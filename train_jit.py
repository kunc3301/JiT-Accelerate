import argparse
import datetime
import numpy as np
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

import utils.misc as misc
import copy
from utils.engine import train_one_epoch, evaluate
from models.denoiser import Denoiser
from dataset.cmpas import cmpas


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B-16', type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--dropout', type=float, default=0.1)

    # training
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=0.)
    parser.add_argument('--lr_schedule', type=str, default='cosine')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--ema_decay1', type=float, default=0.9999)
    parser.add_argument('--ema_decay2', type=float, default=0.9996)
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # dataset
    parser.add_argument('--in_length', default=6, type=int)
    parser.add_argument('--out_length', default=6, type=int)
    parser.add_argument('--threshold', default=100, type=int)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str)
    parser.add_argument('--num_sampling_steps', default=50, type=int)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=8)

    # checkpointing
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--resume', default='')
    parser.add_argument('--save_ckpt_freq', type=int, default=10)
    parser.add_argument('--log_freq', default=100, type=int)

    return parser


def main(args):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    accelerator = Accelerator()

    # print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    # print("Arguments:\n{}".format(args).replace(', ', ',\n'))
    args.output_dir = os.path.join(
        args.output_dir,
        "{}_{}_steps{}_in{}_out{}_lr{}_sched{}_mean{}_std{}_drop{}".format(
            args.model, args.sampling_method, args.num_sampling_steps, 
            args.in_length, args.out_length, args.lr,
            args.lr_schedule, args.P_mean, args.P_std, args.dropout
        )
    )
    if accelerator.is_main_process:
        print("work in", args.output_dir)

    # Create dataset and data loader
    dataset_train = cmpas(
        split='train', in_length=args.in_length, out_length=args.out_length, threshold=args.threshold
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    dataset_test = cmpas(
        split='eval', in_length=args.in_length, out_length=args.out_length, threshold=args.threshold
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=True,
        batch_size=args.gen_bsz,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem
    )

    # Create denoiser
    args.proj_dropout = args.attn_dropout = args.dropout
    model = Denoiser(args)
    # print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    
    # Accelerator preparation
    model, optimizer, data_loader_train, data_loader_test = accelerator.prepare(
        model, optimizer, data_loader_train, data_loader_test)
    model_without_ddp = model.module
    args.device = accelerator.device
    args.accelerator = accelerator

    # Set up TensorBoard logging (only on main process)
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Set seeds for reproducibility
    seed = args.seed + accelerator.process_index
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].to(args.device) for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].to(args.device) for name, _ in model_without_ddp.named_parameters()]
        if accelerator.is_main_process:
            print(f"Resumed checkpoint from {args.resume} for {args.epochs} epochs")

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if accelerator.is_main_process:
                print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        if accelerator.is_main_process:
            print(f"Training from scratch for {args.epochs} epochs")

    # Evaluate generation
    if args.evaluate_gen:
        if accelerator.is_main_process:
            print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.no_grad():
            evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if accelerator.is_main_process and (epoch % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if epoch % args.eval_freq == 0 or epoch + 1 == args.epochs:
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, data_loader_test, args, epoch)
            torch.cuda.empty_cache()

        if accelerator.is_main_process and log_writer is not None:
            log_writer.flush()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
