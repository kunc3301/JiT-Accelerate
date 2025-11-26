import math
import sys
import os
from tqdm import tqdm
import torch
import numpy as np
import utils.misc as misc
from utils.metrics import RainMetricsCalculator
import copy

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import xarray as xr
import cmaps
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


shp_data = shpreader.Reader('/data/nvme3/chenkun/China/bou2_4p.shp')
cmpas_zarr = xr.open_zarr('/data/nvme1/ShortTermForecast/Cmpas/CoarseGrain/2023.zarr')


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def plt_radar(sub, output, date_time=None):

    lon_ticks = np.arange(115, 123, 2)
    lat_ticks = np.arange(27, 36, 1)

    lon_grid, lat_grid = np.meshgrid(cmpas_zarr['longitude'].values[22:-22], sorted(cmpas_zarr['latitude'].values[22:-22], reverse=True))

    ax =  plt.subplot(sub[0],sub[1],sub[2], projection=ccrs.PlateCarree())
    ax.add_geometries(shp_data.geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='dimgray', linewidth=0.5)

    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())

    ax.set_xticklabels(
        ['{:.0f}°E'.format(tick) if tick > 0 else '{:.0f}°W'.format(abs(tick)) for tick in ax.get_xticks()],
        fontsize=8)
    ax.set_yticklabels(
        ['{:.0f}°N'.format(tick) if tick > 0 else '{:.0f}°S'.format(abs(tick)) for tick in ax.get_yticks()],
        fontsize=8)

    ax.xaxis.set_major_formatter(LATITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    ax.contourf(lon_grid, lat_grid, output, levels=[x for x in range(0,50)], cmap=cmaps.WhiteBlueGreenYellowRed, extend='both')
    
    if date_time:
        ax.set_title(date_time)

def train_one_epoch(model, model_without_ddp, data_loader, optimizer, epoch, log_writer=None, args=None):
    model.train(True)
    optimizer.zero_grad()

    # if log_writer is not None:
    #     print('log_dir: {}'.format(log_writer.log_dir))

    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}", disable=not args.accelerator.is_main_process)
    for data_iter_step, (x_prev, x, _) in enumerate(pbar):
        # per iteration (instead of per epoch) lr scheduler
        adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        loss = model(x, x_prev)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        args.accelerator.backward(loss)
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        loss_value_reduce = misc.all_reduce_mean(loss_value, args.accelerator)
        lr = optimizer.param_groups[0]["lr"]
        global_step = epoch * len(data_loader) + data_iter_step

        logs = {'loss': loss_value_reduce, 'lr': lr, 'step': global_step}
        pbar.set_postfix(**logs)

        if log_writer is not None:
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, global_step)
                log_writer.add_scalar('lr', lr, global_step)


def evaluate(model_without_ddp, data_loader_test, args, epoch):
    model_without_ddp.eval()

    # if log_writer is not None:
    #     print('log_dir: {}'.format(log_writer.log_dir))

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    # print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(args.output_dir, f'samples_epoch{epoch:04d}')
    # print("Save to:", save_folder)
    if args.accelerator.is_main_process and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    metrics = RainMetricsCalculator(num_timesteps=args.out_length, thresholds=[0.1, 1.0, 2.5, 5.0, 10.0])
    for data_iter_step, (x_prev, x, date_time) in enumerate(data_loader_test):
        sampled_images = model_without_ddp.generate(x_prev)
        args.accelerator.wait_for_everyone()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2 * args.threshold
        sampled_images = sampled_images.detach().cpu().numpy()
        x = (x + 1) / 2 * args.threshold
        x = x.detach().cpu().numpy()
        x_prev = (x_prev + 1) / 2 * args.threshold
        x_prev = x_prev.detach().cpu().numpy()

        # accumulate metrics
        metrics.process_batch(sampled_images, x)

        # plot figures
        # print(date_time)
        date_time = datetime.strptime(date_time[0], '%Y-%m-%dT%H:%M:%S')
        len_date = args.in_length + args.out_length
        date_time = [datetime.strftime(date_time + timedelta(hours=i), "%Y-%m-%dT%H:%M:%S") for i in range(len_date)]

        plt.figure(figsize=(15, 7))
        for i in range(args.out_length):
            plt_radar([3, args.out_length, i+1], x_prev[0,i], date_time[i])
            plt_radar([3, args.out_length, args.out_length+i+1], x[0,i], date_time[args.out_length+i])
            plt_radar([3, args.out_length, args.out_length*2+i+1], sampled_images[0,i], date_time[args.out_length+i])
        plt.tight_layout()
        plt.savefig(f'{save_folder}/{date_time[0]}.png', dpi=150)
        plt.close()

    args.accelerator.wait_for_everyone()
    # back to no ema
    # print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)
    
    if args.accelerator.is_main_process:
        # print metrics
        metric_dict = metrics.get_metrics()
        for metric_name, metric_values in metric_dict.items():
            print(f"Epoch {epoch} {metric_name}:")
            for idx, threshold in enumerate(metrics.thresholds):
                values_str = ", ".join([f"{v:.4f}" for v in metric_values[idx]])
                print(f"  {threshold} mm/h: {values_str}")

