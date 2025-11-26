import numpy as np

class RainMetricsCalculator:
    def __init__(self, num_timesteps, thresholds=[0.1, 1.0, 2.5, 5.0, 10.0]):

        self.num_timesteps = num_timesteps
        self.thresholds = thresholds
        self.num_thresholds = len(thresholds)

        count = np.zeros((4, self.num_thresholds, num_timesteps), dtype=np.int64)
        self.tp, self.fp, self.fn, self.tn = count[0], count[1], count[2], count[3]

    def process_batch(self, pred_batch, obs_batch):

        bsz = pred_batch.shape[0]
        pred_flat = pred_batch.reshape(bsz, self.num_timesteps, -1)
        obs_flat = obs_batch.reshape(bsz, self.num_timesteps, -1)
        
        for t in range(self.num_timesteps):
            pred_t = pred_flat[:, t, :]
            obs_t = obs_flat[:, t, :]
            
            for idx, threshold in enumerate(self.thresholds):
                pred_binary = (pred_t >= threshold).astype(np.int32)
                obs_binary = (obs_t >= threshold).astype(np.int32)
                
                batch_tp = np.sum(pred_binary & obs_binary)
                batch_fp = np.sum(pred_binary & (1 - obs_binary))
                batch_fn = np.sum((1 - pred_binary) & obs_binary)
                batch_tn = np.sum((1 - pred_binary) & (1 - obs_binary))
                
                self.tp[idx, t] += batch_tp
                self.fp[idx, t] += batch_fp
                self.fn[idx, t] += batch_fn
                self.tn[idx, t] += batch_tn

    def get_metrics(self):  # Dict: (num_thresholds, num_timesteps)

        pod = self.tp / (self.tp + self.fn + 1e-9)
        far = self.fp / (self.tp + self.fp + 1e-9)
        csi = self.tp / (self.tp + self.fp + self.fn + 1e-9)
        acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + 1e-9)
        bias = (self.tp + self.fp) / (self.tp + self.fn + 1e-9)
        
        return {
            'pod': pod,    # detection rate
            'far': far,    # false alarm rate
            'csi': csi,    # critical success index
            'acc': acc,    # accuracy
            'bias': bias   # bias
        }
