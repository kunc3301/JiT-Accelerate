import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
import time


class cmpas(Dataset):
    def __init__(self, split='train', in_length=6, out_length=6, **kwargs) -> None:
        super().__init__()

        self.cmpas_dir = '/data/nvme1/ShortTermForecast/Cmpas/CoarseGrain'
        self.years_list = kwargs.get('years', [2019, 2020, 2021, 2022, 2023])
        self.threshold = kwargs.get('threshold', 100)
        self.in_length = in_length
        self.length = in_length + out_length

        file_list = [f'{self.cmpas_dir}/{year}.zarr' for year in self.years_list]
        self.cmpas = xr.open_mfdataset(
            file_list, engine='zarr', concat_dim='time', combine='nested'
        )

        if split == 'train':
            self.data_list = np.loadtxt("./dataset/train_start_times.txt", dtype=str)
        else:
            self.data_list = np.loadtxt("./dataset/eval_start_times.txt", dtype=str)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):

        # get cmpas: shape = (length=1, lat, lon), range = (0, 1)
        idx = self.cmpas.time.get_index("time").get_loc(self.data_list[index])
        array = self.cmpas['1h_precipitation'].isel(time=slice(idx, idx+self.length)).values[:,22:-22,22:-22]
        array = np.clip(array, 0, self.threshold).astype(np.float32) / self.threshold * 2 - 1  # normalize to (-1, 1)

        return array[:self.in_length], array[self.in_length:], self.data_list[index]


if __name__ == "__main__":

    start_time = time.time()
    dataset = cmpas(split='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    for data in dataloader:
        print(data[0].shape, data[1].shape, data[2])
    print(f'Time taken: {time.time() - start_time:.2f} seconds')


