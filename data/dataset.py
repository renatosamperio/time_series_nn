import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_size, output_size):
        self.data = data
        self.input_size = input_size
        self.output_size = output_size
        # print("*** data.size: %s"%str(len(self.data)))
        # print("*** input_size: %s"%str(self.input_size))
        # print("*** output_size: %s"%str(self.output_size))

    def __len__(self):
        return len(self.data) - self.input_size - self.output_size + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_size]
        y_start_idx = idx + self.input_size
        y_end_idx = idx + self.input_size + self.output_size
        y = self.data[y_start_idx:y_end_idx]
        # print("*** x.len: %s"%len(x))
        # print("*** y.len: %s"%len(y))
        # print("*** y.start_idx: %s"%y_start_idx)
        # print("*** y.end_idx: %s"%y_end_idx)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_data(csv_path, normalize=True):
    data = pd.read_csv(csv_path)
    series = data["Close"].values
    if normalize:
        series = (series - np.mean(series)) / np.std(series)
    return series
