import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_size, output_size):
        self.data = data
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.data) - self.input_size - self.output_size + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_size]
        y_start_idx = idx + self.input_size
        y_end_idx = idx + self.input_size + self.output_size
        y = self.data[y_start_idx:y_end_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_data(csv_path, normalize=True):
    data = pd.read_csv(csv_path)
    data_size = len(data)
    if data_size<1:
        print("Warning: File '"+csv_path+"' is empty")
        return None
    
    series = data["Close"].values
    if normalize:
        series = (series - np.mean(series)) / np.std(series)
    return series
