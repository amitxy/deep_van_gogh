import os
import time
import sys
from torch.utils.data import Dataset, Subset
import numpy as np
import torch

DATA_DIR =  'data/dataset'
DATASET_DIR = f'{DATA_DIR}/Post_Impressionism'
CSV_PATH = f'{DATA_DIR}/classes.csv'
OPTIMIZED_DIR = f'data/optimized/'
MODELS_DIR = f'models/'
SEED = 42


class NumPyDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        self.images = data["images"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = torch.tensor(self.images[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def show_optimization_progress(current_size, total_size):
    sys.stdout.flush()
    percent = (current_size / total_size) * 100
    if percent == 100:
        print(f"\rOptimizing dataset... {percent:.2f}%")
    else:
        sys.stdout.write(f"\rOptimizing dataset... {percent:.2f}%")


def get_opt_dataset(dataset_name, test_idx=None, val_idx=None):
    data = NumPyDataset(os.path.join(OPTIMIZED_DIR, f'{dataset_name}.npz'))
    if test_idx and val_idx:
        return Subset(data, test_idx), Subset(data, val_idx)

    return Subset(data, test_idx if test_idx else val_idx)

def mean_dict(dicts):
    """Returns the mean values of the dictionaries"""
    keys = dicts[0].keys()
    return {key: np.mean([d[key] for d in dicts]) for key in keys if np.issubdtype(type(dicts[0][key]), np.number) }