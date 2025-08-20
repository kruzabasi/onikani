# training/datasets.py
import numpy as np
import torch
from torch.utils.data import Dataset
from data.patcher import patch_and_segment

class SlidingWindowDataset(Dataset):
    """
    Sliding window dataset for PSformer.
    - X: numpy array shape (M, T) where M = channels, T = time length
    - input_length: L (lookback length)
    - patch_size: P
    - horizon: F (forecast horizon, default 1)
    Produces:
      - x_seg: torch.FloatTensor shape (C, N)  where C = M * P, N = L // P
      - y: torch.FloatTensor shape (M, F)
    """
    def __init__(self, X: np.ndarray, input_length: int, patch_size: int, horizon: int = 1):
        assert isinstance(X, np.ndarray) and X.ndim == 2, "X must be ndarray with shape (M, T)"
        self.X = X.astype(np.float32)
        self.M, self.T = self.X.shape
        self.L = input_length
        self.P = patch_size
        if self.L % self.P != 0:
            raise ValueError("input_length must be divisible by patch_size")
        self.N = self.L // self.P
        self.F = horizon
        # number of windows
        self.count = self.T - (self.L + self.F) + 1
        if self.count <= 0:
            raise ValueError("Time length T too small for chosen input_length + horizon")
    def __len__(self):
        return self.count
    def __getitem__(self, idx):
        # window start at idx: use [idx : idx+L] as input, target [idx+L : idx+L+F]
        start = idx
        end = start + self.L
        x_win = self.X[:, start:end]   # shape (M, L)
        y_win = self.X[:, end:end + self.F]  # shape (M, F)
        seg = patch_and_segment(x_win, self.P)  # (C, N)
        # return torch tensors
        return torch.from_numpy(seg), torch.from_numpy(y_win.astype(np.float32))
