# tests/test_dataset.py
import numpy as np
from training.datasets import SlidingWindowDataset

def test_sliding_window_dataset_shapes():
    rng = np.random.RandomState(0)
    M = 3
    T = 40
    L = 16
    P = 4
    F = 1
    X = rng.randn(M, T).astype(float)
    ds = SlidingWindowDataset(X, input_length=L, patch_size=P, horizon=F)
    assert len(ds) == T - (L + F) + 1
    seg, y = ds[0]
    # seg should be (C, N)
    assert seg.shape[0] == M * P
    assert seg.shape[1] == L // P
    assert y.shape == (M, F)
