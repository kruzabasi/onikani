# tests/test_patcher.py
import numpy as np
from data.patcher import patch_and_segment

def test_patch_and_segment_simple():
    M = 2
    L = 8
    P = 2
    X = np.arange(M * L).reshape(M, L).astype(float)
    seg = patch_and_segment(X, P)
    assert seg.shape == (M * P, L // P)
    # first segment expected: concat of first P values of each channel
    expected_first = np.concatenate([X[i, 0:P] for i in range(M)])
    assert (seg[:, 0] == expected_first).all()
