# tests/test_patcher.py
import numpy as np
from data.patcher import patch_and_segment, segment_to_series

def test_patch_and_segment_simple():
    M = 2
    L = 8
    P = 2
    X = np.arange(M * L).reshape(M, L).astype(float)
    seg = patch_and_segment(X, P)
    assert seg.shape == (M * P, L // P)
    expected_first = np.concatenate([X[i, 0:P] for i in range(M)])
    assert (seg[:, 0] == expected_first).all()

def test_segment_inverse_roundtrip():
    rng = np.random.RandomState(0)
    M = 3
    L = 12
    P = 3
    X = rng.randn(M, L).astype(float)
    seg = patch_and_segment(X, P)
    X_rec = segment_to_series(seg, M=M, P=P)
    assert X_rec.shape == X.shape
    assert np.allclose(X_rec, X)
