# data/patcher.py
import numpy as np

def patch_and_segment(X: np.ndarray, P: int) -> np.ndarray:
    """
    Transform multivariate time-series X (M, L) into PSformer segments shape (C, N)
    where:
      - M = number of channels/variables
      - L = look-back length
      - P = patch length (must divide L)
      - N = L // P  (number of patches)
      - C = M * P   (segment dimension)
    Returns:
      segments : np.ndarray of shape (C, N)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy.ndarray with shape (M, L)")
    if X.ndim != 2:
        raise ValueError("X must be 2D array with shape (M, L)")
    M, L = X.shape
    if L % P != 0:
        raise ValueError(f"Patch size P={P} must divide sequence length L={L}")
    N = L // P
    C = M * P
    segments = np.empty((C, N), dtype=X.dtype)
    for j in range(N):
        # gather patch j of each variable and concatenate
        start = j * P
        part_list = [X[i, start:start+P] for i in range(M)]
        segments[:, j] = np.concatenate(part_list, axis=0)
    return segments
