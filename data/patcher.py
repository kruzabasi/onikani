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
        start = j * P
        part_list = [X[i, start:start+P] for i in range(M)]
        segments[:, j] = np.concatenate(part_list, axis=0)
    return segments

def segment_to_series(segments: np.ndarray, M: int, P: int) -> np.ndarray:
    """
    Inverse of patch_and_segment.
    segments: np.ndarray with shape (C, N) where C = M * P
    Returns:
      X: np.ndarray shape (M, L) where L = N * P
    """
    if not isinstance(segments, np.ndarray):
        raise TypeError("segments must be a numpy.ndarray")
    if segments.ndim != 2:
        raise ValueError("segments must be 2D array with shape (C, N)")
    C, N = segments.shape
    if C % P != 0:
        raise ValueError(f"C={C} must be divisible by P={P} to infer M")
    inferred_M = C // P
    if inferred_M != M:
        raise ValueError(f"Provided M={M} does not match inferred M={inferred_M} from segments")
    L = N * P
    X = np.empty((M, L), dtype=segments.dtype)
    for j in range(N):
        col = segments[:, j]
        # split col into M parts each length P
        parts = [col[i * P:(i + 1) * P] for i in range(M)]
        for i in range(M):
            X[i, j * P:(j + 1) * P] = parts[i]
    return X
