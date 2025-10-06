import numpy as np


def euclid(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))


def nearest_index(grid_coords: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid_coords - value)))

