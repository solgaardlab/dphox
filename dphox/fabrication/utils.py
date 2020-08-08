import numpy as np
from typing import Tuple
from shapely.geometry import LineString


def get_point_in_profile(y: float, path_xy: np.ndarray) -> Tuple[float, float]:
    for i in range(path_xy.shape[0] - 1):
        if path_xy[i, 0] <= y <= path_xy[i + 1, 0]:
            if not np.linalg.norm(path_xy[i, :] - path_xy[i + 1, :]) == 0:
                return y, path_xy[i+1][1]


def get_point_idx_in_profile(y: float, path_xy: np.ndarray) -> int:
    for i in range(path_xy.shape[0] - 1):
        if path_xy[i, 0] <= y <= path_xy[i + 1, 0]:
            if not np.linalg.norm(path_xy[i, :] - path_xy[i + 1, :]) == 0:
                return i+1
