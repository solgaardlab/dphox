import numpy as np

from .typing import Tuple


def _um_str(microns: float):
    return f'{np.around(float(microns),3):.3f}'


def circle(radius: float, n: int = 20, xy: Tuple[float, float] = (0, 0)):
    u = np.linspace(0, 2 * np.pi, n + 1)
    polygon = [(radius * np.sin(a) + xy[0], radius * np.cos(a) + xy[1]) for a in u]
    return polygon
