import numpy as np
import scipy.sparse as sp
from .constants import C_0
from .typing import Op, Tuple, List, Shape


def fdtd_curl(dx: np.ndarray) -> Tuple[Op, Op]:
    """
    Generate curl operators for the E and H fields (for FDTD).

    Args:
        pos: pos parameter from `simphox.Simulation`

    Returns:
        Functions for discretized curl of E- and H-fields.
    """

    def de(e, d):
        return (np.roll(e[d], -1, axis=d) - e[d]) / dx[d]

    def dh(h, d):
        return (h[d] - np.roll(h[d], 1, axis=d)) / dx[d]

    def curl(f, df):
        return np.stack([df(f[2], 1) - df(f[1], 2), df(f[0], 2) - df(f[2], 0), df(f[1], 0) - df(f[0], 1)])

    return lambda e: curl(e, de), lambda h: curl(h, dh)


def kron_tile(d: List[sp.spmatrix], s: np.ndarray):
    if len(s) == 3:
        return [sp.kron(d[0], sp.eye(s[1] * s[2])),
                sp.kron(sp.kron(sp.eye(s[0]), d[1]), sp.eye(s[2])),
                sp.kron(sp.eye(s[0] * s[1]), d[2])]
    else:
        return [sp.kron(d[0], sp.eye(s[1])), sp.kron(sp.eye(s[0]), d[1])]
