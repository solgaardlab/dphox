import numpy as np
import scipy.sparse as sp

from .typing import List, Callable


def d2curl_op(d: List[sp.spmatrix]) -> sp.spmatrix:
    o = sp.csr_matrix((d[0].shape[0], d[0].shape[0]))
    return sp.bmat([[o, -d[2], d[1]],
                    [d[2], o, -d[0]],
                    [-d[1], d[0], o]])


def d2curl_fn(f: np.ndarray, df: Callable[[np.ndarray, int], np.ndarray]):
    return np.stack([df(f[2], 1) - df(f[1], 2), df(f[0], 2) - df(f[2], 0), df(f[1], 0) - df(f[0], 1)])


def grid_average(params: np.ndarray) -> np.ndarray:
    if len(params.shape) == 1:
        p = (params + np.roll(params, shift=1) + np.roll(params, shift=-1)) / 3
        return np.stack((p, p, p))
    p = params[..., np.newaxis] if len(params.shape) == 2 else params
    p_x = (p + np.roll(p, shift=1, axis=1) + np.roll(p, shift=1, axis=2) +
           np.roll(p, shift=-1, axis=1) + np.roll(p, shift=-1, axis=2)) / 5
    p_y = (p + np.roll(p, shift=1, axis=0) + np.roll(p, shift=1, axis=2) +
           np.roll(p, shift=-1, axis=0) + np.roll(p, shift=-1, axis=2)) / 5
    p_z = (p + np.roll(p, shift=1, axis=0) + np.roll(p, shift=1, axis=1) +
           np.roll(p, shift=-1, axis=0) + np.roll(p, shift=-1, axis=1)) / 5
    return np.stack([p_x, p_y, p_z])
