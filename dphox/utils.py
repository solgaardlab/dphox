import numpy as np
import scipy.sparse as sp

from .typing import List, Callable


def poynting_z(e: np.ndarray, h: np.ndarray):
    e_cross = np.stack([(e[0] + np.roll(e[0], shift=1, axis=1)) / 2,
                        (e[1] + np.roll(e[1], shift=1, axis=0)) / 2])
    h_cross = np.stack([(h[0] + np.roll(h[0], shift=1, axis=0)) / 2,
                        (h[1] + np.roll(h[1], shift=1, axis=1)) / 2])
    return e_cross[0] * h_cross.conj()[1] - e_cross[1] * h_cross.conj()[0]


def poynting_x(e: np.ndarray, h: np.ndarray):
    e_cross = np.stack([(e[1] + np.roll(e[1], shift=1, axis=1)) / 2,
                        (e[2] + np.roll(e[2], shift=1, axis=0)) / 2])
    h_cross = np.stack([(h[1] + np.roll(h[1], shift=1, axis=0)) / 2,
                        (h[2] + np.roll(h[2], shift=1, axis=1)) / 2])
    return e_cross[1] * h_cross.conj()[2] - e_cross[2] * h_cross.conj()[1]


def overlap(e1: np.ndarray, h1: np.ndarray, e2: np.ndarray, h2: np.ndarray):
    return (np.sum(poynting_z(e1, h2)) * np.sum(poynting_z(e2, h1)) /
            np.sum(poynting_z(e1, h1))).real / np.sum(poynting_z(e2, h2)).real


def d2curl_op(d: List[sp.spmatrix]) -> sp.spmatrix:
    o = sp.csr_matrix((d[0].shape[0], d[0].shape[0]))
    return sp.bmat([[o, -d[2], d[1]],
                    [d[2], o, -d[0]],
                    [-d[1], d[0], o]])


def curl(f: np.ndarray, df: Callable[[np.ndarray, int], np.ndarray], beta: float = None):
    if beta is not None:
        return np.stack([df(f[2], 1) + 1j * beta * f[1],
                         -1j * beta * f[0] - df(f[2], 0),
                         df(f[1], 0) - df(f[0], 1)])
    return np.stack([df(f[2], 1) - df(f[1], 2),
                     df(f[0], 2) - df(f[2], 0),
                     df(f[1], 0) - df(f[0], 1)])


def yee_avg(params: np.ndarray, shift: int = 1) -> np.ndarray:
    if len(params.shape) == 1:
        p = (params + np.roll(params, shift=shift) + np.roll(params, shift=-shift)) / 3
        return np.stack((p, p, p))
    p = params[..., np.newaxis] if len(params.shape) == 2 else params
    p_x = (p + np.roll(p, shift=shift, axis=1)) / 2
    p_y = (p + np.roll(p, shift=shift, axis=0)) / 2
    p_z = (p_y + np.roll(p_y, shift=shift, axis=1)) / 2
    return np.stack([p_x, p_y, p_z])


def pml_params(pos: np.ndarray, t: int, exp_scale: float, log_reflection: float, absorption_corr: float):
    d = np.vstack(((pos[:-1] + pos[1:]) / 2, pos[:-1])).T
    d_pml = np.vstack((
        (d[t] - d[:t]) / (d[t] - pos[0]),
        np.zeros_like(d[t:-t]),
        (d[-t:] - d[-t]) / (pos[-1] - d[-t])
    )).T
    sigma = (exp_scale + 1) * (d_pml ** exp_scale) * log_reflection / (2 * absorption_corr)
    alpha = (1 - d_pml) ** exp_scale
    return sigma, alpha

#
# def subpixel_smoothing(poly: Union[polygon, np.ndarray], grid_dim: Tuple[float, float, float, float]):
#     cairo.
