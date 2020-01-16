import numpy as np
import scipy.sparse as sp

from .typing import List, Callable, Optional


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


def emplot(ax, val, eps, spacing: Optional[float] = None, field_cmap: str = 'RdBu', alpha=0.9):
    nx, ny = val.shape
    extent = (0, int(nx * spacing), 0, int(ny * spacing)) if spacing else (0, nx, 0, ny)
    ax.imshow(eps.T, cmap='gray', origin='lower left', alpha=1, extent=extent)
    ax.imshow(val.T, cmap=field_cmap, origin='lower left', alpha=alpha, extent=extent)
    if spacing:  # in microns!
        ax.set_ylabel(r'$y$ ($\mu$m)')
        ax.set_xlabel(r'$x$ ($\mu$m)')


def field_emplot_re(ax, field: np.ndarray, eps: np.ndarray, spacing: Optional[float] = None):
    emplot(ax, field.real, eps, spacing)


def field_emplot_mag(ax, field: np.ndarray, eps: np.ndarray, spacing: Optional[float] = None):
    emplot(ax, np.abs(field), eps, spacing, field_cmap='hot', alpha=0.8)
