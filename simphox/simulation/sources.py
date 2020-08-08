from typing import Tuple, Callable

from .fdfd import FDFD
from .grid import SimGrid

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    GPU_AVAIL = True
except ImportError:
    GPU_AVAIL = False

from ..typing import Dim


def xs_profile(grid: SimGrid, center: Tuple[float, ...], shape: Tuple[float, ...],
               axis: int = 0, wavelength: float = 1.55, mode_idx: int = 0):
    """

    Args:
        grid: simulation grid (e.g., :code:`FDFD`, :code:`FDTD`, :code:`BPM`)
        center: center tuple of the form :code:`(x, y, z)`
        shape: size of the source
        axis: axis for normal vector of cross-section (one of :code:`(0, 1, 2)`)
        wavelength: wavelength (arb. units, should match with spacing)
        mode_idx: mode index for the eigenmode for source profile
    """
    center = (np.asarray(center) // grid.spacing[0]).astype(np.int)  # assume isotropic for now...
    shape = (np.asarray(shape) // grid.spacing[0]).astype(np.int)
    if grid.ndim == 1:
        raise ValueError(f"Simulation dimension ndim must be 2 or 3 but got {grid.ndim}.")
    if shape.size == 2:
        c = np.roll(center, -axis)
        s0, s1 = int(shape[0] // 2), int(shape[1] // 2)
        slice_arr = np.roll([c[0], slice(c[1] - s0, c[1] - s0 + shape[0]),
                             slice(c[2] - s1, c[2] - s1 + shape[1])], axis)
        s = slice_arr[0], slice_arr[1], slice_arr[2]
    else:
        c = np.roll(center, -axis)
        s0 = int(shape // 2)
        slice_arr = np.roll([c[0], slice(c[1] - s0, c[1] - s0 + shape)], -axis)
        s = slice_arr[0], slice_arr[1]
    mode_eps = grid.eps[s]
    src_fdfd = FDFD(
        shape=mode_eps.shape, spacing=grid.spacing[0],  # TODO (sunil): handle this...
        eps=mode_eps, wavelength=wavelength
    )
    beta, mode = src_fdfd.src(mode_idx=mode_idx, return_beta=True)
    mode = np.expand_dims(mode, axis + 1)
    if grid.ndim == 3:
        mode = np.stack((mode[2], mode[1], mode[0]))  # re-orient the source directions
    return mode, (slice(None), *s)


def tfsf_profile(grid: SimGrid, q_mask: np.ndarray, wavelength: float, k: Dim):
    """TFSF source profile

    Args:
        grid: simulation grid (e.g., :code:`FDFD`, :code:`FDTD`, :code:`BPM`)
        q_mask: mask for scattered field
        wavelength: wavelength (arb. units)
        k: the k-vector (automatically normalized according to wavelength)
    """
    src_fdfd = FDFD(
        shape=grid.shape,
        spacing=grid.spacing,  # TODO (sunil): handle this...
        eps=grid.eps,
        wavelength=wavelength
    )
    mask = q_mask
    q = sp.diags(mask.flatten())
    period = wavelength  # equivalent to period since c = 1!
    k0 = 2 * np.pi / period
    k = np.asarray(k) / (np.sum(k)) * k0
    fsrc = np.einsum('i,j,k->ijk', np.exp(1j * src_fdfd.pos[0][:-1] * k[0]),
                                   np.exp(1j * src_fdfd.pos[1][:-1] * k[1]),
                                   np.exp(1j * src_fdfd.pos[2][:-1] * k[2])).flatten()
    a = src_fdfd.mat
    src = src_fdfd.reshape((q @ a - a @ q) @ fsrc)  # qaaq = quack :)
    return src


def eigenmode_source(grid: SimGrid, center: Tuple[float, ...], size: Tuple[float, ...],
                     axis: int = 0, wavelength: float = 1.55, mode_idx: int = 0, gpu: bool = False):
    """For waveguide-related problems or shining light into a photonic port, an eigenmode source is used.

    Args:
        grid: simulation grid (e.g., :code:`FDFD`, :code:`FDTD`, :code:`BPM`)
        center: center tuple of the form :code:`(x, y, z)`
        size: size of the source
        axis: axis for normal vector of cross-section (one of :code:`(0, 1, 2)`)
        wavelength: wavelength (arb. units, should match with spacing)
        mode_idx: mode index for the eigenmode for source profile
        gpu: place source on the GPU

    Returns:
        Eigenmode source function and region (:code:`slice` object or mask)
    """
    profile, region = xs_profile(grid, center, size, axis, wavelength, mode_idx)
    return cw_source_fn(profile, wavelength, gpu), region


def cw_source(profile: np.ndarray, wavelength: float, t: float, dt: float) -> np.ndarray:
    """CW source array

    Args:
        profile: Profile :math:`\mathbf{\\Psi}`
        wavelength: Wavelength :mode:`\\lambda`
        t: total "on" time
        dt: time step size

    Returns:
        CW source as an ndarray of size :code:`[t/dt, *source_shape]`

    """
    return source(cw_source_fn(profile, wavelength), t, dt)


def source(source_fn: Callable[[float], np.ndarray], t: float, dt: float) -> np.ndarray:
    """Source array given a source function

    Args:
        source_fn: Source function
        t: total "on" time
        dt: time step size

    Returns:
        ndarray of size :code:`[t/dt, *source_shape]`

    """
    ts = np.linspace(0, t, int(t // dt) + 1)
    return np.asarray([source_fn(t) for t in ts])  # not the most efficient, but it'll do for now


def cw_source_fn(profile: np.ndarray, wavelength: float, gpu: bool = False) -> Callable[[float], np.ndarray]:
    """CW source function

    Args:
        profile: Profile :mode:`\mathbf{\\Psi}` (e.g. mode or TFSF) for the input source
        wavelength: Wavelength for CW source
        gpu: place source on the gpu

    Returns:
        the CW source function of time

    """
    profile = cp.asarray(profile) if gpu else profile
    xp = cp if gpu else np
    return lambda t: profile * xp.exp(-1j * 2 * xp.pi * t / wavelength)


def gaussian_source(profiles: np.ndarray, pulse_width: float, center_wavelength: float, dt: float,
                    t0: float = None, linear_chirp: float = 0) -> np.ndarray:
    """Gaussian source array

    Args:
        profiles: profiles defined at individual frequencies
        pulse_width: Gaussian pulse width
        center_wavelength: center wavelength
        dt: time step size
        t0: peak time (default to be central time step)
        linear_chirp: linear chirp coefficient (default to be 0)

    Returns:
        the Gaussian source discretized in time

    """
    k0 = 2 * np.pi / center_wavelength
    t = np.arange(profiles.shape[0]) * dt
    t0 = t[t.size // 2] if t0 is None else t0
    g = np.fft.fft(np.exp(1j * k0 * (t - t0)) * np.exp((-pulse_width + 1j * linear_chirp) * (t - t0) ** 2))
    return np.fft.ifft(g * profiles, axis=0)


def gaussian_source_fn(profiles: np.ndarray, pulse_width: float, center_wavelength: float, dt: float,
                       t0: float = None, linear_chirp: float = 0) -> Callable[[float], np.ndarray]:
    """Gaussian source function

    Args:
        profiles: profiles defined at individual frequencies
        pulse_width: pulse width at individual frequencies
        center_wavelength: center wavelength
        dt: time step size
        t0: peak time (default to be central time step)
        linear_chirp: linear chirp coefficient (default to be 0)

    Returns:
        the Gaussian source function of time

    """
    src = gaussian_source(profiles, pulse_width, center_wavelength, dt, t0, linear_chirp)
    return lambda tt: src[tt // dt]
