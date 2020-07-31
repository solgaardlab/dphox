from typing import Union, Optional, Tuple, Callable

from .fdfd import FDFD
from .grid import SimGrid

import numpy as np
import scipy.sparse as sp

from ..typing import Dim, Dim2


class Profile:
    def __init__(self, mask: np.ndarray, e: np.ndarray, wavelength: float,
                 h: Optional[np.ndarray] = None, beta: float = None):
        """

        Args:
            mask: Mask in the overall grid to place the profile
            e: Electric field (:math:`\mathbf{E}`)
            wavelength: Wavelength (:math:`\\lambda`)
            h: Magnetic field (:math:`\mathbf{H}`)
            beta: Propagation constant (:math:`\\beta`)
        """
        self.beta = beta
        self.mask = mask
        self.indices = self.mask != 0
        self.e_src = e
        self.h_src = h
        self.shape = mask.shape
        self.period = wavelength  # equivalent to period since c = 1!
        self.k0 = 2 * np.pi / self.period
        self.n = self.beta / self.k0
        self.profile = np.zeros_like(self.mask).astype(np.complex128)
        self.profile[self.indices] = self.e_src.flatten()


class XS(Profile):
    def __init__(self, grid: SimGrid, source_region: Union[Tuple[Dim2, Dim2, Dim2], np.ndarray],
                 wavelength: float = 1.55, mode_idx: int = 0):
        """

        Args:
            grid: simulation grid (e.g., :code:`FDFD`, :code:`FDTD`, :code:`BPM`)
            source_region: a tuple of the form :code:`((xmin, xmax), (ymin, ymax), (zmin, zmax))`
                OR a numpy mask array that gives a line or plane cross section. Using :code:`None` for either
                :code:`min` or :code:`max` values results in starting from the edge of the simulation along that
                direction. Using `None` in place of the a range
            wavelength: wavelength (arb. units, should match with spacing)
            mode_idx: mode index for the eigenmode for source profile
        """

        if grid.ndim == 1:
            raise ValueError(f"Simulation dimension ndim must be 2 or 3 but got {grid.ndim}.")

        if isinstance(source_region, np.ndarray):
            mask = source_region
            if not mask.shape == tuple(grid.shape):
                raise ValueError(f"Mask has the wrong shape. Expected {mask.shape} but got {tuple(grid.shape)}")
            mode_eps = grid.eps[mask != 0]
        elif isinstance(source_region, tuple):
            xr, yr, zr = source_region
            mode_eps = grid.eps[xr[0]:xr[1], yr[0]:yr[1], zr[0]:zr[1]]
            mask = np.zeros_like(grid.eps_t, dtype=np.bool)
            mask[:, xr[0]:xr[1], yr[0]:yr[1], zr[0]:zr[1]] = True
        else:
            raise ValueError("Source region malformatted.")
        src_fdfd = FDFD(
            shape=mode_eps.shape,
            spacing=grid.spacing[0],  # TODO (sunil): handle this...
            eps=mode_eps,
            wavelength=wavelength
        )
        beta, mode = src_fdfd.src(mode_idx=mode_idx, return_beta=True)
        e, h = mode, src_fdfd.e2h(mode, beta)
        if grid.ndim == 3:
            e = np.stack((e[2], e[1], e[0]))  # re-orient the source directions
        super(XS, self).__init__(mask, e, wavelength, h, beta)


class TFSF(Profile):
    def __init__(self, grid: SimGrid, q_mask: np.ndarray, wavelength: float, k: Dim):
        """

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
        self.mask = q_mask
        self.shape = q_mask.shape
        self.q = sp.diags(self.mask.flatten())
        self.period = wavelength  # equivalent to period since c = 1!
        self.k0 = 2 * np.pi / self.period
        self.k = np.asarray(k) / (np.sum(k)) * self.k0
        self.fsrc = np.einsum('i,j,k->ijk',
                              np.exp(1j * src_fdfd.pos[0][:-1] * self.k[0]),
                              np.exp(1j * src_fdfd.pos[1][:-1] * self.k[1]),
                              np.exp(1j * src_fdfd.pos[2][:-1] * self.k[2])).flatten()
        a = src_fdfd.mat
        self.e_src = src_fdfd.reshape((self.q @ a - a @ self.q) @ self.fsrc)  # qaaq = quack :)
        self.h_src = src_fdfd.e2h(self.e_src)
        self.profile = self.e_src


def cw_source(profile: Optional[np.ndarray, Profile], wavelength: float, t: float, dt: float) -> np.ndarray:
    """ CW source array

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
    """ Source array given a source function

    Args:
        source_fn: Source function
        t: total "on" time
        dt: time step size

    Returns:
        ndarray of size :code:`[t/dt, *source_shape]`

    """
    ts = np.linspace(0, t, int(t // dt) + 1)
    return np.asarray([source_fn(t) for t in ts])  # not the most efficient, but it'll do for now


def cw_source_fn(profile: Optional[np.ndarray, Profile], wavelength: float) -> Callable[[float], np.ndarray]:
    """ CW source function

    Args:
        profile: Profile :mode:`\mathbf{\\Psi}` (e.g. mode or TFSF) for the input source
        wavelength: Wavelength for CW source

    Returns:
        the CW source function of time

    """
    profile = profile.profile if isinstance(profile, Profile) else profile
    return lambda t: profile * np.exp(1j * 2 * np.pi * t / wavelength)


def gaussian_source(profiles: np.ndarray, pulse_width: float, center_wavelength: float, dt: float,
                    t0: float = None, linear_chirp: float = 0) -> np.ndarray:
    """Gaussian source array

    Args:
        profiles: profiles defined at individual frequencies
        pulse_width: pulse width at individual frequencies
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
