import numpy as np
from typing import Tuple, List

from .grid import SimGrid
from .sources import XSSource
from functools import lru_cache, partial
from ..typing import Shape, Dim, GridSpacing, Optional, Union
from ..utils import sigma


class FDTD(SimGrid):
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray] = 1,
                 wavelength: float = 1.55, pml: Optional[Union[Shape, Dim]] = None):
        super(FDTD, self).__init__(shape, spacing, eps, pml=pml)
        self.dt = 1 / np.sqrt(np.sum(1 / self.spacing ** 2))  # includes courant condition!
        self.wavelength = wavelength

        # stored fields for fdtd
        self.e = np.zeros(self.field_shape, dtype=np.complex128)
        self.h = np.zeros_like(self.e)

        # for pml updates
        self.psi_e = np.zeros_like(self.e)
        self.psi_h = np.zeros_like(self.h)

        self.src = None

        if self.pml_shape is not None:
            out_e, out_h = zip(*[self._cpml(ax) for ax in range(3)])
            self.cpml = np.meshgrid(*out_e, indexing='ij'), np.meshgrid(*out_h, indexing='ij')

    def set_xs_src(self, source_region, mode_idx):
        self.src = XSSource(self, source_region, mode_idx=mode_idx, wavelength=self.wavelength)

    def step(self, e: np.ndarray, h: np.ndarray, psi_e: Optional[np.ndarray] = None,
             psi_h: Optional[np.ndarray] = None):
        if self.src.shape != self.field_shape:
            raise ValueError(f"Expected src shape to match field_shape, but got {self.src.shape} != {self.e.shape}")
        if self.pml_shape is None or psi_e is None or psi_h is None:
            psi_e_next, psi_h_next = None, None
            e_next, h_next = self.curl_h(h) * self.dt / self.eps_t, self.curl_e(e) * self.dt
        else:
            b, c = self.cpml
            psi_e_next = b[0] * psi_e + c[0] * self.curl_h(h)
            psi_h_next = b[1] * psi_h + c[1] * self.curl_e(e)
            e_next = (self.curl_h(h) + psi_e_next) * self.dt / self.eps_t
            h_next = (self.curl_e(e) + psi_h_next) * self.dt
        return e_next, h_next, psi_e_next, psi_h_next

    def _cpml(self, ax: int, a: float = 0, exp_scale: float = 3.5,
             kappa: float = 1, log_reflection: float = 1.6) -> Tuple[np.ndarray, np.ndarray]:
        if self.cell_sizes[ax].size == 1:
            return np.ones(2), np.ones(2)

        _sigma = partial(sigma, t=self.pml_shape[ax], exp_scale=exp_scale,
                         log_reflection=log_reflection, absorption_corr=1)

        def _bc(sigma: np.ndarray, dt):
            b = np.exp(-(a + sigma / kappa) * dt)
            c = (b - 1) * (sigma / (sigma * kappa + a * kappa ** 2))
            return b, c

        return _bc(_sigma(self.pos[ax]), self.dt)

