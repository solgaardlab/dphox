import numpy as np

from .grid import SimGrid
from ..typing import Shape, Dim, GridSpacing, Optional, Union


class FDTD(SimGrid):
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray] = 1,
                 pml: Optional[Union[Shape, Dim]] = None):
        super(FDTD, self).__init__(shape, spacing, eps, pml=pml)
        self.dt = 1 / np.sqrt(np.sum(1 / self.spacing ** 2))

        self._init_fields()

    def step(self, src: np.ndarray):
        if src.shape != self.field_shape:
            raise ValueError(f"Expected src shape to match field_shape, but got {src.shape} != {self.e.shape}")

        self.e += (self.curl_h(self.h) + self.psi_e) * self.dt / self.eps_t
        self.h += (self.curl_e(self.e) + self.psi_h) * self.dt

    def _init_fields(self):
        # stored fields for fdtd
        self.e = np.zeros(self.field_shape, dtype=np.complex128)
        self.h = np.zeros_like(self.e)

        # for the pml
        self.psi_e = np.zeros_like(self.e)
        self.psi_h = np.zeros_like(self.e)

    def step_boundary(self, a: float = 0, exp_scale: float = 3.5, kappa: float = 1, log_reflection: float = 1.6):
        # x = np.asarray([np.arange(t + 1, dtype=np.float64) for t in self.pml_shape])
        for ax in range(self.ndim):
            if self.cell_sizes[ax].size <= 1:
                return np.ones(1), np.ones(1)

            p = self.pos[ax]
            pe, ph = (p[:-1] + p[1:]) / 2, p[:-1]
            t = self.pml_shape[ax]

            psi_e = self.psi_e.transpose((-1, ax))
            psi_h = self.psi_h.transpose((-1, ax))

            sigma = lambda x: (exp_scale + 1) * (x ** exp_scale) * log_reflection / 2
            b = lambda x: np.exp(-(a + sigma(x) / kappa) * self.dt)
            c = lambda x: (b(x) - 1) * (sigma(x) / (sigma(x) * kappa + a * kappa ** 2))
            b_e, c_e = b(pe[:t]), b(ph[:t])

    # @lru_cache
    # def _pml_props(self, a: float = 0, exp_scale: float = 3.5, kappa: float = 1, log_reflection: float = 1.6):
    #     for ax in range(self.ndim):
    #         if self.cell_sizes[ax].size <= 1:
    #             return np.ones(1), np.ones(1)
    #
    #         p = self.pos[ax]
    #         pe, ph = (p[:-1] + p[1:]) / 2, p[:-1]
    #         t = self.pml_shape[ax]
    #
    #         psi_e = self.psi_e.transpose((-1, ax))
    #         psi_h = self.psi_h.transpose((-1, ax))
    #
    #         sigma = lambda x: (exp_scale + 1) * (x ** exp_scale) * log_reflection / 2
    #         b = lambda x: np.exp(-(a + sigma(x) / kappa) * self.dt)
    #         c = lambda x: (b(x) - 1) * (sigma(x) / (sigma(x) * kappa + a * kappa ** 2))
    #         b_e, c_e = b(pe[:t]), b(ph[:t])
