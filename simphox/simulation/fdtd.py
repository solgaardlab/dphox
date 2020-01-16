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

    # def cpml(self, a: float = 1e-8):
    #     sigma =
    #     b = np.exp(a + )  # exponential prefactor for the
    #     c = (b - 1) *