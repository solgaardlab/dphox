from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

from .grid import SimGrid
from ..typing import Shape, Dim, GridSpacing, Optional, Tuple, List, Union, SpSolve, Op

try:  # pardiso (using Intel MKL) is much faster than scipy's solver
    from .mkl import spsolve, feast_eigs
except OSError:  # if mkl isn't installed
    from scipy.sparse.linalg import spsolve


class BPM(SimGrid):
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray] = 1,
                 wavelength: float = 1.55, bloch_phase: Union[Dim, float] = 0.0,
                 pml: Optional[Union[Shape, Dim]] = None, pml_eps: float = 1.0,
                 grid_avg: bool = True, no_grad: bool = True):

        self.wavelength = wavelength
        self.k0 = 2 * np.pi / self.wavelength  # defines the units for the simulation!
        self.no_grad = no_grad

        super(BPM, self).__init__(
            shape=shape,
            spacing=spacing,
            eps=eps,
            bloch_phase=bloch_phase,
            pml=pml,
            pml_eps=pml_eps,
            grid_avg=grid_avg
        )
        self._init_fields()

    def _init_fields(self):
        # stored fields for fdtd
        self.e = np.zeros(self.field_shape, dtype=np.complex128)

    def step(self, z: int, vectorial: bool = True) -> np.ndarray:
        """Build the discrete Maxwell operator :math:`A(k_0)` acting on :math:`\mathbf{e}`.

        The discretized version of Maxwell's equations in frequency domain is:
        .. math::
            \nabla \times \mu^{-1} \nabla \times \mathbf{e} - k_0^2 \epsilon \mathbf{e} = k_0 \mathbf{j},
        which can be written in the form :math:`A \mathbf{e} = \mathbf{b}`, where:
        .. math::
            A = \nabla \times \mu^{-1} \nabla \times - k_0^2 \epsilon \\
            b = k_0 \mathbf{j}
        is an operator representing the discretized EM wave operator at frequency :math:`omega`.

        Returns:
            Electric field operator :math:`A` for solving Maxwell's equations at frequency :math:`omega`.
        """
        if self.ndim == 3:
            es = []
            for idx in range(3):
                ϵ = self.eps_t[idx][:, :, z]
                e_00 = (np.roll(np.roll(ϵ, 1, axis=0), 1, axis=1) / np.roll(ϵ, 1, axis=0) - 1) * np.roll(
                    np.roll(self.e[idx], 1, axis=0), 1, axis=1)
                e_11 = (np.roll(np.roll(ϵ, -1, axis=0), -1, axis=1) / np.roll(ϵ, -1, axis=0) - 1) * np.roll(
                    np.roll(self.e[idx], -1, axis=0), -1, axis=1)
                if vectorial:
                    e_01 = (np.roll(np.roll(ϵ, 1, axis=0), -1, axis=1) / np.roll(ϵ, 1, axis=0) - 1) * np.roll(
                        np.roll(self.e[idx], 1, axis=0), -1, axis=1)
                    e_10 = (np.roll(np.roll(ϵ, -1, axis=0), 1, axis=1) / np.roll(ϵ, -1, axis=0) - 1) * np.roll(
                        np.roll(self.e[idx], -1, axis=0), 1, axis=1)
                    es.append(e_00 + e_01 + e_10 + e_11)
                else:
                    es.append(e_00 + e_11)
        return np.stack(es)

    @property
    @lru_cache()
    def _dxes(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Conditional transformation of self.dxes based on stretched-coordinated perfectly matched layers (SC-PML)

        Returns:
            SC-PML transformation of dxes for the e-fields and h-fields, respectively
        """

        if self.pml_shape is None:
            return np.meshgrid(*self.cell_sizes, indexing='ij'), np.meshgrid(*self.cell_sizes, indexing='ij')
        else:
            dxes_pml_e, dxes_pml_h = [], []
            for ax, p in enumerate(self.pos):
                scpml_e, scpml_h = self.scpml(ax)
                dxes_pml_e.append(self.cell_sizes[ax] * scpml_e)
                dxes_pml_h.append(self.cell_sizes[ax] * scpml_h)
            return np.meshgrid(*dxes_pml_e, indexing='ij'), np.meshgrid(*dxes_pml_h, indexing='ij')

    def scpml(self, ax: int, exp_scale: float = 4, log_reflection: float = -16) -> Tuple[np.ndarray, np.ndarray]:
        if self.cell_sizes[ax].size == 1:
            return np.ones(1), np.ones(1)
        p = self.pos[ax]
        pe, ph = (p[:-1] + p[1:]) / 2, p[:-1]
        absorption_corr = self.k0 * self.pml_eps
        t = self.pml_shape[ax]

        def _scpml(d: np.ndarray):
            d_pml = np.hstack((
                (d[t] - d[:t]) / (d[t] - p[0]),
                np.zeros_like(d[t:-t]),
                (d[-t:] - d[-t]) / (p[-1] - d[-t])
            ))
            return 1 + 1j * (exp_scale + 1) * (d_pml ** exp_scale) * log_reflection / (2 * absorption_corr)

        return _scpml(pe), _scpml(ph)
