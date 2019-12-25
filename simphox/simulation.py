import numpy as np
import scipy.sparse as sp

from .component import Component
from .typing import Shape, Dim, GridSpacing, Optional, Tuple, List, Union, SpSolve
from .constants import C_0, EPS_0, MU_0
from .ops import kron_tile


class Grid:
    def __init__(self, grid_shape: Shape, grid_spacing: GridSpacing, eps: float = 1):
        self.shape = np.asarray(grid_shape, dtype=np.int)
        self.spacing = grid_spacing * np.ones(len(grid_shape)) if isinstance(grid_spacing, float) else np.asarray(
            grid_spacing)
        if not len(self.shape) == len(self.spacing):
            raise ValueError(f'Require len(grid_shape) == len(grid_spacing) but got'
                             f'({len(self.shape)}, {len(self.spacing)})')
        self.dim = len(grid_shape)
        self.boundary_eps = eps
        self.n = np.prod(self.shape)
        self.eps: np.ndarray = np.ones(self.shape) * eps
        self.size = self.spacing * self.shape
        self.dxes = [self.spacing[i] * np.ones((self.shape[i],))
                     if i < self.dim else np.ones((1,)) for i in range(3)]
        self.pos = [np.hstack((0, np.cumsum(dx))) if dx.size > 1 else None for dx in self.dxes]
        if self.dim == 1:
            self.mesh_pos = np.mgrid[:self.size[0]:self.spacing[0]]
        elif self.dim == 2:
            self.mesh_pos = np.stack(np.mgrid[:self.size[0]:self.spacing[0], :self.size[1]:self.spacing[1]],
                                     axis=0)
        else:
            self.mesh_pos = np.stack(np.mgrid[:self.size[0]:self.spacing[0],
                                     :self.size[1]:self.spacing[1],
                                     :self.size[2]:self.spacing[2]], axis=0)
        self.components = []

    def _check_bounds(self, component):
        b = component.bounds
        return b[0] >= 0 and b[1] >= 0 and b[2] <= self.size[0] and b[3] <= self.size[1]

    def fill(self, zmax: float, eps: float):
        if self.dim == 3:
            self.eps[:, :, :int(zmax / self.spacing[2])] = eps
        else:
            raise ValueError('dim must be 3 to fill grid')

    def add(self, component: Component, eps: float, zmin: float = None, thickness: float = None):
        if not self._check_bounds(component):
            raise ValueError('The pattern is out of bounds')
        self.components.append(component)
        mask = component.mask(self.shape[:2], self.spacing)
        if self.dim == 2:
            self.eps = mask * eps
        else:
            zidx = (int(zmin / self.spacing[0]), int((zmin + thickness) / self.spacing[1]))
            self.eps[:, :, zidx[0]:zidx[1]] += (mask * eps)[..., np.newaxis]


class FDFD(Grid):
    def __init__(self, grid_shape: Shape, grid_spacing: GridSpacing, eps: float = 1,
                 pml_shape: Optional[Shape] = None, bloch_phase: Union[Dim, float] = 0.0, wavelength: float = 1.55):
        super(FDFD, self).__init__(grid_shape, grid_spacing, eps)
        self.bloch = np.ones_like(self.shape) * np.exp(1j * np.asarray(bloch_phase)) if isinstance(bloch_phase, float) \
            else np.exp(1j * np.asarray(bloch_phase))
        self.pml_shape = pml_shape
        self.omega = 2 * np.pi * C_0 / (wavelength * 1e-6)
        self.wavelength = wavelength
        if pml_shape is not None and not len(pml_shape) == len(self.shape):
            raise ValueError(f'Need len(pml_shape) == len(grid_shape) but got ({len(pml_shape)}, {len(self.shape)}).')
        if not len(self.bloch) == len(self.shape):
            raise ValueError(
                f'Need len(bloch_phase) == len(grid_shape) but got ({len(self.bloch)}, {len(self.shape)}).')

    @property
    def mat(self):
        """Build the operator :math:`A(\omega)` acting on the electric field :math:`e`,
        which is a discretized version of Maxwell's equations in frequency domain:
        .. math::
            \nabla \times \mu^{-1} \nabla \times e - \omega^2 \epsilon e = i \omega J,
        which can be written in the form :math:`A e = b`, where:
        .. math::
            A = \nabla \times \mu^{-1} \nabla \times  - \omega^2 \epsilon \\
            b = i \omega J
        is an operator representing the discretized EM wave operator at frequency :math:`omega`.

        Returns:
            Electric field operator :math:`A` for solving Maxwell's equations at frequency :math:`omega`.
        """

        curl_f, curl_b = self._curls
        return curl_b @ curl_f / MU_0 - self.omega ** 2 * EPS_0 * sp.kron(sp.eye(3), sp.diags(self.eps.flat))

    A = mat  # alias A (common symbol for FDFD matrix) to mat

    def solve(self, src: np.ndarray, solver_fn: SpSolve):
        return solver_fn(self.mat, src)

    # def modes(self, num_modes, axis: int = 0):
    #     return axis

    @property
    def _curls(self):
        """Gets the forward curl(`curl_f`) and backward curl (`curl_b`) for the FDFD matrix (`FDFD.mat` property)

        Returns:
            Forward and backward curls
        """

        # account for 1d and 2d cases
        b = np.hstack((self.bloch, np.ones((3 - self.dim,), dtype=self.bloch.dtype)))
        s = np.hstack((self.shape, np.ones((3 - self.dim,), dtype=self.shape.dtype)))

        # define grid cell sizes (including pml if necessary)
        dx_f, dx_b = self._dxes
        dx_f = np.meshgrid(*dx_f, indexing='ij')
        dx_b = np.meshgrid(*dx_b, indexing='ij')

        # get forward derivative
        df = [sp.diags([-1, 1, b[ax]], [0, 1, -n + 1], shape=(n, n), dtype=np.complex128)
              if n > 1 else 0 for ax, n in enumerate(s)]  # get single axis forward-derivs
        df = kron_tile(df, s)  # tile over the other axes using sp.kron
        df = [sp.diags(1 / dx_f[ax].ravel()) @ df[ax] for ax in range(len(s))]  # scale by dx (incl pml)

        # get backward derivative
        db = [sp.diags([1, -1, -np.conj(b[ax])], [0, -1, n - 1], shape=(n, n), dtype=np.complex128)
              if n > 1 else 0 for ax, n in enumerate(s)]  # get single axis back-derivs
        db = kron_tile(db, s)  # tile over the other axes using sp.kron
        db = [sp.diags(1 / dx_b[ax].ravel()) @ db[ax] for ax in range(len(s))]  # scale by dx (incl pml)

        # get curls
        o = sp.csr_matrix((self.n, self.n))  # zeros
        curl_f = sp.bmat([[o, -df[2], df[1]],
                          [df[2], o, -df[0]],
                          [-df[1], df[0], o]])
        curl_b = sp.bmat([[o, -db[2], db[1]],
                          [db[2], o, -db[0]],
                          [-db[1], db[0], o]])

        return curl_f, curl_b

    @property
    def _dxes(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Conditional transformation of self.dxes based on stretched-coordinated perfectly matched layers (SC-PML)

        Returns:
            SC-PML transformation of dxes

        """

        if self.pml_shape is None:
            return self.dxes, self.dxes
        else:
            dxes_pml_e, dxes_pml_h = ([], [])
            for ax, p in enumerate(self.pos):
                if self.dxes[ax].size == 1:
                    dxes_pml_e.append(self.dxes[ax])
                    dxes_pml_h.append(self.dxes[ax])
                else:
                    pe, ph = (p[:-1] + p[1:]) / 2, p[:-1]
                    dxes_pml_e.append(self.dxes[ax] * self._scpml(pe, ax))
                    dxes_pml_h.append(self.dxes[ax] * self._scpml(ph, ax))
            return dxes_pml_e, dxes_pml_h

    def _scpml(self, d: np.ndarray, ax: int, exp_scale: float = 4, log_reflection: float = -16):
        absorption_corr = self.omega * self.boundary_eps
        t = self.pml_shape[ax]
        d_pml = np.hstack((
            (d[:t] - np.min(d)) / (d[t] - np.min(d)),
            np.zeros_like(d[t:-t]),
            (np.max(d) - d[-t:]) / (np.max(d) - d[-t])
        ))
        return 1 + 1j * (exp_scale + 1) * log_reflection / 2 * d_pml ** exp_scale / absorption_corr
