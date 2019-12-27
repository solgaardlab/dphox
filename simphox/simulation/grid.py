from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from ..component import Component
from ..ops import d2curl_op, d2curl_fn
from ..ops import grid_average
from ..typing import Shape, Dim, GridSpacing, Optional, Tuple, List, Union, Op


class Grid:
    def __init__(self, grid_shape: Shape, grid_spacing: GridSpacing, eps: Union[float, np.ndarray] = 1.0,
                 mu: Union[float, np.ndarray] = 1.0, enable_grid_averaging: bool = True):
        """Grid object accomodating any electromagnetic simulation strategy (FDFD, FDTD, BPM, etc.)

        Args:
            grid_shape: Tuple of size 1, 2, or 3 representing the number of pixels in the grid
            grid_spacing: Spacing (microns) between each pixel along each axis (must be same dim as `grid_shape`)
            eps: Relative permittivity
            mu: Relative permeability
            enable_grid_averaging: Enable grid averaging for the simulation
        """
        self.shape = np.asarray(grid_shape, dtype=np.int)
        self.spacing = grid_spacing * np.ones(len(grid_shape)) if isinstance(grid_spacing, float) else np.asarray(
            grid_spacing)
        self.dim = len(grid_shape)
        if not self.dim == len(self.spacing):
            raise AttributeError(f'Require len(grid_shape) == len(grid_spacing) but got'
                                 f'({len(self.shape)}, {len(self.spacing)})')
        self.n = np.prod(self.shape)
        self.eps: np.ndarray = np.ones(self.shape) * eps if not isinstance(eps, np.ndarray) else eps
        self.mu: np.ndarray = np.ones(self.shape) * mu if not isinstance(mu, np.ndarray) else mu
        if not tuple(self.shape) == self.eps.shape:
            raise AttributeError(f'Require grid_shape == eps.shape but got'
                                 f'({self.shape}, {self.eps.shape})')
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
        self.enable_grid_averaging = enable_grid_averaging

    def _check_bounds(self, component):
        b = component.bounds
        return b[0] >= 0 and b[1] >= 0 and b[2] <= self.size[0] and b[3] <= self.size[1]

    def fill(self, zmax: float, eps: float) -> None:
        """Fill grid up to `zmax`

        Args:
            zmax: Maximum z of the fill operation
            eps: Relative eps to fill

        Returns:

        """
        if self.dim == 3:
            self.eps[:, :, :int(zmax / self.spacing[2])] = eps
        else:
            raise ValueError('dim must be 3 to fill grid')

    def add(self, component: Component, eps: float, zmin: float = None, thickness: float = None) -> None:
        """Add a component to the grid

        Args:
            component: component to add
            eps: permittivity of the component being added (isotropic only, for now)
            zmin: minimum z extent of the component
            thickness: component thickness (`zmax = zmin + thickness`)

        Returns:

        """
        if not self._check_bounds(component):
            raise ValueError('The pattern is out of bounds')
        self.components.append(component)
        mask = component.mask(self.shape[:2], self.spacing)
        if self.dim == 2:
            self.eps = mask * eps
        else:
            zidx = (int(zmin / self.spacing[0]), int((zmin + thickness) / self.spacing[1]))
            self.eps[:, :, zidx[0]:zidx[1]] += (mask * eps)[..., np.newaxis]

    def reshape(self, v: np.ndarray) -> np.ndarray:
        """A simple method to reshape flat 3d vec array into the same shape

        Args:
            v: vector of size `(3n,)` to rearrange into array of size `(3n,)`

        Returns:

        """
        return np.stack([split_v.reshape(self.shape) for split_v in np.split(v, 3)])


class SimGrid(Grid):
    def __init__(self, grid_shape: Shape, grid_spacing: GridSpacing, eps: Union[float, np.ndarray] = 1,
                 pml_shape: Optional[Union[Shape, Dim]] = None, pml_eps: float = 1.0):
        super(SimGrid, self).__init__(grid_shape, grid_spacing, eps)
        self.pml_shape = np.asarray(pml_shape, dtype=np.int) if pml_shape else None
        self.pml_eps = pml_eps
        if self.pml_shape:
            if np.any(self.pml_shape <= 3) or np.any(self.pml_shape >= self.shape // 2):
                raise AttributeError(f'PML shape must be more than 3 and less than half the shape on each axis.')
        if pml_shape is not None and not len(pml_shape) == len(self.shape):
            raise AttributeError(f'Need len(pml_shape) == len(grid_shape),'
                                 f'got ({len(pml_shape)}, {len(self.shape)}).')

    @lru_cache()
    def deriv(self, back: bool = False) -> List[sp.spmatrix]:
        """Calculate directional derivative (cached, since this does not depend on any params)

        Args:
            back: Return backward derivative

        Returns:
            Discrete directional derivative `d` of the form `(d_x, d_y, d_z)`

        """

        # account for 1d and 2d cases
        b = np.hstack((self.bloch, np.ones((3 - self.dim,), dtype=self.bloch.dtype)))
        s = np.hstack((self.shape, np.ones((3 - self.dim,), dtype=self.shape.dtype)))

        # define grid cell sizes (including pml if necessary)
        dx_f, dx_b = self._dxes

        if back:
            # get backward derivative
            dx = np.meshgrid(*dx_b, indexing='ij')
            d = [sp.diags([1, -1, -np.conj(b[ax])], [0, -1, n - 1], shape=(n, n), dtype=np.complex128)
                 if n > 1 else 0 for ax, n in enumerate(s)]  # get single axis back-derivs
        else:
            # get forward derivative
            dx = np.meshgrid(*dx_f, indexing='ij')
            d = [sp.diags([-1, 1, b[ax]], [0, 1, -n + 1], shape=(n, n), dtype=np.complex128)
                 if n > 1 else 0 for ax, n in enumerate(s)]  # get single axis forward-derivs
        d = [sp.kron(d[0], sp.eye(s[1] * s[2])),
             sp.kron(sp.kron(sp.eye(s[0]), d[1]), sp.eye(s[2])),
             sp.kron(sp.eye(s[0] * s[1]), d[2])]  # tile over the other axes using sp.kron
        d = [sp.diags(1 / dx[ax].ravel()) @ d[ax] for ax in range(len(s))]  # scale by dx (incl pml)

        return d

    @property
    @lru_cache()
    def df(self):
        return self.deriv()

    @property
    @lru_cache()
    def db(self):
        return self.deriv(back=True)

    @property
    @lru_cache()
    def curl_f(self):
        return d2curl_op(self.df)

    @property
    @lru_cache()
    def curl_b(self):
        return d2curl_op(self.db)

    @property
    @lru_cache()
    def curl_e(self) -> Op:
        dx_f, dx_b = self._dxes
        dx = np.meshgrid(*dx_f, indexing='ij')

        def de(e, d):
            return (np.roll(e[d], -1, axis=d) - e[d]) / dx[d]

        return lambda e: d2curl_fn(e, de)

    @property
    @lru_cache()
    def curl_h(self) -> Op:
        dx_f, dx_b = self._dxes
        dx = np.meshgrid(*dx_b, indexing='ij')

        def dh(h, d):
            return (h[d] - np.roll(h[d], 1, axis=d)) / dx[d]

        return lambda h: d2curl_fn(h, dh)

    @property
    def _dxes(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Conditional transformation of self.dxes (will need to be extended by some other class)

        Returns:
            dxes for e and h fields, respectively.
        """
        return self.dxes, self.dxes

    @property
    def eps_t(self):
        return grid_average(self.eps) if self.enable_grid_averaging else np.stack((self.eps, self.eps, self.eps))

    @property
    def mu_t(self):
        return grid_average(self.mu) if self.enable_grid_averaging else np.stack((self.mu, self.mu, self.mu))
