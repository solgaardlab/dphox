from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from ..component import Component
from ..typing import Shape, Dim, GridSpacing, Optional, Tuple, List, Union
from ..utils import d2curl_op, d2curl_fn, grid_average


class Grid:
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray] = 1.0):
        """Grid object accomodating any electromagnetic simulation strategy (FDFD, FDTD, BPM, etc.)

        Args:
            shape: Tuple of size 1, 2, or 3 representing the number of pixels in the grid
            spacing: Spacing (microns) between each pixel along each axis (must be same dim as `grid_shape`)
            eps: Relative permittivity
        """
        self.shape = np.asarray(shape, dtype=np.int)
        self.spacing = spacing * np.ones(len(shape)) if isinstance(spacing, float) else np.asarray(spacing)
        self.ndim = len(shape)
        self.shape3 = np.hstack((self.shape, np.ones((3 - self.ndim,), dtype=self.shape.dtype)))
        if not self.ndim == len(self.spacing):
            raise AttributeError(f'Require len(grid_shape) == len(grid_spacing) but got'
                                 f'({len(self.shape)}, {len(self.spacing)})')
        self.n = np.prod(self.shape)
        self.eps: np.ndarray = np.ones(self.shape) * eps if not isinstance(eps, np.ndarray) else eps
        if not tuple(self.shape) == self.eps.shape:
            raise AttributeError(f'Require grid_shape == eps.shape but got'
                                 f'({self.shape}, {self.eps.shape})')
        self.size = self.spacing * self.shape
        self.cell_sizes = [self.spacing[i] * np.ones((self.shape[i],))
                           if i < self.ndim else np.ones((1,)) for i in range(3)]
        self.pos = [np.hstack((0, np.cumsum(dx))) if dx.size > 1 else None for dx in self.cell_sizes]
        if self.ndim == 1:
            self.mesh_pos = np.mgrid[:self.size[0]:self.spacing[0]]
        elif self.ndim == 2:
            self.mesh_pos = np.stack(np.mgrid[:self.size[0]:self.spacing[0], :self.size[1]:self.spacing[1]],
                                     axis=0)
        else:
            self.mesh_pos = np.stack(np.mgrid[:self.size[0]:self.spacing[0],
                                     :self.size[1]:self.spacing[1],
                                     :self.size[2]:self.spacing[2]], axis=0)
        self.components = []

    def _check_bounds(self, component) -> bool:
        b = component.bounds
        return b[0] >= 0 and b[1] >= 0 and b[2] <= self.size[0] and b[3] <= self.size[1]

    def fill(self, zmax: float, eps: float):
        """Fill grid up to `zmax`, typically used for substrate + cladding epsilon settings

        Args:
            zmax: Maximum z (or final dimension) of the fill operation
            eps: Relative eps to fill

        Returns:

        """
        self.eps[..., :int(zmax / self.spacing[-1])] = eps

    def add(self, component: Component, eps: float, zmin: float = None, thickness: float = None):
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
        if self.ndim == 2:
            self.eps[mask == 1] = eps
        else:
            zidx = (int(zmin / self.spacing[0]), int((zmin + thickness) / self.spacing[1]))
            self.eps[:, :, zidx[0]:zidx[1]] += (mask * eps)[..., np.newaxis] - self.eps[:, :, zidx[0]:zidx[1]]

    def reshape(self, v: np.ndarray) -> np.ndarray:
        """A simple method to reshape flat 3d vec array into the same shape

        Args:
            v: vector of size `(3n,)` to rearrange into array of size `(3n,)`

        Returns:

        """
        return np.stack([split_v.reshape(self.shape3) for split_v in np.split(v, 3)])


class SimGrid(Grid):
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray] = 1,
                 bloch_phase: Union[Dim, float] = 0.0, pml: Optional[Union[int, Shape, Dim]] = None,
                 pml_eps: float = 1.0, grid_avg: bool = True):
        super(SimGrid, self).__init__(shape, spacing, eps)
        self.pml_shape = np.asarray(pml, dtype=np.int) if isinstance(pml, tuple) else pml
        self.pml_shape = np.ones(self.ndim, dtype=np.int) * pml if isinstance(pml, int) else pml
        self.pml_eps = pml_eps
        self.grid_avg = grid_avg
        if self.pml_shape is not None:
            if np.any(self.pml_shape <= 3) or np.any(self.pml_shape >= self.shape // 2):
                raise AttributeError(f'PML shape must be more than 3 and less than half the shape on each axis.')
        if pml is not None and not len(self.pml_shape) == len(self.shape):
            raise AttributeError(f'Need len(pml_shape) == len(grid_shape),'
                                 f'got ({len(pml)}, {len(self.shape)}).')
        self.bloch = np.ones_like(self.shape) * np.exp(1j * np.asarray(bloch_phase)) if isinstance(bloch_phase, float) \
            else np.exp(1j * np.asarray(bloch_phase))
        if not len(self.bloch) == len(self.shape):
            raise AttributeError(f'Need len(bloch_phase) == len(grid_shape),'
                                 f'got ({len(self.bloch)}, {len(self.shape)}).')

    @lru_cache()
    def deriv(self, back: bool = False) -> List[sp.spmatrix]:
        """Calculate directional derivative (cached, since this does not depend on any params)

        Args:
            back: Return backward derivative

        Returns:
            Discrete directional derivative `d` of the form `(d_x, d_y, d_z)`

        """

        # account for 1d and 2d cases
        b = np.hstack((self.bloch, np.ones((3 - self.ndim,), dtype=self.bloch.dtype)))
        s = np.hstack((self.shape, np.ones((3 - self.ndim,), dtype=self.shape.dtype)))

        if back:
            # get backward derivative
            _, dx = self._dxes
            d = [sp.diags([1, -1, -np.conj(b[ax])], [0, -1, n - 1], shape=(n, n), dtype=np.complex128)
                 if n > 1 else 0 for ax, n in enumerate(s)]  # get single axis back-derivs
        else:
            # get forward derivative
            dx, _ = self._dxes
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

    def curl_e(self, e) -> np.ndarray:
        dx, _ = self._dxes

        def de(e_, d):
            return (np.roll(e_, -1, axis=d) - e_) / dx[d]

        return d2curl_fn(e, de)

    def curl_h(self, h) -> np.ndarray:
        _, dx = self._dxes

        def dh(h_, d):
            return (h_ - np.roll(h_, 1, axis=d)) / dx[d]

        return d2curl_fn(h, dh)

    @property
    def _dxes(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Conditional transformation of self.dxes (will need to be extended by some other class)

        Returns:
            dxes for e and h fields, respectively.
        """
        return np.meshgrid(*self.cell_sizes, indexing='ij'), np.meshgrid(*self.cell_sizes, indexing='ij')

    @property
    def eps_t(self):
        return grid_average(self.eps) if self.grid_avg else np.stack((self.eps, self.eps, self.eps))

