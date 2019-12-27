import numpy as np

from ..component import Component
from ..ops import grid_average
from ..typing import Shape, GridSpacing, Union


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
        self.spacing *= 1e-6
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

    @property
    def eps_t(self):
        return grid_average(self.eps) if self.enable_grid_averaging else np.stack((self.eps, self.eps, self.eps))

    @property
    def mu_t(self):
        return grid_average(self.mu) if self.enable_grid_averaging else np.stack((self.mu, self.mu, self.mu))
