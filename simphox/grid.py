import numpy as np

from .component import Component
from .typing import GridShape, GridSpacing


class Grid:
    def __init__(self, shape: GridShape, grid_spacing: GridSpacing, eps: float = 1):
        self.shape = np.asarray(shape)
        self.grid_spacing = grid_spacing * np.ones(len(shape)) if isinstance(grid_spacing, float) else np.asarray(
            grid_spacing)
        self.dim = len(shape)
        self.eps = np.ones(self.shape) * eps
        self.size = self.grid_spacing * self.shape
        if self.dim == 3:
            self.pos = np.stack(np.meshgrid(
                np.arange(0, self.size[0], self.grid_spacing[0]),
                np.arange(0, self.size[1], self.grid_spacing[1]),
                np.arange(0, self.size[2], self.grid_spacing[2])
            ), axis=0)
        else:
            self.pos = np.stack(np.meshgrid(
                np.arange(0, self.size[0], self.grid_spacing[0]),
                np.arange(0, self.size[1], self.grid_spacing[1])
            ), axis=0)
        self.components = []

    def _check_bounds(self, component):
        b = component.bounds
        return b[0] >= 0 and b[1] >= 0 and b[2] <= self.size[0] and b[3] <= self.size[1]

    def fill(self, zmax: float, eps: float):
        if self.dim == 3:
            self.eps[:, :, :int(zmax / self.grid_spacing[2])] = eps
        else:
            raise ValueError('dim must be 3 to fill grid')

    def add(self, component: Component, eps: float, zmin: float = None, thickness: float = None):
        if not self._check_bounds(component):
            raise ValueError('The pattern is out of bounds')
        self.components.append(component)
        mask = component.mask(self.shape[:2], self.grid_spacing)
        if self.dim == 2:
            self.eps = mask * eps
        else:
            zidx = (int(zmin / self.grid_spacing[0]), int((zmin + thickness) / self.grid_spacing[1]))
            self.eps[:, :, zidx[0]:zidx[1]] += (mask * eps)[..., np.newaxis]
