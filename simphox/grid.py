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
            p = self.pos[2]
            zmaxidx = p.flat[np.abs(p - zmax).argmin()]
            self.eps[:, :, :zmaxidx] = eps
        else:
            raise ValueError('dim must be 3 to fill grid')

    def add(self, component: Component, zmin: float = None, thickness: float = None):
        p = self.pos[2]
        zidx = (p.flat[np.abs(p - zmin).argmin()], p.flat[np.abs(p - zmin - thickness).argmin()])
        b = component.bounds
        self.components.append(component)
        if not self._check_bounds(component):
            raise ValueError('The pattern is out of bounds')
        if self.dim == 2:
            self.eps[b[0]:b[2], b[1]:b[3]] = component.mask(self.pos)
        else:
            self.eps[b[0]:b[2], b[1]:b[3], zidx[0]:zidx[1]] = component.mask(self.pos)
