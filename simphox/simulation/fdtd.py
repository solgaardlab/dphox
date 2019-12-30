import numpy as np

from .grid import SimGrid
from ..typing import Shape, Dim, GridSpacing, Optional, Union


class FDTD(SimGrid):
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray] = 1,
                 pml: Optional[Union[Shape, Dim]] = None):
        super(FDTD, self).__init__(shape, spacing, eps, pml=pml)
