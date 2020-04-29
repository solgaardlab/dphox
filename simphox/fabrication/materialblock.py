import numpy as np
from shapely.geometry import LineString, Polygon, LinearRing, MultiLineString
from typing import Tuple, List, Optional

from .material import Material
from ..fabrication.utils import get_point_in_profile, get_point_idx_in_profile


class MaterialBlock:
    def __init__(self, center: Tuple[float, float], dim: Tuple[float, float, float], material: Material,
                 layer: int = 0, name: str = None, planarize: bool = False, full_length: bool=True):
        self.center = np.asarray(center)
        self.dim = np.asarray(dim)
        self.min = self.center - self.dim[:-1] / 2
        self.max = self.center + self.dim[:-1] / 2
        self.layer = layer
        self.name = name
        self.material = material
        self.planarize = planarize
        self.full_length = full_length

    def substrate_deposit(self, height: float, axis: Optional[int] = 1) -> Polygon:
        polygon = LineString(
            coordinates=(
                [self.min[axis], height],
                [self.max[axis], height]
            )
        ).buffer(self.dim[2] / 2, cap_style=2)
        return polygon

    def deposit(self, base_profile: LineString, existing_layers: List[Polygon], axis: Optional[int] = 1):
        path_offset = LineString(base_profile).parallel_offset(-self.dim[2])
        if isinstance(path_offset, MultiLineString):  # fix this
            x = np.hstack([path_offset[0].xy[0], path_offset[1].xy[0]])
            y = np.hstack([path_offset[0].xy[1], path_offset[1].xy[1]])
        else:
            x = path_offset.xy[0]
            y = path_offset.xy[1]

        block_offset_array = np.asarray([x, y]).T

        min_point = np.asarray(get_point_in_profile(self.min[axis], block_offset_array))
        min_idx = get_point_idx_in_profile(self.min[axis], block_offset_array)
        max_point = np.asarray(get_point_in_profile(self.max[axis], block_offset_array))
        max_idx = get_point_idx_in_profile(self.max[axis], block_offset_array)

        # TODO(sunil): If the edge point(s) are missing from shapely's offset curve...
        # there may be a better solution.
        if min_idx is None:
            min_point = np.asarray((base_profile.coords.xy[0][0], base_profile.coords.xy[1][0] + self.dim[2]))
            min_idx = 0
        if max_idx is None:
            max_point = np.asarray((base_profile.coords.xy[0][-1], base_profile.coords.xy[1][-1] + self.dim[2]))
            max_idx = -1

        block_offset_array_bounded = block_offset_array[min_idx + 1:max_idx]

        block_ring = LinearRing(np.vstack([
            np.asarray((min_point[0], 0)),
            min_point,
            [] if self.planarize else block_offset_array_bounded,
            max_point,
            np.asarray((max_point[0], 0)),
        ]))

        polygon = Polygon(block_ring)
        for poly_to_subtract in existing_layers:
            polygon = polygon - poly_to_subtract
        return polygon
