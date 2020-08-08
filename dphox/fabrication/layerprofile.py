import numpy as np
from shapely.geometry import LineString, LinearRing, Polygon, MultiLineString

from .materialblock import MaterialBlock
from .utils import get_point_in_profile, get_point_idx_in_profile
from typing import List, Optional


class LayerProfile:
    def __init__(self, span: float, base_blocks: List[MaterialBlock], base_holes: List[MaterialBlock]=None,
                 axis: Optional[int]=1):

        self.base_path = []
        edges = []
        self.base_path.append((-span / 2, 0))
        self.axis = axis

        # Rising Edges

        for block in sorted(base_blocks, key=lambda b: b.min[self.axis]):
            edges.append([
                (block.min[self.axis], 0),
                (block.min[self.axis], block.dim[2])
            ])
        if base_holes:
            for block in sorted(base_holes, key=lambda b: b.max[self.axis]):
                edges.append([
                    (block.max[self.axis], 0),
                    (block.max[self.axis], block.dim[2])
                ])

        # Falling Edges

        for block in sorted(base_blocks, key=lambda b: b.max[self.axis]):
            edges.append([
                (block.max[self.axis], block.dim[2]),
                (block.max[self.axis], 0)
            ])
        if base_holes:
            for block in sorted(base_holes, key=lambda b: b.min[self.axis]):
                edges.append([
                    (block.min[self.axis], block.dim[2]),
                    (block.min[self.axis], 0)
                ])

        # Sort edges and form profile

        sorted_edges = sorted(edges, key=lambda e: e[0][0])

        for edge in sorted_edges:
            self.base_path += edge

        self.base_path.append((span / 2, 0))

        self.min = -span / 2
        self.max = span / 2

        self.path = LineString(self.base_path)

    def add_block(self, block: MaterialBlock, include_edges=False):
        path_offset = LineString(self.path).parallel_offset(-block.dim[2])

        if isinstance(path_offset, MultiLineString):  # fix this
            x = np.hstack([path_offset[0].xy[0], path_offset[1].xy[0]])
            y = np.hstack([path_offset[0].xy[1], path_offset[1].xy[1]])
        else:
            x = path_offset.xy[0]
            y = path_offset.xy[1]

        block_offset_array = np.asarray([x, y]).T

        # block_offset_array = np.asarray([path_offset.coords.xy[0], path_offset.coords.xy[1]]).T
        path_xy = np.asarray([self.path.coords.xy[0], self.path.coords.xy[1]]).T

        min_point_base = np.asarray(get_point_in_profile(block.min[self.axis], path_xy))
        max_point_base = np.asarray(get_point_in_profile(block.max[self.axis], path_xy))
        min_point_layer = np.asarray(get_point_in_profile(block.min[self.axis], block_offset_array))
        max_point_layer = np.asarray(get_point_in_profile(block.max[self.axis], block_offset_array))
        min_idx = get_point_idx_in_profile(block.min[self.axis], block_offset_array)
        max_idx = get_point_idx_in_profile(block.max[self.axis], block_offset_array)

        # TODO(sunil): If the edge point(s) are missing from shapely's offset curve... there may be a better solution.
        if min_idx is None:
            min_point_layer = np.asarray((self.path.coords.xy[0][0], self.path.coords.xy[1][0] + block.dim[2]))
            min_idx = 0
        if max_idx is None:
            max_point_layer = np.asarray((self.path.coords.xy[0][-1], self.path.coords.xy[1][-1] + block.dim[2]))
            max_idx = -1

        block_offset_array_bounded = block_offset_array[min_idx + 1:max_idx]

        path_xy = np.asarray([self.path.coords.xy[0], self.path.coords.xy[1]]).T

        if include_edges:
            left_path = path_xy[np.where(np.logical_and(block.min[self.axis] > path_xy[:, 0], path_xy[:, 0] >= self.min))]
            right_path = path_xy[np.where(np.logical_and(block.max[self.axis] < path_xy[:, 0], path_xy[:, 0] <= self.max))]
        else:
            left_path = path_xy[np.where(np.logical_and(block.min[self.axis] > path_xy[:, 0], path_xy[:, 0] > self.min))]
            right_path = path_xy[np.where(np.logical_and(block.max[self.axis] < path_xy[:, 0], path_xy[:, 0] < self.max))]

        path_point_lists = [
            left_path,
            min_point_base if min_point_base[0] > self.min else None,
            min_point_layer,
            block_offset_array_bounded,
            max_point_layer,
            max_point_base if max_point_base[0] < self.max else None,
            right_path
        ]
        self.path = LineString(np.vstack([l for l in path_point_lists if l is not None]))

    def etch(self, block: MaterialBlock, include_edges: bool=False):
        # TODO(sunil): Assumes flat surface (probably a good assumption?)
        path_xy = np.asarray([self.path.coords.xy[0], self.path.coords.xy[1]]).T
        min_point_base = np.asarray(get_point_in_profile(block.min[self.axis], path_xy))
        max_point_base = np.asarray(get_point_in_profile(block.max[self.axis], path_xy))
        if min_point_base.size == 0:
            min_point_base = np.asarray((self.path.coords.xy[0][0], self.path.coords.xy[1][0]))
        if max_point_base.size == 0:
            max_point_base = np.asarray((self.path.coords.xy[0][-1], self.path.coords.xy[1][-1]))

        # left_path = path_xy[np.where(np.logical_and(block.min[self.axis] > path_xy[:, 0], path_xy[:, 0] > self.min))]
        # right_path = path_xy[np.where(np.logical_and(block.max[self.axis] < path_xy[:, 0], path_xy[:, 0] < self.max))]

        if include_edges:
            left_path = path_xy[np.where(np.logical_and(block.min[self.axis] > path_xy[:, 0], path_xy[:, 0] >= self.min))]
            right_path = path_xy[np.where(np.logical_and(block.max[self.axis] < path_xy[:, 0], path_xy[:, 0] <= self.max))]
        else:
            left_path = path_xy[np.where(np.logical_and(block.min[self.axis] > path_xy[:, 0], path_xy[:, 0] > self.min))]
            right_path = path_xy[np.where(np.logical_and(block.max[self.axis] < path_xy[:, 0], path_xy[:, 0] < self.max))]

        path_point_lists = [
            left_path,
            min_point_base,
            np.asarray((min_point_base[0], min_point_base[1] - block.dim[2])),
            np.asarray((max_point_base[0], max_point_base[1] - block.dim[2])),
            max_point_base,
            right_path
        ]

        self.path = LineString(np.vstack([l for l in path_point_lists if l is not None]))
        return Polygon(LinearRing(path_point_lists[1:-1]))
