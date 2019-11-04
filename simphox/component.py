import gdspy as gy
import numpy as np

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union
from descartes import PolygonPatch
from skimage.draw import polygon2mask

from .typing import List, Dim2, GridShape


class Path(gy.Path):
    def sbend(self, bend_dim: Dim2, layer: int=0, inverted: bool=False):
        pole_1 = np.asarray((bend_dim[0] / 2, 0))
        pole_2 = np.asarray((bend_dim[0] / 2, (-1) ** inverted * bend_dim[1]))
        pole_3 = np.asarray((bend_dim[0], (-1) ** inverted * bend_dim[1]))
        self.bezier([pole_1, pole_2, pole_3], layer=layer)
        return self

    def dc(self, bend_dim: Dim2, interaction_length: float, end_length: float=0, layer: int=0, inverted: bool=False):
        self.segment(end_length)
        self.sbend(bend_dim, layer, inverted)
        self.segment(interaction_length, layer=layer)
        self.sbend(bend_dim, layer, not inverted)
        self.segment(end_length)
        return self

    def mzi(self, bend_dim: Dim2, interaction_length: float, arm_length: float,
            end_length: float=0, layer: int=0, inverted: bool=False):
        self.segment(end_length)
        self.sbend(bend_dim, layer, inverted)
        self.segment(interaction_length, layer=layer)
        self.sbend(bend_dim, layer, not inverted)
        self.segment(arm_length)
        self.sbend(bend_dim, layer, inverted)
        self.segment(interaction_length, layer=layer)
        self.sbend(bend_dim, layer, not inverted)
        self.segment(end_length)
        return self

    @property
    def shapely(self):
        polygon_list = [Polygon(polygon_point_list) for polygon_point_list in self.polygons]
        pattern = cascaded_union(polygon_list)
        return pattern


class Component:
    def __init__(self, paths: List[Path]):
        self.paths = paths

    @property
    def shapely(self):
        polygon_list = []
        for path in self.paths:
            polygon_list += [Polygon(polygon_point_list) for polygon_point_list in path.polygons]
        return MultiPolygon(polygon_list)

    def mask(self, shape: np.ndarray, grid_spacing: np.ndarray):
        mask = np.zeros(shape)
        for p in self.shapely:
            mask += polygon2mask(shape, (p.exterior.coords.xy / grid_spacing[:, np.newaxis]).T)
        return mask

    @property
    def bounds(self):
        return self.shapely.bounds

    @property
    def size(self):
        b = self.bounds  # (minx, miny, maxx, maxy)
        return b[2] - b[0], b[3] - b[1]  # (maxx - minx, maxy - miny)

    @property
    def center(self):
        b = self.bounds  # (minx, miny, maxx, maxy)
        return (b[2] + b[0]) / 2, (b[3] + b[1]) / 2  # (avgx, avgy)

    def translate(self, dx: float, dy: float):
        for path in self.paths:
            path.translate(dx, dy)
        return self

    def to_gds(self, cell: gy.Cell):
        for path in self.paths:
            cell.add(path)

    def plot(self, ax, color):
        ax.add_patch(PolygonPatch(self.shapely, facecolor=color, edgecolor='none'))
        b = self.bounds
        ax.set_xlim((b[0], b[2]))
        ax.set_ylim((b[1], b[3]))
        ax.set_aspect('equal')


class DC(Component):
    def __init__(self, bend_dim: Dim2, waveguide_width: float,
                 coupling_spacing: float, interaction_length: float, end_length: float=0):
        self.end_length = end_length
        self.bend_dim = bend_dim
        self.waveguide_width = waveguide_width
        self.interaction_length = interaction_length
        self.coupling_spacing = coupling_spacing

        lower_path = Path(waveguide_width).dc(bend_dim, interaction_length, end_length)
        upper_path = Path(waveguide_width).dc(bend_dim, interaction_length, end_length, inverted=True)
        upper_path.translate(dx=0, dy=waveguide_width + 2 * bend_dim[1] + coupling_spacing)

        super(DC, self).__init__([lower_path, upper_path])


class MZI(Component):
    def __init__(self, bend_dim: Dim2, waveguide_width: float, arm_length: float,
                 coupling_spacing: float, interaction_length: float, end_length: float=0):
        self.end_length = end_length
        self.arm_length = arm_length
        self.bend_dim = bend_dim
        self.waveguide_width = waveguide_width
        self.interaction_length = interaction_length
        self.coupling_spacing = coupling_spacing

        lower_path = Path(waveguide_width).mzi(bend_dim, interaction_length, arm_length, end_length)
        upper_path = Path(waveguide_width).mzi(bend_dim, interaction_length, arm_length, end_length, inverted=True)
        upper_path.translate(dx=0, dy=waveguide_width + 2 * bend_dim[1] + coupling_spacing)

        super(MZI, self).__init__([lower_path, upper_path])


class MMI(Component):
    def __init__(self, box_dim: Dim2, waveguide_width: float, interport_distance: float,
                 taper_dim: Dim2, end_length: float=0):
        self.end_length = end_length
        self.waveguide_width = waveguide_width
        self.box_dim = box_dim
        self.interport_distance = interport_distance
        self.taper_dim = taper_dim

        lower_input_path = Path(waveguide_width).segment(end_length).segment(taper_dim[0], final_width=taper_dim[1])
        upper_input_path = Path(waveguide_width).segment(end_length).segment(taper_dim[0], final_width=taper_dim[1])
        upper_input_path.translate(dx=0, dy=interport_distance)
        lower_output_path = Path(taper_dim[1]).segment(end_length).segment(taper_dim[0], final_width=waveguide_width)
        upper_output_path = Path(taper_dim[1]).segment(end_length).segment(taper_dim[0], final_width=waveguide_width)
        lower_output_path.translate(dx=end_length + taper_dim[0] + box_dim[0], dy=0)
        upper_output_path.translate(dx=end_length + taper_dim[0] + box_dim[0], dy=interport_distance)
        box = Path(box_dim[1], (end_length + taper_dim[0], interport_distance / 2)).segment(box_dim[0])

        super(MMI, self).__init__([lower_input_path, upper_input_path, lower_output_path, upper_output_path, box])
