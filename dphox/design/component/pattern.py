from collections import defaultdict

import gdspy as gy
import nazca as nd
from copy import deepcopy as copy
from shapely.vectorized import contains
from shapely.geometry import Polygon, MultiPolygon, CAP_STYLE
from shapely.ops import cascaded_union
from shapely.affinity import translate
from descartes import PolygonPatch
import trimesh
from trimesh import creation, visual

try:
    import plotly.graph_objects as go
except ImportError:
    pass

from ...typing import *


class Path(gy.Path):
    def polynomial_taper(self, length: float, taper_params: Tuple[float, ...],
                         num_taper_evaluations: int = 100, layer: int = 0, inverted: bool = False):
        curr_width = self.w * 2
        taper_params = np.asarray(taper_params)
        self.parametric(lambda u: (length * u, 0),
                        lambda u: (1, 0),
                        final_width=lambda u: curr_width - np.sum(taper_params) +
                                              np.sum(taper_params * (1 - u) ** np.arange(taper_params.size,
                                                                                         dtype=float)) if inverted
                        else curr_width + np.sum(taper_params * u ** np.arange(taper_params.size, dtype=float)),
                        number_of_evaluations=num_taper_evaluations,
                        layer=layer)
        return self

    def sbend(self, bend_dim: Dim2, layer: int = 0, inverted: bool = False, use_radius: bool = False):
        if use_radius is False:
            pole_1 = np.asarray((bend_dim[0] / 2, 0))
            pole_2 = np.asarray((bend_dim[0] / 2, (-1) ** inverted * bend_dim[1]))
            pole_3 = np.asarray((bend_dim[0], (-1) ** inverted * bend_dim[1]))
            self.bezier([pole_1, pole_2, pole_3], layer=layer)
        else:
            if bend_dim[1] > 2 * bend_dim[0]:
                angle = np.pi / 2 * (-1) ** inverted
                self.turn(bend_dim[0], angle, number_of_points=199)
                self.segment(bend_dim[1] - 2 * bend_dim[0])
                self.turn(bend_dim[0], -angle, number_of_points=199)
            else:
                angle = np.arccos(1 - bend_dim[1] / 2 / bend_dim[0]) * (-1) ** inverted
                self.turn(bend_dim[0], angle, number_of_points=199)
                self.turn(bend_dim[0], -angle, number_of_points=199)
        return self

    def dc(self, bend_dim: Dim2, interaction_l: float, end_l: float = 0, layer: int = 0,
           inverted: bool = False, end_bend_dim: Optional[Dim3] = None, use_radius: bool = False):
        if end_bend_dim:
            if end_bend_dim[-1] > 0:
                self.segment(end_bend_dim[-1], layer=layer)
            self.sbend(end_bend_dim[:2], layer, inverted, use_radius)
        if end_l > 0:
            self.segment(end_l, layer=layer)
        self.sbend(bend_dim, layer, inverted, use_radius)
        self.segment(interaction_l, layer=layer)
        self.sbend(bend_dim, layer, not inverted, use_radius)
        if end_l > 0:
            self.segment(end_l, layer=layer)
        if end_bend_dim:
            self.sbend(end_bend_dim[:2], layer, not inverted, use_radius)
            if end_bend_dim[-1] > 0:
                self.segment(end_bend_dim[-1], layer=layer)
        return self

    def mzi(self, bend_dim: Dim2, interaction_l: float, arm_l: float, end_l: float = 0, layer: int = 0,
            inverted: bool = False, end_bend_dim: Optional[Dim3] = None, use_radius: bool = False):
        if end_bend_dim:
            if end_bend_dim[-1] > 0:
                self.segment(end_bend_dim[-1], layer=layer)
            self.sbend(end_bend_dim[:2], layer, inverted)
        if end_l > 0:
            self.segment(end_l, layer=layer)
        self.sbend(bend_dim, layer, inverted, use_radius)
        self.segment(interaction_l, layer=layer)
        self.sbend(bend_dim, layer, not inverted, use_radius)
        self.segment(arm_l, layer=layer)
        self.sbend(bend_dim, layer, inverted, use_radius)
        self.segment(interaction_l, layer=layer)
        self.sbend(bend_dim, layer, not inverted, use_radius)
        if end_l > 0:
            self.segment(end_l, layer=layer)
        if end_bend_dim:
            self.sbend(end_bend_dim[:2], layer, not inverted)
            if end_bend_dim[-1] > 0:
                self.segment(end_bend_dim[-1], layer=layer)
        return self

    def trombone(self, height: float, radius: float):
        self.turn(radius, np.pi / 2, tolerance=0.001).segment(height)
        self.turn(radius, -np.pi, tolerance=0.001).segment(height).turn(radius, np.pi / 2, tolerance=0.001)
        return self

    def to(self, port: Dim2):
        return self.sbend((port[0] - self.x, port[1] - self.y))


class Pattern:
    def __init__(self, *polygons: Union[Path, gy.Polygon, gy.FlexPath, Polygon], shift: Dim2 = (0, 0),
                 call_union: bool = True):
        self.shift = shift
        self.config = copy(self.__dict__)
        self.polys = []
        for shape in polygons:
            if not isinstance(shape, Polygon):
                if isinstance(shape, MultiPolygon):
                    polygons = list(shape)
                else:
                    polygons = shape.get_polygons() if isinstance(shape, gy.FlexPath) else shape.polygons
                self.polys += [Polygon(polygon_point_list) for polygon_point_list in polygons]
            else:
                self.polys.append(shape)
        self.call_union = call_union
        self.pattern = self._pattern()
        if shift != (0, 0):
            self.translate(shift[0], shift[1])

    def _pattern(self) -> MultiPolygon:
        if not self.call_union:
            return MultiPolygon(self.polys)
        else:
            pattern = cascaded_union(self.polys)
            return pattern if isinstance(pattern, MultiPolygon) else MultiPolygon([pattern])

    def mask(self, shape: Shape, grid_spacing: GridSpacing):
        x_, y_ = np.mgrid[0:grid_spacing[0] * shape[0]:grid_spacing[0], 0:grid_spacing[1] * shape[1]:grid_spacing[1]]
        return contains(self.pattern, x_, y_)

    @property
    def bounds(self) -> Dim4:
        return self.pattern.bounds

    @property
    def size(self) -> Dim2:
        b = self.bounds  # (minx, miny, maxx, maxy)
        return b[2] - b[0], b[3] - b[1]  # (maxx - minx, maxy - miny)

    @property
    def center(self) -> Dim2:
        b = self.bounds  # (minx, miny, maxx, maxy)
        return (b[2] + b[0]) / 2, (b[3] + b[1]) / 2  # (avgx, avgy)

    def translate(self, dx: float = 0, dy: float = 0) -> "Pattern":
        self.polys = [translate(path, dx, dy) for path in self.polys]
        self.pattern = self._pattern()
        self.shift = (self.shift[0] + dx, self.shift[1] + dy)
        self.config["shift"] = self.shift
        return self

    def center_align(self, c: Union["Pattern", Tuple[float, float]]) -> "Pattern":
        old_x, old_y = self.center
        center = c if isinstance(c, tuple) else c.center
        self.translate(center[0] - old_x, center[1] - old_y)
        return self

    def horz_align(self, c: Union["Pattern", float], left: bool = True, opposite: bool = False) -> "Pattern":
        x = self.bounds[0] if left else self.bounds[2]
        p = c[0] if isinstance(c, tuple) else (
            c.bounds[0] if left and not opposite or opposite and not left else c.bounds[2])
        self.translate(dx=p - x)
        return self

    def vert_align(self, c: Union["Pattern", float], bottom: bool = True, opposite: bool = False) -> "Pattern":
        x = self.bounds[1] if bottom else self.bounds[3]
        p = c[1] if isinstance(c, tuple) else (
            c.bounds[1] if bottom and not opposite or opposite and not bottom else c.bounds[3])
        self.translate(dy=p - x)
        return self

    def flip(self, horiz: bool = False):
        new_polys = []
        for poly in self.polys:
            points = np.asarray(poly.exterior.coords.xy)
            new_points = np.stack((-points[0], points[1])) if horiz else np.stack((points[0], -points[1]))
            new_polys.append(Polygon(new_points.T))
        self.polys = new_polys
        return self

    def boolean_operation(self, other_pattern, operation):
        if operation == 'intersection':
            returned_object = self.pattern.intersection(other_pattern.pattern)
        elif operation == 'difference':
            returned_object = self.pattern.difference(other_pattern.pattern)
        elif operation == 'union':
            returned_object = self.pattern.union(other_pattern.pattern)
        elif operation == 'symmetric_difference':
            returned_object = self.pattern.symmetric_difference(other_pattern.pattern)
        else:
            raise ValueError (" Not a valid boolean operation: Must be 'intersection', 'difference', 'union', or 'symmetric_difference' ")
        return(pattern_recover(returned_object))

    def to_gds(self, cell: gy.Cell):
        """

        Args:
            cell: GDSPY cell to add polygon

        Returns:

        """
        for path in self.polys:
            cell.add(gy.Polygon(np.asarray(path.exterior.coords.xy).T)) if isinstance(path, Polygon) else cell.add(path)

    def to_trimesh(self, extrude_height: float, start_height: float = 0, engine: str = 'scad') -> trimesh.Trimesh:
        meshes = [trimesh.creation.extrude_polygon(poly, height=extrude_height).apply_translation((0, 0, start_height))
                  for poly in self.polys]
        return trimesh.Trimesh().union(meshes, engine=engine)

    def to_stl(self, filename: str, extrude_height: float, engine: str = 'scad'):
        self.to_trimesh(extrude_height, engine=engine).export(filename)

    def plot(self, ax, color):
        ax.add_patch(PolygonPatch(self.pattern, facecolor=color, edgecolor='none'))
        b = self.bounds
        ax.set_xlim((b[0], b[2]))
        ax.set_ylim((b[1], b[3]))
        ax.set_aspect('equal')

    @property
    def input_ports(self) -> np.ndarray:
        return np.asarray([])

    @property
    def output_ports(self) -> np.ndarray:
        return np.asarray([])

    @property
    def contact_ports(self) -> np.ndarray:
        return np.asarray([])

    @property
    def attachment_ports(self) -> np.ndarray:
        return np.asarray([])

    def grow(self, grow_d: float, subtract: bool = False, cap_style: int = CAP_STYLE.square) -> "Pattern":
        rib_pattern = self.pattern.buffer(grow_d, cap_style=cap_style)
        if subtract:
            rib_pattern = rib_pattern - self.pattern
        return Pattern(rib_pattern if isinstance(rib_pattern, MultiPolygon) else rib_pattern)

    def nazca_cell(self, cell_name: str, layer: Union[int, str]) -> nd.Cell:
        with nd.Cell(cell_name) as cell:
            for poly in self.polys:
                nd.Polygon(points=np.asarray(poly.exterior.coords.xy).T, layer=layer).put()
            for idx, port in enumerate(self.input_ports):
                nd.Pin(f'a{idx}').put(*port, 180)
            for idx, port in enumerate(self.output_ports):
                nd.Pin(f'b{idx}').put(*port)
            for idx, port in enumerate(self.contact_ports):
                nd.Pin(f'c{idx}').put(*port)
            for idx, port in enumerate(self.attachment_ports):
                nd.Pin(f't{idx}').put(*port)
            nd.put_stub()
        return cell

    def metal_contact(self, metal_layers: Tuple[str, ...],
                      via_sizes: Tuple[float, ...] = (0.4, 0.4, 3.6)):
        patterns = []
        for i, metal_layer in enumerate(metal_layers):
            if i % 2 == 0:
                pattern = Pattern(Path(via_sizes[i // 2]).segment(via_sizes[i // 2])).center_align(self)
            else:
                pattern = copy(self)
            patterns.append((pattern.center_align(self), metal_layer))



        return [(pattern, metal_layer) for pattern, metal_layer in patterns]

    def dope(self, dope_stack: Tuple[str, ...], level: int = 1, dope_grow: float = 0.25):
        return copy(self).grow(dope_grow), dope_stack[level]

    def clearout_box(self, clearout_layer: str, clearout_etch_stop_layer: str,
                     dim: Tuple[float, float], clearout_etch_stop_grow: float = 0.5,
                     center: Tuple[float, float] = None):
        center = center if center is not None else self.center
        box = Pattern(Path(dim[1]).segment(dim[0]).translate(dx=0, dy=dim[1] / 2)).center_align(center)
        box_grow = box.grow(clearout_etch_stop_grow)
        return [(box, clearout_layer), (box_grow, clearout_etch_stop_layer)]


class GroupedPattern(Pattern):
    def __init__(self, *patterns: Pattern, shift: Dim2 = (0, 0), call_union: bool = True):
        self.patterns = patterns
        super(GroupedPattern, self).__init__(*sum([list(pattern.polys) for pattern in patterns], []),
                                             shift=shift, call_union=call_union)

    @property
    def input_ports(self) -> np.ndarray:
        input_ports = [c.input_ports for c in self.patterns if c.input_ports.size > 0]
        return np.vstack(input_ports) if len(input_ports) > 0 else np.asarray([])

    @property
    def output_ports(self) -> np.ndarray:
        output_ports = [c.output_ports for c in self.patterns if c.output_ports.size > 0]
        return np.vstack(output_ports) if len(output_ports) > 0 else np.asarray([])

# TODO(nate): find a better place for these functions

def pattern_recover(returned_object):
        if isinstance(returned_object, Polygon):
            collection = MultiPolygon(polygons=[returned_object])
        else:
            collection = MultiPolygon([g for g in returned_object.geoms if isinstance(g,Polygon)])
        return(Pattern(collection))

def get_cubic_taper(change_w):
    return (0, 0, 3 * change_w, -2 * change_w)

def is_adiabatic(taper_params, init_width: float = 0.48, wavelength: float = 1.55, neff: float = 2.75,
                 num_points: int = 100, taper_l: float = 5):
    taper_params = np.asarray(taper_params)
    u = np.linspace(0, 1 + 1 / num_points, num_points)[:, np.newaxis]
    width = init_width + np.sum(taper_params * u ** np.arange(taper_params.size, dtype=float), axis=1)
    theta = np.arctan(np.diff(width) / taper_l * num_points)
    max_pt = np.argmax(theta)
    return theta[max_pt], wavelength / (2 * width[max_pt] * neff)

def get_linear_adiabatic(min_width: float = 0.48, max_width: float = 1,  wavelength: float = 1.55, neff_max: float = 2.75,
                 num_points: int = 100, min_to_max: bool = True, aggressive: bool = False):
    taper_params =(0, max_width - min_width) if min_to_max else (0, min_width - max_width)
    taper_l =  1.1*abs(max_width - min_width) / np.arctan(wavelength / (2 * max_width * neff_max)) if aggressive else 2* abs(max_width - min_width) / np.arctan(wavelength / (2 * max_width * neff_max))
    return taper_l, taper_params
