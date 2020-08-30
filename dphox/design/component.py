from collections import defaultdict

import gdspy as gy
import nazca as nd
from copy import deepcopy as copy
from shapely.vectorized import contains
from shapely.geometry import Polygon, MultiPolygon, CAP_STYLE
from shapely.ops import cascaded_union, polygonize
from shapely.affinity import translate
from descartes import PolygonPatch
import trimesh
from trimesh import creation, visual

try:
    import plotly.graph_objects as go
except ImportError:
    pass

from ..typing import *


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
        return Pattern(rib_pattern.geoms if isinstance(rib_pattern, MultiPolygon) else rib_pattern)

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


class Multilayer:
    def __init__(self,
                 pattern_to_layer: Dict[Union[Pattern, Path, gy.Polygon, gy.FlexPath, Polygon], Union[int, str]]):
        self.pattern_to_layer = {comp: layer if isinstance(comp, Pattern) else Pattern(comp)
                                 for comp, layer in pattern_to_layer.items()}
        self.layer_to_pattern = self._layer_to_pattern()

    @property
    def input_ports(self) -> np.ndarray:
        all_input_ports = [c.input_ports for c in self.pattern_to_layer.keys() if c.input_ports.size > 0]
        return np.vstack(all_input_ports) if len(all_input_ports) > 0 else np.asarray([])

    @property
    def output_ports(self) -> np.ndarray:
        all_output_ports = [c.output_ports for c in self.pattern_to_layer.keys() if c.output_ports.size > 0]
        return np.vstack(all_output_ports) if len(all_output_ports) > 0 else np.asarray([])

    @property
    def contact_ports(self) -> np.ndarray:
        contact_ports = [c.contact_ports for c in self.pattern_to_layer.keys() if c.contact_ports.size > 0]
        return np.vstack(contact_ports) if len(contact_ports) > 0 else np.asarray([])

    @property
    def attachment_ports(self) -> np.ndarray:
        attachment_ports = [c.attachment_ports for c in self.pattern_to_layer.keys() if c.attachment_ports.size > 0]
        return np.vstack(attachment_ports) if len(attachment_ports) > 0 else np.asarray([])

    @property
    def bounds(self) -> Dim4:
        return self.gdspy_cell().get_bounding_box()

    def gdspy_cell(self, cell_name: str = 'dummy') -> gy.Cell:
        cell = gy.Cell(cell_name, exclude_from_current=(cell_name == 'dummy'))
        for pattern, layer in self.pattern_to_layer.items():
            for poly in pattern.polys:
                cell.add(gy.Polygon(np.asarray(poly.exterior.coords.xy).T, layer=layer))
        return cell

    def nazca_cell(self, cell_name: str) -> nd.Cell:
        with nd.Cell(cell_name) as cell:
            for pattern, layer in self.pattern_to_layer.items():
                for poly in pattern.polys:
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

    def _layer_to_pattern(self) -> Dict[Union[int, str], MultiPolygon]:
        layer_to_polys = defaultdict(list)
        for component, layer in self.pattern_to_layer.items():
            layer_to_polys[layer].extend(component.polys)
        pattern_dict = {layer: MultiPolygon(polys) for layer, polys in layer_to_polys.items()}
        # pattern_dict = {layer: (pattern if isinstance(pattern, MultiPolygon) else MultiPolygon([pattern]))
        #                 for layer, pattern in pattern_dict.items()}
        return pattern_dict

    def plot(self, ax, layer_to_color: Dict[Union[int, str], Union[Dim3, str]], alpha: float = 0.5):
        for layer, pattern in self.layer_to_pattern:
            ax.add_patch(PolygonPatch(pattern, facecolor=layer_to_color[layer], edgecolor='none', alpha=alpha))
        b = self.bounds
        ax.set_xlim((b[0], b[2]))
        ax.set_ylim((b[1], b[3]))
        ax.set_aspect('equal')

    def to_trimesh(self, layer_to_zrange: Dict[str, Tuple[float, float]],
                   layer_to_color: Optional[Dict[str, str]] = None, engine: str = 'scad'):
        meshes = []
        for layer, zrange in layer_to_zrange.items():
            zmin, zmax = zrange
            layer_meshes = [
                trimesh.creation.extrude_polygon(poly, height=zmax - zmin).apply_translation((0, 0, zmin))
                for poly in self.layer_to_pattern[layer]]
            mesh = trimesh.Trimesh().union(layer_meshes, engine=engine)
            mesh.visual.vertex_colors = visual.random_color() if layer_to_color is None else layer_to_color[layer]
            meshes.append(mesh)
        return trimesh.Scene(meshes)


class Box(Pattern):
    def __init__(self, box_dim: Dim2, shift: Dim2 = (0, 0)):
        self.box_dim = box_dim

        super(Box, self).__init__(Path(box_dim[1]).segment(box_dim[0]).translate(dx=0, dy=box_dim[1] / 2), shift=shift)


class GratingPad(Pattern):
    def __init__(self, pad_dim: Dim2, taper_l: float, final_width: float, out: bool = False,
                 end_l: Optional[float] = None, bend_dim: Optional[Dim2] = None, shift: Dim2 = (0, 0),
                 layer: int = 0):
        self.pad_dim = pad_dim
        self.taper_l = taper_l
        self.final_width = final_width
        self.out = out
        self.bend_dim = bend_dim
        self.end_l = taper_l if end_l is None else end_l

        if out:
            path = Path(final_width)
            if end_l > 0:
                path.segment(end_l)
            if bend_dim:
                path.sbend(bend_dim)
            super(GratingPad, self).__init__(
                path.segment(taper_l, final_width=pad_dim[1]).segment(pad_dim[0]), shift=shift)
        else:
            path = Path(pad_dim[1]).segment(pad_dim[0]).segment(taper_l, final_width=final_width)
            if bend_dim:
                path.sbend(bend_dim, layer=layer)
            if end_l > 0:
                path.segment(end_l, layer=layer)
            super(GratingPad, self).__init__(path, shift=shift)

    def to(self, port: Dim2):
        if self.out:
            return self.translate(port[0], port[1])
        else:
            bend_y = self.bend_dim[1] if self.bend_dim else 0
            return self.translate(port[0] - self.size[0], port[1] - bend_y)

    @property
    def copy(self) -> "GratingPad":
        return copy(self)


class GroupedPattern(Pattern):
    def __init__(self, *patterns: Pattern, shift: Dim2 = (0, 0), call_union: bool = True):
        self.patterns = patterns
        super(GroupedPattern, self).__init__(*sum([list(pattern.polys) for pattern in patterns], []),
                                             shift=shift, call_union=call_union)

    @classmethod
    def component_with_gratings(cls, component: Pattern, grating: GratingPad) -> "GroupedPattern":
        components = [component]
        out_config = copy(grating.config)
        out_config['out'] = True
        out_grating = GratingPad(**out_config)
        components.extend([grating.copy.to(port) for port in component.input_ports])
        components.extend([out_grating.copy.to(port) for port in component.output_ports])
        return cls(*components)

    @property
    def input_ports(self) -> np.ndarray:
        input_ports = [c.input_ports for c in self.patterns if c.input_ports.size > 0]
        return np.vstack(input_ports) if len(input_ports) > 0 else np.asarray([])

    @property
    def output_ports(self) -> np.ndarray:
        output_ports = [c.output_ports for c in self.patterns if c.output_ports.size > 0]
        return np.vstack(output_ports) if len(output_ports) > 0 else np.asarray([])


class DC(Pattern):
    def __init__(self, bend_dim: Dim2, waveguide_w: float, gap_w: float, interaction_l: float,
                 coupler_boundary_taper_ls: Tuple[float, ...] = (0,),
                 coupler_boundary_taper: Optional[Tuple[Tuple[float, ...]]] = None, end_l: float = 0,
                 end_bend_dim: Optional[Dim3] = None, use_radius: bool = False, shift: Dim2 = (0, 0)):
        """Directional coupler

        Args:
            bend_dim: if use_radius is True (bend_radius, bend_height), else (bend_width, bend_height)
            waveguide_w: waveguide width
            gap_w: gap between the waveguides
            interaction_l: interaction length
            coupler_boundary_taper_ls: coupler boundary tapers length
            coupler_boundary_taper: coupler boundary taper params
            end_l: end length before and after the bends
            end_bend_dim: If specified, places an additional end bend (see DC)
            use_radius: use radius to define bends
            shift:
        """
        self.end_l = end_l
        self.bend_dim = bend_dim
        self.waveguide_w = waveguide_w
        self.interaction_l = interaction_l
        self.gap_w = gap_w
        self.end_bend_dim = end_bend_dim
        self.use_radius = use_radius
        self.coupler_boundary_taper_ls = coupler_boundary_taper_ls
        self.coupler_boundary_taper = coupler_boundary_taper

        interport_distance = waveguide_w + 2 * bend_dim[1] + gap_w
        if end_bend_dim:
            interport_distance += 2 * end_bend_dim[1]

        lower_path = Path(waveguide_w).dc(bend_dim, interaction_l, end_l, end_bend_dim=end_bend_dim,
                                          use_radius=use_radius)
        upper_path = Path(waveguide_w).dc(bend_dim, interaction_l, end_l, end_bend_dim=end_bend_dim,
                                          inverted=True, use_radius=use_radius)
        upper_path.translate(dx=0, dy=interport_distance)

        if coupler_boundary_taper is not None and np.sum(coupler_boundary_taper_ls) > 0:
            current_dc = Pattern(upper_path, lower_path)
            outer_boundary = Waveguide(waveguide_w=2 * waveguide_w + gap_w, length=interaction_l,
                                       taper_params=coupler_boundary_taper,
                                       taper_ls=coupler_boundary_taper_ls).center_align(current_dc)
            dc_interaction = Box((interaction_l, 2 * waveguide_w + gap_w)).center_align(current_dc.center)
            paths_intersection = outer_boundary.pattern.intersection(current_dc.pattern)
            paths_diff = outer_boundary.pattern.union(current_dc.pattern).difference(dc_interaction.pattern)
            paths_full = paths_diff.union(paths_intersection)
            paths = list(polygonize(paths_full))
        else:
            paths = lower_path, upper_path
        super(DC, self).__init__(*paths, shift=shift)
        self.lower_path, self.upper_path = Pattern(lower_path), Pattern(upper_path)

    @property
    def input_ports(self) -> np.ndarray:
        interport_distance = self.waveguide_w + 2 * self.bend_dim[1] + self.gap_w
        if self.end_bend_dim:
            interport_distance += 2 * self.end_bend_dim[1]
        return np.asarray(((0, 0), (0, interport_distance))) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.size[0], 0))

    @property
    def interaction_points(self) -> np.ndarray:
        bl = np.asarray(self.center) - np.asarray((self.interaction_l, self.waveguide_w + self.gap_w)) / 2
        tl = bl + np.asarray((0, self.waveguide_w + self.gap_w))
        br = bl + np.asarray((self.interaction_l, 0))
        tr = tl + np.asarray((self.interaction_l, 0))
        return np.vstack((bl, tl, br, tr))


class MZI(Pattern):
    def __init__(self, bend_dim: Dim2, waveguide_w: float, arm_l: float, gap_w: float,
                 interaction_l: float, end_l: float = 0, end_bend_dim: Optional[Dim3] = None, use_radius: bool = False,
                 shift: Dim2 = (0, 0)):
        self.end_l = end_l
        self.arm_l = arm_l
        self.bend_dim = bend_dim
        self.waveguide_w = waveguide_w
        self.interaction_l = interaction_l
        self.gap_w = gap_w
        self.end_bend_dim = end_bend_dim
        self.use_radius = use_radius

        lower_path = Path(waveguide_w).mzi(bend_dim, interaction_l, arm_l, end_l,
                                           end_bend_dim=end_bend_dim, use_radius=use_radius)
        upper_path = Path(waveguide_w).mzi(bend_dim, interaction_l, arm_l, end_l,
                                           end_bend_dim=end_bend_dim, inverted=True, use_radius=use_radius)
        upper_path.translate(dx=0, dy=waveguide_w + 2 * bend_dim[1] + gap_w)

        super(MZI, self).__init__(lower_path, upper_path, shift=shift)
        self.lower_path, self.upper_path = Pattern(lower_path), Pattern(upper_path)

    @property
    def input_ports(self) -> np.ndarray:
        interport_distance = self.waveguide_w + 2 * self.bend_dim[1] + self.gap_w
        if self.end_bend_dim:
            interport_distance += 2 * self.end_bend_dim[1]
        return np.asarray(((0, 0), (0, interport_distance))) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.size[0], 0))

    @property
    def interaction_points(self) -> np.ndarray:
        input_ports = self.input_ports
        bl = input_ports[0] + np.asarray(self.bend_dim) + np.asarray(self.end_bend_dim) + np.asarray((self.end_l, 0))
        tl = bl + np.asarray((self.waveguide_w + self.gap_w, 0))
        br = bl + np.asarray((self.interaction_l, 0))
        tr = tl + np.asarray((self.interaction_l, 0))
        left_dc_pts = np.vstack((bl, tl, br, tr))
        right_dc_pts = left_dc_pts + np.asarray((self.arm_l + self.bend_dim[0], 0))
        return np.vstack((left_dc_pts, right_dc_pts))


class MMI(Pattern):
    def __init__(self, box_dim: Dim2, waveguide_w: float, interport_distance: float,
                 taper_dim: Dim2, end_l: float = 0, bend_dim: Optional[Tuple[float, float]] = None,
                 use_radius: bool = False, shift: Dim2 = (0, 0)):
        self.end_l = end_l
        self.waveguide_w = waveguide_w
        self.box_dim = box_dim
        self.interport_distance = interport_distance
        self.taper_dim = taper_dim
        self.bend_dim = bend_dim
        self.use_radius = use_radius

        if self.bend_dim:
            center = (end_l + bend_dim[0] + taper_dim[0] + box_dim[0] / 2, interport_distance / 2 + bend_dim[1])
            p_00 = Path(waveguide_w).segment(end_l) if end_l > 0 else Path(waveguide_w)
            p_00.sbend(bend_dim, use_radius=use_radius).segment(taper_dim[0], final_width=taper_dim[1])
            p_01 = Path(waveguide_w, (0, interport_distance + 2 * bend_dim[1]))
            p_01 = p_01.segment(end_l) if end_l > 0 else p_01
            p_01.sbend(bend_dim, inverted=True, use_radius=use_radius).segment(
                taper_dim[0], final_width=taper_dim[1])
        else:
            center = (end_l + taper_dim[0] + box_dim[0] / 2, interport_distance / 2)
            p_00 = Path(waveguide_w).segment(end_l) if end_l > 0 else Path(waveguide_w)
            p_00.segment(taper_dim[0], final_width=taper_dim[1])
            p_01 = copy(p_00).translate(dx=0, dy=interport_distance)
        mmi_start = (center[0] - box_dim[0] / 2, center[1])
        mmi = Path(box_dim[1], mmi_start).segment(box_dim[0])
        p_10 = copy(p_01).rotate(np.pi, center)
        p_11 = copy(p_00).rotate(np.pi, center)

        super(MMI, self).__init__(mmi, p_00, p_01, p_10, p_11, shift=shift)

    @property
    def input_ports(self) -> np.ndarray:
        bend_y = 2 * self.bend_dim[1] if self.bend_dim else 0
        return np.asarray(((0, 0), (0, self.interport_distance + bend_y))) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.size[0], 0))


class Waveguide(Pattern):
    def __init__(self, waveguide_w: float, length: float, taper_ls: Tuple[float, ...] = None,
                 taper_params: Tuple[Tuple[float, ...]] = None,
                 slot_dim: Optional[Dim2] = None, slot_taper_ls: Tuple[float, ...] = 0,
                 slot_taper_params: Tuple[Tuple[float, ...]] = None,
                 num_taper_evaluations: int = 100, shift: Dim2 = (0, 0),
                 symmetric: bool = True):

        """Waveguide class
        Args:
            waveguide_w: waveguide width at the input of the waveguide path
            length: total length of the waveguide
            taper_ls: a tuple of lengths for tapers starting from the left

            symmetric: a temporary toggling variable to turn off the symmetric nature of the waveguide class.

            .
            .
            .
            """
        self.length = length
        self.waveguide_w = waveguide_w
        self.taper_ls = taper_ls
        self.taper_params = taper_params
        self.slot_dim = slot_dim
        self.slot_taper_ls = slot_taper_ls
        self.slot_taper_params = slot_taper_params

        self.pads = []

        p = Path(waveguide_w)
        if taper_params is not None:
            for taper_l, taper_param in zip(taper_ls, taper_params):
                if taper_l > 0:
                    p.polynomial_taper(taper_l, taper_param, num_taper_evaluations)
        if symmetric:
            if not length >= 2 * np.sum(taper_ls):
                raise ValueError(
                    f'Require interaction_l >= 2 * np.sum(taper_ls) but got {length} < {2 * np.sum(taper_ls)}')
            if taper_params is not None:
                p.segment(length - 2 * np.sum(taper_ls))
                for taper_l, taper_param in zip(reversed(taper_ls), reversed(taper_params)):
                    if taper_l > 0:
                        p.polynomial_taper(taper_l, taper_param, num_taper_evaluations, inverted=True)
            else:
                p.segment(length)
        else:
            if not length >= np.sum(taper_ls):
                raise ValueError(f'Require interaction_l >= np.sum(taper_ls) but got {length} < {np.sum(taper_ls)}')
            p.segment(length - np.sum(taper_ls))

        if slot_taper_params:
            center_x = length / 2
            slot = self.__init__(slot_dim[1], slot_dim[0], slot_taper_ls, slot_taper_params).center_align((center_x, 0))
            pattern = Pattern(p).pattern - slot.pattern
            if isinstance(pattern, MultiPolygon):
                slot_waveguide = [Pattern(poly) for poly in pattern]
                super(Waveguide, self).__init__(*slot_waveguide, shift=shift)
            else:
                super(Waveguide, self).__init__(pattern, shift=shift)
        else:
            super(Waveguide, self).__init__(p, shift=shift)

    @property
    def input_ports(self) -> np.ndarray:
        return np.asarray(((0, 0),)) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.size[0], 0))

    #######################################################################################
    ###Adding multilayer directly for testing in a real stack Will be removed if not useful
    def multilayer(self, contact_box_dim: Dim2, clearout_box_dim: Dim2,
                   waveguide_layer: str = 'seam', metal_stack_layers: Tuple[str, ...] = ('m1am', 'm2am'),
                   via_stack_layers: Tuple[str, ...] = ('cbam', 'v1am'),
                   clearout_layer: str = 'tram', clearout_etch_stop_layer: str = 'esam',
                   doping_stack_layer: Optional[str] = None,
                   clearout_etch_stop_grow: float = 0, via_shrink: float = 1, doping_grow: float = 0.25) -> Multilayer:
        return multilayer(self, self.pads, (self.center,), waveguide_layer, metal_stack_layers,
                          via_stack_layers, clearout_layer, clearout_etch_stop_layer, contact_box_dim,
                          clearout_box_dim, doping_stack_layer, clearout_etch_stop_grow, via_shrink, doping_grow)
    #########################################################################################


class LateralNemsPS(GroupedPattern):
    def __init__(self, waveguide_w: float, nanofin_w: float, phaseshift_l: float,
                 gap_w: float, taper_ls: Tuple[float, ...], num_taper_evaluations: int = 100,
                 pad_dim: Optional[Dim3] = None, anchor: Optional[Dim5] = None,
                 gap_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 wg_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 boundary_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 shift: Tuple[float, float] = (0, 0)):
        """NEMS single-mode phase shifter
        Args:
            waveguide_w: waveguide width
            nanofin_w: nanofin width (initial, before tapering)
            phaseshift_l: phase shift length
            gap_w: gap width (initial, before tapering)
            taper_ls:  array of taper lengths
            num_taper_evaluations: number of taper evaluations (see gdspy)
            pad_dim: silicon handle xy size followed by distance between pad and fin to actuate
            anchor: a tuple of parameters corresponding to:
                anchor[0] = tether/electrode x,
                anchor[1] = tether/electrode y,
                anchor[2] = width of a fin for a (possible) tethered spring design,
                anchor[3] = turn radius for adiabatic connection to thin waveguide section,
                anchor[4] = thickness of straight section.
            gap_taper: gap taper polynomial params (recommend same as wg_taper)
            wg_taper: wg taper polynomial params (recommend same as gap_taper)
            shift: translate this component in xy
        """
        self.waveguide_w = waveguide_w
        self.nanofin_w = nanofin_w
        self.phaseshift_l = phaseshift_l
        self.gap_w = gap_w
        self.taper_ls = taper_ls
        self.num_taper_evaluations = num_taper_evaluations
        self.pad_dim = pad_dim
        self.gap_taper = gap_taper
        self.wg_taper = wg_taper
        self.boundary_taper = boundary_taper
        self.anchor = anchor

        if not phaseshift_l >= 2 * np.sum(taper_ls):
            raise ValueError(
                f'Require interaction_l >= 2 * np.sum(taper_ls) but got {phaseshift_l} < {2 * np.sum(taper_ls)}')

        boundary_taper = wg_taper if boundary_taper is None else boundary_taper

        box_w = nanofin_w * 2 + gap_w * 2 + waveguide_w
        wg = Waveguide(waveguide_w, taper_ls=taper_ls, taper_params=wg_taper, length=phaseshift_l,
                       num_taper_evaluations=num_taper_evaluations)
        boundary = Waveguide(box_w, taper_params=boundary_taper, taper_ls=taper_ls, length=phaseshift_l,
                             num_taper_evaluations=num_taper_evaluations).pattern
        gap_path = Waveguide(waveguide_w + gap_w * 2, taper_params=gap_taper,
                             taper_ls=taper_ls, length=phaseshift_l,
                             num_taper_evaluations=num_taper_evaluations).pattern
        nanofins = [Pattern(poly) for poly in (boundary - gap_path)]
        pads, anchors = [], []
        if pad_dim is not None:
            pad = Box(pad_dim[:2]).center_align(wg)
            pad_y = nanofin_w + pad_dim[2] + pad_dim[1] / 2
            pads += [copy(pad).translate(dx=0, dy=-pad_y), copy(pad).translate(dx=0, dy=pad_y)]
        # if anchor is not None:
        #     tether = NemsAnchor(fixed_fin_dim=(phaseshift_l, nanofin_w), bending_fin_dim=(anchor[2], nanofin_w),
        #                         tether_dim=(anchor[0], anchor[1]), loop_connector=(anchor[3], anchor[4]))
        #     top_tether = copy(tether).center_align(nanofins[0]).vert_align(nanofins[0], opposite=True).translate(
        #         dy=-nanofin_w)
        #     bottom_tether = copy(tether).flip().center_align(nanofins[1]).vert_align(nanofins[1], bottom=False,
        #                                                                              opposite=True).translate(
        #         dy=nanofin_w)
        #     anchors += [top_tether, bottom_tether]
        super(LateralNemsPS, self).__init__(*([wg] + nanofins + pads + anchors), shift=shift, call_union=False)
        self.waveguide, self.anchors, self.pads, self.nanofins = wg, anchors, pads, nanofins

    @property
    def input_ports(self) -> np.ndarray:
        return np.asarray((0, 0)) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.phaseshift_l, 0))

    @property
    def attachment_ports(self) -> np.ndarray:
        dy = np.asarray((0, self.nanofin_w / 2 + self.waveguide_w / 2 + self.gap_w))
        center = np.asarray(self.center)
        return np.asarray((center + dy, center - dy))

    # waveguide_layer = 'seam', metal_stack_layers: Tuple[str, ...] = ('m1am', 'm2am'),
    # doping_stack_layer = 'ppam', via_stack_layers = ['cbam', 'v1am'],
    # clearout_layer = 'tram', clearout_etch_stop_layer = 'esam',

    def multilayer(self, contact_box_dim: Dim2, clearout_box_dim: Dim2,
                   waveguide_layer: str = 'seam', metal_stack_layers: Tuple[str, ...] = ('m1am', 'm2am'),
                   via_stack_layers: Tuple[str, ...] = ('cbam', 'v1am'),
                   clearout_layer: str = 'tram', clearout_etch_stop_layer: str = 'esam',
                   doping_stack_layer: Optional[str] = None,
                   clearout_etch_stop_grow: float = 0, via_shrink: float = 1, doping_grow: float = 0.25) -> Multilayer:
        return multilayer(self, self.pads, (self.center,), waveguide_layer, metal_stack_layers,
                          via_stack_layers, clearout_layer, clearout_etch_stop_layer, contact_box_dim,
                          clearout_box_dim, doping_stack_layer, clearout_etch_stop_grow, via_shrink, doping_grow)


class LateralNemsTDC(GroupedPattern):
    def __init__(self, waveguide_w: float, nanofin_w: float, dc_gap_w: float, beam_gap_w: float, bend_dim: Dim2,
                 interaction_l: float, dc_taper_ls: Tuple[float, ...] = None,
                 dc_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 beam_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 end_l: float = 0, end_bend_dim: Optional[Dim3] = None,
                 anchor: Optional[Dim4] = None, pad_dim: Optional[Dim3] = None,
                 middle_fin_dim: Optional[Dim2] = None, middle_fin_pad_dim: Optional[Dim2] = None,
                 use_radius: bool = True, shift: Dim2 = (0, 0)):
        """NEMS tunable directional coupler

        Args:
            waveguide_w: waveguide width
            nanofin_w: nanofin width
            dc_gap_w: directional coupler gap width
            beam_gap_w: gap between the nanofin and the TDC waveguides
            bend_dim: see DC
            interaction_l: interaction length
            end_l: end length before and after the first and last bends
            end_bend_dim: If specified, places an additional end bend (see DC)
            anchor: a tuple of parameters corresponding to:
                anchor[0] = tether/electrode x,
                anchor[1] = tether/electrode y,
                anchor[2] = width of a fin for a (possible) tethered spring design,
                anchor[3] = turn radius for adiabatic connection to thin waveguide section,
                anchor[4] = thickness of straight section.
            pad_dim: If specified, silicon anchor/handle xy size followed by the pad gap
            middle_fin_dim: If specified, place a middle fin in the center of the coupling gap
            middle_fin_pad_dim: If specified, place an anchor pad on the left and right of the middle fin
                (ensure sufficiently far from the bends!).
            use_radius: use radius (see DC)
            shift: translate this component in xy
        """
        self.waveguide_w = waveguide_w
        self.nanofin_w = nanofin_w
        self.interaction_l = interaction_l
        self.end_l = end_l
        self.dc_gap_w = dc_gap_w
        self.beam_gap_w = beam_gap_w
        self.pad_dim = pad_dim
        self.middle_fin_dim = middle_fin_dim
        self.middle_fin_pad_dim = middle_fin_pad_dim
        self.anchor = anchor
        self.use_radius = use_radius

        dc = DC(bend_dim=bend_dim, waveguide_w=waveguide_w, gap_w=dc_gap_w,
                coupler_boundary_taper_ls=dc_taper_ls, coupler_boundary_taper=dc_taper,
                interaction_l=interaction_l, end_bend_dim=end_bend_dim, end_l=end_l, use_radius=use_radius)
        connectors, pads, tethers = [], [], []

        nanofin_y = nanofin_w / 2 + dc_gap_w / 2 + waveguide_w + beam_gap_w
        nanofin = Box((interaction_l, nanofin_w)).center_align(dc)

        if not interaction_l >= 2 * np.sum(dc_taper_ls):
            raise ValueError(
                f'Require interaction_l > 2 * np.sum(dc_taper_ls) but got {interaction_l} < {2 * np.sum(dc_taper_ls)}')

        if beam_taper is None:
            nanofins = [copy(nanofin).translate(dx=0, dy=-nanofin_y), copy(nanofin).translate(dx=0, dy=nanofin_y)]
            if middle_fin_dim is not None:
                nanofins.append(Box(middle_fin_dim).center_align(dc))
        else:
            box_w = (nanofin_w + beam_gap_w + waveguide_w) * 2 + dc_gap_w
            gap_taper_wg_w = (beam_gap_w + waveguide_w) * 2 + dc_gap_w
            # nanofin_box = Box((interaction_l, box_w)).center_align(dc).pattern
            # gap_taper_wg = Waveguide(gap_taper_wg_w, interaction_l, dc_taper_ls, beam_taper).center_align(dc).pattern
            # nanofins = [Pattern(poly) for poly in (nanofin_box - gap_taper_wg)]

            ######### NATE: trying to taper fins of TDC ###################
            boundary = Waveguide(box_w, taper_params=beam_taper, taper_ls=dc_taper_ls,
                                 length=interaction_l).center_align(dc).pattern
            gap_path = Waveguide(gap_taper_wg_w, taper_params=beam_taper, taper_ls=dc_taper_ls,
                                 length=interaction_l).center_align(dc).pattern
            nanofins = [Pattern(poly) for poly in (boundary - gap_path)]
            ######### NATE: trying to taper center of TDC ###################

        # if anchor is not None:
        #     tether = NemsAnchor(fixed_fin_dim=(interaction_l, nanofin_w),
        #                         bending_fin_dim=(anchor[2], nanofin_w),
        #                         tether_dim=(anchor[0], anchor[1]),
        #                         loop_connector=(anchor[3], anchor[4]))
        #     top_tether = copy(tether).center_align(nanofins[0]).vert_align(nanofins[0], opposite=True).translate(
        #         dy=-nanofin_w)
        #     bottom_tether = copy(tether).flip().center_align(nanofins[1]).vert_align(nanofins[1], bottom=False,
        #                                                                              opposite=True).translate(
        #         dy=nanofin_w)
        #     tethers += [top_tether, bottom_tether]

        if pad_dim is not None:
            pad = Box(pad_dim[:2]).center_align(dc)
            pad_y = nanofin_w / 2 + pad_dim[2] + pad_dim[1] / 2 + nanofin_y
            pads += [copy(pad).translate(dx=0, dy=-pad_y), copy(pad).translate(dx=0, dy=pad_y)]

        if middle_fin_pad_dim is not None:
            pad = Box(middle_fin_pad_dim).center_align(dc)
            pad_x = middle_fin_pad_dim[0] / 2 + middle_fin_dim[0] / 2
            pads += [copy(pad).translate(dx=pad_x), copy(pad).translate(dx=pad_x)]

        super(LateralNemsTDC, self).__init__(*([dc] + nanofins + connectors + pads), shift=shift, call_union=False)
        self.dc, self.connectors, self.pads, self.nanofins = dc, connectors, pads, nanofins

    @property
    def input_ports(self) -> np.ndarray:
        return self.dc.input_ports + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.dc.output_ports + self.shift

    @property
    def attachment_ports(self) -> np.ndarray:
        return np.asarray(self.center)

    def multilayer(self, contact_box_dim: Dim2, clearout_box_dim: Dim2,
                   waveguide_layer: str = 'seam', metal_stack_layers: Tuple[str, ...] = ('m1am', 'm2am'),
                   via_stack_layers: Tuple[str, ...] = ('cbam', 'v1am'),
                   clearout_layer: str = 'tram', clearout_etch_stop_layer: str = 'esam',
                   doping_stack_layer: Optional[str] = None,
                   clearout_etch_stop_grow: float = 0, via_shrink: float = 1, doping_grow: float = 0.25) -> Multilayer:
        return multilayer(self, self.pads, (self.center,), waveguide_layer, metal_stack_layers,
                          via_stack_layers, clearout_layer, clearout_etch_stop_layer, contact_box_dim,
                          clearout_box_dim, doping_stack_layer, clearout_etch_stop_grow, via_shrink, doping_grow)


class Interposer(Pattern):
    def __init__(self, waveguide_w: float, n: int, period: float, radius: float,
                 trombone_radius: Optional[float] = None,
                 final_period: Optional[float] = None, self_coupling_extension_dim: Optional[Dim2] = None,
                 horiz_dist: float = 0, num_trombones: int = 1, shift: Dim2 = (0, 0)):
        """

        Args:
            waveguide_w: waveguide width
            n: number of I/O for interposer
            period: initial period entering the interposer
            radius: radius of bends for the interposer
            trombone_radius: trombone bend radius
            final_period: final period for the interposer
            self_coupling_extension_dim: self coupling for alignment
            horiz_dist: additional horizontal distance (usually to make room for wirebonds)
            num_trombones: number of trombones
            shift: translate this component in xy
        """
        trombone_radius = radius if trombone_radius is None else trombone_radius
        final_period = period if final_period is None else final_period
        period_diff = final_period - period
        paths = []
        init_pos = np.zeros((n, 2))
        final_pos = np.zeros_like(init_pos)
        for idx in range(n):
            radius = period_diff / 2 if not radius else radius
            angle_r = np.sign(period_diff) * np.arccos(1 - np.abs(period_diff) / 4 / radius)
            angled_length = np.abs(period_diff / np.sin(angle_r))
            x_length = np.abs(period_diff / np.tan(angle_r))
            angle = angle_r
            path = Path(waveguide_w).segment(length=0).translate(dx=0, dy=period * idx)
            mid = int(np.ceil(n / 2))
            max_length_diff = (angled_length - x_length) * (mid - 1)
            num_trombones = int(
                np.ceil(max_length_diff / 2 / (final_period - 3 * radius))) if not num_trombones else num_trombones
            length_diff = (angled_length - x_length) * idx if idx < mid else (angled_length - x_length) * (n - 1 - idx)
            path.segment(horiz_dist)
            if idx < mid:
                path.turn(radius, -angle)
                path.segment(angled_length * (mid - idx - 1))
                path.turn(radius, angle)
                path.segment(x_length * (idx + 1))
            else:
                path.turn(radius, angle)
                path.segment(angled_length * (mid - n + idx))
                path.turn(radius, -angle)
                path.segment(x_length * (n - idx))
            for _ in range(num_trombones):
                path.trombone(length_diff / 2 / num_trombones, radius=trombone_radius)
            paths.append(path)
            init_pos[idx] = np.asarray((0, period * idx))
            final_pos[idx] = np.asarray((path.x, path.y))

        if self_coupling_extension_dim is not None:
            dx, dy = final_pos[0, 0], final_pos[0, 1]
            radius, grating_length = self_coupling_extension_dim
            self_coupling_path = Path(width=waveguide_w).rotate(-np.pi).translate(dx=dx, dy=dy - final_period)
            self_coupling_path.turn(radius, -np.pi, tolerance=0.001)
            self_coupling_path.segment(length=grating_length + 5)
            self_coupling_path.turn(radius=radius, angle=np.pi / 2, tolerance=0.001)
            self_coupling_path.segment(length=final_period * (n + 1) - 6 * radius)
            self_coupling_path.turn(radius=radius, angle=np.pi / 2, tolerance=0.001)
            self_coupling_path.segment(length=grating_length + 5)
            self_coupling_path.turn(radius=radius, angle=-np.pi, tolerance=0.001)
            paths.append(self_coupling_path)

        super(Interposer, self).__init__(*paths, call_union=False, shift=shift)
        self.self_coupling_path = None if self_coupling_extension_dim is None else paths[-1]
        self.paths = paths
        self.init_pos = init_pos
        self.final_pos = final_pos

    @property
    def input_ports(self) -> np.ndarray:
        return self.init_pos + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.final_pos + self.shift


class NemsAnchor(GroupedPattern):
    def __init__(self, fin_spring_dim: Dim2, connector_dim: Dim2, top_spring_dim: Dim2 = None,
                 straight_connector: Optional[Dim2] = None, loop_connector: Optional[Dim3] = None,
                 pos_electrode_dim: Optional[Dim3] = None, neg_electrode_dim: Optional[Dim2] = None,
                 include_fin_dummy: bool = False):
        """NEMS anchor

        Args:
            fin_spring_dim: fixed fin dimension (x, y)
            top_spring_dim: fin dimension (x, y)
            connector_dim: connector dimension
            straight_connector: straight connector to the fin, box xy (overridden by loop connector)
            loop_connector: loop connector to the fin, xy dim and final width on the top part of loop
            pos_electrode_dim: positive electrode dimension
            neg_electrode_dim: negative electrode dimension
            include_fin_dummy: include fin dummy for mechanical simulation
        """
        self.fin_spring_dim = fin_spring_dim
        self.top_spring_dim = top_spring_dim
        self.connector_dim = connector_dim
        self.straight_connector = straight_connector
        self.loop_connector = loop_connector
        self.pos_electrode_dim = pos_electrode_dim
        self.neg_electrode_dim = neg_electrode_dim
        patterns = []
        c_ports = []

        top_spring_dim = fin_spring_dim if not top_spring_dim else top_spring_dim
        connector = Box(connector_dim).translate()
        if loop_connector is not None and straight_connector is None:
            loop = Pattern(Path(fin_spring_dim[1]).rotate(np.pi).turn(
                loop_connector[1], -np.pi, final_width=loop_connector[2], tolerance=0.001).segment(
                loop_connector[0]).turn(loop_connector[1], -np.pi, final_width=fin_spring_dim[1],
                                        tolerance=0.001).segment(loop_connector[0]))
            loop.center_align(connector).vert_align(connector, bottom=False, opposite=False)
            connector = GroupedPattern(connector, loop)
        elif straight_connector is not None:
            straight = Box(straight_connector)
            connector = GroupedPattern(connector,
                                       copy(straight).horz_align(connector).vert_align(connector, bottom=False,
                                                                                       opposite=True),
                                       copy(straight).horz_align(connector, left=False,
                                                                 opposite=False).vert_align(connector,
                                                                                            bottom=False,
                                                                                            opposite=True))
        a_port = (connector.center[0], connector.bounds[1] + fin_spring_dim[1] / 2)
        if include_fin_dummy:
            patterns.append(Box(fin_spring_dim).center_align(a_port))
        patterns.append(connector)
        if top_spring_dim is not None:
            top_spring = Box(top_spring_dim).center_align(
                connector).vert_align(connector, bottom=True, opposite=True)
            patterns.append(top_spring)
            if pos_electrode_dim is not None:
                pos_electrode = Box((pos_electrode_dim[0], pos_electrode_dim[1])).center_align(top_spring).vert_align(
                    top_spring, opposite=True).translate(dy=pos_electrode_dim[2])
                c_ports.append((pos_electrode.bounds[0], pos_electrode.center[1]))
                c_ports.append((pos_electrode.bounds[1], pos_electrode.center[1]))
                patterns.append(pos_electrode)
            if neg_electrode_dim is not None:
                neg_electrode_left = Box(neg_electrode_dim).horz_align(
                    top_spring, opposite=True).vert_align(top_spring)
                neg_electrode_right = Box(neg_electrode_dim).horz_align(
                    top_spring, left=False, opposite=True).vert_align(top_spring)
                c_ports.append((neg_electrode_left.bounds[0], neg_electrode_left.center[1]))
                c_ports.append((neg_electrode_right.bounds[1], neg_electrode_left.center[1]))
                patterns.extend([neg_electrode_left, neg_electrode_right])

        super(NemsAnchor, self).__init__(*patterns)
        self.translate(-a_port[0], -a_port[1])
    #
    # @property
    # def contact_ports(self) -> np.ndarray:
    #     return np.asarray(self.c_ports)



#
# class NemsMillerNode(GroupedPattern):
#     def __init__(self, waveguide_w: float, upper_interaction_l: float, lower_interaction_l: float,
#                  gap_w: float, bend_radius: float, bend_extension: float, lr_nanofin_w: float,
#                  ud_nanofin_w: float, lr_gap_w: float, ud_gap_w: float, lr_pad_dim: Optional[Dim2] = None,
#                  ud_pad_dim: Optional[Dim2] = None, lr_connector_dim: Optional[Dim2] = None,
#                  ud_connector_dim: Optional[Dim2] = None, shift: Tuple[float, float] = (0, 0)):
#         self.waveguide_w = waveguide_w
#         self.upper_interaction_l = upper_interaction_l
#         self.lower_interaction_l = lower_interaction_l
#         self.bend_radius = bend_radius
#         self.bend_extension = bend_extension
#         self.lr_nanofin_w = lr_nanofin_w
#         self.ud_nanofin_w = ud_nanofin_w
#         self.lr_pad_dim = lr_pad_dim
#         self.ud_pad_dim = ud_pad_dim
#         self.lr_connector_dim = lr_connector_dim
#         self.ud_connector_dim = ud_connector_dim
#         self.gap_w = gap_w
#
#         connectors, pads = [], []
#
#         bend_height = 2 * bend_radius + bend_extension
#         interport_distance = waveguide_w + 2 * bend_height + gap_w
#
#         if not upper_interaction_l <= lower_interaction_l:
#             raise ValueError("Require upper_interaction_l <= lower_interaction_l by convention.")
#
#         lower_path = Path(waveguide_w).dc((bend_radius, bend_height), lower_interaction_l, use_radius=True)
#         upper_path = Path(waveguide_w).dc((bend_radius, bend_height), upper_interaction_l,
#                                           (lower_interaction_l - upper_interaction_l) / 2,
#                                           inverted=True, use_radius=True)
#         upper_path.translate(dx=0, dy=interport_distance)
#
#         dc = Pattern(lower_path, upper_path)
#
#         nanofin_y = ud_nanofin_w / 2 + gap_w / 2 + waveguide_w + ud_gap_w
#         nanofins = [Box((lower_interaction_l, ud_nanofin_w)).center_align(dc).translate(dx=0, dy=-nanofin_y)]
#         pad_y = ud_connector_dim[1] + ud_pad_dim[1] / 2
#         pads += [Box(ud_pad_dim).center_align(nanofins[0]).translate(dy=-pad_y)]
#         connector = Box(ud_connector_dim).center_align(pads[0])
#         connectors += [copy(connector).vert_align(pads[0], bottom=True, opposite=True).horz_align(pads[0]),
#                        copy(connector).vert_align(pads[0], bottom=True, opposite=True).horz_align(pads[0], left=False)]
#
#         nanofin_x = lr_nanofin_w / 2 + lr_gap_w + upper_interaction_l / 2 + bend_radius + waveguide_w / 2
#         pad_x = lr_connector_dim[0] + lr_pad_dim[0] / 2
#         nanofin_y = bend_radius + waveguide_w + gap_w / 2 + bend_extension / 2
#
#         nanofins += [Box((lr_nanofin_w, bend_extension)).center_align(dc).translate(dx=-nanofin_x, dy=nanofin_y),
#                      Box((lr_nanofin_w, bend_extension)).center_align(dc).translate(dx=nanofin_x, dy=nanofin_y)]
#         pads += [Box(lr_pad_dim).center_align(nanofins[1]).translate(dx=-pad_x, dy=0),
#                  Box(lr_pad_dim).center_align(nanofins[2]).translate(dx=pad_x, dy=0)]
#         connector = Box(lr_connector_dim).center_align(pads[1])
#         connectors += [copy(connector).horz_align(pads[1], left=True, opposite=True).vert_align(pads[1]),
#                        copy(connector).horz_align(pads[1], left=True, opposite=True).vert_align(pads[1], bottom=False)]
#         connector = Box(lr_connector_dim).center_align(pads[2])
#         connectors += [copy(connector).horz_align(pads[2], left=False, opposite=True).vert_align(pads[2]),
#                        copy(connector).horz_align(pads[2], left=False, opposite=True).vert_align(pads[2], bottom=False)]
#
#         super(NemsMillerNode, self).__init__(*([dc] + nanofins + connectors + pads), shift=shift)
#         self.dc, self.connectors, self.nanofins, self.pads = dc, connectors, nanofins, pads
#
#     @property
#     def input_ports(self) -> np.ndarray:
#         bend_height = 2 * self.bend_radius + self.bend_extension
#         return np.asarray(((0, 0), (0, self.waveguide_w + 2 * bend_height + self.gap_w)))
#
#     @property
#     def output_ports(self) -> np.ndarray:
#         # TODO(sunil): change this to correct method
#         return self.input_ports + np.asarray((self.size[0], 0))
#
#     def multilayer(self, waveguide_layer: str='seam', metal_stack_layers: Tuple[str, ...] = ('m1am', 'm2am'), via_stack_layers: Tuple[str, ...] = ('cbam', 'v1am'),
#                    clearout_layer: str, clearout_etch_stop_layer: str, contact_box_dim: Dim2, clearout_box_dim: Dim2,
#                    doping_stack_layer: Optional[str] = None,
#                    clearout_etch_stop_grow: float = 0, via_shrink: float = 1, doping_grow: float = 0.25) -> Multilayer:
#         return multilayer(self, self.pads, ((self.center[0], self.pads[1].center[1]),),
#                           waveguide_layer, metal_stack_layers,
#                           via_stack_layers, clearout_layer, clearout_etch_stop_layer, contact_box_dim,
#                           clearout_box_dim, doping_stack_layer, clearout_etch_stop_grow, via_shrink, doping_grow)


class EutecticOctagon(Pattern):
    def __init__(self, width: float):
        a = width / (1 + np.sqrt(2))
        poly = Polygon([(width / 2, a), (a, width / 2), (-a, width / 2), (-width / 2, a),
                        (-width / 2, -a), (-a, -width / 2), (a, -width / 2), (-width / 2, a)])

        super(EutecticOctagon, self).__init__(poly)


# class RingResonator(Pattern):
#     def __init__(self, waveguide_w: float, taper_l: float = 0,
#                  taper_params: Union[np.ndarray, List[float]] = None,
#                  length: float = 5, num_taper_evaluations: int = 100, end_l: float = 0,
#                  shift: Dim2 = (0, 0), layer: int = 0):
#         self.end_l = end_l
#         self.length = length
#         self.waveguide_w = waveguide_w
#         p = Path(waveguide_w).segment(end_l, layer=layer) if end_l > 0 else Path(waveguide_w)
#         if end_l > 0:
#             p.segment(end_l, layer=layer)
#         if taper_l > 0 or taper_params is not None:
#             p.polynomial_taper(taper_l, taper_params, num_taper_evaluations, layer)
#         p.segment(length, layer=layer)
#         if taper_l > 0 or taper_params is not None:
#             p.polynomial_taper(taper_l, taper_params, num_taper_evaluations, layer, inverted=True)
#         if end_l > 0:
#             p.segment(end_l, layer=layer)
#         super(RingResonator, self).__init__(p, shift=shift)
#
#     @property
#     def input_ports(self) -> np.ndarray:
#         return np.asarray((0, 0)) + self.shift
#
#     @property
#     def output_ports(self) -> np.ndarray:
#         return self.input_ports + np.asarray((self.size[0], 0))


class MemsMonitorCoupler(Pattern):
    def __init__(self, waveguide_w: float, interaction_l: float, gap_w: float,
                 end_l: float, detector_wg_l: float, bend_radius: float = 3, pad_dim: Optional[Dim2] = None,
                 rib_pad_w: float = 0):
        self.waveguide_w = waveguide_w
        self.interaction_l = interaction_l
        self.detector_wg_l = detector_wg_l
        self.gap_w = gap_w
        self.end_l = end_l
        self.bend_radius = bend_radius
        self.pad_dim = pad_dim

        pads = []

        waveguide = Path(width=waveguide_w).segment(interaction_l)
        monitor_wg = copy(waveguide).translate(dx=0, dy=gap_w + waveguide_w)
        monitor_left = Path(width=waveguide_w).rotate(np.pi).turn(bend_radius, -np.pi / 2).segment(detector_wg_l).turn(
            bend_radius, -np.pi / 2).translate(dx=0, dy=gap_w + waveguide_w)
        monitor_right = Path(width=waveguide_w).turn(bend_radius, np.pi / 2).segment(detector_wg_l).turn(
            bend_radius, np.pi / 2).translate(dx=interaction_l, dy=gap_w + waveguide_w)
        pad_y = waveguide_w * 3 / 2 + gap_w + pad_dim[1] / 2 + rib_pad_w
        pads.append(
            Path(width=pad_dim[1]).segment(pad_dim[0]).translate(dx=interaction_l / 2 - pad_dim[0] / 2, dy=pad_y))
        if rib_pad_w > 0:
            pads.append(Path(width=pad_dim[1]).segment(pad_dim[0]).translate(
                dx=0, dy=waveguide_w * 3 / 2 + gap_w + rib_pad_w / 2))

        super(MemsMonitorCoupler, self).__init__(waveguide, monitor_wg, monitor_left, monitor_right, *pads)
        self.pads = pads[:1]


def multilayer(waveguide_pattern: Pattern, pads: List[Pattern], clearout_areas: Tuple[Union[Dim2, Pattern], ...],
               waveguide_layer: str, metal_stack_layers: Tuple[str, ...],
               via_stack_layers: Tuple[str, ...], clearout_layer: str, clearout_etch_stop_layer: str, contact_box_dim: Dim2,
               clearout_box_dim: Dim2, doping_stack_layer: Optional[str] = None,
               clearout_etch_stop_grow: float = 0, via_shrink: float = 1, doping_grow: float = 0.25) -> Multilayer:
    pattern_to_layer = {GroupedPattern(*[Box(contact_box_dim).center_align(pad) for pad in pads]): layer
                        for layer in metal_stack_layers}
    pattern_to_layer.update({GroupedPattern(*[Box(contact_box_dim).center_align(pad).grow(-via_shrink)
                                              for pad in pads]): layer for layer in via_stack_layers})
    if doping_stack_layer is not None:
        pattern_to_layer[GroupedPattern(*[Box(pad.size).center_align(pad).grow(doping_grow)
                                          for pad in pads])] = doping_stack_layer

    pattern_to_layer[waveguide_pattern] = waveguide_layer

    for region in clearout_areas:
        clearout = Box(clearout_box_dim).center_align(region if isinstance(region, tuple) else region.center)
        pattern_to_layer[clearout] = clearout_layer
        if clearout_etch_stop_grow > 0 and clearout_etch_stop_layer is not None:
            pattern_to_layer[clearout.grow(clearout_etch_stop_grow)] = clearout_etch_stop_layer

    return Multilayer(pattern_to_layer)
