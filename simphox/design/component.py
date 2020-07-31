import gdspy as gy
import copy
from shapely.vectorized import contains
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union
from descartes import PolygonPatch

try:
    import plotly.graph_objects as go
    import nazca as nd
except ImportError:
    pass

from simphox.typing import *


class Path(gy.Path):
    def poly_taper(self, length: float, taper_params: Union[np.ndarray, List[float]],
                   num_taper_evaluations: int = 100, layer: int = 0, inverted: bool = False):
        curr_width = self.w * 2
        taper_params = np.asarray(taper_params)
        self.parametric(lambda u: (length * u, 0),
                        lambda u: (1, 0),
                        final_width=lambda u: curr_width - np.sum(taper_params) +
                                              np.sum(taper_params * (1 - u) ** np.arange(taper_params.size,
                                                                                         dtype=float)) if inverted
                        else curr_width +
                             np.sum(taper_params * u ** np.arange(taper_params.size, dtype=float)),
                        number_of_evaluations=num_taper_evaluations,
                        layer=layer)
        return self

    def sbend(self, bend_dim: Dim2, layer: int = 0, inverted: bool = False):
        pole_1 = np.asarray((bend_dim[0] / 2, 0))
        pole_2 = np.asarray((bend_dim[0] / 2, (-1) ** inverted * bend_dim[1]))
        pole_3 = np.asarray((bend_dim[0], (-1) ** inverted * bend_dim[1]))
        self.bezier([pole_1, pole_2, pole_3], layer=layer)
        return self

    def dc(self, bend_dim: Dim2, interaction_length: float, end_length: float = 0, layer: int = 0,
           inverted: bool = False, end_bend_dim: Optional[Dim3] = None):
        if end_bend_dim:
            if end_bend_dim[-1] > 0:
                self.segment(end_bend_dim[-1], layer=layer)
            self.sbend(end_bend_dim[:2], layer, inverted)
        if end_length > 0:
            self.segment(end_length, layer=layer)
        self.sbend(bend_dim, layer, inverted)
        self.segment(interaction_length, layer=layer)
        self.sbend(bend_dim, layer, not inverted)
        if end_length > 0:
            self.segment(end_length, layer=layer)
        if end_bend_dim:
            self.sbend(end_bend_dim[:2], layer, not inverted)
            if end_bend_dim[-1] > 0:
                self.segment(end_bend_dim[-1], layer=layer)
        return self

    def mzi(self, bend_dim: Dim2, interaction_length: float, arm_length: float,
            end_length: float = 0, layer: int = 0, inverted: bool = False, end_bend_dim: Optional[Dim3] = None):
        if end_bend_dim:
            if end_bend_dim[-1] > 0:
                self.segment(end_bend_dim[-1], layer=layer)
            self.sbend(end_bend_dim[:2], layer, inverted)
        if end_length > 0:
            self.segment(end_length, layer=layer)
        self.sbend(bend_dim, layer, inverted)
        self.segment(interaction_length, layer=layer)
        self.sbend(bend_dim, layer, not inverted)
        self.segment(arm_length, layer=layer)
        self.sbend(bend_dim, layer, inverted)
        self.segment(interaction_length, layer=layer)
        self.sbend(bend_dim, layer, not inverted)
        if end_length > 0:
            self.segment(end_length, layer=layer)
        if end_bend_dim:
            self.sbend(end_bend_dim[:2], layer, not inverted)
            if end_bend_dim[-1] > 0:
                self.segment(end_bend_dim[-1], layer=layer)
        return self

    def to(self, port: Dim2):
        return self.sbend((port[0] - self.x, port[1] - self.y))


class Component:
    def __init__(self, *polygons: Union[Path, gy.Polygon, gy.FlexPath, Polygon], shift: Dim2 = (0, 0), layer: int = 0):
        self.shift = shift
        self.layer = layer
        self.config = copy.deepcopy(self.__dict__)

        self.polys = polygons
        self.pattern = self._pattern()
        if shift != (0, 0):
            self.translate(shift[0], shift[1])

    def _pattern(self) -> MultiPolygon:
        if not isinstance(self.polys[0], Polygon):
            polygon_list = []
            for shape in self.polys:
                polygons = shape.get_polygons() if isinstance(shape, gy.FlexPath) else shape.polygons
                polygon_list += [Polygon(polygon_point_list) for polygon_point_list in polygons]
        else:
            polygon_list = self.polys
        pattern = cascaded_union(polygon_list)
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

    def translate(self, dx: float = 0, dy: float = 0) -> "Component":
        for path in self.polys:
            path.translate(dx, dy)
        self.pattern = self._pattern()
        self.shift = (self.shift[0] + dx, self.shift[1] + dy)
        self.config["shift"] = self.shift
        return self

    def centered(self, center: Tuple[float, float]) -> "Component":
        old_x, old_y = self.center
        self.translate(center[0] - old_x, center[1] - old_y)
        return self

    def to_gds(self, cell: gy.Cell):
        """

        Args:
            cell: GDSPY cell to add polygon

        Returns:

        """
        for path in self.polys:
            cell.add(gy.Polygon(np.asarray(path.exterior.coords.xy).T)) if isinstance(path, Polygon) else cell.add(path)

    def to_nazca_polys(self):
        return [nd.Polygon(np.asarray(poly.exterior.coords.xy).T) for poly in self.polys]

    def plotly3d(self, thickness: float, color: str, floor: float = 0, resolution: Union[int, Dim3] = (1000, 1000, 10)):
        resolution = (resolution, resolution, resolution) if isinstance(resolution, float) else resolution
        colorscale = [[0, color], [1, color]]

        def extrude(coords):
            v = np.mgrid[floor:floor + thickness:resolution[2] * 1j]
            xs, zs = np.meshgrid(coords[0], v)
            ys = np.tile(coords[1], (xs.shape[0], 1))
            return go.Surface(x=xs, y=ys, z=zs, colorscale=colorscale, showscale=False)

        plotly_objects = []

        xmin, ymin, xmax, ymax = self.bounds
        x_, y_ = np.mgrid[xmin:xmax:resolution[0] * 1j, ymin:ymax:resolution[1] * 1j]
        mask = contains(self.pattern, x_, y_)
        z_ = np.ones_like(mask, dtype=np.float)
        z_[~mask] = np.nan
        for poly in self.pattern:
            ext_coords = np.asarray(poly.exterior.coords.xy)
            plotly_objects.append(extrude(ext_coords))
            for pint in poly.interiors:
                int_coords = np.asarray(pint.coords.xy)
                plotly_objects.append(extrude(int_coords))
        plotly_objects += [go.Surface(x=x_, y=y_, z=floor * z_, colorscale=colorscale, showscale=False),
                           go.Surface(x=x_, y=y_, z=(floor + thickness) * z_, colorscale=colorscale, showscale=False)]
        return plotly_objects

    def plotly2d(self, color: str):
        # currently only works for exterior shapes
        plotly_objects = []

        for poly in self.pattern:
            plotly_objects.append(go.layout.Shape(
                type="path",
                path="".join(["M"] + [f"{point[0]},{point[1]} L" for point in np.asarray(poly.exterior.coords.xy).T])[
                     :-1],
                fillcolor=color,
                line_width=0
            ))

        return plotly_objects

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


class Box(Component):
    def __init__(self, box_dim: Dim2, shift: Dim2 = (0, 0), layer: int = 0):
        self.box_dim = box_dim

        super(Box, self).__init__(Path(box_dim[1]).segment(box_dim[0]).translate(dx=0, dy=box_dim[1] / 2), shift=shift,
                                  layer=layer)


class GratingPad(Component):
    def __init__(self, pad_dim: Dim2, taper_length: float, final_width: float, out: bool = False,
                 end_length: Optional[float] = None, bend_dim: Optional[Dim2] = None, shift: Dim2 = (0, 0),
                 layer: int = 0):
        self.pad_dim = pad_dim
        self.taper_length = taper_length
        self.final_width = final_width
        self.out = out
        self.bend_dim = bend_dim
        self.end_length = taper_length if end_length is None else end_length

        if out:
            path = Path(final_width)
            if end_length > 0:
                path.segment(end_length)
            if bend_dim:
                path.sbend(bend_dim)
            super(GratingPad, self).__init__(
                path.segment(taper_length, final_width=pad_dim[1]).segment(pad_dim[0]), shift=shift, layer=layer)
        else:
            path = Path(pad_dim[1]).segment(pad_dim[0]).segment(taper_length, final_width=final_width)
            if bend_dim:
                path.sbend(bend_dim, layer=layer)
            if end_length > 0:
                path.segment(end_length, layer=layer)
            super(GratingPad, self).__init__(path, shift=shift)

    def to(self, port: Dim2):
        if self.out:
            return self.translate(port[0], port[1])
        else:
            bend_y = self.bend_dim[1] if self.bend_dim else 0
            return self.translate(port[0] - self.size[0], port[1] - bend_y)

    @property
    def copy(self) -> "GratingPad":
        return copy.deepcopy(self)


class GroupedComponent(Component):
    def __init__(self, *components: Component, shift: Dim2 = (0, 0)):
        super(GroupedComponent, self).__init__(*sum([list(component.polys) for component in components], []),
                                               shift=shift)

    @classmethod
    def component_with_gratings(cls, component: Component, grating: GratingPad) -> "GroupedComponent":
        components = [component]
        out_config = copy.deepcopy(grating.config)
        out_config['out'] = True
        out_grating = GratingPad(**out_config)
        components.extend([grating.copy.to(port) for port in component.input_ports])
        components.extend([out_grating.copy.to(port) for port in component.output_ports])
        return cls(*components)


class DC(Component):
    def __init__(self, bend_dim: Dim2, waveguide_width: float, coupling_spacing: float, interaction_length: float,
                 end_length: float = 0, end_bend_dim: Optional[Dim3] = None, shift: Dim2 = (0, 0), layer: int = 0):
        self.end_length = end_length
        self.bend_dim = bend_dim
        self.waveguide_width = waveguide_width
        self.interaction_length = interaction_length
        self.coupling_spacing = coupling_spacing
        self.end_bend_dim = end_bend_dim

        interport_distance = waveguide_width + 2 * bend_dim[1] + coupling_spacing
        if end_bend_dim:
            interport_distance += 2 * end_bend_dim[1]

        lower_path = Path(waveguide_width).dc(bend_dim, interaction_length, end_length, end_bend_dim=end_bend_dim,
                                              layer=layer)
        upper_path = Path(waveguide_width).dc(bend_dim, interaction_length, end_length, end_bend_dim=end_bend_dim,
                                              inverted=True, layer=layer)
        upper_path.translate(dx=0, dy=interport_distance)

        super(DC, self).__init__(lower_path, upper_path, shift=shift, layer=layer)

    @property
    def input_ports(self) -> np.ndarray:
        interport_distance = self.waveguide_width + 2 * self.bend_dim[1] + self.coupling_spacing
        if self.end_bend_dim:
            interport_distance += 2 * self.end_bend_dim[1]
        return np.asarray(((0, 0), (0, interport_distance))) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.size[0], 0))


class MZI(Component):
    def __init__(self, bend_dim: Dim2, waveguide_width: float, arm_length: float, coupling_spacing: float,
                 interaction_length: float, end_length: float = 0, end_bend_dim: Optional[Dim3] = None,
                 shift: Dim2 = (0, 0), layer: int = 0):
        self.end_length = end_length
        self.arm_length = arm_length
        self.bend_dim = bend_dim
        self.waveguide_width = waveguide_width
        self.interaction_length = interaction_length
        self.coupling_spacing = coupling_spacing
        self.end_bend_dim = end_bend_dim

        lower_path = Path(waveguide_width).mzi(bend_dim, interaction_length, arm_length, end_length,
                                               end_bend_dim=end_bend_dim, layer=layer)
        upper_path = Path(waveguide_width).mzi(bend_dim, interaction_length, arm_length, end_length,
                                               end_bend_dim=end_bend_dim, inverted=True, layer=layer)
        upper_path.translate(dx=0, dy=waveguide_width + 2 * bend_dim[1] + coupling_spacing)

        super(MZI, self).__init__(lower_path, upper_path, shift=shift)

    @property
    def input_ports(self) -> np.ndarray:
        interport_distance = self.waveguide_width + 2 * self.bend_dim[1] + self.coupling_spacing
        if self.end_bend_dim:
            interport_distance += 2 * self.end_bend_dim[1]
        return np.asarray(((0, 0), (0, interport_distance))) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.size[0], 0))


class MMI(Component):
    def __init__(self, box_dim: Dim2, waveguide_width: float, interport_distance: float,
                 taper_dim: Dim2, end_length: float = 0, bend_dim: Optional[Tuple[float, float]] = None,
                 shift: Dim2 = (0, 0), layer: int = 0):
        self.end_length = end_length
        self.waveguide_width = waveguide_width
        self.box_dim = box_dim
        self.interport_distance = interport_distance
        self.taper_dim = taper_dim
        self.bend_dim = bend_dim

        if self.bend_dim:
            center = (end_length + bend_dim[0] + taper_dim[0] + box_dim[0] / 2, interport_distance / 2 + bend_dim[1])
            p_00 = Path(waveguide_width).segment(end_length, layer=layer) if end_length > 0 else Path(waveguide_width)
            p_00.sbend(bend_dim).segment(taper_dim[0], final_width=taper_dim[1], layer=layer)
            p_01 = Path(waveguide_width, (0, interport_distance + 2 * bend_dim[1]))
            p_01 = p_01.segment(end_length, layer=layer) if end_length > 0 else p_01
            p_01.sbend(bend_dim, inverted=True).segment(taper_dim[0], final_width=taper_dim[1], layer=layer)
        else:
            center = (end_length + taper_dim[0] + box_dim[0] / 2, interport_distance / 2)
            p_00 = Path(waveguide_width).segment(end_length, layer=layer) if end_length > 0 else Path(waveguide_width)
            p_00.segment(taper_dim[0], final_width=taper_dim[1], layer=layer)
            p_01 = copy.deepcopy(p_00).translate(dx=0, dy=interport_distance)
        mmi_start = (center[0] - box_dim[0] / 2, center[1])
        mmi = Path(box_dim[1], mmi_start).segment(box_dim[0], layer=layer)
        p_10 = copy.deepcopy(p_01).rotate(np.pi, center)
        p_11 = copy.deepcopy(p_00).rotate(np.pi, center)

        super(MMI, self).__init__(mmi, p_00, p_01, p_10, p_11, shift=shift)

    @property
    def input_ports(self) -> np.ndarray:
        bend_y = 2 * self.bend_dim[1] if self.bend_dim else 0
        return np.asarray(((0, 0), (0, self.interport_distance + bend_y))) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.size[0], 0))


class Waveguide(Component):
    def __init__(self, waveguide_width: float, taper_length: float = 0,
                 taper_params: Union[np.ndarray, List[float]] = None,
                 length: float = 5, num_taper_evaluations: int = 100, end_length: float = 0,
                 shift: Dim2 = (0, 0), layer: int = 0):
        self.end_length = end_length
        self.length = length
        self.waveguide_width = waveguide_width
        p = Path(waveguide_width).segment(end_length, layer=layer) if end_length > 0 else Path(waveguide_width)
        if end_length > 0:
            p.segment(end_length, layer=layer)
        if taper_length > 0 or taper_params is not None:
            p.poly_taper(taper_length, taper_params, num_taper_evaluations, layer)
        p.segment(length, layer=layer)
        if taper_length > 0 or taper_params is not None:
            p.poly_taper(taper_length, taper_params, num_taper_evaluations, layer, inverted=True)
        if end_length > 0:
            p.segment(end_length, layer=layer)
        super(Waveguide, self).__init__(p, shift=shift)

    @property
    def input_ports(self) -> np.ndarray:
        return np.asarray((0, 0)) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.size[0], 0))

#
# class RingResonator(Component):
#     def __init__(self, waveguide_width: float, taper_length: float = 0,
#                  taper_params: Union[np.ndarray, List[float]] = None,
#                  length: float = 5, num_taper_evaluations: int = 100, end_length: float = 0,
#                  shift: Dim2 = (0, 0), layer: int = 0):
#         self.end_length = end_length
#         self.length = length
#         self.waveguide_width = waveguide_width
#         p = Path(waveguide_width).segment(end_length, layer=layer) if end_length > 0 else Path(waveguide_width)
#         if end_length > 0:
#             p.segment(end_length, layer=layer)
#         if taper_length > 0 or taper_params is not None:
#             p.poly_taper(taper_length, taper_params, num_taper_evaluations, layer)
#         p.segment(length, layer=layer)
#         if taper_length > 0 or taper_params is not None:
#             p.poly_taper(taper_length, taper_params, num_taper_evaluations, layer, inverted=True)
#         if end_length > 0:
#             p.segment(end_length, layer=layer)
#         super(RingResonator, self).__init__(p, shift=shift)
#
#     @property
#     def input_ports(self) -> np.ndarray:
#         return np.asarray((0, 0)) + self.shift
#
#     @property
#     def output_ports(self) -> np.ndarray:
#         return self.input_ports + np.asarray((self.size[0], 0))
