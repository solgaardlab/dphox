from collections import defaultdict, namedtuple

import gdspy as gy
import nazca as nd
from copy import deepcopy as copy
from shapely.vectorized import contains
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, Point
from shapely.ops import cascaded_union
from shapely.affinity import translate, rotate
from descartes import PolygonPatch
import trimesh
from trimesh import creation

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
                        final_width=lambda u: curr_width - np.sum(taper_params) + np.sum(
                            taper_params * (1 - u) ** np.arange(taper_params.size, dtype=float)) if inverted
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


class Port:
    def __init__(self, x: float, y: float, a: float = 0):
        """Port used by dphox

        Args:
            x: x position of the port
            y: y position of the port
            a: angle (orientation) of the port (in radians)
        """
        self.x = x
        self.y = y
        self.a = a
        self.a_deg = a * 180 / np.pi
        self.xy = (x, y)
        self.xya = (x, y, a)
        self.xya_deg = (x, y, a * 180 / np.pi)


class Pattern:
    """Pattern or layer of material used in a layout

        Args:
            *patterns: A list of polygons specified by the user, can be a Path, gdspy Polygon, shapely Polygon,
            shapely MultiPolygon or Pattern
            call_union: Do not call union
    """

    def __init__(self, *patterns: Union[Path, gy.Polygon, gy.FlexPath, Polygon, MultiPolygon, "Pattern", np.ndarray],
                 call_union: bool = True):
        self.config = copy(self.__dict__)
        self.polys = []
        for pattern in patterns:
            if isinstance(pattern, Pattern):
                self.polys += pattern.polys
            elif isinstance(pattern, np.ndarray):
                self.polys.append(Polygon(pattern))
            elif not isinstance(pattern, Polygon):
                if isinstance(pattern, MultiPolygon):
                    patterns = list(pattern)
                else:
                    patterns = pattern.get_polygons() if isinstance(pattern, gy.FlexPath) else pattern.polygons
                self.polys += [Polygon(polygon_point_list) for polygon_point_list in patterns]
            else:
                self.polys.append(pattern)
        self.call_union = call_union
        self.shapely = self._shapely()
        self.port: Dict[str, Port] = {}
        self.reference_patterns: List[Pattern] = []

    @classmethod
    def from_shapely(cls, shapely_pattern: Union[Polygon, GeometryCollection]) -> "Pattern":
        collection = shapely_pattern if isinstance(shapely_pattern, Polygon) \
            else MultiPolygon([g for g in shapely_pattern.geoms if isinstance(g, Polygon)])
        return cls(collection)

    def _shapely(self) -> MultiPolygon:
        if not self.call_union:
            return MultiPolygon(self.polys)
        else:
            pattern = cascaded_union(self.polys)
            return pattern if isinstance(pattern, MultiPolygon) else MultiPolygon([pattern])

    def mask(self, shape: Shape, grid_spacing: GridSpacing) -> np.ndarray:
        """Pixelized mask used for simulating this component

        Args:
            shape: Shape of the mask
            grid_spacing: The grid spacing resolution to use for the pixellized mask

        Returns:
            An array of indicators of whether a volumetric image contains the mask

        """
        x_, y_ = np.mgrid[0:grid_spacing[0] * shape[0]:grid_spacing[0], 0:grid_spacing[1] * shape[1]:grid_spacing[1]]

        return contains(self.shapely, x_, y_)

    @property
    def bounds(self) -> Dim4:
        """Bounds of the pattern

        Returns:
            Tuple of the form :code:`(minx, miny, maxx, maxy)`

        """
        return self.shapely.bounds

    @property
    def size(self) -> Dim2:
        """Size of the pattern

        Returns:
            Tuple of the form :code:`(sizex, sizey)`

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return b[2] - b[0], b[3] - b[1]  # (maxx - minx, maxy - miny)

    @property
    def center(self) -> Dim2:
        """

        Returns:
            Center for the component

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return (b[2] + b[0]) / 2, (b[3] + b[1]) / 2  # (avgx, avgy)

    def translate(self, dx: float = 0, dy: float = 0) -> "Pattern":
        """Translate patter

        Args:
            dx: Displacement in x
            dy: Displacement in y

        Returns:
            The translated pattern

        """
        self.polys = [translate(path, dx, dy) for path in self.polys]
        self.shapely = self._shapely()
        # any patterns in the element should also be translated
        for pattern in self.reference_patterns:
            pattern.translate(dx, dy)
        for name, port in self.port.items():
            self.port[name] = Port(port.x + dx, port.y + dy, port.a)
        return self

    def align(self, pattern_or_center: Union["Pattern", Tuple[float, float]],
              other: Union["Pattern", Tuple[float, float]] = None) -> "Pattern":
        """Align center of pattern

        Args:
            pattern_or_center: A pattern (align to the pattern's center) or a center point for alignment
            other: If specified, instead of aligning based on this pattern's center,
                align based on another pattern's center and translate accordingly.

        Returns:
            Aligned pattern

        """
        if other is None:
            old_x, old_y = self.center
        else:
            old_x, old_y = other if isinstance(other, tuple) else other.center
        center = pattern_or_center if isinstance(pattern_or_center, tuple) else pattern_or_center.center
        self.translate(center[0] - old_x, center[1] - old_y)
        return self

    def halign(self, c: Union["Pattern", float], left: bool = True, opposite: bool = False) -> "Pattern":
        """Horizontal alignment of pattern

        Args:
            c: A pattern (horizontal align to the pattern's boundary) or a center x for alignment
            left: (if :code:`c` is pattern) Align to left boundary of component, otherwise right boundary
            opposite: (if :code:`c` is pattern) Align opposite faces (left-right, right-left)

        Returns:
            Horizontally aligned pattern

        """
        x = self.bounds[0] if left else self.bounds[2]
        p = c if isinstance(c, float) or isinstance(c, int) \
            else (c.bounds[0] if left and not opposite or opposite and not left else c.bounds[2])
        self.translate(dx=p - x)
        return self

    def valign(self, c: Union["Pattern", float], bottom: bool = True, opposite: bool = False) -> "Pattern":
        """Vertical alignment of pattern

        Args:
            c: A pattern (vertical align to the pattern's boundary) or a center y for alignment
            bottom: (if :code:`c` is pattern) Align to upper boundary of component, otherwise lower boundary
            opposite: (if :code:`c` is pattern) Align opposite faces (upper-lower, lower-upper)

        Returns:
            Vertically aligned pattern

        """
        y = self.bounds[1] if bottom else self.bounds[3]
        p = c if isinstance(c, float) or isinstance(c, int) \
            else (c.bounds[1] if bottom and not opposite or opposite and not bottom else c.bounds[3])
        self.translate(dy=p - y)
        return self

    def flip(self, center: Dim2 = (0, 0), horiz: bool = False) -> "Pattern":
        """Flip the component across a center point (default (0, 0))

        Args:
            center:
            horiz: do horizontal flip, otherwise vertical flip

        Returns:
            Flipped pattern

        """
        new_polys = []
        for poly in self.polys:
            points = np.asarray(poly.exterior.coords.xy)
            new_points = np.stack((-points[0] + 2 * center[0], points[1])) if horiz \
                else np.stack((points[0], -points[1] + 2 * center[1]))
            new_polys.append(Polygon(new_points.T))
        self.polys = new_polys
        self.shapely = self._shapely()
        # any patterns in this pattern should also be flipped
        for pattern in self.reference_patterns:
            pattern.flip(center, horiz)
        self.port = {name: Port(-port.x + 2 * center[0], port.y, port.a) if horiz else
                     Port(port.x, -port.y + 2 * center[1], port.a) for name, port in self.port.items()}
        return self

    def rotate(self, angle: float, origin: str = (0, 0)) -> "Pattern":
        """Runs Shapely's rotate operation on the geometry

        Args:
            angle: rotation angle
            origin: origin of rotation

        Returns:
            Rotated pattern

        """
        self.shapely = rotate(self.shapely, angle, origin)
        self.polys = [poly for poly in self.shapely]
        # any patterns in this pattern should also be rotated
        for pattern in self.reference_patterns:
            pattern.rotate(angle, origin)
        port_to_point = {name: rotate(Point(*port.xy), angle, origin)
                         for name, port in self.port.items()}
        self.port = {name: Port(float(point.x), float(point.y), self.port[name].a + angle / 180 * np.pi)
                     for name, point in port_to_point.items()}
        return self

    @property
    def copy(self) -> "Pattern":
        return copy(self)

    def boolean_operation(self, other_pattern: "Pattern", operation: str):
        op_to_func = {
            'intersection': self.intersection,
            'difference': self.difference,
            'union': self.union,
            'symmetric_difference': self.symmetric_difference
        }
        boolean_func = op_to_func.get(operation,
                                      lambda: f"Not a valid boolean operation: Must be in {op_to_func.keys()}")
        return boolean_func(other_pattern)

    def intersection(self, other_pattern: "Pattern") -> "Pattern":
        return Pattern.from_shapely(self.shapely.intersection(other_pattern.shapely))

    def difference(self, other_pattern: "Pattern") -> "Pattern":
        return Pattern.from_shapely(self.shapely.difference(other_pattern.shapely))

    def union(self, other_pattern: "Pattern") -> "Pattern":
        return Pattern.from_shapely(self.shapely.union(other_pattern.shapely))

    def symmetric_difference(self, other_pattern: "Pattern") -> "Pattern":
        return Pattern.from_shapely(self.shapely.symmetric_difference(other_pattern.shapely))

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
        ax.add_patch(PolygonPatch(self.shapely, facecolor=color, edgecolor='none'))
        b = self.bounds
        ax.set_xlim((b[0], b[2]))
        ax.set_ylim((b[1], b[3]))
        ax.set_aspect('equal')

    def offset(self, grow_d: float) -> "Pattern":
        pattern = self.shapely.buffer(grow_d)
        return Pattern(pattern if isinstance(pattern, MultiPolygon) else pattern)

    def nazca_cell(self, cell_name: str, layer: Union[int, str]) -> nd.Cell:
        with nd.Cell(cell_name) as cell:
            for poly in self.polys:
                nd.Polygon(points=np.asarray(poly.exterior.coords.xy).T, layer=layer).put()
            for name, port in self.port.items():
                nd.Pin(name).put(*port.xya_deg)
            nd.put_stub()
        return cell

    def metal_contact(self, metal_layers: Tuple[str, ...],
                      via_sizes: Tuple[float, ...] = (0.4, 0.4, 3.6)):
        patterns = []
        for i, metal_layer in enumerate(metal_layers):
            if i % 2 == 0:
                pattern = Pattern(Path(via_sizes[i // 2]).segment(via_sizes[i // 2])).align(self)
            else:
                pattern = copy(self)
            patterns.append((pattern.align(self), metal_layer))
        return [(pattern, metal_layer) for pattern, metal_layer in patterns]

    def dope(self, dope_layer: str, dope_grow: float = 0.1):
        return self.copy.offset(dope_grow), dope_layer

    def clearout_box(self, clearout_layer: str, clearout_etch_stop_layer: str,
                     dim: Tuple[float, float], clearout_etch_stop_grow: float = 0.5,
                     center: Tuple[float, float] = None):
        center = center if center is not None else self.center
        box = Pattern(Path(dim[1]).segment(dim[0]).translate(dx=0, dy=dim[1] / 2)).align(center)
        box_grow = box.offset(clearout_etch_stop_grow)
        return [(box, clearout_layer), (box_grow, clearout_etch_stop_layer)]

    def replace(self, pattern: "Pattern", center: Optional[Dim2] = None, raise_port: bool = True):
        pattern_bbox = Pattern(Path(pattern.size[1]).segment(pattern.size[0]))
        align = self if center is None else center
        diff = self.difference(pattern_bbox.align(align))
        new_pattern = Pattern(diff, pattern.align(align), call_union=False)
        if raise_port:
            new_pattern.port = self.port
        return new_pattern

    def to(self, port: Port):
        return self.rotate(port.a_deg).translate(port.x, port.y)

    @classmethod
    def from_gds(cls, filename):
        """The top cell in a given GDS file is assigned to a pattern (not by spec, assumes single layer!)

        Args:
            filename: the GDS file for

        Returns:

        """
        lib = gy.GdsLibrary(infile=filename)
        main_cell = lib.top_level()[0]
        return cls(*main_cell.get_polygons())


# TODO(nate): find a better place for these functions


def cubic_taper(change_w, off: bool = False) -> Tuple[float, ...]:
    if off:
        return 0, change_w  # quick hack to change to linear taper
    else:
        return 0, 0, 3 * change_w, -2 * change_w


def is_adiabatic(taper_params, init_width: float = 0.48, wavelength: float = 1.55, neff: float = 2.75,
                 num_points: int = 100, taper_l: float = 5):
    taper_params = np.asarray(taper_params)
    u = np.linspace(0, 1 + 1 / num_points, num_points)[:, np.newaxis]
    width = init_width + np.sum(taper_params * u ** np.arange(taper_params.size, dtype=float), axis=1)
    theta = np.arctan(np.diff(width) / taper_l * num_points)
    max_pt = np.argmax(theta)
    return theta[max_pt], wavelength / (2 * width[max_pt] * neff)


def get_linear_adiabatic(min_width: float = 0.48, max_width: float = 1, wavelength: float = 1.55,
                         neff_max: float = 2.75, min_to_max: bool = True, aggressive: bool = False):
    taper_params = (0, max_width - min_width) if min_to_max else (0, min_width - max_width)
    taper_l = 1.1 * abs(max_width - min_width) / np.arctan(wavelength / (2 * max_width * neff_max)) \
        if aggressive else 2 * abs(max_width - min_width) / np.arctan(wavelength / (2 * max_width * neff_max))
    return taper_l, taper_params
