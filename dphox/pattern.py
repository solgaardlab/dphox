import dataclasses
from copy import deepcopy as copy

import trimesh
from descartes import PolygonPatch
from shapely.affinity import translate, rotate
from shapely.geometry import GeometryCollection, Point
from shapely.ops import cascaded_union
from shapely.vectorized import contains
from trimesh import creation

from ..typing import *
from loguru import logger

NAZCA_IMPORTED = True

try:
    import plotly.graph_objects as go
except ImportError:
    pass

try:
    import nazca as nd
except ImportError:
    NAZCA_IMPORTED = False


@dataclasses.dataclass
class Port:
    """Port used in components in DPhox

    A port defines the center and angle/orientation in a design.

    Attributes:
        x: x position of the port
        y: y position of the port
        z: z position of the port (optional)
        a: angle (orientation) of the port (in degrees)
        w: the width of the port (optional, specified in design, mostly used for simulation)
        h: the height of the port (optional, not specified in design, mostly used for simulation)
    """
    x: float
    y: float
    a: float = 0
    w: float = 0
    z: float = 0
    h: float = 0

    def __post_init__(self):
        self.xy = (self.x, self.y)
        self.xya = (self.x, self.y, self.a)
        self.center = np.array((self.x, self.y, self.z))

    @property
    def size(self):
        if np.mod(self.a, 90) != 0:
            raise ValueError(f"Require angle to be a multiple a multiple of 90 but got {self.a}")
        return np.array((self.w, 0, self.h)) if np.mod(self.a, 180) != 0 else np.array((0, self.w, self.h))


class Path(gy.Path):
    def polynomial_taper(self, length: float, taper_params: Tuple[float, ...],
                         num_taper_evaluations: int = 100, layer: int = 0, inverted: bool = False):
        """

        Args:
            length: Length of the poly taper
            taper_params: Taper parameters
            num_taper_evaluations: Number of evaluations for the taper (see gdspy documentation)
            layer: Layer for the polynomial taper path
            inverted: Invert the polynomial

        Returns:
            The current path with the polynomial taper added

        """
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

    def sbend(self, bend_dim: Union[Size2, Size3], layer: int = 0, inverted: bool = False, use_radius: bool = False):
        """S bend using a bend dimension.

        Args:
            bend_dim: Bend dimension of the form :code:`(length, height, final_width)` or :code:`(length, height)`,
                where the latter is provided when the sbend is supposed to maintain the same final width throughout.
            layer: Layer for the sbend path.
            inverted: Bend down instead of bending up.
            use_radius: Use a radius/turn instead of bezier curve.

        Returns:
            The current path with the sbend added (see GDSPY chaining).

        """
        curr_width = 2 * self.w
        final_width = bend_dim[-1] if len(bend_dim) == 3 else curr_width
        if use_radius is False:
            pole_1 = np.asarray((bend_dim[0] / 2, 0))
            pole_2 = np.asarray((bend_dim[0] / 2, (-1) ** inverted * bend_dim[1]))
            pole_3 = np.asarray((bend_dim[0], (-1) ** inverted * bend_dim[1]))
            self.bezier([pole_1, pole_2, pole_3], final_width=final_width, layer=layer)
        else:
            halfway_final_width = (final_width + curr_width) / 2
            if bend_dim[1] > 2 * bend_dim[0]:
                angle = np.pi / 2 * (-1) ** inverted
                self.turn(bend_dim[0], angle, final_width=halfway_final_width, number_of_points=199)
                self.segment(bend_dim[1] - 2 * bend_dim[0])
                self.turn(bend_dim[0], -angle, final_width=final_width, number_of_points=199)
            else:
                angle = np.arccos(1 - bend_dim[1] / 2 / bend_dim[0]) * (-1) ** inverted
                self.turn(bend_dim[0], angle, final_width=halfway_final_width, number_of_points=199)
                self.turn(bend_dim[0], -angle, final_width=final_width, number_of_points=199)
        return self

    def dc(self, bend_dim: Union[Size2, Size3], interaction_l: float, end_l: float = 0, layer: int = 0,
           inverted: bool = False, end_bend_dim: Optional[Size3] = None, use_radius: bool = False):
        """Directional coupler waveguide path (includes the top or bottom path, end stub lengths,
        interaction length, additional end bends if necessary).

        Args:
            bend_dim: Bend dimension of the form :code:`(length, height, final_width)` or :code:`(length, height)`,
                where the latter is provided when the sbend is supposed to maintain the same final width throughout.
            interaction_l: Interaction length (length of the middle straight section of the directional coupler).
            end_l: End length
            layer: Layer of the directional coupler
            inverted: Bend down instead of bending up.
            end_bend_dim: Whether to include an additional end bend dimension for the directional coupler
            use_radius: Use a radius/turn instead of bezier curve.

        Returns:

        """
        curr_width = self.w * 2
        if end_bend_dim:
            if end_bend_dim[-1] > 0:
                self.segment(end_bend_dim[-1], layer=layer)
            self.sbend(end_bend_dim[:2], layer, inverted, use_radius)
        if end_l > 0:
            self.segment(end_l, layer=layer)
        self.sbend(bend_dim, layer, inverted, use_radius)
        self.segment(interaction_l, layer=layer)
        self.sbend((*bend_dim[:2], curr_width), layer, not inverted, use_radius)
        if end_l > 0:
            self.segment(end_l, layer=layer)
        if end_bend_dim:
            self.sbend(end_bend_dim[:2], layer, not inverted, use_radius)
            if end_bend_dim[-1] > 0:
                self.segment(end_bend_dim[-1], layer=layer)
        return self

    def trombone(self, height: float, radius: float):
        """Height and radius

        Args:
            height: Height of the trombone structure.
            radius: Radius of the trombone structure.

        Returns:
            The current path extended with trombone design.

        """
        self.turn(radius, 90, tolerance=0.001).segment(height)
        self.turn(radius, -180, tolerance=0.001).segment(height).turn(radius, np.pi / 2, tolerance=0.001)
        return self

    def to(self, port: Port):
        """Connect the path to a port

        Args:
            port:

        Returns:
            The current path extended to connect to the port

        """
        return self.sbend((port.x - self.x, port.y - self.y, port.w))


class Pattern:
    """Pattern corresponding to a patterned layer of material that may be used in a layout.

    A :code:`Pattern` is the core object in DPhox, which enables composition of multiple polygons or
    patterns into a single Pattern. It allows for composition of a myriad of different objects such as
    GDSPY Polygons and Shapely polygons into a single pattern. Since it is based on Shapely, this allows
    :code:`Pattern` to interface with other libraries such as :code:`Trimesh` and simulators codes such as
    MEEP and simphox for simulating the generated designs straightforwardly.

    Args:
        *patterns: A list of polygons specified by the user, can be a Path, gdspy Polygon, shapely Polygon,
            shapely MultiPolygon or Pattern.
        call_union: Call union on the polygons (useful in cases where there are polygons
            that are separated in a path).
        decimal_places: decimal places for rounding (in case of tiny errors in polygons)
    """

    def __init__(self, *patterns: Union["Pattern", Path, PolygonLike],
                 call_union: bool = True, decimal_places: int = 3):
        self.config = copy(self.__dict__)
        self.polys = []
        self.decimal_places = decimal_places
        for pattern in patterns:
            if isinstance(pattern, Pattern):
                self.polys += pattern.polys
            elif isinstance(pattern, np.ndarray):
                self.polys.append(Polygon(pattern))
            elif not isinstance(pattern, Polygon):
                if isinstance(pattern, MultiPolygon):
                    patterns = list(pattern)
                elif isinstance(pattern, gy.FlexPath):
                    patterns = pattern.get_polygons()
                elif isinstance(pattern, Path):
                    patterns = pattern.polygons
                self.polys += [Polygon(polygon_point_list) for polygon_point_list in patterns]
            else:
                self.polys.append(pattern)
        self.call_union = call_union
        self.shapely = self._shapely()
        self.port: Dict[str, Port] = {}
        self.reference_patterns: List[Pattern] = []

    @classmethod
    def from_shapely(cls, shapely_pattern: Union[Polygon, GeometryCollection]) -> "Pattern":
        try:
            collection = shapely_pattern if isinstance(shapely_pattern, Polygon) \
                else MultiPolygon([g for g in shapely_pattern.geoms if isinstance(g, Polygon)])
        except AttributeError:
            logger.exception(f'`shapely_pattern` is not a Polygon or a GeometryCollection it is a {type(shapely_pattern)},'
                             f'so it will be replaced with an empty MultiPolygon')
            collection = MultiPolygon([])
        return cls(collection)

    def _shapely(self) -> MultiPolygon:
        if not self.call_union:
            return MultiPolygon(self.polys)
        else:
            pattern = cascaded_union(self.polys)
            return pattern if isinstance(pattern, MultiPolygon) else MultiPolygon([pattern])

    def mask(self, shape: Shape, spacing: Spacing) -> np.ndarray:
        """Pixelized mask used for simulating this component

        Args:
            shape: Shape of the mask
            spacing: The grid spacing resolution to use for the pixelated mask

        Returns:
            An array of indicators of whether a volumetric image contains the mask

        """
        x_, y_ = np.mgrid[0:spacing[0] * shape[0]:spacing[0], 0:spacing[1] * shape[1]:spacing[1]]
        return contains(self.shapely, x_, y_)

    @property
    def bounds(self) -> Size4:
        """Bounds of the pattern

        Returns:
            Tuple of the form :code:`(minx, miny, maxx, maxy)`

        """
        return self.shapely.bounds

    @property
    def size(self) -> Size2:
        """Size of the pattern

        Returns:
            Tuple of the form :code:`(sizex, sizey)`

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return b[2] - b[0], b[3] - b[1]  # (maxx - minx, maxy - miny)

    @property
    def center(self) -> Size2:
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
            self.port[name] = Port(port.x + dx, port.y + dy, port.a, port.w)
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

    def flip(self, center: Size2 = (0, 0), horiz: bool = False) -> "Pattern":
        """Flip the component across a center point (default (0, 0))

        Args:
            center: The center point about which to flip
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
        self.port = {name: Port(-port.x + 2 * center[0], port.y, -port.a) if horiz else
        Port(port.x, -port.y + 2 * center[1], -port.a) for name, port in self.port.items()}
        return self

    def rotate(self, angle: float, origin: Tuple[float, float] = (0, 0)) -> "Pattern":
        """Runs Shapely's rotate operation on the geometry

        Args:
            angle: rotation angle in degrees
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
        self.port = {name: Port(float(point.x), float(point.y), self.port[name].a + angle)
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

    def nazca_cell(self, cell_name: str, layer: Union[int, str]) -> "nd.Cell":
        if not NAZCA_IMPORTED:
            raise ImportError('Nazca not installed! Please install nazca prior to running nazca_cell().')
        with nd.Cell(cell_name) as cell:
            for poly in self.polys:
                nd.Polygon(points=np.around(np.asarray(poly.exterior.coords.xy).T, decimals=3), layer=layer).put()
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

    def replace(self, pattern: "Pattern", center: Optional[Size2] = None, raise_port: bool = True):
        pattern_bbox = Pattern(Path(pattern.size[1]).segment(pattern.size[0]))
        align = self if center is None else center
        diff = self.difference(pattern_bbox.align(align))
        new_pattern = Pattern(diff, pattern.align(align), call_union=False)
        if raise_port:
            new_pattern.port = self.port
        return new_pattern

    def to(self, port: Port, from_port: Optional[str] = None):
        if from_port is None:
            return self.rotate(port.a).translate(port.x, port.y)
        else:
            return self.rotate(port.a - self.port[from_port].a + 180, origin=self.port[from_port].xy).translate(
                port.x - self.port[from_port].x, port.y - self.port[from_port].y
            )

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
