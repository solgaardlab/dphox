import pickle
from copy import deepcopy as copy
from enum import Enum

import gdspy as gy
import matplotlib.pyplot as plt
import numpy as np
import shapely.wkt as swkt
from descartes import PolygonPatch
from pydantic.dataclasses import dataclass
from shapely.affinity import rotate
from shapely.geometry import box, GeometryCollection, LineString, Point, JOIN_STYLE, CAP_STYLE
from shapely.ops import split, unary_union

from .typing import Dict, Float2, Float3, Float4, List, MultiPolygon, Optional, Polygon, PolygonLike, Shape, Spacing, \
    Union
from .utils import fix_dataclass_init_docs, poly_points, split_holes
from .transform import translate2d, reflect2d, skew2d, rotate2d, scale2d, AffineTransform

NAZCA_IMPORTED = True
PLOTLY_IMPORTED = True
SHAPELYVEC_IMPORTED = True

try:
    import plotly.graph_objects as go
except ImportError:
    PLOTLY_IMPORTED = False

try:
    import nazca as nd
except ImportError:
    NAZCA_IMPORTED = False

try:
    from shapely.vectorized import contains
except ImportError:
    SHAPELYVEC_IMPORTED = False

try:
    HOLOVIEWS_IMPORTED = True
    import holoviews as hv
    from holoviews import opts
    import panel as pn
except ImportError:
    HOLOVIEWS_IMPORTED = False


@fix_dataclass_init_docs
@dataclass
class TaperSpec:
    """A taper specification for waveguide width changes (useful in phase shifting and coupling).

    Attributes:
        length: The taper length
        params: The taper parameters for the polynomial taper.
        num_evaluations: Number of evaluations for the Taper
    """
    params: List[float]
    num_evaluations: int = 100

    @classmethod
    def linear(cls, change_w: float, num_evaluations: int = 100):
        params = [0., change_w]
        return cls(params, num_evaluations)

    @classmethod
    def cubic(cls, change_w: float, num_evaluations: int = 100):
        params = [0., 0., 3 * change_w, -2 * change_w]
        return cls(params, num_evaluations)


@fix_dataclass_init_docs
@dataclass
class Port:
    """Port used in components in DPhox

    A port defines the center, width and angle/orientation of a port in a design. Note that ports always are considered
    to have a width, and if a width is not provided the width is assumed to be 1 in the units of the file.

    Attributes:
        x: x position of the port.
        y: y position of the port.
        w: width of the port
        a: Angle (orientation) of the port (in degrees).
        z: z position of the port (optional, not specified in design, mostly used for simulation).
        h: the height / thickness of the port (optional, not specified in design, mostly used for simulation).
    """
    x: float = 0
    y: float = 0
    a: float = 0
    w: float = 1
    z: float = 0
    h: float = 0

    def __post_init_post_parse__(self):
        self.xy = np.array((self.x, self.y))
        self.xya = np.array((self.x, self.y, self.a))
        self.center = np.array((self.x, self.y, self.z))

    @property
    def size(self):
        """Get the size of the :code:`Port` for simulation-related applications.

        Returns:
            The size of the Port in 3D space, i.e., (x, y, z).

        """
        if np.mod(self.a, 90) != 0:
            raise ValueError(f"Require angle to be a multiple a multiple of 90 but got {self.a}")
        return np.array((self.w, 0, self.h)) if np.mod(self.a, 180) != 0 else np.array((0, self.w, self.h))

    @property
    def shapely(self) -> Polygon:
        """Return the :code:`Polygon` triangle corresponding to the :code:`Port`.

        Based on center and orientation of the :code:`Port`, return the corresponding Shapely triangle.
        This is effectively the inverse of the :code:`from_shapely` classmethod of this class.

        Returns:
            The shapely :code:`Polygon` triangle represented by the :code:`Port`.

        """
        dx, dy = np.sin(self.a * np.pi / 180) * self.w / 2, np.cos(self.a * np.pi / 180) * self.w / 2
        return Polygon(
            [(self.x - dx - dy * 0.75, self.y - dy - dx * 0.75), (self.x + dx - dy * 0.75, self.y + dy - dx * 0.75),
             (self.x, self.y)])

    @classmethod
    def from_shapely(cls, triangle: Polygon, z: float = 0, h: float = 0) -> "Port":
        """Initialize a :code:`Port` using a :code:`LineString` in Shapely.

        The port can be unambiguously defined using a line. The Shapely :code:`Polygon` triangle defines the
        center :math:`x, y` of the line as well as the width :math:`w` of the port. This is effectively the
        inverse of the :code:`shapely` property of this class.

        Args:
            triangle: Triangle representing the port.
            z: The z position of the port.
            h: The height / thickness of the port.

        Returns:
            The :code:`Port` represented by the shapely :code:`Polygon` triangle.

        """
        if not isinstance(triangle, Polygon):
            raise TypeError(f'Input line must be a shapely Polygon but got {type(triangle)}')

        points = poly_points(triangle)
        first, second, port_point, _ = points
        x, y = port_point
        c = (second[1] - first[1]) + (second[0] - first[0]) * 1j
        a = np.angle(c) * 180 / np.pi
        return cls(x, y, a, np.abs(c), z, h)

    @classmethod
    def from_linestring(cls, linestring: LineString, z: float = 0, h: float = 0) -> "Port":
        """Initialize a :code:`Port` using a :code:`LineString` in Shapely.

        The port can be unambiguously defined using a line. The Shapely :code:`Polygon` triangle defines the
        center :math:`x, y` of the line as well as the width :math:`w` of the port. This is effectively the
        inverse of the :code:`shapely` property of this class.

        Args:
            linestring: Shapely :code:`LineString` representing the port.
            z: The z position of the port.
            h: The height / thickness of the port.

        Returns:
            The :code:`Port` represented by the shapely :code:`Polygon` triangle.

        """
        if not isinstance(linestring, LineString):
            raise TypeError(f'Input line must be a shapely LineString but got {type(linestring)}')

        first, second = linestring.coords
        c, = linestring.centroid.coords
        d = (second[1] - first[1]) + (second[0] - first[0]) * 1j
        a = np.angle(d) * 180 / np.pi
        return cls(c[0], c[1], a, np.abs(d), z, h)

    def normal(self, scale: float = 1):
        """The vector normal to the port

        Args:
            scale: The magnitude of the normal vector

        Returns:
            Return the vector normal to the port

        """
        return np.array([np.cos(self.a * np.pi / 180), np.sin(self.a * np.pi / 180)]) * scale

    def transformed_line(self, transform: np.ndarray):
        dx, dy = np.sin(self.a * np.pi / 180) * self.w / 2, np.cos(self.a * np.pi / 180) * self.w / 2
        x, y = self.xy
        return np.array(transform[..., :2, :]) @ np.array([[x - dx, y - dy, 1], [x + dx, y + dy, 1]]).T

    def transform(self, transform: np.ndarray):
        line = self.transformed_line(transform).T
        c = (line[0] + line[1]) / 2
        d = (line[1, 1] - line[0, 1]) + (line[1, 0] - line[0, 0]) * 1j
        self.a = np.angle(d) * 180 / np.pi
        self.x = c[0]
        self.y = c[1]
        self.w = np.abs(d)
        self.xy = np.array((self.x, self.y))
        self.xya = np.array((self.x, self.y, self.a))
        self.center = np.array((self.x, self.y, self.z))
        return self

    def hvplot(self, name: str = 'port'):
        x, y = self.shapely.exterior.coords.xy
        port_xy = self.xy - self.normal(self.w)
        return hv.Polygons([{'x': x, 'y': y}]).opts(
            data_aspect=1, frame_height=200, color='red', line_alpha=0) * hv.Text(*port_xy, name)

    @property
    def copy(self) -> "Port":
        """Return a copy of this port for repeated use.

        Returns:
            A deep copy of this port.

        """
        return pickle.loads(pickle.dumps(self))


class GdspyPath(gy.Path):
    """This is just :code:`gdspy.Path` with some added sugar for tapering, bending, and more. The eventual plan is to
    deprecate this class in favor of an independent Shapely-based path solution.

    See Also:
        https://gdspy.readthedocs.io/en/stable/gettingstarted.html#paths

    """

    def polynomial_taper(self, taper_spec: TaperSpec, layer: int = 0, inverted: bool = False):
        """Polynomial taper for a GDSPY path.

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
        taper_params = np.asarray(taper_spec.params)
        length = taper_spec.length
        if self.direction == '+x':
            self.direction = 0
        elif self.direction == '-x':
            self.direction = np.pi
        elif self.direction == '+y':
            self.direction = np.pi / 2
        elif self.direction == '-y':
            self.direction = -np.pi / 2
        self.parametric(lambda u: (np.cos(self.direction) * length * u, np.sin(self.direction) * length * u),
                        lambda u: (np.cos(self.direction), np.sin(self.direction)),
                        final_width=lambda u: curr_width - np.sum(taper_params) + np.sum(
                            taper_params * (1 - u) ** np.arange(taper_params.size, dtype=float)) if inverted
                        else curr_width + np.sum(taper_params * u ** np.arange(taper_params.size, dtype=float)),
                        number_of_evaluations=taper_spec.num_evaluations, layer=layer)
        return self

    def sbend(self, bend_extent: Union[Float2, Float3], layer: int = 0, inverted: bool = False,
              use_radius: bool = False):
        """S bend using a bend dimension.

        Args:
            bend_extent: Bend dimension of the form :code:`(length, height, final_width)` or :code:`(length, height)`,
                where the latter is provided when the sbend is supposed to maintain the same final width throughout.
            layer: Layer for the sbend path.
            inverted: Bend down instead of bending up.
            use_radius: Use a radius/turn instead of bezier curve.

        Returns:
            The current path with the sbend added (see GDSPY chaining).

        """
        curr_width = 2 * self.w
        final_width = bend_extent[-1] if len(bend_extent) == 3 else curr_width
        if use_radius is False:
            pole_1 = np.asarray((bend_extent[0] / 2, 0))
            pole_2 = np.asarray((bend_extent[0] / 2, (-1) ** inverted * bend_extent[1]))
            pole_3 = np.asarray((bend_extent[0], (-1) ** inverted * bend_extent[1]))
            self.bezier([pole_1, pole_2, pole_3], final_width=final_width, layer=layer)
        else:
            halfway_final_width = (final_width + curr_width) / 2
            if bend_extent[1] > 2 * bend_extent[0]:
                angle = np.pi / 2 * (-1) ** inverted
                self.turn(bend_extent[0], angle, final_width=halfway_final_width, number_of_points=199)
                self.segment(bend_extent[1] - 2 * bend_extent[0])
                self.turn(bend_extent[0], -angle, final_width=final_width, number_of_points=199)
            else:
                angle = np.arccos(1 - bend_extent[1] / 2 / bend_extent[0]) * (-1) ** inverted
                self.turn(bend_extent[0], angle, final_width=halfway_final_width, number_of_points=199)
                self.turn(bend_extent[0], -angle, final_width=final_width, number_of_points=199)
        return self

    def dc(self, bend_extent: Union[Float2, Float3], interaction_l: float, end_l: float = 0, layer: int = 0,
           inverted: bool = False, end_bend_extent: Optional[Float3] = None, use_radius: bool = False):
        """Directional coupler waveguide path (includes the top or bottom path, end stub lengths,
        interaction length, additional end bends if necessary).

        Args:
            bend_extent: Bend dimension of the form :code:`(length, height, final_width)` or :code:`(length, height)`,
                where the latter is provided when the sbend is supposed to maintain the same final width throughout.
            interaction_l: Interaction length (length of the middle straight section of the directional coupler).
            end_l: End length
            layer: Layer of the directional coupler
            inverted: Bend down instead of bending up.
            end_bend_extent: Whether to include an additional end bend dimension for the directional coupler
            use_radius: Use a radius/turn instead of bezier curve.

        Returns:
            The current path with the dc path added (see GDSPY chaining).

        """
        curr_width = self.w * 2
        if end_bend_extent:
            if end_bend_extent[-1] > 0:
                self.segment(end_bend_extent[-1], layer=layer)
            self.sbend(end_bend_extent[:2], layer, inverted, use_radius)
        if end_l > 0:
            self.segment(end_l, layer=layer)
        self.sbend(bend_extent, layer, inverted, use_radius)
        self.segment(interaction_l, layer=layer)
        self.sbend((*bend_extent[:2], curr_width), layer, not inverted, use_radius)
        if end_l > 0:
            self.segment(end_l, layer=layer)
        if end_bend_extent:
            self.sbend(end_bend_extent[:2], layer, not inverted, use_radius)
            if end_bend_extent[-1] > 0:
                self.segment(end_bend_extent[-1], layer=layer)
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
        self.turn(radius, -180, tolerance=0.001).segment(height).turn(radius, 90, tolerance=0.001)
        return self

    def to(self, port: Port):
        """Connect the path to a port

        Args:
            port: Port.

        Returns:
            The current path extended to connect to the port

        """
        return self.sbend((port.x - self.x, port.y - self.y, port.w))


class Pattern:
    """Pattern corresponding to a patterned layer of material that may be used in a layout.

    A :code:`Pattern` is a core object in DPhox, which enables composition of multiple polygons or
    patterns into a single Pattern. It allows for composition of a myriad of different objects such as
    GDSPY Polygons and Shapely polygons into a single pattern. Since :code:`Pattern` is a simple wrapper around Shapely's
    MultiPolygon, this class interfaces easily with other libraries such as :code:`Trimesh` and simulators codes such as
    MEEP and simphox for simulating the generated designs straightforwardly.

    Attributes:
        polygons: the numpy array representation for the polygons in this pattern.
        poses: a pytree of affine matrix transformations (3x3) giving all of the layout info for the pattern.
        decimal_places: decimal places for rounding (in case of tiny errors in polygons)
    """

    def __init__(self, *patterns: Union["Pattern", GdspyPath, PolygonLike], decimal_places: int = 5):
        """Initializer for the pattern class.

        Args:
            *patterns: The patterns (Gdspy, Shapely, numpy array, Pattern)
            decimal_places: decimal places for rounding (in case of tiny errors in polygons)
        """
        self.decimal_places = decimal_places
        self.polygons = []

        for pattern in patterns:
            if isinstance(pattern, Pattern):
                self.polygons.extend(pattern.polygons)
            elif isinstance(pattern, np.ndarray):
                self.polygons.append(pattern)
            elif isinstance(pattern, Polygon):
                pattern = split_holes(pattern)
                self.polygons.extend([poly_points(geom).T for geom in pattern.geoms])
            elif isinstance(pattern, MultiPolygon):
                self.polygons.extend([poly_points(geom).T for geom in split_holes(pattern).geoms])
            elif isinstance(pattern, gy.FlexPath):
                self.polygons.extend(pattern.get_polygons())
            elif isinstance(pattern, GeometryCollection):
                self.polygons.extend([poly_points(geom).T for geom in split_holes(pattern).geoms])
            elif isinstance(pattern, GdspyPath):
                self.polygons.extend(pattern.polygons)
            else:
                raise TypeError(f'Pattern does not accept type {type(pattern)}')

        self.port: Dict[str, Port] = {}
        self.reference_patterns: List[Pattern] = []

    @property
    def all_points(self) -> np.ndarray:
        return np.hstack(self.polygons) if len(self.polygons) > 1 else self.polygons[0]

    def transformed_polygons(self, transform: Union[np.ndarray, List[np.ndarray], AffineTransform]) -> List[np.ndarray]:
        transformer = transform if isinstance(transform, AffineTransform) else AffineTransform(transform)
        return transformer.transform_polygons(self.polygons)

    @property
    def shapely(self) -> MultiPolygon:
        return swkt.loads(swkt.dumps(MultiPolygon([Polygon(p.T) for p in self.polygons]),
                                     rounding_precision=self.decimal_places))

    def shapely_union(self) -> MultiPolygon:
        pattern = unary_union(self.polys)
        return pattern if isinstance(pattern, MultiPolygon) else MultiPolygon([pattern])

    def mask(self, shape: Shape, spacing: Spacing, smooth_feature: float = 0) -> np.ndarray:
        """Pixelized mask used for simulating this component

        Args:
            shape: Shape of the mask
            spacing: The grid spacing resolution to use for the pixelated mask
            smooth_feature: The shapely smooth feature factor, which erodes, dilates twice, then erodes the geometry
                by :code:`smooth_feature` units.

        Returns:
            An array of indicators of whether a volumetric image contains the mask

        """
        if not SHAPELYVEC_IMPORTED:
            raise NotImplementedError('shapely.vectorized is not imported, but this relies on its contains method.')
        x_, y_ = np.mgrid[0:spacing[0] * shape[0]:spacing[0], 0:spacing[1] * shape[1]:spacing[1]]
        geom = self.shapely_union()
        if smooth_feature:
            # erode
            geom = geom.buffer(-smooth_feature, join_style=JOIN_STYLE.round, cap_style=CAP_STYLE.square)
            # dilate twice
            geom = geom.buffer(2 * smooth_feature, join_style=JOIN_STYLE.round, cap_style=CAP_STYLE.square)
            # erode
            geom = geom.buffer(-smooth_feature, join_style=JOIN_STYLE.round, cap_style=CAP_STYLE.square)
        return contains(geom, x_, y_)  # need to use union!

    @property
    def polys(self):
        return self.shapely.geoms

    @property
    def bounds(self) -> Float4:
        """Bounds of the pattern

        Returns:
            Tuple of the form :code:`(minx, miny, maxx, maxy)`

        """
        p = self.all_points
        if p.shape[1] > 0:
            return np.min(p[0]), np.min(p[1]), np.max(p[0]), np.max(p[1])
        else:
            return 0, 0, 0, 0

    @property
    def size(self) -> Float2:
        """Size of the pattern.

        Returns:
            Tuple of the form :code:`(sizex, sizey)`.

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return b[2] - b[0], b[3] - b[1]  # (maxx - minx, maxy - miny)

    @property
    def center(self) -> Float2:
        """Center of the pattern.

        Returns:
            Center for the component.

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return (b[2] + b[0]) / 2, (b[3] + b[1]) / 2  # (avgx, avgy)

    def transform(self, transform: np.ndarray):
        self.polygons = self.transformed_polygons(transform)
        self.port = {name: port.transform(transform) for name, port in self.port.items()}
        for pattern in self.reference_patterns:
            pattern.transform(transform)
        return self

    def translate(self, dx: float = 0, dy: float = 0) -> "Pattern":
        """Translate patter

        Args:
            dx: Displacement in x
            dy: Displacement in y

        Returns:
            The translated pattern

        """
        return self.transform(translate2d((dx, dy)))

    def reflect(self, center: Float2 = (0, 0), horiz: bool = False) -> "Pattern":
        """Reflect the component across a center point (default (0, 0))

        Args:
            center: The center point about which to flip
            horiz: do horizontal flip, otherwise vertical flip

        Returns:
            Flipped pattern

        """
        self.transform(reflect2d(center, horiz))
        for name, port in self.port.items():
            port.a = np.mod(port.a + 180, 360)  # temp fix... need to change how ports are represented
        return self

    def rotate(self, angle: float, origin: Union[Float2, np.ndarray] = (0, 0)) -> "Pattern":
        """Runs Shapely's rotate operation on the geometry about :code:`origin`.

        Args:
            angle: rotation angle in degrees.
            origin: origin of rotation.

        Returns:
            Rotated pattern by :code:`angle` about :code:`origin`

        """
        if angle % 360 != 0:  # to save time, only rotate if the angle is nonzero
            a = angle * np.pi / 180
            self.transform(rotate2d(a, origin))
        return self

    def skew(self, xs: float = 0, ys: float = 0, origin: Optional[Float2] = None) -> "Pattern":
        """Affine skew operation on the geometry about :code:`origin`.

        Args:
            xs: x skew factor.
            ys: y skew factor.
            origin: origin of rotation (uses center of pattern if :code:`None`).

        Returns:
            Rotated pattern by :code:`angle` about :code:`origin`

        """
        return self.transform(skew2d((xs, ys), self.center if origin is None else origin))

    def scale(self, xfact: float = 1, yfact: float = 1, origin: Optional[Float2] = None) -> "Pattern":
        """Affine scale operation on the geometry about :code:`origin`.

        Args:
            xfact: x scale factor.
            yfact: y scale factor.
            origin: origin of rotation (uses center of pattern if :code:`None`).

        Returns:
            Rotated pattern by :code:`angle` about :code:`origin`

        """
        return self.transform(scale2d((xfact, yfact), self.center if origin is None else origin))

    def align(self, pattern_or_center: Union["Pattern", Float2] = (0, 0),
              other: Union["Pattern", Float2] = None) -> "Pattern":
        """Align center of pattern

        Args:
            pattern_or_center: A pattern (align to the pattern's center) or a center point for alignment.
            other: If specified, instead of aligning based on this pattern's center,
                align based on another pattern's center and translate accordingly.

        Returns:
            Aligned pattern

        """
        if other is None:
            old_x, old_y = self.center
        else:
            old_x, old_y = other.center if isinstance(other, Pattern) else other
        center = pattern_or_center.center if isinstance(pattern_or_center, Pattern) else pattern_or_center
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

    def vstack(self, other_pattern: "Pattern", bottom: bool = False) -> "Pattern":
        return self.align(other_pattern).valign(other_pattern, bottom=bottom, opposite=True)

    def hstack(self, other_pattern: "Pattern", left: bool = False) -> "Pattern":
        return self.align(other_pattern).halign(other_pattern, left=left, opposite=True)

    @property
    def copy(self) -> "Pattern":
        """Copies the pattern using deepcopy.

        Returns:
            A copy of the Pattern so that changes do not propagate to the original :code:`Pattern`.

        """
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
        """Intersection between this pattern and provided pattern.

        Apply an intersection operation provided by Shapely's interface.

        See Also:
            https://shapely.readthedocs.io/en/stable/manual.html#object.intersection

        Args:
            other_pattern: Other pattern

        Returns:
            The new intersected pattern

        """
        return Pattern(self.shapely.intersection(other_pattern.shapely))

    def difference(self, other_pattern: "Pattern") -> "Pattern":
        """Difference between this pattern and provided pattern.

        Apply an difference operation provided by Shapely's interface. Note the distinction between
        :code:`difference` and :code:`symmetric_difference`.

        See Also:
            https://shapely.readthedocs.io/en/stable/manual.html#object.difference

        Args:
            other_pattern: Other pattern

        Returns:
            The new differenced pattern

        """
        return Pattern(self.shapely.difference(other_pattern.shapely))

    def union(self, other_pattern: "Pattern") -> "Pattern":
        """Union between this pattern and provided pattern.

        Apply an union operation provided by Shapely's interface.

        See Also:
            https://shapely.readthedocs.io/en/stable/manual.html#object.union

        Args:
            other_pattern: Other pattern

        Returns:
            The new union pattern

        """
        return Pattern(self.shapely.union(other_pattern.shapely))

    def symmetric_difference(self, other_pattern: "Pattern") -> "Pattern":
        """Union between this pattern and provided pattern.

        Apply a symmetric difference operation provided by Shapely's interface.

        See Also:
            https://shapely.readthedocs.io/en/stable/manual.html#object.symmetric_difference

        Args:
            other_pattern: Other pattern

        Returns:
            The new symmetric difference pattern

        """
        return Pattern(self.shapely.symmetric_difference(other_pattern.shapely))

    def to_gdspy(self, cell: gy.Cell):
        """Add to an existing GDSPY cell.

        Args:
            cell: GDSPY cell to add polygon

        Returns:
            The GDSPY cell corresponding to the :code:`Pattern`

        """
        for poly in self.polygons:
            cell.add(gy.Polygon(poly) if isinstance(poly, Polygon) else cell.add(poly))

    def replace(self, pattern: "Pattern", center: Optional[Float2] = None, raise_port: bool = True):
        pattern_bbox = Box.bbox(pattern)
        align = self if center is None else center
        diff = self.difference(pattern_bbox.align(align))
        new_pattern = Pattern(diff, pattern.align(align))
        if raise_port:
            new_pattern.port = self.port
        return new_pattern

    def plot(self, ax: Optional, color: str = 'black'):
        ax = plt.gca() if ax is None else ax
        ax.add_patch(PolygonPatch(self.shapely_union(), facecolor=color, edgecolor='none'))
        b = self.bounds
        ax.set_xlim((b[0], b[2]))
        ax.set_ylim((b[1], b[3]))
        ax.set_aspect('equal')

    def buffer(self, distance: float, join_style=JOIN_STYLE.round, cap_style=CAP_STYLE.square) -> "Pattern":
        pattern = self.shapely_union().buffer(distance, join_style=join_style, cap_style=cap_style)
        return Pattern(pattern if isinstance(pattern, MultiPolygon) else pattern)

    def smooth(self, distance: float, min_area: float = None,
               join_style=JOIN_STYLE.round, cap_style=CAP_STYLE.square) -> "Pattern":
        smoothed = self.buffer(distance,
                               join_style=join_style,
                               cap_style=cap_style).buffer(-distance, join_style=join_style, cap_style=cap_style)
        smoothed_exclude = Pattern(smoothed.shapely_union().union(self.shapely_union()).difference(
            self.shapely_union()))
        min_area = distance ** 2 / 4 if min_area is None else min_area
        self.polygons += [p for p in smoothed_exclude.shapely.geoms if p.area > min_area]
        return self

    def nazca_cell(self, cell_name: str, layer: Union[int, str]) -> "nd.Cell":
        if not NAZCA_IMPORTED:
            raise ImportError('Nazca not installed! Please install nazca prior to running nazca_cell().')
        with nd.Cell(cell_name) as cell:
            for poly in self.polygons:
                nd.Polygon(points=poly, layer=layer).put()
            for name, port in self.port.items():
                nd.Pin(name).put(*port.xya)
            nd.put_stub()
        return cell

    def put(self, port: Port, from_port: Optional[Union[str, Port]] = None):
        if from_port is None:
            return self.rotate(port.a).translate(port.x, port.y)
        else:
            fp = self.port[from_port] if isinstance(from_port, str) else from_port
            return self.rotate(port.a - self.port[from_port].a + 180, origin=fp.xy).translate(port.x - fp.x,
                                                                                              port.y - fp.y)

    def hvplot(self, color: str = 'black', name: str = 'pattern', alpha: float = 0.5, bounds: Optional[Float4] = None,
               plot_ports: bool = True):
        """Plot this device on a matplotlib plot.

        Args:
            color: The color for plotting the pattern.
            alpha: The transparency factor for the plot (to see overlay of structures from many layers).

        Returns:
            The holoviews Overlay for displaying all of the polygons.

        """
        if not HOLOVIEWS_IMPORTED:
            raise ImportError('Holoviews, Panel, and/or Bokeh not yet installed. Check your installation...')
        plots_to_overlay = []
        b = self.bounds if bounds is None else bounds
        geom = self.shapely_union()

        def _holoviews_poly(shapely_poly):
            x, y = poly_points(shapely_poly).T
            holes = [[np.array(hole.coords.xy).T for hole in shapely_poly.interiors]]
            return {'x': x, 'y': y, 'holes': holes}

        polys = [_holoviews_poly(poly) for poly in geom.geoms] if isinstance(geom, MultiPolygon) else [
            _holoviews_poly(geom)]

        plots_to_overlay.append(
            hv.Polygons(polys, label=name).opts(data_aspect=1, frame_height=200, fill_alpha=alpha,
                                                ylim=(b[1], b[3]), xlim=(b[0], b[2]),
                                                color=color, line_alpha=0, tools=['hover']))

        if plot_ports:
            for name, port in self.port.items():
                plots_to_overlay.append(port.hvplot(name))

        return hv.Overlay(plots_to_overlay)

    def __sub__(self, other: "Pattern"):
        return self.difference(other)

    def __add__(self, other: "Pattern"):
        return self.union(other)

    def __mul__(self, other: "Pattern"):
        return self.intersection(other)

    def __truediv__(self, other: "Pattern"):
        return self.symmetric_difference(other)


@fix_dataclass_init_docs
@dataclass
class Box(Pattern):
    """Box with default center at origin

    Attributes:
        extent: Dimension (box width, box height)
        decimal_places: The decimal places to resolve the box.
    """

    extent: Float2 = (1, 1)
    decimal_places: int = 3

    def __post_init_post_parse__(self):
        super(Box, self).__init__(box(-self.extent[0] / 2, -self.extent[1] / 2,
                                      self.extent[0] / 2, self.extent[1] / 2),
                                  decimal_places=self.decimal_places)
        self.port = {
            'c': Port(*self.center),
            'n': Port(self.center[0], self.bounds[3], 90, self.extent[0]),
            'w': Port(self.bounds[0], self.center[1], -180, self.extent[1]),
            'e': Port(self.bounds[2], self.center[1], 0, self.extent[1]),
            's': Port(self.center[0], self.bounds[1], -90, self.extent[0])
        }

    @classmethod
    def bbox(cls, pattern: Pattern) -> "Box":
        """Bounding box for pattern

        Args:
            pattern: The pattern over which to take a bounding box

        Returns:
            A bounding box pattern of the same size as :code:`pattern`

        """
        return cls(pattern.size).align(pattern)

    def expand(self, grow: float) -> "Box":
        """An aligned box that grows by amount :code:`grow`

        Args:
            grow: The amount to grow the box

        Returns:
            The box after the grow transformation

        """
        big_extent = (self.extent[0] + grow, self.extent[1] + grow)
        return Box(big_extent).align(self)

    def hollow(self, thickness: float) -> Pattern:
        """A hollow box of thickness :code:`thickness` on all four sides within the confines of the box extent.

        Args:
            thickness: thickness of the box.

        Returns:
            A box of specified :code:`thickness` with no filling inside.

        """
        return Pattern(
            self.difference(Box((self.extent[0] - 2 * thickness, self.extent[1])).align(self)),
            self.difference(Box((self.extent[0], self.extent[1] - 2 * thickness)).align(self)),
        )

    def cup(self, thickness: float) -> Pattern:
        """Return a cup-shaped (U-shaped) within the confines of the box extent.

        Args:
            thickness: thickness of the border.

        Returns:
            A cup-shaped block of thickness :code:`thickness`.

        """
        return Pattern(
            self.difference(Box((self.extent[0] - 2 * thickness, self.extent[1])).align(self)),
            self.difference(Box((self.extent[0], self.extent[1] - thickness)).align(self).valign(self)),
        )

    def ell(self, thickness: float) -> Pattern:
        """Return an ell-shaped (L-shaped) pattern within the confines of the box extent.

        Args:
            thickness: thickness of the border.

        Returns:
            An L-shaped block of thickness :code:`thickness`.

        """
        return Pattern(
            self.difference(Box((self.extent[0] - thickness, self.extent[1])).align(self).halign(self)),
            self.difference(Box((self.extent[0], self.extent[1] - thickness)).align(self).valign(self)),
        )

    def striped(self, stripe_w: float, pitch: Optional[Float2] = None) -> Pattern:
        """A stripe hatch pattern in the confines of the box, useful for etching and arrays of square holes.

        Args:
            stripe_w: Stripe width (useful for etch holes).
            pitch: Pitch of the stripes

        Returns:
            The striped :code:`Pattern`

        """
        pitch = (stripe_w * 2, stripe_w * 2) if pitch is None else pitch
        patterns = [self.hollow(stripe_w)] if pitch[0] > 0 and pitch[1] > 0 else []
        if pitch[0] > 0 and not 3 * pitch[1] >= self.size[0]:
            xs = np.mgrid[self.bounds[0] + pitch[0]:self.bounds[2]:pitch[0]]
            patterns.append(
                Pattern(*[Box(extent=(stripe_w, self.size[1])).halign(x) for x in xs]).align(
                    self.center))
        if pitch[1] > 0 and not 3 * pitch[1] >= self.size[1]:
            ys = np.mgrid[self.bounds[1] + pitch[1]:self.bounds[3]:pitch[1]]
            patterns.append(
                Pattern(*[Box(extent=(self.size[0], stripe_w)).valign(y) for y in ys]).align(
                    self.center))
        return Pattern(*patterns)

    def flexure(self, spring_extent: Float2, connector_extent: Float2 = None,
                stripe_w: float = 1, symmetric: bool = True, spring_center: bool = False) -> Pattern:
        """A crab-leg flexure (useful for MEMS actuation).

        Args:
            spring_extent: Spring extent (x, y).
            connector_extent: Connector extent (x, y).
            stripe_w: Stripe width (useful for etch holes, calls the striped method).
            symmetric: Whether to specify a symmetric connector.

        Returns:
            The flexure :code:`Pattern`

        """
        spring = Box(extent=spring_extent).align(self)
        connectors = []
        if connector_extent is not None:
            connector = Box(extent=connector_extent).align(self)
            if symmetric:
                connectors += [connector.copy.halign(self), connector.copy.halign(self, left=False)]
            else:
                connectors += [
                    connector.copy.valign(self).halign(self),
                    connector.copy.valign(self).halign(self, left=False)
                ]
        springs = [
            spring.copy.valign(self),
            spring.copy if spring_center else spring.copy.valign(self, bottom=False)
        ]
        return Pattern(self.striped(stripe_w), *springs, *connectors)


@fix_dataclass_init_docs
@dataclass
class Ellipse(Pattern):
    """Ellipse with default center at origin.

    Attributes:
        radius_extent: Dimension (ellipse x radius, ellipse y radius).
        resolution: Resolution is (number of points on circle) / 2.
    """

    radius_extent: Float2 = (1, 1)
    resolution: int = 16

    def __post_init_post_parse__(self):
        super(Ellipse, self).__init__(Point(0, 0).buffer(1, resolution=self.resolution))
        self.scale(*self.radius_extent)


@fix_dataclass_init_docs
@dataclass
class Circle(Pattern):
    """Ellipse with default center at origin.

    Attributes:
        diameter: diameter of the circle.
        resolution: Resolution is (number of points on circle) / 2.
    """

    radius: float = 1
    resolution: int = 16

    def __post_init_post_parse__(self):
        super(Circle, self).__init__(Point(0, 0).buffer(self.radius, resolution=self.resolution))


@fix_dataclass_init_docs
@dataclass
class Sector(Pattern):
    """Sector of a circle with center at origin.

    Attributes:
        radius: radius of the circle boundary of the sector.
        angle: angle of the sector.
        resolution: Resolution is (number of points on circle) / 2.
    """

    radius: float = 1
    angle: float = 180
    resolution: int = 16

    def __post_init_post_parse__(self):
        circle = Point(0, 0).buffer(self.radius, resolution=self.resolution)
        top_splitter = rotate(LineString([(0, self.radius), (0, 0)]), angle=self.angle / 2, origin=(0, 0))
        bottom_splitter = rotate(LineString([(0, 0), (0, self.radius)]), angle=-self.angle / 2, origin=(0, 0))
        super(Sector, self).__init__(split(circle, LineString([*top_splitter.coords, *bottom_splitter.coords]))[1])


@fix_dataclass_init_docs
@dataclass
class StripedBox(Pattern):
    extent: Float2
    stripe_w: float
    pitch: Union[float, Float2]

    def __post_init_post_parse__(self):
        super(StripedBox, self).__init__(Box(self.extent).striped(self.stripe_w, self.pitch))


@fix_dataclass_init_docs
@dataclass
class MEMSFlexure(Pattern):
    extent: Float2
    stripe_w: float
    pitch: Union[float, Float2]
    spring_extent: Float2
    connector_extent: Float2 = None
    spring_center: bool = False

    def __post_init_post_parse__(self):
        super(MEMSFlexure, self).__init__(Box(self.extent).flexure(self.spring_extent, self.connector_extent,
                                                                   self.stripe_w, self.spring_center))
        self.box = Box(self.extent)
        self.reference_patterns.append(self.box)
