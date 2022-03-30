import numpy as np
from dataclasses import dataclass

from shapely.affinity import rotate
from shapely.geometry import box, GeometryCollection, LineString, LinearRing, Point, JOIN_STYLE, CAP_STYLE
from shapely.ops import split, unary_union, polygonize
from copy import deepcopy as copy

from .foundry import CommonLayer, fabricate, Foundry, FABLESS
from .geometry import Geometry
from .port import Port
from .typing import Float2, Float4, List, MultiPolygon, Optional, Polygon, PolygonLike, Shape, Spacing, \
    Union, Iterable
from .utils import DECIMALS, fix_dataclass_init_docs, min_aspect_bounds, poly_points, shapely_patch, split_holes

SHAPELYVEC_IMPORTED = True
GDSPY_IMPORTED = True


try:
    from shapely.vectorized import contains
except ImportError:
    SHAPELYVEC_IMPORTED = False

try:
    import gdspy as gy
except ImportError:
    GDSPY_IMPORTED = False


class Pattern(Geometry):
    """Pattern corresponding to a patterned layer of material that may be used in a layout.

    A :code:`Pattern` is a core object in DPhox, which enables composition of multiple polygons or
    patterns into a single Pattern. It allows for composition of a myriad of different objects such as
    GDSPY Polygons and Shapely polygons into a single pattern. Since :code:`Pattern` is a simple wrapper around Shapely's
    MultiPolygon, this class interfaces easily with other libraries such as :code:`Trimesh` and simulators codes such as
    MEEP and simphox for simulating the generated designs straightforwardly.

    Attributes:
        polygons: the numpy array representation for the polygons in this pattern.
        decimals: decimal places for rounding (in case of tiny errors in polygons)
    """

    def __init__(self, *patterns: Union["Pattern", PolygonLike, List[Union[PolygonLike, "Pattern"]]], decimals: int = 6):
        """Initializer for the pattern class.

        Args:
            *patterns: The patterns (Gdspy, Shapely, numpy array, Pattern)
            decimals: decimal places for rounding (in case of tiny errors in polygons)
        """
        self.decimals = decimals
        super().__init__(get_ndarray_polygons(patterns), {}, [])

    @property
    def shapely(self) -> MultiPolygon:
        return MultiPolygon([Polygon(np.around(p.T, decimals=self.decimals)) for p in self.geoms])

    @property
    def shapely_union(self) -> MultiPolygon:
        pattern = unary_union(self.shapely.geoms)
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
        geom = self.shapely_union
        if smooth_feature:
            # erode
            geom = geom.buffer(-smooth_feature, join_style=JOIN_STYLE.round, cap_style=CAP_STYLE.square)
            # dilate twice
            geom = geom.buffer(2 * smooth_feature, join_style=JOIN_STYLE.round, cap_style=CAP_STYLE.square)
            # erode
            geom = geom.buffer(-smooth_feature, join_style=JOIN_STYLE.round, cap_style=CAP_STYLE.square)
        return contains(geom.buffer(np.max(spacing)), x_, y_)  # need to use union!

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
        return Pattern(self.shapely_union.intersection(other_pattern.shapely_union))

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
        return Pattern(self.shapely_union.difference(other_pattern.shapely_union))

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
        return Pattern(self.shapely_union.union(other_pattern.shapely_union))

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
        return Pattern(self.shapely_union.symmetric_difference(other_pattern.shapely_union))

    def to_gdspy(self, cell):
        """Add to an existing GDSPY cell.

        Args:
            cell: GDSPY cell to add polygon

        Returns:
            The GDSPY cell corresponding to the :code:`Pattern`

        """
        import gdspy as gy
        for poly in self.geoms:
            cell.add(gy.Polygon(poly) if isinstance(poly, Polygon) else cell.add(poly))

    @property
    def bbox_pattern(self) -> "Box":
        """

        Returns:
            The linestring along the diagonal of the bbox

        """
        bbox = Box(self.size).align(self.center)
        bbox.port = self.port_copy
        return bbox

    def replace(self, pattern: "Pattern", center: Optional[Float2] = None, raise_port: bool = True):
        """Replace the polygons in some part of the image with :code:`pattern`.

        This is a useful property for inverse design. Note that the entire bounding box of the pattern is replaced
        by :code:`pattern` which may not always be desirable, but we estimate this is sufficient for most inverse
        design use cases.

        Args:
            pattern: The pattern which replaces this pattern.
            center: The center where to replace the pattern.
            raise_port: Whether to raise the port of this current pattern and add those ports to the new pattern.

        Returns:
            The new pattern with :code:`pattern` replacing the appropriate region of the image.

        """
        align = self if center is None else center
        diff = self.difference(Box(pattern.size).align(align))
        new_pattern = Pattern(diff, pattern.align(align))
        if raise_port:
            new_pattern.port = self.port
        return new_pattern

    def plot(self, ax: Optional = None, color: str = 'gray', plot_ports: bool = True, alpha: float = 1):
        """Plot the pattern

        Args:
            ax: Axis for plotting (if none, use the default matplotlib plot axis
            color: Color of the pattern
            plot_ports: Whether to plot the ports.
            alpha: the alpha property for plotting.

        Returns:

        """
        # import locally since this import takes some time.
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        ax.add_patch(shapely_patch(self.shapely_union, facecolor=color, edgecolor='none', alpha=alpha))
        if plot_ports:
            for name, port in self.port.items():
                port_xy = port.xy - port.tangent(port.w)
                ax.add_patch(shapely_patch(port.shapely,
                                           facecolor='red', edgecolor='none', alpha=alpha))
                ax.text(*port_xy, name)
        b = min_aspect_bounds(self.bounds)
        ax.set_xlim((b[0], b[2]))
        ax.set_ylim((b[1], b[3]))
        ax.set_aspect('equal')

    def buffer(self, distance: float, join_style=JOIN_STYLE.round, cap_style=CAP_STYLE.square) -> "Pattern":
        pattern = self.shapely_union.buffer(distance, join_style=join_style, cap_style=cap_style)
        return Pattern(pattern if isinstance(pattern, MultiPolygon) else pattern)

    def smooth(self, distance: float, min_area: float = None,
               join_style=JOIN_STYLE.round, cap_style=CAP_STYLE.square) -> "Pattern":
        smoothed = self.buffer(distance,
                               join_style=join_style,
                               cap_style=cap_style).buffer(-distance, join_style=join_style, cap_style=cap_style)
        smoothed_exclude = Pattern(smoothed.shapely_union.union(self.shapely_union) - self.shapely_union)
        min_area = distance ** 2 / 4 if min_area is None else min_area
        self.geoms += [p for p in smoothed_exclude.geoms if Polygon(p.T).area > min_area]
        return self

    def nazca_cell(self, cell_name: str, layer: Union[int, str]):
        try:
            import nazca as nd
        except ImportError:
            raise ImportError('Nazca not installed! Please install nazca prior to running nazca_cell().')
        with nd.Cell(cell_name) as cell:
            for poly in self.geoms:
                nd.Polygon(points=poly, layer=layer).to()
            for name, port in self.port.items():
                nd.Pin(name).to(*port.xya)
            nd.put_stub()
        return cell

    def hvplot(self, color: str = 'black', name: str = 'pattern', alpha: float = 0.5, bounds: Optional[Float4] = None,
               plot_ports: bool = True):
        """Plot this device on a matplotlib plot.

        Args:
            color: The color for plotting the pattern.
            name: Name of the pattern / label of the plot.
            alpha: The transparency factor for the plot (to see overlay of structures from many layers).
            plot_ports: Plot the ports (triangle indicators and text labels).

        Returns:
            The holoviews Overlay for displaying all of the polygons.

        """
        import holoviews as hv
        plots_to_overlay = []
        b = min_aspect_bounds(self.bounds) if bounds is None else bounds
        geom = self.shapely_union

        def _holoviews_poly(shapely_poly):
            x, y = poly_points(shapely_poly).T
            holes = [[np.array(hole.coords.xy).T for hole in shapely_poly.interiors]]
            return {'x': x, 'y': y, 'holes': holes}

        polys = [_holoviews_poly(poly) for poly in geom.geoms] if isinstance(geom, MultiPolygon) \
            else [_holoviews_poly(geom)]

        plots_to_overlay.append(
            hv.Polygons(polys, label=name).opts(data_aspect=1, frame_height=200, fill_alpha=alpha,
                                                ylim=(b[1], b[3]), xlim=(b[0], b[2]),
                                                color=color, line_alpha=0, tools=['hover']))

        if plot_ports:
            for name, port in self.port.items():
                plots_to_overlay.append(port.hvplot(name))

        return hv.Overlay(plots_to_overlay)

    def trimesh(self, foundry: Foundry = FABLESS, layer: CommonLayer = CommonLayer.RIDGE_SI):
        """Fabricate this pattern based on a :code:`Foundry`.

        This method is fairly rudimentary and will not implement things like conformal deposition. At the moment,
        you can implement things like rib etches which can be determined using 2d shape operations. Depositions in
        layers above etched layers will just start from the maximum z extent of the previous layer. This is specified
        by the :code:`Foundry` stack.

        Args:
            foundry: The foundry for each layer.
            layer: The layer for this pattern.

        Returns:
            The device :code:`Scene` to visualize.

        """
        return fabricate(
            layer_to_geom={layer: MultiPolygon([Polygon(p.T) for p in self.geoms])},
            foundry=foundry,
        )

    def __sub__(self, other: "Pattern"):
        return self.difference(other)

    def __or__(self, other: "Pattern"):
        return self.union(other)

    def __add__(self, other: "Pattern"):
        return self.union(other)  # TODO: switch to Pattern(self.geoms + other.geoms) after testing

    def __and__(self, other: "Pattern"):
        return self.intersection(other)

    def __xor__(self, other: "Pattern"):
        return self.symmetric_difference(other)

    @property
    def copy(self) -> "Pattern":
        return copy(self)


def get_ndarray_polygons(polylike_list: Iterable[Union["Pattern", PolygonLike, List[Union[PolygonLike, "Pattern"]]]],
                         decimals: int = DECIMALS):
    """A recursive list of lists of polylike objects, which turned into a flat list of 2d ndarray polygons.

    Args:
        polylike_list: List of polygon-like objects including :code:`Pattern`, shapely geometry collections,
            GDSPY geometries, and more.
        decimals: decimal precision of the resulting polygons

    Returns:
        A list of :math:`M` polygons that are each represented as :math:`2 \\times N_m` :code:`ndarray`'s.

    """
    polygons = []
    for pattern in polylike_list:
        if isinstance(pattern, list):
            # recursively apply to the list.
            polygons.extend(sum([get_ndarray_polygons([p]) for p in pattern], []))
        elif isinstance(pattern, Pattern):
            polygons.extend(pattern.geoms)
        elif isinstance(pattern, np.ndarray):
            polygons.append(pattern)
        elif isinstance(pattern, Polygon):
            pattern = split_holes(pattern)
            polygons.extend([poly_points(geom).T for geom in pattern.geoms])
        elif isinstance(pattern, MultiPolygon):
            polygons.extend([poly_points(geom).T for geom in split_holes(pattern).geoms])
        elif isinstance(pattern, GeometryCollection):
            polygons.extend([poly_points(geom).T for geom in split_holes(pattern).geoms])
        elif GDSPY_IMPORTED:
            if isinstance(pattern, gy.FlexPath):
                polygons.extend(pattern.get_polygons())
            elif isinstance(pattern, gy.Path):
                polygons.extend(pattern.polygons)
        else:
            raise TypeError(f'Pattern does not accept type {type(pattern)}')
    return [np.around(p, decimals=decimals) for p in polygons]


@fix_dataclass_init_docs
@dataclass
class Box(Pattern):
    """Box with default center at origin

    Attributes:
        extent: Dimension (box width, box height)
        decimals: The decimal places to resolve the box.
    """

    extent: Float2 = (1, 1)
    decimals: int = 6

    def __post_init__(self):
        super(Box, self).__init__(box(-self.extent[0] / 2, -self.extent[1] / 2,
                                      self.extent[0] / 2, self.extent[1] / 2),
                                  decimals=self.decimals)
        self.port = {
            'c': Port(*self.center),
            'n': Port(self.center[0], self.bounds[3], 90, self.extent[0]),
            'w': Port(self.bounds[0], self.center[1], -180, self.extent[1]),
            'e': Port(self.bounds[2], self.center[1], 0, self.extent[1]),
            's': Port(self.center[0], self.bounds[1], -90, self.extent[0])
        }

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

    def striped(self, stripe_w: float, pitch: Optional[Float2] = None,
                along_x: bool = True, along_y: bool = True, include_boundary: bool = True) -> Pattern:
        """A stripe hatch pattern in the confines of the box, useful for etching and arrays of square holes.

        Args:
            stripe_w: Stripe width (useful for etch holes).
            pitch: Pitch of the stripes
            along_x: Stripes / grating along x
            along_y: Stripes / grating along y

        Returns:
            The striped :code:`Pattern`

        """
        pitch = (stripe_w * 2, stripe_w * 2) if pitch is None else pitch
        patterns = []
        if include_boundary:
            patterns = [self.hollow(stripe_w)] if pitch[0] > 0 and pitch[1] > 0 else []
        if pitch[0] > 0 and not 3 * pitch[1] >= self.size[0] and along_x:
            xs = np.mgrid[self.bounds[0] + pitch[0]:self.bounds[2]:pitch[0]]
            patterns.append(
                Pattern(*[Box(extent=(stripe_w, self.size[1])).halign(x) for x in xs]).align(
                    self.center))
        if pitch[1] > 0 and not 3 * pitch[1] >= self.size[1] and along_y:
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

    def __post_init__(self):
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

    def __post_init__(self):
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

    def __post_init__(self):
        circle = Point(0, 0).buffer(self.radius, resolution=self.resolution)
        top_splitter = rotate(LineString([(0, self.radius), (0, 0)]), angle=self.angle / 2, origin=(0, 0))
        bottom_splitter = rotate(LineString([(0, 0), (0, self.radius)]), angle=-self.angle / 2, origin=(0, 0))
        super(Sector, self).__init__(
            split(circle, LineString([*top_splitter.coords, (0, 0), *bottom_splitter.coords]))[1])


@fix_dataclass_init_docs
@dataclass
class StripedBox(Pattern):
    extent: Float2
    stripe_w: float
    pitch: Union[float, Float2]

    def __post_init__(self):
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

    def __post_init__(self):
        super(MEMSFlexure, self).__init__(Box(self.extent).flexure(self.spring_extent, self.connector_extent,
                                                                   self.stripe_w, self.spring_center))
        self.box = Box(self.extent)
        self.refs.append(self.box)


@fix_dataclass_init_docs
@dataclass
class Quad(Pattern):
    start: Port
    end: Port

    def __post_init__(self):
        super(Quad, self).__init__(Pattern(np.hstack((self.start.line, self.end.line))))
        self.port = {'a0': self.start.copy, 'b0': self.end.copy}


def text(string: str, size: float = 12):
    """A simple function to generate text pattern from a string.

    Args:
        string: The string to turn into a pattern
        size: The fontsize size of the string.

    Returns:
        The text :code:`Pattern`.

    """
    import matplotlib.pyplot as plt
    from matplotlib.textpath import TextPath
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')

    try:
        usetex = True
        path = TextPath((0, 0), string, size=size, usetex=True)
    except FileNotFoundError:
        usetex = False
        path = TextPath((0, 0), string, size=size, usetex=False)
    rings = [LinearRing(p) for i, p in enumerate(path.to_polygons())]
    multipoly = MultiPolygon(polygonize(rings))

    filtered_polys = []
    for ring in rings:
        for poly in multipoly.geoms:
            if (Polygon(poly.exterior) ^ Polygon(ring)).area < 1e-6 and ring.is_ccw == usetex:
                filtered_polys.append(poly)

    return Pattern(filtered_polys)
