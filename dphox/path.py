import numpy as np
from shapely.geometry import LineString, MultiLineString

from .pattern import Pattern
from .geometry import Geometry
from .port import Port
from .typing import CurveLike, CurveTuple, Float4, Iterable, List, Optional, PathWidth, Union
from .utils import DECIMALS, linestring_points, MAX_GDS_POINTS, min_aspect_bounds


class Curve(Geometry):
    """A discrete curve consisting of points and tangents that used to define paths of varying widths.

    Note:
        In our definition of curve, we allow for multiple curve segments that are unconnected to each other.

    Attributes:
        curve: A function :math:`f(t) = (x(t), y(t))`, given :math:`t \\in [0, 1]`, or a length (float),
            or a list of points, or a tuple of points and tangents.
        num_evaluations: Number of evaluations to define :math:`f(t)` (number of points in the curve).
    """

    def __init__(self, *curves: Union[float, "Curve", CurveLike, List[CurveLike]]):
        points, tangents = get_ndarray_curve(curves)
        super().__init__(points, {}, [], tangents)
        self.port = self.path_port()

    def angles(self, full: bool = True):
        """Calculate the angles for the tangents along the curve.

        Args:
            full: Whether to report the angles for the full coalesced curve.

        Returns:
            The angles of the tangents along the curve.

        """

        if full:
            t = np.hstack(self.tangents)
            return np.unwrap(np.arctan2(t[1], t[0]))
        else:
            return [np.unwrap(np.arctan2(t[1], t[0])) for t in self.tangents]

    def total_length(self, full: bool = True):
        """Calculate the total length at the end of each line segment of the curve.

        Args:
            full: Whether to report the lengths of the segments for the full coalesced curve.

        Returns:
            The lengths for the individual line segments of the curve.

        """

        if full:
            return np.cumsum(self.lengths())
        else:
            return [np.cumsum(p) for p in self.lengths(full=False)]

    def lengths(self, full: bool = True):
        """Calculate the lengths of each line segment of the curve.

        Args:
            full: Whether to report the lengths of the segments for the full coalesced curve.

        Returns:
            The lengths for the individual line segments of the curve.

        """

        if full:
            return np.linalg.norm(np.diff(self.points), axis=0)
        else:
            return [np.linalg.norm(np.diff(p), axis=0) for p in self.geoms]

    @property
    def pathlength(self):
        return np.sum(self.lengths(full=True))

    def curvature(self, full: bool = True, min_frac: float = 1e-3):
        """Calculate the curvature vs length along the curve.

        Args:
            full: Whether to report the curvature vs length for the full coalesced curve.
            min_frac: The minimum

        Returns:
            A tuple of the lengths and curvature along the length.

        """
        min_dist = min_frac * np.mean(self.lengths(full=True))
        if full:
            d, a = self.lengths(full=True), self.angles(full=True)
            return np.cumsum(d)[d > min_dist], np.diff(a)[d > min_dist] / d[d > min_dist]
        else:
            return [(np.cumsum(d)[d > min_dist], np.diff(a)[d > min_dist] / d[d > min_dist])
                    for d, a in zip(self.lengths(full=False), self.angles(full=False))]

    @property
    def normals(self):
        """Calculate the normals (perpendicular to the tangents) along the curve.

        Returns:
            The normals for the curve.

        """
        return [np.vstack((-np.sin(a), np.cos(a))) for a in self.angles()]

    def path_port(self, w: float = 1):
        """Get the port and orientations from the normals of the curve assuming it is a piecewise path.

        Note:
            This function will not make sense if there are multiple unconnected curves.
            This is generally reserved for path-related operations.
            Unexpected behavior will occur if this method is used for arbitrary curve sets.

        Args:
            w: width of the port.

        Returns:
            The ports for the curve.

        """
        n = self.normals
        n = (n[0].T[0], n[-1].T[-1])
        p = (self.geoms[0].T[0], self.geoms[-1].T[-1])
        return {
            'a0': Port.from_points(np.array((p[0] + n[0] * w / 2, p[0] - n[0] * w / 2))),
            'b0': Port.from_points(np.array((p[1] - n[1] * w / 2, p[1] + n[1] * w / 2)))
        }

    @property
    def shapely(self):
        """Shapely geometry

        Returns:
            The multiline string for the geometries.

        """
        return MultiLineString([LineString(p) for p in self.geoms])

    def coalesce(self):
        """Coalesce path segments into a single path

        Note:
            Caution: This assumes a C1 path, so paths with discontinuities will have incorrect tangents.

        Returns:
            The coalesced Curve.

        """
        self.geoms = [self.points]
        self.tangents = [np.hstack(self.tangents)]
        return self

    def interpolated(self, distance: float):
        """Interpolated curve such that all segments have equal length :code:`distance`.

        Args:
            distance: The length of segments in the interpolated path.

        Returns:
            The interpolated path.

        """
        return Curve([p.interpolate(distance) for p in self.shapely.geoms])

    def path(self, width: Union[float, Iterable[PathWidth]] = 1, offset: Union[float, Iterable[PathWidth]] = 0,
             decimals: int = DECIMALS) -> Pattern:
        """Path (pattern) converted from this curve using width and offset specifications.

        Args:
            width: Width of the path. If a list of callables, apply a parametric width to each curve segment.
            offset: Offset of the path. If a list of callables, apply a parametric offset to each curve segment.
            decimals: Decimal precision of the path.

        Returns:
            A pattern representing the path.

        """
        path_patterns = []

        widths = [width] * self.num_geoms if not isinstance(width, list) and not isinstance(width, tuple) else width
        offsets = [offset] * self.num_geoms if not isinstance(offset, list) and not isinstance(offset,
                                                                                               tuple) else offset

        if not len(widths) == self.num_geoms:
            raise AttributeError(f"Expected len(widths) == self.num_geoms, but got {len(widths)} != {self.num_geoms}")
        if not len(offsets) == self.num_geoms:
            raise AttributeError(f"Expected len(offsets) == self.num_geoms, but got {len(offsets)} != {self.num_geoms}")

        for segment, tangent, width, offset in zip(self.geoms, self.tangents, widths, offsets):
            if callable(width):
                t = np.linspace(0, 1, segment.shape[1])[:, np.newaxis]
                width = width(t)
            if callable(offset):
                t = np.linspace(0, 1, segment.shape[1])[:, np.newaxis]
                offset = offset(t)
            path_patterns.append(curve_to_path(segment, width, tangent, offset, decimals))

        path = Pattern(path_patterns).set_port({'a0': path_patterns[0].port['a0'],
                                                'b0': path_patterns[-1].port['b0']})
        path.refs.append(self)
        path.curve = self
        return path

    def hvplot(self, color: str = 'black', name: str = 'curve', bounds: Optional[Float4] = None,
               plot_ports: bool = True):
        """Plot this device on a matplotlib plot.

        Args:
            color: The color for plotting the pattern.
            name: Name of the curve.
            bounds: Bounds of the plot.
            plot_ports: Plot the ports of the curve.

        Returns:
            The holoviews Overlay for displaying all of the polygons.

        """
        import holoviews as hv
        plots_to_overlay = []
        b = min_aspect_bounds(self.bounds) if bounds is None else bounds

        for curve in self.geoms:
            plots_to_overlay.append(
                hv.Curve((curve[0], curve[1]), label=name).opts(data_aspect=1, frame_height=200,
                                                                ylim=(b[1], b[3]), xlim=(b[0], b[2]),
                                                                color=color, tools=['hover']))

        if plot_ports:
            for name, port in self.port.items():
                plots_to_overlay.append(port.hvplot(name))

        return hv.Overlay(plots_to_overlay)


def curve_to_path(points: np.ndarray, widths: Union[float, np.ndarray], tangents: np.ndarray,
                  offset: Union[float, np.ndarray] = 0, decimals: int = DECIMALS,
                  max_num_points: int = MAX_GDS_POINTS):
    """Converts a curve to a path.

    Args:
        points: The points along the curve.
        tangents: The normal directions / derivatives evaluated at the points along the curve.
        widths: The widths at each point along the curve (measured perpendicular to the tangents).
        offset: Offset of the path.
        decimals: Number of decimals precision for the curve output.
        max_num_points: Maximum number of points allowed in the curve (otherwise, break it apart).
            Note that the polygon will have twice this amount.

    Returns:
        The resulting Pattern.

    """

    # step 1: find the path polygon points based on the points, tangents, widths, and offset
    angles = np.arctan2(tangents[1], tangents[0])
    w = np.vstack((-np.sin(angles) * widths, np.cos(angles) * widths)) / 2
    off = np.vstack((-np.sin(angles) * offset, np.cos(angles) * offset)) / 2
    top_path = np.around(points + w + off, decimals).T
    bottom_path = np.around(points - w + off, decimals).T
    front_port = np.array([bottom_path[-1], top_path[-1]])
    back_port = np.array([top_path[0], bottom_path[0]])

    # step 2: split the path if there are too many points in it
    num_evaluations = top_path.shape[0]
    num_split = np.ceil(num_evaluations / max_num_points).astype(np.int32)
    ranges = [(i * max_num_points, (i + 1) * max_num_points + 1) for i in range(num_split)]

    # step 3: convert the resulting polygon list into a Pattern whose polygons form the path.
    pattern = Pattern([np.vstack((top_path[s[0]:s[1]], bottom_path[s[0]:s[1]][::-1])).T for s in ranges])
    pattern.port = {
        'a0': Port.from_points(back_port),
        'b0': Port.from_points(front_port)
    }

    return pattern


def get_ndarray_curve(curvelike_list: Iterable[Union[float, "Curve", CurveLike, List[CurveLike]]]):
    """A recursive list of lists of curvelike objects, which turned into a flat list of 2d ndarray polygons.

    Args:
        curvelike_list: List of polygon-like objects including :code:`CurveSet`, shapely linestrings,
            :code:`CurveTuple` (tuple of points and tangents), and more.

    Returns:
        A list of :math:`M` polygons that are each represented as :math:`2 \\times N_m` :code:`ndarray`'s.

    """
    linestrings = []
    tangents = []
    for curve in curvelike_list:
        new_tangents = []
        if isinstance(curve, CurveTuple):
            new_linestrings = [curve.points]
            new_tangents = [curve.tangents]
        elif np.isscalar(curve):
            # just a straight segment
            new_linestrings = np.array(((0, 0), (curve, 0))).T
        elif isinstance(curve, list) or isinstance(curve, tuple):
            # recursively apply to get list.
            new_linestrings_and_tangents = sum([get_ndarray_curve([p]) for p in curve], [])
            new_linestrings, new_tangents = zip(new_linestrings_and_tangents)
        elif isinstance(curve, Curve):
            new_linestrings = curve.geoms
            new_tangents = curve.tangents
        elif isinstance(curve, np.ndarray):
            if curve.ndim != 2 and curve.ndim != 3:
                raise AttributeError("The number of dimensions for the curve must be 2 or 3")
            new_linestrings = [curve] if curve.ndim == 2 else curve.tolist()
        elif isinstance(curve, LineString):
            new_linestrings = linestring_points(curve).T
        elif isinstance(curve, MultiLineString):
            new_linestrings = [linestring_points(geom).T for geom in curve.geoms]
        else:
            raise TypeError(f'Pattern does not accept type {type(curve)}')
        tangents.extend(new_tangents if new_tangents else [np.gradient(p, axis=0) for p in new_linestrings])
        linestrings.extend(new_linestrings)
    return linestrings, tangents


def link(*geoms: Union[Pattern, Curve], front_port: str = 'b0', back_port: str = 'a0'):
    """Link many separate curves or paths into a single geometry, assuming each geometry has a front and back port.

    Note:
        This is a simple linking function that simply uses the type of the first item in the list to attach
        either a set of paths or a set of curves to each other.

    Args:
        geoms: The paths to link, assuming the curve is the first ref in each path.
        front_port: Front port name.
        back_port: Back port name.

    Returns:
        The resulting geometry (path or curve) after linking many curves together into a single one.

    """
    port = geoms[0].port
    for geom in geoms[1:]:
        geom.put(port[front_port], from_port=back_port)
        port[front_port] = geom.port[front_port]
    if isinstance(geoms[0], Pattern):  # assume all patterns
        pattern = Pattern(*geoms).set_port(port)
        pattern.curve = Curve(*[path.curve for path in geoms])
        pattern.refs.append(pattern.curve)
        return pattern
    elif isinstance(geoms[0], Curve):  # assume all curves
        return Curve(*geoms)
    else:
        raise TypeError(f"Geometries must either be a pattern or curve but got {type(geoms[0])}")
