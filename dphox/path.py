from pydantic import Field
from pydantic.dataclasses import dataclass
from shapely.ops import split
from shapely.geometry import Polygon, MultiPolygon, LineString
from scipy.special import fresnel
import numpy as np

from .pattern import Box, Pattern, Port
from .typing import Callable, Float2, List, Optional, Union, Tuple
from .utils import fix_dataclass_init_docs, MAX_GDS_POINTS


def parametric(path: Union[float, Callable], width: Union[Callable, float, Tuple[float, float]],
               num_evaluations: int = 99, max_num_points: int = MAX_GDS_POINTS, decimal_places: int = 5):
    """Given :math:`t` parameter, this function defines

    Args:
        path: The path/curve function or length for a straight path.
        width: The width function as a function of :math:`t` or width for a constant width along the path/curve.
        num_evaluations: Number of points to evaluate the parametric function
        max_num_points: Maximum number of points to evaluate the parametric function (split path if too long)
        decimal_places: Number of decimal places precision

    Returns:
        A tuple of the multipolygon geometry, back port linestring (:code:`a0`), and front port linestring (:code:`b0`).

    """
    u = np.linspace(0, 1, num_evaluations)[:, np.newaxis]
    if isinstance(path, float) or isinstance(path, int):
        path = np.hstack([u * path, np.zeros_like(u)])
        path_diff = np.hstack([np.ones_like(u) / num_evaluations * path, np.zeros_like(u)]).T
    else:
        path = path(u)
        if isinstance(path, tuple):
            path, path_diff = path
            path_diff = path_diff.T
        else:
            path_diff = np.diff(path, axis=0)
            path_diff = np.vstack((path_diff[0], path_diff))
            path_diff = path_diff.T

    widths = width
    if not isinstance(width, float) and not isinstance(width, int):
        widths = taper_fn(width)(u) if isinstance(width, tuple) or isinstance(width, list) else width(u)

    angles = np.arctan2(path_diff[1], path_diff[0])
    width_translation = np.vstack((-np.sin(angles) * widths, np.cos(angles) * widths)).T / 2
    top_path = np.around(path + width_translation, decimal_places)
    bottom_path = np.around(path - width_translation, decimal_places)

    front_port = LineString([bottom_path[-1], top_path[-1]])
    back_port = LineString([top_path[0], bottom_path[0]])

    # TODO: convert to more efficient path fracture
    num_splitters = num_evaluations * 2 // max_num_points
    splitters = [LineString([top_path[(i + 1) * max_num_points // 2], bottom_path[(i + 1) * max_num_points // 2]])
                 for i in range(num_splitters)]
    polygon = Polygon(np.vstack((top_path, bottom_path[::-1])))
    polys = []
    for splitter in splitters:
        geoms = split(polygon, splitter).geoms
        poly, polygon = geoms[0], geoms[1]
        polys.append(poly)
    polys.append(polygon)
    pattern = Pattern(MultiPolygon(polys))
    pattern.port = {
        'a0': Port.from_linestring(back_port),
        'b0': Port.from_linestring(front_port)
    }
    return pattern


def bezier_sbend_fn(bend_x: float, bend_y: float):
    """Bezier sbend functional.

    Args:
        bend_x: Change in :math:`x` due to the bend.
        bend_y: Change in :math:`y` due to the bend.

    Returns:
        A function mapping 0 to 1 to the width of the taper along the path linestring.

    """
    pole_1 = np.asarray((bend_x / 2, 0))
    pole_2 = np.asarray((bend_x / 2, bend_y))
    pole_3 = np.asarray((bend_x, bend_y))

    def _sbend(t: np.ndarray):
        path = 3 * (1 - t) ** 2 * t * pole_1 + 3 * (1 - t) * t ** 2 * pole_2 + t ** 3 * pole_3
        derivative = 3 * (1 - t) ** 2 * pole_1 + 6 * (1 - t) * t * (pole_2 - pole_1) + 3 * t ** 2 * (pole_3 - pole_2)
        return path, derivative

    return _sbend


def taper_fn(taper_params: Union[np.ndarray, Tuple[float]]):
    """Taper functional.

    Args:
        taper_params: Polynomial taper parameter function of the form :math:`f(t; \\mathbf{x}) = \\sum_{n = 1}^N x_nt^n`

    Returns:
         A function mapping 0 to 1 to the width of the taper along the path linestring.

    """
    poly_exp = np.arange(len(taper_params), dtype=float)
    return lambda u: np.sum(taper_params * u ** poly_exp, axis=1)


def euler_bend_fn(radius: float, angle: float = 90):
    """Euler bend functional

    Args:
        radius: Radius of the euler bend.
        angle: Angle change of the euler bend arc.

    Returns:
        A function mapping 0 to 1 to the linestring of the bend path.

    """
    sign = np.sign(angle)
    angle = np.abs(angle / 180 * np.pi)

    def _bend(t: np.ndarray):
        z = np.sqrt(angle * t)
        y, x = fresnel(z / np.sqrt(np.pi / 2))
        path = radius * np.hstack((x, y * sign))
        derivative = radius * angle * np.hstack((np.cos(angle * t), np.sin(angle * t) * sign))
        return path, derivative

    return _bend


def circular_bend_fn(radius: float, angle: float = 90):
    """Circular bend functional

    Args:
        radius: Radius of the circular bend.
        angle: Angle change of the circular arc.

    Returns:
        A function mapping 0 to 1 to the linestring of the bend path.

    """
    sign = np.sign(angle)
    angle = np.abs(angle / 180 * np.pi)

    def _bend(t: np.ndarray):
        x = radius * np.sin(angle * t)
        y = radius * (1 - np.cos(angle * t))
        return np.hstack((x, y * sign)), radius * np.hstack((np.cos(angle * t), np.sin(angle * t) * sign))

    return _bend


def spiral_fn(turns: int, scale: float = 5, separation_scale: float = 1):
    """Spiral functional.

    Args:
        turns: Number of 180 degree turns in the spiral function
        scale: The scale of the spiral function (maps to minimum radius in final implementation relative to scale 1).
        separation_scale: The separation scale for the spiral function (how fast to spiral out relative to scale 1)

    Returns:
        A function mapping 0 to 1 to the linestring of the spiral path.

    """
    def _spiral(t: np.ndarray):
        theta = t * turns * np.pi + 2 * np.pi
        radius = (theta - 2 * np.pi) * separation_scale / scale + 2 * np.pi
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        return scale / np.pi * np.hstack((x, y)), scale / np.pi * np.hstack((y, -x))

    return _spiral

Width = Union[float, List[float], Callable]
Curve = Union[float, Callable]


@fix_dataclass_init_docs
@dataclass
class Straight(Box):
    extent: Float2 = (1, 1)
    decimal_places: int = 3

    def __post_init_post_parse__(self):
        super(Straight, self).__post_init_post_parse__()
        self.port = {
            'a0': self.port['w'],
            'b0': self.port['e']
        }


@fix_dataclass_init_docs
@dataclass
class Taper(Pattern):
    width: Width
    length: float
    num_evaluations: int = 99

    def __post_init_post_parse__(self):
        pattern = parametric(self.length, self.width, num_evaluations=self.num_evaluations)
        super(Taper, self).__init__(pattern)
        self.port = pattern.port


@fix_dataclass_init_docs
@dataclass
class Turn(Pattern):
    width: Width
    radius: float
    angle: float
    euler: float = 0
    num_evaluations: int = 99

    def __post_init_post_parse__(self):
        self.circular = parametric(circular_bend_fn(self.radius, self.angle * (1 - self.euler)), width=self.width,
                                   num_evaluations=self.num_evaluations)
        if self.euler > 0:
            self.euler_start = parametric(euler_bend_fn(self.radius, self.angle * self.euler / 2), width=self.width,
                                          num_evaluations=self.num_evaluations)
            self.euler_end = parametric(euler_bend_fn(self.radius, self.angle * self.euler / 2), width=self.width,
                                        num_evaluations=self.num_evaluations)
            patterns = [self.euler_start, self.circular.put(self.euler_start.port['b0'])]
            patterns.append(self.euler_end.reflect().put(self.circular.port['b0'], from_port='b0'))
            port = {
                'a0': self.euler_start.port['a0'],
                'b0': self.euler_end.port['a0']
            }
        else:
            patterns = [self.circular]
            port = self.circular.port
        super(Turn, self).__init__(*patterns)
        self.port = port


Arc = Turn


@fix_dataclass_init_docs
@dataclass
class Spiral(Pattern):
    width: Width
    n_turns: int
    min_radius: float = 5
    separation: float = 1
    num_evaluations: int = 1000

    def __post_init_post_parse__(self):
        spiral = parametric(path=spiral_fn(self.n_turns, self.min_radius, separation_scale=self.separation),
                            width=self.width, num_evaluations=self.num_evaluations)
        bend = parametric(path=circular_bend_fn(self.min_radius, 180), width=self.width)
        patterns = [spiral.copy.rotate(180), spiral, bend.copy.rotate(90), bend.rotate(-90)]
        super(Spiral, self).__init__(*patterns)
        self.port['a0'] = spiral.port['b0']
        self.port['b0'] = Port(-self.port['a0'].x, self.port['a0'].y, -self.port['a0'].a, self.port['a0'].w)
        self.put(self.port['a0'])


@fix_dataclass_init_docs
@dataclass
class BezierSBend(Pattern):
    width: Width
    dx: float
    dy: float
    num_evaluations: int = 99

    def __post_init_post_parse__(self):
        bend = parametric(path=bezier_sbend_fn(self.dx, self.dy), width=self.width,
                          num_evaluations=self.num_evaluations)
        super(BezierSBend, self).__init__(bend)
        self.port = bend.port


@fix_dataclass_init_docs
@dataclass
class TurnSBend(Pattern):
    width: Width
    angle: float
    min_radius: float
    euler: float = 0
    num_evaluations: float = 99

    def __post_init_post_parse__(self):
        turn_up = Turn(self.width, self.min_radius, self.angle, self.euler, self.num_evaluations)
        turn_down = Turn(self.width, self.min_radius, -self.angle, self.euler, self.num_evaluations)
        segment = Path([turn_up, turn_down])
        super(TurnSBend, self).__init__(segment)
        self.port['a0'] = segment.port['a0']
        self.port['b0'] = segment.port['b0']


Segment = Union[Straight, Taper, Turn, Spiral, BezierSBend, TurnSBend, "Path"]


@fix_dataclass_init_docs
@dataclass
class Path(Pattern):
    """A parametric path for routing in photonic or electronic designs.

    A waveguide is a patterned slab of core material that is capable of guiding light. Here, we define a general
    waveguide class which can be tapered or optionally consist of recursively defined waveguides, i.e. via
    :code:`subtract_waveguide`, that enable the definition of complex adiabatic coupling and perturbation strategies.
    This enables the :code:`Waveguide` class to be used in a variety of contexts including multimode interference.

    Attributes:
        segments: Length or curve operation along the path.
        subtract: A path to subtract from the current path. This is recursively defined, allowing
            for the definition of highly complex waveguiding structures.
    """
    segments: Union[Segment, List[Segment]] = Field(default_factory=list)
    subtract: Optional["Path"] = None
    decimal_places: int = 5

    def __post_init_post_parse__(self):
        self.pathway = []
        self.polygons = []
        self.port = {
            'a0': Port(a=180),
            'b0': Port()
        }  # dummy ports
        # add segments if they exist already
        for segment in self.segments:
            self.add(segment)
        path = Pattern(*self.pathway)
        if self.subtract is not None:
            if self.subtract.size[1] > self.size[1]:
                raise ValueError(f'Require the y extent of this waveguide to be greater than that of '
                                 f'`subtract_waveguide`, but got {self.size[1]} < {self.subtract.size[1]}')
            path = Pattern(self.shapely_union() - self.subtract.shapely_union())
        super(Path, self).__init__(path)
        if self.pathway:
            self.port['a0'] = self.pathway[0].port['a0']
            self.port['b0'] = self.pathway[-1].port['b0']
        else:
            self.port = {
                'a0': Port(a=180),
                'b0': Port()
            }  # dummy ports

    @property
    def wg_path(self) -> Pattern:
        return self

    def add(self, segment: Segment, back: bool = False):
        port_name = 'a0' if back else 'b0'
        if back:  # swap input and output ports
            segment.port = {
                'a0': segment.port['b0'],
                'b0': segment.port['a0']
            }
        self.pathway.append(segment.put(self.port[port_name], from_port='a0'))
        self.polygons += self.pathway[-1].polygons
        self.port[port_name] = self.pathway[-1].port[port_name]
        return self

    def symmetrize(self):
        flipped = self.copy
        flipped.port = {
            'a0': flipped.port['b0'],
            'b0': flipped.port['a0']
        }
        self.add(flipped)
        return self

    def straight(self, extent: Float2, back: bool = False):
        return self.add(Straight(extent), back)

    def taper(self, width: Width, length: float, back: bool = False):
        return self.add(Taper(width, length), back)

    def turn(self, width: Width, radius: float, angle: float, euler: float = 0, back: bool = False):
        return self.add(Turn(width, radius, angle, euler), back)

    def bezier_sbend(self, width: Width, dx: float, dy: float, back: bool = False):
        return self.add(BezierSBend(width, dx, dy), back)

    def turn_sbend(self, width: Width, angle: float, min_radius: float, euler: float = 0, back: bool = False):
        return self.add(TurnSBend(width, angle, min_radius, euler), back)

    def bezier_dc(self, width: Width, dx: float, dy: float, interaction_l: float):
        return self.bezier_sbend(width, dx, dy).straight((interaction_l, self.port['b0'].w)).bezier_sbend(width, dx, -dy)


Path.__pydantic_model__.update_forward_refs()


def straight(extent: Float2):
    return Path().straight(extent)


def taper(width: Width, length: float):
    return Path().taper(width, length)


def taper_waveguide(width: Width, length: float, taper_length: float, symmetric: bool = True, taper_first: bool = True):
    if 2 * taper_length > length:
        raise ValueError(f"Require 2 * taper_length <= length, but got {2 * taper_length} >= {length}.")
    init_w = width[0]
    if taper_first:
        path = Path().taper(width, taper_length).straight((length / 2 - taper_length, np.sum(width)))
    else:
        path = Path().straight((length / 2 - taper_length, init_w)).taper(width, taper_length)
    return path.symmetrize() if symmetric else path


def turn(width: Width, radius: float, angle: float, euler: float = 0):
    return Path().turn(width, radius, angle, euler)


def bezier_sbend(width: Width, dx: float, dy: float):
    return Path().bezier_sbend(width, dx, dy)


def turn_sbend(width: Width, angle: float, min_radius: float, euler: float = 0):
    return Path().turn_sbend(width, angle, min_radius, euler)


def bezier_dc(width: Width, dx: float, dy: float, interaction_l: float):
    return Path().turn_sbend(width, dx, dy, interaction_l)
