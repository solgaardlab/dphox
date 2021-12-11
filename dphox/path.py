from pydantic import Field
from pydantic.dataclasses import dataclass
from shapely.affinity import rotate

from .pattern import Box, Pattern, Port
from .typing import Callable, Float2, List, Optional, Union
from .utils import bezier_sbend_fn, circular_bend_fn, euler_bend_fn, fix_dataclass_init_docs,\
    parametric_fn, spiral_fn

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
    decimal_places: int = 3

    def __post_init_post_parse__(self):
        path, out_port, in_port = parametric_fn(self.length, self.width)
        super(Taper, self).__init__(path)
        self.port['a0'] = Port.from_linestring(in_port)
        self.port['b0'] = Port.from_linestring(out_port)


@fix_dataclass_init_docs
@dataclass
class Turn(Pattern):
    width: Width
    radius: float
    angle: float
    euler: float = 0

    def __post_init_post_parse__(self):
        patterns = []
        path, front, back = parametric_fn(circular_bend_fn(self.radius, self.angle * (1 - self.euler)),
                                          width=self.width)
        if self.euler > 0:
            path1, connect, in_port = parametric_fn(euler_bend_fn(self.radius, self.angle * self.euler / 2),
                                                    width=self.width)
            patterns.append(Pattern(path1))
            patterns.append(Pattern(path).put(Port.from_linestring(connect), from_port=Port.from_linestring(back)))
            path3, connect, out_port = parametric_fn(euler_bend_fn(self.radius, self.angle * self.euler / 2),
                                                     width=self.width)
            patterns.append(Pattern(path3).put(Port.from_linestring(front)))
        else:
            patterns.append(Pattern(path))
            in_port, out_port = back, front

        super(Turn, self).__init__(*patterns)
        self.port['a0'] = Port.from_linestring(in_port)
        self.port['b0'] = Port.from_linestring(out_port)


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
        sp, in_port, _ = parametric_fn(path=spiral_fn(self.n_turns, self.min_radius, separation_scale=self.separation),
                                       width=self.width, num_evaluations=self.num_evaluations)
        bend, _, _ = parametric_fn(path=circular_bend_fn(self.min_radius, 180), width=self.width)
        patterns = [rotate(sp, 180, (0, 0)), sp, rotate(bend, 90, (0, 0)), rotate(bend, -90, (0, 0))]

        super(Spiral, self).__init__(*patterns)
        self.port['a0'] = Port.from_linestring(in_port)
        self.port['b0'] = Port(-self.port['a0'].x, self.port['a0'].y, -self.port['a0'].a, self.port['a0'].w)
        self.put(self.port['a0'])


@fix_dataclass_init_docs
@dataclass
class BezierSBend(Pattern):
    width: Width
    dx: float
    dy: float

    def __post_init_post_parse__(self):
        bend, out_port, in_port = parametric_fn(path=bezier_sbend_fn(self.dx, self.dy), width=self.width)
        super(BezierSBend, self).__init__(bend)
        self.port['a0'] = Port.from_linestring(in_port)
        self.port['b0'] = Port.from_linestring(out_port)


@fix_dataclass_init_docs
@dataclass
class TurnSBend(Pattern):
    width: Width
    angle: float
    min_radius: float
    euler: float = 0

    def __post_init_post_parse__(self):
        turn_up = Turn(self.width, self.min_radius, self.angle, self.euler)
        turn_down = Turn(self.width, self.min_radius, -self.angle, self.euler)
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

    def __post_init_post_parse__(self):
        self.pathway = []
        # add segments if they exist already
        for segment in self.segments:
            self.add(segment)
        path = Pattern(*self.pathway)
        if self.subtract is not None:
            if self.subtract.size[1] >= self.size[1]:
                raise ValueError(f'Require the y extent of this waveguide to be greater than that of '
                                 f'`subtract_waveguide`, but got {self.size[1]} <= {self.subtract.size[1]}')
            path = Pattern(self.shapely_union() - self.subtract.shapely_union())
        super(Path, self).__init__(path)
        self.port['a0'] = self.pathway[0].port['a0'] if self.pathway else Port(a=180)
        self.port['b0'] = self.pathway[-1].port['b0'] if self.pathway else Port()

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

    arc = turn

    def bezier_sbend(self, width: Width, dx: float, dy: float, back: bool = False):
        return self.add(BezierSBend(width, dx, dy), back)

    def turn_sbend(self, width: Width, angle: float, min_radius: float, euler: float = 0, back: bool = False):
        return self.add(TurnSBend(width, angle, min_radius, euler), back)

    def bezier_dc(self, width: Width, dx: float, dy: float, interaction_l: float):
        return self.bezier_sbend(width, dx, dy).straight((interaction_l, self.port['b0'].w)).bezier_sbend(width, dx, -dy)


Path.__pydantic_model__.update_forward_refs()
