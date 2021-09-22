from copy import deepcopy

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass
from shapely.geometry import MultiPolygon, LinearRing, Point
from shapely.ops import unary_union

from .device import Device
from .foundry import CommonLayer
from .pattern import AnnotatedPath, Box, Ellipse, GdspyPath, Pattern, Port, TaperSpec, Sector
from .typing import Float2, Int2, List, Optional, Union
from .utils import fix_dataclass_init_docs

try:
    import plotly.graph_objects as go
except ImportError:
    pass


@fix_dataclass_init_docs
@dataclass
class Waveguide(Pattern):
    """Waveguide, the core photonic structure for guiding light.

    A waveguide is a patterned slab of core material that is capable of guiding light. Here, we define a general
    waveguide class which can be tapered or optionally consist of recursively defined waveguides, i.e. via
    :code:`subtract_waveguide`, that enable the definition of complex adiabatic coupling and perturbation strategies.
    This enables the :code:`Waveguide` class to be used in a variety of contexts including multimode interference.

    Attributes:
        extent: Tuple of waveguide width, length at the input of the waveguide path. Note that the taper
            can extend to be wider than the original extent width.
        taper: A :code:`TaperSpec` or list of :code:`TaperSpec`'s that sequentially taper the waveguide width
            according to a :code:`Path.polynomial_taper` specification.
        subtract_waveguide: A waveguide to subtract from the current waveguide. This is recursively defined, allowing
            for the definition of highly complex waveguiding structures.
        symmetric: Whether to symmetrically apply the taper params. For example, if a :code:`Waveguide`
            were to be tapered by specifying :code:`taper_params` and `taper_ls`, the :code:`symmetric == True`
            case would result in an inverted taper back to the original waveguide width. This is common in
            certain applications such as crossings and phase shifters, thus it is set to :code:`True` by default,
            though there are some devices such as waveguide terminations and escalators where such behavior is
            not required.
    """
    extent: Float2
    taper: Union[TaperSpec, List[TaperSpec]] = Field(default_factory=list)
    subtract_waveguide: Optional["Waveguide"] = None
    symmetric: bool = True

    def __post_init_post_parse__(self):
        waveguide = GdspyPath(self.extent[0])
        self.taper = [self.taper] if isinstance(self.taper, TaperSpec) \
            else self.taper
        taper_l = np.sum([t.length for t in self.taper])
        for taper in self.taper:
            waveguide.polynomial_taper(taper, taper.num_evaluations)
        if self.symmetric:
            if self.extent[1] < 2 * taper_l:
                raise ValueError(
                    f'Require waveguide_extent[1] >= 2 * taper_l but got {self.extent[1]} <= {2 * taper_l}')
            if self.extent[1] != 2 * taper_l:
                waveguide.segment(self.extent[1] - 2 * taper_l)
            for taper in reversed(self.taper):
                waveguide.polynomial_taper(taper, taper.num_evaluations, inverted=True)
        else:
            if not self.extent[1] >= taper_l:
                raise ValueError(f'Require length >= taper_l but got {self.extent[1]} < {taper_l}')
            waveguide.segment(self.extent[1] - taper_l)
        self.final_width = waveguide.w
        waveguide = Pattern(waveguide)
        if self.subtract_waveguide is not None:
            if self.subtract_waveguide.size[1] >= waveguide.size[1]:
                raise ValueError(f'Require the y extent of this waveguide to be greater than that of '
                                 f'`subtract_waveguide`, but got {waveguide.size[1]} <= {self.subtract_waveguide.size[1]}')
            waveguide = Pattern(waveguide.shapely_union() - self.subtract_waveguide.shapely_union())
        super(Waveguide, self).__init__(waveguide)
        self.port['a0'] = Port(0, 0, -180, w=self.extent[0])
        self.port['b0'] = Port(self.size[0], 0, w=self.final_width)

    @property
    def wg_path(self) -> Pattern:
        return self


Waveguide.__pydantic_model__.update_forward_refs()


@fix_dataclass_init_docs
@dataclass
class DC(Pattern):
    """Directional coupler

    A directional coupler is a `symmetric` component (across x and y dimensions) that contains two waveguides that
    interact and couple light by adiabatically bending the waveguides towards each other, interacting over some
    interaction length :code:`interaction_l`, and then adiabatically bending out to the original interport distance.
    An MMI can actually be created if the gap_w is set to be negative.

    Attributes:
        waveguide_w: Waveguide width at the inputs and outputs.
        bend_extent: If use_radius is True (bend_radius, bend_height), else (bend_width, bend_height).
            If a third dimension is specified then the width of the waveguide portion in the interaction
            region is specified, which may be useful for broadband coupling.
        gap_w: Gap between the waveguides in the interaction region.
        interaction_l: Interaction length for the interaction region.
        coupling_waveguide: An optional coupling waveguide to replace
        end_bend_extent: If specified, places an additional end bend
        use_radius: use radius to define bends
    """

    waveguide_w: float
    bend_l: float
    interport_distance: float
    gap_w: float
    interaction_l: float
    end_l: float = 0
    coupler_waveguide_w: Optional[float] = None
    coupling_waveguide: Optional[Waveguide] = None
    use_radius: bool = False

    def __post_init_post_parse__(self):
        self.coupler_waveguide_w = self.waveguide_w if self.coupler_waveguide_w is None else self.coupler_waveguide_w
        bend_extent = (self.bend_l, (self.interport_distance - self.gap_w - self.coupler_waveguide_w) / 2,
                       self.coupler_waveguide_w)

        lower_path = AnnotatedPath(self.waveguide_w).dc(bend_extent=bend_extent, interaction_l=self.interaction_l,
                                                        end_l=self.end_l, use_radius=self.use_radius)
        upper_path = AnnotatedPath(self.waveguide_w).dc(bend_extent=bend_extent, interaction_l=self.interaction_l,
                                                        end_l=self.end_l, inverted=True, use_radius=self.use_radius)
        upper_path.translate(dx=0, dy=self.interport_distance)
        super(DC, self).__init__(lower_path, upper_path)
        if self.coupling_waveguide is not None:
            self.replace(self.coupling_waveguide)
        self.lower_path, self.upper_path = lower_path, upper_path
        self.port['a0'] = Port(0, 0, -180, w=self.waveguide_w)
        self.port['a1'] = Port(0, self.interport_distance, -180, w=self.waveguide_w)
        self.port['b0'] = Port(self.size[0], 0, w=self.waveguide_w)
        self.port['b1'] = Port(self.size[0], self.interport_distance, w=self.waveguide_w)
        self.lower_path.port = {'a0': self.port['a0'], 'b0': self.port['b0']}
        self.lower_path.wg_path = self.lower_path
        self.upper_path.port = {'a0': self.port['a1'], 'b0': self.port['b1']}
        self.upper_path.wg_path = self.upper_path
        self.reference_patterns.extend([self.lower_path, upper_path])

    @property
    def interaction_points(self) -> np.ndarray:
        bl = np.asarray(self.center) - np.asarray((self.interaction_l, self.waveguide_w + self.gap_w)) / 2
        tl = bl + np.asarray((0, self.waveguide_w + self.gap_w))
        br = bl + np.asarray((self.interaction_l, 0))
        tr = tl + np.asarray((self.interaction_l, 0))
        return np.vstack((bl, tl, br, tr))

    @property
    def path_array(self):
        return np.array([self.polys[:3], self.polys[3:]])


@fix_dataclass_init_docs
@dataclass
class Interposer(Pattern):
    """Pitch-changing array of waveguides with path length correction.

    Args:
        waveguide_w: The waveguide waveguide width.
        n: The number of I/O (waveguides) for interposer.
        init_pitch: The initial pitch (distance between successive waveguides) entering the interposer.
        radius: The radius of bends for the interposer.
        trombone_radius: The trombone bend radius for path equalization.
        final_pitch: The final pitch (distance between successive waveguides) for the interposer.
        self_coupling_extension_extent: The self coupling for alignment, which is useful since a major use case of
            the interposer is for fiber array coupling.
        horiz_dist: The additional horizontal distance.
        num_trombones: The number of trombones for path equalization.
        trombone_at_end: Whether to use a path-equalizing trombone near the waveguides spaced at :code:`final_period`.
    """
    waveguide_w: float
    n: int
    init_pitch: float
    radius: float
    trombone_radius: Optional[float] = None
    final_pitch: Optional[float] = None
    self_coupling_extension_extent: Optional[Float2] = None
    self_coupling_final: bool = True
    horiz_dist: float = 0
    num_trombones: int = 1
    trombone_at_end: bool = True

    def __init__(self):
        self.trombone_radius = self.radius if self.trombone_radius is None else self.trombone_radius
        self.final_pitch = self.init_pitch if self.final_pitch is None else self.final_pitch
        pitch_diff = self.final_pitch - self.init_pitch
        paths = []
        init_pos = np.zeros((self.n, 2))
        final_pos = np.zeros_like(init_pos)
        for idx in range(self.n):
            radius = pitch_diff / 2 if not self.radius else self.radius
            angle_r = np.sign(pitch_diff) * np.arccos(1 - np.abs(pitch_diff) / 4 / radius)
            angled_length = np.abs(pitch_diff / np.sin(angle_r))
            x_length = np.abs(pitch_diff / np.tan(angle_r))
            angle = angle_r
            path = GdspyPath(self.waveguide_w).segment(length=0).translate(dx=0, dy=self.init_pitch * idx)
            mid = int(np.ceil(self.n / 2))
            max_length_diff = (angled_length - x_length) * (mid - 1)
            self.num_trombones = int(np.ceil(max_length_diff / 2 / (self.final_pitch - 3 * radius))) \
                if not self.num_trombones else self.num_trombones
            length_diff = (angled_length - x_length) * idx if idx < mid else (angled_length - x_length) * (
                    self.n - 1 - idx)
            if not self.trombone_at_end:
                for _ in range(self.num_trombones):
                    path.trombone(length_diff / 2 / self.num_trombones, radius=self.trombone_radius)
            path.segment(self.horiz_dist)
            if idx < mid:
                path.turn(radius, -angle)
                path.segment(angled_length * (mid - idx - 1))
                path.turn(radius, angle)
                path.segment(x_length * (idx + 1))
            else:
                path.turn(radius, angle)
                path.segment(angled_length * (mid - self.n + idx))
                path.turn(radius, -angle)
                path.segment(x_length * (self.n - idx))
            if self.trombone_at_end:
                for _ in range(self.num_trombones):
                    path.trombone(length_diff / 2 / self.num_trombones, radius=self.trombone_radius)
            paths.append(path)
            init_pos[idx] = np.asarray((0, self.init_pitch * idx))
            final_pos[idx] = np.asarray((path.x, path.y))

        if self.self_coupling_extension_extent is not None:
            if self.self_coupling_final:
                dx, dy = final_pos[0, 0], final_pos[0, 1]
                p = self.final_pitch
                s = 1
            else:
                dx, dy = init_pos[0, 0], init_pos[0, 1]
                p = self.init_pitch
                s = -1
            radius, grating_length = self.self_coupling_extension_extent
            self_coupling_path = GdspyPath(width=self.waveguide_w).rotate(-np.pi * self.self_coupling_final).translate(
                dx=dx,
                dy=dy - p)
            self_coupling_path.turn(radius, -np.pi * s, tolerance=0.001)
            self_coupling_path.segment(length=grating_length + 5)
            self_coupling_path.turn(radius=radius, angle=np.pi / 2 * s, tolerance=0.001)
            self_coupling_path.segment(length=p * (self.n + 1) - 6 * radius)
            self_coupling_path.turn(radius=radius, angle=np.pi / 2 * s, tolerance=0.001)
            self_coupling_path.segment(length=grating_length + 5)
            self_coupling_path.turn(radius=radius, angle=-np.pi * s, tolerance=0.001)
            paths.append(self_coupling_path)

        super(Interposer, self).__init__(*paths)
        self.self_coupling_path = None if self.self_coupling_extension_extent is None else paths[-1]
        self.paths = paths
        self.init_pos = init_pos
        self.final_pos = final_pos
        for idx in range(self.n):
            self.port[f'a{idx}'] = Port(*init_pos[idx], -180, w=self.waveguide_w)
            self.port[f'b{idx}'] = Port(*final_pos[idx], w=self.waveguide_w)


@fix_dataclass_init_docs
@dataclass
class Cross(Pattern):
    """Cross

    Attributes:
        waveguide: waveguide to form the crossing (used to implement tapering)
    """

    waveguide: Waveguide

    def __post_init_post_parse__(self):
        horizontal = self.waveguide.align()
        vertical = self.waveguide.align().copy.rotate(90)
        super(Cross, self).__init__(horizontal, vertical)
        self.port['a0'] = horizontal.port['a0']
        self.port['a1'] = vertical.port['a0']
        self.port['b0'] = horizontal.port['b0']
        self.port['b1'] = vertical.port['b0']


@fix_dataclass_init_docs
@dataclass
class Array(Pattern):
    """Array of boxes or ellipses for 2D photonic crystals.

    This class can generate large circle arrays which may be used for photonic crystal designs or for slow
    light applications.

    Attributes:
        unit: The box or ellipse to repeat in the array
        grid_shape: Number of rows and columns
        pitch: The distance between the circles in the Hole array
        n_points: The number of points in the circle (it can save time to use fewer points).

    """
    unit: Union[Box, Ellipse]
    grid_shape: Int2
    # orientation: Union[float, np.ndarray]
    pitch: Optional[Union[float, Float2]] = None

    def __post_init_post_parse__(self):
        self.pitch = (self.pitch, self.pitch) if isinstance(self.pitch, float) else self.pitch
        super(Array, self).__init__(MultiPolygon([self.unit.copy.translate(i * self.pitch, j * self.pitch)
                                                  for i in range(self.grid_shape[0])
                                                  for j in range(self.grid_shape[1])
                                                  ]))


@fix_dataclass_init_docs
@dataclass
class DelayLine(Pattern):
    """Delay line for unbalanced MZIs.

    Attributes:
        waveguide_w: the waveguide width
        delay_length: the delay line length increase over the straight length
        bend_radius: the bend radius of turns in the squiggle delay line
        straight_length: the comparative straight segment this matches
        number_bend_pairs: the number of bend pairs
        flip: whether to flip the usual direction of the delay line
    """
    waveguide_w: float
    delay_length: float
    bend_radius: float
    straight_length: float
    number_bend_pairs: int = 1
    flip: bool = False

    def __post_init_post_parse__(self):

        if ((2 * np.pi + 4) * self.number_bend_pairs + np.pi - 4) * self.bend_radius >= self.delay_length:
            raise ValueError(
                f"Bends alone exceed the delay length {self.delay_length}"
                f"reduce the bend radius or the number of bend pairs")
        segment_length = (self.delay_length - (
                (2 * np.pi + 4) * self.number_bend_pairs + np.pi - 4) * self.bend_radius) / (
                                 2 * self.number_bend_pairs)
        extra_length = self.straight_length - 4 * self.bend_radius - segment_length
        if extra_length <= 0:
            raise ValueError(
                f"The delay line does not fit in the horizontal distance of"
                f"{self.straight_length} increase the number of bend pairs")
        height = (4 * self.number_bend_pairs - 2) * self.bend_radius
        p = GdspyPath(self.waveguide_w)
        p.segment(length=self.bend_radius)
        p.segment(length=segment_length)

        bend_dir = -1 if self.flip else 1

        for count in range(self.number_bend_pairs):
            p.turn(radius=self.bend_radius, angle=np.pi * bend_dir)
            p.segment(length=segment_length)
            p.turn(radius=self.bend_radius, angle=-np.pi * bend_dir)
            p.segment(length=segment_length)
        p.segment(length=self.bend_radius)
        p.turn(radius=self.bend_radius, angle=-np.pi / 2 * bend_dir)
        p.segment(length=height)
        p.turn(radius=self.bend_radius, angle=np.pi / 2 * bend_dir)
        p.segment(length=extra_length)

        super(DelayLine, self).__init__(p)
        self.port['a0'] = Port(0, 0, -180, w=self.waveguide_w)
        self.port['b0'] = Port(self.bounds[2], 0, w=self.waveguide_w)


@fix_dataclass_init_docs
@dataclass
class WaveguideDevice(Device):
    """Rib waveguide

    Attributes:
        ridge_waveguide: The ridge waveguide (the thick section of the rib), represented as a :code:`Waveguide` object
            to allow features such as tapering and coupling. Generally this should be smaller than
            :code:`slab_waveguide`.
        slab_waveguide: The slab waveguide (the thin section of the rib), represented as a :code:`Waveguide` object
            to allow features such as tapering and coupling. Generally this should be larger than
            :code:`ridge_waveguide`.
        name: The device name.
    """
    ridge_waveguide: Waveguide
    slab_waveguide: Waveguide
    ridge: str = CommonLayer.RIDGE_SI
    rib: str = CommonLayer.RIB_SI
    name: str = "rib_wg"

    def __post_init_post_parse__(self):
        super(WaveguideDevice, self).__init__(self.name, [(self.ridge_waveguide, self.ridge),
                                                          (self.slab_waveguide, self.rib)])
        self.port = self.ridge_waveguide.port


@fix_dataclass_init_docs
@dataclass
class StraightGrating(Device):
    """Straight (non-focusing) grating with partial etch.

    Attributes:
        extent: Dimension of the extent of the grating.
        waveguide: The waveguide to connect to the grating structure (can be tapered if desired)
        pitch: The pitch between the grating teeth.
        duty_cycle: The fill factor for the grating.
        rib_grow: Offset the rib / slab layer in size (usually positive).
        n_teeth: The number of teeth (uses maximum given extent and pitch if not specified).
        name: Name of the device.
        ridge: The ridge layer for the partial etch.
        slab: The slab layer for the partial etch.

    """
    extent: Float2
    waveguide: Waveguide
    pitch: float
    duty_cycle: float = 0.5
    rib_grow: float = 0
    n_teeth: Optional[int] = None
    name: str = 'straight_grating'
    ridge: CommonLayer = CommonLayer.RIDGE_SI
    slab: CommonLayer = CommonLayer.RIB_SI

    def __post_init_post_parse__(self):
        self.stripe_w = self.pitch * (1 - self.duty_cycle)
        slab = (Box(self.extent).hstack(self.waveguide).offset(self.rib_grow), self.slab)
        grating = (Box(self.extent).hstack(self.waveguide).striped(self.stripe_w, (self.pitch, 0)), self.ridge)
        super(StraightGrating, self).__init__(self.name, [slab, grating, (self.waveguide, self.ridge)])
        self.port['a0'] = self.waveguide.port['a0']


@fix_dataclass_init_docs
@dataclass
class FocusingGrating(Device):
    """Focusing grating with partial etch.

    Attributes:
        radius: The radius of the focusing grating structure
        angle: The opening angle for the focusing grating.
        waveguide: The waveguide for the focusing grating.
        pitch: The pitch between the grating teeth.
        duty_cycle: The fill factor for the grating.
        grating_frac: The fraction of the distance radiating from the center occupied by the grating (otherwise ridge).
        resolution: Number of evaluations for the arcs.
        rib_grow: Offset the rib / slab layer in size (usually positive).
        name: Name of the device.
        ridge: The ridge layer for the partial etch.
        slab: The slab layer for the partial etch.

    """
    radius: float
    angle: float
    waveguide: Waveguide
    pitch: float
    duty_cycle: float = 0.5
    grating_frac: float = 1
    resolution: int = 16
    rib_grow: float = 0
    name: str = 'focusing_grating'
    ridge: CommonLayer = CommonLayer.RIDGE_SI
    slab: CommonLayer = CommonLayer.RIB_SI

    def __post_init_post_parse__(self):
        self.stripe_w = self.pitch * (1 - self.duty_cycle)
        grating = Sector(self.radius, self.angle, self.resolution).shapely
        n_periods = int(self.radius * self.grating_frac / self.pitch) - 1
        for n in range(n_periods):
            circle = LinearRing(Point(0, 0).buffer(
                self.radius - self.pitch * (n + 1) + self.stripe_w / 2).exterior.coords).buffer(self.stripe_w / 2)
            grating -= circle
        grating = Pattern(grating).rotate(90).translate(
            np.abs(self.waveguide.final_width / np.tan(self.angle / 360 * np.pi))
        )
        if n_periods <= 0:
            raise ValueError(f'Calculated {n_periods} which is <= 0.'
                             f'Make sure that the pitch is not too big and grating_frac not'
                             f'too small compared to radius.')
        super(FocusingGrating, self).__init__(self.name,
                                              [(Box(grating.size).offset(self.rib_grow).align(grating), self.slab),
                                               (grating, self.ridge),
                                               (self.waveguide, self.ridge)])
        self.port['a0'] = self.waveguide.port['b0']
        self.rotate(180).halign(0)


@fix_dataclass_init_docs
@dataclass
class TapDC(Pattern):
    """Tap directional coupler

    Attributes:
        dc: the directional coupler that acts as a tap coupler
        grating_pad: the grating pad for the tap
    """
    dc: DC
    grating: Union[StraightGrating, FocusingGrating]

    def __post_init_post_parse__(self):
        in_grating = self.grating_pad.copy.to(self.dc.port['b1'])
        out_grating = self.grating_pad.copy.to(self.dc.port['a1'])
        super(TapDC, self).__init__(self.dc, in_grating, out_grating)
        self.port['a0'] = self.dc.port['a0']
        self.port['b0'] = self.dc.port['b0']
        self.reference_patterns.append(self.dc)
        self.wg_path = self.dc.lower_path
