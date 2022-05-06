from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..device import Device
from ..foundry import AIR, CommonLayer, SILICON
from ..parametric import cubic_taper, cubic_taper_fn, dc_path, grating_arc, link, loopback, straight, trombone, turn
from ..pattern import Box, Pattern, Port
from ..typing import Float2, Int2
from ..utils import fix_dataclass_init_docs


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
        radius: The bend radius of the directional coupler
        interport_distance: Distance between the ports of the directional coupler.
        gap_w: Gap between the waveguides in the interaction region.
        interaction_l: Interaction length for the interaction region.
        euler: The euler parameter for the directional coupler bend.
        end_l: End length for the coupler (before and after the bends).
        coupler_waveguide_w: Coupling waveguide width
    """

    waveguide_w: float
    radius: float
    interport_distance: float
    gap_w: float
    interaction_l: float
    euler: float = 0.2
    end_l: float = 0
    coupler_waveguide_w: Optional[float] = None

    def __post_init__(self):
        self.coupler_waveguide_w = self.waveguide_w if self.coupler_waveguide_w is None else self.coupler_waveguide_w
        cw = self.coupler_waveguide_w
        w = self.waveguide_w
        radius, dy = (self.radius, (self.interport_distance - self.gap_w - self.coupler_waveguide_w) / 2)
        width = (w, cubic_taper_fn(w, cw), cw, cubic_taper_fn(cw, w), w) if cw != w else w
        lower_path = link(self.end_l, dc_path(radius, dy, self.interaction_l, self.euler), self.end_l).path(width)
        upper_path = link(self.end_l, dc_path(radius, -dy, self.interaction_l, self.euler), self.end_l).path(width)
        upper_path.translate(dx=0, dy=self.interport_distance)
        super(DC, self).__init__(lower_path, upper_path)
        self.lower_path, self.upper_path = lower_path, upper_path
        self.port['a0'] = Port(0, 0, -180, w=self.waveguide_w)
        self.port['a1'] = Port(0, self.interport_distance, -180, w=self.waveguide_w)
        self.port['b0'] = Port(self.size[0], 0, w=self.waveguide_w)
        self.port['b1'] = Port(self.size[0], self.interport_distance, w=self.waveguide_w)
        self.lower_path.port = {'a0': self.port['a0'].copy, 'b0': self.port['b0'].copy}
        self.upper_path.port = {'a0': self.port['a1'].copy, 'b0': self.port['b1'].copy}
        self.refs.extend([self.lower_path, upper_path])

    @property
    def interaction_points(self) -> np.ndarray:
        bl = np.asarray(self.center) - np.asarray((self.interaction_l, self.waveguide_w + self.gap_w)) / 2
        tl = bl + np.asarray((0, self.waveguide_w + self.gap_w))
        br = bl + np.asarray((self.interaction_l, 0))
        tr = tl + np.asarray((self.interaction_l, 0))
        return np.vstack((bl, tl, br, tr))

    @property
    def path_array(self):
        return np.array([self.polygons[:3], self.polygons[3:]])

    def device(self, layer: str = CommonLayer.RIDGE_SI):
        device = Device('dc', [(self, layer)])
        device.port = self.port_copy
        device.lower_path = self.lower_path
        device.upper_path = self.upper_path
        return device


@fix_dataclass_init_docs
@dataclass
class Cross(Pattern):
    """Cross

    Attributes:
        waveguide: waveguide to form the crossing (used to implement tapering)
    """

    waveguide: Pattern

    def __post_init__(self):
        horizontal = self.waveguide
        vertical = self.waveguide.copy.rotate(90, self.waveguide.center)
        super().__init__(horizontal, vertical)
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
        unit: The pattern to repeat in the array
        grid_shape: Number of rows and columns
        pitch: The distance between the circles in the Hole array

    """
    unit: Pattern
    grid_shape: Int2
    pitch: Optional[Union[float, Float2]] = None

    def __post_init__(self):
        self.pitch = np.array(self.unit.size) * 2 if self.pitch is None else self.pitch
        self.pitch = (self.pitch, self.pitch) if np.isscalar(self.pitch) else self.pitch
        super().__init__([self.unit.copy.translate(i * self.pitch[0], j * self.pitch[1])
                          for i in range(self.grid_shape[0]) for j in range(self.grid_shape[1])
                          ])


@fix_dataclass_init_docs
@dataclass
class Escalator(Device):
    """Escalator device implemented using tapers facing each other

    Attributes:
        bottom_waveguide: The bottom waveguide of escalator represented as a :code:`Waveguide` object
            to allow features such as tapering and coupling.
        top_waveguide: The top waveguide of escalator represented as a :code:`Waveguide` object
            to allow features such as tapering and coupling.
        bottom: Bottom layer of escalator.
        top: Top layer of escalator.
        name: The device name for the escalator.
    """
    bottom_waveguide: Pattern
    top_waveguide: Optional[Pattern] = None
    bottom: str = CommonLayer.RIDGE_SI
    top: str = CommonLayer.RIDGE_SIN
    name: str = "escalator"

    def __post_init__(self):
        pattern_to_layer = [(self.bottom_waveguide, self.bottom)]
        pattern_to_layer += [(self.top_waveguide, self.top)] if self.top_waveguide is not None else []
        super().__init__(self.name, pattern_to_layer)
        self.port = {'a0': self.bottom_waveguide.port['a0'].copy, 'b0': self.bottom_waveguide.port['b0'].copy}


@fix_dataclass_init_docs
@dataclass
class RibDevice(Device):
    """Waveguide cross section allowing specification of ridge and slab waveguides.

    Attributes:
        ridge_waveguide: The ridge waveguide (the thick section of the rib), represented as a :code:`Waveguide` object
            to allow features such as tapering and coupling. Generally this should be smaller than
            :code:`slab_waveguide`. The port of this device is defined using the port of the ridge waveguide.
        slab_waveguide: The slab waveguide (the thin section of the rib), represented as a :code:`Waveguide` object
            to allow features such as tapering and coupling. Generally this should be larger than
            :code:`ridge_waveguide`. If not specified, this merely implements a waveguide pattern.
        ridge: Ridge layer.
        slab: Slab layer.
        name: The device name.
    """
    ridge_waveguide: Pattern
    slab_waveguide: Optional[Pattern] = None
    ridge: str = CommonLayer.RIDGE_SI
    slab: str = CommonLayer.RIB_SI
    name: str = "rib_wg"

    def __post_init__(self):
        pattern_to_layer = [(self.ridge_waveguide, self.ridge)]
        pattern_to_layer += [(self.slab_waveguide, self.slab)] if self.slab_waveguide is not None else []
        super().__init__(self.name, pattern_to_layer)
        self.port = {'a0': self.ridge_waveguide.port['a0'].copy, 'b0': self.ridge_waveguide.port['b0'].copy}


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
        num_periods: The number of periods (uses maximum given extent and pitch if not specified).
        name: Name of the device.
        ridge: The ridge layer for the partial etch.
        slab: The slab layer for the partial etch.

    """
    extent: Float2
    waveguide: Pattern
    pitch: float
    duty_cycle: float = 0.5
    rib_grow: float = 0
    num_periods: Optional[int] = None
    name: str = 'straight_grating'
    ridge: CommonLayer = CommonLayer.RIDGE_SI
    slab: CommonLayer = CommonLayer.RIB_SI

    def __post_init__(self):
        self.stripe_w = self.pitch * (1 - self.duty_cycle)
        slab = (Box(self.extent).hstack(self.waveguide).buffer(self.rib_grow), self.slab)
        grating = (Box(self.extent).hstack(self.waveguide).striped(self.stripe_w, (self.pitch, 0)), self.ridge)
        super().__init__(self.name, [slab, grating, (self.waveguide, self.ridge)])
        self.port['a0'] = self.waveguide.port['a0'].copy


@fix_dataclass_init_docs
@dataclass
class FocusingGrating(Device):
    """Focusing grating with partial etch.

    Attributes:
        angle: The opening angle for the focusing grating.
        waveguide_w: The waveguide width.
        wavelength: wavelength accepted by the grating.
        duty_cycle: duty cycle for the grating
        n_clad: clad material index of refraction (assume oxide by default).
        n_core: core material index of refraction (assume silicon by default).
        fiber_angle: angle of the fiber in degrees from horizontal (not NOT vertical).
        num_periods: number of grating periods
        resolution: Number of evaluations for the curve.
        grating_frac: The fraction of the distance radiating from the center occupied by the grating (otherwise ridge).
        rib_grow: Offset the rib / slab layer in size (usually positive).
        waveguide_extra_l: The extra length of the waveguide
        name: Name of the device.
        ridge: The ridge layer for the partial etch.
        slab: The slab layer for the partial etch.

    """
    waveguide_w: float = 0.5
    angle: float = 22.5
    n_env: int = AIR.n
    n_core: int = SILICON.n
    min_period: int = 40
    num_periods: int = 30
    wavelength: float = 1.55
    fiber_angle: float = 82
    duty_cycle: float = 0.5
    grating_frac: float = 1
    resolution: int = 99
    rib_grow: float = 1
    waveguide_extra_l: float = 0
    name: str = 'focusing_grating'
    ridge: CommonLayer = CommonLayer.RIDGE_SI
    slab: CommonLayer = CommonLayer.RIB_SI

    def __post_init__(self):
        grating_arcs = [grating_arc(self.angle, self.duty_cycle, self.n_core, self.n_env,
                                    self.fiber_angle, self.wavelength, m, resolution=self.resolution)
                        for m in range(self.min_period, self.min_period + self.num_periods)]
        sector = Pattern(np.hstack((np.zeros((2, 1)), grating_arcs[0].curve.geoms[0])))
        grating = Pattern(grating_arcs, sector)
        min_waveguide_l = np.abs(self.waveguide_w / np.tan(np.radians(self.angle)))
        self.waveguide = RibDevice(straight(self.waveguide_extra_l + min_waveguide_l).path(self.waveguide_w),
                                   slab=self.slab, ridge=self.ridge)
        self.waveguide.halign(min_waveguide_l, left=False)
        super().__init__(self.name,
                         [(grating.buffer(self.rib_grow), self.slab),
                          (grating, self.ridge), self.waveguide])
        self.port['a0'] = self.waveguide.port['a0'].copy
        self.translate(*(-self.port['a0'].xy))  # put the a0 port at 0, 0


@fix_dataclass_init_docs
@dataclass
class TapDC(Pattern):
    """Tap directional coupler

    Attributes:
        dc: the directional coupler that acts as a tap coupler
        grating_pad: the grating pad for the tap
        turn_radius: The turn radius for the tap DC
    """
    dc: DC
    radius: float
    angle: float = 90
    euler: float = 0
    ridge: str = CommonLayer.RIDGE_SI
    name: str = 'tap'

    def __post_init__(self):
        tap_turns = [turn(self.radius, -self.angle, self.euler).path(self.dc.waveguide_w).to(self.dc.port['a1']),
                     turn(self.radius, self.angle, self.euler).path(self.dc.waveguide_w).to(self.dc.port['b1'])]
        super().__init__(self.dc, tap_turns)
        self.port['a0'] = self.dc.port['a0']
        self.port['b0'] = self.dc.port['b0']
        self.port['a1'] = tap_turns[0].port['b0']
        self.port['b1'] = tap_turns[1].port['b0']
        self.wg_path = self.dc.lower_path

    def with_gratings(self, grating: Union[StraightGrating, FocusingGrating]):
        device = Device('tap_dc_with_grating', [(self, grating.ridge)])
        device.port = self.port_copy
        device.place(grating, self.port['a1'])
        device.place(grating, self.port['b1'])
        return device


@fix_dataclass_init_docs
@dataclass
class Interposer(Pattern):
    """Pitch-changing array of waveguides with path length correction.

    Attributes:
        waveguide_w: The waveguide width.
        n: The number of I/O (waveguides) for interposer.
        init_pitch: The initial pitch (distance between successive waveguides) entering the interposer.
        radius: The radius of bends for the interposer.
        trombone_radius: The trombone bend radius for path equalization.
        final_pitch: The final pitch (distance between successive waveguides) for the interposer.
        self_coupling_extension_extent: The self coupling for alignment, which is useful since a major use case of
            the interposer is for fiber array coupling.
        additional_x: The additional horizontal distance (useful in fiber array coupling for wirebond clearance).
        num_trombones: The number of trombones for path equalization.
        trombone_at_end: Whether to use a path-equalizing trombone near the waveguides spaced at :code:`final_period`.
    """
    waveguide_w: float
    n: int
    init_pitch: float
    final_pitch: float
    radius: Optional[float] = None
    euler: float = 0
    trombone_radius: float = 5
    self_coupling_final: bool = True
    self_coupling_init: bool = False
    self_coupling_radius: float = None
    self_coupling_extension: float = 0
    additional_x: float = 0
    num_trombones: int = 1
    trombone_at_end: bool = True

    def __post_init__(self):
        w = self.waveguide_w
        pitch_diff = self.final_pitch - self.init_pitch
        self.radius = np.abs(pitch_diff) / 2 if self.radius is None else self.radius
        r = self.radius

        paths = []
        init_pos = np.zeros((2, self.n))
        init_pos[1] = self.init_pitch * np.arange(self.n)
        init_pos = init_pos.T
        final_pos = np.zeros_like(init_pos)

        if np.abs(1 - np.abs(pitch_diff) / 4 / r) > 1:
            raise ValueError(f"Radius {r} needs to be at least abs(pitch_diff) / 2 = {np.abs(pitch_diff) / 2}.")
        angle_r = np.sign(pitch_diff) * np.arccos(1 - np.abs(pitch_diff) / 4 / r)
        angled_length = np.abs(pitch_diff / np.sin(angle_r))
        x_length = np.abs(pitch_diff / np.tan(angle_r))
        mid = int(np.ceil(self.n / 2))
        angle = float(np.degrees(angle_r))

        for idx in range(self.n):
            init_pos[idx] = np.asarray((0, self.init_pitch * idx))
            length_diff = (angled_length - x_length) * idx if idx < mid else (angled_length - x_length) * (self.n - 1 - idx)
            segments = []
            trombone_section = [trombone(self.trombone_radius,
                                         length_diff / 2 / self.num_trombones, self.euler)] * self.num_trombones
            if not self.trombone_at_end:
                segments += trombone_section
            segments.append(self.additional_x)
            if idx < mid:
                segments += [turn(r, -angle, self.euler), angled_length * (mid - idx - 1),
                             turn(r, angle, self.euler), x_length * (idx + 1)]
            else:
                segments += [turn(r, angle, self.euler), angled_length * (mid - self.n + idx),
                             turn(r, -angle, self.euler), x_length * (self.n - idx)]
            if self.trombone_at_end:
                segments += trombone_section
            paths.append(link(*segments).path(w).to(init_pos[idx]))
            final_pos[idx] = paths[-1].port['b0'].xy

        if self.self_coupling_final:
            scr = self.final_pitch / 4 if self.self_coupling_radius is None else self.self_coupling_radius
            dx, dy = final_pos[0, 0], final_pos[0, 1]
            p = self.final_pitch
            port = Port(dx, dy - p, -180)
            extension = (self.self_coupling_extension, p * (self.n + 1) - 6 * scr)
            paths.append(loopback(scr, self.euler, extension).path(w).to(port))
        if self.self_coupling_init:
            scr = self.init_pitch / 4 if self.self_coupling_radius is None else self.self_coupling_radius
            dx, dy = init_pos[0, 0], init_pos[0, 1]
            p = self.init_pitch
            port = Port(dx, dy - p)
            extension = (self.self_coupling_extension, p * (self.n + 1) - 6 * scr)
            paths.append(loopback(scr, self.euler, extension).path(w).to(port))

        port = {**{f'a{idx}': Port(*init_pos[idx], -180, w=self.waveguide_w) for idx in range(self.n)},
                **{f'b{idx}': Port(*final_pos[idx], w=self.waveguide_w) for idx in range(self.n)},
                'l0': paths[-1].port['a0'], 'l1': paths[-1].port['b0']}

        super().__init__(*paths)
        self.self_coupling_path = None if self.self_coupling_extension is None else paths[-1]
        self.paths = paths
        self.init_pos = init_pos
        self.final_pos = final_pos
        self.port = port

    def device(self, layer: str = CommonLayer.RIDGE_SI):
        return Device('interposer', [(self, layer)]).set_port(self.port_copy)

    def with_gratings(self, grating: FocusingGrating, layer: str = CommonLayer.RIDGE_SI):
        interposer = self.device(layer)
        interposer.port = self.port
        for idx in range(6):
            interposer.place(grating, self.port[f'b{idx}'], from_port=grating.port['a0'])
        interposer.place(grating, self.port['l0'], from_port=grating.port['a0'])
        interposer.place(grating, self.port['l1'], from_port=grating.port['a0'])
        return interposer

@fix_dataclass_init_docs
@dataclass
class ArrayWaveguideGrating(Pattern):
    """

    """
    pass


@fix_dataclass_init_docs
@dataclass
class TSplitter(Pattern):
    """Pitch-changing array of waveguides with path length correction.

    Args:
        waveguide_w: The waveguide width.
        splitter_l: The splitter length (ignoring the turns).
        radius: The radius.
        splitter_mmi_w: Splitter MMI width (use twice the waveguide width by default).
        input_l: The input extension length.
        output_l: The output extension length.
    """
    waveguide_w: float
    splitter_l: float
    radius: float
    splitter_mmi_w: Optional[float] = None
    input_l: float = 0
    output_l: float = 0

    def __post_init__(self):
        self.splitter_mmi_w = 2 * self.waveguide_w if self.splitter_mmi_w is None else self.splitter_mmi_w
        splitter_taper = cubic_taper(init_w=self.waveguide_w,
                                     change_w=self.splitter_mmi_w - self.waveguide_w,
                                     length=self.input_l + self.splitter_l,
                                     taper_length=self.splitter_l,
                                     symmetric=False,
                                     taper_first=False)
        turn_shift = self.splitter_mmi_w / 2 - self.waveguide_w / 2
        upturn = link(turn(self.radius), self.output_l).path(self.waveguide_w)
        downturn = link(turn(self.radius, -90), self.output_l).path(self.waveguide_w)
        super(TSplitter, self).__init__(splitter_taper,
                                        upturn.to(splitter_taper.port['b0']).translate(dy=turn_shift),
                                        downturn.to(splitter_taper.port['b0']).translate(dy=-turn_shift))
        self.port = {'a0': splitter_taper.port['a0'], 'b0': upturn.port['b0'], 'b1': downturn.port['b0']}
