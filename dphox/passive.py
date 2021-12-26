from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPolygon

from .device import Device
from .foundry import AIR, CommonLayer, SILICON
from .prefab import dc, grating_arc, straight, ring, turn
from .pattern import Box, Pattern, Port
from .typing import Float2, Int2, Optional, Union
from .utils import DEFAULT_RESOLUTION, fix_dataclass_init_docs


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
        bend_radius: The bend radius of the directional coupler
        gap_w: Gap between the waveguides in the interaction region.
        interaction_l: Interaction length for the interaction region.
        coupler_waveguide_w: Coupling waveguide width
        end_bend_extent: If specified, places an additional end bend
        use_radius: use radius to define bends
    """

    waveguide_w: float
    bend_radius: float
    interport_distance: float
    gap_w: float
    interaction_l: float
    euler: float = 0.2
    end_l: float = 0
    coupler_waveguide_w: Optional[float] = None
    use_radius: bool = False

    def __post_init__(self):
        self.coupler_waveguide_w = self.waveguide_w if self.coupler_waveguide_w is None else self.coupler_waveguide_w
        radius, dy, w = (self.bend_radius, (self.interport_distance - self.gap_w - self.coupler_waveguide_w) / 2,
                     self.coupler_waveguide_w)
        lower_path = dc(radius, dy, self.interaction_l, self.euler).path(self.waveguide_w)
        upper_path = dc(radius, -dy, self.interaction_l, self.euler).path(self.waveguide_w)
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
        n_points: The number of points in the circle (it can save time to use fewer points).

    """
    unit: Pattern
    grid_shape: Int2
    # orientation: Union[float, np.ndarray]
    pitch: Optional[Union[float, Float2]] = None

    def __post_init__(self):
        self.pitch = (self.pitch, self.pitch) if isinstance(self.pitch, float) else self.pitch
        super().__init__(MultiPolygon([self.unit.copy.translate(i * self.pitch, j * self.pitch)
                                       for i in range(self.grid_shape[0])
                                       for j in range(self.grid_shape[1])
                                       ]))


@fix_dataclass_init_docs
@dataclass
class WaveguideDevice(Device):
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
        self.waveguide = WaveguideDevice(straight(self.waveguide_extra_l + min_waveguide_l).path(self.waveguide_w),
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


def circle(radius: float, resolution: int = DEFAULT_RESOLUTION):
    """A circle of specified radius.

    Args:
        radius: The radius of the circle.
        resolution: Number of evaluations for each turn.

    Returns:
        The circle pattern.

    """
    return ring(radius, resolution).pattern


def ellipse(radius_x: float, radius_y: float, resolution: int = DEFAULT_RESOLUTION):
    """An ellipse of specified x and y radii.

    Args:
        radius_x: The x radius of the circle.
        radius_y: The y radius of the circle.
        resolution: Number of evaluations for each turn.

    Returns:
        The ellipse pattern.

    """
    return circle(1, resolution).scale(radius_x, radius_y).pattern
