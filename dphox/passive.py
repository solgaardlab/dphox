import numpy as np
from dataclasses import dataclass
from shapely.geometry import LinearRing, MultiPolygon, Point

from .device import Device
from .foundry import CommonLayer
from .parametric import bezier_dc
from .pattern import Box, Ellipse, Pattern, Port, Sector
from .typing import Float2, Int2, Optional, Union
from .utils import fix_dataclass_init_docs

try:
    import plotly.graph_objects as go
except ImportError:
    pass


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
        coupler_waveguide_w: Coupling waveguide width
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
    use_radius: bool = False

    def __post_init__(self):
        self.coupler_waveguide_w = self.waveguide_w if self.coupler_waveguide_w is None else self.coupler_waveguide_w
        dx, dy, w = (self.bend_l, (self.interport_distance - self.gap_w - self.coupler_waveguide_w) / 2,
                     self.coupler_waveguide_w)
        lower_path = bezier_dc(dx, dy, self.interaction_l).path(self.waveguide_w)
        upper_path = bezier_dc(dx, -dy, self.interaction_l).path(self.waveguide_w)
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


@fix_dataclass_init_docs
@dataclass
class Cross(Pattern):
    """Cross

    Attributes:
        waveguide: waveguide to form the crossing (used to implement tapering)
    """

    waveguide: Pattern

    def __post_init__(self):
        horizontal = self.waveguide.align()
        vertical = self.waveguide.align().copy.rotate(90)
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
        unit: The box or ellipse to repeat in the array
        grid_shape: Number of rows and columns
        pitch: The distance between the circles in the Hole array
        n_points: The number of points in the circle (it can save time to use fewer points).

    """
    unit: Union[Box, Ellipse]
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
    ridge_waveguide: Pattern
    slab_waveguide: Pattern
    ridge: str = CommonLayer.RIDGE_SI
    slab: str = CommonLayer.RIB_SI
    name: str = "rib_wg"

    def __post_init__(self):
        super().__init__(self.name, [(self.ridge_waveguide, self.ridge),
                                     (self.slab_waveguide, self.slab)])
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
    waveguide: Pattern
    pitch: float
    duty_cycle: float = 0.5
    rib_grow: float = 0
    n_teeth: Optional[int] = None
    name: str = 'straight_grating'
    ridge: CommonLayer = CommonLayer.RIDGE_SI
    slab: CommonLayer = CommonLayer.RIB_SI

    def __post_init__(self):
        self.stripe_w = self.pitch * (1 - self.duty_cycle)
        slab = (Box(self.extent).hstack(self.waveguide).buffer(self.rib_grow), self.slab)
        grating = (Box(self.extent).hstack(self.waveguide).striped(self.stripe_w, (self.pitch, 0)), self.ridge)
        super().__init__(self.name, [slab, grating, (self.waveguide, self.ridge)])
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
    waveguide: WaveguideDevice
    pitch: float
    duty_cycle: float = 0.5
    grating_frac: float = 1
    resolution: int = 16
    rib_grow: float = 0
    name: str = 'focusing_grating'
    ridge: CommonLayer = CommonLayer.RIDGE_SI
    slab: CommonLayer = CommonLayer.RIB_SI

    def __post_init__(self):
        self.stripe_w = self.pitch * (1 - self.duty_cycle)
        grating = Sector(self.radius, self.angle, self.resolution).shapely
        n_periods = int(self.radius * self.grating_frac / self.pitch) - 1
        for n in range(n_periods):
            circle = LinearRing(Point(0, 0).buffer(
                self.radius - self.pitch * (n + 1) + self.stripe_w / 2).exterior.coords).buffer(self.stripe_w / 2)
            grating -= circle
        grating = Pattern(grating).rotate(90).translate(
            np.abs(self.waveguide.port['b0'].w / np.tan(self.angle / 360 * np.pi))
        )
        self.waveguide.slab = self.slab
        self.waveguide.ridge = self.ridge
        if n_periods <= 0:
            raise ValueError(f'Calculated {n_periods} which is <= 0.'
                             f'Make sure that the pitch is not too big and grating_frac not'
                             f'too small compared to radius.')
        super().__init__(self.name,
                         [(grating.buffer(self.rib_grow + self.pitch), self.slab),
                          (grating, self.ridge), self.waveguide])
        self.port['a0'] = self.waveguide.port['b0'].copy
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

    def __post_init__(self):
        in_grating = self.grating_pad.copy.put(self.dc.port['b1'])
        out_grating = self.grating_pad.copy.put(self.dc.port['a1'])
        super().__init__(self.dc, in_grating, out_grating)
        self.port['a0'] = self.dc.port['a0']
        self.port['b0'] = self.dc.port['b0']
        self.refs.append(self.dc)
        self.wg_path = self.dc.lower_path
