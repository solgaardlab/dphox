from ..typing import *
from .pattern import Pattern, Path, Port

from copy import deepcopy as copy
from shapely.geometry import MultiPolygon, box, MultiPoint
import numpy as np
import nazca as nd

from ..utils import circle

try:
    import plotly.graph_objects as go
except ImportError:
    pass


class Box(Pattern):
    """Box with default center at origin

    Args:
        box_dim: Box dimension (box width, box height)
    """

    def __init__(self, box_dim: Dim2, decimal_places: int = 3):
        self.box_dim = box_dim
        super(Box, self).__init__(box(-box_dim[0] / 2, -box_dim[1] / 2, box_dim[0] / 2, box_dim[1] / 2),
                                  decimal_places=decimal_places)

    @classmethod
    def bbox(cls, pattern: Pattern) -> Pattern:
        """Bounding box for pattern

        Args:
            pattern: The pattern over which to take a bounding box

        Returns:
            A bounding box pattern of the same size as :code:`pattern`

        """
        return cls(pattern.size).align(pattern)

    def expand(self, grow: float) -> Pattern:
        """An aligned box that grows by amount :code:`grow`

        Args:
            grow: The amount to grow the box

        Returns:
            The box after the grow transformation

        """
        big_box_dim = (self.box_dim[0] + grow, self.box_dim[1] + grow)
        return Box(big_box_dim).align(self)

    def hollow(self, thickness: float) -> Pattern:
        """A hollow box of thickness :code:`thickness` on all four sides.

        Args:
            thickness: Thickness of the box

        Returns:

        """
        return Pattern(
            self.difference(Box((self.box_dim[0] - 2 * thickness, self.box_dim[1])).align(self)),
            self.difference(Box((self.box_dim[0], self.box_dim[1] - 2 * thickness)).align(self)),
        )

    def u(self, thickness: float) -> Pattern:
        return Pattern(
            self.difference(Box((self.box_dim[0] - 2 * thickness, self.box_dim[1])).align(self)),
            self.difference(Box((self.box_dim[0], self.box_dim[1] - thickness)).align(self).valign(self)),
        )

    def striped(self, stripe_w: float, pitch: Optional[Dim2] = None) -> Pattern:
        pitch = (stripe_w * 2, stripe_w * 2) if pitch is None else pitch
        patterns = [self.hollow(stripe_w)] if pitch[0] > 0 and pitch[1] > 0 else []
        if pitch[0] > 0 and not 3 * pitch[1] >= self.size[0]:
            # edges of etch holes are really thick
            # TODO: make the edges lean toward large holes over small holes. currently attepting to subtract the last pitch
            xs = np.mgrid[self.bounds[0] + pitch[0]:self.bounds[2]:pitch[0]]
            patterns.append(Pattern(*[Box((stripe_w, self.size[1])).halign(x) for x in xs], call_union=False).align(self.center))
        if pitch[1] > 0 and not 3 * pitch[1] >= self.size[1]:
            ys = np.mgrid[self.bounds[1] + pitch[1]:self.bounds[3]:pitch[1]]
            patterns.append(Pattern(*[Box((self.size[0], stripe_w)).valign(y) for y in ys], call_union=False).align(self.center))
        return Pattern(*patterns, call_union=False)

    def flexure(self, spring_dim: Dim2, connector_dim: Dim2 = None, symmetric_connector: bool = True,
                stripe_w: float = 1) -> Pattern:
        spring = Box(spring_dim).align(self)
        connector = Box(connector_dim).align(self)
        connectors = []
        if symmetric_connector:
            connectors += [connector.copy.halign(self), connector.copy.halign(self, left=False)]
        else:
            connectors += [
                connector.copy.valign(self).halign(self),
                connector.copy.valign(self).halign(self, left=False)
            ]
        return Pattern(self.striped(stripe_w),
                       spring.copy.valign(self), spring.copy.valign(self, bottom=False), *connectors, call_union=False)


class DC(Pattern):
    def __init__(self, bend_dim: Dim2, waveguide_w: float, gap_w: float, interaction_l: float,
                 coupler_boundary_taper_ls: Tuple[float, ...] = (0,),
                 coupler_boundary_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 end_bend_dim: Optional[Dim3] = None, use_radius: bool = False):
        """Directional coupler

        Args:
            bend_dim: if use_radius is True (bend_radius, bend_height), else (bend_width, bend_height)
            waveguide_w: waveguide width
            gap_w: gap between the waveguides
            interaction_l: interaction length
            coupler_boundary_taper_ls: coupler boundary tapers length
            coupler_boundary_taper: coupler boundary taper params
            end_bend_dim: If specified, places an additional end bend
            use_radius: use radius to define bends
        """
        self.bend_dim = bend_dim
        self.waveguide_w = waveguide_w
        self.interaction_l = interaction_l
        self.gap_w = gap_w
        self.end_bend_dim = end_bend_dim
        self.use_radius = use_radius
        self.coupler_boundary_taper_ls = coupler_boundary_taper_ls
        self.coupler_boundary_taper = coupler_boundary_taper

        interport_distance = waveguide_w + 2 * bend_dim[1] + gap_w
        if end_bend_dim:
            interport_distance += 2 * end_bend_dim[1]

        lower_path = Path(waveguide_w).dc(bend_dim, interaction_l, end_l=0, end_bend_dim=end_bend_dim,
                                          use_radius=use_radius)
        upper_path = Path(waveguide_w).dc(bend_dim, interaction_l, end_l=0, end_bend_dim=end_bend_dim,
                                          inverted=True, use_radius=use_radius)
        upper_path.translate(dx=0, dy=interport_distance)

        if coupler_boundary_taper is not None and np.sum(coupler_boundary_taper_ls) > 0:
            current_dc = Pattern(upper_path, lower_path)
            outer_boundary = Waveguide(waveguide_w=2 * waveguide_w + gap_w, length=interaction_l,
                                       taper_params=coupler_boundary_taper,
                                       taper_ls=coupler_boundary_taper_ls).align(current_dc)
            center_wg = Box((interaction_l, waveguide_w)).align(current_dc.center)
            dc_interaction = Pattern(copy(center_wg).translate(dy=-gap_w / 2 - waveguide_w / 2),
                                     copy(center_wg).translate(dy=gap_w / 2 + waveguide_w / 2))
            cuts = dc_interaction.shapely - outer_boundary.shapely
            # hacky way to make sure polygons are completely separated
            dc_without_interaction = current_dc.shapely - Box((dc_interaction.size[0],
                                                               dc_interaction.size[1] * 2)).align(current_dc).shapely
            paths = [dc_without_interaction, dc_interaction.shapely - cuts]
        else:
            paths = lower_path, upper_path
        super(DC, self).__init__(*paths, call_union=False)
        self.lower_path, self.upper_path = Pattern(lower_path), Pattern(upper_path)
        self.port['a0'] = Port(0, 0, -np.pi)
        self.port['a1'] = Port(0, interport_distance, -np.pi)
        self.port['b0'] = Port(self.size[0], 0)
        self.port['b1'] = Port(self.size[0], interport_distance)

    @property
    def interaction_points(self) -> np.ndarray:
        bl = np.asarray(self.center) - np.asarray((self.interaction_l, self.waveguide_w + self.gap_w)) / 2
        tl = bl + np.asarray((0, self.waveguide_w + self.gap_w))
        br = bl + np.asarray((self.interaction_l, 0))
        tr = tl + np.asarray((self.interaction_l, 0))
        return np.vstack((bl, tl, br, tr))


class MMI(Pattern):
    def __init__(self, box_dim: Dim2, waveguide_w: float, interport_distance: float,
                 taper_dim: Dim2, end_l: float = 0, bend_dim: Optional[Tuple[float, float]] = None,
                 use_radius: bool = False):
        self.end_l = end_l
        self.waveguide_w = waveguide_w
        self.box_dim = box_dim
        self.interport_distance = interport_distance
        self.taper_dim = taper_dim
        self.bend_dim = bend_dim
        self.use_radius = use_radius

        if self.bend_dim:
            center = (end_l + bend_dim[0] + taper_dim[0] + box_dim[0] / 2, interport_distance / 2 + bend_dim[1])
            p_00 = Path(waveguide_w).segment(end_l) if end_l > 0 else Path(waveguide_w)
            p_00.sbend(bend_dim, use_radius=use_radius).segment(taper_dim[0], final_width=taper_dim[1])
            p_01 = Path(waveguide_w, (0, interport_distance + 2 * bend_dim[1]))
            p_01 = p_01.segment(end_l) if end_l > 0 else p_01
            p_01.sbend(bend_dim, inverted=True, use_radius=use_radius).segment(taper_dim[0], final_width=taper_dim[1])
        else:
            center = (end_l + taper_dim[0] + box_dim[0] / 2, interport_distance / 2)
            p_00 = Path(waveguide_w).segment(end_l) if end_l > 0 else Path(waveguide_w)
            p_00.segment(taper_dim[0], final_width=taper_dim[1])
            p_01 = copy(p_00).translate(dx=0, dy=interport_distance)
        mmi_start = (center[0] - box_dim[0] / 2, center[1])
        mmi = Path(box_dim[1], mmi_start).segment(box_dim[0])
        p_10 = copy(p_01).rotate(np.pi, center)
        p_11 = copy(p_00).rotate(np.pi, center)
        super(MMI, self).__init__(mmi, p_00, p_01, p_10, p_11)
        bend_y = 2 * self.bend_dim[1] if self.bend_dim else 0
        self.port['a0'] = Port(0, 0, -np.pi)
        self.port['a1'] = Port(0, self.interport_distance + bend_y, -np.pi)
        self.port['b0'] = Port(self.size[0], 0)
        self.port['b1'] = Port(self.size[0], self.interport_distance + bend_y)


class GratingPad(Pattern):
    def __init__(self, pad_dim: Dim2, taper_l: float, final_w: float, out: bool = False,
                 end_l: Optional[float] = None, bend_dim: Optional[Dim2] = None, layer: int = 0):
        self.pad_dim = pad_dim
        self.taper_l = taper_l
        self.final_w = final_w
        self.out = out
        self.bend_dim = bend_dim
        self.end_l = taper_l if end_l is None else end_l

        if out:
            path = Path(final_w)
            if end_l > 0:
                path.segment(end_l)
            if bend_dim:
                path.sbend(bend_dim)
            super(GratingPad, self).__init__(path.segment(taper_l, final_width=pad_dim[1]).segment(pad_dim[0]))
        else:
            path = Path(pad_dim[1]).segment(pad_dim[0]).segment(taper_l, final_width=final_w)
            if bend_dim:
                path.sbend(bend_dim, layer=layer)
            if end_l > 0:
                path.segment(end_l, layer=layer)
            super(GratingPad, self).__init__(path)
        self.port['a0'] = Port(0, 0)


class Interposer(Pattern):
    def __init__(self, waveguide_w: float, n: int, period: float, radius: float,
                 trombone_radius: Optional[float] = None, final_period: Optional[float] = None,
                 self_coupling_extension_dim: Optional[Dim2] = None, self_coupling_final: bool = True,
                 horiz_dist: float = 0, num_trombones: int = 1, trombone_at_end: bool = True):
        """Pitch-changing array of waveguides

        Args:
            waveguide_w: waveguide width
            n: number of I/O for interposer
            period: initial period entering the interposer
            radius: radius of bends for the interposer
            trombone_radius: trombone bend radius
            final_period: final period for the interposer
            self_coupling_extension_dim: self coupling for alignment
            horiz_dist: additional horizontal distance
            num_trombones: number of trombones
        """
        trombone_radius = radius if trombone_radius is None else trombone_radius
        final_period = period if final_period is None else final_period
        period_diff = final_period - period
        paths = []
        init_pos = np.zeros((n, 2))
        final_pos = np.zeros_like(init_pos)
        for idx in range(n):
            radius = period_diff / 2 if not radius else radius
            angle_r = np.sign(period_diff) * np.arccos(1 - np.abs(period_diff) / 4 / radius)
            angled_length = np.abs(period_diff / np.sin(angle_r))
            x_length = np.abs(period_diff / np.tan(angle_r))
            angle = angle_r
            path = Path(waveguide_w).segment(length=0).translate(dx=0, dy=period * idx)
            mid = int(np.ceil(n / 2))
            max_length_diff = (angled_length - x_length) * (mid - 1)
            num_trombones = int(np.ceil(max_length_diff / 2 / (final_period - 3 * radius))) \
                if not num_trombones else num_trombones
            length_diff = (angled_length - x_length) * idx if idx < mid else (angled_length - x_length) * (n - 1 - idx)
            if not trombone_at_end:
                for _ in range(num_trombones):
                    path.trombone(length_diff / 2 / num_trombones, radius=trombone_radius)
            path.segment(horiz_dist)
            if idx < mid:
                path.turn(radius, -angle)
                path.segment(angled_length * (mid - idx - 1))
                path.turn(radius, angle)
                path.segment(x_length * (idx + 1))
            else:
                path.turn(radius, angle)
                path.segment(angled_length * (mid - n + idx))
                path.turn(radius, -angle)
                path.segment(x_length * (n - idx))
            if trombone_at_end:
                for _ in range(num_trombones):
                    path.trombone(length_diff / 2 / num_trombones, radius=trombone_radius)
            paths.append(path)
            init_pos[idx] = np.asarray((0, period * idx))
            final_pos[idx] = np.asarray((path.x, path.y))

        if self_coupling_extension_dim is not None:
            if self_coupling_final:
                dx, dy = final_pos[0, 0], final_pos[0, 1]
                p = final_period
                s = 1
            else:
                dx, dy = init_pos[0, 0], init_pos[0, 1]
                p = period
                s = -1
            radius, grating_length = self_coupling_extension_dim
            self_coupling_path = Path(width=waveguide_w).rotate(-np.pi * self_coupling_final).translate(dx=dx, dy=dy - p)
            self_coupling_path.turn(radius, -np.pi * s, tolerance=0.001)
            self_coupling_path.segment(length=grating_length + 5)
            self_coupling_path.turn(radius=radius, angle=np.pi / 2 * s, tolerance=0.001)
            self_coupling_path.segment(length=p * (n + 1) - 6 * radius)
            self_coupling_path.turn(radius=radius, angle=np.pi / 2 * s, tolerance=0.001)
            self_coupling_path.segment(length=grating_length + 5)
            self_coupling_path.turn(radius=radius, angle=-np.pi * s, tolerance=0.001)
            paths.append(self_coupling_path)

        super(Interposer, self).__init__(*paths, call_union=False)
        self.self_coupling_path = None if self_coupling_extension_dim is None else paths[-1]
        self.paths = paths
        self.init_pos = init_pos
        self.final_pos = final_pos
        for idx in range(n):
            self.port[f'a{idx}'] = Port(*init_pos[idx], -np.pi)
            self.port[f'b{idx}'] = Port(*final_pos[idx])


class Waveguide(Pattern):
    def __init__(self, waveguide_w: float, length: float, taper_ls: Tuple[float, ...] = 0,
                 taper_params: Tuple[Tuple[float, ...], ...] = None,
                 slot_dim: Optional[Dim2] = None, slot_taper_ls: Tuple[float, ...] = 0,
                 slot_taper_params: Tuple[Tuple[float, ...]] = None,
                 num_taper_evaluations: int = 100, symmetric: bool = True, rotate_angle: float = None):
        """Waveguide class
        Args:
            waveguide_w: waveguide width at the input of the waveguide path
            length: total length of the waveguide
            taper_ls: a tuple of lengths for tapers starting from the left
            taper_params: a tuple of taper params successively :code:`Path.polynomial_taper`
            symmetric: a toggling variable to turn off the symmetric nature of the waveguide class.
            slot_dim: initial slot length and width
            slot_taper_ls: slot taper lengths (in style of waveguide class)
            slot_taper_params: slot taper parameters
            num_taper_evaluations: number of taper evaluations
            symmetric: symmetric
        """
        self.length = length
        self.waveguide_w = waveguide_w
        self.taper_ls = taper_ls
        self.taper_params = taper_params
        self.slot_dim = slot_dim
        self.slot_taper_ls = slot_taper_ls
        self.slot_taper_params = slot_taper_params

        self.pads = []

        p = Path(waveguide_w)

        if taper_params is not None:
            for taper_l, taper_param in zip(taper_ls, taper_params):
                if taper_l > 0 and taper_param is not None:
                    p.polynomial_taper(taper_l, taper_param, num_taper_evaluations)
        if symmetric:
            if not length >= 2 * np.sum(taper_ls):
                raise ValueError(
                    f'Require length >= 2 * np.sum(taper_ls) but got {length} < {2 * np.sum(taper_ls)}')
            if taper_params is not None:
                p.segment(length - 2 * np.sum(taper_ls))
                for taper_l, taper_param in zip(reversed(taper_ls), reversed(taper_params)):
                    if taper_l > 0 and taper_param is not None:
                        p.polynomial_taper(taper_l, taper_param, num_taper_evaluations, inverted=True)
            else:
                p.segment(length)
        else:
            if not length >= np.sum(taper_ls):
                raise ValueError(f'Require length >= np.sum(taper_ls) but got {length} < {np.sum(taper_ls)}')
            p.segment(length - np.sum(taper_ls))

        if rotate_angle is not None:
            p.rotate(rotate_angle)

        if slot_taper_params:
            center_x = length / 2
            slot = self.__init__(slot_dim[1], slot_dim[0], slot_taper_ls, slot_taper_params).align((center_x, 0))
            pattern = Pattern(p).shapely - slot.shapely
            if isinstance(pattern, MultiPolygon):
                slot_waveguide = [Pattern(poly) for poly in pattern]
                super(Waveguide, self).__init__(*slot_waveguide)
            else:
                super(Waveguide, self).__init__(pattern)
        else:
            super(Waveguide, self).__init__(p)
        self.port['a0'] = Port(0, 0, -np.pi)
        self.port['b0'] = Port(self.size[0], 0)


class AlignmentCross(Pattern):
    def __init__(self, line_dim: Dim2):
        """Alignment cross

        Args:
            line_dim: line dimension (length, thickness)
        """
        self.line_dim = line_dim
        p = Path(line_dim[1], (0, 0)).segment(line_dim[0], "+y")
        q = Path(line_dim[1], (-line_dim[0] / 2, line_dim[0] / 2)).segment(line_dim[0], "+x")
        super(AlignmentCross, self).__init__(p, q)


class HoleArray(Pattern):
    def __init__(self, diameter: float, grid_shape: Tuple[int, int], pitch: Optional[float] = None, n_points: int = 8):
        self.diameter = diameter
        self.pitch = pitch
        self.grid_shape = grid_shape
        super(HoleArray, self).__init__(MultiPolygon([(circle(radius=0.5 * diameter, n=n_points,
                                                              xy=(i * pitch, j * pitch)), [])
                                                      for i in range(grid_shape[0]) for j in range(grid_shape[1])
                                                      ]), call_union=False)


class DelayLine(Pattern):
    def __init__(self, waveguide_width: float, delay_length: float, bend_radius: float, straight_length: float,
                 number_bend_pairs: int = 1, flip: bool = False):
        """Delay Line
        Args:
            waveguide_width: the waveguide width
            delay_length: the delay line length increase over the straight length
            bend_radius: the bend radius of turns in the squiggle delay line
            straight_length: the comparative straight segment this matches
            flip: whether to flip the usual direction of the delay line
        """
        self.waveguide_width = waveguide_width
        self.delay_length = delay_length
        self.bend_radius = bend_radius
        self.straight_length = straight_length
        self.number_bend_pairs = number_bend_pairs
        self.flip = flip

        if ((2 * np.pi + 4) * number_bend_pairs + np.pi - 4) * bend_radius >= delay_length:
            raise ValueError(
                f"Bends alone exceed the delay length {delay_length}"
                f"reduce the bend radius or the number of bend pairs")
        segment_length = (delay_length - ((2 * np.pi + 4) * number_bend_pairs + np.pi - 4) * bend_radius) / (
            2 * number_bend_pairs)
        extra_length = straight_length - 4 * bend_radius - segment_length
        if extra_length <= 0:
            raise ValueError(
                f"The delay line does not fit in the horizontal distance of"
                f"{straight_length} increase the number of bend pairs")
        height = (4 * number_bend_pairs - 2) * bend_radius
        p = Path(waveguide_width)
        p.segment(length=bend_radius)
        p.segment(length=segment_length)

        bend_dir = -1 if flip else 1

        for count in range(number_bend_pairs):
            p.turn(radius=bend_radius, angle=np.pi * bend_dir)
            p.segment(length=segment_length)
            p.turn(radius=bend_radius, angle=-np.pi * bend_dir)
            p.segment(length=segment_length)
        p.segment(length=bend_radius)
        p.turn(radius=bend_radius, angle=-np.pi / 2 * bend_dir)
        p.segment(length=height)
        p.turn(radius=bend_radius, angle=np.pi / 2 * bend_dir)
        p.segment(length=extra_length)

        super(DelayLine, self).__init__(p)
        self.port['a0'] = Port(0, 0, -np.pi)
        self.port['b0'] = Port(self.bounds[2], 0)


class TapDC(Pattern):
    def __init__(self, dc: DC, grating_pad: GratingPad):
        in_grating, out_grating = grating_pad.copy.to(dc.port['b1']), grating_pad.copy.to(dc.port['a1'])
        super(TapDC, self).__init__(dc, in_grating, out_grating)
        self.port['a0'] = dc.port['a0']
        self.port['b0'] = dc.port['b0']
