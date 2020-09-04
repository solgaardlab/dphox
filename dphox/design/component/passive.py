from ...typing import *
from .pattern import Pattern, Path, get_cubic_taper, GroupedPattern

from copy import deepcopy as copy
from shapely.ops import polygonize
from shapely.geometry import MultiPolygon
from .multilayer import Multilayer

try:
    import plotly.graph_objects as go
except ImportError:
    pass


class Box(Pattern):
    def __init__(self, box_dim: Dim2, shift: Dim2 = (0, 0)):
        self.box_dim = box_dim
        super(Box, self).__init__(Path(box_dim[1]).segment(box_dim[0]).translate(dx=0, dy=box_dim[1] / 2), shift=shift)


class DC(Pattern):
    def __init__(self, bend_dim: Dim2, waveguide_w: float, gap_w: float, interaction_l: float,
                 coupler_boundary_taper_ls: Tuple[float, ...] = (0,),
                 coupler_boundary_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 end_bend_dim: Optional[Dim3] = None, use_radius: bool = False, shift: Dim2 = (0, 0)):
        """Directional coupler

        Args:
            bend_dim: if use_radius is True (bend_radius, bend_height), else (bend_width, bend_height)
            waveguide_w: waveguide width
            gap_w: gap between the waveguides
            interaction_l: interaction length
            coupler_boundary_taper_ls: coupler boundary tapers length
            coupler_boundary_taper: coupler boundary taper params
            end_l: end length before and after the bends
            end_bend_dim: If specified, places an additional end bend (see DC)
            use_radius: use radius to define bends
            shift:
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
                                       taper_ls=coupler_boundary_taper_ls).center_align(current_dc)
            center_wg = Box((interaction_l, waveguide_w)).center_align(current_dc.center)
            dc_interaction = GroupedPattern(copy(center_wg).translate(dy=-gap_w / 2 - waveguide_w / 2),
                                            copy(center_wg).translate(dy=gap_w / 2 + waveguide_w / 2))
            cuts = dc_interaction.pattern - outer_boundary.pattern
            # hacky way to make sure polygons are completely separated
            dc_without_interaction = current_dc.pattern - Box((dc_interaction.size[0],
                                                               dc_interaction.size[1] * 2)).center_align(current_dc).pattern
            paths = [dc_without_interaction, dc_interaction.pattern - cuts]
        else:
            paths = lower_path, upper_path
        super(DC, self).__init__(*paths, shift=shift)
        self.lower_path, self.upper_path = Pattern(lower_path), Pattern(upper_path)

    @property
    def input_ports(self) -> np.ndarray:
        interport_distance = self.waveguide_w + 2 * self.bend_dim[1] + self.gap_w
        if self.end_bend_dim:
            interport_distance += 2 * self.end_bend_dim[1]
        return np.asarray(((0, 0), (0, interport_distance))) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.size[0], 0))

    @property
    def interaction_points(self) -> np.ndarray:
        bl = np.asarray(self.center) - np.asarray((self.interaction_l, self.waveguide_w + self.gap_w)) / 2
        tl = bl + np.asarray((0, self.waveguide_w + self.gap_w))
        br = bl + np.asarray((self.interaction_l, 0))
        tr = tl + np.asarray((self.interaction_l, 0))
        return np.vstack((bl, tl, br, tr))


class MZI(Pattern):
    def __init__(self, bend_dim: Dim2, waveguide_w: float, arm_l: float, gap_w: float,
                 interaction_l: float, end_l: float = 0, end_bend_dim: Optional[Dim3] = None, use_radius: bool = False,
                 shift: Dim2 = (0, 0)):
        self.end_l = end_l
        self.arm_l = arm_l
        self.bend_dim = bend_dim
        self.waveguide_w = waveguide_w
        self.interaction_l = interaction_l
        self.gap_w = gap_w
        self.end_bend_dim = end_bend_dim
        self.use_radius = use_radius

        lower_path = Path(waveguide_w).mzi(bend_dim, interaction_l, arm_l, end_l,
                                           end_bend_dim=end_bend_dim, use_radius=use_radius)
        upper_path = Path(waveguide_w).mzi(bend_dim, interaction_l, arm_l, end_l,
                                           end_bend_dim=end_bend_dim, inverted=True, use_radius=use_radius)
        upper_path.translate(dx=0, dy=waveguide_w + 2 * bend_dim[1] + gap_w)

        super(MZI, self).__init__(lower_path, upper_path, shift=shift)
        self.lower_path, self.upper_path = Pattern(lower_path), Pattern(upper_path)

    @property
    def input_ports(self) -> np.ndarray:
        interport_distance = self.waveguide_w + 2 * self.bend_dim[1] + self.gap_w
        if self.end_bend_dim:
            interport_distance += 2 * self.end_bend_dim[1]
        return np.asarray(((0, 0), (0, interport_distance))) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.size[0], 0))

    @property
    def interaction_points(self) -> np.ndarray:
        input_ports = self.input_ports
        bl = input_ports[0] + np.asarray(self.bend_dim) + np.asarray(self.end_bend_dim) + np.asarray((self.end_l, 0))
        tl = bl + np.asarray((self.waveguide_w + self.gap_w, 0))
        br = bl + np.asarray((self.interaction_l, 0))
        tr = tl + np.asarray((self.interaction_l, 0))
        left_dc_pts = np.vstack((bl, tl, br, tr))
        right_dc_pts = left_dc_pts + np.asarray((self.arm_l + self.bend_dim[0], 0))
        return np.vstack((left_dc_pts, right_dc_pts))


class MMI(Pattern):
    def __init__(self, box_dim: Dim2, waveguide_w: float, interport_distance: float,
                 taper_dim: Dim2, end_l: float = 0, bend_dim: Optional[Tuple[float, float]] = None,
                 use_radius: bool = False, shift: Dim2 = (0, 0)):
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
            p_01.sbend(bend_dim, inverted=True, use_radius=use_radius).segment(
                taper_dim[0], final_width=taper_dim[1])
        else:
            center = (end_l + taper_dim[0] + box_dim[0] / 2, interport_distance / 2)
            p_00 = Path(waveguide_w).segment(end_l) if end_l > 0 else Path(waveguide_w)
            p_00.segment(taper_dim[0], final_width=taper_dim[1])
            p_01 = copy(p_00).translate(dx=0, dy=interport_distance)
        mmi_start = (center[0] - box_dim[0] / 2, center[1])
        mmi = Path(box_dim[1], mmi_start).segment(box_dim[0])
        p_10 = copy(p_01).rotate(np.pi, center)
        p_11 = copy(p_00).rotate(np.pi, center)

        super(MMI, self).__init__(mmi, p_00, p_01, p_10, p_11, shift=shift)

    @property
    def input_ports(self) -> np.ndarray:
        bend_y = 2 * self.bend_dim[1] if self.bend_dim else 0
        return np.asarray(((0, 0), (0, self.interport_distance + bend_y))) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.size[0], 0))


class GratingPad(Pattern):
    def __init__(self, pad_dim: Dim2, taper_l: float, final_width: float, out: bool = False,
                 end_l: Optional[float] = None, bend_dim: Optional[Dim2] = None, shift: Dim2 = (0, 0),
                 layer: int = 0):
        self.pad_dim = pad_dim
        self.taper_l = taper_l
        self.final_width = final_width
        self.out = out
        self.bend_dim = bend_dim
        self.end_l = taper_l if end_l is None else end_l

        if out:
            path = Path(final_width)
            if end_l > 0:
                path.segment(end_l)
            if bend_dim:
                path.sbend(bend_dim)
            super(GratingPad, self).__init__(
                path.segment(taper_l, final_width=pad_dim[1]).segment(pad_dim[0]), shift=shift)
        else:
            path = Path(pad_dim[1]).segment(pad_dim[0]).segment(taper_l, final_width=final_width)
            if bend_dim:
                path.sbend(bend_dim, layer=layer)
            if end_l > 0:
                path.segment(end_l, layer=layer)
            super(GratingPad, self).__init__(path, shift=shift)

    def to(self, port: Dim2):
        if self.out:
            return self.translate(port[0], port[1])
        else:
            bend_y = self.bend_dim[1] if self.bend_dim else 0
            return self.translate(port[0] - self.size[0], port[1] - bend_y)

    @property
    def copy(self) -> "GratingPad":
        return copy(self)


class Interposer(Pattern):
    def __init__(self, waveguide_w: float, n: int, period: float, radius: float,
                 trombone_radius: Optional[float] = None,
                 final_period: Optional[float] = None, self_coupling_extension_dim: Optional[Dim2] = None,
                 horiz_dist: float = 0, num_trombones: int = 1, shift: Dim2 = (0, 0)):
        """

        Args:
            waveguide_w: waveguide width
            n: number of I/O for interposer
            period: initial period entering the interposer
            radius: radius of bends for the interposer
            trombone_radius: trombone bend radius
            final_period: final period for the interposer
            self_coupling_extension_dim: self coupling for alignment
            horiz_dist: additional horizontal distance (usually to make room for wirebonds)
            num_trombones: number of trombones
            shift: translate this component in xy
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
            num_trombones = int(
                np.ceil(max_length_diff / 2 / (final_period - 3 * radius))) if not num_trombones else num_trombones
            length_diff = (angled_length - x_length) * idx if idx < mid else (angled_length - x_length) * (n - 1 - idx)
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
            for _ in range(num_trombones):
                path.trombone(length_diff / 2 / num_trombones, radius=trombone_radius)
            paths.append(path)
            init_pos[idx] = np.asarray((0, period * idx))
            final_pos[idx] = np.asarray((path.x, path.y))

        if self_coupling_extension_dim is not None:
            dx, dy = final_pos[0, 0], final_pos[0, 1]
            radius, grating_length = self_coupling_extension_dim
            self_coupling_path = Path(width=waveguide_w).rotate(-np.pi).translate(dx=dx, dy=dy - final_period)
            self_coupling_path.turn(radius, -np.pi, tolerance=0.001)
            self_coupling_path.segment(length=grating_length + 5)
            self_coupling_path.turn(radius=radius, angle=np.pi / 2, tolerance=0.001)
            self_coupling_path.segment(length=final_period * (n + 1) - 6 * radius)
            self_coupling_path.turn(radius=radius, angle=np.pi / 2, tolerance=0.001)
            self_coupling_path.segment(length=grating_length + 5)
            self_coupling_path.turn(radius=radius, angle=-np.pi, tolerance=0.001)
            paths.append(self_coupling_path)

        super(Interposer, self).__init__(*paths, call_union=False, shift=shift)
        self.self_coupling_path = None if self_coupling_extension_dim is None else paths[-1]
        self.paths = paths
        self.init_pos = init_pos
        self.final_pos = final_pos

    @property
    def input_ports(self) -> np.ndarray:
        return self.init_pos + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.final_pos + self.shift


class Waveguide(Pattern):
    def __init__(self, waveguide_w: float, length: float, taper_ls: Tuple[float, ...] = None,
                 taper_params: Tuple[Tuple[float, ...]] = None,
                 slot_dim: Optional[Dim2] = None, slot_taper_ls: Tuple[float, ...] = 0,
                 slot_taper_params: Tuple[Tuple[float, ...]] = None,
                 num_taper_evaluations: int = 100, shift: Dim2 = (0, 0),
                 symmetric: bool = True, rotate_angle = None):

        """Waveguide class
        Args:
            waveguide_w: waveguide width at the input of the waveguide path
            length: total length of the waveguide
            taper_ls: a tuple of lengths for tapers starting from the left

            symmetric: a temporary toggling variable to turn off the symmetric nature of the waveguide class.

            .
            .
            .
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
                if taper_l > 0:
                    p.polynomial_taper(taper_l, taper_param, num_taper_evaluations)
        if symmetric:
            if not length >= 2 * np.sum(taper_ls):
                raise ValueError(
                    f'Require interaction_l >= 2 * np.sum(taper_ls) but got {length} < {2 * np.sum(taper_ls)}')
            if taper_params is not None:
                p.segment(length - 2 * np.sum(taper_ls))
                for taper_l, taper_param in zip(reversed(taper_ls), reversed(taper_params)):
                    if taper_l > 0:
                        p.polynomial_taper(taper_l, taper_param, num_taper_evaluations, inverted=True)
            else:
                p.segment(length)
        else:
            if not length >= np.sum(taper_ls):
                raise ValueError(f'Require interaction_l >= np.sum(taper_ls) but got {length} < {np.sum(taper_ls)}')
            p.segment(length - np.sum(taper_ls))

        if rotate_angle is not None:
            p.rotate(rotate_angle)

        if slot_taper_params:
            center_x = length / 2
            slot = self.__init__(slot_dim[1], slot_dim[0], slot_taper_ls, slot_taper_params).center_align((center_x, 0))
            pattern = Pattern(p).pattern - slot.pattern
            if isinstance(pattern, MultiPolygon):
                slot_waveguide = [Pattern(poly) for poly in pattern]
                super(Waveguide, self).__init__(*slot_waveguide, shift=shift)
            else:
                super(Waveguide, self).__init__(pattern, shift=shift)
        else:
            super(Waveguide, self).__init__(p, shift=shift)

    @property
    def input_ports(self) -> np.ndarray:
        return np.asarray(((0, 0),)) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.size[0], 0))


class AlignmentMark(Pattern):
	def __init__(self, waveguide_w: float, length: float, shift: Dim2 = (0, 0), layer: int = 0):
		self.length = length
		self.waveguide_w = waveguide_w
		p = Path(waveguide_w, (0,0)).segment(length, "+y", layer=layer)
		q = Path(waveguide_w, (-length/2, length/2)).segment(length, "+x", layer=layer)
		super(AlignmentMark, self).__init__(p,q, shift=shift)

