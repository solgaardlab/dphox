from ...typing import *
from .pattern import Pattern, Path, GroupedPattern
from .passive import Waveguide, DC, Box
from .multilayer import Multilayer

from copy import deepcopy as copy

try:
    import plotly.graph_objects as go
except ImportError:
    pass


class LateralNemsPS(GroupedPattern):
    def __init__(self, waveguide_w: float, nanofin_w: float, phaseshift_l: float,
                 gap_w: float, taper_ls: Tuple[float, ...], num_taper_evaluations: int = 100,
                 pad_dim: Optional[Dim3] = None,
                 gap_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 wg_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 boundary_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 shift: Tuple[float, float] = (0, 0),
                 rib_brim_taper: Optional[Tuple[Tuple[float, ...]]] = None):
        """NEMS single-mode phase shifter
        Args:
            waveguide_w: waveguide width
            nanofin_w: nanofin width (initial, before tapering)
            phaseshift_l: phase shift length
            gap_w: gap width (initial, before tapering)
            taper_ls:  array of taper lengths
            num_taper_evaluations: number of taper evaluations (see gdspy)
            pad_dim: silicon handle xy size followed by distance between pad and fin to actuate
            gap_taper: gap taper polynomial params (recommend same as wg_taper)
            wg_taper: wg taper polynomial params (recommend same as gap_taper)
            shift: translate this component in xy
        """
        self.waveguide_w = waveguide_w
        self.nanofin_w = nanofin_w
        self.phaseshift_l = phaseshift_l
        self.gap_w = gap_w
        self.taper_ls = taper_ls
        self.num_taper_evaluations = num_taper_evaluations
        self.pad_dim = pad_dim
        self.gap_taper = gap_taper
        self.wg_taper = wg_taper
        self.boundary_taper = boundary_taper
        self.rib_brim_taper = rib_brim_taper 

        if not phaseshift_l >= 2 * np.sum(taper_ls):
            raise ValueError(
                f'Require interaction_l >= 2 * np.sum(taper_ls) but got {phaseshift_l} < {2 * np.sum(taper_ls)}')

        boundary_taper = wg_taper if boundary_taper is None else boundary_taper

        box_w = nanofin_w * 2 + gap_w * 2 + waveguide_w
        wg = Waveguide(waveguide_w, taper_ls=taper_ls, taper_params=wg_taper, length=phaseshift_l,
                    num_taper_evaluations=num_taper_evaluations)
        boundary = Waveguide(box_w, taper_params=boundary_taper, taper_ls=taper_ls, length=phaseshift_l,
                            num_taper_evaluations=num_taper_evaluations).pattern
        gap_path = Waveguide(waveguide_w + gap_w * 2, taper_params=gap_taper,
                            taper_ls=taper_ls, length=phaseshift_l,
                            num_taper_evaluations=num_taper_evaluations).pattern
        nanofins = [Pattern(poly) for poly in (boundary - gap_path)]

        if rib_brim_taper is not None:
            rib_brim = Waveguide(waveguide_w, taper_ls=taper_ls, taper_params=rib_brim_taper, length=phaseshift_l,
                        num_taper_evaluations=num_taper_evaluations)
            rib_brim = [Pattern(poly) for poly in (rib_brim.pattern - wg.pattern)]
        else:
            rib_brim = []

        pads, anchors = [], []
        if pad_dim is not None:
            pad = Box(pad_dim[:2]).center_align(wg)
            pad_y = nanofin_w + pad_dim[2] + pad_dim[1] / 2
            pads += [copy(pad).translate(dx=0, dy=-pad_y), copy(pad).translate(dx=0, dy=pad_y)]
        super(LateralNemsPS, self).__init__(*([wg] + nanofins + pads + anchors + rib_brim), shift=shift, call_union=False)
        self.waveguide, self.anchors, self.pads, self.nanofins, self.rib_brim = wg, anchors, pads, nanofins, rib_brim

    @property
    def input_ports(self) -> np.ndarray:
        return np.asarray((0, 0)) + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.input_ports + np.asarray((self.phaseshift_l, 0))

    @property
    def attachment_ports(self) -> np.ndarray:
        dy = np.asarray((0, self.nanofin_w / 2 + self.waveguide_w / 2 + self.gap_w))
        center = np.asarray(self.center)
        return np.asarray((center + dy, center - dy))


class LateralNemsTDC(GroupedPattern):
    def __init__(self, waveguide_w: float, nanofin_w: float, dc_gap_w: float, beam_gap_w: float, bend_dim: Dim2,
                 interaction_l: float, dc_taper_ls: Tuple[float, ...] = None,
                 dc_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 beam_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 end_l: float = 0, end_bend_dim: Optional[Dim3] = None,
                 pad_dim: Optional[Dim3] = None,
                 middle_fin_dim: Optional[Dim2] = None, middle_fin_pad_dim: Optional[Dim2] = None,
                 use_radius: bool = True, shift: Dim2 = (0, 0)):
        """NEMS tunable directional coupler

        Args:
            waveguide_w: waveguide width
            nanofin_w: nanofin width
            dc_gap_w: directional coupler gap width
            beam_gap_w: gap between the nanofin and the TDC waveguides
            bend_dim: see DC
            interaction_l: interaction length
            end_l: end length before and after the first and last bends
            end_bend_dim: If specified, places an additional end bend (see DC)
            pad_dim: If specified, silicon anchor/handle xy size followed by the pad gap
            middle_fin_dim: If specified, place a middle fin in the center of the coupling gap
            middle_fin_pad_dim: If specified, place an anchor pad on the left and right of the middle fin
                (ensure sufficiently far from the bends!).
            use_radius: use radius (see DC)
            shift: translate this component in xy
        """
        self.waveguide_w = waveguide_w
        self.nanofin_w = nanofin_w
        self.interaction_l = interaction_l
        self.end_l = end_l
        self.dc_gap_w = dc_gap_w
        self.beam_gap_w = beam_gap_w
        self.pad_dim = pad_dim
        self.middle_fin_dim = middle_fin_dim
        self.middle_fin_pad_dim = middle_fin_pad_dim
        self.use_radius = use_radius

        dc = DC(bend_dim=bend_dim, waveguide_w=waveguide_w, gap_w=dc_gap_w,
                coupler_boundary_taper_ls=dc_taper_ls, coupler_boundary_taper=dc_taper,
                interaction_l=interaction_l, end_bend_dim=end_bend_dim, end_l=end_l, use_radius=use_radius)
        connectors, pads, tethers = [], [], []

        nanofin_y = nanofin_w / 2 + dc_gap_w / 2 + waveguide_w + beam_gap_w
        nanofin = Box((interaction_l, nanofin_w)).center_align(dc)

        if dc_taper_ls is not None:
            if not interaction_l >= 2 * np.sum(dc_taper_ls):
                raise ValueError(
                    f'Require interaction_l > 2 * np.sum(dc_taper_ls) but got {interaction_l} < {2 * np.sum(dc_taper_ls)}')

        if beam_taper is None:
            nanofins = [copy(nanofin).translate(dx=0, dy=-nanofin_y), copy(nanofin).translate(dx=0, dy=nanofin_y)]
            if middle_fin_dim is not None:
                nanofins.append(Box(middle_fin_dim).center_align(dc))
        else:
            box_w = (nanofin_w + beam_gap_w + waveguide_w) * 2 + dc_gap_w
            gap_taper_wg_w = (beam_gap_w + waveguide_w) * 2 + dc_gap_w
            # nanofin_box = Box((interaction_l, box_w)).center_align(dc).pattern
            # gap_taper_wg = Waveguide(gap_taper_wg_w, interaction_l, dc_taper_ls, beam_taper).center_align(dc).pattern
            # nanofins = [Pattern(poly) for poly in (nanofin_box - gap_taper_wg)]

            ######### NATE: trying to taper fins of TDC ###################
            boundary = Waveguide(box_w, taper_params=beam_taper, taper_ls=dc_taper_ls,
                                 length=interaction_l).center_align(dc).pattern
            gap_path = Waveguide(gap_taper_wg_w, taper_params=beam_taper, taper_ls=dc_taper_ls,
                                 length=interaction_l).center_align(dc).pattern
            nanofins = [Pattern(poly) for poly in (boundary - gap_path)]
            ######### NATE: trying to taper center of TDC ###################

        if pad_dim is not None:
            pad = Box(pad_dim[:2]).center_align(dc)
            pad_y = nanofin_w / 2 + pad_dim[2] + pad_dim[1] / 2 + nanofin_y
            pads += [copy(pad).translate(dx=0, dy=-pad_y), copy(pad).translate(dx=0, dy=pad_y)]

        if middle_fin_pad_dim is not None:
            pad = Box(middle_fin_pad_dim).center_align(dc)
            pad_x = middle_fin_pad_dim[0] / 2 + middle_fin_dim[0] / 2
            pads += [copy(pad).translate(dx=pad_x), copy(pad).translate(dx=pad_x)]

        super(LateralNemsTDC, self).__init__(*([dc] + nanofins + connectors + pads), shift=shift, call_union=False)
        self.dc, self.connectors, self.pads, self.nanofins = dc, connectors, pads, nanofins

    @property
    def input_ports(self) -> np.ndarray:
        return self.dc.input_ports + self.shift

    @property
    def output_ports(self) -> np.ndarray:
        return self.dc.output_ports + self.shift

    @property
    def attachment_ports(self) -> np.ndarray:
        dy = np.asarray((0, self.nanofin_w / 2 + self.waveguide_w + self.dc_gap_w / 2 + self.beam_gap_w))
        center = np.asarray(self.center)
        return np.asarray((center + dy, center - dy))


class NemsAnchor(GroupedPattern):
    def __init__(self, fin_spring_dim: Dim2, connector_dim: Dim2, top_spring_dim: Dim2 = None,
                 straight_connector: Optional[Dim2] = None, loop_connector: Optional[Dim3] = None,
                 pos_electrode_dim: Optional[Dim3] = None, neg_electrode_dim: Optional[Dim2] = None,
                 include_fin_dummy: bool = False):
        """NEMS anchor

        Args:
            fin_spring_dim: fixed fin dimension (x, y)
            top_spring_dim: fin dimension (x, y)
            connector_dim: connector dimension
            straight_connector: straight connector to the fin, box xy (overridden by loop connector)
            loop_connector: loop connector to the fin, xy dim and final width on the top part of loop
            pos_electrode_dim: positive electrode dimension
            neg_electrode_dim: negative electrode dimension
            include_fin_dummy: include fin dummy for mechanical simulation
        """
        self.fin_spring_dim = fin_spring_dim
        self.top_spring_dim = top_spring_dim
        self.connector_dim = connector_dim
        self.straight_connector = straight_connector
        self.loop_connector = loop_connector
        self.pos_electrode_dim = pos_electrode_dim
        self.neg_electrode_dim = neg_electrode_dim
        patterns, c_ports, pads, doped_elems = [], [], [], []

        top_spring_dim = fin_spring_dim if not top_spring_dim else top_spring_dim
        connector = Box(connector_dim).translate()
        doped_elems.append(connector)
        shuttle = copy(connector)
        if loop_connector is not None and straight_connector is None:
            loop = Pattern(Path(fin_spring_dim[1]).rotate(np.pi).turn(
                loop_connector[1], -np.pi, final_width=loop_connector[2], tolerance=0.001).segment(
                loop_connector[0]).turn(loop_connector[1], -np.pi, final_width=fin_spring_dim[1],
                                        tolerance=0.001).segment(loop_connector[0]))
            loop.center_align(connector).vert_align(connector, bottom=False, opposite=False)
            connector = GroupedPattern(connector, loop)
        elif straight_connector is not None:
            straight = Box(straight_connector)
            connector = GroupedPattern(connector,
                                       copy(straight).horz_align(connector).vert_align(connector, bottom=False,
                                                                                       opposite=True),
                                       copy(straight).horz_align(connector, left=False,
                                                                 opposite=False).vert_align(connector,
                                                                                            bottom=False,
                                                                                            opposite=True),
            # adding more straight connectors for mirror symmetric mechanics
                                       copy(straight).horz_align(connector, left=False,
                                                                opposite=False).vert_align(
                                                                                        copy(connector).translate(connector.size[0],connector.size[1]),),
                                       copy(straight).horz_align(connector).vert_align(
                                                                                        copy(connector).translate(connector.size[0],connector.size[1]),))
                                                                                        # bottom=False,
                                                                                        # opposite=True))
            # adding more straight connectors for mirror symmetric mechanics

        a_port = (connector.center[0], connector.bounds[1])
        a_mirror_port = (connector.center[0], connector.bounds[3])
        if include_fin_dummy:
            # patterns.append(Box(fin_spring_dim).center_align(a_port)) # this dummy is at the real fin location
            patterns.append(Box(fin_spring_dim).center_align(a_mirror_port)) # this is the mirror image dummy for mechanics
        patterns.append(connector)
        if top_spring_dim is not None:
            top_spring = Box(top_spring_dim).center_align(
                connector).vert_align(connector, bottom=True, opposite=True)
            top_spring = Box(top_spring_dim).center_align(
                shuttle).vert_align(shuttle, bottom=True, opposite=True)
            bottom_spring = Box(top_spring_dim).center_align(
                shuttle).vert_align(shuttle, bottom=False, opposite=True)
            
            patterns.extend([top_spring,bottom_spring])
            doped_elems.extend([top_spring,bottom_spring])

            if pos_electrode_dim is not None:
                pos_electrode = Box((pos_electrode_dim[0], pos_electrode_dim[1])).center_align(top_spring).vert_align(
                    top_spring, opposite=True).translate(dy=pos_electrode_dim[2])
                patterns.append(pos_electrode)
                pads.append(pos_electrode)
                doped_elems.append(copy(pos_electrode))
            if neg_electrode_dim is not None:
                neg_electrode_left = Box(neg_electrode_dim).horz_align( #moving alignment to account for bottom spring
                    bottom_spring, opposite=True).vert_align(bottom_spring)
                neg_electrode_right = Box(neg_electrode_dim).horz_align( #moving alignment to account for bottom spring
                    bottom_spring, left=False, opposite=True).vert_align(bottom_spring)
                patterns.extend([neg_electrode_left, neg_electrode_right])
                pads.extend([neg_electrode_left, neg_electrode_right])
                doped_elems.append(GroupedPattern(neg_electrode_left, neg_electrode_right))

        super(NemsAnchor, self).__init__(*patterns)
        self.translate(-a_port[0], -a_port[1])
        self.pads = [pad.translate(-a_port[0], -a_port[1]) for pad in pads]
        self.dope_patterns = [doped_elem.translate(-a_port[0], -a_port[1]) for doped_elem in doped_elems]

    @property
    def contact_ports(self) -> np.ndarray:
        return np.asarray([pad.center for pad in self.pads])


class MemsMonitorCoupler(Pattern):
    def __init__(self, waveguide_w: float, interaction_l: float, gap_w: float,
                 end_l: float, detector_wg_l: float, bend_radius: float = 3, pad_dim: Optional[Dim2] = None,
                 rib_pad_w: float = 0):
        self.waveguide_w = waveguide_w
        self.interaction_l = interaction_l
        self.detector_wg_l = detector_wg_l
        self.gap_w = gap_w
        self.end_l = end_l
        self.bend_radius = bend_radius
        self.pad_dim = pad_dim

        pads = []

        waveguide = Path(width=waveguide_w).segment(interaction_l)
        monitor_wg = copy(waveguide).translate(dx=0, dy=gap_w + waveguide_w)
        monitor_left = Path(width=waveguide_w).rotate(np.pi).turn(bend_radius, -np.pi / 2).segment(detector_wg_l).turn(
            bend_radius, -np.pi / 2).translate(dx=0, dy=gap_w + waveguide_w)
        monitor_right = Path(width=waveguide_w).turn(bend_radius, np.pi / 2).segment(detector_wg_l).turn(
            bend_radius, np.pi / 2).translate(dx=interaction_l, dy=gap_w + waveguide_w)
        pad_y = waveguide_w * 3 / 2 + gap_w + pad_dim[1] / 2 + rib_pad_w
        pads.append(
            Path(width=pad_dim[1]).segment(pad_dim[0]).translate(dx=interaction_l / 2 - pad_dim[0] / 2, dy=pad_y))
        if rib_pad_w > 0:
            pads.append(Path(width=pad_dim[1]).segment(pad_dim[0]).translate(
                dx=0, dy=waveguide_w * 3 / 2 + gap_w + rib_pad_w / 2))

        super(MemsMonitorCoupler, self).__init__(waveguide, monitor_wg, monitor_left, monitor_right, *pads)
        self.pads = pads[:1]

# class NemsMillerNode(GroupedPattern):
#     def __init__(self, waveguide_w: float, upper_interaction_l: float, lower_interaction_l: float,
#                  gap_w: float, bend_radius: float, bend_extension: float, lr_nanofin_w: float,
#                  ud_nanofin_w: float, lr_gap_w: float, ud_gap_w: float, lr_pad_dim: Optional[Dim2] = None,
#                  ud_pad_dim: Optional[Dim2] = None, lr_connector_dim: Optional[Dim2] = None,
#                  ud_connector_dim: Optional[Dim2] = None, shift: Tuple[float, float] = (0, 0)):
#         self.waveguide_w = waveguide_w
#         self.upper_interaction_l = upper_interaction_l
#         self.lower_interaction_l = lower_interaction_l
#         self.bend_radius = bend_radius
#         self.bend_extension = bend_extension
#         self.lr_nanofin_w = lr_nanofin_w
#         self.ud_nanofin_w = ud_nanofin_w
#         self.lr_pad_dim = lr_pad_dim
#         self.ud_pad_dim = ud_pad_dim
#         self.lr_connector_dim = lr_connector_dim
#         self.ud_connector_dim = ud_connector_dim
#         self.gap_w = gap_w
#
#         connectors, pads = [], []
#
#         bend_height = 2 * bend_radius + bend_extension
#         interport_distance = waveguide_w + 2 * bend_height + gap_w
#
#         if not upper_interaction_l <= lower_interaction_l:
#             raise ValueError("Require upper_interaction_l <= lower_interaction_l by convention.")
#
#         lower_path = Path(waveguide_w).dc((bend_radius, bend_height), lower_interaction_l, use_radius=True)
#         upper_path = Path(waveguide_w).dc((bend_radius, bend_height), upper_interaction_l,
#                                           (lower_interaction_l - upper_interaction_l) / 2,
#                                           inverted=True, use_radius=True)
#         upper_path.translate(dx=0, dy=interport_distance)
#
#         dc = Pattern(lower_path, upper_path)
#
#         nanofin_y = ud_nanofin_w / 2 + gap_w / 2 + waveguide_w + ud_gap_w
#         nanofins = [Box((lower_interaction_l, ud_nanofin_w)).center_align(dc).translate(dx=0, dy=-nanofin_y)]
#         pad_y = ud_connector_dim[1] + ud_pad_dim[1] / 2
#         pads += [Box(ud_pad_dim).center_align(nanofins[0]).translate(dy=-pad_y)]
#         connector = Box(ud_connector_dim).center_align(pads[0])
#         connectors += [copy(connector).vert_align(pads[0], bottom=True, opposite=True).horz_align(pads[0]),
#                        copy(connector).vert_align(pads[0], bottom=True, opposite=True).horz_align(pads[0], left=False)]
#
#         nanofin_x = lr_nanofin_w / 2 + lr_gap_w + upper_interaction_l / 2 + bend_radius + waveguide_w / 2
#         pad_x = lr_connector_dim[0] + lr_pad_dim[0] / 2
#         nanofin_y = bend_radius + waveguide_w + gap_w / 2 + bend_extension / 2
#
#         nanofins += [Box((lr_nanofin_w, bend_extension)).center_align(dc).translate(dx=-nanofin_x, dy=nanofin_y),
#                      Box((lr_nanofin_w, bend_extension)).center_align(dc).translate(dx=nanofin_x, dy=nanofin_y)]
#         pads += [Box(lr_pad_dim).center_align(nanofins[1]).translate(dx=-pad_x, dy=0),
#                  Box(lr_pad_dim).center_align(nanofins[2]).translate(dx=pad_x, dy=0)]
#         connector = Box(lr_connector_dim).center_align(pads[1])
#         connectors += [copy(connector).horz_align(pads[1], left=True, opposite=True).vert_align(pads[1]),
#                        copy(connector).horz_align(pads[1], left=True, opposite=True).vert_align(pads[1], bottom=False)]
#         connector = Box(lr_connector_dim).center_align(pads[2])
#         connectors += [copy(connector).horz_align(pads[2], left=False, opposite=True).vert_align(pads[2]),
#                        copy(connector).horz_align(pads[2], left=False, opposite=True).vert_align(pads[2], bottom=False)]
#
#         super(NemsMillerNode, self).__init__(*([dc] + nanofins + connectors + pads), shift=shift)
#         self.dc, self.connectors, self.nanofins, self.pads = dc, connectors, nanofins, pads
#
#     @property
#     def input_ports(self) -> np.ndarray:
#         bend_height = 2 * self.bend_radius + self.bend_extension
#         return np.asarray(((0, 0), (0, self.waveguide_w + 2 * bend_height + self.gap_w)))
#
#     @property
#     def output_ports(self) -> np.ndarray:
#         # TODO(sunil): change this to correct method
#         return self.input_ports + np.asarray((self.size[0], 0))
#
#     def multilayer(self, waveguide_layer: str='seam', metal_stack_layers: Tuple[str, ...] = ('m1am', 'm2am'), via_stack_layers: Tuple[str, ...] = ('cbam', 'v1am'),
#                    clearout_layer: str, clearout_etch_stop_layer: str, contact_box_dim: Dim2, clearout_box_dim: Dim2,
#                    doping_stack_layer: Optional[str] = None,
#                    clearout_etch_stop_grow: float = 0, via_shrink: float = 1, doping_grow: float = 0.25) -> Multilayer:
#         return multilayer(self, self.pads, ((self.center[0], self.pads[1].center[1]),),
#                           waveguide_layer, metal_stack_layers,
#                           via_stack_layers, clearout_layer, clearout_etch_stop_layer, contact_box_dim,
#                           clearout_box_dim, doping_stack_layer, clearout_etch_stop_grow, via_shrink, doping_grow)


# class RingResonator(Pattern):
#     def __init__(self, waveguide_w: float, taper_l: float = 0,
#                  taper_params: Union[np.ndarray, List[float]] = None,
#                  length: float = 5, num_taper_evaluations: int = 100, end_l: float = 0,
#                  shift: Dim2 = (0, 0), layer: int = 0):
#         self.end_l = end_l
#         self.length = length
#         self.waveguide_w = waveguide_w
#         p = Path(waveguide_w).segment(end_l, layer=layer) if end_l > 0 else Path(waveguide_w)
#         if end_l > 0:
#             p.segment(end_l, layer=layer)
#         if taper_l > 0 or taper_params is not None:
#             p.polynomial_taper(taper_l, taper_params, num_taper_evaluations, layer)
#         p.segment(length, layer=layer)
#         if taper_l > 0 or taper_params is not None:
#             p.polynomial_taper(taper_l, taper_params, num_taper_evaluations, layer, inverted=True)
#         if end_l > 0:
#             p.segment(end_l, layer=layer)
#         super(RingResonator, self).__init__(p, shift=shift)
#
#     @property
#     def input_ports(self) -> np.ndarray:
#         return np.asarray((0, 0)) + self.shift
#
#     @property
#     def output_ports(self) -> np.ndarray:
#         return self.input_ports + np.asarray((self.size[0], 0))
