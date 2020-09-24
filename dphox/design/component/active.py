from ...typing import *
from .pattern import Pattern, Path, GroupedPattern, get_linear_adiabatic, cubic_taper, Port
from .passive import Waveguide, DC, Box
from .multilayer import Multilayer, Via

from copy import deepcopy as copy

try:
    import plotly.graph_objects as go
except ImportError:
    pass


class LateralNemsPS(GroupedPattern):
    def __init__(self, waveguide_w: float, nanofin_w: float, phaseshift_l: float,
                 gap_w: float, taper_l: float, end_ls: Tuple[float] = (0,),
                 num_taper_evaluations: int = 100, pad_dim: Optional[Dim3] = None,
                 gap_taper: Optional[Tuple[float, ...]] = None, wg_taper: Optional[Tuple[float, ...]] = None,
                 boundary_taper: Optional[Tuple[float, ...]] = None,
                 end_taper: Optional[Tuple[Tuple[float, ...]]] = None):
        """NEMS single-mode phase shifter
        Args:
            waveguide_w: waveguide width
            nanofin_w: nanofin width (initial, before tapering)
            phaseshift_l: phase shift length
            gap_w: gap width (initial, before tapering)
            taper_l: taper length
            end_ls: end waveguide lengths
            num_taper_evaluations: number of taper evaluations (see gdspy)
            pad_dim: silicon handle xy size followed by distance between pad and fin to actuate
            gap_taper: gap taper polynomial params (recommend same as wg_taper)
            wg_taper: wg taper polynomial params (recommend same as gap_taper)
            end_taper: end taper for the transition into the phase shifter region
        """
        self.waveguide_w = waveguide_w
        self.nanofin_w = nanofin_w
        self.phaseshift_l = phaseshift_l
        self.gap_w = gap_w
        self.taper_l = taper_l
        self.end_ls = end_ls
        self.end_taper = end_taper
        self.num_taper_evaluations = num_taper_evaluations
        self.pad_dim = pad_dim
        self.gap_taper = gap_taper
        self.wg_taper = wg_taper
        self.boundary_taper = boundary_taper

        if not phaseshift_l >= 2 * taper_l:
            raise ValueError(
                f'Require interaction_l >= 2 * taper_l but got {phaseshift_l} < {2 * taper_l}')

        boundary_taper = wg_taper if boundary_taper is None else boundary_taper

        wg_taper = (wg_taper,) if end_taper is None else (*end_taper, wg_taper)
        box_w = nanofin_w * 2 + gap_w * 2 + waveguide_w
        wg = Waveguide(waveguide_w, taper_ls=(*end_ls, taper_l), taper_params=wg_taper, length=phaseshift_l + 2 * np.sum(end_ls),
                       num_taper_evaluations=num_taper_evaluations)
        boundary = Waveguide(box_w, taper_ls=(taper_l,), taper_params=(boundary_taper,), length=phaseshift_l,
                             num_taper_evaluations=num_taper_evaluations).align(wg).pattern
        gap_path = Waveguide(waveguide_w + gap_w * 2, taper_params=(gap_taper,), taper_ls=(taper_l,), length=phaseshift_l,
                             num_taper_evaluations=num_taper_evaluations).align(wg).pattern
        nanofins = [Pattern(poly) for poly in (boundary - gap_path)]

        pads, anchors = [], []
        if pad_dim is not None:
            pad = Box(pad_dim[:2]).align(wg)
            pad_y = nanofin_w + pad_dim[2] + pad_dim[1] / 2
            pads += [copy(pad).translate(dx=0, dy=-pad_y), copy(pad).translate(dx=0, dy=pad_y)]
        super(LateralNemsPS, self).__init__(*([wg] + nanofins + pads + anchors), call_union=False)
        self.waveguide, self.anchors, self.pads, self.nanofins = wg, anchors, pads, nanofins
        dy = np.asarray((0, self.nanofin_w / 2 + self.waveguide_w / 2 + self.gap_w))
        center = np.asarray(self.center)
        self.port['a0'] = Port(0, 0, -np.pi)
        self.port['b0'] = Port(self.phaseshift_l, 0)
        self.port['ps0'] = Port(gap_path.bounds[0], 0, -np.pi)
        self.port['ps1'] = Port(gap_path.bounds[1], 0)
        self.port['fin0'] = Port(*(center + dy))
        self.port['fin1'] = Port(*(center - dy))


class LateralNemsTDC(GroupedPattern):
    def __init__(self, waveguide_w: float, nanofin_w: float, dc_gap_w: float, beam_gap_w: float, bend_dim: Dim2,
                 interaction_l: float, dc_taper_ls: Tuple[float, ...] = None,
                 dc_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 beam_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 boundary_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 end_bend_dim: Optional[Dim3] = None,
                 pad_dim: Optional[Dim4] = None, use_radius: bool = True):
        """NEMS tunable directional coupler

        Args:
            waveguide_w: waveguide width
            nanofin_w: nanofin width
            dc_gap_w: directional coupler gap width
            beam_gap_w: gap between the nanofin and the TDC waveguides
            bend_dim: see DC
            interaction_l: interaction length
            dc_taper_ls: DC taper lengths
            dc_taper: tapering of the boundary of the directional coupler
            beam_taper: tapering of the lower boundary of the fin
            boundary_taper: tapering of the upper boundary of the fin (currently not implemented)
            end_bend_dim: If specified, places an additional end bend (see DC)
            pad_dim: If specified, silicon gnd pad xy size followed by connector dimensions (distance to guide, width)
            use_radius: use radius (see DC)
        """
        self.waveguide_w = waveguide_w
        self.nanofin_w = nanofin_w
        self.interaction_l = interaction_l
        self.dc_gap_w = dc_gap_w
        self.beam_gap_w = beam_gap_w
        self.pad_dim = pad_dim
        self.use_radius = use_radius

        dc = DC(bend_dim=bend_dim, waveguide_w=waveguide_w, gap_w=dc_gap_w,
                coupler_boundary_taper_ls=dc_taper_ls, coupler_boundary_taper=dc_taper,
                interaction_l=interaction_l, end_bend_dim=end_bend_dim, use_radius=use_radius)
        connectors, pads, gnd_connections, rib_brim = [], [], [], []

        nanofin_y = nanofin_w / 2 + dc_gap_w / 2 + waveguide_w + beam_gap_w
        nanofin = Box((interaction_l, nanofin_w)).align(dc)

        if dc_taper_ls is not None:
            if not interaction_l >= 2 * np.sum(dc_taper_ls):
                raise ValueError(
                    f'Require interaction_l > 2 * np.sum(dc_taper_ls) but got {interaction_l} < {2 * np.sum(dc_taper_ls)}')

        if beam_taper is None:
            nanofins = [copy(nanofin).translate(dx=0, dy=-nanofin_y), copy(nanofin).translate(dx=0, dy=nanofin_y)]
        else:
            box_w = (nanofin_w + beam_gap_w + waveguide_w) * 2 + dc_gap_w
            gap_taper_wg_w = (beam_gap_w + waveguide_w) * 2 + dc_gap_w
            # nanofin_box = Box((interaction_l, box_w)).align(dc).pattern
            nanofin_box = Waveguide(box_w, interaction_l, dc_taper_ls, boundary_taper).align(dc).pattern
            gap_taper_wg = Waveguide(gap_taper_wg_w, interaction_l, dc_taper_ls, beam_taper).align(dc).pattern
            nanofins = [Pattern(poly) for poly in (nanofin_box - gap_taper_wg)]

            ######### NATE: trying to taper fins of TDC ###################
            # boundary = Waveguide(box_w, taper_params=beam_taper, taper_ls=dc_taper_ls,
            #                      length=interaction_l).center_align(dc).pattern
            # gap_path = Waveguide(gap_taper_wg_w, taper_params=beam_taper, taper_ls=dc_taper_ls,
            #                      length=interaction_l).center_align(dc).pattern
            # nanofins = [Pattern(poly) for poly in (boundary - gap_path)]
            ######### NATE: trying to taper center of TDC ###################

        patterns = [dc] + nanofins + connectors + pads

        # TODO(Nate): make the brim connector to ground standard for 220nm, rework the taper helpers
        if pad_dim is not None:
            brim_l, brim_taper = get_linear_adiabatic(min_width=waveguide_w, max_width=1, aggressive=True)
            brim_taper = cubic_taper(brim_taper[1])
            gnd_contact_dim = pad_dim[2:]

            if not bend_dim[1] > 2 * bend_dim[0] + 2 * brim_l:
                raise ValueError(
                    f'Not enough room in s-bend to ground waveguide segment of length'
                    f'{bend_dim[1] - 2 * bend_dim[0]} need at least {2 * brim_l + gnd_contact_dim[-1]}')

            if not (pad_dim[0] + (waveguide_w / 2 + np.sum(brim_taper) / 2 + gnd_contact_dim[0])) < bend_dim[0]:
                raise ValueError(
                    f'Not enough room in s-bend to ground waveguide with bend_dim[0] of {bend_dim[0]}'
                    f'need at least {(pad_dim[0] + (waveguide_w / 2 + np.sum(brim_taper) / 2 + gnd_contact_dim[0]))}')

            rib_brim, gnd_connections, pads = [], [], []
            dx_brim = bend_dim[0]
            dy_brim = bend_dim[1] / 2
            min_x, min_y, max_x, max_y = dc.pattern.bounds
            flip_x = flip_y = True
            for x in (min_x + dx_brim, max_x - dx_brim):
                for y in (min_y + dy_brim, max_y - dy_brim):
                    flip_y = not flip_y
                    rib_brim.append(Waveguide(waveguide_w, taper_ls=(brim_l,), taper_params=(brim_taper,),
                                              length=2 * brim_l + gnd_contact_dim[-1], rotate_angle=np.pi / 2).translate(dx=x,
                                                                                                                         dy=y - brim_l))
                    if flip_x:
                        gnd_connections.append(
                            Box(gnd_contact_dim[:2]).translate(dx=x + waveguide_w / 2,
                                                               dy=y))
                    else:
                        gnd_connections.append(
                            Box(gnd_contact_dim[:2]).translate(dx=x - waveguide_w / 2 - gnd_contact_dim[0],
                                                               dy=y))
                    pads.append(
                        Box(pad_dim[:2]).align(rib_brim[-1]).halign(gnd_connections[-1],
                                                                    left=flip_x,
                                                                    opposite=True))
                flip_x = not flip_x
            rib_brim = [Pattern(poly) for brim in rib_brim for poly in (brim.pattern - dc.pattern)]
            patterns += gnd_connections + rib_brim + pads
        super(LateralNemsTDC, self).__init__(*patterns, call_union=False)
        self.dc, self.connectors, self.pads, self.nanofins = dc, connectors, pads, nanofins
        if pad_dim is not None:
            self.gnd_connections, self.rib_brim = gnd_connections, rib_brim
        self.port = self.dc.port
        dy = np.asarray((0, self.nanofin_w / 2 + self.waveguide_w + self.dc_gap_w / 2 + self.beam_gap_w))
        center = np.asarray(self.center)
        self.port['t0'] = Port(*(center + dy))
        self.port['t1'] = Port(*(center - dy))
        gnd_labels = ['gnd0_l_0', 'gnd0_u_0', 'gnd0_l_1', 'gnd0_u_1']
        angle = 0
        for gnd_label, pad in zip(gnd_labels, pads):
            center = pad.center
            self.port[gnd_label] = Port(*(center), a=angle)
            angle += np.pi


class NemsAnchor(GroupedPattern):
    def __init__(self, fin_spring_dim: Dim2, shuttle_dim: Dim2, top_spring_dim: Dim2 = None,
                 straight_connector: Optional[Dim2] = None, tether_connector: Optional[Dim3] = None,
                 pos_electrode_dim: Optional[Dim3] = None, neg_electrode_dim: Optional[Dim2] = None,
                 include_fin_dummy: bool = False, attach_comb: bool = False, tooth_dim: Dim3 = (0.3, 3, 0.15)):
        """NEMS anchor

        Args:
            fin_spring_dim: fixed fin dimension (x, y)
            top_spring_dim: fin dimension (x, y)
            shuttle_dim: shuttle dimension
            straight_connector: straight connector to the fin, box xy (overridden by loop connector)
            tether_connector: tether connector to the fin, xy dim and segment length on the top part of loop
            pos_electrode_dim: positive electrode dimension
            neg_electrode_dim: negative electrode dimension
            include_fin_dummy: include fin dummy for for mechanical support
            attach_comb: attach a comb drive to the shuttle (only if pos_electrode_dim specified!)
            tooth_dim: (length, width, inter-tooth gap)
        """
        self.fin_spring_dim = fin_spring_dim
        self.top_spring_dim = top_spring_dim
        self.shuttle_dim = shuttle_dim
        self.straight_connector = straight_connector
        self.tether_connector = tether_connector
        self.pos_electrode_dim = pos_electrode_dim
        self.neg_electrode_dim = neg_electrode_dim
        patterns, c_ports, pads, springs = [], [], [], []

        top_spring_dim = fin_spring_dim if not top_spring_dim else top_spring_dim
        connector = Box(shuttle_dim).translate()
        shuttle = copy(connector)
        comb = None
        if tether_connector is not None and straight_connector is None:
            s = tether_connector
            loop = Pattern(Path(fin_spring_dim[1]).sbend((-s[0], s[1])).sbend((s[0], s[1])).segment(
                tether_connector[2]).sbend((s[0], -s[1])).sbend((-s[0], -s[1])).segment(tether_connector[2]))
            straight = Box(s[-2:])
            straight.align(connector).valign(connector, bottom=False, opposite=True)
            loop.align(straight).valign(straight, bottom=False, opposite=True)
            connector = GroupedPattern(shuttle, straight, loop)
        elif straight_connector is not None:
            straight = Box(straight_connector)
            connector = GroupedPattern(connector,
                                       copy(straight).halign(connector).valign(connector, bottom=False,
                                                                               opposite=True),
                                       copy(straight).halign(connector, left=False,
                                                             opposite=False).valign(connector,
                                                                                    bottom=False,
                                                                                    opposite=True),
                                       # adding more straight connectors for mirror symmetric mechanics
                                       copy(straight).halign(connector, left=False, opposite=False).valign(
                                           copy(connector).translate(connector.size[0], connector.size[1]), ),
                                       copy(straight).halign(connector).valign(
                                           copy(connector).translate(connector.size[0], connector.size[1]), ),
                                       copy(straight).align(connector).valign(
                                           copy(connector).translate(connector.size[0], connector.size[1]), )
                                       )
        if include_fin_dummy and not attach_comb:
            # this is the mirror image dummy for mechanics
            patterns.append(Box(fin_spring_dim).align((connector.center[0], connector.bounds[3])))
        patterns.append(connector)
        if top_spring_dim is not None:
            top_spring = Box(top_spring_dim).align(shuttle).valign(shuttle, bottom=True, opposite=True)
            bottom_spring = Box(top_spring_dim).align(shuttle).valign(shuttle, bottom=False, opposite=True)
            if pos_electrode_dim is not None:
                pos_electrode = Box((pos_electrode_dim[0], pos_electrode_dim[1])).align(top_spring).valign(
                    top_spring, opposite=True).translate(dy=pos_electrode_dim[2])
                patterns.append(pos_electrode)
                pads.append(pos_electrode)
                patterns.extend([top_spring, bottom_spring])
                springs.extend([top_spring, bottom_spring])
                if attach_comb:
                    dx_teeth = tooth_dim[2] + tooth_dim[0]
                    num_teeth = int((min(pos_electrode_dim[0], shuttle_dim[0]) - tooth_dim[1]) // (2 * dx_teeth))
                    if num_teeth <= 0:
                        raise ValueError('Electrode dim is too small to hold comb teeth.')
                    tooth = Box(tooth_dim[:2])
                    upper_comb = GroupedPattern(*[copy(tooth).translate(dx_teeth * 2 * n)
                                                  for n in range(num_teeth)])
                    lower_comb = GroupedPattern(*[copy(tooth).translate(dx_teeth * (2 * n + 1))
                                                  for n in range(num_teeth - 1)])
                    upper_comb.align(pos_electrode).valign(pos_electrode, opposite=True, bottom=False)
                    lower_comb.align(shuttle).valign(shuttle, opposite=True)
                    comb = GroupedPattern(upper_comb, lower_comb)
                    patterns.append(comb)
            else:
                if attach_comb:
                    raise AttributeError('Must specify pos_electrode_dim if attach_comb is True')
                pads.append(shuttle)
            if neg_electrode_dim is not None:
                # moving alignment to account for bottom spring
                neg_electrode_left = Box(neg_electrode_dim).halign(
                    bottom_spring, opposite=True).valign(bottom_spring)
                neg_electrode_right = Box(neg_electrode_dim).halign(
                    bottom_spring, left=False, opposite=True).valign(bottom_spring)
                patterns.extend([neg_electrode_left, neg_electrode_right])
                pads.extend([neg_electrode_left, neg_electrode_right])

        super(NemsAnchor, self).__init__(*patterns)
        shift = (-connector.center[0], -connector.bounds[1])
        self.translate(*shift)
        self.pads = [pad.translate(*shift) for pad in pads]
        self.springs = [s.translate(*shift) for s in springs]
        self.shuttle = shuttle.translate(*shift) if pos_electrode_dim is not None else shuttle
        self.comb = comb.translate(*shift) if comb is not None else comb
        for idx, pad in enumerate(self.pads):
            self.port[f'c{idx}'] = Port(*pad.center, -np.pi)
            self.port[f'd{idx}'] = Port(*pad.center)


class GndWaveguide(GroupedPattern):
    def __init__(self, waveguide_w: float, length: float, rib_brim_w: float, gnd_connector_dim: Optional[Dim2],
                 gnd_contact_dim: Optional[Dim2], flip: bool=False, wg_taper: Optional[Tuple[float]] = None,
                 symmetric_taper: bool = False, symmetric: bool = False):
        """Grounded waveguide, typically required for photonic MEMS, consisting of a rib brim
        around an (optionally) tapered waveguide.

        Args:
            waveguide_w:
            length:
            gnd_contact_dim:
            rib_brim_w:
            gnd_connector_dim:
            flip:
            wg_taper:
            symmetric_taper:
            symmetric: symmetric gnd connector connections (top and bottom of brim)
        """
        self.waveguide_w = waveguide_w
        self.rib_brim_w = rib_brim_w
        self.length = length
        self.gnd_contact_dim = gnd_contact_dim
        self.gnd_connector_dim = gnd_connector_dim

        pads = []

        # TODO(): remove the hidden hardcoding
        brim_l, brim_taper = get_linear_adiabatic(min_width=waveguide_w, max_width=rib_brim_w, aggressive=True)
        brim_taper = cubic_taper(brim_taper[1])

        wg = Waveguide(waveguide_w=waveguide_w, length=length, taper_ls=(length,), taper_params=(wg_taper,),
                       symmetric=symmetric_taper)
        rib_brim = Waveguide(waveguide_w=waveguide_w, length=2 * brim_l, taper_ls=(brim_l,),
                             taper_params=(brim_taper,)).align(wg)
        gnd_connection = Box(gnd_connector_dim).align(wg).valign(wg, bottom=not flip, opposite=True)
        rib_brim = [Pattern(poly) for poly in (rib_brim.pattern - wg.pattern)]

        if gnd_contact_dim is not None:
            pad = Box(gnd_contact_dim).align(gnd_connection).valign(gnd_connection, bottom=not flip, opposite=True)
            pads.append(pad)

        patterns = rib_brim + [wg, gnd_connection]

        super(GndWaveguide, self).__init__(*patterns)
        self.wg, self.rib_brim, self.pads = wg, rib_brim, pads
        self.port['a0'] = Port(0, 0, -np.pi)
        self.port['b0'] = Port(length, 0)
        if gnd_contact_dim is not None:
            self.port['gnd1'] = Port(*pads[0].center, -np.pi)
            self.port['gnd0'] = Port(*pads[0].center)


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


class LateralNemsPSFull(Multilayer):
    def __init__(self, ps: LateralNemsPS, anchor: NemsAnchor,
                 metal_via: Via, pad_via: Via, trace_w: float,
                 ridge: str, rib: str, shuttle_dope: str,
                 spring_dope: str, pad_dope: str, pos_metal: str, gnd_metal: str):
        self.config = {
            'ps': ps.config,
            'anchor': anchor.config,
            'metal_via': metal_via.config,
            'pad_via': pad_via.config,
            'trace_w': trace_w,
            'ridge': ridge,
            'rib': rib,
            'pos_metal': pos_metal,
            'gnd_metal': gnd_metal,
            'shuttle_dope': shuttle_dope,
            'spring_dope': spring_dope,
            'pad_dope': pad_dope
        }
        top = copy(anchor).translate(*ps.port['t0'].xy)
        bot = copy(anchor).flip().translate(*ps.port['t1'].xy)
        full_ps = GroupedPattern(top, bot, ps)
        vias = metal_via.pattern_to_layer + pad_via.pattern_to_layer
        dopes = [top.shuttle.dope(shuttle_dope), bot.shuttle.dope(shuttle_dope)] + \
                [s.dope(spring_dope) for s in top.springs + bot.springs] + \
                [s.dope(pad_dope) for s in top.pads + bot.pads]
        super(LateralNemsPSFull, self).__init__([(full_ps, ridge), (ps.rib_brim, rib)] + vias + dopes)


class NemsMillerNode(GroupedPattern):
    def __init__(self, waveguide_w: float, upper_interaction_l: float, lower_interaction_l: float,
                 gap_w: float, bend_radius: float, bend_extension: float, lr_nanofin_w: float,
                 ud_nanofin_w: float, lr_gap_w: float, ud_gap_w: float, lr_pad_dim: Optional[Dim2] = None,
                 ud_pad_dim: Optional[Dim2] = None, lr_connector_dim: Optional[Dim2] = None,
                 ud_connector_dim: Optional[Dim2] = None):
        self.waveguide_w = waveguide_w
        self.upper_interaction_l = upper_interaction_l
        self.lower_interaction_l = lower_interaction_l
        self.bend_radius = bend_radius
        self.bend_extension = bend_extension
        self.lr_nanofin_w = lr_nanofin_w
        self.ud_nanofin_w = ud_nanofin_w
        self.lr_pad_dim = lr_pad_dim
        self.ud_pad_dim = ud_pad_dim
        self.lr_connector_dim = lr_connector_dim
        self.ud_connector_dim = ud_connector_dim
        self.gap_w = gap_w

        connectors, pads = [], []

        bend_height = 2 * bend_radius + bend_extension
        interport_distance = waveguide_w + 2 * bend_height + gap_w

        if not upper_interaction_l <= lower_interaction_l:
            raise ValueError("Require upper_interaction_l <= lower_interaction_l by convention.")

        lower_path = Path(waveguide_w).dc((bend_radius, bend_height), lower_interaction_l, use_radius=True)
        upper_path = Path(waveguide_w).dc((bend_radius, bend_height), upper_interaction_l,
                                          (lower_interaction_l - upper_interaction_l) / 2,
                                          inverted=True, use_radius=True)
        upper_path.translate(dx=0, dy=interport_distance)

        dc = Pattern(lower_path, upper_path)

        nanofin_y = ud_nanofin_w / 2 + gap_w / 2 + waveguide_w + ud_gap_w
        nanofins = [Box((lower_interaction_l, ud_nanofin_w)).align(dc).translate(dx=0, dy=-nanofin_y)]
        pad_y = ud_connector_dim[1] + ud_pad_dim[1] / 2
        pads += [Box(ud_pad_dim).align(nanofins[0]).translate(dy=-pad_y)]
        connector = Box(ud_connector_dim).align(pads[0])
        connectors += [copy(connector).valign(pads[0], bottom=True, opposite=True).halign(pads[0]),
                       copy(connector).valign(pads[0], bottom=True, opposite=True).halign(pads[0], left=False)]

        nanofin_x = lr_nanofin_w / 2 + lr_gap_w + upper_interaction_l / 2 + bend_radius + waveguide_w / 2
        pad_x = lr_connector_dim[0] + lr_pad_dim[0] / 2
        nanofin_y = bend_radius + waveguide_w + gap_w / 2 + bend_extension / 2

        nanofins += [Box((lr_nanofin_w, bend_extension)).align(dc).translate(dx=-nanofin_x, dy=nanofin_y),
                     Box((lr_nanofin_w, bend_extension)).align(dc).translate(dx=nanofin_x, dy=nanofin_y)]
        pads += [Box(lr_pad_dim).align(nanofins[1]).translate(dx=-pad_x, dy=0),
                 Box(lr_pad_dim).align(nanofins[2]).translate(dx=pad_x, dy=0)]
        connector = Box(lr_connector_dim).align(pads[1])
        connectors += [copy(connector).halign(pads[1], left=True, opposite=True).valign(pads[1]),
                       copy(connector).halign(pads[1], left=True, opposite=True).valign(pads[1], bottom=False)]
        connector = Box(lr_connector_dim).align(pads[2])
        connectors += [copy(connector).halign(pads[2], left=False, opposite=True).valign(pads[2]),
                       copy(connector).halign(pads[2], left=False, opposite=True).valign(pads[2], bottom=False)]

        super(NemsMillerNode, self).__init__(*([dc] + nanofins + connectors + pads))
        self.dc, self.connectors, self.nanofins, self.pads = dc, connectors, nanofins, pads

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
