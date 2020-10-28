from ..typing import *
from .pattern import Pattern, Path, get_linear_adiabatic, cubic_taper, Port
from .passive import Waveguide, DC, Box
from .multilayer import Multilayer, Via

from copy import deepcopy as copy

SIMPHOX_IMPORTED = False
MEEP_IMPORTED = False  # for meep sims

try:
    SIMPHOX_IMPORTED = True
    from simphox.device import ModeDevice, MaterialBlock, SILICON, Material

    AIR = Material('Air', (0, 0, 0), 1)
except ImportError:
    SIMPHOX_IMPORTED = False
    pass


class LateralNemsPS(Pattern):
    def __init__(self, waveguide_w: float, nanofin_w: float, phaseshift_l: float, gap_w: float,
                 taper_l: float, fin_end_bend_dim: Dim2, rib_etch_grow: float, gnd_connector: Optional[Dim3] = None,
                 gnd_pad_dim: Optional[Dim2] = None, end_ls: Tuple[float] = (0,), num_taper_evaluations: int = 100,
                 gap_taper: Optional[Tuple[float, ...]] = None, wg_taper: Optional[Tuple[float, ...]] = None,
                 boundary_taper: Optional[Tuple[float, ...]] = None,
                 end_taper: Optional[Tuple[Tuple[float, ...]]] = None, gnd_connector_idx: int = -1):
        """NEMS single-mode phase shifter
        Args:
            waveguide_w: waveguide width
            nanofin_w: nanofin width (initial, before tapering)
            phaseshift_l: phase shift length
            gap_w: gap width (initial, before tapering)
            gnd_connector: tuple of the form (rib_brim_w, connectorx, connectory)
            gnd_pad_dim: add a pad to the ground connector
            fin_end_bend_dim: adiabatic fin bending
            rib_etch_grow: rib etch grow (extra growth accounts for foreshortening and/or misalignment)
            taper_l: taper length at start and end (including fins)
            end_ls: end waveguide lengths (not including fins)
            num_taper_evaluations: number of taper evaluations (see gdspy)
            gap_taper: gap taper polynomial params (recommend same as wg_taper)
            wg_taper: wg taper polynomial params (recommend same as gap_taper)
            end_taper: end taper for the transition into the phase shifter region
            gnd_connector_idx: the ground connector will be connected to the corresponding end-length section on the PS
        """
        self.waveguide_w = waveguide_w
        self.nanofin_w = nanofin_w
        self.phaseshift_l = phaseshift_l
        self.gap_w = gap_w
        self.taper_l = taper_l
        self.end_ls = end_ls
        self.end_taper = end_taper
        self.num_taper_evaluations = num_taper_evaluations
        self.gap_taper = gap_taper
        self.wg_taper = wg_taper
        self.boundary_taper = boundary_taper
        self.fin_end_bend_dim = fin_end_bend_dim
        self.gnd_pad_dim = gnd_pad_dim
        self.gnd_connector = gnd_connector
        self.gnd_connector_idx = gnd_connector_idx
        self.rib_etch_grow = rib_etch_grow

        if not phaseshift_l >= 2 * taper_l:
            raise ValueError(f'Require interaction_l >= 2 * taper_l but got {phaseshift_l} < {2 * taper_l}')

        boundary_taper = wg_taper if boundary_taper is None else boundary_taper

        nanofins = []

        if nanofin_w is not None:
            wg_taper = (wg_taper,) if end_taper is None else (*end_taper, wg_taper)
            wg = Waveguide(waveguide_w, taper_ls=(*end_ls, taper_l), taper_params=wg_taper,
                           length=phaseshift_l + 2 * np.sum(end_ls),
                           num_taper_evaluations=num_taper_evaluations)
            _gap_w = gap_w + np.sum([np.sum(et) for et in end_taper]) / 2
            box_w = nanofin_w * 2 + _gap_w * 2 + waveguide_w
            boundary = Waveguide(box_w, taper_ls=(taper_l,), taper_params=(boundary_taper,), length=phaseshift_l,
                                 num_taper_evaluations=num_taper_evaluations).align(wg).shapely
            gap_path = Waveguide(waveguide_w + _gap_w * 2, taper_params=(gap_taper,), taper_ls=(taper_l,),
                                 length=phaseshift_l,
                                 num_taper_evaluations=num_taper_evaluations).align(wg).shapely
            nanofins = [Pattern(poly) for poly in (boundary - gap_path)]
            if fin_end_bend_dim is not None:
                nanofin_adiabatic = Pattern(Path(nanofin_w).sbend(fin_end_bend_dim))
                nanofin_height = waveguide_w / 2 + _gap_w + nanofin_w / 2
                nanofin_ends = [
                    nanofin_adiabatic.copy.translate(nanofins[0].bounds[2], nanofin_height),
                    nanofin_adiabatic.copy.flip().translate(nanofins[1].bounds[2], -nanofin_height),
                    nanofin_adiabatic.copy.flip(horiz=True).translate(nanofins[0].bounds[0], nanofin_height),
                    nanofin_adiabatic.copy.flip(horiz=True).flip().translate(nanofins[1].bounds[0], -nanofin_height)
                ]
            else:
                nanofin_ends = []
            patterns = [wg] + nanofins + nanofin_ends
        else:
            wg_taper = (wg_taper,) if end_taper is None else (*end_taper, wg_taper)
            wg = Waveguide(waveguide_w, taper_ls=(*end_ls, taper_l), taper_params=wg_taper,
                           length=phaseshift_l + 2 * np.sum(end_ls),
                           num_taper_evaluations=num_taper_evaluations, slot_dim=(phaseshift_l, gap_w),
                           slot_taper_ls=(taper_l,), slot_taper_params=(gap_taper,))
            patterns = [wg]

        rib_brim, gnd_pads, rib_etch = [], [], []

        gnd_connector_idx = len(end_ls) - 1 if gnd_connector_idx == -1 else gnd_connector_idx
        rib_brim_x = float(np.sum(end_ls[:gnd_connector_idx]))

        if gnd_connector is not None:
            final_width = waveguide_w + np.sum([np.sum(t) for t in end_taper])
            rib_brim = Waveguide(waveguide_w=waveguide_w, length=end_ls[gnd_connector_idx],
                                 taper_ls=(end_ls[gnd_connector_idx] / 2, end_ls[gnd_connector_idx] / 2),
                                 taper_params=(cubic_taper(gnd_connector[0] - waveguide_w),
                                               cubic_taper(final_width - gnd_connector[0])),
                                 symmetric=False
                                 )
            # TODO(Nate): make not clunky, and foundry agnostic
            #  (rule for how etches are handled at different foundries?)
            #  clunky way to separate seam and ream masks for now
            rib_etch = Waveguide(waveguide_w=waveguide_w + 2 * rib_etch_grow, length=end_ls[gnd_connector_idx],
                                 taper_ls=(end_ls[gnd_connector_idx] / 2, end_ls[gnd_connector_idx] / 2),
                                 taper_params=(cubic_taper(gnd_connector[0] - waveguide_w),
                                               cubic_taper(final_width - gnd_connector[0])),
                                 symmetric=False
                                 )
            rib_brims_etch = [copy(rib_etch).translate(rib_brim_x),
                              copy(rib_etch).flip(horiz=True).translate(wg.size[0])]

            rib_brims = [copy(rib_brim).translate(rib_brim_x),
                         copy(rib_brim).flip(horiz=True).translate(wg.size[0])]

            gnd_connections = [
                Box(gnd_connector[1:]).align(rib_brims[0]).valign(rib_brim, bottom=True, opposite=True),
                Box(gnd_connector[1:]).align(rib_brims[0]).valign(rib_brim, bottom=False, opposite=True),
                Box(gnd_connector[1:]).align(rib_brims[1]).valign(rib_brim, bottom=True, opposite=True),
                Box(gnd_connector[1:]).align(rib_brims[1]).valign(rib_brim, bottom=False, opposite=True)
            ]

            if gnd_pad_dim is not None:
                gnd_pads += [
                    Box(gnd_pad_dim).align(rib_brims[0]).valign(gnd_connections[0], bottom=True, opposite=True),
                    Box(gnd_pad_dim).align(rib_brims[0]).valign(gnd_connections[1], bottom=False, opposite=True),
                    Box(gnd_pad_dim).align(rib_brims[1]).valign(gnd_connections[2], bottom=True, opposite=True),
                    Box(gnd_pad_dim).align(rib_brims[1]).valign(gnd_connections[3], bottom=False, opposite=True)
                ]

            rib_brim = Pattern(*rib_brims)
            rib_brim = [Pattern(poly) for poly in rib_brim.shapely - wg.shapely]
            rib_etch = Pattern(*rib_brims_etch)
            rib_etch = [Pattern(poly) for poly in rib_etch.shapely - wg.shapely]

            patterns.extend(rib_brim + gnd_connections + gnd_pads)

        super(LateralNemsPS, self).__init__(*patterns, call_union=False)
        self.waveguide, self.nanofins, self.rib_brim, self.gnd_pads, self.pads = wg, nanofins, rib_etch, gnd_pads, gnd_pads + gnd_connections
        dy = np.asarray((0, self.nanofin_w / 2 + self.waveguide_w / 2 + self.gap_w))
        center = np.asarray(self.center)
        self.port['a0'] = Port(0, 0, -np.pi)
        self.port['b0'] = Port(self.phaseshift_l + 2 * np.sum(self.end_ls), 0)
        if nanofin_w is not None:
            self.port['ps0'] = Port(gap_path.bounds[0], 0, -np.pi)
            self.port['ps1'] = Port(gap_path.bounds[2], 0)
            self.port['fin0'] = Port(*(center + dy))
            self.port['fin1'] = Port(*(center - dy), np.pi)

    def effective_index(self, waveguide_h: float = 0.22, grid_spacing: float = 0.01,
                        wavelength: float = 1.55, wg_z: float = 1, sim_size: Dim2 = (2.5, 2)):
        if not SIMPHOX_IMPORTED:
            raise ImportError("This method requires simphox to be imported")
        waveguide_w_change = np.sum([np.sum(et) for et in self.end_taper])
        waveguide_w = self.waveguide_w + waveguide_w_change
        mode_device = ModeDevice(
            wg=MaterialBlock((waveguide_w, waveguide_h), SILICON),
            sub=MaterialBlock(sim_size, AIR),
            size=sim_size,
            wavelength=wavelength,
            wg_height=wg_z,
            spacing=grid_spacing
        )
        nanofin_ps = MaterialBlock((self.nanofin_w, waveguide_h), SILICON)
        return mode_device.solve(mode_device.single(lat_ps=nanofin_ps, sep=self.gap_w))

    def update(self, new: bool = True, **kwargs):
        """Update this class with a new set of parameters using config

        Args:
            new: Return new instance instead of updating
            **kwargs: all of the arguments to update

        Returns:

        """
        config = copy(self.config)
        config.update(kwargs)
        if not new:
            self.__init__(**config)
            return self
        else:
            return LateralNemsPS(**config)


class LateralNemsTDC(Pattern):
    def __init__(self, waveguide_w: float, nanofin_w: float, dc_gap_w: float, beam_gap_w: float, bend_dim: Dim2,
                 interaction_l: float, fin_end_bend_dim: Dim2, gnd_wg: Dim4, rib_etch_grow: float,
                 dc_taper_ls: Tuple[float, ...] = None,
                 dc_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 beam_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 boundary_taper: Optional[Tuple[Tuple[float, ...]]] = None,
                 end_bend_dim: Optional[Dim3] = None, dc_end_l: float = 0, use_radius: bool = True):
        """NEMS tunable directional coupler

        Args:
            waveguide_w: waveguide width
            nanofin_w: nanofin width
            dc_gap_w: directional coupler gap width
            beam_gap_w: gap between the nanofin and the TDC waveguides
            bend_dim: see DC
            interaction_l: interaction length
            fin_end_bend_dim: adiabatic transition for the fin end bend dim
            gnd_wg: ground waveguide dimensions
            rib_etch_grow: rib etch grow (extra growth accounts for foreshortening and/or misalignment)
            dc_taper_ls: DC taper lengths
            dc_taper: tapering of the boundary of the directional coupler
            beam_taper: tapering of the lower boundary of the fin
            boundary_taper: tapering of the upper boundary of the fin (currently not implemented)
            end_bend_dim: If specified, places an additional end bend (see DC)
            dc_end_l: End length for the directional coupler
            use_radius: use radius (see DC)
        """
        self.waveguide_w = waveguide_w
        self.nanofin_w = nanofin_w
        self.interaction_l = interaction_l
        self.dc_gap_w = dc_gap_w
        self.beam_gap_w = beam_gap_w
        self.use_radius = use_radius
        self.dc_end_l = dc_end_l
        self.dc_taper = dc_taper
        self.dc_taper_ls = dc_taper_ls
        self.fin_end_bend_dim = fin_end_bend_dim
        self.gnd_wg = gnd_wg
        self.bend_dim = bend_dim
        self.boundary_taper = boundary_taper
        self.beam_taper = beam_taper
        self.end_bend_dim = end_bend_dim
        self.rib_etch_grow = rib_etch_grow

        dc = DC(bend_dim=bend_dim, waveguide_w=waveguide_w, gap_w=dc_gap_w,
                coupler_boundary_taper_ls=dc_taper_ls, coupler_boundary_taper=dc_taper,
                interaction_l=interaction_l + 2 * dc_end_l, end_bend_dim=end_bend_dim, use_radius=use_radius)
        connectors, gnd_wg_pads, gnd_connections, rib_etch = [], [], [], []

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
            nanofin_box = Waveguide(box_w, interaction_l, dc_taper_ls, boundary_taper).align(dc).shapely
            gap_taper_wg = Waveguide(gap_taper_wg_w, interaction_l, dc_taper_ls, beam_taper).align(dc).shapely
            nanofins = [Pattern(poly) for poly in (nanofin_box - gap_taper_wg)]

        nanofin_adiabatic = Pattern(Path(nanofin_w).sbend(fin_end_bend_dim))
        nanofin_height = (nanofin_w / 2 + beam_gap_w + waveguide_w) + dc_gap_w / 2
        nanofin_ends = Pattern(
            nanofin_adiabatic.copy.translate(nanofins[0].bounds[2], nanofin_height),
            nanofin_adiabatic.copy.flip().translate(nanofins[1].bounds[2], -nanofin_height),
            nanofin_adiabatic.copy.flip(horiz=True).translate(nanofins[0].bounds[0], nanofin_height),
            nanofin_adiabatic.copy.flip(horiz=True).flip().translate(nanofins[1].bounds[0], -nanofin_height)
        ).align(dc)

        patterns = [dc] + nanofins + connectors + gnd_wg_pads + [nanofin_ends]

        # TODO(Nate): make the brim connector to ground standard for 220nm, rework the taper helpers
        if gnd_wg is not None:
            brim_l, brim_taper = get_linear_adiabatic(min_width=waveguide_w, max_width=1, aggressive=True)
            brim_taper = cubic_taper(brim_taper[1])
            gnd_contact_dim = gnd_wg[2:]

            if not bend_dim[1] > 2 * bend_dim[0] + 2 * brim_l:
                raise ValueError(
                    f'Not enough room in s-bend to ground waveguide segment of length'
                    f'{bend_dim[1] - 2 * bend_dim[0]} need at least {2 * brim_l + gnd_contact_dim[-1]}')

            if not (gnd_wg[0] + (waveguide_w / 2 + np.sum(brim_taper) / 2 + gnd_contact_dim[0])) < bend_dim[0]:
                raise ValueError(
                    f'Not enough room in s-bend to ground waveguide with bend_dim[0] of {bend_dim[0]}'
                    f'need at least {(gnd_wg[0] + (waveguide_w / 2 + np.sum(brim_taper) / 2 + gnd_contact_dim[0]))}')

            rib_etch, gnd_connections, gnd_wg_pads = [], [], []
            dx_brim = bend_dim[0]
            dy_brim = (dc_gap_w + waveguide_w + bend_dim[1]) / 2
            min_x, min_y, max_x, max_y = dc.shapely.bounds
            # remaking the y coordinate center referenced for symmetry
            center_y = (max_y + min_y) / 2 - gnd_contact_dim[-1] / 2
            flip_x = flip_y = True
            rib_brim = []
            for x in (min_x + dx_brim, max_x - dx_brim):
                for y in (center_y - dy_brim, center_y + dy_brim):
                    flip_y = not flip_y
                    rib_brim += [Waveguide(
                        waveguide_w,
                        taper_ls=(brim_l,),
                        taper_params=(brim_taper,),
                        length=2 * brim_l + gnd_contact_dim[-1],
                        rotate_angle=np.pi / 2).translate(dx=x, dy=y - brim_l)]

                    # TODO(Nate): make not clunky, and foundry agnostic
                    #  (rule for how etches are handled at different foundries?)
                    #  clunky way to separate seam and ream masks for now
                    rib_etch.append(
                        Waveguide(
                            waveguide_w + 2 * rib_etch_grow, taper_ls=(brim_l,),
                            taper_params=(brim_taper,), length=2 * brim_l + gnd_contact_dim[-1],
                            rotate_angle=np.pi / 2).translate(dx=x, dy=y - brim_l)
                    )
                    gnd_connections.append(
                        Box(gnd_contact_dim[:2]).align(rib_brim[-1]).halign(rib_brim[-1],
                                                                            opposite=True,
                                                                            left=flip_x))
                    gnd_wg_pads.append(
                        Box(gnd_wg[:2]).align(rib_brim[-1]).halign(gnd_connections[-1],
                                                                  left=flip_x,
                                                                  opposite=True))
                flip_x = not flip_x
            rib_etch = [Pattern(poly) for brim in rib_etch for poly in (brim.shapely - dc.shapely)]
            patterns += gnd_connections + rib_brim + gnd_wg_pads
        super(LateralNemsTDC, self).__init__(*patterns, call_union=False)
        self.dc, self.connectors, self.pads, self.nanofins = dc, connectors, gnd_wg_pads, nanofins
        self.gnd_connections, self.rib_brim = gnd_connections, rib_etch
        self.port = self.dc.port
        self.gnd_wg_pads = gnd_wg_pads
        dy = np.asarray((0, self.nanofin_w / 2 + self.waveguide_w + self.dc_gap_w / 2 + self.beam_gap_w))
        center = np.asarray(self.center)
        self.port['fin0'] = Port(*(center + dy))
        self.port['fin1'] = Port(*(center - dy), np.pi)

    def update(self, new: bool = True, **kwargs):
        """Update this class with a new set of parameters using config

        Args:
            new: Return new instance instead of updating
            **kwargs: all of the arguments to update

        Returns:

        """
        config = copy(self.config)
        config.update(kwargs)
        if not new:
            self.__init__(**config)
            return self
        else:
            return LateralNemsTDC(**config)


class NemsAnchor(Pattern):
    def __init__(self, fin_dim: Dim2, shuttle_dim: Dim2, spring_dim: Dim2 = None,
                 straight_connector: Optional[Dim2] = None, tether_connector: Optional[Dim4] = None,
                 pos_electrode_dim: Optional[Dim3] = None, gnd_electrode_dim: Optional[Dim2] = None,
                 include_support_spring: bool = False, tooth_param: Dim3 = None, shuttle_stripe_w: float = 1):
        """NEMS anchor (the main MEMS section for the phase shifter and tunable directional coupler)

        Args:
            fin_dim: fixed fin dimension (x, y)
            spring_dim: fin dimension (x, y)
            shuttle_dim: shuttle dimension
            straight_connector: straight connector to the fin, box xy (overridden by loop connector)
            tether_connector: tether connector to the fin, xy dim and segment length on the top part of loop
            pos_electrode_dim: positive electrode dimension
            gnd_electrode_dim: negative electrode dimension
            include_support_spring: include extra spring at top for for mechanical support
            tooth_param: (length, width, inter-tooth gap) (suggested: (0.3, 3, 0.15))
            shuttle_stripe_w: design an etch hole shuttle consisting of stripes of width ``shuttle_stripe_w``
                (if 0, do not add a stripped shuttle).
        """
        self.fin_dim = fin_dim
        self.spring_dim = spring_dim
        self.shuttle_dim = shuttle_dim
        self.straight_connector = straight_connector
        self.tether_connector = tether_connector
        self.pos_electrode_dim = pos_electrode_dim
        self.gnd_electrode_dim = gnd_electrode_dim
        self.include_support_spring = include_support_spring
        self.tooth_param = tooth_param
        self.shuttle_stripe_w = shuttle_stripe_w

        patterns, pads, springs, pos_pads, gnd_pads = [], [], [], [], []

        spring_dim = fin_dim if not spring_dim else spring_dim
        shuttle = Box(shuttle_dim).translate()
        comb = None
        connector = shuttle.copy
        if tether_connector is not None and straight_connector is None:
            s = tether_connector
            loop = Pattern(Path(fin_dim[1]).sbend((s[0], s[1])).segment(fin_dim[0]).sbend((s[0], -s[1])))
            straight = Box((shuttle_dim[0], s[-2]))
            straight.align(shuttle).valign(shuttle, bottom=False, opposite=True)
            loop.align(straight).valign(straight, bottom=False, opposite=True)
            connector = Pattern(straight, loop)
            patterns.append(connector)
        elif straight_connector is not None:
            straight = Box(straight_connector)
            fin_connectors = [
                straight.copy.halign(shuttle).valign(shuttle, bottom=False, opposite=True),
                straight.copy.halign(shuttle, left=False,
                                     opposite=False).valign(shuttle, bottom=False, opposite=True),
            ]
            patterns.extend(fin_connectors)
            fat = Box((shuttle_dim[0], straight_connector[1])).align(shuttle).valign(shuttle, opposite=True)
            if include_support_spring and tooth_param is None:
                # this is the mirror image dummy for mechanics
                dummy_fin = Box(fin_dim).align(shuttle).valign(fat, opposite=False, bottom=False)
                patterns.append(dummy_fin)
                end_connector = Pattern(dummy_fin, fat)
            else:
                end_connector = fat

            connector = Pattern(shuttle, *fin_connectors, end_connector)

        if spring_dim is not None:
            top_spring = Box(spring_dim).align(shuttle).valign(shuttle, bottom=True, opposite=True)
            bottom_spring = Box(spring_dim).align(shuttle).valign(shuttle, bottom=False, opposite=True)
            if pos_electrode_dim is not None:
                pos_alignment_pattern = connector if include_support_spring and tooth_param is None else top_spring
                pos_electrode = Box(pos_electrode_dim[:2]).align(top_spring).valign(
                    pos_alignment_pattern, opposite=True).translate(dy=pos_electrode_dim[2])
                patterns.append(pos_electrode)
                pads.append(pos_electrode)
                pos_pads.append(pos_electrode)
                patterns.extend([top_spring, bottom_spring])
                springs.extend([top_spring, bottom_spring])
                if straight_connector is not None:
                    shuttle = Box((shuttle_dim[0], shuttle_dim[1] + straight_connector[1])).valign(shuttle)
            else:
                if tooth_param is not None:
                    raise AttributeError('Must specify pos_electrode_dim if attach_comb is True')
                if straight_connector is not None:
                    shuttle = Box((shuttle_dim[0], shuttle_dim[1] + straight_connector[1])).valign(shuttle)
                pads.append(shuttle.copy)
                pos_pads.append(pads[-1])
            if tooth_param is not None:
                comb = SimpleComb(tooth_param, shuttle_pad_dim=shuttle.size).align(shuttle).valign(shuttle, opposite=True)
                patterns.append(comb)

            patterns.append(shuttle if shuttle_stripe_w == 0 else shuttle.striped(shuttle_stripe_w))

            if gnd_electrode_dim is not None:
                # moving alignment to account for bottom spring
                gnd_electrode_left = Box(gnd_electrode_dim).halign(
                    bottom_spring, opposite=True).valign(bottom_spring)
                gnd_electrode_right = Box(gnd_electrode_dim).halign(
                    bottom_spring, left=False, opposite=True).valign(bottom_spring)
                patterns.extend([gnd_electrode_left, gnd_electrode_right])
                pads.extend([gnd_electrode_left, gnd_electrode_right])
                gnd_pads = [gnd_electrode_left, gnd_electrode_right]

        super(NemsAnchor, self).__init__(*patterns, call_union=False)
        self.pads, self.springs, self.shuttle, self.comb = pads, springs, shuttle, comb
        self.pos_pads, self.gnd_pads = pos_pads, gnd_pads
        shift = [-connector.center[0], -connector.bounds[1]]
        shift[1] -= -tether_connector[-1] if tether_connector is not None and straight_connector is None else 0
        self.reference_patterns = self.pads + [self.shuttle] + self.springs
        self.reference_patterns += [self.comb] if tooth_param is not None else []
        self.translate(*shift)

    def update(self, new: bool = True, **kwargs):
        """Update this class with a new set of parameters using config

        Args:
            new: Return new instance instead of updating
            **kwargs: all of the arguments to update

        Returns:

        """
        config = copy(self.config)
        config.update(kwargs)
        if not new:
            self.__init__(**config)
            return self
        else:
            return NemsAnchor(**config)


class ContactWaveguide(Pattern):
    def __init__(self, waveguide_w: float, length: float, rib_taper_param: Tuple[float, ...],
                 gnd_connector_dim: Optional[Dim2], gnd_contact_dim: Optional[Dim2],
                 rib_etch_grow: float, flipped: bool = False):
        """Contacted waveguide, typically required for photonic MEMS, consisting of a rib brim
        around an (optionally) tapered waveguide.

        Args:
            waveguide_w: waveguide width
            length: length of waveguide
            rib_taper_param: rib taper params
            gnd_contact_dim:
            gnd_connector_dim:
            rib_etch_grow:
            flipped:
        """
        self.waveguide_w = waveguide_w
        self.rib_taper_param = rib_taper_param
        self.length = length
        self.gnd_contact_dim = gnd_contact_dim
        self.gnd_connector_dim = gnd_connector_dim
        self.rib_etch_grow = rib_etch_grow
        self.flipped = flipped

        pads = []

        # TODO(): remove the hidden hardcoding
        # min_brim_l, brim_taper = get_linear_adiabatic(min_width=waveguide_w, max_width=sum(rib_taper_param) + waveguide_w, aggressive=True)
        # if not sum(rib_taper_param) > 0 or not min_brim_l <= length / 2:
        #     raise ValueError(f"Expected sum(rib_taper_param) > 0 and min_brim_l <= length / 2, but got:"
        #                      f"sum(rib_taper_param), min_brim_l, length / 2 = {sum(rib_taper_param), min_brim_l, length / 2}")

        wg = Waveguide(waveguide_w=waveguide_w, length=length)
        rib_brim_etch = Waveguide(waveguide_w=waveguide_w + 2 * rib_etch_grow, length=length, taper_ls=(length / 2,),
                                  taper_params=(rib_taper_param,)).align(wg)
        gnd_connection = Box(gnd_connector_dim).align(wg).valign(wg, bottom=not flipped, opposite=True)
        rib_brim = Waveguide(waveguide_w=waveguide_w, length=length, taper_ls=(length / 2,),
                             taper_params=(rib_taper_param,)).align(wg)
        rib_brim = [Pattern(poly) for poly in (rib_brim.shapely - wg.shapely)]
        rib_brim_etch = [Pattern(poly) for poly in (rib_brim_etch.shapely - wg.shapely)]

        if gnd_contact_dim is not None:
            pad = Box(gnd_contact_dim).align(gnd_connection).valign(gnd_connection, bottom=not flipped, opposite=True)
            pads.append(pad)

        patterns = rib_brim + [wg, gnd_connection] + pads

        super(ContactWaveguide, self).__init__(*patterns)
        self.wg, self.rib_brim, self.pads = wg, rib_brim_etch, pads
        self.rib_brim_w = sum(rib_taper_param) + waveguide_w
        self.reference_patterns = [wg] + rib_brim_etch + pads
        self.port['a0'] = Port(0, 0, -np.pi)
        self.port['b0'] = Port(length, 0)


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


class SimpleComb(Pattern):
    def __init__(self, tooth_param: Dim3, overlap: float = 0, side_align: bool = False,
                 edge_tooth_factor: int = 3, shuttle_pad_dim: Optional[Dim2] = None,
                 pos_pad_dim: Optional[Dim2] = None):
        """

        Args:

            tooth_param: width, height and inter-tooth spacing between teeth
            shuttle_pad_dim: ground electrode dimension
            pos_pad_dim: positive electrode dimension (aligned to gnd pad based on ``side_align`` boolean)
            overlap: overlap of the comb teeth on pos and gnd sides
            edge_tooth_factor: integer number of times bigger the edge teeth are vs other teeth (to avoid snapping)

        """
        self.tooth_param = tooth_param
        self.shuttle_pad_dim = shuttle_pad_dim
        self.pos_pad_dim = pos_pad_dim
        self.overlap = overlap
        self.edge_tooth_factor = edge_tooth_factor
        self.side_align = side_align

        dx_teeth = tooth_param[2] + tooth_param[0]
        num_teeth = int((shuttle_pad_dim[0] - tooth_param[1]) // (2 * dx_teeth))
        if num_teeth <= 0:
            raise ValueError('Electrode dim is too small to hold comb teeth.')
        tooth = Box(tooth_param[:2])
        fat_tooth = Box((tooth_param[0] * edge_tooth_factor, tooth_param[1]))
        upper_teeth = [tooth.copy.translate(dx_teeth * 2 * n) for n in range(num_teeth)]
        upper_teeth = upper_teeth
        lower_teeth = [tooth.copy.translate(dx_teeth * (2 * n + 1)) for n in range(num_teeth - 1)]
        shuttle_comb = Pattern(*upper_teeth)
        pos_comb = Pattern(*lower_teeth)
        edges = [fat_tooth.copy.align(shuttle_comb).halign(shuttle_comb, left=False, opposite=True).translate(tooth_param[0]),
                 fat_tooth.copy.align(shuttle_comb).halign(shuttle_comb, left=True, opposite=True).translate(-tooth_param[0])]
        shuttle_comb = Pattern(shuttle_comb, *edges)
        pos_comb.align(shuttle_comb).translate(0, -overlap + tooth_param[1])
        comb = Pattern(shuttle_comb, pos_comb)
        patterns = [comb]

        if shuttle_pad_dim is not None and pos_pad_dim is not None:
            shuttle_pad = Box(shuttle_pad_dim)
            pos_pad = Box(pos_pad_dim)
            comb.align(shuttle_pad).valign(shuttle_pad, opposite=True, bottom=True)
            pos_pad.halign(shuttle_pad, left=True) if side_align else pos_pad.align(shuttle_pad)
            pos_pad.valign(comb, bottom=True, opposite=True)
            patterns += [shuttle_pad, pos_pad]
        else:
            shuttle_pad = pos_pad = None

        super(SimpleComb, self).__init__(*patterns)
        self.reference_patterns = patterns
        self.shuttle_pad = shuttle_pad
        self.pos_pad = pos_pad
        self.shuttle_comb = shuttle_comb
        self.pos_comb = pos_comb
        if pos_pad is not None:
            self.port['pos'] = Port(pos_pad.center[0], pos_pad.bounds[3], np.pi / 2)

    def clearout(self, buffer: float = 1):
        bounding_pattern = Pattern(*self.reference_patterns[:-1])
        size = (bounding_pattern.size[0] + buffer, bounding_pattern.size[1])
        return Box(size).align(bounding_pattern)


class LateralNemsFull(Multilayer):
    def __init__(self, device: Union[LateralNemsPS, LateralNemsTDC], anchor: NemsAnchor,
                 gnd_via: Via, pos_via: Via, trace_w: float,
                 pos_box_w: float, gnd_box_h: float, clearout_dim: Dim2, dope_grow: float, dope_expand: float,
                 ridge: str, rib: str, shuttle_dope: str,
                 spring_dope: str, pad_dope: str, pos_metal: str, gnd_metal: str,
                 clearout_layer: str, clearout_etch_stop_layer: str, clearout_etch_stop_grow: float,
                 dual_drive_metal: bool = False):
        """Full multilayer NEMS design assuming positive and ground pads defined in silicon layer

        Args:
            device: phase shifter or tunable directional coupler
            anchor: top anchor (None in pull-in case)
            gnd_via: gnd ``Via`` connection
            pos_via: pos ``Via`` connection
            trace_w: trace width
            pos_box_w: Extension for the positive box
            gnd_box_h: Extension for the negative box
            clearout_dim: clearout_w, clearout_h_added (relative to box height matched at edges of pos electrodes)
            ridge: ridge layer
            rib: rib layer
            shuttle_dope: shuttle dope layer
            spring_dope: spring dope layer
            pad_dope: pad dope layer
            pos_metal: pos terminal layer
            gnd_metal: gnd terminal layer
            clearout_layer: clearout layer
            clearout_etch_stop_layer: clearout etch stop layer
            clearout_etch_stop_grow: grow the etch stop layer around the clearout region to protect sidewall
            dual_drive_metal: allow for dual drive control
        """
        device_name = 'tdc' if 'interaction_l' in device.__dict__ else 'ps'
        self.trace_w = trace_w
        self.ridge = ridge
        self.rib = rib
        self.pos_metal = pos_metal
        self.gnd_metal = gnd_metal
        self.shuttle_dope = shuttle_dope
        self.spring_dope = spring_dope
        self.pad_dope = pad_dope
        self.pos_box_w = pos_box_w
        self.gnd_box_h = gnd_box_h
        self.dope_grow = dope_grow
        self.dope_expand = dope_expand
        self.clearout_dim = clearout_dim
        self.clearout_layer = clearout_layer
        self.clearout_etch_stop_layer = clearout_etch_stop_layer
        self.clearout_etch_stop_grow = clearout_etch_stop_grow
        self.dual_drive_metal = dual_drive_metal

        self.config = copy(self.__dict__)
        self.config.update({
            device_name: device.config,
            'anchor': anchor.config,
            'pos_via': pos_via.config,
            'gnd_via': gnd_via.config,
        })

        top = anchor.copy.to(device.port['fin0'])
        bot = anchor.copy.to(device.port['fin1'])
        full = Pattern(top, bot, device)
        vias = []
        device_pads = device.pads if device.pads is not None else []
        dopes = [s.expand(dope_expand).dope(shuttle_dope, dope_grow)
                 for s in [top.shuttle, bot.shuttle] if shuttle_dope is not None] + \
                [s.expand(dope_expand).dope(spring_dope, dope_grow)
                 for s in top.springs + bot.springs if spring_dope is not None] + \
                [s.expand(dope_expand).dope(pad_dope, dope_grow)
                 for s in top.pads + bot.pads + device_pads if pad_dope is not None]
        metals = []
        port = {}
        gnd_pads = []
        if gnd_metal is not None:
            if top.gnd_pads:
                gnd_pads = top.gnd_pads + bot.gnd_pads
            elif device_name == 'ps' and anchor.gnd_electrode_dim is None:
                if not device.gnd_pads:
                    raise ValueError('No ground pads available...')
                gnd_pads = device.gnd_pads
        if gnd_pads:
            gnd = Pattern(*gnd_pads)
            gnd_box = Box((gnd.size[0], gnd.size[1] + 2 * gnd_box_h)).hollow(trace_w).align(gnd)
            metals.append((gnd_box, gnd_metal))
            vias.extend(sum([gnd_via.copy.align(pad).pattern_to_layer for pad in gnd_pads], []))
            port['gnd_l'] = Port(gnd_box.bounds[0] + trace_w / 2, gnd_box.bounds[3], np.pi / 2)
            port['gnd_r'] = Port(gnd_box.bounds[2] - trace_w / 2, gnd_box.bounds[3], np.pi / 2)
        if top.pos_pads and pos_metal is not None:
            pos_pads = top.pos_pads + bot.pos_pads
            pos = Pattern(*pos_pads)
            if dual_drive_metal:
                pos_box = Box((pos.size[0] + 2 * pos_box_w, pos.size[1])).u(trace_w).align(pos)
            else:
                pos_box = Box((pos.size[0] + 2 * pos_box_w, pos.size[1])).hollow(trace_w).align(pos)
            metals.append((pos_box, pos_metal))
            vias.extend(sum([pos_via.copy.align(pad).pattern_to_layer for pad in pos_pads], []))
            port['pos_l'] = Port(pos_box.bounds[0], pos_box.center[1], -np.pi)
            port['pos_b'] = Port(pos_box.center[0], pos_box.bounds[1], -np.pi / 2)
            port['pos_r'] = Port(pos_box.bounds[2], pos_box.center[1], 0)
            clearout_h = pos_pads[0].bounds[1] - pos_pads[1].bounds[3]
            clearout = full.clearout_box(clearout_layer, clearout_etch_stop_layer, (clearout_dim[0],
                                                                                    clearout_h + clearout_dim[1]))
        else:
            clearout = full.clearout_box(clearout_layer, clearout_etch_stop_layer, clearout_dim)
        # TODO(sunil): make a name attribute for each pattern instead?
        if device_name == 'tdc':
            gnd_pads = device.gnd_wg_pads
            gnd = Pattern(*gnd_pads)
            gnd_box = Box((gnd.size[0], gnd.size[1])).hollow(trace_w).align(gnd)
            metals.append((gnd_box, gnd_metal))
            vias.extend(sum([gnd_via.copy.align(pad).pattern_to_layer for pad in gnd_pads], []))
            if anchor.gnd_electrode_dim is None:
                port['gnd_l'] = Port(gnd_box.bounds[0] + trace_w / 2, gnd_box.bounds[3], np.pi / 2)
                port['gnd_r'] = Port(gnd_box.bounds[2] - trace_w / 2, gnd_box.bounds[3], np.pi / 2)

        rib_brim = [(rb, rib) for rb in device.rib_brim if rib is not None]
        super(LateralNemsFull, self).__init__([(full, ridge)] + clearout + rib_brim + vias + dopes + metals)
        self.port = port
        self.port.update(device.port)

    @classmethod
    def from_config(cls, config):
        """Initialize via configuration dictionary (useful for importing from a JSON file)

        Args:
            config:

        Returns:

        """
        return cls(**_handle_nems_config(config))

    def update(self, new: bool = True, **kwargs):
        """Update this config with a new set of parameters

        Args:
            new: Return new instance instead of updating
            **kwargs: all of the arguments to update

        Returns:

        """
        config = copy(self.config)
        config.update(kwargs)
        if 'tdc' in kwargs and 'ps' in config:
            del config['ps']
        if 'ps' in kwargs and 'tdc' in config:
            del config['tdc']
        if not new:
            self.__init__(**_handle_nems_config(config))
            return self
        else:
            return LateralNemsFull(**_handle_nems_config(config))


class NemsMillerNode(Multilayer):
    def __init__(self, waveguide_w: float, upper_interaction_l: float, lower_interaction_l: float,
                 gap_w: float, bend_radius: float, upper_bend_extension: float, lower_bend_extension: float,
                 ps_comb: SimpleComb, tdc_comb: SimpleComb, comb_wg: ContactWaveguide, gnd_wg: ContactWaveguide,
                 pos_via: Via, gnd_via: Via, tdc_pad_dim: Dim4, ps_clearout_dim: Dim2,
                 ps_spring_dim: Dim2, tdc_spring_dim: Dim2, ps_shuttle_w: float, tdc_shuttle_w: float, end_l: float,
                 trace_w: float, connector_dim: Dim2, ridge: str, rib: str, dope: str, pos_metal: str, gnd_metal: str,
                 clearout_buffer_w: float, clearout_layer: str, clearout_etch_stop_layer: str,
                 clearout_etch_stop_grow: float, dope_grow: float, dope_expand: float):
        self.waveguide_w = waveguide_w
        self.upper_interaction_l = upper_interaction_l
        self.lower_interaction_l = lower_interaction_l
        self.bend_radius = bend_radius
        self.upper_bend_extension = upper_bend_extension
        self.lower_bend_extension = lower_bend_extension
        self.gap_w = gap_w
        self.tdc_pad_dim = tdc_pad_dim
        self.connector_dim = connector_dim
        self.pos_via = pos_via
        self.gnd_via = gnd_via
        self.trace_w = trace_w
        self.ridge = ridge
        self.rib = rib
        self.dope = dope
        self.pos_metal = pos_metal
        self.gnd_metal = gnd_metal
        self.clearout_layer = clearout_layer
        self.clearout_etch_stop_layer = clearout_etch_stop_layer
        self.dope_grow = dope_grow
        self.dope_expand = dope_expand
        self.clearout_etch_stop_grow = clearout_etch_stop_grow
        self.ps_spring_dim = ps_spring_dim
        self.tdc_spring_dim = tdc_spring_dim
        self.end_l = end_l

        self.config = copy(self.__dict__)
        self.config.update({
            'ps_comb': ps_comb.config,
            'tdc_comb': tdc_comb.config,
            'comb_wg': comb_wg.config,
            'gnd_wg': gnd_wg.config,
            'pos_via': pos_via.config,
            'gnd_via': gnd_via.config,
        })

        lower_bend_height = 2 * bend_radius + lower_bend_extension
        upper_bend_height = 2 * bend_radius + upper_bend_extension
        interport_w = waveguide_w + upper_bend_height + lower_bend_height + gap_w

        if not upper_interaction_l <= lower_interaction_l:
            raise ValueError("Require upper_interaction_l <= lower_interaction_l by convention.")

        lower_path = Path(waveguide_w).dc((bend_radius, lower_bend_height), lower_interaction_l, use_radius=True)
        upper_path = Path(waveguide_w).dc((bend_radius, upper_bend_height), upper_interaction_l,
                                          (lower_interaction_l - upper_interaction_l) / 2,
                                          inverted=True, use_radius=True)
        upper_path.translate(dx=0, dy=interport_w)

        dc = Pattern(lower_path, upper_path)
        ridge_patterns = [dc]

        # ps comb drive attachment

        wg = Waveguide(waveguide_w, ps_comb.shuttle_pad.size[0])
        ps_comb_1 = comb_wg.copy.halign(wg)
        ps_comb_2 = comb_wg.copy.halign(wg, left=False)
        ps_comb_connector = Pattern(ps_comb_1, ps_comb_2)
        ps_comb_rib_etch = Pattern(*ps_comb_1.rib_brim, *ps_comb_2.rib_brim)
        ps_comb.align(ps_comb_connector, ps_comb.shuttle_pad).valign(ps_comb_connector, opposite=True)

        # tdc comb drive attachment

        wg = Waveguide(waveguide_w, tdc_comb.shuttle_pad.size[0])
        tdc_comb_1 = comb_wg.copy.halign(wg)
        tdc_comb_2 = comb_wg.copy.halign(wg, left=False)
        tdc_comb_connector = Pattern(tdc_comb_1, tdc_comb_2)
        tdc_comb_rib_etch = Pattern(*tdc_comb_1.rib_brim, *tdc_comb_2.rib_brim)
        tdc_comb.align(tdc_comb_connector, tdc_comb.shuttle_pad).valign(tdc_comb_connector, opposite=True)

        # clamped flexures
        ps_connector_dim = (1, upper_interaction_l + 2 * bend_radius - waveguide_w)
        tdc_connector_dim = (1, tdc_shuttle_w + bend_radius / 2)  # TODO(sunil): fix dis

        # comb drive definitions

        ps_comb_drive = Multilayer([(ps_comb_connector, ridge), (ps_comb_rib_etch, rib), (ps_comb, ridge)] +
                                   pos_via.copy.align(ps_comb.pos_pad).pattern_to_layer +
                                   [(ps_comb.pos_pad.copy, pos_metal),
                                    (ps_comb.clearout(), clearout_layer),
                                    (ps_comb.clearout().offset(clearout_etch_stop_grow), clearout_etch_stop_layer),
                                    Box.bbox(ps_comb).expand(dope_expand).dope(dope, dope_grow)
                                    ])
        ps_connect_port = Port(bend_radius + (lower_interaction_l - upper_interaction_l) / 2,
                               interport_w - bend_radius - upper_bend_extension / 2 - ps_comb.shuttle_pad.size[0] / 2,
                               np.pi / 2)
        ps_comb_drives = [ps_comb_drive.copy.to(ps_connect_port),
                          ps_comb_drive.copy.flip().to(
                              ps_connect_port).translate(bend_radius * 2 + upper_interaction_l)]

        # TODO(sunil): bad decomposition here... these should be multilayers!
        tdc_comb_drive = Multilayer([(tdc_comb_connector, ridge), (tdc_comb_rib_etch, rib), (tdc_comb, ridge)] +
                                    pos_via.copy.align(tdc_comb.pos_pad).pattern_to_layer +
                                    [(tdc_comb.pos_pad.copy, pos_metal),
                                     (tdc_comb.clearout(), clearout_layer),
                                     (tdc_comb.clearout().offset(clearout_etch_stop_grow), clearout_etch_stop_layer),
                                     Box.bbox(tdc_comb).expand(dope_expand).dope(dope, dope_grow)])
        tdc_connect_port = Port(2 * bend_radius + tdc_spring_dim[0], lower_bend_height, 0)
        tdc_comb_drives = [tdc_comb_drive.copy.to(tdc_connect_port),
                           tdc_comb_drive.copy.flip(horiz=True).to(tdc_connect_port).translate(
                               lower_interaction_l - tdc_spring_dim[0] * 2)]

        comb_drive_p2l = sum([cd.pattern_to_layer for cd in ps_comb_drives + tdc_comb_drives], [])
        ps_flexure = Box((ps_comb.pos_pad.size[0] - gnd_wg.length + tdc_connector_dim[0], ps_shuttle_w)).flexure(
            (upper_bend_extension, ps_spring_dim[1]), ps_connector_dim
        ).rotate(90).align(ps_comb_drives[0]).halign(
            bend_radius + (lower_interaction_l - upper_interaction_l) / 2 + waveguide_w / 2,
            opposite=True)
        tdc_flexure = Box((lower_interaction_l - 2 * tdc_spring_dim[0] - gnd_wg.length, tdc_shuttle_w)).flexure(
            (lower_interaction_l + bend_radius, ps_spring_dim[1]), tdc_connector_dim, False).align(
            dc).valign(lower_bend_height - waveguide_w / 2, opposite=True, bottom=False)
        ridge_patterns += [ps_flexure, tdc_flexure]

        # ground waveguide connections

        gnd_wgs = [gnd_wg.copy.to(Port(0, 0, -np.pi)), gnd_wg.copy.flip().to(Port(0, interport_w, -np.pi)),
                   gnd_wg.copy.to(Port(dc.size[0], interport_w)), gnd_wg.copy.flip().to(Port(dc.size[0], 0))]
        gnd_wg_rib_etch = [(Pattern(*gwg.rib_brim), rib) for gwg in gnd_wgs]
        gnd_wg_pattern = Pattern(*gnd_wgs)

        ridge_patterns += gnd_wgs

        gnd_trace = Box(gnd_wg_pattern.size).align(gnd_wg_pattern).hollow(trace_w)
        gnd_vias = sum([gnd_via.copy.align(gwg.pads[0]).pattern_to_layer for gwg in gnd_wgs], [])

        ps_flexure_clearout = Box((upper_interaction_l + 2 * bend_radius +
                                   2 * comb_wg.gnd_connector_dim[
                                       1] + 2 * comb_wg.rib_brim_w + waveguide_w + clearout_buffer_w,
                                   ps_comb.pos_pad.size[0])).align(ps_flexure)
        ps_clearout = Box((ps_shuttle_w + clearout_buffer_w, ps_spring_dim[0])).align(ps_flexure)
        tdc_clearout = Box((lower_interaction_l - bend_radius + ps_clearout_dim[0],
                            lower_bend_height + bend_radius + comb_wg.rib_brim_w / 2 + waveguide_w / 2
                            )).align(dc).valign(0)
        wg_clearout_1 = Box((upper_bend_extension, comb_wg.rib_brim_w + clearout_buffer_w)).rotate(90).align(
            (bend_radius + (lower_interaction_l - upper_interaction_l) / 2,
             interport_w - bend_radius - upper_bend_extension / 2))
        wg_clearout_2 = wg_clearout_1.copy.translate(bend_radius * 2 + upper_interaction_l)

        tdc_pad_bbox = Pattern(tdc_comb_drives[0].layer_to_pattern[ridge], tdc_comb_drives[1].layer_to_pattern[ridge])
        pos_trace = Box((dc.size[0] - 2 * trace_w, tdc_pad_bbox.bounds[3] + trace_w + gnd_wg.size[1])).align(
            dc).valign(tdc_pad_bbox, bottom=False).hollow(trace_w)
        pos_trace = pos_trace.difference(
            Box((tdc_pad_bbox.size[0], 2 * trace_w)).align(pos_trace).valign(pos_trace, bottom=False))

        clearout = Pattern(ps_flexure_clearout, ps_clearout, tdc_clearout, wg_clearout_1, wg_clearout_2)
        super(NemsMillerNode, self).__init__([(Pattern(*ridge_patterns, call_union=False), ridge),
                                              (gnd_trace, gnd_metal)]
                                             + gnd_wg_rib_etch + gnd_vias + comb_drive_p2l +
                                             [(clearout, clearout_layer),
                                              (clearout.offset(clearout_etch_stop_grow), clearout_etch_stop_layer),
                                              (pos_trace, pos_metal)]
                                             )
        self.port['gnd_l'] = Port(gnd_trace.bounds[0] + trace_w / 2, gnd_trace.bounds[3], np.pi / 2)
        self.port['gnd_r'] = Port(gnd_trace.bounds[2] - trace_w / 2, gnd_trace.bounds[3], np.pi / 2)
        self.port['pos_l'] = ps_comb_drives[0].port['pos']
        self.port['pos_r'] = ps_comb_drives[1].port['pos']
        self.port['pos_c'] = Port(pos_trace.bounds[0], pos_trace.center[1], np.pi)
        self.port['a0'] = Port(-gnd_wg.size[0], 0, -np.pi)
        self.port['a1'] = Port(-gnd_wg.size[0], interport_w, -np.pi)
        self.port['b0'] = Port(dc.size[0] + gnd_wg.size[0], 0)
        self.port['b1'] = Port(dc.size[0] + gnd_wg.size[0], interport_w)
        self.interport_w = interport_w
        self.gnd_wg_rib_etch = gnd_wg_rib_etch


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


def _handle_nems_config(config):
    anchor = NemsAnchor(**config['anchor']) if isinstance(config['anchor'], dict) else config['anchor']
    pos_via = Via(**config['pos_via']) if isinstance(config['pos_via'], dict) else config['pos_via']
    gnd_via = Via(**config['gnd_via']) if isinstance(config['gnd_via'], dict) else config['gnd_via']
    for key in ('anchor', 'pos_via', 'gnd_via'):
        del config[key]
    if 'tdc' in config:
        device = LateralNemsTDC(**config['tdc']) if isinstance(config['tdc'], dict) else config['tdc']
        del config['tdc']
        # TODO(sunil): can't have both ps and tdc in the config, should be a warning.
        if 'ps' in config:
            del config['ps']
    elif 'ps' in config:
        device = LateralNemsPS(**config['ps']) if isinstance(config['ps'], dict) else config['ps']
        del config['ps']
    else:
        raise AttributeError('Config not supported')
    return {
        'device': device,
        'anchor': anchor,
        'pos_via': pos_via,
        'gnd_via': gnd_via,
        **config
    }
