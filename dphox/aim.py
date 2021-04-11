from .component import *

# Solgaard lab AIM PDK

waveguide_w = 0.48
phaseshift_l_pull_apart = 90
phaseshift_l_pull_in = 40
interaction_l_pull_apart = 100
interaction_l_pull_in = 50
end_l = 5
tether_phaseshift_l = 75
tether_interaction_l = 100
interport_w = 50
gap_w = 0.3
dc_radius = 15
test_bend_h = (interport_w - gap_w - waveguide_w) / 2


class AIMDC(DC):
    def __init__(self, bend_dim=(dc_radius, test_bend_h), waveguide_w=waveguide_w,
                 gap_w=gap_w, interaction_l=37.8, use_radius=True,
                 coupler_boundary_taper_ls: Tuple[float, ...] = (0,),
                 coupler_boundary_taper: Optional[Tuple[Tuple[float, ...]]] = None
                 ):
        super(AIMDC, self).__init__(bend_dim=bend_dim, waveguide_w=waveguide_w,
                                    gap_w=gap_w, interaction_l=interaction_l, use_radius=use_radius,
                                    coupler_boundary_taper_ls=coupler_boundary_taper_ls,
                                    coupler_boundary_taper=coupler_boundary_taper)


class AIMNemsPS(LateralNemsPS):
    def __init__(self, waveguide_w=0.48, nanofin_w=0.22, phaseshift_l=phaseshift_l_pull_apart,
                 gap_w=0.10, num_taper_evaluations=100, gnd_connector=(2, 0.2, 5),
                 gnd_pad_dim=None, taper_l=0, end_ls=(end_l,), gap_taper=None, wg_taper=None, boundary_taper=None,
                 fin_end_bend_dim=(2, 1), end_taper=((0, -0.08),), gnd_connector_idx=-1, rib_etch_grow=0.1):
        super(AIMNemsPS, self).__init__(waveguide_w=waveguide_w, nanofin_w=nanofin_w, phaseshift_l=phaseshift_l,
                                        gap_w=gap_w, num_taper_evaluations=num_taper_evaluations,
                                        gnd_connector=gnd_connector, taper_l=taper_l, gnd_pad_dim=gnd_pad_dim,
                                        end_ls=end_ls, gap_taper=gap_taper, wg_taper=wg_taper,
                                        boundary_taper=boundary_taper, fin_end_bend_dim=fin_end_bend_dim,
                                        end_taper=end_taper, gnd_connector_idx=gnd_connector_idx,
                                        rib_etch_grow=rib_etch_grow)


class AIMNemsTDC(LateralNemsTDC):
    def __init__(self, waveguide_w=0.48, nanofin_w=0.22, interaction_l=interaction_l_pull_apart,
                 dc_gap_w=0.2, beam_gap_w=0.1, bend_dim=(10, 24.66), gnd_wg=(2, 2, 2, 0.75),
                 use_radius=True, dc_end_l=0, dc_taper_ls=None, dc_taper=None, beam_taper=None, fin_end_bend_dim=(2, 1),
                 rib_etch_grow=0.25):
        super(AIMNemsTDC, self).__init__(waveguide_w=waveguide_w, nanofin_w=nanofin_w, interaction_l=interaction_l,
                                         dc_gap_w=dc_gap_w, beam_gap_w=beam_gap_w, bend_dim=bend_dim, gnd_wg=gnd_wg,
                                         use_radius=use_radius, dc_end_l=dc_end_l, dc_taper_ls=dc_taper_ls,
                                         dc_taper=dc_taper, beam_taper=beam_taper, fin_end_bend_dim=fin_end_bend_dim,
                                         rib_etch_grow=rib_etch_grow)


class AIMNemsActuator(NemsActuator):
    def __init__(self, fin_dim=(100, 0.22), shuttle_dim=(50, 2), spring_dim=None, straight_connector=(0.25, 1),
                 tether_connector=(2, 1, 0.5, 1), pos_electrode_dim=(90, 4, 0.5), gnd_electrode_dim=(3, 4),
                 include_support_spring=True, shuttle_stripe_w=1, tooth_param=None):
        super(AIMNemsActuator, self).__init__(fin_dim=fin_dim, shuttle_dim=shuttle_dim, spring_dim=spring_dim,
                                            straight_connector=straight_connector, tether_connector=tether_connector,
                                            pos_electrode_dim=pos_electrode_dim, gnd_electrode_dim=gnd_electrode_dim,
                                            include_support_spring=include_support_spring,
                                            shuttle_stripe_w=shuttle_stripe_w,
                                            tooth_param=tooth_param)


class AIMNemsFull(LateralNemsFull):
    def __init__(self, device, actuator, clearout_dim,
                 pos_box_w=8, gnd_box_h=0,
                 mid_via=Via((0.4, 0.4), 0.1, metal=['m1am', 'm2am'], via=['cbam', 'v1am'], shape=(2, 2), pitch=1),
                 top_via=Via((3.6, 3.6), (1.0, 1.5), metal=['m2am', 'mlam'], via='vaam'),
                 trace_w=3):
        super(AIMNemsFull, self).__init__(device=device, actuator=actuator, mid_via=mid_via,
                                          top_via=top_via, trace_w=trace_w, pos_box_w=pos_box_w,
                                          gnd_box_h=gnd_box_h, clearout_dim=clearout_dim, clearout_etch_stop_grow=0.5,
                                          dope_expand=0.3, dope_grow=0.1, ridge='seam', rib='ream', shuttle_dope='pdam',
                                          spring_dope='pdam', pad_dope='pppam', pos_metal='mlam',
                                          gnd_metal='m2am', clearout_layer='clearout', clearout_etch_stop_layer='snam')


class AIMMockThermal(ThermalPS):
    def __init__(self, waveguide_w=0.48, ps_w=2, length=80,
                 via=Via((0.4, 0.4), 0.1, metal=['m1am', 'm2am'], via=['v1am'])):
        super(AIMMockThermal, self).__init__(waveguide_w, ps_w, length, via, ridge='ream', ps_layer='m1am')

ps_pull_apart = AIMNemsPS()
ps_pull_in = AIMNemsPS(phaseshift_l=phaseshift_l_pull_in, gnd_pad_dim=(3, 4))
tdc_pull_apart = AIMNemsTDC()
tdc_pull_in = AIMNemsTDC(interaction_l=interaction_l_pull_in)

pull_apart_actuator = AIMNemsActuator()
pull_in_actuator = AIMNemsActuator(
    fin_dim=(50, 0.22), shuttle_dim=(40, 3),
    pos_electrode_dim=None, gnd_electrode_dim=None,
    spring_dim=None, include_support_spring=False, shuttle_stripe_w=0
)
tether_actuator_tdc = AIMNemsActuator(
    spring_dim=(tether_interaction_l + 5, 0.22),
    pos_electrode_dim=(tether_interaction_l - 5, 4, 0.5),
    fin_dim=(tether_interaction_l, 0.4),
    shuttle_dim=(10, 3),
    straight_connector=None,
    include_support_spring=False
)
tether_actuator_ps = AIMNemsActuator(
    shuttle_dim=(10, 3),
    spring_dim=(tether_phaseshift_l + 10, 0.22),
    pos_electrode_dim=(tether_phaseshift_l, 4, 0.5),
    fin_dim=(tether_phaseshift_l, 0.22),
    straight_connector=None,
    include_support_spring=False
)
tether_actuator_ps_comb = tether_actuator_ps.update(tooth_param=(0.3, 2, 0.15, 0.5),
                                                pos_electrode_dim=(tether_phaseshift_l, 4, 3.2),
                                                shuttle_dim=(30, 3))

pull_in_full_ps = AIMNemsFull(device=ps_pull_in, actuator=pull_in_actuator, gnd_box_h=5,
                              clearout_dim=(phaseshift_l_pull_in, 0.3)).update(shuttle_dope=None)
pull_in_full_tdc = AIMNemsFull(device=tdc_pull_in, actuator=pull_in_actuator,
                               clearout_dim=(interaction_l_pull_in, 0.3),
                               gnd_box_h=10, pos_box_w=12).update(shuttle_dope=None)
pull_apart_full_ps = AIMNemsFull(device=ps_pull_apart, actuator=pull_apart_actuator,
                                 clearout_dim=(phaseshift_l_pull_apart, 0.3),
                                 gnd_box_h=10, pos_box_w=18)
pull_apart_full_tdc = AIMNemsFull(device=tdc_pull_apart, actuator=pull_apart_actuator,
                                  clearout_dim=(interaction_l_pull_apart, 0.3),
                                  gnd_box_h=10, pos_box_w=15)

tether_ps = ps_pull_apart.update(
    end_ls=(5, 5),
    end_taper=((0.0,), (0.0, -0.08),),
    gnd_connector_idx=0,
    phaseshift_l=tether_phaseshift_l,
    taper_l=5,
    wg_taper=cubic_taper(-0.05),
    gap_taper=cubic_taper(-0.05)
)

tether_tdc = tdc_pull_apart.update(
    interaction_l=tether_interaction_l,
    dc_end_l=5,
    dc_taper_ls=(5,),
    dc_taper=(cubic_taper(-0.05),),
    beam_taper=(cubic_taper(-0.05),)
)

tether_full_ps = pull_apart_full_ps.update(
    actuator=tether_actuator_ps,
    ps=tether_ps,
    clearout_dim=(tether_phaseshift_l + 5, 0.5),
    gnd_box_h=10.4,
)

tether_full_tdc = pull_apart_full_tdc.update(
    actuator=tether_actuator_tdc,
    tdc=tether_tdc,
    clearout_dim=(tether_interaction_l + 5, 0.5),
    pos_box_w=11,
    gnd_box_h=10.4,
)

tether_full_comb_ps = pull_apart_full_ps.update(
    actuator=tether_actuator_ps_comb,
    ps=tether_ps,
    clearout_dim=(tether_phaseshift_l + 5, 0.5),
    gnd_box_h=25
)

miller_node = NemsMillerNode(
    waveguide_w=0.48, upper_interaction_l=30, lower_interaction_l=120,
    gap_w=0.3, bend_radius=5, upper_bend_extension=58, lower_bend_extension=10,
    tdc_pad_dim=(55, 5, 1, 0.22), connector_dim=(0.1, 0.5),
    ps_comb=SimpleComb(
        tooth_param=(0.3, 3, 0.15),
        shuttle_pad_dim=(20, 3),
        pos_pad_dim=(20, 6),
        edge_tooth_factor=5,
        overlap=0.5
    ),
    tdc_comb=SimpleComb(
        tooth_param=(0.3, 3, 0.15),
        shuttle_pad_dim=(15, 3),
        pos_pad_dim=(15, 6),
        edge_tooth_factor=5,
        overlap=0.5
    ),
    comb_wg=ContactWaveguide(
        waveguide_w=0.48, length=5, gnd_contact_dim=(1, 2),
        rib_taper_param=cubic_taper(1.52), gnd_connector_dim=(1, 2),
        flipped=False, rib_etch_grow=0.25
    ),
    gnd_wg=ContactWaveguide(
        waveguide_w=0.48, length=5, gnd_contact_dim=(3, 3),
        rib_taper_param=cubic_taper(1.52), gnd_connector_dim=(0.5, 2),
        flipped=False, rib_etch_grow=0.25
    ),
    ps_spring_dim=(40, 0.25), tdc_spring_dim=(20, 0.25),
    ps_shuttle_w=25, tdc_shuttle_w=15, clearout_etch_stop_grow=0.5, clearout_buffer_w=2,
    ridge='seam', rib='ream', dope='pppam', comb_dope='pdam', pos_metal='m2am',
    gnd_metal='m1am', clearout_layer='clearout', clearout_etch_stop_layer='snam',
    gnd_via=Via((0.4, 0.4), 0.1, metal=['m1am', 'm2am'], via=['cbam', 'v1am'], shape=(2, 2), pitch=1),
    pos_via=Via((0.4, 0.4), 0.1, metal=['m1am', 'm2am'], via=['cbam', 'v1am'], shape=(15, 2), pitch=1),
    trace_w=3, dope_expand=0.3, dope_grow=0.1, ps_clearout_dim=(4, 1.9), end_l=0
)
