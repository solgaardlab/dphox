from .component import *

# Solgaard lab AIM PDK

aim_phaseshift_l_pull_apart = 90
aim_phaseshift_l_pull_in = 40
aim_interaction_l_pull_apart = 100
aim_interaction_l_pull_in = 50
aim_end_l = 5  # single taper at the end
clearout_h_pull_in = 3
clearout_h_pull_apart = 10
tether_phaseshift_l = 75
tether_interaction_l = 100

aim_ps_pull_apart = LateralNemsPS(
    waveguide_w=0.48,
    nanofin_w=0.22,
    phaseshift_l=aim_phaseshift_l_pull_apart,
    gap_w=0.10,
    num_taper_evaluations=100,
    gnd_connector=(2, 0.2, 3),
    taper_l=0,
    end_ls=(aim_end_l,),
    gap_taper=None,
    wg_taper=None,
    boundary_taper=None,
    fin_end_bend_dim=(2, 1),
    end_taper=((0, -0.08),),
    gnd_connector_idx=-1
)

aim_ps_pull_in = aim_ps_pull_apart.update(phaseshift_l=aim_phaseshift_l_pull_in)

aim_tdc_pull_apart = LateralNemsTDC(
    waveguide_w=0.48,
    nanofin_w=0.22,
    interaction_l=aim_interaction_l_pull_apart,
    dc_gap_w=0.2,
    beam_gap_w=0.1,
    bend_dim=(10, 24.66),
    gnd_wg=(2, 2, 2, 0.75),
    use_radius=True,
    dc_end_l=0,
    dc_taper_ls=None,
    dc_taper=None,
    beam_taper=None,
    fin_end_bend_dim=(2, 1)
)

aim_tdc_pull_in = aim_tdc_pull_apart.update(interaction_l=aim_interaction_l_pull_in)

aim_pull_apart_anchor = NemsAnchor(
    fin_dim=(100, 0.22),
    shuttle_dim=(50, 2),
    spring_dim=None,
    straight_connector=(0.25, 1),
    tether_connector=(3, 1, 0.5, 1),
    pos_electrode_dim=(90, 4, 0.5),
    gnd_electrode_dim=(3, 4),
    attach_comb=False,
    include_fin_dummy=True
)

aim_pull_in_anchor = NemsAnchor(
    fin_dim=(50, 0.22),
    shuttle_dim=(40, 3),
    pos_electrode_dim=None,
    gnd_electrode_dim=None,
    spring_dim=None,
    straight_connector=(0.25, 1),
    tether_connector=(3, 1, 0.5, 1),
    attach_comb=False,
    include_fin_dummy=False
)

aim_pull_in_full_ps = LateralNemsFull(
    device=aim_ps_pull_in,
    anchor=aim_pull_in_anchor,
    gnd_via=Via((0.4, 0.4), 0.1, metal='m1am', via='cbam', shape=(2, 2), pitch=1),
    pos_via=Via((0.4, 0.4), 0.1, metal=['m1am', 'm2am'], via=['cbam', 'v1am'], shape=(20, 2), pitch=1),
    trace_w=4,
    pos_box_w=12,
    gnd_box_h=10,
    clearout_dim=(aim_phaseshift_l_pull_in, clearout_h_pull_in),
    dope_expand=0.25,
    dope_grow=0.1,
    ridge='seam',
    rib='ream',
    shuttle_dope='pdam',
    spring_dope='pdam',
    pad_dope='pppam',
    pos_metal='m2am',
    gnd_metal='m1am',
    clearout_layer='clearout',
    clearout_etch_stop_layer='snam'
)

aim_tether_anchor_tdc = aim_pull_apart_anchor.update(
    fin_dim=(tether_interaction_l, 0.4),
    shuttle_dim=(5, 2),
    spring_dim=(tether_interaction_l + 5, 0.22),
    straight_connector=None,
    tether_connector=(2, 1, 0.5, 1),
    pos_electrode_dim=(tether_interaction_l - 5, 4, 0.5),
    gnd_electrode_dim=(3, 3),
    include_fin_dummy=False
)

aim_tether_anchor_ps = aim_tether_anchor_tdc.update(
    shuttle_dim=(10, 2),
    spring_dim=(tether_phaseshift_l + 10, 0.22),
    pos_electrode_dim=(tether_phaseshift_l, 4, 0.5)
)

aim_pull_in_full_tdc = aim_pull_in_full_ps.update(tdc=aim_tdc_pull_in,
                                                  clearout_dim=(aim_interaction_l_pull_in, clearout_h_pull_in))

aim_pull_apart_full_ps = aim_pull_in_full_ps.update(anchor=aim_pull_apart_anchor,
                                                    ps=aim_ps_pull_apart,
                                                    clearout_dim=(aim_phaseshift_l_pull_apart,
                                                                  clearout_h_pull_apart))

aim_pull_apart_full_tdc = aim_pull_apart_full_ps.update(
    tdc=aim_tdc_pull_apart,
    clearout_dim=(aim_interaction_l_pull_apart - 10,
                  clearout_h_pull_apart))


aim_tether_ps = aim_ps_pull_apart.update(
    end_ls=(5, 5),
    end_taper=((0.0,), (0.0, -0.08),),
    gnd_connector_idx=0,
    phaseshift_l=tether_phaseshift_l,
    taper_l=5,
    wg_taper=cubic_taper(-0.05),
    gap_taper=cubic_taper(-0.05)
)

aim_tether_tdc = aim_tdc_pull_apart.update(
    interaction_l=tether_interaction_l,
    dc_end_l=5,
    dc_taper_ls=(5,),
    dc_taper=(cubic_taper(-0.05),),
    beam_taper=(cubic_taper(-0.05),)
)

aim_tether_full_ps = aim_pull_apart_full_ps.update(
    anchor=aim_tether_anchor_ps,
    ps=aim_tether_ps,
    clearout_dim=(tether_phaseshift_l + 5, clearout_h_pull_apart))

aim_tether_full_tdc = aim_pull_apart_full_tdc.update(
    anchor=aim_tether_anchor_tdc,
    tdc=aim_tether_tdc,
    clearout_dim=(tether_interaction_l - 5, clearout_h_pull_apart))

