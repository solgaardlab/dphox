from .component import *

# Solgaard lab AIM PDK

aim_phaseshift_l = 90
aim_interaction_l = 100

aim_ps = LateralNemsPS(
    waveguide_w=0.48,
    nanofin_w=0.22,
    phaseshift_l=aim_phaseshift_l,
    gap_w=0.10,
    num_taper_evaluations=100,
    gnd_connector=(2, 0.2, 3),
    taper_l=0,
    end_ls=(5,),
    gap_taper=None,
    wg_taper=None,
    boundary_taper=None,
    fin_end_bend_dim=(2, 1),
    end_taper=((0, -0.08),),
    gnd_connector_idx=-1
)

aim_tdc = LateralNemsTDC(
    waveguide_w=0.48,
    nanofin_w=0.22,
    interaction_l=aim_interaction_l,
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
    shuttle_dim=(40, 5),
    pos_electrode_dim=None,
    gnd_electrode_dim=None,
    spring_dim=None,
    straight_connector=(0.25, 1),
    tether_connector=(3, 1, 0.5, 1),
    attach_comb=False,
    include_fin_dummy=True
)

aim_pull_in_full_ps = LateralNemsFull(
    device=aim_ps,
    anchor=aim_pull_in_anchor,
    gnd_via=Via((0.4, 0.4), 0.1, metal='m1am', via='cbam', shape=(4, 4), pitch=1),
    pos_via=Via((0.4, 0.4), 0.1, metal=['m1am', 'm2am'], via=['cbam', 'v1am'], shape=(20, 4), pitch=1),
    trace_w=4,
    pos_box_w=12,
    gnd_box_h=10,
    clearout_dim=(aim_phaseshift_l, 3),
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

aim_pull_in_full_tdc = aim_pull_in_full_ps.update(tdc=aim_tdc,
                                                  clearout_dim=(aim_interaction_l, 3))

aim_pull_apart_full_ps = aim_pull_in_full_ps.update(anchor=aim_pull_apart_anchor,
                                                    clearout_dim=(aim_phaseshift_l, 12.5))

aim_pull_apart_full_tdc = aim_pull_apart_full_ps.update(tdc=aim_tdc,
                                                        clearout_dim=(aim_interaction_l, 12.5))


