import itertools

import nazca as nd
from datetime import date
from dphox.aim import *
from dphox.layout import NazcaLayout

chip = NazcaLayout(
    passive_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35a_passive.gds',
    waveguides_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35_waveguides.gds',
    active_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35a_active.gds',
)

# #Please leave this so Nate can run this quickly
# chip = NazcaLayout(
#     passive_filepath='../../../20200819_sjby_aim_run/APSUNY_v35a_passive.gds',
#     waveguides_filepath='../../../20200819_sjby_aim_run/APSUNY_v35_waveguides.gds',
#     active_filepath='../../../20200819_sjby_aim_run/APSUNY_v35a_active.gds',
# )

# chip params

chip_h = 1973  # total height allowed by Xian
chip_w = 12000
perimeter_w = 50  # along the perimeter, include dicing trench of this width
edge_shift_dim = (-56, 20)  # shift entire perimeter box in xy
mesh_chiplet_x = 30
test_chiplet_x = 7670
chiplet_divider_x = 7530

# component params

n_pads_eu = (344, 12)
n_pads_bp = (69, 3)
n_test = 17
dc_radius = 15
aggressive_dc_radius = 5
pdk_dc_radius = 25
sep = 30
gnd_length = 15
standard_grating_interport = 127
mesh_layer_x = 450

# testing params

waveguide_w = 0.48
wg_filler = 15
test_interport_w = 50
test_gap_w = 0.3
test_gap_w_aggressive = 0.15
test_gap_w_short = 0.25
test_gap_w_invdes = 0.4
test_gap_w_invdes_bb = 0.98
test_bend_dim = test_interport_w / 2 - test_gap_w / 2 - waveguide_w / 2
test_bend_dim_short = test_interport_w / 2 - test_gap_w_short / 2 - waveguide_w / 2
test_bend_dim_aggressive = test_interport_w / 2 - test_gap_w_aggressive / 2 - waveguide_w / 2
test_bend_dim_invdes = test_interport_w / 2 - test_gap_w_invdes / 2 - waveguide_w / 2
test_bend_dim_invdes_bb = test_interport_w / 2 - test_gap_w_invdes_bb / 2 - waveguide_w / 2
test_tdc_interport_w = 50
test_tdc_interaction_l = 100
test_tdc_interaction_short_l = 50
miller_node_interport_w = 88.75
miller_node_bend_dim = miller_node_interport_w / 2 - test_gap_w / 2 - waveguide_w / 2
pull_in_phaseshift_l = 50
test_tdc_radius = 10
test_tdc_bend_dim = test_tdc_interport_w / 2 - test_gap_w / 2 - waveguide_w / 2
mesh_interport_w = 50
mesh_phaseshift_l = 90
tether_phaseshift_l = 75
tether_interaction_l = 100
detector_route_loop = (20, 30, 40)  # height, length, relative starting x for loops around detectors
tapline_x_start = 600
# x for the 8 taplines, numpy gives errors for some reason, so need to use raw python
tapline_x = [tapline_x_start + x for x in
             [0, 400, 700, 1000, 1400, 1800, 2090, 2400]]
tapline_y = 162  # y for the taplines
grating_array_xy = (600, 125)

# spacing of test array probe pads
test_pad_x = [tapline_x[0] - 80, tapline_x[1] - 80, tapline_x[2] - 250, tapline_x[3] - 190,
              tapline_x[4] - 80, tapline_x[5] - 372, tapline_x[6] - 260, tapline_x[7] - 280]

# bond pad (testing)

left_bp_x = 100
right_bp_x = 3070
test_pad_y = 185
test_bp_w = 212
via_y = -770

# Basic components

print('Compiling basic components...')

dc = AIMDC().nazca_cell('dc', layer='seam')
dc_short = AIMDC(bend_dim=(aggressive_dc_radius, test_bend_dim_short), gap_w=test_gap_w_short,
                 interaction_l=22).nazca_cell('dc_short', layer='seam')
dc_aggressive = AIMDC(bend_dim=(aggressive_dc_radius, test_bend_dim_aggressive), gap_w=test_gap_w_aggressive,
                      interaction_l=9).nazca_cell('dc_aggressive', layer='seam')
dc_invdes = AIMDC(bend_dim=(aggressive_dc_radius, test_bend_dim_invdes), gap_w=test_gap_w_invdes,
                  interaction_l=5, coupler_boundary_taper_ls=(1,),
                  coupler_boundary_taper=(cubic_taper(-0.16),)).replace(
    Pattern.from_gds('alex_directional.gds')
).nazca_cell('dc_invdes', layer='seam')
dc_invdes_bb1 = AIMDC(bend_dim=(aggressive_dc_radius, test_bend_dim_invdes_bb), gap_w=test_gap_w_invdes_bb,
                      interaction_l=6.08).replace(
    Pattern.from_gds('coupler_480nm_BW100nm.gds')
).nazca_cell('dc_invdes_bb100nm', layer='seam')
dc_invdes_bb2 = AIMDC(bend_dim=(aggressive_dc_radius, test_bend_dim_invdes_bb), gap_w=test_gap_w_invdes_bb,
                      interaction_l=6.08).replace(
    Pattern.from_gds('coupler_480nm_BW200nm.gds')
).nazca_cell('dc_invdes_bb200nm', layer='seam')
dc_millernode = AIMDC(bend_dim=(aggressive_dc_radius, miller_node_bend_dim)).nazca_cell('dc_millernode', layer='seam')
mesh_dc = chip.pdk_dc(radius=pdk_dc_radius, interport_w=mesh_interport_w)
tap_detector = chip.bidirectional_tap(10, mesh_bend=True)
tdc = pull_apart_full_tdc.nazca_cell('test_tdc')
gnd_wg = chip.gnd_wg()
mesh_ps = chip.device_linked([pull_apart_full_ps, sep, tap_detector])
alignment_mark = chip.alignment_mark()
alignment_mark_small = chip.alignment_mark((50, 5), name='alignment_mark_small')
grating = chip.pdk_cells['cl_band_vertical_coupler_si']
detector = chip.pdk_cells['cl_band_photodetector_digital']

delay_line_50 = chip.delay_line(name='delay_line_50')
delay_line_200 = chip.delay_line(delay_length=200, straight_length=100, flip=True, name='delay_line_200')

print('Compiling mesh components...')

thermal_ps = chip.thermal_ps((tap_detector, sep))
dc_dummy = chip.waveguide(mesh_dc.pin['b0'].x - mesh_dc.pin['a0'].x)
mzi_node_nems = chip.mzi_node(chip.device_array([mesh_ps] * 2, mesh_interport_w, name='nems_double_ps'), mesh_dc,
                              name='nems_mzi')
mzi_node_thermal = chip.mzi_node(chip.device_array([thermal_ps] * 2, mesh_interport_w, name='thermal_double_ps'),
                                 mesh_dc, name='thermal_mzi')
mzi_node_nems_detector = chip.mzi_node(chip.device_array([mesh_ps] * 2, mesh_interport_w, name='nems_double_ps_test'),
                                       mesh_dc,
                                       detector=chip.pdk_cells['cl_band_photodetector_digital'], name='nems_mzi_test')
mzi_node_thermal_detector = chip.mzi_node(chip.device_array([thermal_ps] * 2, mesh_interport_w,
                                                            name='thermal_double_ps_test'),
                                          mesh_dc, detector=chip.pdk_cells['cl_band_photodetector_digital'],
                                          name='thermal_mzi_test')
mzi_dummy_nems = chip.mzi_dummy(mesh_ps, dc_dummy, name='mzi_dummy_nems')
mzi_dummy_thermal = chip.mzi_dummy(thermal_ps, dc_dummy, name='mzi_dummy_thermal')
nems_mesh = chip.triangular_mesh(5, mzi_node_nems, mzi_dummy_nems,
                                 mesh_ps, mesh_interport_w, name='triangular_mesh_nems')
thermal_mesh = chip.triangular_mesh(5, mzi_node_thermal, mzi_dummy_thermal,
                                    thermal_ps, mesh_interport_w, name='triangular_mesh_thermal')

grating_array = chip.grating_array(18, period=127, link_end_gratings_radius=10)

interposer = chip.interposer(
    n=14, waveguide_w=0.48, period=50,
    final_period=standard_grating_interport, radius=50, trombone_radius=10,
    self_coupling_extension_dim=(30, 200),
    with_gratings=True, horiz_dist=200, num_trombones=2
)

bp_array = chip.bond_pad_array(n_pads_bp, stagger_x_frac=0.4)
bp_array_testing = chip.bond_pad_array((2, n_test))
eu_array = chip.eutectic_array(n_pads_eu)
autoroute_4 = chip.autoroute_turn(7, level=2, turn_radius=8,
                                  connector_x=0, connector_y=20,
                                  final_period=18.5, width=4)
autoroute_4_extended = chip.autoroute_turn(7, level=2, turn_radius=8,
                                           connector_x=9, connector_y=28,
                                           final_period=18.5, width=4)
autoroute_4_nems_gnd = chip.autoroute_turn(7, level=2, turn_radius=8,
                                           connector_x=8, connector_y=16,
                                           final_period=18.5, width=4)
autoroute_4_nems_pos = chip.autoroute_turn(7, level=2, turn_radius=8,
                                           connector_x=1, connector_y=28,
                                           final_period=18.5, width=4)
autoroute_8 = chip.autoroute_turn(7, level=2, turn_radius=8,
                                  connector_x=0, connector_y=0,
                                  final_period=18.5, width=8)
autoroute_8_extended = chip.autoroute_turn(7, level=2, turn_radius=8, connector_x=9, connector_y=10,
                                           final_period=18.5, width=8)


# Test structures

# Shortcut to keep same params as default while only changing tapers


def ps_taper(taper_length: float, taper_change: float):
    return dict(
        taper_l=taper_length,
        gap_taper=cubic_taper(taper_change),
        wg_taper=cubic_taper(taper_change)
    )


def tdc_taper(taper_length: float, taper_change: float):
    return dict(
        dc_taper_ls=(taper_length,),
        dc_taper=(cubic_taper(taper_change),),
        beam_taper=(cubic_taper(taper_change),)
    )


'''
Pull-apart phase shifter or PSV3

Motivation: modify the gap of the pull-apart phase shifter
Motivation: reduce the waveguide width to encourage more phase shift per unit length in center
Motivation: modify fin width to change stiffness / phase shift per unit length

'''

print('Defining pull-apart ps structures...')

pull_apart_gap = [pull_apart_full_ps.update(ps=ps_pull_apart.update(gap_w=gap_w))
                  for gap_w in (0.1, 0.15, 0.2)]
pull_apart_taper = [pull_apart_full_ps.update(ps=ps_pull_apart.update(**ps_taper(taper_l, taper_change)))
                    for taper_change in (-0.05, -0.1, -0.15) for taper_l in (20, 30, 40)]
pull_apart_fin = [pull_apart_full_ps.update(ps=ps_pull_apart.update(nanofin_w=nanofin_w))
                  for nanofin_w in (0.15, 0.2, 0.25)]
pull_apart_stiff = [pull_apart_full_ps.update(anchor=pull_apart_anchor.update(spring_dim=spring_dim))
                    for spring_dim in [(100, 0.3), (100, 0.4)]]
pull_apart_ps = [chip.mzi_arms([delay_line_50, ps], [delay_line_200],
                               interport_w=test_interport_w, name=f'pull_apart_{i}')
                 for i, ps in enumerate(pull_apart_gap + pull_apart_taper + pull_apart_fin + pull_apart_stiff)]

'''
Pull-in phase shifter or PSV1

Note: To increase pull-in voltage, phase shift length is made shorter.
Motivation: attempt pull-in phase shifter idea with tapering to reduce pull-in voltage (for better or worse...)
and phase shift length, modify fin width to change phase shift per unit length

'''

print('Defining pull-in ps structures...')

pull_in_gap = [pull_in_full_ps.update(ps=ps_pull_in.update(gap_w=gap_w))
               for gap_w in (0.1, 0.125, 0.15, 0.2, 0.25)]
pull_in_taper = [pull_in_full_ps.update(
    ps=ps_pull_in.update(**ps_taper(taper_l, taper_change))
)
    for taper_change in (-0.05, -0.1, -0.15) for taper_l in (10, 15)
]
pull_in_fin = [pull_in_full_ps.update(ps=ps_pull_in.update(nanofin_w=nanofin_w, gap_w=gap_w))
               for nanofin_w in (0.15, 0.2, 0.25) for gap_w in (0.1, 0.15)]
pull_in_ps = [chip.mzi_arms([delay_line_50, ps], [delay_line_200],
                            interport_w=test_interport_w, name=f'pull_in_{i}')
              for i, ps in enumerate(pull_in_gap + pull_in_taper + pull_in_fin)]

'''
Pull-apart TDC

Motivation: Symmetric TDC requires very small critical dimensions and
the asymmetric case requires a wider gap for mode pertubation and
realistically better care in length but this is a test case
tapers are the only way to reach these aggressive goals
Note: Tapering and dc_gap is MOST important

'''

print('Defining pull-apart tdc structures...')

pull_apart_tdc_devices = [
    pull_apart_full_tdc.update(
        tdc=tdc_pull_apart.update(
            dc_gap_w=gap_w,
            bend_dim=(test_tdc_radius, test_tdc_interport_w / 2 - gap_w / 2 - waveguide_w / 2),
            **tdc_taper(20, taper_change)
        ),
        dual_drive_metal=(gap_w == 0.3)
    )
    for gap_w in (0.100, 0.125, .150, 0.300)
    for taper_change in (0, -0.16, -0.32, -0.52)
]
pull_apart_tdc_devices += [pull_apart_full_tdc.update(
    tdc=tdc_pull_apart.update(
        interaction_l=test_tdc_interaction_short_l,
        dc_gap_w=0.125, bend_dim=(test_tdc_radius, test_tdc_interport_w / 2 - 0.125 / 2 - waveguide_w / 2),
        **tdc_taper(20, -0.52)
    ),
    anchor=pull_apart_anchor.update(pos_electrode_dim=(test_tdc_interaction_short_l, 4, 0.5),
                                    spring_dim=(test_tdc_interaction_short_l, 0.22),
                                    fin_dim=(test_tdc_interaction_short_l, 0.15)),
    clearout_dim=(test_tdc_interaction_short_l, 0.3),
    pos_box_w=9
)]
pull_apart_tdc = [dev.nazca_cell(f'pull_apart_tdc_{i}') for i, dev in enumerate(pull_apart_tdc_devices)]

'''
Pull-in TDC

Motivation: attempt pull-in TDC with tapering to reduce the beat length of the TDC
Tapering and dc_gap is MOST important

'''

print('Defining pull-in tdc structures...')

pull_in_tdc_devices = [
    pull_in_full_tdc.update(
        tdc=tdc_pull_in.update(
            dc_gap_w=gap_w,
            bend_dim=(test_tdc_radius, test_tdc_interport_w / 2 - gap_w / 2 - waveguide_w / 2),
            **tdc_taper(20, taper_change)
        ),
        dual_drive_metal=(gap_w == 0.3)
    )
    for gap_w in (0.100, 0.125, 0.150, 0.300)
    for taper_change in (0, -0.16, -0.32, -0.52)
]
pull_in_tdc_devices += [pull_in_full_tdc.update(
    tdc=tdc_pull_in.update(
        interaction_l=test_tdc_interaction_short_l,
        dc_gap_w=0.125, bend_dim=(test_tdc_radius, test_tdc_interport_w / 2 - 0.125 / 2 - waveguide_w / 2),
        **tdc_taper(20, -0.52)
    ),
    anchor=pull_in_anchor.update(shuttle_dim=(test_tdc_interaction_short_l, 5)),
    clearout_dim=(test_tdc_interaction_short_l, 0.1)
)]
pull_in_tdc = [dev.nazca_cell(f'pull_in_tdc_{i}') for i, dev in enumerate(pull_in_tdc_devices)]

'''
VIP Test structures

Motivation: Test structures necessary for reference measurements

'''
print('Defining VIP structures...')

delay_arms = chip.mzi_arms([delay_line_50, gnd_wg, 60, gnd_wg],
                           [delay_line_200],
                           interport_w=test_interport_w,
                           name='bare_mzi_arms')
delay_arms_gnded = chip.mzi_arms([delay_line_50, gnd_wg, 60, gnd_wg],
                                 [delay_line_200],
                                 interport_w=test_interport_w,
                                 name='bare_mzi_arms_gnded')

vip_column = []
for dev, cell_name in zip((dc, dc_short, dc_invdes, dc_aggressive), ('ref_dc', 'ref_dc_short',
                                                                     'ref_dc_invdes', 'ref_dc_aggressive')):
    with nd.Cell(name=cell_name) as cell:
        dc_r = dev.put()
        dc_r.raise_pins(['a0', 'a1', 'b0', 'b1'])
    vip_column.append(cell)

vip_column += [
    chip.mzi_node(delay_arms_gnded, dc, include_input_ps=False, name='bare_mzi_gnded'),
    chip.mzi_node(delay_arms, dc, include_input_ps=False, name='bare_mzi')
]


# Testing bend Radii 10,5,2.5,1

def bend_exp(name='bends_1_1'):
    bend_radius = float(name.split('_')[-2])
    delay_length = 4 * np.pi * bend_radius if 3 * np.pi * bend_radius > 50 else 50
    straight_length = delay_length / 2 if 3 * np.pi * bend_radius > 50 else 25
    with nd.Cell(name=name) as bend_exp:
        first_dc = dc.put()
        delay_line = chip.delay_line(delay_length=delay_length, straight_length=straight_length,
                                     bend_radius=bend_radius, name=f'delay_line_{name}')
        l_arm = [delay_line for _ in range(int(name.split('_')[-1]))]
        mzi_arms = chip.mzi_arms(l_arm, [wg_filler, ],
                                 interport_w=test_interport_w,
                                 name=f'bare_mzi_arms_{name}').put(first_dc.pin['b0'])
        nd.Pin('a0').put(first_dc.pin['a0'])
        nd.Pin('a1').put(first_dc.pin['a1'])
        mzi_arms.raise_pins(['b0', 'b1'])
    return bend_exp


bend_exp_names = [f'bends_{br}_{i}' for i in [2, 4, 8] for br in [1, 2.5, 5]]
bend_exp_names += [f'bends_10_{i}' for i in [2, 3]]
vip_column += [bend_exp(name=bend_exp_name) for bend_exp_name in bend_exp_names]

'''
Tether test structures

Motivation: Tether the phase shifting block/fin instead of using a bending fin
Tradeoff is that we can decouple the photonics from the MEMS (no dependence on fin spring constant),
but we lose extra mechanical support and are susceptible to warping/cantilever effects (which motivates this test).

'''

print('Defining tether structures...')

tether = [
             tether_full_ps.update(
                 ps=tether_ps.update(phaseshift_l=psl, **ps_taper(taper_l, taper_change)),
                 anchor=tether_anchor_ps.update(
                     spring_dim=(psl + 10, 0.22),
                     pos_electrode_dim=(psl, 4, 0.5),
                     fin_dim=(psl, 0.22)
                 ),
                 clearout_dim=(psl + 5, 0.3),
             )
             for psl in (60, 80) for taper_l, taper_change in ((5, -0.05), (10, -0.1), (15, -0.1))
         ] + [
             tether_full_tdc.update(
                 tdc=tether_tdc.update(interaction_l=il, **tdc_taper(taper_l, taper_change)),
                 anchor=tether_anchor_tdc.update(
                     spring_dim=(il + 5, 0.22),
                     pos_electrode_dim=(il - 5, 4, 0.5),
                     fin_dim=(il, 0.3),
                 ),
                 clearout_dim=(il + 5, 0.3),
             )
             for il in (80, 100) for taper_l, taper_change in
             ((10, -0.1), (15, -0.1), (20, -0.16), (20, -0.32), (20, -0.52))
         ] + [
             tether_full_tdc.update(tdc=tether_tdc.update(interaction_l=test_tdc_interaction_short_l),
                                    anchor=tether_anchor_tdc.update(
                                        spring_dim=(test_tdc_interaction_short_l + 12, 0.22),
                                        fin_dim=(test_tdc_interaction_short_l, 0.22),
                                        pos_electrode_dim=(test_tdc_interaction_short_l, 4, 0.5)
                                    ),
                                    clearout_dim=(test_tdc_interaction_short_l + 5, 0.3),
                                    pos_box_w=12,
                                    )
         ]

tether_column = [chip.mzi_arms([dev], [1], name=f'tether_{i}') if i < 6 else dev.nazca_cell(f'tether_{i}')
                 for i, dev in enumerate(tether)]
tether_dcs = [dc if i < 6 else None for i in range(n_test)]

'''
Aggressive test structures (two columns)

Motivation: Change the couplers to be shorter in this stack and test the resulting MZIs (for use in future runs)
This includes inv design and aggressive, broadband directional couplers

'''

print('Defining aggressive structures...')

tether_ps_safe = tether_full_ps.update(
    ps=tether_ps.update(phaseshift_l=80),
    anchor=tether_anchor_tdc.update(
        spring_dim=(80 + 10, 0.22),
        pos_electrode_dim=(75, 4, 0.4),
        fin_dim=(80, 0.3),
        shuttle_dim=(10, 1.5),
        shuttle_stripe_w=0
    ),
    clearout_dim=(85, 0.3)
)

aggressive = [
                 tether_full_ps.update(ps=tether_ps.update(taper_l=10,
                                                           boundary_taper=cubic_taper(-0.25),
                                                           wg_taper=cubic_taper(-0.4),
                                                           gap_taper=cubic_taper(-0.4),
                                                           ))  # slot
             ] + [tether_full_comb_ps] + [tether_ps_safe] * 2 + [pull_in_full_ps] * 2 + [tether_ps_safe] * 2 + [
                 tether_full_ps.update(
                     ps=tether_ps.update(phaseshift_l=psl, **ps_taper(taper_l, taper_change)),
                     anchor=tether_anchor_tdc.update(
                         spring_dim=(psl + 10, 0.22),
                         pos_electrode_dim=(psl - 5, 4, 0.4),
                         fin_dim=(psl, 0.3),
                         shuttle_dim=(10, 1.5),
                         shuttle_stripe_w=0
                     ),
                     clearout_dim=(psl + 5, 0.3)
                 )
                 for psl in (50, 70) for taper_l, taper_change in ((10, -0.05), (15, -0.1), (20, -0.1))
             ] + [
                 tether_full_tdc.update(tdc=tether_tdc.update(**tdc_taper(taper_l, taper_change)))
                 for taper_l, taper_change in ((20, -0.32), (20, -0.42), (20, -0.52))
             ]
aggressive_column = [chip.mzi_arms([dev], [1], name=f'aggressive_{i}') if i < 14
                     else dev.nazca_cell(f'aggressive_{i}') for i, dev in enumerate(aggressive)]

aggressive_column_dcs = [dc_short] * 2 \
                        + [dc_aggressive, dc_invdes, dc_aggressive, dc_invdes, dc_invdes_bb1, dc_invdes_bb2] \
                        + [dc_short] * 6 + [None] * 3

'''
Extreme phase shifters

Motivation: 
Stiff pull-in phase shifter (resonance)
Stiff pull-in phase shifter thicker fins (higher pull-in)
More inv design and aggressive coupler variations

'''

extreme = [
              pull_in_full_ps.update(
                  ps=ps_pull_in.update(phaseshift_l=psl, **ps_taper(taper_l, taper_change)),
                  anchor=pull_in_anchor.update(shuttle_dim=(psl, 3), spring_dim=(psl, 0.22)),
                  clearout_dim=(psl, 0.3)
              )
              for psl in (30, 40) for taper_l, taper_change in ((5, -0.1), (10, -0.1), (15, -0.1))
          ] + [
              pull_in_full_ps.update(
                  ps=ps_pull_in.update(phaseshift_l=psl, nanofin_w=nanofin_w),
                  anchor=pull_in_anchor.update(shuttle_dim=(psl, 3), spring_dim=(psl, 0.22)),
                  clearout_dim=(psl, 0.3)
              )
              for psl in (20, 30) for nanofin_w in (0.2, 0.25, 0.3)
          ] + [
              tether_full_ps.update(
                  ps=tether_ps.update(phaseshift_l=psl),
                  anchor=tether_anchor_ps.update(
                      spring_dim=(psl + 10, 0.22),
                      pos_electrode_dim=(psl - 5, 4, 0.3),
                      fin_dim=(psl, 0.15),
                      shuttle_dim=(5, 1.5),
                      shuttle_stripe_w=0
                  ),
                  clearout_dim=(psl + 5, 0.3)
              ) for psl in (15, 30)
          ] + [
              tether_full_tdc.update(
                  ps=tether_tdc.update(interaction_l=il, dc_gap_w=0.125,
                                       bend_dim=(
                                           test_tdc_radius, test_tdc_interport_w / 2 - 0.125 / 2 - waveguide_w / 2), ),
                  anchor=tether_anchor_tdc.update(
                      spring_dim=(il, 0.22),
                      pos_electrode_dim=(il - 5, 4, 0.4),
                      fin_dim=(il, 0.3),
                      shuttle_dim=(10, 1.5),
                      shuttle_stripe_w=0
                  ),
                  clearout_dim=(il, 0.3)
              ) for il in (15, 20, 25)
          ]
extreme_column = [chip.mzi_arms([dev], [1], name=f'extreme_{i}') if i < 14
                  else dev.nazca_cell(f'extreme_{i}') for i, dev in enumerate(aggressive)]

extreme_column_dcs = [dc_aggressive] * 10 + [dc_short] * 4 + [None] * 3

# testing tap lines
testing_tap_line = chip.tap_line(n_test, name='tapline')
testing_tap_line_tdc = chip.tap_line(n_test, inter_wg_dist=200, name='tapline_tdc')
testing_tap_line_aggressive = chip.tap_line(n_test, inter_wg_dist=290, name='tapline_aggressive')
testing_tap_line_extreme = chip.tap_line(n_test, inter_wg_dist=270, name='tapline_extreme')

test_columns = []


# test structures between the meshes

def autoroute_node_detector(p1, n2, n1, p2,
                            r1=8, r2=4, r3=4, r4=8,
                            s1=27, s2=19, s3=15, s4=33,
                            a1=90, a2=-90, a3=90, a4=-90):
    chip.v1_via_4.put(p1)
    chip.m1_ic.bend(r1, a1).put(p1)
    chip.m1_ic.strt(s1).put()
    chip.v1_via_4.put(n2)
    chip.m1_ic.bend(r2, a2).put(n2)
    chip.m1_ic.strt(s2).put()
    chip.m2_ic.bend(r3, a3).put(n1)
    chip.m2_ic.strt(s3).put()
    chip.m2_ic.bend(r4, a4).put(p2)
    chip.m2_ic.strt(s4).put()


# test structure grid
for col, ps_column in enumerate((pull_apart_ps, pull_in_ps)):
    test_columns.append(chip.test_column(
        ps_column, testing_tap_line, f'ps_{col}', autoroute_node_detector, dc, left_pad_orientation=False
    ))

for col, tdc_column in enumerate((pull_apart_tdc, pull_in_tdc)):
    test_columns.append(
        chip.test_column(tdc_column, testing_tap_line_tdc, f'tdc_{col}', autoroute_node_detector)
    )

test_columns.append(
    chip.test_column(vip_column, testing_tap_line, f'vip', autoroute_node_detector)
)

test_columns.append(
    chip.test_column(tether_column, testing_tap_line, f'tether', autoroute_node_detector, dc=tether_dcs)
)

# TODO(sunil): dummy, replace with more variations!
test_columns.append(
    chip.test_column(extreme_column, testing_tap_line_extreme, f'extreme', autoroute_node_detector,
                     dc=extreme_column_dcs)
)

test_columns.append(
    chip.test_column(aggressive_column, testing_tap_line_aggressive, f'aggressive', autoroute_node_detector,
                     dc=aggressive_column_dcs)
)

# test structures between the meshes


middle_mesh_pull_apart = [
    chip.mzi_node(chip.mzi_arms([ps], [1], interport_w=mesh_interport_w, name=f'middle_mesh_pull_apart_ps_{i}'),
                  dc, include_input_ps=False,
                  name=f'mzi_middle_mesh_pull_apart_ps_{i}') for i, ps in enumerate((pull_apart_full_ps,
                                                                                     pull_apart_full_ps.update(
                                                                                         ps=ps_pull_apart.update(
                                                                                             **ps_taper(-0.05, 30))
                                                                                     )))
]

middle_mesh_pull_in = [
    chip.mzi_node(chip.mzi_arms([delay_line_50, ps],
                                [delay_line_200],
                                interport_w=mesh_interport_w, name=f'middle_mesh_pull_in_ps_{i}'),
                  dc, include_input_ps=False,
                  name=f'mzi_middle_mesh_pull_in_ps_{i}') for i, ps in enumerate((pull_in_full_ps,
                                                                                  pull_in_full_ps.update(
                                                                                      ps=ps_pull_in.update(
                                                                                          **ps_taper(-0.05, 10))
                                                                                  )))
]

middle_mesh_tdc = [pull_apart_full_tdc.nazca_cell('middle_mesh_pull_apart_tdc'),
                   pull_in_full_tdc.nazca_cell('middle_mesh_pull_in_tdc'),
                   pull_apart_full_tdc.update(
                       tdc=tdc_pull_apart.update(**tdc_taper(-0.2, 40))
                   ).nazca_cell('middle_mesh_pull_apart_taper_tdc')]

with nd.Cell('miller_node_full') as miller_node_full:
    miller_node_cell = miller_node.nazca_cell('miller_node').put()
    miller_node_cell.raise_pins()
    dcm = dc_millernode.put(miller_node_cell.pin['a0'], flip=True)
    nd.Pin('a0').put(dcm.pin['b0'])
    nd.Pin('a1').put(dcm.pin['b1'])

middle_mesh_test_structures = middle_mesh_pull_apart + middle_mesh_pull_in + middle_mesh_tdc + [miller_node_full]

# gnd pad (testing side)
with nd.Cell('gnd_pad') as gnd_pad:
    chip.ml_ic.strt(width=1716, length=60).put()

with nd.Cell('test_pad') as test_pad:
    for i in range(17):
        chip.ml_ic.strt(width=90, length=60).put(0, 101 * i)

chiplet_divider = chip.dice_box((100, 1973))
chip_horiz_dice = chip.dice_box((chip_w, perimeter_w))
chip_vert_dice = chip.dice_box((perimeter_w, chip_h))

print('Setting up mesh chiplet...')

# Chip construction
with nd.Cell('mesh_chiplet') as mesh_chiplet:
    nems = nems_mesh.put(0, 750, flip=True)
    thermal = thermal_mesh.put(0, 1000)
    input_interposer = interposer.put(thermal.pin['a4'])
    output_interposer = interposer.put(thermal.pin['b4'], flip=True)
    mzi_node_thermal = mzi_node_thermal_detector.put(input_interposer.pin['a6'])

    # routing code for the meshes
    bp_array_nems = bp_array.put(-180, -40)
    eu_array_nems = eu_array.put(-180, 200)
    bp_array_thermal = bp_array.put(-180, 1778, flip=True)
    eu_array_thermal = eu_array.put(-180, 1538, flip=True)

    # all ranges are [inclusive, exclusive) as is convention in python range() method
    # TODO(someone not Nate): double check the remapped mesh connections
    # add more when test structures are added in between the meshes
    eu_bp_port_ranges_m1 = [(0, 2),  # layer 0 /input
                            (5, 7),  # 2 lines into this layer
                            (15, 20),  # backward mesh output detector layer
                            (28, 30), (38, 40),  # layer 1
                            (50, 53), (61, 64),  # 3 lines into this layer
                            (73, 76), (84, 87),  # layer 2
                            (95, 99), (107, 111),  # 4 lines into this layer
                            (118, 122), (130, 134),  # layer 3
                            (140, 145), (153, 158),  # 5 lines into this layer
                            (163, 168), (176, 181),  # layer 4
                            (186, 191), (199, 204),  # 5 lines into this layer
                            (210, 214), (222, 226),  # layer 5
                            (233, 237), (245, 249),  # 4 lines into this layer
                            (257, 260), (268, 271),  # layer 6
                            (280, 283), (291, 294),  # 3 lines into this layer
                            (304, 306), (314, 316),  # layer 7
                            (327, 329),  # 2 lines into this layer
                            (337, 342)]  # forward mesh output detector layer

    eu_bp_m2_idx = np.hstack([np.arange(*r) for r in eu_bp_port_ranges_m1])

    used_connections = set()
    counter = 0
    for idx in eu_bp_m2_idx:
        pin_x = eu_array_nems.pin[f'o{idx}'].x
        closest = (0, 0)
        closest_dist = np.inf
        for i, j in itertools.product(range(n_pads_bp[0]), range(3)):
            dist = np.abs(bp_array_nems.pin[f'u{i},{j}'].x - pin_x)
            if dist < closest_dist and (i, j) not in used_connections:
                closest = (i, j)
                closest_dist = dist
        used_connections.add(closest)
        i, j = closest
        chip.m2_ic.strt(100 * (2 - j), width=8).put(bp_array_nems.pin[f'u{i},{j}'])
        chip.m2_ic.bend_strt_bend_p2p(eu_array_nems.pin[f'o{idx}'], radius=8, width=8).put()
        chip.m2_ic.strt(100 * (2 - j), width=8).put(bp_array_thermal.pin[f'u{i},{j}'])
        chip.m2_ic.bend_strt_bend_p2p(eu_array_thermal.pin[f'o{idx}'], radius=8, width=8).put()

    # Added the Shared GND and Anode Vbias lines
    eu_bp_port_blocks_m2 = [
        (7, 11), (11, 15), (30, 34), (34, 38), (53, 57),
        (57, 61), (76, 80), (80, 84), (99, 103), (103, 107),
        (122, 126), (126, 130), (145, 149), (149, 153), (168, 172),
        (172, 176), (191, 195), (195, 199), (214, 218), (218, 222),
        (237, 241), (241, 245), (260, 264), (264, 268), (283, 287),
        (287, 291), (306, 310), (310, 314), (329, 333), (333, 337)
    ]

    eu_bp_port_blocks = [np.arange(*r) for r in eu_bp_port_blocks_m2]

    for block in eu_bp_port_blocks:
        closest_pair = (0, (0, 0))
        closest_dist = np.inf
        start_idx = None
        for idx in block:
            if start_idx is None:
                start_idx = idx
            pin_x = eu_array_nems.pin[f'o{idx}'].x
            for i, j in itertools.product(range(n_pads_bp[0]), range(3)):
                dist = np.abs(bp_array_nems.pin[f'u{i},{j}'].x - pin_x)
                if dist < closest_dist and (i, j) not in used_connections:
                    closest_pair = (idx, (i, j))
                    closest_dist = dist
        # connect all pins in a block
        chip.m2_ic.strt_p2p(eu_array_nems.pin[f'o{start_idx}'], eu_array_nems.pin[f'o{idx}'], width=12).put()
        chip.m1_ic.strt_p2p(eu_array_nems.pin[f'o{start_idx}'], eu_array_nems.pin[f'o{idx}'], width=12).put()
        chip.m2_ic.strt_p2p(eu_array_thermal.pin[f'o{start_idx}'], eu_array_thermal.pin[f'o{idx}'], width=12).put()
        chip.m1_ic.strt_p2p(eu_array_thermal.pin[f'o{start_idx}'], eu_array_thermal.pin[f'o{idx}'], width=12).put()
        chip.m2_ic.strt_p2p(eu_array_nems.pin[f'i{start_idx}'], eu_array_nems.pin[f'i{idx}'], width=12).put()
        chip.m1_ic.strt_p2p(eu_array_nems.pin[f'i{start_idx}'], eu_array_nems.pin[f'i{idx}'], width=12).put()
        chip.m2_ic.strt_p2p(eu_array_thermal.pin[f'i{start_idx}'], eu_array_thermal.pin[f'i{idx}'], width=12).put()
        chip.m1_ic.strt_p2p(eu_array_thermal.pin[f'i{start_idx}'], eu_array_thermal.pin[f'i{idx}'], width=12).put()

        used_connections.add(closest_pair[1])
        i, j = closest_pair[1]
        idx = closest_pair[0]
        chip.m2_ic.strt(100 * (2 - j), width=8).put(bp_array_nems.pin[f'u{i},{j}'])
        chip.m2_ic.bend_strt_bend_p2p(eu_array_nems.pin[f'o{idx}'], radius=4, width=8).put()
        chip.m2_ic.strt(100 * (2 - j), width=19).put(bp_array_thermal.pin[f'u{i},{j}'])
        chip.m2_ic.bend_strt_bend_p2p(eu_array_thermal.pin[f'o{idx}'], radius=4, width=19).put()

    '''
    Full Test pin list:
    20, 21, 23, 24
    43, 44, 66, 67
    69, 89, 90
    92, 112, 113
    115, 135, 136
    138, 158, 159 double check these pins
    161, 181, 182 double check these pins
    184, 204, 205 double check these pins
    207, 227, 228 double check these pins
    319, 320
    342, 343
    '''
    eu_bp_port_tests_m1 = [
        (20, 22), (23, 25),  # test MZI # moved to test routing
        (43, 45), (66, 68),  # test MZI # moved to test routing
        (69, 70), (89, 91),
        (92, 93), (112, 114),
        (115, 116), (135, 137),
        (207, 208), (227, 229),
        (230, 231), (250, 252),
        (253, 254), (273, 275),
        (276, 277), (296, 298),
        (299, 301),
        (319, 321), (322, 324),
        (342, 344)
    ]

    eu_bp_test_m1_idx = np.hstack([np.arange(*r) for r in eu_bp_port_tests_m1])
    pad_assignments = [(2, 0), (5, 0), (4, 2), (6, 1),
                       (8, 1), (8, 2), (13, 0), (13, 1),
                       (13, 2), (17, 2), (18, 1), (18, 2),
                       (23, 0), (22, 2), (25, 0), (27, 1), (27, 2),
                       (41, 1), (46, 0), (45, 2),
                       (48, 0), (50, 1), (51, 0), (50, 2),
                       (54, 1), (54, 2), (55, 0),
                       (59, 0), (59, 1), (59, 2), (60, 1),
                       (63, 2), (64, 0), (64, 1), (64, 2),
                       (66, 1), (67, 0)
                       ]

    left_most = np.NINF
    for idx, assignment in zip(eu_bp_test_m1_idx, pad_assignments):
        i, j = assignment
        chip.v1_via_8.put(bp_array_nems.pin[f'u{i},{j}'])
        chip.m1_ic.bend_strt_bend_p2p(eu_array_nems.pin[f'o{idx}'], radius=4, width=8).put()
        chip.v1_via_8.put()
        chip.v1_via_8.put(bp_array_thermal.pin[f'u{i},{j}'])
        chip.m1_ic.bend_strt_bend_p2p(eu_array_thermal.pin[f'o{idx}'], radius=4, width=8).put()
        chip.v1_via_8.put()

    pin_num = 0
    num_ps_middle_mesh = len(middle_mesh_pull_apart + middle_mesh_pull_in)

    mzi_node_nems = mzi_node_nems_detector.put(input_interposer.pin['a7'], flip=True)
    alignment_mark.put(-500, 0)
    alignment_mark.put(7000, 0)
    alignment_mark.put(7000, 1700)
    alignment_mark.put(-500, 1700)

    mesh_ts_idx = 0
    middle_mesh_ts_layers = [3, 4, 5, 9, 10, 11, 12, 13]
    miller_node_pin = None

    for layer in range(15):
        # autoroute
        autoroute_nems_gnd = autoroute_4_nems_gnd.put(layer * mesh_layer_x + 8.05, 536.77, flop=True)
        autoroute_nems_pos = autoroute_4_nems_pos.put(layer * mesh_layer_x - 10, 550, flop=True)
        autoroute_nems_cathode = autoroute_4.put(layer * mesh_layer_x + 178, 542)
        autoroute_nems_anode = autoroute_4_extended.put(layer * mesh_layer_x + 178, 550)
        for i in range(7):
            chip.v1_via_4.put(autoroute_nems_gnd.pin[f'a{i}'], flop=True)
        autoroute_thermal_gnd = autoroute_8.put(layer * mesh_layer_x, 1228, flop=True, flip=True)
        autoroute_thermal_pos = autoroute_8_extended.put(layer * mesh_layer_x, 1218, flop=True, flip=True)
        autoroute_therm_cathode = autoroute_4.put(layer * mesh_layer_x + 178, 1208, flip=True)
        autoroute_therm_anode = autoroute_4_extended.put(layer * mesh_layer_x + 178, 1200, flip=True)

        if layer in middle_mesh_ts_layers:
            # mid-mesh test structures
            test_structure = middle_mesh_test_structures[mesh_ts_idx]
            shift_x = 140 * (mesh_ts_idx >= num_ps_middle_mesh) - (layer == 13) * 90
            len_x = 0 * (mesh_ts_idx >= num_ps_middle_mesh)
            # Nate: added a quick 0, -20 hack to fix drc
            ts = test_structure.put(1250 + shift_x + mesh_layer_x * (layer - 3),
                                    output_interposer.pin['a7'].y + 20 - 4 * (layer == 13), flip=True)
            chip.m1_ic.bend_strt_bend_p2p(ts.pin['gnd_l'], autoroute_nems_gnd.pin['b5'], radius=4).put()
            chip.v1_via_4.put()
            if layer == 13:
                chip.m2_ic.bend_strt_bend_p2p(ts.pin['pos_c'], autoroute_nems_pos.pin['a6'], radius=8).put()
                chip.m2_ic.bend_strt_bend_p2p(ts.pin['pos_l'], autoroute_nems_pos.pin['a5'], radius=8).put()
                x = chip.m2_ic.strt(60).put(ts.pin['pos_r'])
                chip.v1_via_4.put()
                miller_node_pin = x.pin['b0']
            else:
                chip.m2_ic.bend_strt_bend_p2p(ts.pin['pos_l'], autoroute_nems_pos.pin['a6'], radius=4).put()
            # chip.v1_via_4.put(flop=True)
            chip.waveguide_ic.strt(shift_x + len_x).put(ts.pin['a0'])
            chip.waveguide_ic.bend(radius=7, angle=-90).put()
            chip.waveguide_ic.strt(15 + 4 * (layer == 13)).put()
            bend = chip.waveguide_ic.bend(radius=7, angle=-90).put()
            grating.put(bend.pin['b0'].x, bend.pin['b0'].y, -90)
            chip.waveguide_ic.strt(7 + shift_x + len_x).put(ts.pin['a1'])
            chip.waveguide_ic.bend(radius=7, angle=-90).put()
            chip.waveguide_ic.strt(113 + 44 * (layer == 13)).put()
            chip.waveguide_ic.bend(radius=7, angle=-90).put()
            bend = chip.waveguide_ic.strt(7).put()
            grating.put(bend.pin['b0'].x, bend.pin['b0'].y, -90)
            d1 = detector.put(ts.pin['b0'], flip=True)
            d2 = detector.put(ts.pin['b1'])
            if layer == 13:
                chip.m1_ic.bend_strt_bend_p2p(d2.pin['n'], autoroute_nems_anode.pin[f'b5'], radius=4).put()
                chip.v1_via_4.put()
                chip.m2_ic.bend_strt_bend_p2p(d2.pin['p'], autoroute_nems_cathode.pin[f'a5'], radius=4).put()
            else:
                chip.m2_ic.bend_strt_bend_p2p(d2.pin['n'], autoroute_nems_anode.pin[f'b5'], radius=4).put()
                chip.m1_ic.bend_strt_bend_p2p(d2.pin['p'], autoroute_nems_cathode.pin[f'b5'], radius=4).put()
                chip.v1_via_4.put()
            chip.m2_ic.bend_strt_bend_p2p(d1.pin['n'], autoroute_nems_anode.pin[f'b6'], radius=4).put()
            chip.m1_ic.bend_strt_bend_p2p(d1.pin['p'], autoroute_nems_cathode.pin[f'b6'], radius=4).put()
            chip.v1_via_4.put()
            chip.v1_via_4.put(d1.pin['p'])
            chip.v1_via_4.put(d2.pin['n' if layer == 13 else 'p'])
            mesh_ts_idx += 1


        def extra_bend_p2p(p1, p2, radius, angle, strt, use_m1=False):
            if use_m1:
                chip.v1_via_4.put(p1)
            ic = chip.m1_ic if use_m1 else chip.m2_ic
            ic.bend(radius, angle).put(p1)
            ic.strt(strt).put()
            ic.bend_strt_bend_p2p(p2, radius=radius).put()
            if use_m1:
                chip.v1_via_4.put()


        if layer == 2:
            # nems and thermal test node
            chip.m2_ic.bend_strt_bend_p2p(mzi_node_nems.pin['n1'], autoroute_nems_anode.pin['a5'], radius=8).put()
            chip.m2_ic.bend_strt_bend_p2p(mzi_node_nems.pin['n2'], autoroute_nems_anode.pin['a6'], radius=8).put()
            chip.m1_ic.bend_strt_bend_p2p(mzi_node_nems.pin['p1'], autoroute_nems_cathode.pin['a5'], radius=8).put()
            chip.m2_ic.bend_strt_bend_p2p(mzi_node_nems.pin['p2'], autoroute_nems_cathode.pin['a6'], radius=8).put()
            chip.v1_via_4.put(autoroute_nems_cathode.pin['p1'], flop=True)
            chip.m2_ic.bend_strt_bend_p2p(mzi_node_thermal.pin['n1'], autoroute_therm_anode.pin['a5'], radius=8).put()
            chip.m2_ic.bend_strt_bend_p2p(mzi_node_thermal.pin['n2'], autoroute_therm_anode.pin['a6'], radius=8).put()
            chip.m1_ic.bend_strt_bend_p2p(mzi_node_thermal.pin['p1'], autoroute_therm_cathode.pin['a5'], radius=8).put()
            chip.m2_ic.bend_strt_bend_p2p(mzi_node_thermal.pin['p2'], autoroute_therm_cathode.pin['a6'], radius=8).put()
            chip.v1_via_4.put(autoroute_therm_cathode.pin['p1'], flop=True)

        if layer == 13:
            # mesh DC test
            strt = chip.waveguide_ic.strt(200).put(output_interposer.pin['a5'])
            mdc = mesh_dc.put(strt.pin['b0'])
            d1 = detector.put(mdc.pin['b0'], flip=True)
            d2 = detector.put(mdc.pin['b1'], flip=True)
            extra_bend_p2p(d1.pin['n'], autoroute_therm_anode.pin['a5'], 6, 180, 20)
            extra_bend_p2p(d2.pin['n'], autoroute_therm_anode.pin['a6'], 6, 180, 20)
            extra_bend_p2p(d1.pin['p'], autoroute_therm_cathode.pin['a5'], 3, -180, 20)
            extra_bend_p2p(d2.pin['p'], autoroute_therm_cathode.pin['a6'], 3, -180, 20)

        if layer == 14:
            # mesh TDC test
            dev = tdc.put(output_interposer.pin['a7'])
            chip.waveguide_ic.bend(8, -180).put()
            chip.waveguide_ic.strt(length=10).put()
            d1 = detector.put(flip=True)
            chip.waveguide_ic.bend(8, 180).put(dev.pin['b1'])
            chip.waveguide_ic.strt(length=10).put()
            d2 = detector.put(flip=True)
            chip.m2_ic.bend_strt_bend_p2p(d2.pin['n'], autoroute_nems_anode.pin['a5'], radius=4).put()
            chip.m2_ic.bend_strt_bend_p2p(d1.pin['n'], autoroute_nems_anode.pin['a6'], radius=4).put()
            chip.m2_ic.bend_strt_bend_p2p(d2.pin['p'], autoroute_nems_cathode.pin['a5'], radius=4).put()
            chip.m2_ic.bend_strt_bend_p2p(d1.pin['p'], autoroute_nems_cathode.pin['a6'], radius=4).put()
            chip.m1_ic.bend_strt_bend_p2p(dev.pin['gnd_r'], autoroute_nems_gnd.pin['a5'], radius=4).put()
            chip.m2_ic.bend_strt_bend_p2p(dev.pin['pos_r'], autoroute_nems_pos.pin['a5'], radius=4).put()
            chip.v1_via_4.put(dev.pin['gnd_r'], flop=True)

            # mesh tap test
            test_tap = tap_detector.put(output_interposer.pin['a6'].x - 40,
                                        output_interposer.pin['a6'].y, flip=False)
            d = detector.put(test_tap.pin['a0'])
            extra_bend_p2p(d.pin['n'], autoroute_therm_anode.pin['a5'], 4, -90, 60)
            extra_bend_p2p(d.pin['p'], autoroute_therm_cathode.pin['a5'], 10, -90, 70)

        if layer == 14:  # see miller_node_pin, hacky way to add final connection for the miller node test
            chip.m1_ic.bend_strt_bend_p2p(miller_node_pin, autoroute_nems_pos.pin['a6'], radius=8).put()
            chip.v1_via_4.put()

        for pin_nems, pin_thermal in zip(reversed([autoroute_nems_pos.pin[f'p{n}'] for n in range(7)]),
                                         reversed([autoroute_thermal_pos.pin[f'p{n}'] for n in range(7)])):
            if pin_num < n_pads_eu[0]:
                chip.m2_ic.bend_strt_bend_p2p(pin_nems, eu_array_nems.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.m2_ic.bend_strt_bend_p2p(pin_thermal, eu_array_thermal.pin[f'i{pin_num}'], radius=8, width=8).put()
            pin_num += 1
        for i, pins in enumerate(zip(reversed([autoroute_nems_gnd.pin[f'p{n}'] for n in range(7)]),
                                     reversed([autoroute_thermal_gnd.pin[f'p{n}'] for n in range(7)]))):
            pin_nems, pin_thermal = pins
            if pin_num < n_pads_eu[0]:
                chip.m1_ic.bend_strt_bend_p2p(pin_nems, eu_array_nems.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.v1_via_4.put(pin_nems)
                # extra thick traces to increase surface area
                width = 8 if i <= 1 else 20  # hacky rule to avoid shorting
                chip.m1_ic.bend_strt_bend_p2p(pin_thermal, eu_array_thermal.pin[f'i{pin_num}'], radius=width,
                                              width=width).put()
                chip.v1_via_8.put(pin_thermal)
            pin_num += 1 if i % 2 else 0
        pin_num += 1
        for i, pins in enumerate(zip([autoroute_nems_cathode.pin[f'p{n}'] for n in range(7)],
                                     [autoroute_therm_cathode.pin[f'p{n}'] for n in range(7)])):
            pin_nems, pin_thermal = pins
            if pin_num < n_pads_eu[0]:
                chip.m1_ic.bend_strt_bend_p2p(pin_nems, eu_array_nems.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.v1_via_4.put(pin_nems)
                chip.m1_ic.bend_strt_bend_p2p(pin_thermal, eu_array_thermal.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.v1_via_8.put(pin_thermal)
            pin_num += 1 if i % 2 else 0
        pin_num += 1
        for pin_nems, pin_thermal in zip([autoroute_nems_anode.pin[f'p{n}'] for n in range(7)],
                                         [autoroute_therm_anode.pin[f'p{n}'] for n in range(7)]):
            if pin_num < n_pads_eu[0]:
                chip.m2_ic.bend_strt_bend_p2p(pin_nems, eu_array_nems.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.m2_ic.bend_strt_bend_p2p(pin_thermal, eu_array_thermal.pin[f'i{pin_num}'], radius=8, width=8).put()
            pin_num += 1
        pin_num += 1

print('Setting up test chiplet...')
# Test chiplet layout
with nd.Cell('test_chiplet') as test_chiplet:
    # place test taplines down at non-overlapping locations
    detector_x = []
    ga = grating_array.put(*grating_array_xy, -90)
    gs_list = []
    for i, item in enumerate(zip(tapline_x, test_columns)):
        x, gridsearch = item
        gs = gridsearch.put(x, tapline_y)
        chip.waveguide_ic.bend_strt_bend_p2p(ga.pin[f'a{2 * i + 1}'], gs.pin['out'], radius=10).put()
        chip.waveguide_ic.bend_strt_bend_p2p(ga.pin[f'a{2 * i + 2}'], gs.pin['in'], radius=10).put()
        detector_x.append([gs.pin[f'd{j}'].x for j in range(n_test)])
        gs_list.append(gs)

    # put bond pad arrays on the left and right of the testing area
    bp_array_left = bp_array_testing.put(left_bp_x, test_bp_w)
    bp_array_right = bp_array_testing.put(right_bp_x, test_bp_w)

    # alignment_mark.put(300, 0)
    alignment_mark_small.put(150, 0)
    alignment_mark_small.put(150, 1770)
    alignment_mark_small.put(3120, 1770)
    alignment_mark_small.put(3120, 0)

    # place ground bus
    gnd_pad.put(-15, 995)

    # place test pads
    test_pads = [test_pad.put(x, test_pad_y) for i, x in enumerate(test_pad_x) if i != 4]
    dual_drive_tps = [test_pad.put(tapline_x[3] - 270, test_pad_y),
                      test_pad.put(tapline_x[4] - 375, test_pad_y)]

    for i in range(n_test):

        # detector wire connections
        chip.m2_ic.bend(26, -90).put(bp_array_left.pin[f'u{0},{i}'])
        p = chip.m2_ic.strt(2994).put()
        chip.m1_ic.bend(20, -90).put(bp_array_left.pin[f'u{1},{i}'])
        chip.m1_ic.strt(2900).put()
        chip.m1_ic.bend(28, -90).put(bp_array_right.pin[f'd{1},{i}'])
        chip.m1_ic.strt(2994).put()
        chip.m2_ic.bend(22, -90).put(bp_array_right.pin[f'd{0},{i}'])
        chip.m2_ic.strt(2900).put()
        chip.v1_via_4.put(bp_array_right.pin[f'd{1},{i}'])
        chip.v1_via_4.put(bp_array_left.pin[f'u{1},{i}'])

        # loop ground wires around detectors
        cx = 10
        cy = p.pin['a0'].y - 80
        chip.va_via.put(cx, cy)
        for j, x in enumerate(detector_x):
            chip.m2_ic.strt(x[i] - detector_route_loop[2] - cx).put(cx, cy)
            chip.m2_ic.bend(radius=4, angle=90).put()
            chip.m2_ic.strt(detector_route_loop[0]).put()
            chip.m2_ic.bend(radius=4, angle=-90).put()
            ground_wire = chip.m2_ic.strt(detector_route_loop[1]).put()
            chip.m2_ic.bend(radius=4, angle=-90).put()
            chip.m2_ic.strt(detector_route_loop[0]).put()
            chip.m2_ic.bend(radius=4, angle=90).put()
            cx = nd.cp.x()
            # gnd connection
            if f'gnd{i}' in gs_list[j].pin:
                chip.m2_ic.bend_strt_bend_p2p(gs_list[j].pin[f'gnd{i}'], radius=4).put()
                chip.v1_via_4.put(flop=True)
            # pos electrode connection
            if f'pos{i}' in gs_list[j].pin:
                pin = gs_list[j].pin[f'pos{i}']
                # ensures that this route will not intersect any of the current routes
                offset = ground_wire.pin['a0'].y + 6 - pin.y
                offset = -offset if pin.a == 180 else offset
                chip.m2_ic.sbend(radius=4, offset=offset).put(pin)
                idx = j if j < 4 else j - 1  # hack: ignore vip col
                chip.m2_ic.strt(test_pads[idx].bbox[0] - nd.cp.x()).put(*nd.cp.get_xy(), 0)
                # connect to the pad using via
                chip.va_via.put()
            if (j == 2 or j == 3) and n_test - 5 <= i < n_test - 1:
                pin = gs_list[j].pin[f'pos2_{i}']
                chip.m2_ic.strt(20).put(pin)
                idx = (j - 1) // 2
                chip.m2_ic.bend(radius=4, angle=-90).put()
                chip.m2_ic.strt(dual_drive_tps[idx].bbox[0] - nd.cp.x()).put(*nd.cp.get_xy(), 0)
                chip.va_via.put()

# Final chip layout

print('Assembling final layout including other GDS designs...')
bb_goos_1 = nd.netlist.load_gds('broadband_goos_gratings_1.gds', newcellname='broadband_grating_1')
bb_goos_2 = nd.netlist.load_gds('broadband_goos_gratings_2.gds', newcellname='broadband_grating_2')

with nd.Cell('aim_layout') as aim_layout:
    mesh_chiplet.put(mesh_chiplet_x)
    test_chiplet.put(test_chiplet_x)
    chiplet_divider.put(chiplet_divider_x, -standard_grating_interport + 20)
    bb_goos_1.put(chiplet_divider_x + 350, standard_grating_interport - 3)
    bb_goos_2.put(input_interposer.bbox[0] + chip_w - 535, standard_grating_interport - 3)
    chip_horiz_dice.put(input_interposer.bbox[0] + edge_shift_dim[0],
                        -standard_grating_interport + edge_shift_dim[1] - perimeter_w)
    chip_horiz_dice.put(input_interposer.bbox[0] + edge_shift_dim[0],
                        -standard_grating_interport + edge_shift_dim[1] + chip_h)
    chip_vert_dice.put(input_interposer.bbox[0] + edge_shift_dim[0],
                       -standard_grating_interport + edge_shift_dim[1])
    chip_vert_dice.put(input_interposer.bbox[0] + chip_w - perimeter_w + edge_shift_dim[0],
                       -standard_grating_interport + edge_shift_dim[1])

nd.export_gds(filename=f'aim-layout-{str(date.today())}-submission', topcells=[aim_layout])
# Please leave this so Nate can run this quickly
# nd.export_gds(filename=f'../../../20200819_sjby_aim_run/drc finalization/aim-layout-{str(date.today())}-submission', topcells=[aim_layout])
