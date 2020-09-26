import itertools

import nazca as nd
import numpy as np
from typing import Optional

from dphox.design.aim import AIMNazca
from dphox.design.component import cubic_taper
from datetime import date
from tqdm import tqdm

chip = AIMNazca(
    passive_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35a_passive.gds',
    waveguides_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35_waveguides.gds',
    active_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35a_active.gds',
)

# #Please leave this so Nate can run this quickly
# chip = AIMNazca(
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
n_pads_bp = (70, 3)
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
test_gap_w_invdes = 0.6
test_interaction_l_short = 22
test_interaction_l_invdes = 3.72
test_interaction_l_aggressive = 9
test_bend_dim = test_interport_w / 2 - test_gap_w / 2 - waveguide_w / 2
test_bend_dim_short = test_interport_w / 2 - test_gap_w_short / 2 - waveguide_w / 2
test_bend_dim_aggressive = test_interport_w / 2 - test_gap_w_aggressive / 2 - waveguide_w / 2
test_bend_dim_invdes = test_interport_w / 2 - test_gap_w_invdes / 2 - waveguide_w / 2
test_tdc_interport_w = 50
test_tdc_interaction_l = 100
test_tdc_interaction_l_extr = 50
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
tapline_x = [tapline_x_start + x for x in [0, 400, 700, 1000, 1400, 1800, 2100, 2400]]  # TODO(Nate): update this spacing based final array
tapline_y = 162  # y for the taplines
grating_array_xy = (600, 125)

# spacing of test array  probe pads
test_pad_x = [tapline_x[0] - 80, tapline_x[1] - 80, tapline_x[2] - 250, tapline_x[3] - 250,
              tapline_x[4] - 80, tapline_x[5] - 292, tapline_x[6] - 250, tapline_x[7] - 250]

# bond pad (testing)

left_bp_x = 100
right_bp_x = 3070
test_pad_y = 995
test_bp_w = 212
via_y = -770

# Basic components
# TODO(Nate): Double check the defaults of these parameters

dc = chip.custom_dc(bend_dim=(dc_radius, test_bend_dim))[0]
dc_short = chip.custom_dc(bend_dim=(aggressive_dc_radius, test_bend_dim_short), gap_w=test_gap_w_aggressive,
                          interaction_l=test_interaction_l_aggressive)[0]
dc_aggressive = chip.custom_dc(bend_dim=(aggressive_dc_radius, test_bend_dim_aggressive), gap_w=test_gap_w_aggressive,
                               interaction_l=test_interaction_l_aggressive)[0]
dc_invdes = chip.custom_dc_taper(bend_dim=(aggressive_dc_radius, test_bend_dim_invdes), gap_w=test_gap_w_invdes,
                                 interaction_l=test_interaction_l_invdes)[0]
mesh_dc = chip.pdk_dc(radius=pdk_dc_radius, interport_w=mesh_interport_w)
tap_detector = chip.bidirectional_tap(10, mesh_bend=True)
pull_apart_anchor = chip.nems_anchor()
pull_apart_anchor_comb = chip.nems_anchor(attach_comb=True)
pull_in_anchor = chip.nems_anchor(shuttle_dim=(40, 5), fin_dim=(50, 0.15),
                                  pos_electrode_dim=None, neg_electrode_dim=None)
tdc_anchor = chip.nems_anchor(shuttle_dim=(test_tdc_interaction_l, 5),
                              pos_electrode_dim=None, neg_electrode_dim=None)
tdc = chip.nems_tdc(anchor=tdc_anchor, bend_dim=(test_tdc_radius, test_tdc_bend_dim))
gnd_wg = chip.gnd_wg()
ps = chip.nems_ps(anchor=pull_apart_anchor, tap_sep=(tap_detector, sep))
ps_no_anchor = chip.nems_ps()
alignment_mark = chip.alignment_mark()
alignment_mark_small = chip.alignment_mark((50, 5))
gnd_wg = chip.gnd_wg()
grating = chip.pdk_cells['cl_band_vertical_coupler_si']
detector = chip.pdk_cells['cl_band_photodetector_digital']

delay_line_50 = chip.delay_line()
delay_line_200 = chip.delay_line(delay_length=200, straight_length=100)

# Mesh generation

thermal_ps = chip.thermal_ps((tap_detector, sep))
dc_dummy = chip.waveguide(mesh_dc.pin['b0'].x - mesh_dc.pin['a0'].x)
mzi_node_nems = chip.mzi_node(chip.double_ps(ps, mesh_interport_w, name='nems_double_ps'), mesh_dc)
mzi_node_thermal = chip.mzi_node(chip.double_ps(thermal_ps, mesh_interport_w, name='thermal_double_ps'), mesh_dc)
mzi_node_nems_detector = chip.mzi_node(chip.double_ps(ps, mesh_interport_w, name='nems_double_ps'), mesh_dc,
                                       detector=chip.pdk_cells['cl_band_photodetector_digital'])
mzi_node_thermal_detector = chip.mzi_node(chip.double_ps(thermal_ps, mesh_interport_w, name='thermal_double_ps'),
                                          mesh_dc, detector=chip.pdk_cells['cl_band_photodetector_digital'])
mzi_dummy_nems = chip.mzi_dummy(ps, dc_dummy)
mzi_dummy_thermal = chip.mzi_dummy(thermal_ps, dc_dummy)
nems_mesh = chip.triangular_mesh(5, mzi_node_nems, mzi_dummy_nems, ps, mesh_interport_w)
thermal_mesh = chip.triangular_mesh(5, mzi_node_thermal, mzi_dummy_thermal, thermal_ps, mesh_interport_w)

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


def pull_apart_taper_dict(taper_change: float, taper_length: float):
    return dict(
        taper_l=taper_length,
        gap_taper=cubic_taper(taper_change),
        wg_taper=cubic_taper(taper_change)
    )


def pull_in_dict(phaseshift_l: float = pull_in_phaseshift_l, taper_change: float = None, taper_length: float = None,
                 clearout_dim: Optional[float] = None):
    # DOne: modify this to taper the pull-in fin adiabatically using rib_brim_taper
    # Became irrelevant with new ps class structure
    clearout_dim = (phaseshift_l - 10, 3) if clearout_dim is None else clearout_dim
    if taper_change is None or taper_length is None:
        return dict(
            phaseshift_l=phaseshift_l, clearout_box_dim=clearout_dim, gnd_connector=None
        )
    else:
        return dict(
            phaseshift_l=phaseshift_l, clearout_box_dim=clearout_dim,
            taper_l=taper_length, gap_taper=cubic_taper(taper_change),
            wg_taper=cubic_taper(taper_change), gnd_connector=None
        )


def taper_dict_tdc(taper_change: float, taper_length: float):
    return dict(
        dc_taper_ls=(taper_length,), dc_taper=(cubic_taper(taper_change),), beam_taper=(cubic_taper(taper_change),)
    )


'''
Pull-apart phase shifter or PSV3
'''

# Motivation: modify the gap of the pull-apart phase shifter

pull_apart_gap = [
    chip.mzi_arms([delay_line_50, chip.nems_ps(gap_w=gap_w, anchor=pull_apart_anchor, name=f'ps_gap_{gap_w}')],
                  [delay_line_200],
                  interport_w=test_interport_w,
                  name=f'pull_apart_gap_{gap_w}')
    for gap_w in (0.1, 0.15, 0.2, 0.25)]

# Motivation: reduce the waveguide width to encourage more phase shift per unit length in center
pull_apart_taper = [
    chip.mzi_arms([delay_line_50, chip.nems_ps(anchor=pull_apart_anchor, **pull_apart_taper_dict(taper_change, taper_length), name=f'ps_taper_{taper_change}_{taper_length}')],
                  [delay_line_200],
                  interport_w=test_interport_w,
                  name=f'pull_apart_taper_{taper_change}_{taper_length}')
    for taper_change in (-0.05, -0.1, -0.15) for taper_length in (20, 30, 40)]

# Motivation: modify fin width to change stiffness / phase shift per unit length
pull_apart_fin = [
    chip.mzi_arms([delay_line_50, chip.nems_ps(anchor=pull_apart_anchor, nanofin_w=nanofin_w, name=f'ps_fin_{nanofin_w}')],
                  [delay_line_200],
                  interport_w=test_interport_w,
                  name=f'pull_apart_fin_{nanofin_w}')
    for nanofin_w in (0.15, 0.2, 0.22, 0.25)]

'''
Pull-in phase shifter or PSV1
'''

# Motivation: attempt pull-in phase shifter idea with tapering to reduce pull-in voltage (for better or worse...)
# and phase shift length
pull_in_gap = [
    chip.mzi_arms([delay_line_50, chip.nems_ps(anchor=pull_in_anchor, gap_w=gap_w, **pull_in_dict(pull_in_phaseshift_l),
                                               name=f'ps_gap_{gap_w}'), wg_filler, gnd_wg],
                  [delay_line_200, gnd_wg],
                  interport_w=test_interport_w,
                  name=f'pull_in_gap_{gap_w}')
    for gap_w in (0.1, 0.125, 0.15, 0.2, 0.25)]

# Motivation: attempt pull-in phase shifter idea with tapering to reduce pull-in voltage (for better or worse...)
# and phase shift length. To increase pull-in voltage, phase shift length is made shorter.
pull_in_taper = [
    chip.mzi_arms([delay_line_50, chip.nems_ps(anchor=pull_in_anchor, **pull_in_dict(pull_in_phaseshift_l,
                                                                                                         taper_change, taper_length),
                                               name=f'ps_taper_{taper_change}_{taper_length}'), wg_filler, gnd_wg],
                  [delay_line_200, gnd_wg],
                  interport_w=test_interport_w,
                  name=f'pull_in_taper_{taper_change}_{taper_length}')
    for taper_change in (-0.05, -0.1, -0.15) for taper_length in (10, 20)]

# Motivation: attempt pull-in phase shifter idea with modifying fin width / phase shift per unit length
pull_in_fin = [
    chip.mzi_arms([delay_line_50,
                   chip.nems_ps(anchor=pull_in_anchor, nanofin_w=nanofin_w, gap_w=gap_w, **pull_in_dict(pull_in_phaseshift_l),
                                name=f'ps_fin_{nanofin_w}_{gap_w}'), wg_filler, gnd_wg],
                  [delay_line_200, gnd_wg],
                  interport_w=test_interport_w,
                  name=f'pull_in_fin_{nanofin_w}_{gap_w}')
    for nanofin_w in (0.15, 0.2, 0.22) for gap_w in (0.1, 0.15)]


# TODO(Nate): hone in on the best ranges to get an operational device

'''
Pull-apart TDC
'''

pull_apart_gap_tdc = []  # Captured below

# Motivation: Symmtric TDC requires very small critical dimensions and
# the aymmtric case requires a wider gap for mode purtubation and
# realistically better care in length but this is a test case
# tapers are the only way to reach these aggressive goals

# TODO(Nate): Tapering and dc_gap is MOST important
pull_apart_taper_tdc = [
    chip.nems_tdc(anchor=pull_apart_anchor, dc_gap_w=gap_w, bend_dim=(test_tdc_radius, test_tdc_interport_w / 2 - gap_w / 2 - waveguide_w / 2), **taper_dict_tdc(taper_change, taper_length))
    for gap_w in (0.100, 0.125, .150, 0.300) for taper_change in (0, -0.16, -0.32, -0.52) for taper_length in (20,)]

taper_change, taper_length, gap_w = -0.52, 20, 0.125
pull_apart_taper_tdc += [chip.nems_tdc(interaction_l=test_tdc_interaction_l_extr, anchor=pull_apart_anchor, dc_gap_w=gap_w, bend_dim=(test_tdc_radius, test_tdc_interport_w / 2 - gap_w / 2 - waveguide_w / 2), **taper_dict_tdc(taper_change, taper_length))]

# Motivation: fin thickness alone doesn't seem to be a critical parameter
# pull_apart_fin_tdc = [chip.nems_tdc(anchor=pull_apart_anchor, nanofin_w=nanofin_w) for nanofin_w in (0.15, 0.22)]
pull_apart_fin_tdc = []


'''
Pull-in TDC
'''

pull_in_gap_tdc = []  # Captured below

# Motivation: attempt pull-in TDC with tapering to reduce the beat length of the TDC
# Tapering and dc_gap is MOST important
pull_in_taper_tdc = [
    chip.nems_tdc(anchor=tdc_anchor, dc_gap_w=gap_w, bend_dim=(test_tdc_radius, test_tdc_interport_w / 2 - gap_w / 2 - waveguide_w / 2), **taper_dict_tdc(taper_change, taper_length))
    for gap_w in (0.100, 0.125, .150, 0.300) for taper_change in (0, -0.16, -0.32, -0.52) for taper_length in (20,)]
taper_change, taper_length, gap_w = -0.52, 20, 0.125
pull_in_taper_tdc += [chip.nems_tdc(interaction_l=test_tdc_interaction_l_extr, anchor=tdc_anchor, dc_gap_w=gap_w, bend_dim=(test_tdc_radius, test_tdc_interport_w / 2 - gap_w / 2 - waveguide_w / 2), **taper_dict_tdc(taper_change, taper_length))]
# Motivation: fin thickness alone doesn't seem to be a critical parameter
# pull_in_fin_tdc = [chip.nems_tdc(anchor=tdc_anchor, nanofin_w=nanofin_w) for nanofin_w in (0.1, 0.22)]
pull_in_fin_tdc = []


# Motivation: Test sructures necessary for reference meaurements

delay_arms = chip.mzi_arms([delay_line_50, gnd_wg],
                           [delay_line_200, gnd_wg],
                           interport_w=test_interport_w,
                           name='bare_mzi_arms')
delay_arms_gnded = chip.mzi_arms([delay_line_50, gnd_wg],
                                 [delay_line_200, gnd_wg],
                                 interport_w=test_interport_w,
                                 name='bare_mzi_arms_gnded')

with nd.Cell(name='ref_dc') as ref_dc:
    dc_r = dc.put()
    d = detector.put(dc_r.pin['b1'], flip=True)
    nd.Pin('p1').put(d.pin['p'])
    nd.Pin('n1').put(d.pin['n'])
    d = detector.put(dc_r.pin['b0'])
    nd.Pin('p2').put(d.pin['p'])
    nd.Pin('n2').put(d.pin['n'])
    dc_r.raise_pins(['a0', 'a1', 'b0', 'b1'])

with nd.Cell(name='ref_dc_short') as ref_dc_short:
    dc_r = dc_short.put()
    d = detector.put(dc_r.pin['b1'], flip=True)
    nd.Pin('p1').put(d.pin['p'])
    nd.Pin('n1').put(d.pin['n'])
    d = detector.put(dc_r.pin['b0'])
    nd.Pin('p2').put(d.pin['p'])
    nd.Pin('n2').put(d.pin['n'])
    dc_r.raise_pins(['a0', 'a1', 'b0', 'b1'])

with nd.Cell(name='ref_dc_invdes') as ref_dc_invdes:
    dc_r = dc_invdes.put()
    d = detector.put(dc_r.pin['b1'], flip=True)
    nd.Pin('p1').put(d.pin['p'])
    nd.Pin('n1').put(d.pin['n'])
    d = detector.put(dc_r.pin['b0'])
    nd.Pin('p2').put(d.pin['p'])
    nd.Pin('n2').put(d.pin['n'])
    dc_r.raise_pins(['a0', 'a1', 'b0', 'b1'])

with nd.Cell(name='ref_dc_aggressive') as ref_dc_aggressive:
    dc_r = dc_aggressive.put()
    d = detector.put(dc_r.pin['b1'], flip=True)
    nd.Pin('p1').put(d.pin['p'])
    nd.Pin('n1').put(d.pin['n'])
    d = detector.put(dc_r.pin['b0'])
    nd.Pin('p2').put(d.pin['p'])
    nd.Pin('n2').put(d.pin['n'])
    dc_r.raise_pins(['a0', 'a1', 'b0', 'b1'])

# Testing bend Radii 10,5,2.5,1


def bend_exp(name='bends_1_1'):
    bend_radius = float(name.split('_')[-2])
    delay_length = 4 * np.pi * bend_radius if 3 * np.pi * bend_radius > 50 else 50
    straight_length = delay_length / 2 if 3 * np.pi * bend_radius > 50 else 25
    with nd.Cell(name=name) as bend_exp:
        first_dc = dc.put()
        delay_line = chip.delay_line(delay_length=delay_length, straight_length=straight_length, bend_radius=bend_radius)
        l_arm = [delay_line for count in range(int(name.split('_')[-1]))]
        mzi_arms = chip.mzi_arms(l_arm,
                                 [wg_filler, ],
                                 interport_w=test_interport_w,
                                 name='bare_mzi_arms').put(first_dc.pin['b0'])
        nd.Pin('a0').put(first_dc.pin['a0'])
        nd.Pin('a1').put(first_dc.pin['a1'])

        d = detector.put(mzi_arms.pin['b1'], flip=True)
        nd.Pin('p1').put(d.pin['p'])
        nd.Pin('n1').put(d.pin['n'])
        d = detector.put(mzi_arms.pin['b0'])
        nd.Pin('p2').put(d.pin['p'])
        nd.Pin('n2').put(d.pin['n'])

        mzi_arms.raise_pins(['b0', 'b1'])
    return bend_exp


dc_short = chip.custom_dc(bend_dim=(aggressive_dc_radius, test_bend_dim_short), gap_w=test_gap_w_aggressive,
                          interaction_l=test_interaction_l_aggressive)[0]
dc_aggressive = chip.custom_dc(bend_dim=(aggressive_dc_radius, test_bend_dim_aggressive), gap_w=test_gap_w_aggressive,
                               interaction_l=test_interaction_l_aggressive)[0]
dc_invdes = chip.custom_dc(bend_dim=(aggressive_dc_radius, test_bend_dim_invdes), gap_w=test_gap_w_invdes,
                           interaction_l=test_interaction_l_invdes)[0]

reference_devices = [
    ref_dc,
    chip.mzi_node_test(delay_arms_gnded,
                       dc, include_input_ps=False,
                       detector=detector,
                       name='bare_mzi_gnded'),
    chip.mzi_node_test(delay_arms,
                       dc, include_input_ps=False,
                       detector=detector,
                       name='bare_mzi'),
    ref_dc_short,
    ref_dc_aggressive,
    ref_dc_invdes
]

bend_exp_names = [f'bends_{br}_{i}' for i in [2, 4, 8] for br in [1, 2.5, 5]]
bend_exp_names += [f'bends_10_{i}' for i in [2, 3]]
# bend_exp_names += [f'bends_15_{i}' for i in [2, 3]]
reference_devices += [bend_exp(name=bend_exp_name) for bend_exp_name in bend_exp_names]
print(len(reference_devices))


def tether_ps(phaseshift_l=tether_phaseshift_l, taper_l=5, taper_change=-0.05):
    anchor_tether = chip.nems_anchor(
        fin_dim=(phaseshift_l, 0.4), shuttle_dim=(10, 2), spring_dim=(phaseshift_l + 10, 0.22), straight_connector=None,
        tether_connector=(2, 1, 0.5, 1), pos_electrode_dim=(phaseshift_l, 4, 0.5), neg_electrode_dim=(3, 3),
        include_fin_dummy=False, name=f'anchor_tether_ps_{phaseshift_l}_{taper_l}_{taper_change}',
    )
    return chip.mzi_arms(
        [delay_line_50, chip.nems_ps(end_ls=(5, 5), end_taper=((0.0,), (0.0, -0.08),), taper_l=taper_l,
                                     wg_taper=cubic_taper(taper_change), gap_taper=cubic_taper(taper_change), gnd_connector_idx=0,
                                     phaseshift_l=phaseshift_l, anchor=anchor_tether, clearout_box_dim=(phaseshift_l + 5, 12.88))],
        [delay_line_200],
        interport_w=test_interport_w,
        name=f'pull_apart_tether_{phaseshift_l}_{taper_l}_{taper_change}'
    )


def tether_ps_slot(phaseshift_l=tether_phaseshift_l, taper_l=5, taper_change=-0.3):
    anchor_tether = chip.nems_anchor(
        fin_dim=(phaseshift_l, 0.4), shuttle_dim=(10, 2), spring_dim=(phaseshift_l + 10, 0.22), straight_connector=None,
        tether_connector=(2, 1, 0.5, 1), pos_electrode_dim=(phaseshift_l, 4, 1.5), neg_electrode_dim=(3, 3),
        include_fin_dummy=False, name=f'anchor_tether_ps_{phaseshift_l}_{taper_l}_{taper_change}',
    )
    return chip.mzi_arms(
        [chip.nems_ps(end_ls=(5, 5), end_taper=((0.0,), (0.0, -0.08),), taper_l=taper_l, boundary_taper=cubic_taper(taper_change),
                      wg_taper=cubic_taper(-0.32), gap_taper=cubic_taper(-0.32), gnd_connector_idx=0,
                      phaseshift_l=phaseshift_l, anchor=anchor_tether, clearout_box_dim=(phaseshift_l + 5, 12.88))],
        [tether_phaseshift_l + 10],
        interport_w=test_interport_w,
        name=f'tether_slot_{phaseshift_l}_{taper_l}_{taper_change}'
    )


def tether_tdc(interaction_l=tether_interaction_l, taper_l=5, taper_change=-0.05):
    anchor_tether = chip.nems_anchor(
        fin_dim=(interaction_l, 0.4), shuttle_dim=(5, 2), spring_dim=(interaction_l + 5, 0.22), straight_connector=None,
        tether_connector=(2, 1, 0.5, 1), pos_electrode_dim=(interaction_l - 5, 4, 0.5), neg_electrode_dim=(3, 3),
        include_fin_dummy=False, name=f'anchor_tether_tdc_{interaction_l}_{taper_l}_{taper_change}'
    )
    return chip.nems_tdc(anchor=anchor_tether, interaction_l=interaction_l,
                         dc_taper_ls=(taper_l,), dc_taper=(cubic_taper(taper_change),),
                         beam_taper=(cubic_taper(taper_change),), clearout_box_dim=(interaction_l + 5, 12.88),
                         name=f'pull_apart_tdc_{interaction_l}_{taper_l}_{taper_change}', dc_end_l=5,
                         metal_extension=6)


# testing tap lines
testing_tap_line = chip.tap_line(n_test)
testing_tap_line_tdc = chip.tap_line(n_test, inter_wg_dist=200)
testing_tap_line_aggressive = chip.tap_line(n_test, inter_wg_dist=280)

# make sure there are 17 structures per column
ps_columns = [
    pull_apart_gap + pull_apart_taper + pull_apart_fin,
    pull_in_gap + pull_in_taper + pull_in_fin
]

# make sure there are 17 structures per column
tdc_columns = [
    pull_apart_gap_tdc + pull_apart_taper_tdc + pull_apart_fin_tdc,
    pull_in_gap_tdc + pull_in_taper_tdc + pull_in_fin_tdc
]

# make sure there are 17 structures per column
vip_columns = [reference_devices]

tether_column = [
    tether_ps(psl, taper_l, taper_change) for psl in (60, 80) for taper_l, taper_change in ((5, -0.05), (10, -0.1), (15, -0.1))
] + [
    tether_tdc(il, taper_l, taper_change) for il in (80, 100) for taper_l, taper_change in ((10, -0.1), (15, -0.1), (20, -0.16))
    # Nate: Making these more agreesive b/c the waveguide width it's thin enough, 400nm needs to be a test pt, actually the taper chagnes needed to be doubled
] + [
    tether_tdc(il, taper_l, taper_change) for il in (80, 100) for taper_l, taper_change in ((20, -0.32), (20, -0.52))
] + [
    tether_tdc(30, 5, -0.1)
]

aggressive_column = [
    tether_ps_slot(tether_phaseshift_l, taper_length)
    for taper_length in (5, 10) for taper_change in (-0.25, -0.3, -0.35)  # slot variations
] + [chip.mzi_arms([chip.nems_ps(anchor=pull_apart_anchor)], [tether_phaseshift_l + 10]) for _ in range(2)] + [
    chip.mzi_arms([chip.nems_ps(anchor=pull_in_anchor, **pull_in_dict(pull_in_phaseshift_l))],
                  [pull_in_phaseshift_l + 10]) for _ in range(2)
] + [
    tether_ps(100, taper_l, taper_change) for taper_l, taper_change in ((10, -0.05), (15, -0.1), (20, -0.1),
                                                                        (25, -0.1))
] + [
    tether_tdc(il, taper_l, taper_change) for il in (100,) for taper_l, taper_change in ((20, -0.32), (20, -0.42),
                                                                                         (20, -0.52))
]

# TODO(sunil): MANUAL insert test structures
aggressive_column_dcs = [dc_short] * 6 + [dc_aggressive, dc_invdes, dc_aggressive, dc_invdes] + [dc_short] * 7

gridsearches = []

# TODO(Nate): Match this to the fill of the test array
# Number of test structures in each tap line, comment this out when not needed (when all are n_test)
gridsearch_ls = [len(ps_columns[0]), len(ps_columns[1]), len(tdc_columns[0]), len(tdc_columns[1]),
                 len(vip_columns[0]), len(tether_column), len(tdc_columns[1]), len(aggressive_column)]

print(gridsearch_ls)


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
for col, ps_columns in enumerate(ps_columns):
    with nd.Cell(f'gridsearch_{col}') as gridsearch:
        line = testing_tap_line.put()
        for i, ps in enumerate(ps_columns):
            # all structures for a tap line should be specified here
            node = chip.mzi_node_test(ps, dc, include_input_ps=False,
                                      detector=detector,
                                      name=f'test_mzi_{ps.name}').put(line.pin[f'a{2 * i + 1}'])
            # mzi_node_test tracks multiple gnd and pos lines in the upper and lower arms

            autoroute_node_detector(node.pin['p1'], node.pin['n2'], node.pin['n1'], node.pin['p2'])
            nd.Pin(f'd{i}').put(node.pin['b0'])  # this is useful for autorouting the gnd path
            gnd_l, gnd_u, pos_l, pos_u = None, None, None, None,
            for pin in node.pin.keys():
                if pin.split('_')[0] == 'gnd0':
                    if pin.split('_')[1] == 'l':
                        if gnd_l is not None:
                            chip.m2_ic.bend_strt_bend_p2p(node.pin[gnd_l], node.pin[pin], radius=8).put()
                        gnd_l = pin
                    if pin.split('_')[1] == 'u':
                        if gnd_u is not None:
                            chip.m2_ic.bend_strt_bend_p2p(node.pin[gnd_u], node.pin[pin], radius=8).put()
                        gnd_u = pin
                    gnd_pin = pin
                if pin.split('_')[0] == 'pos0':
                    if pin.split('_')[1] == 'l':
                        if pos_l is not None:
                            chip.m2_ic.bend_strt_bend_p2p(node.pin[pos_l], node.pin[pin], radius=8).put()
                        pos_l = pin
                    if pin.split('_')[1] == 'u':
                        if pos_u is not None:
                            chip.m2_ic.bend_strt_bend_p2p(node.pin[pos_u], node.pin[pin], radius=8).put()
                        pos_u = pin
                    pos_pin = pin

            if (gnd_u is not None) and (gnd_l is not None):
                chip.m2_ic.bend_strt_bend_p2p(node.pin[gnd_u], node.pin[gnd_l], radius=8).put()
                gnd_pin = gnd_u
                nd.Pin(f'gnd{i}').put(node.pin[gnd_pin])
            elif(gnd_u is not None) or (gnd_l is not None):
                gnd_pin = gnd_u if gnd_u is not None else gnd_l
                nd.Pin(f'gnd{i}').put(node.pin[gnd_pin])

            if (pos_u is not None) and (pos_l is not None):
                chip.m2_ic.bend_strt_bend_p2p(node.pin[pos_u], node.pin[pos_l], radius=8).put()
                pos_pin = pos_u
                nd.Pin(f'pos{i}').put(node.pin[pos_pin])
            elif(pos_u is not None) or (pos_l is not None):
                pos_pin = pos_u if pos_u is not None else pos_l
                nd.Pin(f'pos{i}').put(node.pin[pos_pin])

            if 'pos0' in node.pin:
                nd.Pin(f'pos{i}').put(node.pin['pos0'])  # hard coded special case that I know about the array elements
            if 'gnd0' in node.pin:
                nd.Pin(f'gnd{i}').put(node.pin['gnd0'])
        nd.Pin('in').put(line.pin['in'])
        nd.Pin('out').put(line.pin['out'])
    gridsearches.append(gridsearch)

for col, tdc_column in enumerate(tdc_columns):
    with nd.Cell(f'gridsearch_{col + len(ps_columns)}') as gridsearch:
        line = testing_tap_line_tdc.put()
        for i, _tdc in enumerate(tdc_column):
            # all structures for a tap line should be specified here
            dev = _tdc.put(line.pin[f'a{2 * i + 1}'])
            d1 = detector.put(dev.pin['b0'])
            d2 = detector.put(dev.pin['b1'], flip=True)
            autoroute_node_detector(d2.pin['p'], d1.pin['n'], d2.pin['n'], d1.pin['p'])
            nd.Pin(f'd{i}').put(dev.pin['b0'])  # this is useful for autorouting the gnd path
            gnd_l, gnd_u = None, None
            for pin in dev.pin.keys():
                if pin.split('_')[0] == 'gnd0' and len(pin.split('_')) > 1:
                    if pin.split('_')[1] == 'l':
                        if gnd_l is not None:
                            chip.m2_ic.strt_p2p(dev.pin[gnd_l], dev.pin[pin]).put()
                        gnd_l = pin
                    if pin.split('_')[1] == 'u':
                        if gnd_u is not None:
                            chip.m2_ic.strt_p2p(dev.pin[gnd_u], dev.pin[pin]).put()
                        gnd_u = pin
                    gnd_pin = pin
            chip.m2_ic.ubend_p2p(dev.pin['gnd0_u_0'], dev.pin['gnd0_l_0'], radius=10).put()

            if 'pos1' in dev.pin:
                nd.Pin(f'pos{i}').put(dev.pin['pos1'])
            if 'gnd0' in dev.pin:
                nd.Pin(f'gnd{i}').put(dev.pin['gnd0'])
            elif 'gnd0_u_1' in dev.pin:
                nd.Pin(f'gnd{i}').put(dev.pin['gnd0_u_1'])
        nd.Pin('in').put(line.pin['in'])
        nd.Pin('out').put(line.pin['out'])
    gridsearches.append(gridsearch)


for col, vip_column in enumerate(vip_columns):
    with nd.Cell(f'gridsearch_{col + len(vip_columns)}') as gridsearch:
        line = testing_tap_line.put()
        for i, vip in enumerate(vip_column):
            # all structures for a tap line should be specified here
            _vip = vip.put(line.pin[f'a{2 * i + 1}'])
            autoroute_node_detector(_vip.pin['p1'], _vip.pin['n2'], _vip.pin['n1'], _vip.pin['p2'])
            nd.Pin(f'd{i}').put(_vip.pin['b0'])  # this is useful for autorouting the gnd path
        nd.Pin('in').put(line.pin['in'])
        nd.Pin('out').put(line.pin['out'])
    gridsearches.append(gridsearch)


# tether gridsearch
with nd.Cell(f'gridsearch_tether') as gridsearch:
    line = testing_tap_line.put()
    for i, device in enumerate(tether_column):
        if i < 6:
            dev = chip.mzi_node_test(device, dc, include_input_ps=False,
                                     detector=detector,
                                     name=f'test_mzi_{ps.name}').put(line.pin[f'a{2 * i + 1}'])
            autoroute_node_detector(dev.pin['p1'], dev.pin['n2'], dev.pin['n1'], dev.pin['p2'])
        else:
            # all structures for a tap line should be specified here
            dev = device.put(line.pin[f'a{2 * i + 1}'])
            d1 = detector.put(dev.pin['b0'])
            d2 = detector.put(dev.pin['b1'], flip=True)
            autoroute_node_detector(d2.pin['p'], d1.pin['n'], d2.pin['n'], d1.pin['p'])
        nd.Pin(f'd{i}').put(dev.pin['b0'])  # this is useful for autorouting the gnd path
        if 'pos1' in dev.pin:
            nd.Pin(f'pos{i}').put(dev.pin['pos1'])
        if 'gnd0' in dev.pin:
            nd.Pin(f'gnd{i}').put(dev.pin['gnd0'])
    nd.Pin('in').put(line.pin['in'])
    nd.Pin('out').put(line.pin['out'])
gridsearches.append(gridsearch)


# aggressive gridsearch
with nd.Cell(f'gridsearch_aggressive') as gridsearch:
    line = testing_tap_line_aggressive.put()
    for i, device in enumerate(zip(aggressive_column, aggressive_column_dcs)):
        ps, dc = device
        dev = chip.mzi_node_test(ps, dc, include_input_ps=False,
                                 detector=detector,
                                 name=f'test_mzi_{ps.name}').put(line.pin[f'a{2 * i + 1}'])
        autoroute_node_detector(dev.pin['p1'], dev.pin['n2'], dev.pin['n1'], dev.pin['p2'])
        nd.Pin(f'd{i}').put(dev.pin['b0'])  # this is useful for autorouting the gnd path
        if 'pos1' in dev.pin:
            nd.Pin(f'pos{i}').put(dev.pin['pos1'])
        if 'gnd0' in dev.pin:
            nd.Pin(f'gnd{i}').put(dev.pin['gnd0'])
    nd.Pin('in').put(line.pin['in'])
    nd.Pin('out').put(line.pin['out'])
aggressive_gridsearch = gridsearch


# test structures between the meshes


middle_mesh_pull_apart = [
    chip.mzi_node_test(chip.mzi_arms([ps], [1], interport_w=mesh_interport_w), dc, include_input_ps=False,
                       name=f'meshtest_mzi_{ps.name}') for ps in (chip.nems_ps(anchor=pull_apart_anchor),
                                                                  chip.nems_ps(anchor=pull_apart_anchor,
                                                                               **pull_apart_taper_dict(-0.05, 30)))
]


middle_mesh_pull_in = [
    chip.mzi_node_test(chip.mzi_arms([delay_line_50, ps, gnd_wg],
                                     [delay_line_200, gnd_wg],
                                     interport_w=mesh_interport_w),
                       dc, include_input_ps=False,
                       name=f'meshtest_mzi_{ps.name}') for ps in (chip.nems_ps(anchor=pull_in_anchor,
                                                                               **pull_in_dict(pull_in_phaseshift_l)),
                                                                  chip.nems_ps(anchor=pull_in_anchor,
                                                                               **pull_in_dict(pull_in_phaseshift_l,
                                                                                              -0.05, 20)))
]
middle_mesh_tdc = [chip.nems_tdc(anchor=pull_apart_anchor), chip.nems_tdc(anchor=tdc_anchor),
                   chip.nems_tdc(anchor=pull_apart_anchor, **taper_dict_tdc(-0.2, 40))]

middle_mesh_test_structures = middle_mesh_pull_apart + middle_mesh_pull_in + middle_mesh_tdc

# gnd pad (testing side)
with nd.Cell('gnd_pad') as gnd_pad:
    chip.ml_ic.strt(width=1716, length=60).put()

with nd.Cell('test_pad') as test_pad:
    chip.ml_ic.strt(width=1716, length=60).put()

chiplet_divider = chip.dice_box((100, 1973))
chip_horiz_dice = chip.dice_box((chip_w, perimeter_w))
chip_vert_dice = chip.dice_box((perimeter_w, chip_h))

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
    eu_bp_port_ranges_m1 = [(0, 2),                  # layer 0 /input
                            (5, 7),                  # 2 lines into this layer
                            (15, 20),                # backward mesh output detector layer
                            # (20, 22), (23, 25),      # test MZI # moved to test routing
                            (28, 30), (38, 40),      # layer 1
                            # (43, 45), (46, 48),      # test MZI # moved to test routing
                            (50, 53), (61, 64),      # 3 lines into this layer
                            (73, 76), (84, 87),      # layer 2
                            (95, 99), (107, 111),    # 4 lines into this layer
                            (118, 122), (130, 134),  # layer 3
                            (140, 145), (153, 158),  # 5 lines into this layer
                            (163, 168), (176, 181),  # layer 4
                            (186, 191), (199, 204),  # 5 lines into this layer
                            (210, 214), (222, 226),  # layer 5
                            (233, 237), (245, 249),  # 4 lines into this layer
                            (257, 260), (268, 271),  # layer 6
                            (280, 283), (291, 294),  # 3 lines into this layer
                            (304, 306), (314, 316),  # layer 7
                            (327, 329),              # 2 lines into this layer
                            (337, 342)]              # forward mesh output detector layer

    eu_bp_m1_idx = np.hstack([np.arange(*r) for r in eu_bp_port_ranges_m1])

    used_connections = set()
    for idx in tqdm(eu_bp_m1_idx):
        pin_x = eu_array_nems.pin[f'o{idx}'].x
        closest = (0, 0)
        closest_dist = np.inf
        for i, j in itertools.product(range(70), range(3)):
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

    # TODO: M2 and M1 are flipped in varibale names around eu_bp_ variables, eh not too important now
    # TODO(someone not Nate): double check the remapped mesh connections

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

    for block in tqdm(eu_bp_port_blocks):
        closest_pair = (0, (0, 0))
        closest_dist = np.inf
        start_idx = None
        for idx in block:
            if start_idx is None:
                start_idx = idx
            pin_x = eu_array_nems.pin[f'o{idx}'].x
            for i, j in itertools.product(range(70), range(3)):
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
        chip.m2_ic.bend_strt_bend_p2p(eu_array_nems.pin[f'o{idx}'], radius=8, width=8).put()
        chip.m2_ic.strt(100 * (2 - j), width=19).put(bp_array_thermal.pin[f'u{i},{j}'])
        chip.m2_ic.bend_strt_bend_p2p(eu_array_thermal.pin[f'o{idx}'], radius=8, width=19).put()
        # TODO: Add M1 layer to thermal gnd blocks to push down resistance
        # chip.v1_via_4.put()

    # TODO:In mesh test Structure BP routing, MANUALLY ROUTE Last Part BEFORE HANDING OFF
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
    # TODO(someone not Nate): double check the remapped mesh connections
    eu_bp_port_tests_m1 = [
        (20, 22), (23, 25),      # test MZI # moved to test routing
        (43, 45), (66, 68),      # test MZI # moved to test routing
        (69, 70), (89, 91),
        (92, 93), (112, 114),
        (115, 116), (135, 137),
        (138, 139), (158, 160),
        (161, 162), (181, 183),
        (184, 185), (204, 206),
        (207, 208), (227, 229),
        (319, 321),
        (342, 344)
    ]

    eu_bp_test_m1_idx = np.hstack([np.arange(*r) for r in eu_bp_port_tests_m1])

    left_most = np.NINF
    for idx in tqdm(eu_bp_test_m1_idx):
        pin_x = eu_array_nems.pin[f'o{idx}'].x
        closest = (0, 0)
        closest_dist = np.inf
        for i, j in itertools.product(range(70), range(3)):
            dist = (bp_array_nems.pin[f'u{i},{j}'].x - pin_x)
            # hacking this to prevent crossings in test layer
            if dist < closest_dist and bp_array_nems.pin[f'u{i},{j}'].x > left_most and (i, j) not in used_connections:
                closest = (i, j)
                closest_dist = dist
        left_most = pin_x
        used_connections.add(closest)
        i, j = closest
        chip.v1_via_8.put(bp_array_nems.pin[f'u{i},{j}'])
        chip.m2_ic.strt(100 * (2 - j), width=8).put(bp_array_nems.pin[f'u{i},{j}'])
        # chip.m1_ic.bend_strt_bend_p2p(eu_array_nems.pin[f'o{idx}'], radius=8, width=8).put()
        chip.v1_via_8.put()

        chip.v1_via_8.put(bp_array_thermal.pin[f'u{i},{j}'])
        chip.m2_ic.strt(100 * (2 - j), width=8).put(bp_array_thermal.pin[f'u{i},{j}'])
        # chip.m1_ic.bend_strt_bend_p2p(eu_array_thermal.pin[f'o{idx}'], radius=8, width=8).put()
        chip.v1_via_8.put()

    pin_num = 0
    num_ps_middle_mesh = len(middle_mesh_pull_apart + middle_mesh_pull_in)

    mzi_node_nems = mzi_node_nems_detector.put(input_interposer.pin['a7'], flip=True)
    alignment_mark.put(-500, 0)
    alignment_mark.put(7000, 0)
    alignment_mark.put(7000, 1700)
    alignment_mark.put(-500, 1700)

    for layer in range(15):
        # autoroute
        autoroute_nems_gnd = autoroute_4_nems_gnd.put(layer * mesh_layer_x + 8.05, 536, flop=True)
        autoroute_nems_pos = autoroute_4_nems_pos.put(layer * mesh_layer_x - 10, 550, flop=True)
        autoroute_nems_cathode = autoroute_4.put(layer * mesh_layer_x + 178, 542)
        autoroute_nems_anode = autoroute_4_extended.put(layer * mesh_layer_x + 178, 550)
        autoroute_thermal_gnd = autoroute_8.put(layer * mesh_layer_x, 1228, flop=True, flip=True)
        autoroute_thermal_pos = autoroute_8_extended.put(layer * mesh_layer_x, 1218, flop=True, flip=True)
        autoroute_therm_cathode = autoroute_4.put(layer * mesh_layer_x + 178, 1208, flip=True)
        autoroute_therm_anode = autoroute_4_extended.put(layer * mesh_layer_x + 178, 1200, flip=True)

        if layer >= 3 and layer < len(middle_mesh_test_structures) + 3:
            # mid-mesh test structures
            ts_idx = layer - 3
            test_structure = middle_mesh_test_structures[ts_idx]
            shift_x = 40 * (ts_idx >= num_ps_middle_mesh)
            # Nate: added a quick 0 , -20 hack to fix drc
            shift_hack = 0 if 'tdc' in test_structure.cell_name else -20
            ts = test_structure.put(shift_hack + 1325 + shift_x - 30 * (ts_idx < num_ps_middle_mesh) + mesh_layer_x * ts_idx,
                                    output_interposer.pin['a7'].y + 20, flip=True)
            chip.waveguide_ic.strt(shift_x).put(ts.pin['a0'])
            chip.waveguide_ic.bend(radius=7, angle=-90).put()
            chip.waveguide_ic.strt(15).put()
            bend = chip.waveguide_ic.bend(radius=7, angle=-90).put()
            grating.put(bend.pin['b0'].x, bend.pin['b0'].y, -90)
            chip.waveguide_ic.strt(7 + shift_x).put(ts.pin['a1'])
            chip.waveguide_ic.bend(radius=7, angle=-90).put()
            chip.waveguide_ic.strt(113).put()
            chip.waveguide_ic.bend(radius=7, angle=-90).put()
            bend = chip.waveguide_ic.strt(7).put()
            grating.put(bend.pin['b0'].x, bend.pin['b0'].y, -90)
            d1 = detector.put(ts.pin['b0'], flip=True)
            d2 = detector.put(ts.pin['b1'])
            chip.m2_ic.bend_strt_bend_p2p(d2.pin['n'], autoroute_nems_anode.pin['b5'], radius=4).put()
            chip.m1_ic.bend_strt_bend_p2p(d2.pin['p'], autoroute_nems_cathode.pin['b5'], radius=4).put()
            chip.v1_via_4.put()
            chip.m2_ic.bend_strt_bend_p2p(d1.pin['n'], autoroute_nems_anode.pin['b6'], radius=4).put()
            chip.m1_ic.bend_strt_bend_p2p(d1.pin['p'], autoroute_nems_cathode.pin['b6'], radius=4).put()
            chip.v1_via_4.put()
            chip.v1_via_4.put(d1.pin['p'])
            chip.v1_via_4.put(d2.pin['p'])

            gnd_l, gnd_u, pos_l, pos_u = None, None, None, None,
            for pin in ts.pin.keys():
                if pin.split('_')[0] == 'gnd0' and len(pin.split('_')) > 1:
                    if pin.split('_')[1] == 'l':
                        if gnd_l is not None:
                            chip.m2_ic.strt_p2p(ts.pin[gnd_l], ts.pin[pin]).put()
                        gnd_l = pin
                    if pin.split('_')[1] == 'u':
                        if gnd_u is not None:
                            chip.m2_ic.strt_p2p(ts.pin[gnd_u], ts.pin[pin]).put()
                        gnd_u = pin
                    gnd_pin = pin
                if pin.split('_')[0] == 'pos0' and len(pin.split('_')) > 1:
                    if pin.split('_')[1] == 'l':
                        if pos_l is not None:
                            chip.m2_ic.bend_strt_bend_p2p(ts.pin[pos_l], ts.pin[pin], radius=8).put()
                        pos_l = pin
                    if pin.split('_')[1] == 'u':
                        if pos_u is not None:
                            chip.m2_ic.bend_strt_bend_p2p(ts.pin[pos_u], ts.pin[pin], radius=8).put()
                        pos_u = pin
                    pos_pin = pin

            if (gnd_u is not None) and (gnd_l is not None):
                chip.m2_ic.ubend_p2p(ts.pin[gnd_l], ts.pin[gnd_u], radius=8).put()
                gnd_pin = gnd_u
            elif(gnd_u is not None) or (gnd_l is not None):
                gnd_pin = gnd_u if gnd_u is not None else gnd_l

            # if (pos_u is not None) and (pos_l is not None):
            #     chip.m2_ic.bend_strt_bend_p2p(ts.pin[pos_u], ts.pin[pos_l], radius=8).put()
            #     pos_pin = pos_u
            # elif(pos_u is not None) or (pos_l is not None):
            #     pos_pin = pos_u if pos_u is not None else pos_l

            if 'pos0' in ts.pin:
                chip.m2_ic.bend_strt_bend_p2p(ts.pin['pos0'], autoroute_nems_pos.pin['a6'], radius=4).put()
            elif pos_pin is not None:
                chip.m2_ic.bend_strt_bend_p2p(ts.pin[pos_pin], autoroute_nems_pos.pin['a6'], radius=4).put()
            if 'gnd1' in ts.pin:
                chip.m2_ic.bend_strt_bend_p2p(ts.pin['gnd1'], autoroute_nems_gnd.pin['a6'], radius=4).put()
            elif gnd_pin is not None:
                chip.m2_ic.bend_strt_bend_p2p(ts.pin[gnd_pin], autoroute_nems_gnd.pin['a6'], radius=4).put()

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
            # TODO(sunil): edit this
            # nems and thermal test node
            chip.m2_ic.bend_strt_bend_p2p(mzi_node_nems.pin['n1'], autoroute_nems_anode.pin['a5'], radius=8).put()
            chip.m2_ic.bend_strt_bend_p2p(mzi_node_nems.pin['n2'], autoroute_nems_anode.pin['a6'], radius=8).put()
            chip.m1_ic.bend_strt_bend_p2p(mzi_node_nems.pin['p1'], autoroute_nems_cathode.pin['a5'], radius=8).put()
            chip.m2_ic.bend_strt_bend_p2p(mzi_node_nems.pin['p2'], autoroute_nems_cathode.pin['a6'], radius=8).put()
            chip.v1_via_4.put(mzi_node_nems.pin['p1'])
            chip.v1_via_4.put(autoroute_nems_cathode.pin['p1'], flop=True)
            chip.m2_ic.bend_strt_bend_p2p(mzi_node_thermal.pin['n1'], autoroute_therm_anode.pin['a5'], radius=8).put()
            chip.m2_ic.bend_strt_bend_p2p(mzi_node_thermal.pin['n2'], autoroute_therm_anode.pin['a6'], radius=8).put()
            chip.m1_ic.bend_strt_bend_p2p(mzi_node_thermal.pin['p1'], autoroute_therm_cathode.pin['a5'], radius=8).put()
            chip.m2_ic.bend_strt_bend_p2p(mzi_node_thermal.pin['p2'], autoroute_therm_cathode.pin['a6'], radius=8).put()
            chip.v1_via_4.put(mzi_node_thermal.pin['p1'])
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
            chip.waveguide_ic.bend(10, -180).put()
            d1 = detector.put(flip=True)
            chip.waveguide_ic.bend(10, 180).put(dev.pin['b1'])
            d2 = detector.put(flip=True)
            chip.m2_ic.bend_strt_bend_p2p(d2.pin['n'], autoroute_nems_anode.pin['a5'], radius=4).put()
            chip.m2_ic.bend_strt_bend_p2p(d1.pin['n'], autoroute_nems_anode.pin['a6'], radius=4).put()
            chip.m2_ic.bend_strt_bend_p2p(d2.pin['p'], autoroute_nems_cathode.pin['a5'], radius=4).put()
            chip.m2_ic.bend_strt_bend_p2p(d1.pin['p'], autoroute_nems_cathode.pin['a6'], radius=4).put()

            # Nate: needed gnding, so I copied this here, sorry for repeating codee
            gnd_l, gnd_u, pos_l, pos_u = None, None, None, None,
            for pin in dev.pin.keys():
                if pin.split('_')[0] == 'gnd0' and len(pin.split('_')) > 1:
                    if pin.split('_')[1] == 'l':
                        if gnd_l is not None:
                            chip.m2_ic.strt_p2p(dev.pin[gnd_l], dev.pin[pin]).put()
                        gnd_l = pin
                    if pin.split('_')[1] == 'u':
                        if gnd_u is not None:
                            chip.m2_ic.strt_p2p(dev.pin[gnd_u], dev.pin[pin]).put()
                        gnd_u = pin
                    gnd_pin = pin
                if pin.split('_')[0] == 'pos0' and len(pin.split('_')) > 1:
                    if pin.split('_')[1] == 'l':
                        if pos_l is not None:
                            chip.m2_ic.bend_strt_bend_p2p(dev.pin[pos_l], dev.pin[pin], radius=8).put()
                        pos_l = pin
                    if pin.split('_')[1] == 'u':
                        if pos_u is not None:
                            chip.m2_ic.bend_strt_bend_p2p(dev.pin[pos_u], dev.pin[pin], radius=8).put()
                        pos_u = pin
                    pos_pin = pin

            # if (gnd_u is not None) and (gnd_l is not None):
            #     chip.m2_ic.ubend_p2p(dev.pin[gnd_l], dev.pin[gnd_u], radius=8).put()
            #     gnd_pin = gnd_u
            # elif(gnd_u is not None) or (gnd_l is not None):
            #     gnd_pin = gnd_u if gnd_u is not None else gnd_l

            # Nate quick hack because I know device orientation
            chip.m2_ic.ubend_p2p(dev.pin['gnd0_u_0'], dev.pin['gnd0_l_0'], radius=8).put()

            if 'pos1' in dev.pin:
                chip.m2_ic.bend_strt_bend_p2p(dev.pin['pos1'], autoroute_nems_pos.pin['a5'], radius=4).put()
            elif pos_pin is not None:
                chip.m2_ic.bend_strt_bend_p2p(dev.pin[pos_pin], autoroute_nems_pos.pin['a5'], radius=4).put()
            if 'gnd1' in dev.pin:
                chip.m2_ic.bend_strt_bend_p2p(dev.pin['gnd1'], autoroute_nems_gnd.pin['a5'], radius=4).put()
            elif gnd_pin is not None:
                chip.m2_ic.bend_strt_bend_p2p(dev.pin[gnd_pin], autoroute_nems_gnd.pin['a5'], radius=4).put()

            # mesh tap test
            test_tap = tap_detector.put(output_interposer.pin['a6'].x - 40,
                                        output_interposer.pin['a6'].y, flip=False)
            d = detector.put(test_tap.pin['a0'])
            extra_bend_p2p(d.pin['n'], autoroute_therm_anode.pin['a5'], 4, -90, 60)
            extra_bend_p2p(d.pin['p'], autoroute_therm_cathode.pin['a5'], 10, -90, 70)

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


# Test chiplet layout
with nd.Cell('test_chiplet') as test_chiplet:
    # place test taplines down at non-overlapping locations
    detector_x = []
    test_structures = gridsearches + gridsearches[2:3] + [aggressive_gridsearch]  # TODO: change this once all 8 columns are added
    ga = grating_array.put(*grating_array_xy, -90)
    gs_list = []
    for i, item in enumerate(zip(tapline_x, test_structures)):
        x, gridsearch = item
        gs = gridsearch.put(x, tapline_y)
        chip.waveguide_ic.bend_strt_bend_p2p(ga.pin[f'a{2 * i + 1}'], gs.pin['out'], radius=10).put()
        chip.waveguide_ic.bend_strt_bend_p2p(ga.pin[f'a{2 * i + 2}'], gs.pin['in'], radius=10).put()
        detector_x.append([gs.pin[f'd{j}'].x for j in range(gridsearch_ls[i])])
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
    gnd_pad.put(-15, test_pad_y)

    # TODO(Nate): space thse pads as columns are filled
    # place test pads

    # TODO(Nate): for probes, we can either manually cut them in the end or have a semiautomated way of doing it (modify below)
    test_pads = [test_pad.put(x, test_pad_y) for x in test_pad_x]

    for i in range(min(gridsearch_ls)):  # change this to n_test when all structures are filled in each column

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
                chip.m2_ic.bend_strt_bend_p2p(gs_list[j].pin[f'gnd{i}'], radius=8).put()  # for anchor gnd pin
            # pos electrode connection
            if f'pos{i}' in gs_list[j].pin:
                pin = gs_list[j].pin[f'pos{i}']
                # ensures that this route will not intersect any of the current routes
                offset = ground_wire.pin['a0'].y + 6 - pin.y
                offset = -offset if pin.a == 180 else offset
                chip.m2_ic.sbend(radius=4, offset=offset).put(pin)
                chip.m2_ic.strt(test_pads[j].bbox[0] - nd.cp.x()).put(*nd.cp.get_xy(), 0)
                # connect to the pad using via
                chip.va_via.put()


# Final chip layout
with nd.Cell('aim_layout') as aim_layout:
    mesh_chiplet.put(mesh_chiplet_x)
    test_chiplet.put(test_chiplet_x)
    chiplet_divider.put(chiplet_divider_x, -standard_grating_interport + 20)
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
# nd.export_gds(filename=f'../../../20200819_sjby_aim_run/aim-layout-{str(date.today())}-submission', topcells=[aim_layout])
