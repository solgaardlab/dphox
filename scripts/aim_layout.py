import itertools

import nazca as nd
import numpy as np
from dphox.design.aim import AIMNazca
from dphox.design.component import cubic_taper
from datetime import date
from tqdm import tqdm

chip = AIMNazca(
    passive_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35a_passive.gds',
    waveguides_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35_waveguides.gds',
    active_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35a_active.gds',
)


# component params
n_pads_eu = (344, 12)
n_pads_bp = (70, 3)
n_test = 17
dc_radius = 15
pdk_dc_radius = 25
sep = 30
gnd_length = 15

# testing params
waveguide_w = 0.48
test_interport_w = 50
test_gap_w = 0.3
test_bend_dim = test_interport_w / 2 - test_gap_w / 2 - waveguide_w / 2
test_tdc_interport_w = 50
test_tdc_interaction_l = 100
pull_in_phaseshift_l = 50
test_tdc_bend_dim = test_tdc_interport_w / 2 - test_gap_w / 2 - waveguide_w / 2

mesh_interport_w = 50
mesh_phaseshift_l = 100
detector_loopback_params = (5, 20)

# Basic components

dc = chip.custom_dc(bend_dim=(dc_radius, test_bend_dim))[0]
mesh_dc = chip.pdk_dc(radius=pdk_dc_radius, interport_w=mesh_interport_w)
tap = chip.bidirectional_tap(10, mesh_bend=True)
pull_apart_anchor = chip.nems_anchor()
pull_in_anchor = chip.nems_anchor(shuttle_dim=(40, 5), fin_spring_dim=(50, 0.15),
                                  pos_electrode_dim=None, neg_electrode_dim=None)
tdc_anchor = chip.nems_anchor(shuttle_dim=(test_tdc_interaction_l, 5),
                              pos_electrode_dim=None, neg_electrode_dim=None)
tdc = chip.nems_tdc(anchor=tdc_anchor)
ps = chip.nems_ps(anchor=pull_apart_anchor, tap_sep=(tap, sep))
ps_no_anchor = chip.nems_ps()
alignment_mark = chip.alignment_mark()
gnd_wg = chip.gnd_wg()

# Mesh generation

thermal_ps = chip.thermal_ps((tap, sep))
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
    final_period=127, radius=50, trombone_radius=10,
    self_coupling_extension_dim=(30, 200),
    with_gratings=True, horiz_dist=200, num_trombones=2
)

bp_array = chip.bond_pad_array(n_pads_bp, stagger_x_frac=0.4)
bp_array_testing = chip.bond_pad_array((2, n_test))
eu_array = chip.eutectic_array(n_pads_eu)
autoroute_4 = chip.autoroute_turn(7, level=2, turn_radius=8, connector_x=0, connector_y=20,
                                  final_period=18.5, width=4)
autoroute_4_extended = chip.autoroute_turn(7, level=2, turn_radius=8, connector_x=9, connector_y=28,
                                           final_period=18.5, width=4)
autoroute_4_nems_gnd = chip.autoroute_turn(7, level=2, turn_radius=8, connector_x=8, connector_y=16,
                                           final_period=18.5, width=4)
autoroute_4_nems_pos = chip.autoroute_turn(7, level=2, turn_radius=8, connector_x=1, connector_y=28,
                                           final_period=18.5, width=4)
autoroute_8 = chip.autoroute_turn(7, level=2, turn_radius=8, connector_x=0, connector_y=0,
                                  final_period=18.5, width=8)
autoroute_8_extended = chip.autoroute_turn(7, level=2, turn_radius=8, connector_x=9, connector_y=10,
                                           final_period=18.5, width=8)


# Test structures

# Shortcut to keep same params as default while only changing tapers


def pull_apart_taper_dict(taper_change: float, taper_length: float):
    return dict(
        taper_ls=(2, 0.15, 0.2, 0.15, 2, taper_length),
        gap_taper=(
            (0.66 + 2 * 0.63,), (0, -1 * (.30 + 2 * 0.63),), (0,), (0, (.30 + 2 * 0.63),),
            cubic_taper(-0.74 - 2 * 0.63), cubic_taper(taper_change)),
        wg_taper=((0,), (0,), (0,), (0,), cubic_taper(-0.08),
                  cubic_taper(taper_change)),
        boundary_taper=(
            (0.66 + 2 * 0.63,), (0,), (0,), (0,), cubic_taper(-0.74 - 2 * 0.63), (0,)),
        rib_brim_taper=(
            cubic_taper(2 * .66), (0,), (0,), (0,), cubic_taper(-0.74 * 2),
            cubic_taper(taper_change))
    )


def pull_in_dict(phaseshift_l: float = 100, taper_change: float = None, taper_length: float = None):
    # TODO: modify this to taper the pull-in fin adiabatically using rib_brim_taper
    if taper_change is None or taper_length is None:
        return dict(
            phaseshift_l=phaseshift_l, clearout_box_dim=(phaseshift_l - 10, 3),
            taper_ls=(0,), gap_taper=None, wg_taper=None, boundary_taper=None, rib_brim_taper=None
        )
    else:
        return dict(
            phaseshift_l=phaseshift_l, clearout_box_dim=(phaseshift_l - 10, 3),
            taper_ls=(taper_length,), gap_taper=(cubic_taper(taper_change),),
            wg_taper=(cubic_taper(taper_change),), boundary_taper=((0,),), rib_brim_taper=None
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
    chip.singlemode_ps(chip.nems_ps(gap_w=gap_w, anchor=pull_apart_anchor, name=f'ps_gap_{gap_w}'),
                       interport_w=test_interport_w,
                       phaseshift_l=mesh_phaseshift_l, name=f'pull_apart_gap_{gap_w}')
    for gap_w in (0.1, 0.15, 0.2, 0.25)]

# Motivation: reduce the waveguide width to encourage more phase shift per unit length in center
pull_apart_taper = [
    chip.singlemode_ps(chip.nems_ps(anchor=pull_apart_anchor, **pull_apart_taper_dict(taper_change, taper_length)
                                    , name=f'ps_taper_{taper_change}_{taper_length}'),
                       interport_w=test_interport_w,
                       phaseshift_l=mesh_phaseshift_l, name=f'pull_apart_taper_{taper_change}_{taper_length}')
    for taper_change in (-0.1, -0.15) for taper_length in (20, 30, 40)]

# Motivation: modify fin width to change stiffness / phase shift per unit length
pull_apart_fin = [
    chip.singlemode_ps(chip.nems_ps(anchor=pull_apart_anchor,
                                    nanofin_w=nanofin_w, name=f'ps_fin_{nanofin_w}'),
                       interport_w=test_interport_w,
                       phaseshift_l=mesh_phaseshift_l, name=f'pull_apart_fin_{nanofin_w}')
    for nanofin_w in (0.15, 0.2, 0.25)]

'''
Pull-in phase shifter or PSV1
'''

# Motivation: attempt pull-in phase shifter idea with tapering to reduce pull-in voltage (for better or worse...)
# and phase shift length

pull_in_gap = [
    chip.singlemode_ps_ext_gnd(chip.nems_ps(anchor=pull_in_anchor, gap_w=gap_w, **pull_in_dict(pull_in_phaseshift_l),
                                            name=f'ps_gap_{gap_w}'), gnd_length,
                               interport_w=test_interport_w,
                               phaseshift_l=pull_in_phaseshift_l + 2 * gnd_length, name=f'pull_in_gap_{gap_w}')
    for gap_w in (0.1, 0.15, 0.2)]

# Motivation: attempt pull-in phase shifter idea with tapering to reduce pull-in voltage (for better or worse...)
# and phase shift length. To increase pull-in, phase shift length is made shorter.
pull_in_taper = [
    chip.singlemode_ps_ext_gnd(chip.nems_ps(anchor=pull_in_anchor, **pull_in_dict(pull_in_phaseshift_l,
                                                                                  taper_change, taper_length),
                                            name=f'ps_taper_{taper_change}_{taper_length}'), gnd_length,
                               interport_w=test_interport_w,
                               phaseshift_l=pull_in_phaseshift_l + 2 * gnd_length,
                               name=f'pull_in_taper_{taper_change}_{taper_length}')
    for taper_change in (-0.1, -0.15) for taper_length in (10, 20)]

# Motivation: attempt pull-in phase shifter idea with modifying fin width / phase shift per unit length
pull_in_fin = [
    chip.singlemode_ps_ext_gnd(
        chip.nems_ps(anchor=pull_in_anchor, nanofin_w=nanofin_w, **pull_in_dict(pull_in_phaseshift_l),
                     name=f'ps_fin_{nanofin_w}'), gnd_length,
        interport_w=test_interport_w,
        phaseshift_l=pull_in_phaseshift_l + 2 * gnd_length, name=f'pull_in_fin_{nanofin_w}')
    for nanofin_w in (0.15, 0.2)]

# Motivation: attempt pull-in phase shifter idea with tapering to reduce pull-in voltage (for better or worse...)
# and phase shift length
pull_apart_gap_tdc = [chip.nems_tdc(anchor=pull_apart_anchor, dc_gap_w=gap_w) for gap_w in (0.1, 0.15, 0.2)]

# Motivation: attempt pull-in phase shifter idea with tapering to reduce pull-in voltage (for better or worse...)
# and phase shift length
pull_apart_taper_tdc = [
    chip.nems_tdc(anchor=pull_apart_anchor, **taper_dict_tdc(taper_change, taper_length))
    for taper_change in (-0.2, -0.3) for taper_length in (20, 40)]

# Motivation: attempt pull-in phase shifter idea with modifying fin width / phase shift per unit length
pull_apart_fin_tdc = [chip.nems_tdc(anchor=pull_apart_anchor, nanofin_w=nanofin_w) for nanofin_w in (0.15, 0.2)]

# Motivation: attempt pull-in TDC with varying gap to adjust pull-in voltage (for better or worse...)
# and phase shift length
pull_in_gap_tdc = [chip.nems_tdc(anchor=tdc_anchor, dc_gap_w=gap_w) for gap_w in (0.1, 0.15, 0.2)]

# Motivation: attempt pull-in TDC with tapering to reduce the beat length of the TDC
pull_in_taper_tdc = [
    chip.nems_tdc(anchor=tdc_anchor, **taper_dict_tdc(taper_change, taper_length))
    for taper_change in (-0.2, -0.3) for taper_length in (20, 40)
]

# Motivation: attempt pull-in phase shifter idea with modifying fin width / phase shift per unit length
pull_in_fin_tdc = [chip.nems_tdc(anchor=tdc_anchor, nanofin_w=nanofin_w) for nanofin_w in (0.1, 0.2)]

# testing tap lines
testing_tap_line = chip.tap_line(n_test)
testing_tap_line_tdc = chip.tap_line(n_test, inter_wg_dist=200)

ps_columns = [
    pull_apart_gap + pull_apart_taper + pull_apart_fin,
    pull_in_gap + pull_in_taper + pull_in_fin
]

tdc_columns = [
    pull_apart_gap_tdc + pull_apart_taper_tdc + pull_apart_fin_tdc,
    pull_in_gap_tdc + pull_in_taper_tdc + pull_in_fin_tdc
]

gridsearches = []

# Number of test structures in each tap line, comment this out when not needed (when all are n_test)
gridsearch_ls = [len(ps_columns[0]), len(ps_columns[1]), len(tdc_columns[0]), len(tdc_columns[1])] * 2


def route_detector(p1, n2, n1, p2):
    # annoying routing hard-coding... TODO(sunil): make this better
    chip.v1_via_array_4.put(p1)
    chip.m1_ic.bend(8, 90).put(p1)
    chip.m1_ic.strt(27).put()
    chip.v1_via_array_4.put(n2)
    chip.m1_ic.bend(4, -90).put(n2)
    chip.m1_ic.strt(19).put()
    chip.m2_ic.bend(4, 90).put(n1)
    chip.m2_ic.strt(15).put()
    chip.m2_ic.bend(8, -90).put(p2)
    chip.m2_ic.strt(33).put()


for col, ps_columns in enumerate(ps_columns):
    with nd.Cell(f'gridsearch_{col}') as gridsearch:
        line = testing_tap_line.put()
        for i, ps in enumerate(ps_columns):
            # all structures for a tap line should be specified here
            node = chip.mzi_node(ps, dc, include_input_ps=False,
                                 detector=chip.pdk_cells['cl_band_photodetector_digital'],
                                 name=f'test_mzi_{ps.name}'
                                 ).put(line.pin[f'a{2 * i + 1}'])
            route_detector(node.pin['p1'], node.pin['n2'], node.pin['n1'], node.pin['p2'])
            nd.Pin(f'd{i}').put(node.pin['b0'])  # this is useful for autorouting the gnd path
        nd.Pin('in').put(line.pin['in'])
        nd.Pin('out').put(line.pin['out'])
    gridsearches.append(gridsearch)

for col, tdc_column in enumerate(tdc_columns):
    with nd.Cell(f'gridsearch_{col + len(ps_columns)}') as gridsearch:
        line = testing_tap_line_tdc.put()
        for i, tdc in enumerate(tdc_column):
            # all structures for a tap line should be specified here
            _tdc = tdc.put(line.pin[f'a{2 * i + 1}'])
            detector = chip.pdk_cells['cl_band_photodetector_digital']
            d1 = detector.put(_tdc.pin['b0'])
            d2 = detector.put(_tdc.pin['b1'], flip=True)
            route_detector(d2.pin['p'], d1.pin['n'], d2.pin['n'], d1.pin['p'])
            nd.Pin(f'd{i}').put(_tdc.pin['b0'])  # this is useful for autorouting the gnd path
        nd.Pin('in').put(line.pin['in'])
        nd.Pin('out').put(line.pin['out'])
    gridsearches.append(gridsearch)

chiplet_divider = chip.dice_box((100, 2000))

# test pad
with nd.Cell('gnd_pad') as gnd_pad:
    chip.ml_ic.strt(width=1716, length=60).put()
    chip.va_via.put(50, -813, array=[1, [1, 0], n_test, [0, 100]])

with nd.Cell('test_pad') as test_pad:
    chip.ml_ic.strt(width=1716, length=60).put()
    chip.va_via.put(50, -778, array=[1, [1, 0], n_test, [0, 100]])

# Chip construction

# TODO(sunil): remove hardcoding! tsk tsk...
with nd.Cell('aim') as aim:
    nems = nems_mesh.put(0, 750, flip=True)
    thermal = thermal_mesh.put(0, 1000)
    input_interposer = interposer.put(thermal.pin['a4'])
    output_interposer = interposer.put(thermal.pin['b4'], flip=True)
    mzi_node_thermal_detector.put(input_interposer.pin['a6'])
    mdc = mesh_dc.put(output_interposer.pin['a6'])
    mzi_node_nems_detector.put(input_interposer.pin['a7'], flip=True)
    alignment_mark.put(-500, 0)
    alignment_mark.put(-500 + 8400, 0)

    # routing code for the meshes
    bp_array_nems = bp_array.put(-180, -40)
    eu_array_nems = eu_array.put(-180, 200)
    bp_array_thermal = bp_array.put(-180, 1778, flip=True)
    eu_array_thermal = eu_array.put(-180, 1538, flip=True)

    # all ranges are [inclusive, exclusive) as is convention in python range() method
    # add more when test structures are added in between the meshes
    eu_bp_port_ranges_m1 = [(0, 2), (5, 7), (15, 17), (20, 25),
                            (28, 30), (38, 40), (43, 48),
                            (50, 53), (61, 64),
                            (73, 76), (84, 87),
                            (96, 100), (107, 111),
                            (118, 122), (130, 134),
                            (139, 144), (153, 158),
                            (162, 167), (176, 181),
                            (185, 190), (199, 204),
                            (210, 214), (222, 226),
                            (232, 237), (245, 249),
                            (257, 260), (268, 271),
                            (280, 283), (291, 294),
                            (304, 306), (314, 316),
                            (327, 329), (337, 339)]

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

    # TODO: incomplete... fill out these ranges and do routing
    eu_bp_port_blocks_m2 = [(7, 10), (11, 14), (30, 33), (34, 37)]

    pin_num = 0
    for layer in range(15):
        a1_nems_left = autoroute_4_nems_gnd.put(layer * 450 + 8.05, 536, flop=True)
        a2_nems_left = autoroute_4_nems_pos.put(layer * 450 - 10, 550, flop=True)
        # for i in range(7):
        #     chip.v1_via_array_4.put(a1_nems_left.pin[f'a{i}'])
        a1_nems_right = autoroute_4.put(layer * 450 + 178, 542)
        a2_nems_right = autoroute_4_extended.put(layer * 450 + 178, 550)
        a1_thermal_left = autoroute_8.put(layer * 450, 1228, flop=True, flip=True)
        a2_thermal_left = autoroute_8_extended.put(layer * 450, 1218, flop=True, flip=True)
        a1_thermal_right = autoroute_4.put(layer * 450 + 178, 1208, flip=True)
        a2_thermal_right = autoroute_4_extended.put(layer * 450 + 178, 1200, flip=True)

        for pin_nems, pin_thermal in zip(reversed([a2_nems_left.pin[f'p{n}'] for n in range(7)]),
                                         reversed([a2_thermal_left.pin[f'p{n}'] for n in range(7)])):
            if pin_num < n_pads_eu[0]:
                chip.m2_ic.bend_strt_bend_p2p(pin_nems, eu_array_nems.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.m2_ic.bend_strt_bend_p2p(pin_thermal, eu_array_thermal.pin[f'i{pin_num}'], radius=8,
                                              width=8).put()
            pin_num += 1
        for i, pins in enumerate(zip(reversed([a1_nems_left.pin[f'p{n}'] for n in range(7)]),
                                     reversed([a1_thermal_left.pin[f'p{n}'] for n in range(7)]))):
            pin_nems, pin_thermal = pins
            if pin_num < n_pads_eu[0]:
                chip.m1_ic.bend_strt_bend_p2p(pin_nems, eu_array_nems.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.v1_via_array_4.put(pin_nems)
                chip.m1_ic.bend_strt_bend_p2p(pin_thermal, eu_array_thermal.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.v1_via_array_8.put(pin_thermal)
                chip.v1_via_array_8.put(layer * 450 - 26.5, 1204, 90,
                                        array=[6, [-18.5, 0], 6, [0, -mesh_interport_w]])
                chip.v1_via_array_8.put(layer * 450 - 26.5, 1204 - 7.5, 90,
                                        array=[6, [-18.5, 0], 6, [0, -mesh_interport_w]])
            pin_num += 1 if i % 2 else 0
        pin_num += 1
        for i, pins in enumerate(zip([a1_nems_right.pin[f'p{n}'] for n in range(7)],
                                     [a1_thermal_right.pin[f'p{n}'] for n in range(7)])):
            pin_nems, pin_thermal = pins
            if pin_num < n_pads_eu[0]:
                chip.m1_ic.bend_strt_bend_p2p(pin_nems, eu_array_nems.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.v1_via_array_4.put(pin_nems)
                chip.m1_ic.bend_strt_bend_p2p(pin_thermal, eu_array_thermal.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.v1_via_array_8.put(pin_thermal)
            pin_num += 1 if i % 2 else 0
        pin_num += 1
        for pin_nems, pin_thermal in zip([a2_nems_right.pin[f'p{n}'] for n in range(7)],
                                         [a2_thermal_right.pin[f'p{n}'] for n in range(7)]):
            if pin_num < n_pads_eu[0]:
                chip.m2_ic.bend_strt_bend_p2p(pin_nems, eu_array_nems.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.m2_ic.bend_strt_bend_p2p(pin_thermal, eu_array_thermal.pin[f'i{pin_num}'], radius=8,
                                              width=8).put()
            pin_num += 1
        pin_num += 1

    # place test gridsearches down at non-overlapping locations
    gridsearch_x = [8300, 8700, 9000, 9300, 9700, 10100, 10400, 10700]
    detector_x = []
    gridsearches = gridsearches + gridsearches  # TODO: change this once all 8 columns are added
    ga = grating_array.put(8300, 125, -90)
    for i, item in enumerate(zip(gridsearch_x, gridsearches)):
        x, gridsearch = item
        gs = gridsearch.put(x - 40, 162)
        chip.waveguide_ic.bend_strt_bend_p2p(ga.pin[f'a{2 * i + 1}'], gs.pin['out'], radius=10).put()
        chip.waveguide_ic.bend_strt_bend_p2p(ga.pin[f'a{2 * i + 2}'], gs.pin['in'], radius=10).put()
        detector_x.append([gs.pin[f'd{j}'].x for j in range(gridsearch_ls[i])])

    chiplet_divider.put(7540, -127)

    # put bond pad arrays on the left and right of the testing area
    bp_array_left = bp_array_testing.put(7780, 212)
    bp_array_right = bp_array_testing.put(12000 + input_interposer.bbox[0] - 232, 212)

    for i in range(min(gridsearch_ls)):  # change this to n_test when all structures are filled in each column
        # detector wire connections
        chip.m2_ic.bend(26, -90).put(bp_array_left.pin[f'u{0},{i}'])
        p = chip.m2_ic.strt(3000 - 6).put()
        chip.m1_ic.bend(20, -90).put(bp_array_left.pin[f'u{1},{i}'])
        chip.m1_ic.strt(3000 - 100).put()
        chip.m1_ic.bend(28, -90).put(bp_array_right.pin[f'd{1},{i}'])
        chip.m1_ic.strt(3000 - 6).put()
        chip.m2_ic.bend(22, -90).put(bp_array_right.pin[f'd{0},{i}'])
        chip.m2_ic.strt(3000 - 100).put()
        chip.v1_via.put(bp_array_right.pin[f'd{1},{i}'])
        chip.v1_via.put(bp_array_left.pin[f'u{1},{i}'])

        # loop ground wires around detectors
        cx = 7700
        for x in detector_x:
            chip.m2_ic.strt(x[i] - 40 - cx).put(cx, p.pin['a0'].y - 80)
            chip.m2_ic.bend(radius=4, angle=90).put()
            chip.m2_ic.strt(20).put()
            chip.m2_ic.bend(radius=4, angle=-90).put()
            chip.m2_ic.strt(30).put()
            chip.m2_ic.bend(radius=4, angle=-90).put()
            chip.m2_ic.strt(20).put()
            chip.m2_ic.bend(radius=4, angle=90).put()
            cx = nd.cp.x()

    # place test pads (the first is gnd, needs some manual adjustment later on)
    gx = gridsearch_x
    test_pad_x = [gx[0] - 120, gx[1] - 120,
                  gx[2] - 290, gx[3] - 290,
                  gx[4] - 120, gx[5] - 120,
                  gx[6] - 290, gx[7] - 290]
    gnd_pad.put(7690, 971)

    # for probes, we can either manually cut them in the end or have a semiautomated way of doing it
    for x in test_pad_x:
        test_pad.put(x, 971)

    # Boundary indicators (REMOVE IN FINAL LAYOUT)
    chip.dice_box((12000, 100)).put(input_interposer.bbox[0] - 50, -227)
    chip.dice_box((12000, 100)).put(input_interposer.bbox[0] - 50, 2100 - 227)

nd.export_gds(filename=f'aim-layout-{str(date.today())}-submission', topcells=[aim])
