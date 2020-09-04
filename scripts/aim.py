import nazca as nd
import numpy as np
from dphox.design.aim import AIMNazca
from datetime import date

chip = AIMNazca(
    passive_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35a_passive.gds',
    waveguides_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35_waveguides.gds',
    active_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35a_active.gds',
)


def get_cubic_taper(change_w):
    return (0, 0, 3 * change_w, -2 * change_w)


def is_adiabatic(taper_params, init_width: float = 0.48, wavelength: float = 1.55, neff: float = 2.75,
                 num_points: int = 100, taper_l: float = 5):
    taper_params = np.asarray(taper_params)
    u = np.linspace(0, 1 + 1 / num_points, num_points)[:, np.newaxis]
    width = init_width + np.sum(taper_params * u ** np.arange(taper_params.size, dtype=float), axis=1)
    theta = np.arctan(np.diff(width) / taper_l * num_points)
    max_pt = np.argmax(theta)
    return theta[max_pt], wavelength / (2 * width[max_pt] * neff)

waveguide_w = .48
dc_radius = 25
sep = 30

# testing params
test_interport_w = 25
test_gap_w = 0.3
test_bend_dim = test_interport_w / 2 - test_gap_w / 2 - waveguide_w / 2

mesh_interport_w = 50
mesh_phaseshift_l = 100

# Basic components

dc = chip.custom_dc(bend_dim=(dc_radius, test_bend_dim))[0]
mesh_dc = chip.pdk_dc(radius=dc_radius, interport_w=mesh_interport_w)
tap = chip.bidirectional_tap(10, mesh_bend=True)
anchor = chip.nems_anchor()
ps = chip.nems_ps(anchor=anchor, tap_sep=(tap, sep))
ps_no_anchor = chip.nems_ps()
alignment_mark = chip.alignment_mark(100, 48)

# Mesh generation

thermal_ps = chip.thermal_ps((tap, sep))
dc_dummy = chip.waveguide(mesh_dc.pin['b0'].x - mesh_dc.pin['a0'].x)
mzi_node_nems = chip.mzi_node(chip.double_ps(ps, mesh_interport_w, name='nems_double_ps'), mesh_dc)
mzi_node_thermal = chip.mzi_node(chip.double_ps(thermal_ps, mesh_interport_w, name='thermal_double_ps'), mesh_dc)
mzi_dummy_nems = chip.mzi_dummy(ps, dc_dummy)
mzi_dummy_thermal = chip.mzi_dummy(thermal_ps, dc_dummy)
nems_mesh = chip.triangular_mesh(5, mzi_node_nems, mzi_dummy_nems, ps, mesh_interport_w)
thermal_mesh = chip.triangular_mesh(5, mzi_node_thermal, mzi_dummy_thermal, thermal_ps, mesh_interport_w)

interposer = chip.interposer(
    n=14, waveguide_w=0.48, period=50,
    final_period=127, radius=50, trombone_radius=10,
    self_coupling_extension_dim=(30, 200),
    with_gratings=True, horiz_dist=200, num_trombones=2
)
bp_array = chip.bond_pad_array()
eu_array = chip.eutectic_array()
autoroute_simple_1 = chip.autoroute_turn(7, level=1, turn_radius=8, connector_x=0, connector_y=20)
autoroute_simple_2 = chip.autoroute_turn(7, level=2, turn_radius=8, connector_x=0, connector_y=20)

# Test structures

psv3_gap = [
    chip.singlemode_ps(chip.nems_ps(gap_w=gap_w, anchor=anchor), interport_w=test_interport_w,
                       phaseshift_l=mesh_phaseshift_l)
    for gap_w in (0.2, 0.25, 0.3, 0.35, 0.4)]
testing_tap_line = chip.testing_tap_line(15)
with nd.Cell('gridsearch') as gridsearch:
    line = testing_tap_line.put()
    for i, ps in enumerate(psv3_gap):  # all structures for a tap line should be specified here
        chip.mzi_node(ps, dc, include_input_ps=False,
                      detector=chip.pdk_cells['cl_band_photodetector_digital'],
                      detector_loopback_params=(5, 15)).put(line.pin[f'a{i}'])

# Chip construction

# TODO(sunil): remove hardcoding! tsk tsk...
with nd.Cell('aim') as aim:
    nems = nems_mesh.put(0, 750, flip=True)
    thermal = thermal_mesh.put(0, 1000)
    input_interposer = interposer.put(thermal.pin['a4'])
    output_interposer = interposer.put(thermal.pin['b4'], flip=True)
    mzi_node_thermal.put(input_interposer.pin['a6'])
    mzi_node_nems.put(input_interposer.pin['a7'], flip=True)
    num_ports = 344
    alignment_mark.put(-500,1700)
    alignment_mark.put(-500+8520, 1700)

    # routing code
    bp_array_nems = bp_array.put(-180, -40)
    eu_array_nems = eu_array.put(-180, 200)
    bp_array_thermal = bp_array.put(-180, 1778, flip=True)
    eu_array_thermal = eu_array.put(-180, 1538, flip=True)
    pin_num = 0
    for layer in range(15):
        a1_nems_left = autoroute_simple_1.put(layer * 450, 550, flop=True)
        a2_nems_left = autoroute_simple_2.put(layer * 450, 550, flop=True)
        a1_nems_right = autoroute_simple_1.put(layer * 450 + 178, 550)
        a2_nems_right = autoroute_simple_2.put(layer * 450 + 178, 550)
        a1_thermal_left = autoroute_simple_1.put(layer * 450, 1200, flop=True, flip=True)
        a2_thermal_left = autoroute_simple_2.put(layer * 450, 1200, flop=True, flip=True)
        a1_thermal_right = autoroute_simple_1.put(layer * 450 + 178, 1200, flip=True)
        a2_thermal_right = autoroute_simple_2.put(layer * 450 + 178, 1200, flip=True)

        for pin_nems, pin_thermal in zip(reversed([a2_nems_left.pin[f'p{n}'] for n in range(7)]),
                                         reversed([a2_thermal_left.pin[f'p{n}'] for n in range(7)])):
            if pin_num < num_ports:
                chip.m2_ic.bend_strt_bend_p2p(pin_nems, eu_array_nems.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.m2_ic.bend_strt_bend_p2p(pin_thermal, eu_array_thermal.pin[f'i{pin_num}'], radius=8,
                                              width=8).put()
            pin_num += 1
        for i, pins in enumerate(zip(reversed([a1_nems_left.pin[f'p{n}'] for n in range(7)]),
                                     reversed([a1_thermal_left.pin[f'p{n}'] for n in range(7)]))):
            pin_nems, pin_thermal = pins
            if pin_num < num_ports:
                chip.m1_ic.bend_strt_bend_p2p(pin_nems, eu_array_nems.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.m1_ic.bend_strt_bend_p2p(pin_thermal, eu_array_thermal.pin[f'i{pin_num}'], radius=8,
                                              width=8).put()
            pin_num += 1 if i % 2 else 0
        pin_num += 1
        for i, pins in enumerate(zip([a1_nems_right.pin[f'p{n}'] for n in range(7)],
                                     [a1_thermal_right.pin[f'p{n}'] for n in range(7)])):
            pin_nems, pin_thermal = pins
            if pin_num < num_ports:
                chip.m1_ic.bend_strt_bend_p2p(pin_nems, eu_array_nems.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.m1_ic.bend_strt_bend_p2p(pin_thermal, eu_array_thermal.pin[f'i{pin_num}'], radius=8,
                                              width=8).put()
            pin_num += 1 if i % 2 else 0
        pin_num += 1
        for pin_nems, pin_thermal in zip([a2_nems_right.pin[f'p{n}'] for n in range(7)],
                                         [a2_thermal_right.pin[f'p{n}'] for n in range(7)]):
            if pin_num < num_ports:
                chip.m2_ic.bend_strt_bend_p2p(pin_nems, eu_array_nems.pin[f'i{pin_num}'], radius=8, width=8).put()
                chip.m2_ic.bend_strt_bend_p2p(pin_thermal, eu_array_thermal.pin[f'i{pin_num}'], radius=8,
                                              width=8).put()
            pin_num += 1
        pin_num += 1
    for n in range(9):
        gridsearch.put(8000 + n * 400, 200)

nd.export_gds(filename=f'aim-layout-{str(date.today())}-submission.design', topcells=[aim])
