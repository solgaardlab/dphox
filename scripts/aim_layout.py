import nazca as nd
import numpy as np
from dphox.design.aim import AIMNazca
from dphox.design.component import get_cubic_taper
from datetime import date

chip = AIMNazca(
    passive_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35a_passive.gds',
    waveguides_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35_waveguides.gds',
    active_filepath='/Users/sunilpai/Documents/research/dphox/aim_lib/APSUNY_v35a_active.gds',
)


def get_bend_dim_from_interport_w(interport_w, gap_w, waveguide_w=0.48):
    return interport_w / 2 - gap_w / 2 - waveguide_w / 2


def is_adiabatic(taper_params, init_width: float = 0.48, wavelength: float = 1.55, neff: float = 2.75,
                 num_points: int = 100, taper_l: float = 5):
    taper_params = np.asarray(taper_params)
    u = np.linspace(0, 1 + 1 / num_points, num_points)[:, np.newaxis]
    width = init_width + np.sum(taper_params * u ** np.arange(taper_params.size, dtype=float), axis=1)
    theta = np.arctan(np.diff(width) / taper_l * num_points)
    max_pt = np.argmax(theta)
    return theta[max_pt], wavelength / (2 * width[max_pt] * neff)


dc_radius = 15
pdk_dc_radius = 25
sep = 30

# testing params
test_interport_w = 50
test_gap_w = 0.3
test_bend_dim = get_bend_dim_from_interport_w(test_interport_w, test_gap_w)
test_tdc_interport_w = 50
test_tdc_interaction_l = 100
pull_in_phaseshift_l = 50
test_tdc_bend_dim = get_bend_dim_from_interport_w(test_tdc_interport_w, test_gap_w)

mesh_interport_w = 50
mesh_phaseshift_l = 100
detector_loopback_params = (5, 20)

# Basic components

dc = chip.custom_dc(bend_dim=(dc_radius, test_bend_dim))[0]
mesh_dc = chip.pdk_dc(radius=pdk_dc_radius, interport_w=mesh_interport_w)
tap = chip.bidirectional_tap(10, mesh_bend=True)
pull_apart_anchor = chip.nems_anchor()
pull_in_anchor = chip.nems_anchor(connector_dim=(40, 5), fin_spring_dim=(50, 0.15),
                                  pos_electrode_dim=None, neg_electrode_dim=None)
tdc_anchor = chip.nems_anchor(connector_dim=(test_tdc_interaction_l, 5),
                              pos_electrode_dim=None, neg_electrode_dim=None)
tdc = chip.nems_tdc(anchor=tdc_anchor)
ps = chip.nems_ps(anchor=pull_apart_anchor, tap_sep=(tap, sep))
ps_no_anchor = chip.nems_ps()

# Mesh generation

thermal_ps = chip.thermal_ps((tap, sep))
dc_dummy = chip.waveguide(mesh_dc.pin['b0'].x - mesh_dc.pin['a0'].x)
mzi_node_nems = chip.mzi_node(chip.double_ps(ps, mesh_interport_w, name='nems_double_ps'), mesh_dc)
mzi_node_thermal = chip.mzi_node(chip.double_ps(thermal_ps, mesh_interport_w, name='thermal_double_ps'), mesh_dc)
mzi_node_nems_detector = chip.mzi_node(chip.double_ps(ps, mesh_interport_w,
                                                      name='nems_double_ps'), mesh_dc,
                                       detector=chip.pdk_cells['cl_band_photodetector_digital'])
mzi_node_thermal_detector = chip.mzi_node(chip.double_ps(thermal_ps, mesh_interport_w,
                                                         name='thermal_double_ps'), mesh_dc,
                                          detector=chip.pdk_cells['cl_band_photodetector_digital'])
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
pp_array = chip.bond_pad_array((1, 3), pitch=550, pad_dim=(60, 500), use_labels=False)
bp_array = chip.bond_pad_array()
bp_array_testing = chip.bond_pad_array((2, 17))
eu_array = chip.eutectic_array()
autoroute_simple_1 = chip.autoroute_turn(7, level=1, turn_radius=8, connector_x=0, connector_y=12)
autoroute_simple_2 = chip.autoroute_turn(7, level=2, turn_radius=8, connector_x=8, connector_y=4)


# Test structures

# Shortcut to keep same params as default while only changing tapers


def pull_apart_taper_dict(taper_change: float, taper_length: float):
    return dict(
        taper_ls=(2, 0.15, 0.2, 0.15, 2, taper_length),
        gap_taper=(
            (0.66 + 2 * 0.63,), (0, -1 * (.30 + 2 * 0.63),), (0,), (0, (.30 + 2 * 0.63),),
            get_cubic_taper(-0.74 - 2 * 0.63), get_cubic_taper(taper_change)),
        wg_taper=((0,), (0,), (0,), (0,), get_cubic_taper(-0.08),
                  get_cubic_taper(taper_change)),
        boundary_taper=(
            (0.66 + 2 * 0.63,), (0,), (0,), (0,), get_cubic_taper(-0.74 - 2 * 0.63), (0,)),
        rib_brim_taper=(
            get_cubic_taper(2 * .66), (0,), (0,), (0,), get_cubic_taper(-0.74 * 2),
            get_cubic_taper(taper_change))
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
            taper_ls=(taper_length,), gap_taper=(get_cubic_taper(taper_change),),
            wg_taper=(get_cubic_taper(taper_change),), boundary_taper=((0,),), rib_brim_taper=None
        )


def taper_dict_tdc(taper_change: float, taper_length: float):
    return dict(
        dc_taper_ls=(taper_length,), dc_taper=(get_cubic_taper(taper_change),),
        beam_taper=(get_cubic_taper(taper_change),)
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
    chip.singlemode_ps(chip.nems_ps(anchor=pull_in_anchor, gap_w=gap_w, **pull_in_dict(pull_in_phaseshift_l),
                                    name=f'ps_gap_{gap_w}'),
                       interport_w=test_interport_w,
                       phaseshift_l=pull_in_phaseshift_l, name=f'pull_in_gap_{gap_w}')
    for gap_w in (0.1, 0.15, 0.2)]

# Motivation: attempt pull-in phase shifter idea with tapering to reduce pull-in voltage (for better or worse...)
# and phase shift length. To increase pull-in, phase shift length is made shorter.
pull_in_taper = [
    chip.singlemode_ps(chip.nems_ps(anchor=pull_in_anchor, **pull_in_dict(pull_in_phaseshift_l,
                                                                          taper_change, taper_length),
                                    name=f'ps_taper_{taper_change}_{taper_length}'),
                       interport_w=test_interport_w,
                       phaseshift_l=pull_in_phaseshift_l, name=f'pull_in_taper_{taper_change}_{taper_length}')
    for taper_change in (-0.1, -0.15) for taper_length in (10, 20)]

# Motivation: attempt pull-in phase shifter idea with modifying fin width / phase shift per unit length
pull_in_fin = [
    chip.singlemode_ps(chip.nems_ps(anchor=pull_in_anchor, nanofin_w=nanofin_w, **pull_in_dict(pull_in_phaseshift_l),
                                    name=f'ps_fin_{nanofin_w}'),
                       interport_w=test_interport_w,
                       phaseshift_l=pull_in_phaseshift_l, name=f'pull_in_fin_{nanofin_w}')
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

testing_tap_line = chip.testing_tap_line(17)
testing_tap_line_tdc = chip.testing_tap_line(17, inter_wg_dist=200)

ps_columns = [
    pull_apart_gap + pull_apart_taper + pull_apart_fin,
    pull_in_gap + pull_in_taper + pull_in_fin
]

tdc_columns = [
    pull_apart_gap_tdc + pull_apart_taper_tdc + pull_apart_fin_tdc,
    pull_in_gap_tdc + pull_in_taper_tdc + pull_in_fin_tdc
]

gridsearches = []

for col, tdc_column in enumerate(ps_columns):
    with nd.Cell(f'gridsearch_{col}') as gridsearch:
        line = testing_tap_line.put()
        for i, ps in enumerate(tdc_column):
            # all structures for a tap line should be specified here
            chip.mzi_node(ps, dc, include_input_ps=False,
                          detector=chip.pdk_cells['cl_band_photodetector_digital'],
                          name=f'test_mzi_{ps.name}'
                          ).put(line.pin[f'a{2 * i + 1}'])
    gridsearches.append(gridsearch)

for col, tdc_column in enumerate(tdc_columns):
    with nd.Cell(f'gridsearch_{col + len(ps_columns)}') as gridsearch:
        line = testing_tap_line_tdc.put()
        for i, tdc in enumerate(tdc_column):
            # all structures for a tap line should be specified here
            tdc.put(line.pin[f'a{2 * i + 1}'])
    gridsearches.append(gridsearch)

# Chip construction

# TODO(sunil): remove hardcoding! tsk tsk...
with nd.Cell('aim') as aim:
    nems = nems_mesh.put(0, 750, flip=True)
    thermal = thermal_mesh.put(0, 1000)
    input_interposer = interposer.put(thermal.pin['a4'])
    output_interposer = interposer.put(thermal.pin['b4'], flip=True)
    mzi_node_thermal_detector.put(input_interposer.pin['a6'])
    mesh_dc.put(output_interposer.pin['a6'])
    mzi_node_nems_detector.put(input_interposer.pin['a7'], flip=True)
    num_ports = 344

    # routing code
    bp_array_nems = bp_array.put(-180, -40)
    eu_array_nems = eu_array.put(-180, 200)
    bp_array_thermal = bp_array.put(-180, 1778, flip=True)
    eu_array_thermal = eu_array.put(-180, 1538, flip=True)

    pin_num = 0
    for layer in range(15):
        a1_nems_left = autoroute_simple_1.put(layer * 450, 550, flop=True)
        a2_nems_left = autoroute_simple_2.put(layer * 450, 542, flop=True)
        a1_nems_right = autoroute_simple_1.put(layer * 450 + 178, 550)
        a2_nems_right = autoroute_simple_2.put(layer * 450 + 178, 542)
        a1_thermal_left = autoroute_simple_1.put(layer * 450, 1200, flop=True, flip=True)
        a2_thermal_left = autoroute_simple_2.put(layer * 450, 1208, flop=True, flip=True)
        a1_thermal_right = autoroute_simple_1.put(layer * 450 + 178, 1200, flip=True)
        a2_thermal_right = autoroute_simple_2.put(layer * 450 + 178, 1208, flip=True)

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
    gridsearches[0].put(8300, 150)
    gridsearches[1].put(8700, 150)
    gridsearches[2].put(9000, 150)
    gridsearches[3].put(9300, 150)
    chip.dice_box((100, 2000)).put(7600, -127)
    bp_array_testing.put(7800, 200)
    bp_array_testing.put(12000 + input_interposer.bbox[0] - 200, 200)
    pp_array.put(8250, 700)

    for i in range(16):
        for j in range(5):
            if j == 2:
                chip.m2_ic.strt(3240).put(7700, 215 + i * 100 + j * 6)
            else:
                chip.m2_ic.strt(3240).put(7700, 215 + i * 100 + j * 6)

    # Boundary indicators (REMOVE IN FINAL LAYOUT)
    chip.dice_box((12000, 100)).put(input_interposer.bbox[0] - 50, -227)
    chip.dice_box((12000, 100)).put(input_interposer.bbox[0] - 50, 2100 - 227)

nd.export_gds(filename=f'aim-layout-{str(date.today())}-submission', topcells=[aim])
