import nazca as nd
import numpy as np
from dphox.design.aim import AIMNazca

if __name__ == 'main':

    chip = AIMNazca(
        passive_filepath='/home/exx/Documents/research/dphox/aim_lib/APSUNY_v35a_passive.gds',
        waveguides_filepath='/home/exx/Documents/research/dphox/aim_lib/APSUNY_v35_waveguides.gds',
        active_filepath='/home/exx/Documents/research/dphox/aim_lib/APSUNY_v35a_active.gds',
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


    # print(is_adiabatic((0, 0.66)))

    waveguide_w = .48
    length = 100
    taper_ls = (5, 5)
    # taper_params = (get_cubic_taper(0.66), get_cubic_taper(-0.74))
    taper_params = None
    dc_radius = 25
    sep = 30

    test_interport_w = 25
    mesh_interport_w = 50
    mesh_phaseshift_l = 100
    clearout_dim = (90, 6.5)
    gap_w = 0.3
    test_bend_dim = test_interport_w / 2 - gap_w / 2 - waveguide_w / 2
    trace_locations = (168, 440 - 168)  # measured in layout

    # Basic components

    dc = chip.custom_dc(bend_dim=(dc_radius, test_bend_dim))[0]
    mesh_dc = chip.pdk_dc(radius=dc_radius, interport_w=mesh_interport_w)
    tap = chip.bidirectional_tap(10, mesh_bend=True)
    anchor = chip.nems_anchor()
    ps = chip.nems_ps(waveguide_w=waveguide_w, phaseshift_l=length, wg_taper=taper_params, gap_taper=taper_params,
                      taper_ls=taper_ls, anchor=anchor, pad_dim=None, clearout_box_dim=clearout_dim, tap_sep=(tap, sep))

    ps_no_anchor = chip.nems_ps(waveguide_w=waveguide_w, phaseshift_l=length, wg_taper=taper_params,
                                gap_taper=taper_params, taper_ls=taper_ls)

    tdc = chip.nems_tdc(waveguide_w=0.48, nanofin_w=0.22,
                        interaction_l=41, dc_gap_w=0.2, beam_gap_w=0.15,
                        bend_dim=(10, 20), pad_dim=(50, 5, 2), anchor=None,
                        middle_fin_dim=None, use_radius=True,
                        clearout_box_dim=(65, 3), dc_taper_ls=(5, 5),
                        dc_taper=((0, -0.16), (0, 0.7)), beam_taper=((0, -0.16), (0, 0.7)),
                        clearout_etch_stop_grow=0.5, diff_ps=None, name='nems_tdc')

    tdc_notaper = chip.nems_tdc(waveguide_w=0.48, nanofin_w=0.22,
                                interaction_l=41, dc_gap_w=0.2, beam_gap_w=0.15,
                                bend_dim=(10, 20), pad_dim=(50, 5, 2), anchor=None,
                                middle_fin_dim=None, use_radius=True,
                                clearout_box_dim=(65, 3), dc_taper_ls=(0,), clearout_etch_stop_grow=0.5,
                                diff_ps=None, name='nems_tdc')

    # Mesh generation

    thermal_ps = chip.thermal_ps((tap, sep))
    dc_dummy = chip.waveguide(mesh_dc.pin['b0'].x - mesh_dc.pin['a0'].x)
    mzi_node_nems = chip.mzi_node(chip.double_ps(ps, mesh_interport_w, name='nems_double_ps'), mesh_dc)
    mzi_node_thermal = chip.mzi_node(chip.double_ps(thermal_ps, mesh_interport_w, name='thermal_double_ps'), mesh_dc)
    tdc_node = chip.tdc_node(chip.double_ps(ps, mesh_interport_w), tdc, tap, tap)
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
    autoroute_simple_1 = chip.autoroute_turn(7, turn_radius=8)
    autoroute_simple_2 = chip.autoroute_turn(7, level=2, turn_radius=8)

    # Test structures

    psv3_gap = [
        chip.singlemode_ps(chip.nems_ps(gap_w=gap_w), interport_w=test_interport_w, phaseshift_l=mesh_phaseshift_l)
        for gap_w in (0.2, 0.25, 0.3, 0.35, 0.4)]
    testing_tap_line = chip.testing_tap_line(15)
    gridsearch = chip.ps_tester(testing_tap_line, psv3_gap, dc)

    # Chip construction

    nems = nems_mesh.put(0, 750, flip=True)
    thermal = thermal_mesh.put(0, 1000)
    input_interposer = interposer.put(thermal.pin['a4'])
    output_interposer = interposer.put(thermal.pin['b4'], flip=True)
    mzi_node_thermal.put(input_interposer.pin['a6'])
    mzi_node_nems.put(input_interposer.pin['a7'], flip=True)
    bp_array.put(-180, -40)
    eu_array.put(-180, 200)
    bp_array.put(-180, 1700)
    eu_array.put(-180, 1340)
    for layer in range(15):
        autoroute_simple_1.put(layer * 450, 550, flop=True)
        autoroute_simple_2.put(layer * 450, 550, flop=True)
        autoroute_simple_1.put(layer * 450 + 178, 550)
        autoroute_simple_2.put(layer * 450 + 178, 550)
    for n in range(9):
        gridsearch.put(8000 + n * 400, 200)

    nd.export_gds(filename='test.gds')
