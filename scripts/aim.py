import nazca as nd
from dphox.design.aim import AIMNazca

if __name__ == 'main':

    chip = AIMNazca(
        passive_filepath='/home/exx/Documents/research/dphox/aim_lib/APSUNY_v35a_passive.gds',
        waveguides_filepath='/home/exx/Documents/research/dphox/aim_lib/APSUNY_v35_waveguides.gds',
        active_filepath='/home/exx/Documents/research/dphox/aim_lib/APSUNY_v35a_active.gds',
    )

    waveguide_w = .48
    length = 100
    taper_ls = (5, 5)
    taper_params = ((0, 0.66), (0, -0.74))
    dc_radius = 15

    test_interport_w = 25
    mesh_interport_w = 50
    mesh_phaseshift_l = 100
    clearout_dim = (90, 6.5)
    gap_w = 0.3
    test_bend_dim = test_interport_w / 2 - gap_w / 2 - waveguide_w / 2

    # Basic components

    dc = chip.custom_dc(bend_dim=(dc_radius, test_bend_dim))[0]
    t_waveguide = chip.waveguide(length, taper_ls=taper_ls, taper_params=taper_params)
    anti_waveguide = chip.waveguide(length, taper_ls=taper_ls, taper_params=taper_params, symmetric=False)

    anchor = chip.nems_anchor()
    ps = chip.nems_ps(waveguide_w=waveguide_w, phaseshift_l=length, wg_taper=taper_params, gap_taper=taper_params,
                      taper_ls=taper_ls, anchor=anchor, pad_dim=None, clearout_box_dim=clearout_dim)

    ps_no_anchor = chip.nems_ps(waveguide_w=waveguide_w, phaseshift_l=length, wg_taper=taper_params,
                                gap_taper=taper_params, taper_ls=taper_ls)

    tdc = chip.nems_tdc(waveguide_w=0.48, nanofin_w=0.22,
                        interaction_l=41, dc_gap_w=0.2, beam_gap_w=0.15,
                        bend_dim=(10, 20), pad_dim=(50, 5, 2), anchor=None,
                        middle_fin_dim=None, use_radius=True, contact_box_dim=(50, 10),
                        clearout_box_dim=(65, 3), dc_taper_ls=(5, 5),
                        dc_taper=((0, -0.16), (0, 0.7)), beam_taper=((0, -0.16), (0, 0.7)),
                        clearout_etch_stop_grow=0.5, diff_ps=None, name='nems_tdc')

    tdc_notaper = chip.nems_tdc(waveguide_w=0.48, nanofin_w=0.22,
                                interaction_l=41, dc_gap_w=0.2, beam_gap_w=0.15,
                                bend_dim=(10, 20), pad_dim=(50, 5, 2), anchor=None,
                                middle_fin_dim=None, use_radius=True, contact_box_dim=(50, 10),
                                clearout_box_dim=(65, 3), dc_taper_ls=(0,), clearout_etch_stop_grow=0.5,
                                diff_ps=None, name='nems_tdc')

    # Mesh generation

    mesh_dc = chip.pdk_dc(radius=dc_radius, interport_w=mesh_interport_w)
    dc_dummy = chip.waveguide(mesh_dc.pin['b0'].x - mesh_dc.pin['a0'].x)
    tap = chip.bidirectional_tap(10, mesh_bend=True)
    mzi_node_nems = chip.mzi_node(chip.double_ps(ps, mesh_interport_w, name='nems_double_ps'),
                                  mesh_dc, tap, tap, sep=30)
    mzi_node_thermal = chip.mzi_node(chip.double_ps(chip.thermal_ps(), mesh_interport_w, name='thermal_double_ps'),
                                     mesh_dc, tap, tap, sep=30)
    tdc_node = chip.tdc_node(chip.double_ps(ps, mesh_interport_w), tdc, tap, tap)
    mzi_dummy_nems = chip.mzi_dummy(ps, dc_dummy, tap, tap, sep=30)
    mzi_dummy_thermal = chip.mzi_dummy(chip.thermal_ps(), dc_dummy, tap, tap, sep=30)
    nems_mesh = chip.triangular_mesh(5, mzi_node_nems, mzi_dummy_nems, mesh_interport_w)
    thermal_mesh = chip.triangular_mesh(5, mzi_node_thermal, mzi_dummy_thermal, mesh_interport_w)

    interposer = chip.interposer(
        n=14, waveguide_w=0.48, period=50,
        final_period=127, radius=50, trombone_radius=10,
        self_coupling_extension_dim=(30, 200),
        with_gratings=True, horiz_dist=200
    )
    bp_array = chip.bond_pad_array((60, 3))

    # Test structures

    psv3_gap = [
        chip.singlemode_ps(chip.nems_ps(gap_w=gap_w), interport_w=test_interport_w, phaseshift_l=mesh_phaseshift_l)
        for gap_w in (0.2, 0.25, 0.3, 0.35, 0.4)]
    testing_tap_line = chip.testing_tap_line(15)
    gridsearch = chip.ps_tester(testing_tap_line, psv3_gap, dc)

    # Chip construction

    nems = nems_mesh.put(0, 750, flip=True)
    thermal = thermal_mesh.put(0, 1000)
    interposer.put(thermal.pin['a4'])
    interposer.put(thermal.pin['b4'], flip=True)
    bp_array.put(0, 0)
    bp_array.put(0, 1440)
    for n in range(10):
        gridsearch.put(7500 + n * 400, 200)

    nd.export_gds(filename='test.gds')
