import nazca as nd
from dphox.design.aim import AIMNazca, mzi_node, tdc_node

if __name__ == 'main':

    chip = AIMNazca(
        passive_filepath='/home/exx/Documents/research/dphox/aim_lib/APSUNY_v35a_passive.gds',
        waveguides_filepath='/home/exx/Documents/research/dphox/aim_lib/APSUNY_v35_waveguides.gds',
        active_filepath='/home/exx/Documents/research/dphox/aim_lib/APSUNY_v35a_active.gds',
    )

    waveguide_w = 0.48
    interport_w = 25
    gap_w = 0.3
    device_dc = interport_w / 2 - gap_w / 2 - waveguide_w / 2

    nems_mesh = chip.triangular_nems_mzi_mesh(
        n=5, waveguide_w=0.48, nanofin_w=0.1,
        nanofin_radius=2, connector_tether_dim=None,
        interport_w=50, arm_l=50,
        ps_gap_w=0.15, pad_dim=(50, 15, 2),
        contact_box_dim=(40, 10),
        clearout_box_dim=(50, 2), radius=15, end_l=30
    )

    thermal_mesh = chip.triangular_thermal_mzi_mesh(
        n=5, waveguide_w=0.48,
        interport_w=50, radius=15, end_l=20
    )

    interposer = chip.interposer(
        n=14, waveguide_w=0.48, period=50,
        final_period=127, radius=60, trombone_radius=10,
        self_coupling_extension_dim=(30, 200),
        with_gratings=True,
        horiz_dist=200
    )

    bend_dim = interport_w / 2 - gap_w / 2 - waveguide_w / 2

    dc = chip.custom_dc(bend_dim=(15, bend_dim))[0]
    psv3_gap = [chip.nems_singlemode_ps(gap_w=gap_w, interport_w=interport_w) for i, gap_w in
                enumerate((0.2, 0.25, 0.3, 0.35, 0.4))]
    psv3_taper = [chip.nems_singlemode_ps(gap_taper=(0, -taper_change), wg_taper=(0, -taper_change), taper_l=5,
                                          interport_w=interport_w) for i, taper_change in
                  enumerate((0.2, 0.25, 0.3, 0.35, 0.4))]
    psv3_tether = [chip.nems_singlemode_ps(gap_w=gap_w, connector_tether_dim=(2, 0.5, 50, 0.15),
                                           interport_w=interport_w) for i, gap_w in enumerate((0.2, 0.3, 0.4))]

    tdc_tether = chip.nems_tdc(waveguide_w=0.48, nanofin_w=0.1, nanofin_radius=2, interaction_l=30, end_l=5,
                               dc_gap_w=0.15, beam_gap_w=0.2,
                               bend_dim=(20, 10), pad_dim=(30, 5, 2), use_radius=True, dc_taper_l=5,
                               dc_taper=(0, -0.35), beam_taper=(0, -0.2),
                               contact_box_dim=(10, 5), clearout_box_dim=(15, 2),
                               connector_tether_dim=(2, 0.5, 40, 0.15), middle_fin_dim=None)

    ps_tether = chip.nems_ps(waveguide_w=0.48, nanofin_w=0.1, nanofin_radius=2, phaseshift_l=50,
                             end_l=5, connector_tether_dim=(2, 0.5, 50, 0.15), pad_dim=(50, 7.5, 2), gap_w=0.15,
                             taper_l=0,
                             contact_box_dim=(50, 5), clearout_box_dim=(50, 3))

    ps = chip.nems_ps()

    tdc = chip.nems_tdc(waveguide_w=0.48, nanofin_w=0.1, nanofin_radius=2, interaction_l=40, end_l=5, dc_gap_w=0.3,
                        beam_gap_w=0.15,
                        bend_dim=(10, 10), pad_dim=(10, 5, 2), use_radius=True, dc_taper_l=5, dc_taper=(0, -0.3),
                        beam_taper=(0, -0.2),
                        contact_box_dim=(10, 5), clearout_box_dim=(15, 2))

    testing_tap_line = chip.testing_tap_line(25)

    detector = chip.pdk_cells['cl_band_photodetector_digital']
    grating = chip.pdk_cells['cl_band_vertical_coupler_si']

    with nd.Cell('tier1') as gridsearch:
        line = testing_tap_line.put()
        port_idx = 0
        for i, ps in enumerate(psv3_gap):
            port_idx += 1
            mzi_node(ps, dc, include_input_ps=False, detector=detector).put(line.pin[f'a{port_idx}'])
        for i, ps in enumerate(psv3_taper):
            port_idx += 1
            mzi_node(ps, dc, include_input_ps=False, detector=detector).put(line.pin[f'a{port_idx}'])
        for i, ps in enumerate(psv3_tether):
            port_idx += 1
            mzi_node(ps, dc, include_input_ps=False, detector=detector).put(line.pin[f'a{port_idx}'])

    nems = nems_mesh.put(0, 750, flip=True)
    thermal = thermal_mesh.put(0, 1000)
    interposer.put(thermal.pin['a4'])
    interposer.put(thermal.pin['b4'], flip=True)
    tdc_tether.put(2000, 1000)
    ps_tether.put(3000, 1200)
    tdc.put(4000, 1400)
    ps.put(5000, 1600)
    gridsearch.put(8000, 0)

    nd.export_gds(filename='test.gds')
