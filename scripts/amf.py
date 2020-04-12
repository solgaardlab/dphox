import nazca as nd
import numpy as np
from simphox.gds.nazca import PhotonicChip, NazcaVisualizer
from simphox.constants import AMF_STACK

if __name__ == 'main':
    waveguide_w = 0.5
    interport_w = 70
    arm_l = 80
    gap_w = 0.3
    interaction_l = 40
    interaction_l_id = 5.72
    gap_w_id = 1
    radius_id = (50 - 2 * waveguide_w - gap_w) / 4
    cp_radius = 35
    trench_gap = 12

    mzi_kwargs = {
        'gap_w': gap_w,
        'interaction_l': interaction_l,
        'interport_w': interport_w,
        'arm_l': arm_l,
        'radius': cp_radius,
        'trench_gap': trench_gap
    }

    idmzi_kwargs = {
        'gap_w': gap_w_id,
        'interaction_l': interaction_l_id,
        'interport_w': 50,
        'arm_l': arm_l,
        'radius': radius_id,
        'trench_gap': 12
    }

    idcp_kwargs = {
        'gap_w': gap_w_id,
        'interaction_l': interaction_l_id,
        'interport_w': 50,
        'arm_l': arm_l,
        'radius': radius_id
    }

    chip = PhotonicChip(AMF_STACK, waveguide_w)
    CHIPLET_SEP = 170

    with nd.Cell('meshes_chiplet') as meshes_chiplet:
        # useful constants
        cp_len = 308.937
        min_dist = 163.5
        ground_dist = 87

        # triangular meshes

        mesh = chip.triangular_mesh(n=6, **mzi_kwargs).put(0, 0)
        chip.triangular_mesh(n=6, **mzi_kwargs, ignore_internal_sampling=True).put(0, 15 * interport_w, flip=True)

        # small test structures

        splitter_tree = chip.splitter_tree_4(**mzi_kwargs, tap_notch=1)
        splitter_tree.put(0, 6 * interport_w)
        mzi_no_tap = chip.mzi(**mzi_kwargs, with_grating_taps=False, with_gratings=False, output_phase_shift=True,
                              input_phase_shift=False)
        mzi_tap = chip.mzi(**mzi_kwargs, with_grating_taps=True, with_gratings=False, output_phase_shift=True,
                           input_phase_shift=False)
        mzi_no_tap.put(mesh.bbox[2] - mzi_no_tap.bbox[2], 8 * interport_w)
        mzi_tap.put(mesh.bbox[2] - mzi_tap.bbox[2], 6 * interport_w)

        gratings_turn = chip.grating_array(4, period=70, connector_x=20, turn_radius=10)
        gratings_3 = chip.grating_array(3, period=70, connector_x=20, turn_radius=10)  # length=20)
        gratings_4 = chip.grating_array(4, period=70, connector_x=10)
        gratings_2 = chip.grating_array(2, period=70)
        gratings_4.put(splitter_tree.bbox[2], 6 * interport_w)
        gratings_4.put(mesh.bbox[2] - mzi_no_tap.bbox[2], 6 * interport_w, flop=True)
        chip.cutback_coupler_test(3, gap_w, interaction_l, interport_w, cp_radius).put(cp_len * 5.5 + arm_l,
                                                                                       6 * interport_w)
        gratings_4.put(cp_len * 5.5 + arm_l, 6 * interport_w, flop=True)
        chip.sampling_test(gap_ws=[0.3, 0.35, 0.4, 0.45], **mzi_kwargs).put(cp_len * 11.75 + arm_l, 6 * interport_w)
        chip.coupler(gap_w, interaction_l, interport_w, arm_l, cp_radius, with_gratings=False).put(cp_len * 8.5 + arm_l,
                                                                                                   6 * interport_w)
        gratings_4.put(cp_len * 11.75 + arm_l, 6 * interport_w, flop=True)
        gratings_4.put(cp_len * 13.75 + 2 * arm_l, 6 * interport_w)
        gratings_2.put(cp_len * 8.5 + arm_l, 6 * interport_w, flop=True)
        gratings_2.put(cp_len * 9.5 + arm_l, 6 * interport_w)

        # gratings and interposers

        interposer = chip.interposer(16, period=70, final_period=127, radius=75, horiz_dist=400)
        interposer.put(0, 0, flop=True)
        interposer.put(mesh.bbox[2])

        connection_array = np.asarray([
            [0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 0, 0, 1],
            [0, 0, 1, 1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 0, 0],
        ])

        end_x = []
        end_lengths = []
        widths = []

        num_layers = connection_array.shape[0]

        for idx, cxns in enumerate(connection_array):
            num_cxns = np.sum(cxns)
            start_x = min_dist + idx * cp_len
            chip.autoroute_turn(cxns, period=70, final_period=20,
                                pin_prefix=f'v{idx // 2}.{idx % 2}.').put(start_x, 0)
            chip.autoroute_turn(cxns, period=70, final_period=15,
                                pin_prefix=f'g{idx // 2}.{idx % 2}.', connector_x=65).put(start_x - ground_dist, 0,
                                                                                          flop=True)
            chip.autoroute_turn(cxns, period=70, final_period=20,
                                pin_prefix=f'v{idx // 2}.{idx % 2}.').put(start_x, 15 * interport_w, flip=True)
            chip.autoroute_turn(cxns, period=70, final_period=15,
                                pin_prefix=f'g{idx // 2}.{idx % 2}.', connector_x=65).put(start_x - ground_dist,
                                                                                          15 * interport_w,
                                                                                          flop=True, flip=True)
            end_x.append(start_x - (num_cxns + 1) * 7.5 - 65 - ground_dist)
            end_x.append(start_x + np.arange(num_cxns) * 20 + 15)
            if idx < num_layers // 2:
                end_lengths.append((num_cxns) * 15)
                end_lengths.append(np.arange(num_cxns) * 15)
            else:
                end_lengths.append(0)
                end_lengths.append(np.flipud(np.arange(num_cxns)) * 15)
            widths.append(num_cxns * 15)
            widths.append(np.ones(num_cxns) * 15)

        end_x, end_lengths, widths = np.hstack(end_x), np.hstack(end_lengths), np.hstack(widths)

        # bond pad arrays
        pad_start = -900
        bond_pad_array_top = chip.bond_pad_array(n_pads=39)
        bond_pad_array_bot = chip.bond_pad_array(n_pads=38)
        bond_pad_array_bot.put(pad_start + 100, -7 * interport_w)
        bond_pad_array_top.put(pad_start, -7 * interport_w - 200)
        bond_pad_array_bot.put(pad_start + 100, 23 * interport_w + 30)
        bond_pad_array_top.put(pad_start, 23 * interport_w + 230)

        pad_routes = chip.autoroute_interposer(end_x=end_x - pad_start, widths=widths, start_length=0,
                                               end_lengths=end_lengths, start_y=-7 * interport_w - 200, end_y=-15)

        pad_routes.put(pad_start, 0)
        pad_routes.put(pad_start, 15 * interport_w, flip=True)

    meshes_chiplet.put(CHIPLET_SEP / 2 - meshes_chiplet.bbox[0], 200 - meshes_chiplet.bbox[1])

    meshes_chiplet_x = meshes_chiplet.bbox[2] - meshes_chiplet.bbox[0] + CHIPLET_SEP / 2

    with nd.Cell('sensor_and_inv_design_chiplet') as sensor_and_inv_design_chiplet:
        # sensor network and splitter trees

        radius = 10
        cp_len = 317.282
        ground_dist = 87 - 0.004
        inter_heater_dist = 240.782
        inter_heater_dist_2 = 401.556
        horiz_dist = 250

        sensor_layer = chip.splitter_layer_4(**mzi_kwargs, tap_notch=0)
        sensor_layer.put(0, 0)
        splitter_tree_dist = sensor_layer.bbox[2] + 9 * radius
        splitter_tree = chip.splitter_tree_4(**mzi_kwargs, tap_notch=0, input_phase_shift=True)
        splitter_tree.put(splitter_tree_dist)
        splitter_tree.put(splitter_tree_dist, 4 * interport_w)

        # test structures
        splitter_tree_x = sensor_layer.bbox[2] + 9 * radius
        splitter_tree_test = chip.splitter_tree_4(**mzi_kwargs, tap_notch=0, input_phase_shift=False)
        splitter_tree_test.put(-horiz_dist - 150, 19 * interport_w)

        with nd.Cell('invdes_mesh') as invdes_mesh:
            splitter_tree_invdes = chip.splitter_tree_4(**idmzi_kwargs, tap_notch=0, input_phase_shift=False,
                                                        interaction_block=True)
            splitter_tree_invdes.put(0, 100)
            mzi_invdes = chip.mzi(**idmzi_kwargs, with_grating_taps=False,
                                  with_gratings=False, output_phase_shift=False, tap_notch=0,
                                  input_phase_shift=False, interaction_block=True).put(0, 0)
            coupler_invdes = chip.coupler(**idcp_kwargs, interaction_block=True, with_gratings=False).put(0, 300)

            grating = nd.netlist.load_gds('grating_jv.gds', newcellname='grating_jv')
            coupler = nd.netlist.load_gds('coupler.gds', newcellname='coupler_id')

            for i in range(4):
                grating.put(0, (i + 2) * 50, flop=True)
                grating.put(splitter_tree_invdes.bbox[2], (i + 2) * 50)
            for i in range(2):
                grating.put(0, i * 50, flop=True)
                grating.put(mzi_invdes.bbox[2], i * 50)
                grating.put(0, i * 50 + 300, flop=True)
                grating.put(coupler_invdes.bbox[2], i * 50 + 300)
            start_coupler_x = arm_l + 2 * radius_id
            coupler.put(start_coupler_x, 2 * radius_id - 0.1 + 300)
            for i in range(3):
                for j in range(2):
                    s = start_coupler_x + 2 * radius_id + interaction_l_id
                    y = 2 * radius_id - 0.1 + 100 * i
                    coupler.put(start_coupler_x + s * j, y)
                    if i == 1:
                        coupler.put(s * (j + 2) + 2 * radius_id, 2 * radius_id - 0.1 + 150)
                    elif i == 2:
                        chip.waveguide_ic.strt(interaction_l_id).put(s * (j + 2) + 2 * radius_id,
                                                                     2 * radius_id - 0.1 + 200 + gap_w_id + waveguide_w)
                    else:
                        chip.waveguide_ic.strt(interaction_l_id).put(s * (j + 2) + 2 * radius_id,
                                                                     2 * radius_id - 0.1 + 100)

        id_x = splitter_tree_test.bbox[2] - 250
        invdes_mesh.put(id_x, 22 * interport_w, flip=True)

        mzi = chip.mzi(**mzi_kwargs, with_grating_taps=False,
                       with_gratings=False, output_phase_shift=False, tap_notch=0,
                       input_phase_shift=False).put(-horiz_dist - 150, 17 * interport_w)
        chip.cutback_coupler_test(4, gap_w, interaction_l, interport_w, cp_radius).put(-horiz_dist - 150,
                                                                                       13 * interport_w)
        chip.coupler(gap_w, interaction_l, interport_w, arm_l, cp_radius, with_gratings=False).put(-horiz_dist,
                                                                                                   11 * interport_w)

        # gratings and interposers

        interposer_8 = chip.interposer(6, period=140, final_period=127, radius=32, horiz_dist=horiz_dist)
        interposer_8.put(0, 0, flop=True)

        gratings_4 = chip.grating_array(4, period=70)
        gratings_2 = chip.grating_array(2, period=70)
        gratings_6 = chip.grating_array(6, period=70)
        gratings_6.put(-horiz_dist - 150, 17 * interport_w, flop=True)
        gratings_4.put(-horiz_dist + splitter_tree_test.bbox[2] - 150, 19 * interport_w)
        gratings_4.put(-horiz_dist - 150, 13 * interport_w, flop=True)
        gratings_2.put(mzi.bbox[2], 17 * interport_w)
        gratings_2.put(-horiz_dist, 11 * interport_w, flop=True)
        gratings_2.put(-horiz_dist + cp_len, 11 * interport_w)

        # sensor-to-fiber connectors

        sensor_connector = chip.sensor_connector(n=4, radius=radius, wrap_l=5, curr_period=140, final_period=70,
                                                 sensor_x=sensor_layer.bbox[2] + 185 + horiz_dist, sensor_y=0,
                                                 connector_x=sensor_layer.bbox[2] + 185 + horiz_dist + 4 * radius,
                                                 connector_y=0)
        mux_connector = chip.sensor_connector(n=4, radius=radius, curr_period=140, final_period=70,
                                              connector_x=4 * radius, connector_y=0)
        u_connector_top = chip.u_connector(radius=radius, connector_xl=4 * radius,
                                           connector_xr=splitter_tree_dist + splitter_tree.bbox[2],
                                           connector_y=2 * interport_w - 2 * radius)
        u_connector_bot = chip.u_connector(radius=radius, connector_xl=5 * radius,
                                           connector_xr=splitter_tree_dist + splitter_tree.bbox[2],
                                           connector_y=8 * interport_w - 2 * radius)

        sensor_connector.put(sensor_layer.bbox[2], 0)
        mux_connector.put(sensor_layer.bbox[2], interport_w)
        u_connector_bot.put(splitter_tree_dist + splitter_tree.bbox[2], 2 * interport_w)
        u_connector_top.put(splitter_tree_dist + splitter_tree.bbox[2], 6 * interport_w)

        # traces
        cxns = [1, 0, 1, 0, 1, 0, 1, 0]
        cxns_tree = [0, 1, 0, 0, 0, 1, 0, 0]

        start_splitter_x = splitter_tree_x + 2 * arm_l + 3.5

        end_x = []
        end_lengths = []
        widths = []

        chip.autoroute_turn(cxns, period=70, final_period=15, pin_prefix=f'g0.').put(cp_len + 7, 0)
        chip.autoroute_turn(cxns, period=70, final_period=25, pin_prefix=f'v0.').put(cp_len + 7 - ground_dist, 0,
                                                                                     flop=True)
        end_x.extend([cp_len + 7 - 25 * np.flipud(np.arange(4)) - 15 - ground_dist, cp_len + 7 + 37.5])
        end_lengths.extend([25 * np.arange(4), 0])
        widths.extend([np.ones(4) * 15, 60])
        chip.autoroute_turn(cxns, period=70, final_period=15, pin_prefix=f'v1.').put(start_splitter_x, 0)
        chip.autoroute_turn(cxns, period=70, final_period=25, pin_prefix=f'g1.').put(start_splitter_x - ground_dist, 0,
                                                                                     flop=True)
        chip.autoroute_turn(cxns, period=70, final_period=25, pin_prefix=f'g2.').put(
            start_splitter_x + inter_heater_dist, 0)
        chip.autoroute_turn(cxns, period=70, final_period=15, pin_prefix=f'v2.').put(
            start_splitter_x + inter_heater_dist - ground_dist, 0, flop=True)
        end_x.extend([start_splitter_x - 25 * np.flipud(np.arange(4)) - ground_dist - 15, start_splitter_x + 37.5,
                      start_splitter_x + inter_heater_dist - 37.5 - ground_dist,
                      start_splitter_x + inter_heater_dist + 25 * np.arange(4) + 15])
        #     end_lengths.extend([25 * np.arange(4), 0, 0, 25 * np.flipud(np.arange(4))])
        end_lengths.extend([25 * np.arange(4), 0, 0, np.zeros(4)])
        widths.extend([np.ones(4) * 15, 60, 60, np.ones(4) * 15])
        chip.autoroute_turn(cxns_tree, period=70, final_period=25, pin_prefix=f'v3.').put(
            start_splitter_x + inter_heater_dist * 2, 0)
        chip.autoroute_turn(cxns_tree, period=70, final_period=15, pin_prefix=f'g3.').put(
            start_splitter_x + inter_heater_dist * 2 - ground_dist, 0, flop=True)
        chip.autoroute_turn(cxns_tree, period=70, final_period=25, pin_prefix=f'v4.').put(
            start_splitter_x + inter_heater_dist * 3, 0)
        t = chip.autoroute_turn(cxns_tree, period=70, final_period=15, pin_prefix=f'g4.').put(
            start_splitter_x + inter_heater_dist * 3 - ground_dist, 0, flop=True)
        end_x.extend([start_splitter_x + inter_heater_dist * 2 - ground_dist - 22.5,
                      start_splitter_x + inter_heater_dist * 2 + 25 * np.arange(2) + 15,
                      start_splitter_x + inter_heater_dist * 3 - ground_dist - 22.5,
                      start_splitter_x + inter_heater_dist * 3 + 25 * np.arange(2) + 15])
        end_lengths.extend([0, 25 * np.flipud(np.arange(2)), 0, 25 * np.flipud(np.arange(2))])
        widths.extend([30, np.ones(2) * 15, 30, np.ones(2) * 15])

        # bond pad arrays
        end_x, end_lengths, widths = np.hstack(end_x), np.hstack(end_lengths), np.hstack(widths)
        pad_start = -100
        bond_pad_array_top = chip.bond_pad_array(n_pads=11)
        bond_pad_array_bot = chip.bond_pad_array(n_pads=10)
        bond_pad_array_bot.put(pad_start, -3 * interport_w)
        bond_pad_array_top.put(pad_start - 100, -3 * interport_w - 200)
        bond_pad_array_bot.put(pad_start - horiz_dist + 150, 26 * interport_w + 30)
        bond_pad_array_top.put(pad_start - horiz_dist + 50, 26 * interport_w + 230)

        pad_routes = chip.autoroute_interposer(end_x=end_x - pad_start + 100, widths=widths, start_length=0,
                                               end_lengths=end_lengths, start_y=-3 * interport_w - 200, end_y=-15)

        pad_routes.put(pad_start - 100, 0)

        cxns_top1 = [1, 1, 0, 1, 1, 0]
        cxns_top2 = [1, 1, 1, 1, 0, 0]

        cxns_top_id1 = [1, 0, 1, 0, 1, 0]
        cxns_top_id2 = [0, 0, 0, 1, 0, 0]

        cxns_top_id3 = [1, 1]

        cp_len_id = interaction_l_id + 4 * radius_id

        with nd.Cell(name='test_routes') as test_routes:
            end_x = []
            end_lengths = []
            widths = []

            start_x = cp_len + 7
            chip.autoroute_turn(cxns_top1, period=70, final_period=25, pin_prefix=f'vt0.').put(start_x, 0)
            chip.autoroute_turn(cxns_top1, period=70, final_period=15, pin_prefix=f'gt0.').put(start_x - ground_dist, 0,
                                                                                               flop=True)
            end_x.extend([start_x - ground_dist - 37.5, start_x + 25 * np.arange(4) + 15])
            end_lengths.extend([0, np.zeros(4)])  # 25 * np.arange(4)
            widths.extend([60, np.ones(4) * 15])
            chip.autoroute_turn(cxns_top2, period=70, final_period=25, pin_prefix=f'vt1.').put(
                start_x + inter_heater_dist_2, 0)
            chip.autoroute_turn(cxns_top2, period=70, final_period=15, pin_prefix=f'gt1.').put(
                start_x + inter_heater_dist_2 - ground_dist, 0, flop=True)
            end_x.extend([start_x + inter_heater_dist_2 - ground_dist - 37.5,
                          start_x + inter_heater_dist_2 + 25 * np.arange(4) + 15])
            end_lengths.extend([0, np.zeros(4)])
            widths.extend([60, np.ones(4) * 15])

            chip.autoroute_turn(cxns_top_id1, period=50, final_period=20, pin_prefix=f'vt2.').put(
                id_x + horiz_dist + 313.5 + cp_len_id, 0)
            chip.autoroute_turn(cxns_top_id1, period=50, final_period=15, pin_prefix=f'gt2.', connector_x=40).put(
                id_x + horiz_dist + 313.5 + cp_len_id - ground_dist, 0, flop=True)
            end_x.extend([id_x + horiz_dist + 313.5 + cp_len_id - ground_dist - 70,
                          id_x + horiz_dist + 313.5 + cp_len_id + 20 * np.arange(3) + 15])
            end_lengths.extend([0, np.zeros(3)])
            widths.extend([45, np.ones(3) * 15])

            chip.autoroute_turn(cxns_top_id2, period=50, final_period=20, pin_prefix=f'vt3.').put(
                id_x + horiz_dist + 313.5 + 3 * cp_len_id + arm_l, 0)
            chip.autoroute_turn(cxns_top_id2, period=50, final_period=15, pin_prefix=f'gt3.').put(
                id_x + horiz_dist + 313.5 + 3 * cp_len_id + arm_l - ground_dist, 0, flop=True)
            end_x.extend([id_x + horiz_dist + 313.5 + 3 * cp_len_id + arm_l - ground_dist - 15,
                          id_x + horiz_dist + 313.5 + 3 * cp_len_id + arm_l + 15])
            end_lengths.extend([0, 0])
            widths.extend([15, 15])

            id_x2 = id_x + horiz_dist + 313.5 + 4 * cp_len_id + arm_l * 2 + 300

            chip.heater(arm_l).put(id_x2 - ground_dist + 3.5, 0)
            chip.heater(arm_l).put(id_x2 - ground_dist + 3.5, 50)
            chip.trench_ic.strt(arm_l).put(id_x2 - ground_dist + 3.5, trench_gap)
            chip.trench_ic.strt(arm_l).put(id_x2 - ground_dist + 3.5, 50 + trench_gap)
            chip.trench_ic.strt(arm_l).put(id_x2 - ground_dist + 3.5, -trench_gap)
            chip.trench_ic.strt(arm_l).put(id_x2 - ground_dist + 3.5, 50 - trench_gap)
            chip.autoroute_turn(cxns_top_id3, period=50, final_period=20, pin_prefix=f'vt1.', turn_offset=True,
                                connector_x=25).put(id_x2, 0)
            chip.autoroute_turn(cxns_top_id3, period=50, final_period=15, pin_prefix=f'gt1.', connector_x=40).put(
                id_x2 - ground_dist, 0, flop=True)
            end_x.extend([id_x2 - ground_dist - 62.5,
                          id_x2 + 40 + 20 * np.arange(2)])
            end_lengths.extend([0, np.zeros(2)])
            widths.extend([30, 15 * np.ones(2)])

            end_x, end_lengths, widths = np.hstack(end_x), np.hstack(end_lengths), np.hstack(widths)
            pad_routes_top = chip.autoroute_interposer(end_x=end_x - pad_start - 200, widths=widths,
                                                       start_length=0,
                                                       end_lengths=end_lengths, start_y=-3 * interport_w - 200,
                                                       end_y=-15)
            pad_routes_top.put(pad_start + 200)

        test_routes.put(-horiz_dist - 150, 22 * interport_w, flip=True)

        drop_port_r = chip.drop_port_array([0, 1, 3, 4, 5, 7], period=70, final_taper_width=0.2).put(
            splitter_tree.bbox[2] + splitter_tree_x, 0)
        drop_port_l = chip.drop_port_array([1, 3, 5, 7], period=70, final_taper_width=0.2).put(0, flop=True)

        invdes_chiplet = nd.netlist.load_gds('invdes_chiplet.gds', newcellname='invdes_chiplet', cellname='Design2')
        invdes_chiplet.put(-invdes_chiplet.bbox[0] + invdes_chiplet.bbox[2] + splitter_tree.bbox[2] + 250,
                           22 * interport_w - invdes_chiplet.bbox[3] + 25.109, flop=True)

    sensor_and_inv_design_chiplet_x = sensor_and_inv_design_chiplet.bbox[2] - sensor_and_inv_design_chiplet.bbox[
        0] + meshes_chiplet_x + 2 * CHIPLET_SEP

    sensor_and_inv_design_chiplet.put(sensor_and_inv_design_chiplet_x - sensor_and_inv_design_chiplet.bbox[2],
                                      -sensor_and_inv_design_chiplet.bbox[1] + 200)

    post_processing_chiplet = nd.netlist.load_gds('post_processing_chiplet.gds', newcellname='post_processing_chiplet')
    post_processing_chiplet_x = -post_processing_chiplet.bbox[0] + sensor_and_inv_design_chiplet_x + 2 * CHIPLET_SEP
    post_processing_chiplet.put(post_processing_chiplet_x, 175)

    achip_chiplet = nd.netlist.load_gds('achip_chiplet.gds', newcellname='achip_chiplet')
    achip_chiplet_x = post_processing_chiplet_x + post_processing_chiplet.bbox[2] - post_processing_chiplet.bbox[
        0] + CHIPLET_SEP
    achip_chiplet.put(achip_chiplet_x + 2 * CHIPLET_SEP - 20, -achip_chiplet.bbox[1] + 100)

    trench_chiplet = chip.chiplet_trench([meshes_chiplet_x + CHIPLET_SEP,
                                          sensor_and_inv_design_chiplet_x + CHIPLET_SEP,
                                          post_processing_chiplet_x + post_processing_chiplet.bbox[2] -
                                          post_processing_chiplet.bbox[0] + CHIPLET_SEP - 30], y_line_w=200).put(0, 0)

    from datetime import date

    nd.export_gds(filename=f'amf-{str(date.today())}.gds')
    # import matplotlib.pyplot as plt
    # plt.figure(dpi=500, figsize=(8, 3))
    # nv = NazcaVisualizer()
    # nv.add_cell(meshes_chiplet)
    # nv.plot(plt.axes())
    # plt.savefig('nazca.pdf')