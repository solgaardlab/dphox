import nazca as nd
import numpy as np
from simphox.design.aim import AIMPhotonicChip

if __name__ == 'main':

    chip = AIMPhotonicChip(
        passive_filepath='/Users/sunilpai/Documents/research/simphox/aim_lib/APSUNY_v35a_passive.gds',
        waveguides_filepath='/Users/sunilpai/Documents/research/simphox/aim_lib/APSUNY_v35_waveguides.gds'
    )

    waveguide_w = 0.5
    interport_w = 70
    arm_l = 150
    end_l = 202
    tdc_interaction_w = 100
    mzi_interation_w = 45
    gap_w = 0.3
    gap_w_id = 0.6
    cp_radius = 35
    trench_gap = 12

    dc_kwargs = {
        'gap_w': gap_w,
        'interaction_l': tdc_interaction_w,
        'interport_w': interport_w,
        'end_l': end_l,
        'radius': cp_radius
    }

    mzi_kwargs = {
        'gap_w': gap_w,
        'interaction_l': mzi_interation_w,
        'interport_w': interport_w,
        'end_l': end_l - 120,
        'arm_l': arm_l,
        'radius': cp_radius
    }

    with nd.Cell('mems_phase_shifter_chiplet') as mems_phase_shifter_chiplet:
        dc_l = chip.cl_band_splitter_4port_si.put(20, 700)
        upper_arm = chip.cl_band_waveguide_si(length=170).put(dc_l.pin['b0'])
        tap_upper = chip.cl_band_1p_tap_si.put(flip=True)
        lower_arm = chip.cl_band_waveguide_si(length=170).put(dc_l.pin['b1'])
        tap_lower = chip.cl_band_1p_tap_si.put()
        chip.cl_band_waveguide_si(angle=90).put(tap_upper.pin['b1'])
        chip.cl_band_waveguide_si(length=25).put()
        chip.cl_band_waveguide_si(angle=-90).put()
        chip.cl_band_waveguide_si(length=200).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)
        chip.cl_band_waveguide_si(angle=-90).put(tap_lower.pin['b1'])
        chip.cl_band_waveguide_si(length=25).put()
        chip.cl_band_waveguide_si(angle=90).put()
        chip.cl_band_waveguide_si(length=200).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)
        dc_r = chip.cl_band_splitter_4port_si.put(tap_upper.pin['b0'])
        chip.cl_band_waveguide_si(angle=-90).put(dc_l.pin['a0'])
        chip.cl_band_waveguide_si(angle=90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), 90)
        chip.cl_band_waveguide_si(angle=90).put(dc_l.pin['a1'])
        chip.cl_band_waveguide_si(angle=-90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), 90)
        chip.cl_band_waveguide_si(angle=90).put(dc_r.pin['b0'])
        chip.cl_band_waveguide_si(angle=-90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)
        chip.cl_band_waveguide_si(angle=-90).put(dc_r.pin['b1'])
        chip.cl_band_waveguide_si(angle=90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)
        dc = chip.dc(**dc_kwargs).put(0, 500)
        chip.cl_band_vertical_coupler_si.put(dc.pin['a0'].x, dc.pin['a0'].y, 90)
        chip.cl_band_vertical_coupler_si.put(dc.pin['a1'].x, dc.pin['a1'].y, 90)
        chip.cl_band_vertical_coupler_si.put(dc.pin['b0'].x, dc.pin['b0'].y, -90)
        chip.cl_band_vertical_coupler_si.put(dc.pin['b1'].x, dc.pin['b1'].y, -90)

        mzi = chip.mzi(**mzi_kwargs).put(0, -120)
        chip.cl_band_vertical_coupler_si.put(mzi.pin['a0'].x, mzi.pin['a0'].y, 90)
        chip.cl_band_vertical_coupler_si.put(mzi.pin['a1'].x, mzi.pin['a1'].y, 90)
        chip.cl_band_vertical_coupler_si.put(mzi.pin['b0'].x, mzi.pin['b0'].y, -90)
        chip.cl_band_vertical_coupler_si.put(mzi.pin['b1'].x, mzi.pin['b1'].y, -90)
        mps = chip.microbridge_ps(bridge_w=5, bridge_l=100,
                                  tether_l=10, tether_w=5,
                                  block_w=1, block_l=arm_l, radius=2)
        mps.put(mzi.pin['c1'], flip=True)
        mps.put(mzi.pin['c0'])
        mps.put(upper_arm.pin['a0'].x + 10, upper_arm.pin['a0'].y)
        mps.put(lower_arm.pin['a0'].x + 10, lower_arm.pin['a0'].y, flip=True)
        bridge_l = 100
        mtdc = chip.microbridge_ps(bridge_w=5, bridge_l=bridge_l,
                                   tether_l=10, tether_w=5,
                                   block_w=0.48, block_l=tdc_interaction_w, radius=1)
        mtdc.put(dc.pin['c1'], flip=True, flop=True)
        mtdc.put(dc.pin['c0'], flop=True)

        use_mps = [False, True, True, True, False]
        use_bus = [True, False, True, False, True]
        for idx, racetrack_l in enumerate(np.linspace(100, 500, 5)):
            interaction_l = 5
            radius = 20
            waveguide = chip.cl_band_waveguide_si(length=300).put(0, idx * 100, 0)
            chip.cl_band_vertical_coupler_si.put(waveguide.pin['a0'].x, waveguide.pin['a0'].y, 90)
            rr = chip.ring_resonator(radius=20, gap_w=0.2, racetrack_l=racetrack_l, interaction_l=interaction_l,
                                     interaction_angle=30).put(waveguide.pin['b0'])
            waveguide = chip.cl_band_waveguide_si(length=300).put(rr.pin['b0'])
            chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)
            if use_mps[idx]:
                # phase shifter
                mps.put(rr.pin['c0'].x + interaction_l / 2 - arm_l / 2,
                        rr.pin['c0'].y + waveguide_w + 2 * radius + gap_w / 2, flip=True)
            if use_bus[idx]:
                # bus
                chip.cl_band_waveguide_si(length=5).put(rr.pin['c0'].x + interaction_l / 2 - 2.5,
                                                        rr.pin['c0'].y + 2 * waveguide_w + 2 * radius + gap_w)

        shallow_trench = chip.shallow_trench(length=500, width=900)
        shallow_trench2 = chip.shallow_trench(length=500, width=600)
        shallow_trench.put(65, shallow_trench.bbox[3] / 2 + 100)
        shallow_trench2.put(35, -shallow_trench.bbox[3] / 2 - 480)

        tdc = nd.load_gds('tdc_v2.gds')  # insert rebecca's filepath (or scripted cell) here
        tdc.put(0, -550, flip=True)
        static = nd.load_gds('static.gds')  # insert nate's filepath (or scripted cell) here
        static.put(1100, -850)

    mems_phase_shifter_chiplet.put(0, 0)

    nd.export_gds(filename='mems_phase_shifter_chiplet.gds')

