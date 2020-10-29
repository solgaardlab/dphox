import itertools

import nazca as nd
from datetime import date
from dphox.gf import *
from dphox.layout import NazcaLayout
from dphox.constants import GF_PSUEDO_STACK, GF_PSUEDO_PDK

chip = NazcaLayout(passive_filepath='../../../20201016_sjby_gf_test/empty.gds', waveguides_filepath='../../../20201016_sjby_gf_test/empty.gds',
                   active_filepath='../../../20201016_sjby_gf_test/empty.gds', stack=GF_PSUEDO_STACK, pdk_dict=GF_PSUEDO_PDK,
                   accuracy=0.005, waveguide_w=0.5, pcb_style=True)


# Basic components

print('Compiling basic components...')

dc = GFDC().nazca_cell('dc', layer='seam')
# dc_short = AIMDC(bend_dim=(aggressive_dc_radius, test_bend_dim_short), gap_w=test_gap_w_short,
#                  interaction_l=22).nazca_cell('dc_short', layer='seam')
# dc_aggressive = AIMDC(bend_dim=(aggressive_dc_radius, test_bend_dim_aggressive), gap_w=test_gap_w_aggressive,
#                       interaction_l=9).nazca_cell('dc_aggressive', layer='seam')
# dc_invdes = AIMDC(bend_dim=(aggressive_dc_radius, test_bend_dim_invdes), gap_w=test_gap_w_invdes,
#                   interaction_l=5, coupler_boundary_taper_ls=(1,),
#                   coupler_boundary_taper=(cubic_taper(-0.16),)).replace(
#     Pattern.from_gds('alex_directional.gds')
# ).nazca_cell('dc_invdes', layer='seam')
# mesh_dc = chip.pdk_dc(radius=pdk_dc_radius, interport_w=mesh_interport_w)
# tap_detector = chip.bidirectional_tap(10, mesh_bend=True)
tdc = pull_apart_full_tdc.nazca_cell('tunable_splitter')
ps = pull_apart_full_ps.nazca_cell('phase shifter')
gnd_wg = chip.gnd_wg()
# mesh_ps = chip.device_linked([pull_apart_full_ps, sep, tap_detector])
alignment_mark = chip.alignment_mark()
alignment_mark_small = chip.alignment_mark((50, 5), name='alignment_mark_small')

grating = fake_grating_coupler.nazca_cell('dummy_c_band_grating_coupler')
pdk_dc = fake_pdk_dc.nazca_cell('dummy_c_band_2x2_Spliter')
detector = fake_photodetector.nazca_cell('dummy_c_band_photodetector')

delay_line_50 = chip.delay_line(waveguide_width=0.5, name='delay_line_50')
delay_line_200 = chip.delay_line(waveguide_width=0.5, delay_length=200, straight_length=100, flip=False, name='delay_line_200')

# Small Test Structure Arrays

print('Combining basic components...')


def pa_nems_MZI(length, no_MZI=False):
    with nd.Cell(f'pa_mems_mzi_{length}') as mems_mzi:
        # ps = pull_apart_full_ps.update(ps=ps_pull_apart.update(phaseshift_l=length)).nazca_cell(f'phase_shifter_{length}')
        ps = pull_apart_full_ps.update(
            ps=ps_pull_apart.update(phaseshift_l=length),
            anchor=pull_apart_anchor.update(pos_electrode_dim=(length - 10, 4, 0.5),
                                            fin_dim=(length, 0.22)),
            clearout_dim=(length, 0.3), gnd_box_h=11, single_metal=True
        ).nazca_cell(f'pull_apart_phase_shifter_{length}')
        if no_MZI:
            ps = chip.waveguide_ic.strt(length)
        buffer_length = 15
        gc = grating.put(0, 0, -180)
        chip.waveguide_ic.strt(buffer_length).put(gc.pin['a0'])
        dc1 = pdk_dc.put()

        chip.waveguide_ic.strt(buffer_length).put()
        delay_line_50.put()
        chip.waveguide_ic.strt(buffer_length).put()
        ps.put()
        chip.waveguide_ic.strt(buffer_length).put()
        dc2 = pdk_dc.put()
        chip.waveguide_ic.strt(buffer_length).put()
        detector.put()

        chip.waveguide_ic.strt(buffer_length).put(dc2.pin['a1'])
        dl_200 = delay_line_200.put()
        chip.waveguide_ic.strt_p2p(dl_200.pin['b0'], dc1.pin['b1']).put()
        chip.waveguide_ic.strt(buffer_length).put(dc1.pin['a1'])
        grating.put()
        chip.waveguide_ic.strt(buffer_length).put(dc2.pin['b1'])
        detector.put()
    return mems_mzi


def pi_nems_MZI(length, no_MZI=False):
    with nd.Cell(f'pi_mems_mzi_{length}') as mems_mzi:
        ps = pull_in_full_ps.update(
            ps=ps_pull_in.update(phaseshift_l=length),
            anchor=pull_apart_anchor.update(fin_dim=(length, 0.22), shuttle_dim=(length - 10, 3),
                                            pos_electrode_dim=None, gnd_electrode_dim=None,
                                            spring_dim=None, include_support_spring=False, shuttle_stripe_w=0),
            clearout_dim=(length, 0.3), gnd_box_h=8, single_metal=True
        ).nazca_cell(f'pull_in_phase_shifter_{length}')
        if no_MZI:
            ps = chip.waveguide_ic.strt(length)
        buffer_length = 15
        gc = grating.put(0, 0, -180)
        chip.waveguide_ic.strt(buffer_length).put(gc.pin['a0'])
        dc1 = pdk_dc.put()

        chip.waveguide_ic.strt(buffer_length).put()
        delay_line_50.put()
        chip.waveguide_ic.strt(buffer_length).put()
        ps.put()
        chip.waveguide_ic.strt(buffer_length).put()
        dc2 = pdk_dc.put()
        chip.waveguide_ic.strt(buffer_length).put()
        detector.put()

        chip.waveguide_ic.strt(buffer_length).put(dc2.pin['a1'])
        dl_200 = delay_line_200.put()
        chip.waveguide_ic.strt_p2p(dl_200.pin['b0'], dc1.pin['b1']).put()
        chip.waveguide_ic.strt(buffer_length).put(dc1.pin['a1'])
        grating.put()
        chip.waveguide_ic.strt(buffer_length).put(dc2.pin['b1'])
        detector.put()
    return mems_mzi


def tdc_taper(taper_length: float, taper_change: float):
    return dict(
        dc_taper_ls=(taper_length,),
        dc_taper=(cubic_taper(taper_change),),
        beam_taper=(cubic_taper(taper_change),)
    )


def tdc_test_cell(length, dc_gap_w=0.2):
    shuttle_dim = (50, 2)
    if length < shuttle_dim[0]:
        shuttle_dim = (np.around(length - 10, 0), 2)
    with nd.Cell(f'tunable_splitter_{length}_{dc_gap_w}_test_cell') as tdc_cell:
        buffer_length = 25
        # interaction_length = length
        length = length
        tdc = pull_apart_full_tdc.update(
            tdc=tdc_pull_apart.update(
                interaction_l=length, dc_gap_w=dc_gap_w,
                **tdc_taper(5, -0.1)
            ),
            anchor=pull_apart_anchor.update(pos_electrode_dim=(length - 10, 4, 0.5),
                                            spring_dim=(length, 0.2),
                                            fin_dim=(length, 0.2),
                                            shuttle_dim=shuttle_dim),
            pos_via=Via((0.12, 0.12), 0.1, metal=['mlam'], via=['cbam'], shape=(length - 15, 2), pitch=1),
            clearout_dim=(length, 0.3),
            gnd_box_h=11,
            separate_fin_drive=True, single_metal=True,
        ).nazca_cell(f'tunable_splitter_{length}_{dc_gap_w}')
        gc = grating.put(0, 0, -180)
        chip.waveguide_ic.strt(buffer_length).put(gc.pin['a0'])
        tdc_placed = tdc.put()

        chip.waveguide_ic.strt(buffer_length).put()
        detector.put()

        # chip.waveguide_ic.strt(buffer_length).put(dc2.pin['a1'])
        # dl_200 = delay_line_200.put()
        # chip.waveguide_ic.strt_p2p(dl_200.pin['b0'], dc1.pin['b1']).put()
        chip.waveguide_ic.strt(buffer_length).put(tdc_placed.pin['a1'])
        grating.put()
        chip.waveguide_ic.strt(buffer_length).put(tdc_placed.pin['b1'])
        detector.put()

    return tdc_cell


chip_y = 1000
chip_x = 2000

bond_pads = chip.bond_pad_array(n_pads=(20, 1), pitch=100,
                                pad_dim=(80, 80), stagger_x_frac=0.5)
with nd.Cell('test_set_1_pa_ps_ts300') as ts1:
    # TODO(Nate): move reference MZI to only test cell 2
    # TODO(Nate): 6 TDCs (200/300), 5 MZIs (3 pull apart, 2 pull in), 1 refernce MZI, 19 pads per row
    dy = 100
    x_offset = 200
    y_offset = 150
    dx_ps = (chip_x - x_offset) / 3
    dx_tdc = (chip_x - x_offset) / 3

    bond_pads.put(0, 80)

    for m, ps_length in enumerate([100, 125, 150]):
        pa_nems_MZI(ps_length).put(x_offset + m * dx_ps, y_offset)
    for m, tdc_length in enumerate([47.3, 141.9, 236.5]):
        tdc_test_cell(tdc_length, dc_gap_w=0.3).put(x_offset + m * dx_tdc, 150 + dy)
    # nems_MZI(ps_length, True).put(x_offset + m * dx_ps, y_offset + dy)
    bond_pads.put(0, y_offset + dy + 200)

with nd.Cell('test_set_2_pi_ps_ts200') as ts2:
    dy = 100
    x_offset = 200
    y_offset = 150
    dx_ps = (chip_x - x_offset) / 3
    dx_tdc = (chip_x - x_offset) / 3
    bond_pads.put(0, 80)

    for m, ps_length in enumerate([50, 100]):
        pi_nems_MZI(ps_length).put(x_offset + m * dx_ps, y_offset)
    pi_nems_MZI(ps_length + 10, True).put(x_offset + (m + 1) * dx_ps, y_offset)
    for m, tdc_length in enumerate([18.8, 56.5, 94.1]):
        tdc_test_cell(tdc_length).put(x_offset + m * dx_tdc, 150 + dy)
    bond_pads.put(0, y_offset + dy + 200)


with nd.Cell('gf_layout') as gf_layout:
    ts1.put()
    ts2.put(0, 500)
    alignment_mark.put(25, 200)
    alignment_mark.put(25, 700)
    alignment_mark.put(chip_x - 125, 700)
    alignment_mark.put(chip_x - 125, 200)

nd.export_gds(filename=f'../../../20201016_sjby_gf_test/gf-layout-{str(date.today())}-submission', topcells=[gf_layout])
