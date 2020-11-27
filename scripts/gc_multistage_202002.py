import itertools
import numpy as np

import nazca as nd
from datetime import date
# from dphox.aim import *
from dphox.layout import NazcaLayout
from dphox.constants import get_sinx280_stack, GCMESHMSDLA_PDK
from dphox.component.active import VerticalPS, SurfaceMEMs, Microbridge

gap = 0.3
mechanical_thickness = 0.2
metal_thickness = 0.1
waveguide_w = 1.4
sinx280_STACK = get_sinx280_stack(gap=gap, mechanical_thickness=mechanical_thickness, metal_thickness=metal_thickness)
chip = NazcaLayout(
    passive_filepath='../../../20201027_sinx280/GC_MultiStage_202002_labeled.gds',
    waveguides_filepath='../../../20201027_sinx280/GC_MultiStage_202002.gds',
    active_filepath='../../../20201027_sinx280/GC_MultiStage_202002.gds',
    stack=sinx280_STACK,
    pdk_dict=GCMESHMSDLA_PDK,
    accuracy=0.001,
    waveguide_w=waveguide_w
)


def ps_280x(name, waveguide_w, gap, ps_w, total_length, width, interaction_l,
            mb_width, mb_length, anchor_dim, mb_underetch_dim):
    ps_pattern = VerticalPS(waveguide_w=waveguide_w, gap=gap, ps_w=ps_w, total_length=total_length, width=width, interaction_l=interaction_l)
    microbridge = Microbridge(width=mb_width, length=mb_length, underetch_dim=mb_underetch_dim, anchor_dim=anchor_dim)
    full_ps = SurfaceMEMs(device=ps_pattern, actuator=microbridge, symmetric=True, bottom_cladding_layer='box', wg_layer='nit1', cladding_layer='tox', sacrificial_layer='asi1', mechanical_layer='nit2', top_metal='metal1')
    return(full_ps.nazca_cell(name))


# dictionary defining the cells

# This rebuilds the sinx chiplet using the previous layout
with nd.Cell('sinx_wg_chiplet') as wg_chiplet:
    chip.pdk_cells['mmi_exp'].put(0, 0)
    chip.pdk_cells['gc_exp'].put(0, 3600)
    chip.pdk_cells['mmi_mzi'].put(0, 2 * 3600)
    chip.pdk_cells['dc_exp'].put(0, 3 * 3600)
    nd.Pin('a0').put(15200 / 2, 14400 / 2, -180)


def photonic_mems_chip(wg_chiplet, ps_w, length):

    with nd.Cell(f'sinx_chip_psw{np.around(ps_w,3)*1000:.0f}_length{np.around(length,3):.0f}') as chiplet:
        wg_chiplet.put()
        # test area to see if I can position the ps correctly
        # 1st waveguide is 550um from the bottom
        test_ps = ps_280x(name='test_ps', waveguide_w=waveguide_w, gap=gap, ps_w=ps_w, total_length=length, width=9 * waveguide_w, interaction_l=length,
                          mb_width=15, mb_length=length, anchor_dim=(15, 15), mb_underetch_dim=(4, 4, 2))
        # on MMIs
        for ind in range(13):
            test_ps.put(550, 650 + ind * 200 - 14400 / 2)
        # on MZI arms
        for ind in range(7):
            test_ps.put(0, 2 * 3600 + 650 + ind * 400 - 14400 / 2)

        # on DCs
        for ind in range(13):
            test_ps.put(550, 3 * 3600 + 650 + ind * 200 - 14400 / 2)

        # Add Rows of metal pads to manually route to
        for ind_x in range(-19, 19):
            if abs(ind_x) <= 2:
                continue
            pad_size = 150
            # offset_neg_x=-15200/2+1400
            chip.ml_ic.strt(length=pad_size, width=pad_size).put(2 * pad_size * (ind_x + 0.5), 14400 / 2 - 425)
            # chip.ml_ic.strt(length=pad_size, width=pad_size).put(2 * pad_size * (ind_x + 0.5), -14400 / 2 + 425)
            # chip.ml_ic.strt_p2p()

            # traces for MMIs

            if ind_x in range(-19, (-19 + 13)):
                pad_pt = (pad_size / 2 + 2 * pad_size * (ind_x + 0.5), 14400 / 2 - 425)
                mid_pt = (pad_size / 2 + 2 * pad_size * (ind_x + 0.5), 650 + (ind_x + 19) * 200 - 14400 / 2)
                ps_pt = (550 - 5, 650 + (ind_x + 19) * 200 - 14400 / 2)
                # chip.ml_ic.strt_p2p(pad_pt, mid_pt).put()
                # chip.ml_ic.strt_p2p(mid_pt, ps_pt).put()
                chip.ml_ic.mamba([pad_pt, mid_pt, ps_pt], width=pad_size / 2).put('org', 0)
                # chip.ml_ic.strt(length=1000, width=pad_size / 2).put(2 * pad_size * (ind_x + 0.5), 14400 / 2 - 425, -90)

            # traces for MZIs part 1

            if ind_x in range((-19 + 13), (-19 + 17)):
                pad_pt = (pad_size / 2 + 2 * pad_size * (ind_x + 0.5), 14400 / 2 - 425)
                mid_pt = (pad_size / 2 + 2 * pad_size * (ind_x + 0.5), 2 * 3600 + 650 + (ind_x + 19 - 13) * 400 - 14400 / 2)
                ps_pt = (-5, 2 * 3600 + 650 + (ind_x + 19 - 13) * 400 - 14400 / 2)
                # chip.ml_ic.strt_p2p(pad_pt, mid_pt).put()
                # chip.ml_ic.strt_p2p(mid_pt, ps_pt).put()
                chip.ml_ic.mamba([pad_pt, mid_pt, ps_pt], width=pad_size / 2).put('org', 0)
                # chip.ml_ic.strt(length=1000, width=pad_size / 2).put(2 * pad_size * (ind_x + 0.5), 14400 / 2 - 425, -90)

            # traces for DCs

            if ind_x in range((19 - 16), 19 - 3):
                pad_pt = (pad_size / 2 + 2 * pad_size * (ind_x + 0.5), 14400 / 2 - 425)
                mid_pt = (pad_size / 2 + 2 * pad_size * (ind_x + 0.5), 3 * 3600 + 650 + (18 - 3 - ind_x) * 200 - 14400 / 2)
                ps_pt = (550 + 5 + length, 3 * 3600 + 650 + (18 - 3 - ind_x) * 200 - 14400 / 2)
                # chip.ml_ic.strt_p2p(pad_pt, mid_pt).put()
                # chip.ml_ic.strt_p2p(mid_pt, ps_pt).put()
                chip.ml_ic.mamba([pad_pt, mid_pt, ps_pt], width=pad_size / 2).put('org', 0)
                # chip.ml_ic.strt(length=1000, width=pad_size / 2).put(2 * pad_size * (ind_x + 0.5), 14400 / 2 - 425, -90)

            # traces for MZIs part 1
            if ind_x in range((19 - 3), 19):
                pad_pt = (pad_size / 2 + 2 * pad_size * (ind_x + 0.5), 14400 / 2 - 425)
                mid_pt = (pad_size / 2 + 2 * pad_size * (ind_x + 0.5), 2 * 3600 + 650 + (18 - ind_x + 4) * 400 - 14400 / 2)
                ps_pt = (5 + length, 2 * 3600 + 650 + (18 - ind_x + 4) * 400 - 14400 / 2)
                # chip.ml_ic.strt_p2p(pad_pt, mid_pt).put()
                # chip.ml_ic.strt_p2p(mid_pt, ps_pt).put()
                chip.ml_ic.mamba([pad_pt, mid_pt, ps_pt], width=pad_size / 2).put('org', 0)

    return chiplet


test_array = {
    # 6 most interior array where amsl litho is best
    (0, 1): (0.2, 125),
    (-1, 0): (0.3, 125),
    (0, 0): (0.4, 125),
    (-1, -1): (0.2, 100),
    (0, -1): (0.3, 100),
    (-1, -2): (0.4, 100),
    # top/bot = 125/100 length structures
    # 4 additional centr slots can go to in-between values
    (-2, 0): (0.25, 125),
    (1, 0): (0.35, 125),
    (-2, -1): (0.25, 100),
    (1, -1): (0.35, 100),
    # pairs on the edges can be extreme/useless cases with refs
    (-3, 0): (0.15, 200),
    # (-3, -1): (0.35, 500),
    # (2, 0): (0.25, 500),
    (2, -1): (0.45, 100),
    # (-1, 2): (0.25, 500),
    (0, 2): (0.15, 200),
    (-1, -3): (0.45, 100),
    # (0, -3): (0.35, 500),


}


with nd.Cell('sinx280_ps_v2') as layout:
    # layout ASML matrix  that was used
    matrix_shift_x, matrix_shift_y = 7500, 7500
    cell_size_x, cell_size_y = 15000, 15000
    for ind_x in range(-3, 3):
        for ind_y in range(-3, 3):
            if abs(ind_x + 0.5) + abs(ind_y + 0.5) > 3.5:
                continue
            # use dictionaty of chip values here
            try:
                ps_w, length = test_array[(ind_x, ind_y)]
                photonic_mems_chip(wg_chiplet, ps_w=ps_w, length=length).put(cell_size_x * ind_x + matrix_shift_x, cell_size_y * ind_y + matrix_shift_y)
            except:
                wg_chiplet.put(cell_size_x * ind_x + matrix_shift_x, cell_size_y * ind_y + matrix_shift_y)

    # alignment mark guesses
    # 850 from the left/right
    # 450 from top/bot
    x_shift = -1 * (15200 - 15000) / 2
    y_shift = -1 * (14400 - 15000) / 2
    a_NW = (x_shift + - 2 * 15000 + 850, y_shift + 1 * 15000 + + 14400 - 450)
    a_NE = (x_shift + 1 * 15000 + 15200 - 850, y_shift + 1 * 15000 + 14400 - 450)
    a_SE = (x_shift + 1 * 15000 + 15200 - 850, y_shift + -2 * 15000 + 450)
    a_SW = (x_shift + -2 * 15000 + 850, y_shift + -2 * 15000 + 450)
    # chip.ml_ic.strt(length=100, width=100).put(a_NW)
    # chip.ml_ic.strt(length=100, width=100).put(a_NE)
    # chip.ml_ic.strt(length=100, width=100).put(a_SE)
    # chip.ml_ic.strt(length=100, width=100).put(a_SW)

    # wg_chiplet.put()

nd.export_gds(filename=f'../../../20201027_sinx280/sinx-ps-layout-{str(date.today())}-{np.around(mechanical_thickness,3)*1000:.0f}nm-MEMs-{np.around(gap,3)*1000:.0f}nm-gap', topcells=[layout])
print(a_NW, a_NE, a_SE, a_SW)
