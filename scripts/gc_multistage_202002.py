import itertools

import nazca as nd
from datetime import date
# from dphox.aim import *
from dphox.layout import NazcaLayout
from dphox.constants import sinx280_STACK, GCMESHMSDLA_PDK


chip = NazcaLayout(
    passive_filepath='../../../20201027_sinx280/GC_MultiStage_202002_labeled.gds',
    waveguides_filepath='../../../20201027_sinx280/GC_MultiStage_202002.gds',
    active_filepath='../../../20201027_sinx280/GC_MultiStage_202002.gds',
    stack=sinx280_STACK,
    pdk_dict=GCMESHMSDLA_PDK,
    accuracy=0.001,
    waveguide_w=1.4
)


chip.pdk_cells['dc_exp'].put()

# This rebuilds the sinx chiplet using the previous layout
with nd.Cell('sinx_ps_layout') as layout:
    chip.pdk_cells['mmi_exp'].put(0, 0)
    chip.pdk_cells['gc_exp'].put(0, 3600)
    chip.pdk_cells['mmi_mzi'].put(0, 2 * 3600)
    chip.pdk_cells['dc_exp'].put(0, 3 * 3600)

nd.export_gds(filename=f'../../../20201027_sinx280/sinx-ps-layout-{str(date.today())}-submission', topcells=[layout])
