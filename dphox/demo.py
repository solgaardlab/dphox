from .foundry import CommonLayer
from .active import ThermalPS, DC, MZI, Mesh
from .device import Via
from .passive import GratingPad, TapDC, Waveguide

ps = ThermalPS(Waveguide((1, 10)), ps_w=2, via=Via((0.4, 0.4), 0.1))
dc = DC(waveguide_w=1, interaction_l=2, bend_l=5, interport_distance=10, gap_w=0.5)
tap = TapDC(DC(waveguide_w=1, interaction_l=0.1, bend_l=2, interport_distance=4, gap_w=0.25),
            GratingPad(pad_extent=(3, 3), taper_l=2, final_w=1, end_l=0, out=True))

mzi = MZI(dc, top_internal=[ps], bottom_internal=[ps.copy], top_external=[ps.copy], bottom_external=[ps.copy])
tapped_mzi = MZI(dc, top_internal=[5, tap, 5, ps], bottom_internal=[5, tap.copy, 5, ps.copy],
                 top_external=[5, tap.copy, 5, ps.copy], bottom_external=[5, tap.copy, 5, ps.copy])

mesh = Mesh(mzi, 6)
tapped_mesh = Mesh(tapped_mzi, 6)
