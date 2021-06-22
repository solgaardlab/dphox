from .component.active import ThermalPS, DC, MZI, Mesh, Via
from .component.passive import GratingPad, TapDC

ps = ThermalPS(waveguide_w=1, ps_w=2, length=10, ridge='ream', ps_layer='m1am',
               via=Via((0.4, 0.4), 0.1, metal=['m1am', 'm2am'], via=['v1am']))
dc = DC(waveguide_w=1, interaction_l=2, bend_dim=(5, 5), gap_w=0.5)
tap = TapDC(DC(waveguide_w=1, interaction_l=0.1, bend_dim=(2, 2), gap_w=0.25),
            GratingPad(pad_dim=(3, 3), taper_l=2, final_w=1, end_l=0, out=True))

mzi = MZI(dc,
          top_internal=[ps],
          bottom_internal=[ps.copy],
          top_external=[ps.copy],
          bottom_external=[ps.copy], ridge='ream')

tapped_mzi = MZI(dc,
                 top_internal=[5, tap, 5, ps],
                 bottom_internal=[5, tap.copy, 5, ps.copy],
                 top_external=[5, tap.copy, 5, ps.copy],
                 bottom_external=[5, tap.copy, 5, ps.copy], ridge='ream')

mesh = Mesh(mzi, 6)
tapped_mesh = Mesh(tapped_mzi, 6)
