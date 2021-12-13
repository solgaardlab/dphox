from .active import Clearout, GndAnchorWaveguide, LateralNemsPS, MEMSFlexure, LocalMesh, MZI, PullInNemsActuator, \
    PullOutNemsActuator, ThermalPS, Via
from .foundry import CommonLayer
from .passive import DC, WaveguideDevice
from .path import Path, taper_waveguide, straight
from .pattern import Box
from .utils import cubic_taper

ps = ThermalPS(straight((10, 1)), ps_w=4, via=Via((0.4, 0.4), 0.1))
dc = DC(waveguide_w=1, interaction_l=2, bend_l=5, interport_distance=10, gap_w=0.5)
mzi = MZI(dc, top_internal=[ps.copy], bottom_internal=[ps.copy], top_external=[ps.copy], bottom_external=[ps.copy])
# mesh = LocalMesh(mzi, 6)


def lateral_nems_ps(ps_l=100, anchor_length=3, clearout_height=12, via_extent=(0.5, 0.5),
                    ps_taper_change=-0.2, flexure_box_w=31, nominal_gap=0.201, waveguide_w=0.5,
                    nanofin_w=0.2, taper_l=10, anchor_taper_l=1.4, pull_in=False, trace_w=1):

    ps_w = waveguide_w + 2 * nominal_gap + 2 * nanofin_w
    gap_w = waveguide_w + 2 * nominal_gap

    psw = Path(
        segments=[straight((ps_l, ps_w))],
        subtract=Path(
            segments=[taper_waveguide(cubic_taper(gap_w, ps_taper_change), ps_l, taper_l)],
            subtract=taper_waveguide(cubic_taper(waveguide_w, ps_taper_change), ps_l, taper_l)
        )
    )

    via_low = Via(via_extent=via_extent,
                  boundary_grow=0.25,
                  metal=[CommonLayer.METAL_1],
                  via=[CommonLayer.VIA_SI_1]
                  )
    via_high = Via(via_extent=via_extent,
                   boundary_grow=0.25,
                   metal=[CommonLayer.METAL_1, CommonLayer.METAL_2],
                   via=[CommonLayer.VIA_SI_1, CommonLayer.VIA_1_2]
                   )

    gaw = GndAnchorWaveguide(
        rib_waveguide=WaveguideDevice(
            ridge_waveguide=Path(
                segments=[taper_waveguide(cubic_taper(ps_w + 0.1, 1), 2 * anchor_length, anchor_taper_l, symmetric=False)],
                subtract=Path(
                    segments=[taper_waveguide(cubic_taper(gap_w, 1), 2 * anchor_length, anchor_taper_l, symmetric=False)],
                    subtract=straight((2 * anchor_length, 0.5))
                )
            ),
            slab_waveguide=taper_waveguide(cubic_taper(0.5, 1), anchor_length, anchor_taper_l + 0.1),
        ),
        gnd_pad=Box((1, 3)),
        gnd_connector=Box((0.5, 2)),
        via=via_high,
        offset_into_rib=0.1
    )

    pina = PullInNemsActuator(
        pos_pad=Box((ps_l, 2)),
        connector=Box((0.3, 0.3)),
        via=via_low
    )

    pona = PullOutNemsActuator(
        pos_pad=Box((ps_l, 2)),
        connector=Box((0.2, 0.5)),
        pad_sep=0.2,
        flexure=MEMSFlexure((flexure_box_w, 4.5),
                            stripe_w=0.5,
                            pitch=0.5,
                            spring_extent=(ps_l + anchor_length * 2, 0.2)),
        via=via_low
    )

    clr = Clearout(
        clearout_etch=Box((ps_l, clearout_height)),
        clearout_etch_stop_grow=0.5
    )

    return LateralNemsPS(
        phase_shifter_waveguide=psw,
        gnd_anchor_waveguide=gaw,
        actuator=pina if pull_in else pona,
        clearout=clr,
        trace_w=trace_w,
        clearout_pos_sep=10,
        clearout_gnd_sep=2
    ) #.smooth_layer(0.19, CommonLayer.RIDGE_SI)
