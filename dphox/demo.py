from .active import Clearout, GndAnchorWaveguide, LateralNemsPS, MEMSFlexure, Mesh, MZI, PullInNemsActuator, \
    PullOutNemsActuator, ThermalPS, Via
from .foundry import CommonLayer
from .passive import DC, TaperSpec, Waveguide, WaveguideDevice
from .pattern import Box

ps = ThermalPS(Waveguide((1, 10)), ps_w=2, via=Via((0.4, 0.4), 0.1))
dc = DC(waveguide_w=1, interaction_l=2, bend_l=5, interport_distance=10, gap_w=0.5)
mzi = MZI(dc, top_internal=[ps], bottom_internal=[ps.copy], top_external=[ps.copy], bottom_external=[ps.copy])


# mesh = Mesh(mzi, 6)

def lateral_nems_ps(ps_l=100, anchor_length=3, clearout_height=12, via_extent=(0.5, 0.5),
                    flexure_box_w=30, nominal_gap=0.201, waveguide_w=0.5,
                    nanofin_w=0.2, taper_l=10, pull_in=False, trace_w=1):
    psw = Waveguide(
        extent=(waveguide_w + 2 * nominal_gap + 2 * nanofin_w, ps_l),
        subtract_waveguide=Waveguide(
            extent=(waveguide_w + 2 * nominal_gap, ps_l),  # gap = 0.7 - 0.5
            taper=TaperSpec.cubic(taper_l, -0.2),
            subtract_waveguide=Waveguide(
                extent=(waveguide_w, ps_l),
                taper=TaperSpec.cubic(taper_l, -0.2)
            )
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
            ridge_waveguide=Waveguide((waveguide_w + 2 * nominal_gap + 2 * nanofin_w + 0.1, anchor_length),
                                      taper=TaperSpec.cubic(1.4, 1),
                                      symmetric=False,
                                      subtract_waveguide=Waveguide(
                                          extent=(waveguide_w + 2 * nominal_gap, anchor_length),
                                          taper=TaperSpec.cubic(1.4, 1),
                                          symmetric=False,
                                          subtract_waveguide=Waveguide(
                                              (0.5, anchor_length),
                                          ))),
            slab_waveguide=Waveguide((0.5, anchor_length),
                                     TaperSpec.cubic(1.5, 1))),
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
    )
