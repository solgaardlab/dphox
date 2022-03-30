from .foundry import AIR, CommonLayer, SILICON
from .parametric import cubic_taper, straight
from .pattern import Box
from .prefab.active import Clearout, GndAnchorWaveguide, LateralNemsPS, LocalMesh, MEMSFlexure, \
    MZI, PullInNemsActuator, PullOutNemsActuator, ThermalPS, Via
from .prefab.passive import DC, FocusingGrating, RibDevice

ps = ThermalPS(straight(5).path(0.5), ps_w=2, via=Via((0.4, 0.4), 0.1))
dc = DC(waveguide_w=0.5, interaction_l=2, radius=2.5, interport_distance=5, gap_w=0.25)
mzi = MZI(dc, top_internal=[ps.copy], bottom_internal=[ps.copy], top_external=[ps.copy], bottom_external=[ps.copy])
mesh = LocalMesh(mzi, 6)
grating = FocusingGrating(
    n_env=AIR.n,
    n_core=SILICON.n,
    min_period=10,
    num_periods=10,
    wavelength=1.55,
    fiber_angle=82,
    duty_cycle=0.5,
    waveguide_w=0.5
)


def lateral_nems_ps(ps_l=100, anchor_length=3.1, clearout_height=12, via_extent=(0.5, 0.5),
                    ps_taper_change=-0.2, flexure_box_w=31, nominal_gap=0.201, waveguide_w=0.5,
                    nanofin_w=0.2, taper_l=10, anchor_taper_l=1.4, pull_in=False, trace_w=1, smooth: float = 0):
    ps_w = waveguide_w + 2 * nominal_gap + 2 * nanofin_w
    gap_w = waveguide_w + 2 * nominal_gap

    ps_box = straight(ps_l).path(ps_w)
    ps_gap = cubic_taper(gap_w, ps_taper_change, ps_l, taper_l)
    ps_waveguide = cubic_taper(waveguide_w, ps_taper_change, ps_l, taper_l)
    psw = ps_box - ps_gap + ps_waveguide
    psw.port = ps_waveguide.port

    via_low = Via(via_extent=via_extent, boundary_grow=0.25,
                  metal=[CommonLayer.METAL_1], via=[CommonLayer.VIA_SI_1])
    via_high = Via(via_extent=via_extent, boundary_grow=0.25,
                   metal=[CommonLayer.METAL_1, CommonLayer.METAL_2],
                   via=[CommonLayer.VIA_SI_1, CommonLayer.VIA_1_2])

    gaw_rib = cubic_taper(ps_w + 0.1, 1, 2 * anchor_length, anchor_taper_l, symmetric=False)
    gaw_gap = cubic_taper(gap_w, 1, 2 * anchor_length, anchor_taper_l, symmetric=False)
    gaw_waveguide = straight(2 * anchor_length).path(waveguide_w)
    gaw_slab = cubic_taper(0.5, 1, anchor_length, anchor_taper_l + 0.1)

    gaw = GndAnchorWaveguide(
        rib_waveguide=RibDevice(
            ridge_waveguide=(gaw_rib - gaw_gap + gaw_waveguide).set_port(gaw_waveguide.port),
            slab_waveguide=gaw_slab,
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

    ps = LateralNemsPS(
        waveguide_w=waveguide_w,
        phase_shifter_waveguide=psw,
        gnd_anchor_waveguide=gaw,
        actuator=pina if pull_in else pona,
        clearout=clr,
        trace_w=trace_w,
        clearout_pos_sep=10,
        clearout_gnd_sep=2
    )

    return ps.smooth_layer(smooth, CommonLayer.RIDGE_SI) if smooth > 0 else ps
