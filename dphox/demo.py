from .foundry import AIR, CommonLayer, SILICON
from .parametric import cubic_taper, straight
from .pattern import Box, Ellipse
from .device import Device
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


def lateral_nems_ps(ps_l=100, anchor_length=3.1, anchor_w=5, clearout_height=12, via_extent=(4, 4),
                    ps_taper_w=0.3, flexure_box_extent=(31, 4.5), nominal_gap=0.201, waveguide_w=0.5,
                    nanofin_w=0.2, taper_l=10, anchor_taper_l=1.4, pull_in=False, trace_w=3, smooth: float = 0,
                    clearout_pos_sep: float = 10, clearout_gnd_sep: float = 2, pos_w: float = 10, gnd_pad_dim: float = (5, 5),
                    extra_clearout_dim=(3, 5), clearout_etch_stop_grow=2, gnd_connector_dim=(1, 2), final_anchor_w=0.5,
                    etch_stop_gap = 0.18, via_boundary_w: float = 1.5, via_high_l: float = 20):
    ps_w = waveguide_w + 2 * nominal_gap + 2 * nanofin_w
    gap_w = waveguide_w + 2 * nominal_gap

    # center phase shifting region

    ps_box = straight(ps_l).path(ps_w)
    ps_gap = cubic_taper(gap_w, ps_taper_w - waveguide_w, ps_l, taper_l)
    ps_waveguide = cubic_taper(waveguide_w, ps_taper_w - waveguide_w, ps_l, taper_l)
    psw = ps_box - ps_gap + ps_waveguide
    psw.port = ps_waveguide.port

    # vias to connect different metal layers to the MEMS actuators

    via_low = Via(via_extent=via_extent, boundary_grow=via_boundary_w,
                  metal=[CommonLayer.METAL_1], via=[CommonLayer.VIA_SI_1])
    

    via_high = Via(via_extent=via_extent, boundary_grow=via_boundary_w,
                   metal=[CommonLayer.METAL_2],
                   via=[CommonLayer.VIA_1_2])

    via_high.translate(via_high_l)

    via_high.add(Box((via_high_l, via_extent[1] + 2 * via_boundary_w)).align(via_high.center).halign(via_high, left=False), CommonLayer.METAL_1)
    via_high.add(Box((via_high_l, via_extent[1] + 2 * via_boundary_w)).align(via_high.center).halign(via_high, left=False), CommonLayer.METAL_2)
    
    via_full = Device("via", [via_low.copy, via_high])

    # ground anchor waveguide

    gaw_rib = cubic_taper(ps_w + 0.1, anchor_w, 2 * anchor_length, anchor_taper_l, symmetric=False)
    gaw_gap = cubic_taper(gap_w, anchor_w, 2 * anchor_length, anchor_taper_l, symmetric=False)
    gaw_waveguide = straight(2 * anchor_length).path(waveguide_w)
    gaw_slab = cubic_taper(final_anchor_w, anchor_w - final_anchor_w + 0.5, anchor_length * 2 - anchor_taper_l, anchor_taper_l + 0.1, symmetric=True)

    gaw = GndAnchorWaveguide(
        rib_waveguide=RibDevice(
            ridge_waveguide=(gaw_rib - gaw_gap + gaw_waveguide).set_port(gaw_waveguide.port),
            slab_waveguide=gaw_slab,
            slab=CommonLayer.RIB_SI_2
        ),
        gnd_pad=Box(gnd_pad_dim),
        gnd_connector=Box(gnd_connector_dim),
        via=via_low,
        offset_into_rib=0.1,
        
    )

    pina = PullInNemsActuator(
        pos_pad=Box((ps_l, pos_w)),
        connector=Box((0.3, 0.3)),
        via=via_full
    )

    pona = PullOutNemsActuator(
        pos_pad=Box((ps_l, pos_w)),
        connector=Box((0.2, 0.8)),
        pad_sep=0.2,
        flexure=MEMSFlexure(flexure_box_extent,
                            stripe_w=0.5,
                            pitch=0.5,
                            spring_extent=(ps_l + anchor_length * 2, 0.2)),
        via=via_full,
        stop_extender=Box(((ps_l + extra_clearout_dim[0] - flexure_box_extent[0])/ 2, flexure_box_extent[1] / 2)),
        stop_bump=Ellipse((2.5 * etch_stop_gap, etch_stop_gap)) + Box((5 * etch_stop_gap, etch_stop_gap)).valign(0),
        stop_gap=(etch_stop_gap, etch_stop_gap),
        dope_expand_tuple=(0.3, 0.3)
    )

    clr = Clearout(
        clearout_etch=Box((ps_l + extra_clearout_dim[0], clearout_height + extra_clearout_dim[1])),
        clearout_etch_stop_grow=clearout_etch_stop_grow
    )

    ps = LateralNemsPS(
        waveguide_w=waveguide_w,
        phase_shifter_waveguide=psw,
        gnd_anchor_waveguide=gaw,
        actuator=pina if pull_in else pona,
        clearout=clr,
        trace_w=trace_w,
        clearout_pos_sep=clearout_pos_sep,
        clearout_gnd_sep=clearout_gnd_sep
    )

    # ps.add(ps.bbox_pattern.buffer(3), layer=CommonLayer.PAD_OPEN)

    return ps.smooth_layer(smooth, CommonLayer.RIDGE_SI) if smooth > 0 else ps
