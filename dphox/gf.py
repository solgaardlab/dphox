from .component import *

# Solgaard lab GF PDK

waveguide_w = 0.5
phaseshift_l_pull_apart = 90
phaseshift_l_pull_in = 40
interaction_l_pull_apart = 100
interaction_l_pull_in = 50
end_l = 5
tether_phaseshift_l = 75
tether_interaction_l = 100
interport_w = 50
gap_w = 0.3
dc_radius = 15
test_bend_h = (interport_w - gap_w - waveguide_w) / 2


class GFDC(DC):
    def __init__(self, bend_dim=(dc_radius, test_bend_h), waveguide_w=waveguide_w,
                 gap_w=gap_w, interaction_l=37.8, use_radius=True,
                 coupler_boundary_taper_ls: Tuple[float, ...] = (0,),
                 coupler_boundary_taper: Optional[Tuple[Tuple[float, ...]]] = None
                 ):
        super(GFDC, self).__init__(bend_dim=bend_dim, waveguide_w=waveguide_w,
                                   gap_w=gap_w, interaction_l=interaction_l, use_radius=use_radius,
                                   coupler_boundary_taper_ls=coupler_boundary_taper_ls,
                                   coupler_boundary_taper=coupler_boundary_taper)


class GFNemsPS(LateralNemsPS):
    def __init__(self, waveguide_w=waveguide_w, nanofin_w=0.2, phaseshift_l=phaseshift_l_pull_apart,
                 gap_w=0.10, num_taper_evaluations=100, gnd_connector=(2, 0.2, 5),
                 gnd_pad_dim=None, taper_l=0, end_ls=(end_l,), gap_taper=None, wg_taper=None, boundary_taper=None,
                 fin_end_bend_dim=(2, 1), end_taper=((0, -0.08),), gnd_connector_idx=-1, rib_etch_grow=0.2):
        super(GFNemsPS, self).__init__(waveguide_w=waveguide_w, nanofin_w=nanofin_w, phaseshift_l=phaseshift_l,
                                       gap_w=gap_w, num_taper_evaluations=num_taper_evaluations,
                                       gnd_connector=gnd_connector, taper_l=taper_l, gnd_pad_dim=gnd_pad_dim,
                                       end_ls=end_ls, gap_taper=gap_taper, wg_taper=wg_taper,
                                       boundary_taper=boundary_taper, fin_end_bend_dim=fin_end_bend_dim,
                                       end_taper=end_taper, gnd_connector_idx=gnd_connector_idx,
                                       rib_etch_grow=rib_etch_grow)


class GFNemsTDC(LateralNemsTDC):
    def __init__(self, waveguide_w=waveguide_w, nanofin_w=0.2, interaction_l=interaction_l_pull_apart,
                 dc_gap_w=0.2, beam_gap_w=0.1, bend_dim=(10, 24.66), gnd_wg=(2, 2, 2, 0.75),
                 use_radius=True, dc_end_l=0, dc_taper_ls=None, dc_taper=None, beam_taper=None, fin_end_bend_dim=(2, 1),
                 rib_etch_grow=0.25):
        super(GFNemsTDC, self).__init__(waveguide_w=waveguide_w, nanofin_w=nanofin_w, interaction_l=interaction_l,
                                        dc_gap_w=dc_gap_w, beam_gap_w=beam_gap_w, bend_dim=bend_dim, gnd_wg=gnd_wg,
                                        use_radius=use_radius, dc_end_l=dc_end_l, dc_taper_ls=dc_taper_ls,
                                        dc_taper=dc_taper, beam_taper=beam_taper, fin_end_bend_dim=fin_end_bend_dim,
                                        rib_etch_grow=rib_etch_grow)


class GFNemsAnchor(NemsAnchor):
    def __init__(self, fin_dim=(100, 0.2), shuttle_dim=(50, 2), spring_dim=None, straight_connector=(0.25, 1),
                 tether_connector=(2, 1, 0.5, 1), pos_electrode_dim=(90, 4, 0.5), gnd_electrode_dim=(3, 4),
                 include_support_spring=True, shuttle_stripe_w=1):
        super(GFNemsAnchor, self).__init__(fin_dim=fin_dim, shuttle_dim=shuttle_dim, spring_dim=spring_dim,
                                           straight_connector=straight_connector, tether_connector=tether_connector,
                                           pos_electrode_dim=pos_electrode_dim, gnd_electrode_dim=gnd_electrode_dim,
                                           include_support_spring=include_support_spring, shuttle_stripe_w=shuttle_stripe_w)


class GFNemsFull(LateralNemsFull):
    def __init__(self, device, anchor, clearout_dim,
                 pos_box_w=8, gnd_box_h=8,
                 gnd_via=Via((0.12, 0.12), 0.1, metal='mlam', via='cbam', shape=(2, 2), pitch=1),
                 pos_via=Via((0.12, 0.12), 0.1, metal=['mlam'], via=['cbam'], shape=(20, 2), pitch=1),
                 trace_w=3, separate_fin_drive=False, single_metal=True):
        super(GFNemsFull, self).__init__(device=device, anchor=anchor, gnd_via=gnd_via,
                                         pos_via=pos_via, trace_w=trace_w, pos_box_w=pos_box_w,
                                         gnd_box_h=gnd_box_h, clearout_dim=clearout_dim, clearout_etch_stop_grow=0.5,
                                         dope_expand=0.5, dope_grow=0.0, ridge='seam', rib='ream', shuttle_dope='pdam',
                                         spring_dope='pdam', pad_dope='pppam', pos_metal='mlam',
                                         gnd_metal='mlam', clearout_layer='clearout', clearout_etch_stop_layer='snam', separate_fin_drive=separate_fin_drive)


class GFFakeGratingCoupler(FakeGratingCoupler):
    def __init__(self, waveguide_w=0.5, bounds=(40, 40), wg_layer='paam', box_layer='tzam'):
        super(GFFakeGratingCoupler, self).__init__(waveguide_w=waveguide_w, bounds=bounds, wg_layer=wg_layer, box_layer=box_layer)


class GFFakePDKDC(FakePDKDC):
    def __init__(self, waveguide_w=0.5, bounds=(51.63, 42.75), interport_w=42.75, wg_layer='paam', box_layer='tzam'):
        super(GFFakePDKDC, self).__init__(waveguide_w=waveguide_w, bounds=bounds, interport_w=interport_w, wg_layer=wg_layer, box_layer=box_layer)


class GFFakePhotodetector(FakePhotodetector):
    def __init__(self, waveguide_w=0.5, bounds=(60, 10), wg_layer='paam', box_layer='tzam'):
        super(GFFakePhotodetector, self).__init__(waveguide_w=waveguide_w, bounds=bounds, wg_layer=wg_layer, box_layer=box_layer)


fake_grating_coupler = GFFakeGratingCoupler()
fake_pdk_dc = GFFakePDKDC()
fake_photodetector = GFFakePhotodetector()

ps_pull_apart = GFNemsPS()
ps_pull_in = GFNemsPS(phaseshift_l=phaseshift_l_pull_in, gnd_pad_dim=(3, 4))
tdc_pull_apart = GFNemsTDC()
tdc_pull_in = GFNemsTDC(interaction_l=interaction_l_pull_in)

pull_apart_anchor = GFNemsAnchor()
pull_in_anchor = GFNemsAnchor(
    fin_dim=(50, 0.22), shuttle_dim=(40, 3),
    pos_electrode_dim=None, gnd_electrode_dim=None,
    spring_dim=None, include_support_spring=False, shuttle_stripe_w=0
)


pull_in_full_ps = GFNemsFull(device=ps_pull_in, anchor=pull_in_anchor,
                             clearout_dim=(phaseshift_l_pull_in, 0.3))
pull_in_full_tdc = GFNemsFull(device=tdc_pull_in, anchor=pull_in_anchor,
                              clearout_dim=(interaction_l_pull_in, 0.3),
                              gnd_box_h=10, pos_box_w=12)
pull_apart_full_ps = GFNemsFull(device=ps_pull_apart, anchor=pull_apart_anchor,
                                clearout_dim=(phaseshift_l_pull_apart, 0.3),
                                gnd_box_h=10, pos_box_w=18)
pull_apart_full_tdc = GFNemsFull(device=tdc_pull_apart, anchor=pull_apart_anchor,
                                 clearout_dim=(interaction_l_pull_apart, 0.3),
                                 gnd_box_h=10, pos_box_w=15, separate_fin_drive=True)
