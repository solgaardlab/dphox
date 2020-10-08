#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:19:16 2020
@author: Sunil Pai, Nate Abebe, Rebecca Hwang, Yu Miao
"""

from .component import *
from ..constants import AIM_PDK_WAVEGUIDE_PATH, AIM_PDK_PASSIVE_PATH, AIM_PDK_ACTIVE_PATH, AIM_STACK, AIM_PDK
import numpy as np
import nazca as nd
from ..typing import Optional


class AIMNazca:
    def __init__(self, passive_filepath: str = AIM_PDK_PASSIVE_PATH, waveguides_filepath: str = AIM_PDK_WAVEGUIDE_PATH,
                 active_filepath: str = AIM_PDK_ACTIVE_PATH, stack: Dict = AIM_STACK, pdk_dict: Dict = AIM_PDK,
                 accuracy: float = 0.001, waveguide_w: float = 0.48):
        self.passive = nd.load_gds(passive_filepath, asdict=True, topcellsonly=False)
        self.waveguides = nd.load_gds(waveguides_filepath, asdict=True, topcellsonly=False)
        self.active = nd.load_gds(active_filepath, asdict=True, topcellsonly=False)
        self.lib = {**self.passive, **self.waveguides, **self.active}
        self.stack = stack
        self.pdk_dict = pdk_dict
        self.pdk_cells = {}
        for layer_name in stack['layers']:
            nd.add_layer(name=layer_name, layer=stack['layers'][layer_name], overwrite=True, accuracy=accuracy)

        for pdk_elem_name, pdk_elem_pins in self.pdk_dict.items():
            with nd.Cell(f'nazca_{pdk_elem_name}') as cell:
                self.lib[pdk_elem_name].put()
                for pin_name, pin_assignment in pdk_elem_pins.items():
                    nd.Pin(pin_name).put(pin_assignment)
            self.pdk_cells[pdk_elem_name] = cell

        for xs_name in stack['cross_sections']:
            for layer_dict in stack['cross_sections'][xs_name]:
                nd.add_layer2xsection(
                    xsection=xs_name,
                    accuracy=accuracy,
                    overwrite=True,
                    **layer_dict)

        self.waveguide_ic = nd.interconnects.Interconnect(width=waveguide_w, xs='waveguide_xs')
        self.pad_ic = nd.interconnects.Interconnect(width=60, xs='pad_xs')
        self.dicing_ic = nd.interconnects.Interconnect(width=100, xs='dicing_xs')
        self.m1_ic = nd.interconnects.Interconnect(width=4, xs='m1_xs')
        self.m2_ic = nd.interconnects.Interconnect(width=4, xs='m2_xs')
        self.ml_ic = nd.interconnects.Interconnect(width=100, xs='ml_xs')
        self.v1_via = Via((0.4, 0.4), 0.1, metal=['m1am', 'm2am'], via='v1am').nazca_cell('v1_via')
        self.va_via = Via((3.6, 3.6), 1.5, metal=['mlam', 'm2am'], via='vaam').nazca_cell('va_via')
        self.v1_via_4 = Via((0.4, 0.4), 0.15, shape=(1, 4), pitch=1,
                            metal=['m1am', 'm2am'], via='v1am').nazca_cell('v1_via_4')
        self.v1_via_8 = Via((0.4, 0.4), 0.15, shape=(1, 8), pitch=1,
                            metal=['m1am', 'm2am'], via='v1am').nazca_cell('v1_via_8')

    def nems_tdc(self, waveguide_w: float = 0.48, nanofin_w: float = 0.22,
                 interaction_l: float = 100, dc_gap_w: float = 0.2, beam_gap_w: float = 0.1,
                 bend_dim: Dim2 = (10, 24.66), pad_dim: Dim4 = (2, 2, 2, 0.75), anchor: nd.Cell = None,
                 use_radius: bool = True, clearout_box_dim: Dim2 = (100, 2.5), dc_taper_ls: Tuple[float, ...] = None,
                 dc_taper=None, beam_taper=None, clearout_etch_stop_grow: float = 0.5,
                 dope_grow: float = 0.25, dope_expand: float = 0.1, fin_adiabatic_bend_dim: Optional[Dim2] = (2, 1),
                 dc_end_l: float = 0, metal_extension: float = 3,
                 diff_ps: Optional[nd.Cell] = None, name: str = 'nems_tdc') -> nd.Cell:
        c = LateralNemsTDC(waveguide_w=waveguide_w, nanofin_w=nanofin_w,
                           interaction_l=interaction_l, dc_gap_w=dc_gap_w, beam_gap_w=beam_gap_w,
                           bend_dim=bend_dim, gnd_wg=pad_dim, use_radius=use_radius, dc_taper_ls=dc_taper_ls,
                           dc_taper=dc_taper, beam_taper=beam_taper, fin_end_bend_dim=fin_adiabatic_bend_dim,
                           dc_end_l=dc_end_l)
        pad_to_layer = sum([pad.metal_contact(('cbam', 'm1am', 'v1am', 'm2am')) for pad in c.pads], [])
        clearout = c.clearout_box(clearout_layer='clearout', clearout_etch_stop_layer='snam',
                                  clearout_etch_stop_grow=clearout_etch_stop_grow, dim=clearout_box_dim)
        ridge_etch = [(brim, 'ream') for brim in c.rib_brim]
        dopes = [p.expand(dope_expand).dope('pppam', dope_grow) for p in c.pads]
        device = Multilayer([(c, 'seam')] + pad_to_layer + clearout + ridge_etch + dopes)
        if anchor is None:
            cell = device.nazca_cell(name)
        else:
            with nd.Cell(name) as cell:
                tdc = device.nazca_cell('tdc').put()
                top_anchor = anchor.put(tdc.pin['fin0'])
                bottom_anchor = anchor.put(tdc.pin['fin1'])
                self.metal_box(top_anchor, bottom_anchor, interaction_l, m2_extension=metal_extension)
                nd.Pin('a0').put(tdc.pin['a0'])
                nd.Pin('b0').put(tdc.pin['b0'])
                nd.Pin('a1').put(tdc.pin['a1'])
                nd.Pin('b1').put(tdc.pin['b1'])
                tdc.raise_pins(['gnd_l', 'gnd_r', 'pos_l', 'pos_r'])
        return self.tdc_node(diff_ps, cell) if diff_ps is not None else cell

    def nems_ps(self, anchor: nd.Cell, waveguide_w: float = 0.48, nanofin_w: float = 0.22, phaseshift_l: float = 90,
                gap_w: float = 0.10, pad_dim: Optional[Dim3] = None, clearout_box_dim: Dim2 = (90, 13.8),
                clearout_etch_stop_grow: float = 0.5, num_taper_evaluations: int = 100,
                tap_sep: Optional[Tuple[nd.Cell, float]] = None,
                gnd_connector: Optional[Dim3] = (2, 0.2, 3), taper_l: float = 0, end_ls: Tuple[float, ...] = (5,),
                gap_taper: Optional[Tuple[float, ...]] = None, wg_taper: Optional[Tuple[float, ...]] = None,
                boundary_taper: Optional[Tuple[float, ...]] = None, fin_adiabatic_bend_dim: Optional[Dim2] = (2, 1),
                end_taper: Optional[Tuple[Tuple[float, ...], ...]] = ((0, -0.08),), gnd_connector_idx: int = -1,
                metal_extension: float = 10,
                name: str = 'nems_ps') -> nd.Cell:
        c = LateralNemsPS(waveguide_w=waveguide_w, nanofin_w=nanofin_w, phaseshift_l=phaseshift_l, gap_w=gap_w,
                          num_taper_evaluations=num_taper_evaluations,
                          fin_end_bend_dim=fin_adiabatic_bend_dim, gnd_connector_idx=gnd_connector_idx,
                          gap_taper=gap_taper, wg_taper=wg_taper, end_ls=end_ls, end_taper=end_taper,
                          taper_l=taper_l, boundary_taper=boundary_taper, gnd_connector=gnd_connector)
        # pad_to_layer = sum([pad.metal_contact(('cbam', 'm1am', 'v1am', 'm2am')) for pad in c.pads], [])
        clearout = c.clearout_box(clearout_layer='clearout', clearout_etch_stop_layer='snam',
                                  clearout_etch_stop_grow=clearout_etch_stop_grow, dim=clearout_box_dim)
        ridge_etch = [(brim, 'ream') for brim in c.rib_brim]
        device = Multilayer([(c, 'seam')] + clearout + ridge_etch)

        with nd.Cell(name) as cell:
            ps = device.nazca_cell('ps').put()
            top_anchor = anchor.put(ps.pin['fin0'])
            bottom_anchor = anchor.put(ps.pin['fin1'])
            self.metal_box(top_anchor, bottom_anchor, phaseshift_l, m2_extension=metal_extension)
            nd.Pin('a0').put(ps.pin['a0'])
            if tap_sep is not None:
                tap, sep = tap_sep
                self.waveguide_ic.strt(sep).put(ps.pin['b0'])
                t = tap.put()
                nd.Pin('b0').put(t.pin['b0'])
            else:
                nd.Pin('b0').put(ps.pin['b0'])
        return cell

    def metal_box(self, top_anchor: nd.Instance, bottom_anchor: nd.Instance, length: float, m2_extension: float = 4):
        has_c1 = 'c1' in top_anchor.pin
        m2_radius = (top_anchor.pin['c0'].y - bottom_anchor.pin['c0'].y) / 2
        self.m2_ic.strt(length / 2 + m2_extension * has_c1).put(top_anchor.pin['c0'])
        self.m2_ic.bend(m2_radius, 180).put()
        x = self.m2_ic.strt(length + 2 * m2_extension * has_c1).put()
        nd.Pin('pos0').put(x.pin['a0'])
        nd.Pin('pos1').put(x.pin['b0'])
        self.m2_ic.bend(m2_radius, 180).put()
        self.m2_ic.strt(length / 2 + m2_extension * has_c1).put()
        if has_c1:
            interpad_distance_x = top_anchor.pin['c1'].x - top_anchor.pin['c2'].x
            interpad_distance_y = top_anchor.pin['c2'].y - bottom_anchor.pin['c2'].y + 2
            m1_radius = (top_anchor.bbox[3] - top_anchor.pin['c1'].y) + 4
            self.m1_ic.bend(m1_radius, 90).put(top_anchor.pin['c1'].x, top_anchor.pin['c1'].y, 90)
            v0 = nd.cp.get_xya()
            self.m1_ic.strt(interpad_distance_x - m1_radius * 2).put()
            v1 = nd.cp.get_xya()
            self.m1_ic.bend(m1_radius, 90).put()
            self.v1_via_4.put(*v0)
            self.v1_via_4.put(*v1)
            nd.Pin('gnd0').put(v0[0], v0[1], 360)
            nd.Pin('gnd1').put(*v1)
            self.m2_ic.strt(interpad_distance_y).put(bottom_anchor.pin['c1'].x, bottom_anchor.pin['c1'].y - 1, 90)
            self.m2_ic.strt(interpad_distance_y).put(bottom_anchor.pin['c2'].x, bottom_anchor.pin['c2'].y - 1, 90)
            for pin in (top_anchor.pin['c1'], top_anchor.pin['c2'], bottom_anchor.pin['c1'], bottom_anchor.pin['c2']):
                self.v1_via.put(pin)

    def gnd_wg(self, waveguide_w: float = 0.48, length: float = 13, gnd_contact_dim: Optional[Dim2] = (2, 2),
               rib_brim_w: float = 1.5, gnd_connector_dim: Optional[Dim2] = (1, 2),
               flip: bool = False, dope_grow: float = 0.25, name='gnd_wg') -> nd.Cell:
        c = GndWaveguide(waveguide_w=waveguide_w, length=length, gnd_contact_dim=gnd_contact_dim,
                         rib_brim_w=rib_brim_w, gnd_connector_dim=gnd_connector_dim, flip=flip)
        pad_to_layer = sum([pad.metal_contact(('cbam', 'm1am', 'v1am', 'm2am')) for pad in c.pads], [])
        dopes = list(zip([p.offset(dope_grow) for p in c.pads], ('pppam',)))
        ridge_etch = [(brim, 'ream') for brim in c.rib_brim]
        device = Multilayer([(c, 'seam')] + pad_to_layer + ridge_etch + dopes)
        return device.nazca_cell(name)

    def autoroute_turn(self, n: Union[int, List, np.ndarray], level: int = 1,
                       period: float = 50, final_period: float = 20, width: float = 8,
                       connector_x: float = 0, connector_y: float = 0, turn_radius: float = 0, overlap: float = 0):
        mt_ic = self.m1_ic if level == 1 else self.m2_ic
        with nd.Cell(f'autoroute_turn_{period}_{final_period}') as autoroute_turn:
            route_arr = np.ones(n) if isinstance(n, int) else np.asarray(n)
            route_num = 0
            for m, route_idx in enumerate(route_arr):
                if route_idx > 0:
                    start = mt_ic.strt(route_num * final_period + connector_x, width).put(-overlap, m * period)
                    if turn_radius > 0:
                        mt_ic.bend(turn_radius, -90, width).put()
                    mt_ic.strt(connector_y, width).put()
                    output = mt_ic.strt(m * period, width).put()
                    nd.Pin(f'a{int(route_num)}').put(start.pin['a0'])
                    nd.Pin(f'b{int(route_num)}').put(start.pin['b0'].x, start.pin['b0'].y, 180)
                    nd.Pin(f'p{int(route_num)}', pin=output.pin['b0']).put()
                    route_num += 1
            nd.put_stub()
        return autoroute_turn

    def nems_anchor(self, fin_dim: Dim2 = (100, 0.15), shuttle_dim: Dim2 = (50, 2),
                    spring_dim: Dim2 = None, straight_connector: Optional[Dim2] = (0.25, 1),
                    tether_connector: Optional[Dim4] = (3, 1, 0.5, 1),
                    pos_electrode_dim: Optional[Dim3] = (90, 4, 0.5),
                    neg_electrode_dim: Optional[Dim2] = (3, 4), dope_expand: float = 0.25, attach_comb: bool = False,
                    dope_grow: float = 0.1, include_fin_dummy: bool = True, name: str = 'nems_anchor'):
        c = NemsAnchor(fin_dim=fin_dim, shuttle_dim=shuttle_dim,
                       spring_dim=spring_dim, straight_connector=straight_connector,
                       tether_connector=tether_connector, pos_electrode_dim=pos_electrode_dim,
                       gnd_electrode_dim=neg_electrode_dim, attach_comb=attach_comb,
                       include_fin_dummy=include_fin_dummy)
        pads, dopes = [], []
        if pos_electrode_dim is not None:
            if neg_electrode_dim is None:
                raise ValueError('must specify neg_electrode_dim if pos_electrode_dim specified but got None')
            pads.extend(sum([pad.metal_contact(('cbam', 'm1am', 'v1am', 'm2am')) for pad in c.pads[:1]], []))
            pads.extend(sum([pad.metal_contact(('cbam', 'm1am')) for pad in c.pads[1:]], []))
            dopes.extend([p.expand(dope_expand).dope('pppam', dope_grow) for p in c.pads])
            dopes.extend([p.expand(dope_expand).dope('pdam', dope_grow) for p in c.springs])
            dopes.append(c.shuttle.expand(dope_expand).dope('pdam', dope_grow))
        else:
            pads.extend(c.shuttle.metal_contact(('cbam', 'm1am', 'v1am', 'm2am')))
            dopes.extend([p.expand(dope_expand).dope('pppam', dope_grow) for p in c.pads])
        device = Multilayer([(c, 'seam')] + pads + dopes)
        return device.nazca_cell(name)

    def double_ps(self, ps: nd.Cell, interport_w: float = 40, name: str = 'double_ps'):
        with nd.Cell(name) as cell:
            pl = ps.put()
            nd.Pin('a0').put(pl.pin['a0'])
            nd.Pin('b0').put(pl.pin['b0'])
            pu = ps.put(0, interport_w)
            nd.Pin('a1').put(pu.pin['a0'])
            nd.Pin('b1').put(pu.pin['b0'])
            if 'pos0' in pu.pin:  # equivalent to "raising pins" but not all phase shifters have these pins...
                nd.Pin('pos0').put(pu.pin['pos0'])
                nd.Pin('pos1').put(pu.pin['pos1'])
            if 'gnd0' in pu.pin:
                nd.Pin('gnd0').put(pu.pin['gnd0'])
                nd.Pin('gnd1').put(pu.pin['gnd1'])
        return cell

    def mzi_arms(self, lower_arm: List[Union[nd.Cell, float, Multilayer]] = None,
                 upper_arm: List[Union[nd.Cell, float, Multilayer]] = None, interport_w: float = 50,
                 name: str = 'mzi_arm'):
        def _put_device(dev: Union[nd.Cell, float, Multilayer]):
            if isinstance(dev, (float, int)):
                return self.waveguide_ic.strt(dev).put()
            elif isinstance(dev, Multilayer):
                device = dev.nazca_cell(f'{name}_dev').put()
                device.raise_pins(['gnd_l', 'pos_l', 'gnd_r', 'pos_r'])
                return device
            else:
                return dev.put()

        with nd.Cell(name) as cell:
            l_devices = [_put_device(dev) for dev in lower_arm]
            nd.cp.goto(0, interport_w)
            u_devices = [_put_device(dev) for dev in upper_arm]
            l_device, u_device = l_devices[0], u_devices[0]
            nd.Pin('a0').put(l_device.pin['a0'])
            nd.Pin('a1').put(u_device.pin['a0'])

            # Use waveguide to complete the arm in case of unequal lengths
            l_device, u_device = l_devices[-1], u_devices[-1]
            lx, ly, la = l_device.pin['b0'].xya()
            ux, uy, ua = u_device.pin['b0'].xya()
            fill_length = abs(lx - ux)
            if ux > lx:
                fill_pin = l_device.pin['b0']
                wg_filler = self.waveguide_ic.strt(fill_length).put(fill_pin)
                nd.Pin('b0').put(wg_filler.pin['b0'])
                nd.Pin('b1').put(u_device.pin['b0'])
            elif ux < lx:
                fill_pin = u_device.pin['b0']
                wg_filler = self.waveguide_ic.strt(fill_length).put(fill_pin)
                nd.Pin('b0').put(l_device.pin['b0'])
                nd.Pin('b1').put(wg_filler.pin['b0'])
            else:
                nd.Pin('b0').put(l_device.pin['b0'])
                nd.Pin('b1').put(u_device.pin['b0'])
        return cell

    def thermal_ps(self, tap_sep: Optional[Tuple[nd.Cell, float]] = None):
        with nd.Cell('thermal_ps') as cell:
            ps = self.pdk_cells['cl_band_thermo_optic_phase_shifter']
            _ps = ps.put(0, 0, 0)
            nd.Pin('a0').put(0, 0, 180)
            nd.Pin('b0').put(100, 0, 0)
            nd.Pin('p').put(_ps.pin['p'])
            self.m2_ic.bend(5.5, 90, width=8).put(_ps.pin['p'].x - 3, _ps.pin['p'].y, 90)
            self.m2_ic.strt(38.8, width=8).put()
            nd.Pin('n').put(_ps.pin['n'])
            self.m2_ic.bend(15.5, 90, width=8).put(_ps.pin['n'].x + 3, _ps.pin['n'].y, 90)
            self.m2_ic.strt(39.8, width=8).put()
            if tap_sep is not None:
                tap, sep = tap_sep
                self.waveguide_ic.strt(sep).put(100, 0, 0)
                t = tap.put()
                nd.Pin('b0').put(t.pin['b0'])
            else:
                nd.Pin('b0').put(100, 0, 0)
        return cell

    def waveguide(self, length: float, waveguide_w: float = 0.48, taper_ls: Tuple[float, ...] = (0,),
                  taper_params: Optional[Tuple[Tuple[float, ...], ...]] = None, symmetric: bool = True):
        c = Waveguide(waveguide_w, length, taper_params=taper_params,
                      num_taper_evaluations=100, symmetric=symmetric, taper_ls=taper_ls)
        device = Multilayer([(c, 'seam')])
        return device.nazca_cell('waveguide')

    def alignment_mark(self, dim: Dim2 = (100, 10)):
        c = AlignmentCross(dim)
        device = Multilayer([(c, 'm1am'), (c.copy, 'm2am'), (c.copy, 'mlam')])
        return device.nazca_cell('alignment_mark')

    def interposer(self, waveguide_w: float, n: int, period: float, radius: float,
                   trombone_radius: Optional[float] = None, num_trombones: int = 1,
                   final_period: Optional[float] = None, self_coupling_extension_dim: Optional[Dim2] = None,
                   horiz_dist: float = 0, with_gratings: bool = True):
        c = Interposer(waveguide_w, n, period, radius, trombone_radius, final_period,
                       self_coupling_extension_dim, horiz_dist, num_trombones=num_trombones)
        device = Multilayer([(c, 'seam')])
        if with_gratings:
            with nd.Cell(f'interposer_with_gratings_{n}_{period}_{final_period}_{radius}') as cell:
                interposer = device.nazca_cell('interposer').put()
                if self_coupling_extension_dim is not None:
                    x, y = interposer.pin['b0'].x, interposer.pin['b0'].y
                    self.grating_array(n + 2, period=final_period).put(x, y - final_period)
                else:
                    self.grating_array(n, period=final_period).put(interposer.pin['b0'])
                for idx in range(n):
                    nd.Pin(f'a{idx}').put(interposer.pin[f'a{idx}'])
            return cell
        else:
            return device.nazca_cell(f'interposer_{n}_{period}_{final_period}_{radius}')

    def bond_pad_array(self, n_pads: Shape2, pitch: Union[float, Dim2] = 100,
                       pad_dim: Dim2 = (40, 40), labels: Optional[np.ndarray] = None,
                       use_labels: bool = True, stagger_x_frac: float = 0, use_ml_only: bool = False):
        pad_w, pad_l = pad_dim
        pitch = pitch if isinstance(pitch, tuple) else (pitch, pitch)
        with nd.Cell(name=f'bond_pad_array_{n_pads}_{pitch}') as bond_pad_array:
            for i in range(n_pads[0]):
                for j in range(n_pads[1]):
                    x_loc = i * pitch[0] + stagger_x_frac * j * pitch[0]
                    pad = self.ml_ic.strt(length=pad_l, width=pad_w).put(x_loc, j * pitch[1], 270)
                    if not use_ml_only:
                        self.m2_ic.strt(-6, width=8).put()
                        self.va_via.put(nd.cp.x(), nd.cp.y(), 90)
                        self.m2_ic.strt(6, width=8).put(x_loc, j * pitch[1], 270)
                        self.va_via.put()
                        # TODO(Nate): Add M1 connections for routing
                    nd.Pin(f'u{i},{j}').put(pad.pin['a0'])
                    nd.Pin(f'd{i},{j}').put(pad.pin['b0'])
                    if use_labels:
                        x = nd.text(text=f'{i + 1 if labels is None else labels[i]}', align='rc',
                                    layer='seam', height=pad_dim[0] / 2)
                        y = nd.text(text=f'{j + 1 if labels is None else labels[i]}', align='rc',
                                    layer='seam', height=pad_dim[0] / 2)
                        y.put(x_loc, -pad_l / 2 + j * pitch[1])
                        x.put(x_loc, 0.27 * pad_l + j * pitch[1])
            nd.put_stub()
        return bond_pad_array

    def eutectic_array(self, n_pads: Shape2, pitch: float = 20, width: float = 12, strip: bool = True):
        def oct(w: float):
            a = w / (1 + np.sqrt(2)) / 2
            return [(w / 2, a), (a, w / 2), (-a, w / 2), (-w / 2, a),
                    (-w / 2, -a), (-a, -w / 2), (a, -w / 2), (w / 2, -a)]

        pitch = pitch if isinstance(pitch, tuple) else (pitch, pitch)
        with nd.Cell(name=f'eutectic_pad_array_{n_pads}_{pitch}') as bond_pad_array:
            pads = [
                nd.Polygon(oct(width), layer=AIM_STACK['layers']['m1am']),  # m1am
                nd.Polygon(nd.geom.box(0.4, 0.4), layer=AIM_STACK['layers']['v1am']),  # v1am
                nd.Polygon(oct(width), layer=AIM_STACK['layers']['m2am']),  # m2am
                nd.Polygon(nd.geom.box(3.6, 3.6), layer=AIM_STACK['layers']['vaam']),
                nd.Polygon(oct(width), layer=AIM_STACK['layers']['mlam']),
            ]
            for i in range(n_pads[0]):
                for j in range(n_pads[1]):
                    for k, pad in enumerate(pads):
                        pad.put(i * pitch[0], j * pitch[1] + 5.2 * (k == 1) - 0.8 * (k == 3), 270)
                    if j == 0:
                        nd.Pin(f'o{i}').put(i * pitch[0], j * pitch[1], -90)
                    elif j == n_pads[1] - 1:
                        nd.Pin(f'i{i}').put(i * pitch[0], j * pitch[1], 90)
                if strip:
                    self.m2_ic.strt_p2p(bond_pad_array.pin[f'o{i}'], bond_pad_array.pin[f'i{i}'], width=width).put()
                    self.m1_ic.strt_p2p(bond_pad_array.pin[f'o{i}'], bond_pad_array.pin[f'i{i}'],
                                        width=width).put()  # adding m1 to bring resistance down
            nd.put_stub()
        return bond_pad_array

    def custom_dc(self, waveguide_w: float = 0.48, bend_dim: Dim2 = (20, 50.78 / 2), gap_w: float = 0.3,
                  interaction_l: float = 37.8, end_bend_dim: Optional[Dim3] = None,
                  use_radius: bool = True) -> Tuple[nd.Cell, nd.Cell]:
        dc = DC(bend_dim, waveguide_w, gap_w, interaction_l, (0,), None, end_bend_dim, use_radius)
        return dc.nazca_cell('dc', layer='seam'), dc.upper_path.nazca_cell('bendy_dc_dummy', layer='seam')

    def custom_dc_taper(self, waveguide_w: float = 0.48, bend_dim: Dim2 = (20, 50.78 / 2), gap_w: float = 0.3,
                        interaction_l: float = 37.8, end_bend_dim: Optional[Dim3] = None,
                        use_radius: bool = True, taper_ls=(1,), coupler_boundary_taper=(0, 0.02)) -> Tuple[
        nd.Cell, nd.Cell]:
        dc = DC(bend_dim, waveguide_w, gap_w, interaction_l, taper_ls, coupler_boundary_taper, end_bend_dim, use_radius)
        return dc.nazca_cell('dc', layer='seam'), dc.upper_path.nazca_cell('bendy_dc_dummy', layer='seam')

    def pdk_dc(self, radius: float, interport_w: float) -> nd.Cell:
        with nd.Cell('mzi_bb') as cell:
            _mzi_bb = self.pdk_cells['cl_band_splitter_4port_si'].put()
            mzi_bb_interport_w = _mzi_bb.pin['a1'].y - _mzi_bb.pin['a0'].y
            angle = np.arccos(1 - (interport_w - mzi_bb_interport_w) / 4 / radius) * 180 / np.pi
            for pin_name in ('a0', 'a1', 'b1', 'b0'):
                self.waveguide_ic.bend(radius, angle).put(_mzi_bb.pin[pin_name])
                self.waveguide_ic.bend(radius, -angle).put()
                nd.Pin(pin_name).put()
                angle = -angle
        return cell

    def bidirectional_tap(self, radius: float, mesh_bend: bool = False):
        def metal_route(ic, level: int):
            if level == 2:
                ic.strt(4).put()
                ic.bend(4, -90).put()
                ic.strt(16).put()
                ic.bend(4, -90).put()
                ic.strt(31.5).put()
                ic.bend(4, 90).put()
            else:
                ic.strt(14).put()
                ic.bend(4, -90).put()
                ic.strt(16).put()
                ic.bend(4, -90).put()
                ic.strt(23.5).put()
                ic.bend(4, 90).put()
                self.v1_via_4.put()

        with nd.Cell(name=f'bidirectional_tap_{radius}') as cell:
            tap = self.pdk_cells['cl_band_1p_tap_si'].put()
            nd.Pin('a0').put(tap.pin['a0'])
            nd.Pin('a1').put(tap.pin['a1'])
            nd.Pin('b0').put(tap.pin['b0'])
            nd.Pin('b1').put(tap.pin['b1'])
            self.waveguide_ic.bend(radius, angle=-90).put(tap.pin['a1'])
            if mesh_bend:
                self.waveguide_ic.strt(radius / 4).put()
                self.waveguide_ic.bend(radius, angle=-90).put()
            pd0 = self.pdk_cells['cl_band_photodetector_digital'].put(flip=True)
            self.v1_via_4.put(pd0.pin['p'])
            self.m1_ic.bend(4, 90).put(pd0.pin['p'])
            metal_route(self.m1_ic, level=1)
            self.m2_ic.bend(4, 90).put(pd0.pin['n'])
            metal_route(self.m2_ic, level=2)
            self.waveguide_ic.bend(radius, angle=90).put(tap.pin['b1'])
            if mesh_bend:
                self.waveguide_ic.strt(radius / 4).put()
                self.waveguide_ic.bend(radius, angle=90).put()
            pd1 = self.pdk_cells['cl_band_photodetector_digital'].put()
            self.v1_via_4.put(pd1.pin['p'])
            self.m1_ic.bend(4, -90).put(pd1.pin['p'])
            metal_route(self.m1_ic, level=1)
            self.m2_ic.bend(4, -90).put(pd1.pin['n'])
            metal_route(self.m2_ic, level=2)

        return cell

    def triangular_mesh(self, n: int, node: nd.Cell, dummy: Optional[nd.Cell] = None,
                        ps: Optional[nd.Cell] = None, interport_w: float = 50):
        num_straight = (n - 1) - np.hstack([np.arange(1, n), np.arange(n - 2, 0, -1)]) - 1
        n_layers = num_straight.size
        curr_x = 0
        with nd.Cell(name=f'triangular_mesh_{n}') as triangular_mesh:
            for layer in range(n_layers):
                for idx in range(n):
                    if (n + layer) % 2 == idx % 2 and num_straight[layer] < idx < n - 1:
                        _node = node.put(curr_x, interport_w * idx)
                        end = _node
                        if layer == n_layers - 1 and ps is not None:
                            ps.put(_node.pin['b0'])
                            end = ps.put(_node.pin['b1'])
                    elif idx <= num_straight[layer] or (idx == n - 1 and (n + layer) % 2 == idx % 2):
                        _dummy = dummy.put(curr_x, interport_w * idx)
                        end = ps.put(_dummy.pin['b0']) if layer == n_layers - 1 and ps is not None else _dummy
                curr_x = end.pin['b0'].x
            for idx in range(n):
                nd.Pin(f'a{idx}').put(0, interport_w * idx, -180)
                nd.Pin(f'b{idx}').put(curr_x, interport_w * idx, 0)
        return triangular_mesh

    def mzi_node(self, diff_ps: nd.Cell, dc: nd.Cell, tap_internal: Optional[nd.Cell] = None,
                 tap_external: Optional[nd.Cell] = None, name: Optional[str] = 'mzi',
                 include_input_ps: bool = True, grating: Optional[nd.Cell] = None, detector: Optional[nd.Cell] = None,
                 detector_loopback_params: Dim2 = None, sep: float = 0):
        with nd.Cell(name=name) as node:
            if include_input_ps:
                input_ps = diff_ps.put()
                first_dc = dc.put(input_ps.pin['b0'])
                nd.Pin('a0').put(input_ps.pin['a0'])
                nd.Pin('a1').put(input_ps.pin['a1'])
            else:
                first_dc = dc.put()
                nd.Pin('a0').put(first_dc.pin['a0'])
                nd.Pin('a1').put(first_dc.pin['a1'])
            if tap_internal is not None:
                upper_sampler = tap_internal.put(first_dc.pin['b0'])
                lower_sampler = tap_internal.put(first_dc.pin['b1'])
                if sep > 0:
                    conn = self.waveguide_ic.strt(sep).put(upper_sampler.pin['b0'])
                    self.waveguide_ic.strt(sep).put(lower_sampler.pin['b0'])
                    internal_ps = diff_ps.put(conn.pin['b0'])
                else:
                    internal_ps = diff_ps.put(upper_sampler.pin['b0'])
            else:
                internal_ps = diff_ps.put(first_dc.pin['b0'])
            internal_ps.raise_pins(['gnd_l', 'pos_l', 'gnd_r', 'pos_r'])
            second_dc = dc.put(internal_ps.pin['b0'])
            if tap_external is not None:
                upper_sampler = tap_external.put(second_dc.pin['b0'])
                lower_sampler = tap_external.put(second_dc.pin['b1'])
                nd.Pin('b0').put(upper_sampler.pin['b0'])
                nd.Pin('b1').put(lower_sampler.pin['b0'])
            else:
                nd.Pin('b0').put(second_dc.pin['b0'])
                nd.Pin('b1').put(second_dc.pin['b1'])
            if grating is not None:
                grating.put(node.pin['b0'].x, node.pin['b0'].y, -90)
            if detector is not None:
                if detector_loopback_params is not None:
                    self.waveguide_ic.bend(detector_loopback_params[0], 180).put(node.pin['b1'])
                    self.waveguide_ic.strt(detector_loopback_params[1]).put()
                    d = detector.put()
                    nd.Pin('p1').put(d.pin['p'])
                    nd.Pin('n1').put(d.pin['n'])
                    self.waveguide_ic.bend(detector_loopback_params[0], -180).put(node.pin['b0'])
                    self.waveguide_ic.strt(detector_loopback_params[1]).put()
                    d = detector.put()
                    nd.Pin('p2').put(d.pin['p'])
                    nd.Pin('n2').put(d.pin['n'])
                else:
                    d = detector.put(node.pin['b1'], flip=True)
                    nd.Pin('p1').put(d.pin['p'])
                    nd.Pin('n1').put(d.pin['n'])
                    d = detector.put(node.pin['b0'])
                    nd.Pin('p2').put(d.pin['p'])
                    nd.Pin('n2').put(d.pin['n'])
            nd.put_stub()
        return node

    def tdc_node(self, diff_ps: nd.Cell, tdc: nd.Cell, tap: Optional[nd.Cell] = None,
                 detector: Optional[nd.Cell] = None, detector_loopback_radius: float = 5):
        with nd.Cell(name=f'tdc') as node:
            input_ps = diff_ps.put()
            _tdc = tdc.put(input_ps.pin['b0'])
            nd.Pin('a0').put(input_ps.pin['a0'])
            nd.Pin('a1').put(input_ps.pin['a1'])
            if tap is not None:
                upper_sampler = tap.put(_tdc.pin['b0'])
                lower_sampler = tap.put(_tdc.pin['b1'])
                nd.Pin('b0').put(upper_sampler.pin['b0'])
                nd.Pin('b1').put(lower_sampler.pin['b0'])
            else:
                nd.Pin('b0').put(_tdc.pin['b0'])
                nd.Pin('b1').put(_tdc.pin['b1'])
            _tdc.raise_pins(['gnd_l', 'pos_l', 'gnd_r', 'pos_r'])
            if detector is not None:
                if detector_loopback_radius > 0:
                    self.waveguide_ic.bend(detector_loopback_radius, 180).put(node.pin['b1'])
                    detector.put()
                    self.waveguide_ic.bend(detector_loopback_radius, -180).put(node.pin['b0'])
                    detector.put()
                else:
                    detector.put(node.pin['b1'])
                    detector.put(node.pin['b0'])
        return node

    def grating_array(self, n: int, period: float, turn_radius: float = 0,
                      connector_x: float = 0, connector_y: float = 0, link_end_gratings_radius: float = 0):
        with nd.Cell(name=f'grating_array_{n}_{period}') as gratings:
            for idx in range(n):
                start = self.waveguide_ic.strt(length=connector_x).put(0, period * idx)
                if turn_radius > 0:
                    self.waveguide_ic.bend(turn_radius, angle=90).put()
                self.waveguide_ic.strt(length=connector_y).put()
                self.pdk_cells['cl_band_vertical_coupler_si'].put(nd.cp.x(), nd.cp.y(), -90)
                nd.Pin(f'a{idx}').put(start.pin['a0'])
            if link_end_gratings_radius > 0:
                r = link_end_gratings_radius
                self.waveguide_ic.bend(radius=r, angle=-180).put(gratings.pin['a0'])
                self.waveguide_ic.strt(length=205).put()
                self.waveguide_ic.bend(radius=r, angle=90).put()
                self.waveguide_ic.strt(length=period * (n - 1) - 6 * r).put()
                self.waveguide_ic.bend(radius=r, angle=90).put()
                self.waveguide_ic.strt(length=205).put()
                self.waveguide_ic.bend(radius=r, angle=-180).put()
            nd.put_stub()
        return gratings

    def tap_line(self, n_taps: int, radius: float = 5, inter_tap_gap: float = 60,
                 inter_wg_dist: float = 350):
        with nd.Cell(name=f'tap_line_{n_taps}_{radius}_{inter_tap_gap}') as tap_line:
            inp = self.waveguide_ic.strt(0).put(0, 0, 90)
            nd.Pin('in').put(inp.pin['a0'])
            for idx in range(n_taps):
                tap = self.pdk_cells['cl_band_1p_tap_si'].put()
                self.waveguide_ic.bend(radius, -90).put(tap.pin['a1'])
                nd.Pin(f'a{2 * idx}').put()
                self.waveguide_ic.bend(radius, 90).put(tap.pin['b1'])
                nd.Pin(f'a{2 * idx + 1}').put()
                self.waveguide_ic.strt(inter_tap_gap / 2 if idx == n_taps - 1 else inter_tap_gap).put(tap.pin['b0'])
            self.waveguide_ic.bend(radius, 90).put()
            self.waveguide_ic.strt(inter_wg_dist).put()
            self.waveguide_ic.bend(radius, 90).put()
            out = self.waveguide_ic.strt(n_taps * (inter_tap_gap + 40) - inter_tap_gap / 2).put()
            nd.Pin('out').put(out.pin['b0'])
            nd.put_stub()
        return tap_line

    def dice_box(self, dim: Dim2 = (100, 2000), bottom_left_corner: Dim2 = (0, 0)):
        with nd.Cell(name=f'dice_box_{dim[0]}_{dim[1]}') as dice_box:
            nd.Polygon(nd.geom.box(*dim),
                       layer=AIM_STACK['layers']['diam']).put(bottom_left_corner[0],
                                                              bottom_left_corner[1] + dim[1] / 2)
        return dice_box

    def drop_port_array(self, n: Union[int, List[int]], waveguide_w: float, period: float, final_taper_width: float):
        with nd.Cell(name=f'drop_port_array_{n}_{period}') as drop_port_array:
            idxs = np.arange(n) if isinstance(n, int) else n
            for idx in idxs:
                self.waveguide_ic.taper(length=50, width1=waveguide_w, width2=final_taper_width).put(0, period * float(
                    idx))
                self.waveguide_ic.bend(radius=10, angle=180, width=final_taper_width).put()
        return drop_port_array

    def tdc_dummy(self, ps: nd.Cell, dc_dummy: nd.Cell, tap: Optional[nd.Cell] = None):
        with nd.Cell(name=f'tdc_dummy') as dummy:
            input_ps = ps.put()
            tdc_dummy = dc_dummy.put(input_ps.pin['b0'])
            nd.Pin('a0').put(input_ps.pin['a0'])
            if tap is not None:
                lower_sampler = tap.put(tdc_dummy.pin['b0'])
                nd.Pin('b0').put(lower_sampler.pin['b0'])
            else:
                nd.Pin('b0').put(tdc_dummy.pin['b0'])
        return dummy

    def mzi_dummy(self, ps: nd.Cell, dc_dummy: nd.Cell, tap_internal: Optional[nd.Cell] = None,
                  tap_external: Optional[nd.Cell] = None, sep: float = 0):
        with nd.Cell(name=f'tdc_dummy') as dummy:
            input_ps = ps.put()
            _dc_dummy = dc_dummy.put(input_ps.pin['b0'])
            if tap_internal is not None:
                lower_sampler = tap_internal.put(_dc_dummy.pin['b0'])
                if sep > 0:
                    conn = self.waveguide_ic.strt(sep).put(lower_sampler.pin['b0'])
                    internal_ps = ps.put(conn.pin['b0'])
                else:
                    internal_ps = ps.put(lower_sampler.pin['b0'])
            else:
                internal_ps = ps.put(_dc_dummy.pin['b0'])
            _dc_dummy = dc_dummy.put(internal_ps.pin['b0'])
            if tap_external is not None:
                lower_sampler = tap_external.put(_dc_dummy.pin['b0'])
                nd.Pin('b0').put(lower_sampler.pin['b0'])
            else:
                nd.Pin('b0').put(_dc_dummy.pin['b0'])
        return dummy

    def cell_to_stl(self, cell: nd.Cell):
        multi = Multilayer.from_nazca_cell(cell)
        multi.fill_material('oxide', growth=1, centered_layer='seam')
        trimeshes = multi.to_trimesh_dict(
            layer_to_zrange=self.stack['zranges'], process_extrusion=self.stack['process_extrusion'])
        for layer, mesh in trimeshes.items():
            mesh.export('{}_{}.stl'.format(cell.cell_name, layer))

    def delay_line(self, waveguide_width: float = 0.48, delay_length: float = 50,
                   bend_radius: float = 5, straight_length: float = 25, number_bend_pairs: int = 1,
                   flip: bool = False):
        return DelayLine(waveguide_width=waveguide_width, delay_length=delay_length,
                         bend_radius=bend_radius, straight_length=straight_length,
                         number_bend_pairs=number_bend_pairs, flip=flip).nazca_cell('delay_line', 'seam')

    def test_column(self, device_list: List[nd.Cell], tapline: nd.Cell, col_name: str,
                    autoroute_node_detector: Callable, dc: Union[nd.Cell, List[nd.Cell]] = None,
                    left_pad_orientation: bool = True):
        with nd.Cell(col_name) as test_column:
            line = tapline.put()
            detector = self.pdk_cells['cl_band_photodetector_digital']
            for i, device in enumerate(device_list):
                # all structures for a tap line should be specified here
                if dc is not None:
                    _dc = dc if isinstance(dc, nd.Cell) else dc[i]
                    if _dc is not None:
                        node = self.mzi_node(device, _dc, include_input_ps=False,
                                             name=f'mzi_{col_name}_{i}').put(line.pin[f'a{2 * i + 1}'])
                    else:
                        node = device.put(line.pin[f'a{2 * i + 1}'])
                else:
                    node = device.put(line.pin[f'a{2 * i + 1}'])
                node.raise_pins(['gnd_r', 'pos_r' if left_pad_orientation else 'pos_l'], [f'gnd{i}', f'pos{i}'])
                d1 = detector.put(node.pin['b0'])
                d2 = detector.put(node.pin['b1'], flip=True)
                autoroute_node_detector(d2.pin['p'], d1.pin['n'], d2.pin['n'], d1.pin['p'])
                nd.Pin(f'd{i}').put(node.pin['b0'])  # this is useful for autorouting the gnd path
            nd.Pin('in').put(line.pin['in'])
            nd.Pin('out').put(line.pin['out'])
        return test_column


def _mzi_angle(waveguide_w: float, gap_w: float, interport_w: float, radius: float):
    return np.arccos(1 - (interport_w - gap_w - waveguide_w) / 4 / radius) * 180 / np.pi
