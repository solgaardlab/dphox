#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:19:16 2020
@author: Sunil Pai, Nate Abebe, Rebecca Hwang, Yu Miao
"""

from . import *
from ..constants import AIM_PDK_WAVEGUIDE_PATH, AIM_PDK_PASSIVE_PATH, AIM_PDK_ACTIVE_PATH, AIM_STACK, AIM_PDK
import numpy as np
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

    def nems_tdc(self, waveguide_w: float = 0.48, nanofin_w: float = 0.22,
                 interaction_l: float = 50, end_l: float = 5, dc_gap_w: float = 0.2, beam_gap_w: float = 0.15,
                 bend_dim: Dim2 = (10, 20), pad_dim: Dim3 = (50, 5, 2), anchor=None,
                 middle_fin_dim=None, use_radius: bool = True, contact_box_dim: Dim2 = (50, 10),
                 clearout_box_dim: Dim2 = (65, 3), dc_taper_ls: Tuple[float, ...] = None,
                 dc_taper=None, beam_taper=None, clearout_etch_stop_grow: float = 0.5,
                 diff_ps: Optional[nd.Cell] = None,
                 name: str = 'nems_tdc') -> nd.Cell:
        c = LateralNemsTDC(waveguide_w=waveguide_w, nanofin_w=nanofin_w,
                           interaction_l=interaction_l, end_l=end_l, dc_gap_w=dc_gap_w, beam_gap_w=beam_gap_w,
                           bend_dim=bend_dim, pad_dim=pad_dim, anchor=anchor,
                           middle_fin_dim=middle_fin_dim, use_radius=use_radius, dc_taper_ls=dc_taper_ls,
                           dc_taper=dc_taper, beam_taper=beam_taper)
        device = c.multilayer(waveguide_layer='seam', metal_stack_layers=['m1am', 'm2am'],
                              doping_stack_layer='ppam', via_stack_layers=['cbam', 'v1am'],
                              clearout_layer='tram', clearout_etch_stop_layer='esam',
                              contact_box_dim=contact_box_dim, clearout_box_dim=clearout_box_dim,
                              clearout_etch_stop_grow=clearout_etch_stop_grow)
        cell = device.nazca_cell(name)
        return self.tdc_node(diff_ps, cell) if diff_ps is not None else cell

    def nems_ps(self, waveguide_w: float = 0.48, nanofin_w: float = 0.22, phaseshift_l: float = 100,
                gap_w: float = 0.15, taper_ls: Tuple[float, ...] = (5,),
                pad_dim: Dim3 = (50, 5, 2), anchor: Tuple[float, ...] = None, contact_box_dim: Dim2 = (50, 5),
                clearout_box_dim: Dim2 = (65, 3), clearout_etch_stop_grow: float = 0.5,
                gap_taper=None, wg_taper=None, num_taper_evaluations: int = 100,
                name: str = 'nems_ps') -> nd.Cell:
        c = LateralNemsPS(waveguide_w=waveguide_w, nanofin_w=nanofin_w, phaseshift_l=phaseshift_l, gap_w=gap_w,
                          num_taper_evaluations=num_taper_evaluations, pad_dim=pad_dim,
                          anchor=anchor, gap_taper=gap_taper, wg_taper=wg_taper,
                          taper_ls=taper_ls)
        device = c.multilayer(waveguide_layer='seam', metal_stack_layers=['m1am', 'm2am'],
                              doping_stack_layer='ppam', via_stack_layers=['cbam', 'v1am'],
                              clearout_layer='tram', clearout_etch_stop_layer='esam',
                              contact_box_dim=contact_box_dim, clearout_box_dim=clearout_box_dim,
                              clearout_etch_stop_grow=clearout_etch_stop_grow)
        return device.nazca_cell(name)

    def nems_anchor(self, fixed_fin_dim: Dim2, bending_fin_dim: Dim2, tether_dim: Dim2,
                    loop_connector: Optional[Dim2] = None, name: str = 'nems_anchor'):
        device = Multilayer({NemsAnchor(fixed_fin_dim=fixed_fin_dim, bending_fin_dim=bending_fin_dim,
                                        tether_dim=tether_dim, loop_connector=loop_connector): 'seam'})
        return device.nazca_cell(name)

    def double_ps(self, ps: nd.Cell, interport_w: float = 40,
                  name: str = 'double_ps'):
        with nd.Cell(name) as cell:
            pl = ps.put()
            nd.Pin('a0').put(pl.pin['a0'])
            nd.Pin('b0').put(pl.pin['b0'])
            pu = ps.put(0, interport_w)
            nd.Pin('a1').put(pu.pin['a0'])
            nd.Pin('b1').put(pu.pin['b0'])
        return cell

    def singlemode_ps(self, ps: nd.Cell, interport_w: float, phaseshift_l: float,
                      top: bool = True, name: str = 'nems_singlemode_ps'):
        with nd.Cell(name) as cell:
            pl = ps.put() if top else self.waveguide_ic.strt(phaseshift_l).put()
            nd.Pin('a0').put(pl.pin['a0'])
            nd.Pin('b0').put(pl.pin['b0'])
            pu = self.waveguide_ic.strt(phaseshift_l).put(0, interport_w) if top else ps.put(0, interport_w)
            nd.Pin('a1').put(pu.pin['a0'])
            nd.Pin('b1').put(pu.pin['b0'])
        return cell

    def thermal_ps(self):
        with nd.Cell('thermal_double_ps') as cell:
            self.pdk_cells['cl_band_thermo_optic_phase_shifter'].put(0, 0, 0)
            nd.Pin('a0').put(0, 0, -180)
            nd.Pin('b0').put(100, 0, 0)
        return cell

    def waveguide(self, length: float, waveguide_w: float = 0.48, taper_ls: Tuple[float, ...] = (0,),
                  taper_params: Optional[Tuple[Tuple[float, ...], ...]] = None, symmetric: bool = True):
        c = Waveguide(waveguide_w, length, taper_params=taper_params,
                      num_taper_evaluations=100, symmetric=symmetric, taper_ls=taper_ls)
        device = c.multilayer(waveguide_layer='seam', metal_stack_layers=['m1am', 'm2am'],
                              doping_stack_layer='ppam', via_stack_layers=['cbam', 'v1am'],
                              clearout_layer='tram', clearout_etch_stop_layer='esam')
        return device.nazca_cell('test_waveguide')

    def nems_miller_node(self, waveguide_w: float, upper_interaction_l: float, lower_interaction_l: float,
                         gap_w: float, bend_radius: float, bend_extension: float, lr_nanofin_w: float,
                         ud_nanofin_w: float, lr_gap_w: float, ud_gap_w: float,
                         contact_box_dim: Dim2, clearout_box_dim: Dim2, clearout_etch_stop_grow: float = 0.5,
                         lr_pad_dim: Optional[Dim2] = None,
                         ud_pad_dim: Optional[Dim2] = None, lr_connector_dim: Optional[Dim2] = None,
                         ud_connector_dim: Optional[Dim2] = None, name: str = 'nems_miller_node') -> nd.Cell:
        c = NemsMillerNode(waveguide_w=waveguide_w, upper_interaction_l=upper_interaction_l, gap_w=gap_w,
                           lower_interaction_l=lower_interaction_l, bend_radius=bend_radius,
                           bend_extension=bend_extension, lr_nanofin_w=lr_nanofin_w, ud_nanofin_w=ud_nanofin_w,
                           lr_gap_w=lr_gap_w, ud_gap_w=ud_gap_w, lr_pad_dim=lr_pad_dim,
                           ud_pad_dim=ud_pad_dim, lr_connector_dim=lr_connector_dim, ud_connector_dim=ud_connector_dim)
        device = c.multilayer(waveguide_layer='seam', metal_stack_layers=['m1am', 'm2am'],
                              doping_stack_layer='ppam', via_stack_layers=['cbam', 'v1am'],
                              clearout_layer='tram', clearout_etch_stop_layer='esam',
                              contact_box_dim=contact_box_dim, clearout_box_dim=clearout_box_dim,
                              clearout_etch_stop_grow=clearout_etch_stop_grow)
        return device.nazca_cell(name)

    def interposer(self, waveguide_w: float, n: int, period: float, radius: float,
                   trombone_radius: Optional[float] = None,
                   final_period: Optional[float] = None, self_coupling_extension_dim: Optional[Dim2] = None,
                   horiz_dist: float = 0, with_gratings: bool = True):
        device = Multilayer({Interposer(waveguide_w, n, period, radius, trombone_radius, final_period,
                                        self_coupling_extension_dim, horiz_dist): 'seam'})
        if with_gratings:
            with nd.Cell(f'interposer_with_gratings_{n}_{period}_{final_period}_{radius}') as cell:
                interposer = device.nazca_cell('interposer').put()
                if self_coupling_extension_dim is not None:
                    x, y = interposer.pin['b0'].x, interposer.pin['b0'].y
                    self.grating_array(n + 2, period=final_period).put(x, y - final_period)
                else:
                    self.grating_array(n, period=final_period).put(interposer.pin['b0'])
            return cell
        else:
            return device.nazca_cell(f'interposer_{n}_{period}_{final_period}_{radius}')

    def bond_pad(self, pad_w: float = 40, pad_l: float = 40):
        with nd.Cell(name='bond_pad') as bond_pad:
            self.pad_ic.strt(length=pad_l, width=pad_w).put()
        return bond_pad

    def bond_pad_array(self, n_pads: Shape2 = (60, 6), pitch: Union[float, Dim2] = 100,
                       pad_dim: Dim2 = (40, 40), labels: Optional[np.ndarray] = None):
        lattice_bond_pads = []
        pad_w, pad_l = pad_dim
        pitch = pitch if isinstance(pitch, tuple) else (pitch, pitch)
        with nd.Cell(name=f'bond_pad_array_{n_pads}_{pitch}') as bond_pad_array:
            pad = self.bond_pad(pad_w=pad_w, pad_l=pad_l)
            for i in range(n_pads[0]):
                for j in range(n_pads[1]):
                    lattice_bond_pads.append(pad.put(i * pitch[0], j * pitch[1], 270))
                    message = nd.text(text=f'{i + 1 if labels is None else labels[i]}', align='cc',
                                      layer='seam', height=pad_dim[0] / 2)
                    message.put(-pitch[0] / 2 + i * pitch[0], -pad_l / 2 + j * pitch[1])
        return bond_pad_array

    def custom_dc(self, waveguide_w: float = 0.48, bend_dim: Dim2 = (20, 50.78 / 2), gap_w: float = 0.3,
                  interaction_l: float = 40, end_l: float = 0, end_bend_dim: Optional[Dim3] = None,
                  use_radius: bool = True) -> Tuple[nd.Cell, nd.Cell]:
        dc = DC(bend_dim, waveguide_w, gap_w, interaction_l, (0,), None, end_l, end_bend_dim, use_radius)
        return dc.nazca_cell('dc', layer='seam'), dc.upper_path.nazca_cell('bendy_dc_dummy', layer='seam')

    def pdk_dc(self, radius: float, interport_w: float) -> nd.Cell:
        return _dc_bb_bends(self, self.pdk_cells['cl_band_splitter_4port_si'], radius, interport_w)

    def bidirectional_tap(self, radius: float, mesh_bend: bool = False):
        with nd.Cell(name=f'bidirectional_tap_{radius}') as cell:
            tap = self.pdk_cells['cl_band_1p_tap_si'].put()
            nd.Pin('a0').put(tap.pin['a0'])
            nd.Pin('a1').put(tap.pin['a1'])
            nd.Pin('b0').put(tap.pin['b0'])
            nd.Pin('b1').put(tap.pin['b1'])
            self.waveguide_ic.bend(radius, angle=-90).put(tap.pin['a1'])
            if mesh_bend:
                pd0 = self.pdk_cells['cl_band_photodetector_digital'].put(flip=True)
            else:
                pd0 = self.pdk_cells['cl_band_photodetector_digital'].put()
            nd.Pin('p0').put(pd0.pin['p'])
            nd.Pin('n0').put(pd0.pin['n'])
            self.waveguide_ic.bend(radius, angle=90).put(tap.pin['b1'])
            pd1 = self.pdk_cells['cl_band_photodetector_digital'].put()
            nd.Pin('p1').put(pd1.pin['p'])
            nd.Pin('n1').put(pd1.pin['n'])
        return cell

    def triangular_mesh(self, n: int, node: nd.Cell, dummy: Optional[nd.Cell] = None,
                        interport_w: float = 50, end_l: float = 30):
        num_straight = (n - 1) - np.hstack([np.arange(1, n), np.arange(n - 2, 0, -1)]) - 1
        n_layers = num_straight.size
        curr_x = 0
        with nd.Cell(name=f'triangular_mesh_{n}') as triangular_mesh:
            for layer in range(n_layers):
                for idx in range(n):
                    if (n + layer) % 2 == idx % 2 and num_straight[layer] < idx < n - 1:
                        _node = node.put(curr_x, interport_w * idx)
                        self.waveguide_ic.strt(end_l).put(_node.pin['b0'])
                        end = self.waveguide_ic.strt(end_l).put(_node.pin['b1'])
                    elif idx <= num_straight[layer] or (idx == n - 1 and (n + layer) % 2 == idx % 2):
                        _dummy = dummy.put(curr_x, interport_w * idx)
                        end = self.waveguide_ic.strt(end_l).put(_dummy.pin['b0'])
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
                    detector.put()
                else:
                    detector.put(node.pin['b1'])
                detector.put(node.pin['b0'])
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
                      connector_x: float = 0, connector_y: float = 0):
        with nd.Cell(name=f'grating_array_{n}_{period}') as gratings:
            for idx in range(n):
                self.waveguide_ic.strt(length=connector_x).put(0, period * idx)
                if turn_radius > 0:
                    self.waveguide_ic.bend(turn_radius, angle=90).put()
                self.waveguide_ic.strt(length=connector_y).put()
                self.pdk_cells['cl_band_vertical_coupler_si'].put(nd.cp.x(), nd.cp.y(), -90)
        return gratings

    def testing_tap_line(self, n_taps: int, radius: float = 5, inter_tap_gap: float = 40):
        with nd.Cell(name=f'testing_tap_line_{n_taps}_{radius}_{inter_tap_gap}') as testing_tap_line:
            self.pdk_cells['cl_band_vertical_coupler_si'].put(0, 0, -90)
            grating = self.waveguide_ic.bend(radius, 90).put()
            pin = grating.pin['b0']
            for idx in range(n_taps):
                self.waveguide_ic.strt(inter_tap_gap).put(pin)
                tap = self.pdk_cells['cl_band_1p_tap_si'].put()
                self.waveguide_ic.bend(radius, -90).put(tap.pin['a1'])
                self.waveguide_ic.strt(2 * radius).put()
                nd.Pin(f'a{2 * idx}').put()
                self.waveguide_ic.bend(radius, 180).put(tap.pin['b1'])
                self.waveguide_ic.bend(radius, -90).put()
                nd.Pin(f'a{2 * idx + 1}').put()
                pin = tap.pin['b0']
            self.waveguide_ic.strt(inter_tap_gap).put(pin)
            self.waveguide_ic.bend(radius, 90).put()
            self.pdk_cells['cl_band_vertical_coupler_si'].put(nd.cp.x(), nd.cp.y(), -90)
            nd.put_stub()
        return testing_tap_line

    def dice_box(self, bottom_left_corner: Dim2 = (0, 0), dim: Dim2 = (2000, 100)):
        with nd.Cell(name=f'dice_box') as dice_box:
            nd.Polygon(nd.geom.box(*dim)).put(bottom_left_corner[0] - dim[0] / 2,
                                              bottom_left_corner[1] - dim[1] / 2)
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

    def ps_tester(self, testing_tap_line: nd.Cell, ps_list: List[nd.Cell], dc: nd.Cell,
                  detector_loopback_params: Tuple[float, float] = (5, 10), name: str = 'ps_tester'):
        with nd.Cell(name) as ps_tester:
            line = testing_tap_line.put()
            for i, ps in enumerate(ps_list):
                self.mzi_node(ps, dc, include_input_ps=False,
                              detector=self.pdk_cells['cl_band_photodetector_digital'],
                              detector_loopback_params=detector_loopback_params).put(line.pin[f'a{i}'])
        return ps_tester


def _mzi_angle(waveguide_w: float, gap_w: float, interport_w: float, radius: float):
    return np.arccos(1 - (interport_w - gap_w - waveguide_w) / 4 / radius) * 180 / np.pi


def _dc_bb_bends(chip: AIMNazca, mzi_bb: nd.Cell, bend_radius: float, interport_w: float):
    with nd.Cell('mzi_bb') as cell:
        _mzi_bb = mzi_bb.put()
        mzi_bb_interport_w = _mzi_bb.pin['a1'].y - _mzi_bb.pin['a0'].y
        angle = np.arccos(1 - (interport_w - mzi_bb_interport_w) / 4 / bend_radius) * 180 / np.pi
        for pin_name in ('a0', 'a1', 'b1', 'b0'):
            chip.waveguide_ic.bend(bend_radius, angle).put(_mzi_bb.pin[pin_name])
            chip.waveguide_ic.bend(bend_radius, -angle).put()
            nd.Pin(pin_name).put()
            angle = -angle
    return cell
