#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:19:16 2020
@author: Yu Miao, Sunil Pai, Nate Abebe, Rebecca Hwang
"""

import nazca as nd
from typing import Dict

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

    def nems_tdc(self, waveguide_w, nanofin_w, interaction_l, end_l, dc_gap_w,
                 beam_gap_w, bend_dim, pad_dim, connector_dim, middle_fin_dim, use_radius,
                 contact_box_dim, clearout_box_dim, clearout_etch_stop_grow,
                 diff_ps: Optional[nd.Cell] = None) -> nd.Cell:
        c = LateralNemsTDC(waveguide_w=waveguide_w, nanofin_w=nanofin_w, interaction_l=interaction_l,
                           end_l=end_l, dc_gap_w=dc_gap_w, beam_gap_w=beam_gap_w,
                           bend_dim=bend_dim, pad_dim=pad_dim, connector_dim=connector_dim,
                           middle_fin_dim=middle_fin_dim, use_radius=use_radius)
        device = c.multilayer(waveguide_layer='seam', metal_stack_layers=['m1am', 'm2am'],
                              doping_stack_layer='ppam', via_stack_layers=['cbam', 'v1am'],
                              clearout_layer='tram', clearout_etch_stop_layer='esam',
                              contact_box_dim=contact_box_dim, clearout_box_dim=clearout_box_dim,
                              clearout_etch_stop_grow=clearout_etch_stop_grow)
        cell = device.nazca_cell('nems_tdc')
        return _tdc_node(diff_ps, cell) if diff_ps is not None else cell

    def nems_ps(self, waveguide_w, nanofin_w, phaseshift_l, end_l, gap_w,
                taper_l, pad_dim, connector_dim, contact_box_dim, clearout_box_dim, clearout_etch_stop_grow,
                gap_taper=None, wg_taper=None, num_taper_evaluations=100) -> nd.Cell:
        c = LateralNemsPS(waveguide_w=waveguide_w, nanofin_w=nanofin_w, phaseshift_l=phaseshift_l,
                          end_l=end_l, gap_w=gap_w, taper_l=taper_l, num_taper_evaluations=num_taper_evaluations,
                          pad_dim=pad_dim, connector_dim=connector_dim, gap_taper=gap_taper, wg_taper=wg_taper)
        device = c.multilayer(waveguide_layer='seam', metal_stack_layers=['m1am', 'm2am'],
                              doping_stack_layer='ppam', via_stack_layers=['cbam', 'v1am'],
                              clearout_layer='tram', clearout_etch_stop_layer='esam',
                              contact_box_dim=contact_box_dim, clearout_box_dim=clearout_box_dim,
                              clearout_etch_stop_grow=clearout_etch_stop_grow)
        return device.nazca_cell('nems_ps')

    def nems_diff_ps(self, waveguide_w, nanofin_w, interport_w, phaseshift_l, end_l, gap_w,
                     taper_l, pad_dim, connector_dim,
                     contact_box_dim, clearout_box_dim, clearout_etch_stop_grow, gap_taper=None, wg_taper=None,
                     num_taper_evaluations=100) -> nd.Cell:
        c = LateralNemsDiffPS(waveguide_w=waveguide_w, nanofin_w=nanofin_w, interport_w=interport_w,
                              phaseshift_l=phaseshift_l, end_l=end_l, gap_w=gap_w, taper_l=taper_l,
                              num_taper_evaluations=num_taper_evaluations,
                              pad_dim=pad_dim, connector_dim=connector_dim, gap_taper=gap_taper, wg_taper=wg_taper)
        device = c.multilayer(waveguide_layer='seam', metal_stack_layers=['m1am', 'm2am'],
                              doping_stack_layer='ppam', via_stack_layers=['cbam', 'v1am'],
                              clearout_layer='tram', clearout_etch_stop_layer='esam',
                              contact_box_dim=contact_box_dim, clearout_box_dim=clearout_box_dim,
                              clearout_etch_stop_grow=clearout_etch_stop_grow)
        return device.nazca_cell('nems_diff_ps')

    def bond_pad(self, pad_w: float = 60, pad_l: float = 60):
        with nd.Cell(name='bond_pad') as bond_pad:
            self.pad_ic.strt(length=pad_l, width=pad_w).put()
        return bond_pad

    def bond_pad_array(self, n_pads: int = 25, pitch: float = 100,
                       pad_dim: Dim2 = (60, 60), labels: Optional[np.ndarray] = None):
        lattice_bond_pads = []
        pad_w, pad_l = pad_dim
        with nd.Cell(name=f'bond_pad_array_{n_pads}_{pitch}') as bond_pad_array:
            pad = self.bond_pad(pad_w=pad_w, pad_l=pad_l)
            for i in range(n_pads):
                lattice_bond_pads.append(pad.put(i * pitch, 0, 270))
                message = nd.text(text=f'{i + 1 if labels is None else labels[i]}', align='cc', layer='ream', height=30)
                message.put(-pitch / 2 + i * pitch, -pad_l / 2)
        return bond_pad_array

    def custom_dc(self, waveguide_w: float, bend_dim: Dim2, gap_w: float, interaction_l: float, end_l: float = 0,
                  end_bend_dim: Optional[Dim3] = None, use_radius: bool = True) -> Tuple[nd.Cell, nd.Cell]:
        dc = DC(bend_dim, waveguide_w, gap_w, interaction_l, end_l, end_bend_dim, use_radius)
        return dc.nazca_cell('dc', layer='seam'), dc.upper_path.nazca_cell('bendy_dc_dummy', layer='seam')

    def pdk_dc(self, radius: float, interport_w: float) -> nd.Cell:
        return _dc_bb_bends(self, self.pdk_cells['cl_band_splitter_4port_si'], radius, interport_w)

    def bidirectional_tap(self, radius: float):
        with nd.Cell(name=f'bidirectional_tap_{radius}') as cell:
            tap = self.pdk_cells['cl_band_1p_tap_si'].put()
            nd.Pin('a0').put(tap.pin['a0'])
            nd.Pin('a1').put(tap.pin['a1'])
            nd.Pin('b0').put(tap.pin['b0'])
            nd.Pin('b1').put(tap.pin['b1'])
            self.waveguide_ic.bend(radius, angle=90).put(tap.pin['a1'])
            pd0 = self.pdk_cells['cl_band_photodetector_digital'].put()
            nd.Pin('p0').put(pd0.pin['p'])
            nd.Pin('n0').put(pd0.pin['n'])
            self.waveguide_ic.bend(radius, angle=-90).put(tap.pin['b1'])
            pd1 = self.pdk_cells['cl_band_photodetector_digital'].put()
            nd.Pin('p1').put(pd1.pin['p'])
            nd.Pin('n1').put(pd1.pin['n'])
        return cell

    def triangular_mzi_mesh(self, n: int, waveguide_w: float, nanofin_w: float, arm_l: float, end_l: float,
                            ps_gap_w: float, interport_w: float, pad_dim: Optional[Dim2],
                            connector_dim: Optional[Dim2], contact_box_dim: Dim2, clearout_box_dim: Dim2, radius: float,
                            taper_l: float = 0, clearout_etch_stop_grow: Dim2 = 0.5,
                            gap_taper: Optional[Union[np.ndarray, Tuple[float, ...]]] = None,
                            wg_taper: Optional[Union[np.ndarray, Tuple[float, ...]]] = None,
                            num_taper_evaluations: int = 100, tap_radius: float = 5, custom_dc: bool = False,
                            interaction_l: float = None, gap_w: float = None):
        diff_ps = self.nems_diff_ps(waveguide_w, nanofin_w, interport_w, arm_l, 0, ps_gap_w, taper_l,
                                    pad_dim, connector_dim, contact_box_dim, clearout_box_dim,
                                    clearout_etch_stop_grow, gap_taper, wg_taper, num_taper_evaluations)
        ps = self.nems_ps(waveguide_w, nanofin_w, arm_l, 0, ps_gap_w, taper_l,
                          pad_dim, connector_dim, contact_box_dim, clearout_box_dim,
                          clearout_etch_stop_grow, gap_taper, wg_taper, num_taper_evaluations)
        if custom_dc:
            bend_height = (interport_w - gap_w - waveguide_w) / 2
            if interaction_l is None or gap_w is None:
                raise ValueError('Must specify interaction_l and gap_w.')
            dc, dc_dummy = self.custom_dc(waveguide_w, (radius, bend_height), gap_w, interaction_l, 0)
        else:
            dc = self.pdk_dc(radius, interport_w)
            dc_width = dc.pin['b0'].x - dc.pin['a0'].x
            dc_dummy = Waveguide(waveguide_w, dc_width).nazca_cell('straight_dc_dummy', 'seam')
        tap = self.bidirectional_tap(tap_radius) if tap_radius > 0 else None
        node, dummy = _mzi_node(diff_ps, dc, tap), _mzi_dummy(ps, dc_dummy, tap)
        return _triangular_mesh(n, self.waveguide_ic, node, dummy, interport_w, end_l)

    def triangular_tdc_mesh(self, n: int, waveguide_w: float, nanofin_w: float, arm_l: float, gap_w: float,
                            interaction_l: float, interport_w: float, ps_taper_l: float, ps_pad_dim: Optional[Dim2],
                            ps_connector_dim: Optional[Dim2], tdc_nanofin_w: float, end_l: float, tdc_gap_w: float,
                            beam_gap_w: float, tdc_pad_dim, tdc_connector_dim, middle_fin_dim,
                            ps_contact_box_dim: Dim2, ps_clearout_box_dim: Dim2,
                            tdc_contact_box_dim: Dim2, tdc_clearout_box_dim: Dim2, radius: float,
                            clearout_etch_stop_grow: Dim2, gap_taper: Union[np.ndarray, Tuple[float, ...]],
                            wg_taper: Union[np.ndarray, Tuple[float, ...]], num_taper_evaluations: int = 100,
                            tap_radius: float = 10, use_radius: bool = True):
        bend_dim = (radius, (interport_w - gap_w - waveguide_w) / 2)
        diff_ps = self.nems_diff_ps(waveguide_w, nanofin_w, interport_w, arm_l, 0, gap_w, ps_taper_l,
                                    ps_pad_dim, ps_connector_dim, ps_contact_box_dim, ps_clearout_box_dim,
                                    clearout_etch_stop_grow, gap_taper, wg_taper, num_taper_evaluations)
        ps = self.nems_ps(waveguide_w, tdc_nanofin_w, arm_l, 0, gap_w, ps_taper_l,
                          ps_pad_dim, ps_connector_dim, ps_contact_box_dim, ps_clearout_box_dim,
                          clearout_etch_stop_grow, gap_taper, wg_taper, num_taper_evaluations)
        tdc = self.nems_tdc(waveguide_w, nanofin_w, interaction_l, 0, tdc_gap_w, beam_gap_w, bend_dim,
                            tdc_pad_dim, tdc_connector_dim, middle_fin_dim, use_radius, tdc_contact_box_dim,
                            tdc_clearout_box_dim, clearout_etch_stop_grow)
        dc_width = tdc.bbox[2] - tdc.bbox[0]
        dc_dummy = Waveguide(waveguide_w, dc_width).nazca_cell('straight_tdc_dummy', 'seam')
        tap = self.bidirectional_tap(tap_radius) if tap_radius > 0 else None
        node, dummy = _mzi_node(diff_ps, tdc, tap), _mzi_dummy(ps, dc_dummy, tap)
        return _triangular_mesh(n, self.waveguide_ic, node, dummy, interport_w, end_l)


def _mzi_angle(waveguide_w: float, gap_w: float, interport_w: float, radius: float):
    return np.arccos(1 - (interport_w - gap_w - waveguide_w) / 4 / radius) * 180 / np.pi


def _mzi_node(diff_ps: nd.Cell, dc: nd.Cell, tap: Optional[nd.Cell] = None):
    with nd.Cell(name=f'mzi') as node:
        input_ps = diff_ps.put()
        first_dc = dc.put(input_ps.pin['b0'])
        internal_ps = diff_ps.put(first_dc.pin['b0'])
        second_dc = dc.put(internal_ps.pin['b0'])
        nd.Pin('a0').put(input_ps.pin['a0'])
        nd.Pin('a1').put(input_ps.pin['a1'])
        if tap is not None:
            upper_sampler = tap.put(second_dc.pin['b0'])
            lower_sampler = tap.put(second_dc.pin['b1'])
            nd.Pin('b0').put(upper_sampler.pin['b0'])
            nd.Pin('b1').put(lower_sampler.pin['b0'])
        else:
            nd.Pin('b0').put(second_dc.pin['b0'])
            nd.Pin('b1').put(second_dc.pin['b1'])
    return node


def _tdc_node(diff_ps: nd.Cell, tdc: nd.Cell, tap: Optional[nd.Cell] = None):
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
    return node


def _tdc_dummy(ps: nd.Cell, dc_dummy: nd.Cell, tap: Optional[nd.Cell] = None):
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


def _mzi_dummy(ps: nd.Cell, dc_dummy: nd.Cell, tap: Optional[nd.Cell] = None):
    with nd.Cell(name=f'tdc_dummy') as dummy:
        input_ps = ps.put()
        _dc_dummy = dc_dummy.put(input_ps.pin['b0'])
        internal_ps = ps.put(_dc_dummy.pin['b0'])
        _dc_dummy = dc_dummy.put(internal_ps.pin['b0'])
        if tap is not None:
            lower_sampler = tap.put(_dc_dummy.pin['b0'])
            nd.Pin('b0').put(lower_sampler.pin['b0'])
        else:
            nd.Pin('b0').put(_dc_dummy.pin['b0'])
    return dummy


def _dc_bb_bends(chip: AIMNazca, mzi_bb: nd.Cell, bend_radius: float, interport_w: float):
    with nd.Cell('mzi_bb') as cell:
        _mzi_bb = mzi_bb.put()
        mzi_bb_interport_w = _mzi_bb.pin['a0'].y - _mzi_bb.pin['a1'].y
        angle = np.arccos(1 - (interport_w - mzi_bb_interport_w) / 4 / bend_radius) * 180 / np.pi
        for pin_name in ('a1', 'a0', 'b0', 'b1'):
            chip.waveguide_ic.bend(bend_radius, angle).put(_mzi_bb.pin[pin_name])
            chip.waveguide_ic.bend(bend_radius, -angle).put()
            nd.Pin(pin_name).put()
            angle = -angle
    return cell


def _triangular_mesh(n: int, waveguide_ic: nd.interconnects.Interconnect,
                     node: nd.Cell, dummy: nd.Cell, interport_w: float, end_l: float):
    num_straight = (n - 1) - np.hstack([np.arange(1, n), np.arange(n - 2, 0, -1)]) - 1
    n_layers = num_straight.size
    curr_x = 0
    with nd.Cell(name=f'triangular_mesh_{n}') as triangular_mesh:
        for layer in range(n_layers):
            for idx in range(n):
                if layer % 2 == idx % 2 and num_straight[layer] < idx < n - 1:
                    _node = node.put(curr_x, interport_w * (idx + 1))
                    waveguide_ic.strt(end_l).put(_node.pin['b0'])
                    end = waveguide_ic.strt(end_l).put(_node.pin['b1'])
                elif idx <= num_straight[layer] or (idx == n - 1 and layer % 2 == idx % 2):
                    _dummy = dummy.put(curr_x, interport_w * idx)
                    end = waveguide_ic.strt(end_l).put(_dummy.pin['b0'])
            curr_x = end.pin['b0'].x
    return triangular_mesh