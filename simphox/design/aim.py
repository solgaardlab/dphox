#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:19:16 2020

@author: YuMiao
"""

import nazca as nd
import numpy as np
import nazca.geometries as geom
from ..typing import Optional, List, Tuple


class AIMPhotonicChip:
    def __init__(self, passive_filepath: str, waveguides_filepath: str, waveguide_w: float = 0.48,
                 accuracy: float = 0.001):
        self.waveguide_w = waveguide_w
        self.passive = nd.load_gds(passive_filepath, asdict=True, topcellsonly=False)
        self.waveguides = nd.load_gds(waveguides_filepath, asdict=True, topcellsonly=False)
        # Define layers
        # todo(sunil): use AIM_STACK instead of hardcoding
        nd.add_layer(name='ZLAM', layer=(701, 727), overwrite=True, accuracy=accuracy)
        nd.add_layer(name='REAM', layer=(702, 727), overwrite=True, accuracy=accuracy)
        nd.add_layer(name='SEAM', layer=(709, 727), overwrite=True, accuracy=accuracy)

        nd.add_layer(name='FNAM', layer=(733, 727), overwrite=True, accuracy=accuracy)
        nd.add_layer(name='SNAM', layer=(735, 727), overwrite=True, accuracy=accuracy)

        nd.add_layer(name='TZAM', layer=(737, 727), overwrite=True, accuracy=accuracy)
        nd.add_layer(name='DIAM', layer=(726, 727), overwrite=True, accuracy=accuracy)

        nd.add_layer(name='BSEAMFILL', layer=(727, 727), overwrite=True, accuracy=accuracy)
        nd.add_layer(name='BFNAMFILL', layer=(734, 727), overwrite=True, accuracy=accuracy)
        nd.add_layer(name='BSNAMFILL', layer=(736, 727), overwrite=True, accuracy=accuracy)

        nd.add_layer(name='WGKOAM', layer=(802, 727), overwrite=True, accuracy=accuracy)
        nd.add_layer(name='METKOAM', layer=(803, 727), overwrite=True, accuracy=accuracy)
        nd.add_layer(name='ABSTRACTAM', layer=(804, 727), overwrite=True, accuracy=accuracy)

        self._pdk()

    def _mzi_angle(self, gap_w: float, interport_w: float, radius: float):
        return np.arccos(1 - (interport_w - gap_w - self.waveguide_w) / 4 / radius) * 180 / np.pi

    def _pdk(self):
        # Add pins
        # silicon level device
        # 1. silicon low loss waveguide
        # 1mm
        # with nd.Cell(name='nazca_cl_band_low_loss_wg_1_mm') as self.cl_band_low_loss_wg_1_mm:
        # self.passive['cl_band_low_loss_wg_1_mm'].put()
        # self.passive['cl_band_low_loss_wg_1_mm'].pin['a0'] = nd.Pin('a0').put(0, 0, 180)
        # self.passive['cl_band_low_loss_wg_1_mm'].pin['b0'] = nd.Pin('b0').put(1000, 0, 0)
        # # 2mm
        # self.cl_band_low_loss_wg_2_mm = self.passive['cl_band_low_loss_wg_2_mm']
        # self.cl_band_low_loss_wg_2_mm.pin['a0'] = nd.Pin('a0').put(0, 0, 180)
        # self.cl_band_low_loss_wg_2_mm.pin['b0'] = nd.Pin('b0').put(2000, 0, 0)
        # # 0.5mm
        # self.cl_band_low_loss_wg_0p5_mm = self.passive['cl_band_low_loss_wg_0p5_mm']
        # self.cl_band_low_loss_wg_0p5_mm.put()
        # self.cl_band_low_loss_wg_0p5_mm.pin['a0'] = nd.Pin('a0').put(0, 0, 180)
        # self.cl_band_low_loss_wg_0p5_mm.pin['b0'] = nd.Pin('b0').put(500, 0, 0)

        # 2. silicon coupler
        # grating coupler (vertical couple)
        with nd.Cell(name='nazca_cl_band_vertical_coupler_si') as self.cl_band_vertical_coupler_si:
            self.passive['cl_band_vertical_coupler_si'].put()
            self.passive['cl_band_vertical_coupler_si'].pin['b0'] = nd.Pin('b0').put(0, 0, -90)
            nd.put_stub()
            # edge coupler
            # Notice: for edge coupler, x=0 of the cell has to put at the edge of the design area
        with nd.Cell(name='nazca_cl_band_edge_coupler_si') as self.cl_band_edge_coupler_si:
            self.passive['cl_band_edge_coupler_si'].put()
            # self.passive['cl_band_edge_coupler_si'].pin['a0'] = nd.Pin('a0').put(0, 0, 180)
            # self.passive['cl_band_edge_coupler_si'].pin['b0'] = nd.Pin('b0').put(400, 0, 0)
            self.passive['cl_band_edge_coupler_si'].pin['b0'] = nd.Pin('b0').put(0, 0, 180)
            self.passive['cl_band_edge_coupler_si'].pin['a0'] = nd.Pin('a0').put(400, 0, 0)

        # 3. silicon 4 port 50/50 splitter
        with nd.Cell(name='nazca_cl_band_splitter_4port_si') as self.cl_band_splitter_4port_si:
            self.passive['cl_band_splitter_4port_si'].put()
            self.passive['cl_band_splitter_4port_si'].pin['a0'] = nd.Pin('a0').put(0, 5, 180)
            self.passive['cl_band_splitter_4port_si'].pin['a1'] = nd.Pin('a1').put(0, -5, 180)
            self.passive['cl_band_splitter_4port_si'].pin['b0'] = nd.Pin('b0').put(200, 5, 0)
            self.passive['cl_band_splitter_4port_si'].pin['b1'] = nd.Pin('b1').put(200, -5, 0)

        # 4. silicon Y junction
        with nd.Cell(name='nazca_cl_band_splitter_3port_si') as self.cl_band_splitter_3port_si:
            self.passive['cl_band_splitter_3port_si'].put()
            self.passive['cl_band_splitter_3port_si'].pin['a0'] = nd.Pin('a0').put(0, 0, 180)
            self.passive['cl_band_splitter_3port_si'].pin['b0'] = nd.Pin('b0').put(100, 5, 0)
            self.passive['cl_band_splitter_3port_si'].pin['b1'] = nd.Pin('b1').put(100, -5, 0)

        # silicon nitride hybride escalator device
        # 5. escalator
        # Nitride waveguide to silicon waveguide
        with nd.Cell(name='nazca_cl_band_escalator_FN_SE') as self.cl_band_escalator_FN_SE:
            self.passive['cl_band_escalator_FN_SE'].put()
            self.passive['cl_band_escalator_FN_SE'].pin['a0'] = nd.Pin('a0').put(0, 0, 180)
            self.passive['cl_band_escalator_FN_SE'].pin['b0'] = nd.Pin('b0').put(40, 0, 0)
            # Nitride waveguide to nitride slot waveguide
        with nd.Cell(name='nazca_cl_band_escalator_FN_FNSN') as self.cl_band_escalator_FN_FNSN:
            self.passive['cl_band_escalator_FN_FNSN'].put()
            self.passive['cl_band_escalator_FN_FNSN'].pin['a0'] = nd.Pin('a0').put(0, 0, 180)
            self.passive['cl_band_escalator_FN_FNSN'].pin['b0'] = nd.Pin('b0').put(40, 0, 0)

        # Nitride layer device
        # 2b. nitride coupler
        # grating coupler (vertical couple)
        with nd.Cell(name='nazca_cl_band_vertical_coupler_FN') as self.cl_band_vertical_coupler_FN:
            self.passive['cl_band_vertical_coupler_FN'].put()
            self.passive['cl_band_vertical_coupler_FN'].pin['b0'] = nd.Pin('b0').put(0, 0, -90)
            # edge coupler
            # Notice: for edge coupler, x=0 of the cell has to put at the edge of the design area
        with nd.Cell(name='nazca_cl_band_edge_coupler_FN') as self.cl_band_edge_coupler_FN:
            self.passive['cl_band_edge_coupler_FN'].put()
            # self.passive['cl_band_edge_coupler_FN'].pin['a0'] = nd.Pin('a0').put(0, 0, 180)
            # self.passive['cl_band_edge_coupler_FN'].pin['b0'] = nd.Pin('b0').put(300, 0, 0)
            self.passive['cl_band_edge_coupler_FN'].pin['a0'] = nd.Pin('a0').put(300, 0, 0)
            self.passive['cl_band_edge_coupler_FN'].pin['b0'] = nd.Pin('b0').put(0, 0, 180)

        # 3b. nitride 4 port 50/50 splitter
        with nd.Cell(name='nazca_cl_band_splitter_4port_FN') as self.cl_band_splitter_4port_FN:
            self.passive['cl_band_splitter_4port_FN'].put()
            self.passive['cl_band_splitter_4port_FN'].pin['a0'] = nd.Pin('a0').put(0, 5, 180)
            self.passive['cl_band_splitter_4port_FN'].pin['a1'] = nd.Pin('a1').put(0, -5, 180)
            self.passive['cl_band_splitter_4port_FN'].pin['b0'] = nd.Pin('b0').put(400, 5, 0)
            self.passive['cl_band_splitter_4port_FN'].pin['b1'] = nd.Pin('b1').put(400, -5, 0)

        # 4b. nitride Y junction (first nitride layer)
        with nd.Cell(name='nazca_cl_band_splitter_3port_FN') as self.cl_band_splitter_3port_FN:
            self.passive['cl_band_splitter_3port_FN'].put()
            self.passive['cl_band_splitter_3port_FN'].pin['a0'] = nd.Pin('a0').put(0, 0, 180)
            self.passive['cl_band_splitter_3port_FN'].pin['b0'] = nd.Pin('b0').put(200, 5, 0)
            self.passive['cl_band_splitter_3port_FN'].pin['b1'] = nd.Pin('b1').put(200, -5, 0)

        # ADD WAVEGUIDES GDS
        with nd.Cell(name='nazca_si_480nm_offset_30um') as self.si_480nm_offset_30um:
            self.waveguides['si_480nm_offset_30um'].put()
            self.waveguides['si_480nm_offset_30um'].pin['a0'] = nd.Pin('a0').put(0, 0, 180)
            self.waveguides['si_480nm_offset_30um'].pin['b0'] = nd.Pin('b0').put(50, -30, 0)
            nd.put_stub()

        with nd.Cell(name='nazca_cl_band_1p_tap_si') as self.cl_band_1p_tap_si:
            self.passive['cl_band_1p_tap_si'].put()
            self.passive['cl_band_1p_tap_si'].pin['a0'] = nd.Pin('a0').put(0, 5, 180)
            self.passive['cl_band_1p_tap_si'].pin['a1'] = nd.Pin('a1').put(0, -5, 180)
            self.passive['cl_band_1p_tap_si'].pin['b0'] = nd.Pin('b0').put(40, 5, 0)
            self.passive['cl_band_1p_tap_si'].pin['b1'] = nd.Pin('b1').put(40, -5, 0)

    # parameterized single mode waveguide for silicon and nitride
    # 6a. silicon single mode waveguide
    def cl_band_waveguide_si(self, length: float = 30, angle: float = 0, radius: Optional[float] = None):
        """Create a length parameterized silicon SM WG

        Args:
            length (float): length of the SM WG

        Returns:
            Cell: SM WG element

        Note: PDK specify radius should be larger than 5um
            We choose:
                radius = 10um
                width = 0.48um
        """
        with nd.Cell(name='nazca_cl_band_waveguide_si') as cl_band_waveguide_si:
            nd.add_xsection('xs_si')
            nd.add_layer2xsection(xsection='xs_si', layer='SEAM')
            ic = nd.interconnects.Interconnect(xs='xs_si', radius=10 if radius is None else radius,
                                               width=self.waveguide_w)
            i1 = ic.bend(angle=angle, arrow=False).put() if angle != 0 else ic.strt(length=length, arrow=False).put()
            # add pin
            nd.Pin('a0', pin=i1.pin['a0']).put()
            nd.Pin('b0', pin=i1.pin['b0']).put()
        return cl_band_waveguide_si

    # 6b. first layer nitride single mode waveguide
    def cl_band_waveguide_FN(self, length: float = 30, angle: float = 0, radius: Optional[float] = None):
        """Create a length parameterized silicon SM WG

        Args:
            length (float): length of the SM WG

        Returns:
            Cell: SM WG element

        Note: PDK specify radius should be larger than 100um
            We choose:
                radius = 100um
                width = 1.5um
        """

        with nd.Cell(name='nazca_cl_band_waveguide_FN') as cl_band_waveguide_FN:
            nd.add_xsection('xs_FN')
            nd.add_layer2xsection(xsection='xs_FN', layer='FNAM')
            ic = nd.interconnects.Interconnect(xs='xs_FN', radius=100 if radius is None else radius, width=1.5)
            i1 = ic.bend(angle=angle, arrow=False).put() if angle != 0 else ic.strt(length=length, arrow=False).put()

            # add pin
            nd.Pin('a0', pin=i1.pin['a0']).put()
            nd.Pin('b0', pin=i1.pin['b0']).put()
        return cl_band_waveguide_FN

    # 6c. first layer nitride single mode waveguide
    def cl_band_waveguide_FNSN(self, length: float = 30, turn: bool = False, angle: float = 90,
                               radius: Optional[float] = None):
        """Create a length parameterized silicon SM WG

        Args:
            length: length of the SM WG

        Returns:
            Cell: SM WG element

        Note: PDK specify radius should be larger than 100um
            We choose:
                radius = 40um
                width = 1.1um
        """

        with nd.Cell(name='nazca_cl_band_waveguide_FNSN') as cl_band_waveguide_FNSN:
            nd.add_xsection('xs_FNSN')
            nd.add_layer2xsection(xsection='xs_FNSN', layer='FNAM')
            nd.add_layer2xsection(xsection='xs_FNSN', layer='SNAM')
            ic = nd.interconnects.Interconnect(xs='xs1', radius=40 if radius is None else radius, width=1.1)
            i1 = ic.bend(angle=angle, arrow=False).put() if turn else ic.strt(length=length, arrow=False).put()
            # add pin
            nd.Pin('a0', pin=i1.pin['a0']).put()
            nd.Pin('b0', pin=i1.pin['b0']).put()
        return cl_band_waveguide_FNSN

    def taper_length(self, start_width, end_width, wavelength):
        if start_width > end_width:
            start_width, end_width = end_width, start_width
        return (end_width ** 2 - start_width ** 2) / (2 * wavelength)

    def horntaper_si(self, start_width, end_width, wavelength, n=500, name=None, xya=None):
        with nd.Cell(name=f"{'cl_band_horntaper_si' if name is None else name}") as taper:
            if start_width < end_width:
                small_width, large_width = start_width, end_width
            else:
                small_width, large_width = end_width, start_width

            xpoints1 = np.linspace(-0.5 * large_width, -0.5 * small_width, n // 2)
            xpoints2 = np.linspace(0.5 * small_width, 0.5 * large_width, n // 2)
            xpoints = np.concatenate((xpoints1, xpoints2))

            ypoints = (4 * xpoints ** 2 - end_width ** 2) / (2 * wavelength)

            taper_points = list(map(lambda x, y: (x, y), xpoints, ypoints))

            nd.Polygon(points=taper_points, layer='SEAM').put(*((0, 0, 0) if xya is None else xya))

            if start_width < end_width:
                nd.Pin('a0').put(0, -np.abs(ypoints).max(), -90)
                nd.Pin('b0').put(0, 0, 90)

            else:
                nd.Pin('b0').put(0, 0, -90)
                nd.Pin('a0').put(0, ypoints.max(), 90)

            nd.put_stub()

        return taper

    def dc(self, gap_w: float, interaction_l: float, interport_w: float, end_l: float, radius: float):
        with nd.Cell(name='dc') as dc:
            nd.add_xsection('xs_si')
            nd.add_layer2xsection(xsection='xs_si', layer='SEAM')
            ic = nd.interconnects.Interconnect(xs='xs_si', radius=radius,
                                               width=self.waveguide_w)
            angle = self._mzi_angle(gap_w, interport_w, radius)
            # upper path
            nd.Pin('a0').put(0, 0, -180)
            ic.strt(length=end_l).put(0, 0, 0)
            _, iw, _ = coupler_path(ic, angle, interaction_l, radius)
            nd.Pin('c0').put(iw.pin['a0'])
            ic.strt(length=end_l).put()
            nd.Pin('b0').put()

            # lower path
            nd.Pin('a1').put(0, interport_w, -180)
            ic.strt(length=end_l).put(0, interport_w, 0)
            _, iw, _ = coupler_path(ic, -angle, interaction_l, radius)
            nd.Pin('c1').put(iw.pin['a0'])
            ic.strt(length=end_l).put()
            nd.Pin('b1').put()

        return dc

    def mzi(self, gap_w: float, interaction_l: float, interport_w: float, arm_l: float, end_l: float, radius: float):
        with nd.Cell(name='mzi') as mzi:
            nd.add_xsection('xs_si')
            nd.add_layer2xsection(xsection='xs_si', layer='SEAM')
            ic = nd.interconnects.Interconnect(xs='xs_si', radius=radius,
                                               width=self.waveguide_w)
            angle = self._mzi_angle(gap_w, interport_w, radius)

            # upper path
            nd.Pin('a0').put(0, 0, -180)
            ic.strt(length=end_l).put(0, 0, 0)
            coupler_path(ic, angle, interaction_l, radius)
            nd.Pin('c0').put()
            ic.strt(length=arm_l).put()
            coupler_path(ic, angle, interaction_l, radius)
            ic.strt(length=end_l).put()
            nd.Pin('b0').put()

            # lower path
            nd.Pin('a1').put(0, interport_w, -180)
            ic.strt(length=end_l).put(0, interport_w, 0)
            coupler_path(ic, -angle, interaction_l, radius)
            nd.Pin('c1').put()
            ic.strt(length=arm_l).put()
            coupler_path(ic, -angle, interaction_l, radius)
            ic.strt(length=end_l).put()
            nd.Pin('b1').put()

        return mzi

    def microbridge_ps(self, bridge_w: float, bridge_l: float, tether_w: float,
                       tether_l: float, block_w: float, block_l: float,
                       radius: float = 10, ring_shape: bool = True):
        with nd.Cell(name='microbridge_ps') as microbridge_ps:
            nd.add_xsection('xs_sin')
            nd.add_layer2xsection(xsection='xs_sin', layer='FNAM')
            ic = nd.interconnects.Interconnect(xs='xs_sin', radius=radius,
                                               width=block_w)

            if ring_shape:
                ic.strt(block_l).put(0, 0)
                ic.bend(angle=180).put()
                ic.strt(block_l).put()
                ic.bend(angle=180).put()
            else:
                ic.strt(block_l).put(0, 0)

            tether_points = geom.box(length=tether_l, width=tether_w)
            bridge_points = geom.box(length=bridge_l, width=bridge_w)

            block_len = 2 * radius * ring_shape + block_w / 2

            nd.Polygon(points=tether_points, layer='FNAM').put(block_l / 2 - tether_l / 2,
                                                               block_len + tether_w / 2)
            nd.Polygon(points=bridge_points, layer='FNAM').put(block_l / 2 - bridge_l / 2,
                                                               block_len + tether_w + bridge_w / 2)

        return microbridge_ps

    def comb_drive_ps(self, cblock_dim: Tuple[float, float],
                      teeth_ys: List[float], big_spring_ys: List[float], anchor_spring_ys: List[float], n_teeth: int,
                      teeth_vert_sep: float, ps_dim: Tuple[float, float], anchor_l: float = 6,
                      big_spring_dim: Tuple[float, float] = (65, 2), anchor_spring_dim: Tuple[float, float] = (15, 9.4),
                      pad_l: float = 100, pad_connect_dim: Tuple[float, float] = (10, 5),
                      spring_edge_w: float = 1, spring_w: float = 0.15, tooth_dim: Tuple[float, float] = (0.15, 2),
                      tooth_pitch: float = 0.6, radius: float = 10):

        with nd.Cell(name='comb_drive') as comb_drive_ps:
            nd.add_xsection('xs_si')
            nd.add_layer2xsection(xsection='xs_si', layer='SEAM')
            ic = nd.interconnects.Interconnect(xs='xs_si', radius=radius, width=spring_w)

            cblock = nd.Polygon(geom.box(length=cblock_dim[0], width=cblock_dim[1]), layer='SEAM')
            connector = nd.Polygon(geom.box(length=pad_connect_dim[0], width=pad_connect_dim[1]), layer='SEAM')
            pad = nd.Polygon(geom.box(length=pad_l, width=pad_l), layer='SEAM')
            ps = ic.strt(length=ps_dim[0], width=ps_dim[1])

            ps.put(0, 0)
            cx, cy = ps.bbox[2] / 2, ps.bbox[3] + pad_connect_dim[1] / 2
            connector.put(cx - connector.bbox[2] / 2, cy)
            cblock.put(cx - cblock.bbox[2] / 2, cy + connector.bbox[3] + cblock_dim[1] / 2)
            connector.put(cx - connector.bbox[2] / 2, cy + connector.bbox[3] + cblock_dim[1] + pad_connect_dim[1] / 2)
            pad.put(cx - pad.bbox[2] / 2, cy + connector.bbox[3] + cblock_dim[1] + pad_connect_dim[1] + pad_l / 2)

            def soft_spring(ss_dim: Tuple[float, float], y: float, with_anchor: bool = False):
                ic.strt(-ss_dim[0], width=spring_w).put(-cblock_dim[0] / 2 + ps.bbox[2] / 2, y)
                ic.strt(ss_dim[1], width=spring_edge_w).put(nd.cp.x() - spring_edge_w / 2, nd.cp.y() - spring_w / 2,
                                                            90)
                ic.strt(ss_dim[0], width=spring_w).put(nd.cp.x() + spring_edge_w / 2, nd.cp.y() - spring_w / 2)
                anchor_x, anchor_y = nd.cp.x() - spring_w, nd.cp.y() - spring_w / 2 - anchor_spring_dim[1] / 2
                if with_anchor:
                    ic.strt(width=anchor_l, length=-anchor_l).put(anchor_x, anchor_y)
                    ic.strt(width=spring_w, length=-ss_dim[1]).put(anchor_x - anchor_l, anchor_y - anchor_l / 2 + spring_w / 2)
                    ic.strt(width=spring_w, length=-ss_dim[1]).put(anchor_x - anchor_l, anchor_y + anchor_l / 2 - spring_w / 2)
                ic.strt(ss_dim[0], width=spring_w).put(cblock_dim[0] / 2 + ps.bbox[2] / 2, y)
                ic.strt(ss_dim[1], width=spring_edge_w).put(nd.cp.x() + spring_edge_w / 2, nd.cp.y() - spring_w / 2, 90)
                ic.strt(ss_dim[0], width=spring_w).put(nd.cp.x() - spring_edge_w / 2, nd.cp.y() - spring_w / 2, -180)
                anchor_x, anchor_y = nd.cp.x() + spring_w, nd.cp.y() - spring_w / 2 - anchor_spring_dim[1] / 2
                if with_anchor:
                    ic.strt(width=anchor_l, length=anchor_l).put(anchor_x, anchor_y)
                    ic.strt(width=spring_w, length=ss_dim[1]).put(anchor_x + anchor_l, anchor_y - anchor_l / 2 + spring_w / 2)
                    ic.strt(width=spring_w, length=ss_dim[1]).put(anchor_x + anchor_l, anchor_y + anchor_l / 2 - spring_w / 2)

            for y in teeth_ys:
                ic.strt(tooth_pitch * n_teeth + tooth_dim[0]).put(cblock_dim[0] / 2 + ps.bbox[2] / 2, y + cy)
                x = nd.cp.x()
                for idx in range(n_teeth):
                    ic.strt(width=tooth_dim[1], length=-tooth_dim[0]).put(x - idx * tooth_pitch,
                                                                          y + cy - tooth_dim[1] / 2 - spring_w / 2)
                    ic.strt(width=tooth_dim[1], length=-tooth_dim[0]).put(x - idx * tooth_pitch - tooth_pitch / 2,
                                                                          y + cy - tooth_dim[1] / 2 - spring_w / 2 -
                                                                          teeth_vert_sep)
                ic.strt(tooth_pitch * n_teeth + tooth_dim[0]).put(-cblock_dim[0] / 2 + ps.bbox[2] / 2, y + cy, 180)
                x = nd.cp.x()
                for idx in range(n_teeth):
                    ic.strt(width=tooth_dim[1], length=tooth_dim[0]).put(x + idx * tooth_pitch,
                                                                         y + cy - tooth_dim[1] / 2 - spring_w / 2, 0)
                    ic.strt(width=tooth_dim[1], length=-tooth_dim[0]).put(x + idx * tooth_pitch - tooth_pitch / 2,
                                                                          y + cy - tooth_dim[1] / 2 - spring_w / 2 -
                                                                          teeth_vert_sep)

            for y in anchor_spring_ys:
                soft_spring(anchor_spring_dim, y + cy, with_anchor=True)

            for y in big_spring_ys:
                soft_spring(big_spring_dim, y + cy)

        return comb_drive_ps


def coupler_path(ic: nd.interconnects.Interconnect, angle: float, interaction_l: float, radius: float = 35):
    input_waveguide = ic.bend(radius=radius, angle=angle).put()
    ic.bend(radius=radius, angle=-angle).put()
    interaction_waveguide = ic.strt(length=interaction_l).put()
    ic.bend(radius=radius, angle=-angle).put()
    output_waveguide = ic.bend(radius=radius, angle=angle).put()
    return input_waveguide, interaction_waveguide, output_waveguide


def trombone(ic: nd.interconnects.Interconnect, height: float, radius: float = 10):
    ic.bend(radius, 90).put()
    ic.strt(height).put()
    ic.bend(radius, -180).put()
    ic.strt(height).put()
    ic.bend(radius, 90).put()
