#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:19:16 2020

@author: YuMiao
"""

import nazca as nd
import numpy as np
from ..constants import AIM_PDK_PASSIVE_PATH, AIM_PDK_WAVEGUIDE_PATH
from ..typing import Optional, List, Tuple


class AIMPhotonicChip:
    def __init__(self, passive_filepath: str, waveguides_filepath: str, waveguide_w: float = 0.48,
                 accuracy: float = 0.001):
        self.waveguide_w = waveguide_w
        self.passive = nd.load_gds(AIM_PDK_PASSIVE_PATH, asdict=True, topcellsonly=False)
        self.waveguides = nd.load_gds(AIM_PDK_WAVEGUIDE_PATH, asdict=True, topcellsonly=False)
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

        # Add pins
        # silicon level device
        # 1. silicon low loss waveguide
        # 1mm
        with nd.Cell(name='nazca_cl_band_low_loss_wg_1_mm') as self.cl_band_low_loss_wg_1_mm:
            self.passive['cl_band_low_loss_wg_1_mm'].put()
            self.passive['cl_band_low_loss_wg_1_mm'].pin['a0'] = nd.Pin('a0').put(0, 0, 180)
            self.passive['cl_band_low_loss_wg_1_mm'].pin['b0'] = nd.Pin('b0').put(1000, 0, 0)
            # 2mm
        with nd.Cell(name='nazca_cl_band_low_loss_wg_2_mm') as self.cl_band_low_loss_wg_2_mm:
            self.passive['cl_band_low_loss_wg_2_mm'].put()
            self.passive['cl_band_low_loss_wg_2_mm'].pin['a0'] = nd.Pin('a0').put(0, 0, 180)
            self.passive['cl_band_low_loss_wg_2_mm'].pin['b0'] = nd.Pin('b0').put(2000, 0, 0)
            # 0.5mm
        with nd.Cell(name='nazca_cl_band_low_loss_wg_0p5_mm') as self.cl_band_low_loss_wg_0p5_mm:
            self.passive['cl_band_low_loss_wg_0p5_mm'].put()
            self.passive['cl_band_low_loss_wg_0p5_mm'].pin['a0'] = nd.Pin('a0').put(0, 0, 180)
            self.passive['cl_band_low_loss_wg_0p5_mm'].pin['b0'] = nd.Pin('b0').put(500, 0, 0)

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
            self.passive['cl_band_edge_coupler_si'].pin['a0'] = nd.Pin('a0').put(0, 0, 180)
            self.passive['cl_band_edge_coupler_si'].pin['b0'] = nd.Pin('b0').put(400, 0, 0)

        # 3. silicon 4 port 50/50 splitter
        with nd.Cell(name='nazca_cl_band_splitter_4port_si') as self.cl_band_splitter_4port_si:
            self.passive['cl_band_splitter_4port_si'].put()
            self.passive['cl_band_splitter_4port_si'].pin['b0'] = nd.Pin('a0').put(0, 5, 180)
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
            self.passive['cl_band_edge_coupler_FN'].pin['a0'] = nd.Pin('a0').put(0, 0, 180)
            self.passive['cl_band_edge_coupler_FN'].pin['b0'] = nd.Pin('b0').put(300, 0, 0)

        # 3b. nitride 4 port 50/50 splitter
        with nd.Cell(name='nazca_cl_band_splitter_4port_si') as self.cl_band_splitter_4port_FN:
            self.passive['cl_band_splitter_4port_FN'].put()
            self.passive['cl_band_splitter_4port_FN'].pin['b0'] = nd.Pin('a0').put(0, 5, 180)
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

    # parameterized single mode waveguide for silicon and nitride
    # 6a. silicon single mode waveguide
    def cl_band_waveguide_si(self, length=30, turn=False, angle=90, radius=None):
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
        if radius == None:
            radius = 10

        with nd.Cell(name='nazca_cl_band_waveguide_si') as C:
            nd.add_xsection('xs_si')
            nd.add_layer2xsection(xsection='xs_si', layer='SEAM')
            ic = nd.interconnects.Interconnect(xs='xs_si', radius=radius, width=self.waveguide_w)
            if turn:
                i1 = ic.bend(angle=angle, arrow=False).put()
            else:
                i1 = ic.strt(length=length, arrow=False).put()
            # add pin
            nd.Pin('a0', pin=i1.pin['a0']).put()
            nd.Pin('b0', pin=i1.pin['b0']).put()
        return C

    # 6b. first layer nitride single mode waveguide
    def cl_band_waveguide_FN(self, length: float = 30, turn: bool = False, angle: float = 90,
                             radius: Optional[float] = None):
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

        with nd.Cell(name='nazca_cl_band_waveguide_FN') as C:
            nd.add_xsection('xs_FN')
            nd.add_layer2xsection(xsection='xs_FN', layer='FNAM')
            ic = nd.interconnects.Interconnect(xs='xs_FN', radius=100 if radius is None else radius, width=1.5)
            i1 = ic.bend(angle=angle, arrow=False).put() if turn else ic.strt(length=length, arrow=False).put()

            # add pin
            nd.Pin('a0', pin=i1.pin['a0']).put()
            nd.Pin('b0', pin=i1.pin['b0']).put()
        return C

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

        with nd.Cell(name='nazca_cl_band_waveguide_FNSN') as C:
            nd.add_xsection('xs_FNSN')
            nd.add_layer2xsection(xsection='xs_FNSN', layer='FNAM')
            nd.add_layer2xsection(xsection='xs_FNSN', layer='SNAM')
            ic = nd.interconnects.Interconnect(xs='xs1', radius=40 if radius is None else radius, width=1.1)
            i1 = ic.bend(angle=angle, arrow=False).put() if turn else ic.strt(length=length, arrow=False).put()
            # add pin
            nd.Pin('a0', pin=i1.pin['a0']).put()
            nd.Pin('b0', pin=i1.pin['b0']).put()
        return C

    # Define taper length
    def taper_length(self, start_width, end_width, wavelength):
        if start_width > end_width:
            start_width, end_width = end_width, start_width
        return (end_width ** 2 - start_width ** 2) / (2 * wavelength)

    # Define horn taper
    def horntaper_si(self, start_width, end_width, wavelength, n=500, name=None, xya=None):
        with nd.Cell(name='{}'.format('cl_band_horntaper_si' if name is None else name)) as taper:
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
