#!/usr/bin/env python3
import sys
sys.path.insert(0,'../')
print(sys.path)
import simphox

from simphox.typing import Tuple, Optional
from simphox.constants import AMF_STACK

import nazca as nd
import numpy as np

# NOTE: This file currently contains privileged information. Only distribute to those who have signed NDA with AMF.
# This NDA is a MUTUAL CONFIDENTIALITY AGREEMENT signed on 13th March 2020 by Advanced Micro Foundry Pte. Ltd (Co. Reg.
# No. 20170322R) of 11 Science Park Road Singapore 117685 and Olav Solgaard, Stanford University,
# Stanford, CA 94305, USA. Much contribution to this code (aside from authors of this repo) comes from much work done at
# Politecnico de Milano, specifically by Maziyar Milanizadeh. Any further distribution of this work must be done with
# permission of the authors of this file and that of the Polimi group.


class PhotonicChip:
    def __init__(self, process_stack: dict, waveguide_w: float, accuracy: float = 0.001):
        """

        Args:
            process_stack: The stack, in JSON format
            waveguide_w: The width of the waveguide (μm)
        """
        for layer_name in process_stack['layers']:
            nd.add_layer(name=layer_name, layer=process_stack['layers'][layer_name], accuracy=accuracy, overwrite=True)
        for xs_name in process_stack['cross_sections']:
            for layer_dict in process_stack['cross_sections'][xs_name]:
                nd.add_layer2xsection(
                    xsection=xs_name,
                    accuracy=accuracy,
                    overwrite=True,
                    **layer_dict)
        xs = nd.get_xsection('waveguide_xs')
        xs.os = 0.0  # define strt - bend offset
        xs.minimum_radius = 10
        xs = nd.get_xsection('heater_xs')
        xs.os = 0.0
        
        self.waveguide_w = waveguide_w
        self.waveguide_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=10, xs='waveguide_xs')
        self.slab_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=10, xs='slab_xs')
        self.grating_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=10, xs='grating_xs')
        self.heater_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=10, xs='heater_xs')
        self.via_heater_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=10, xs='via_heater_xs')
        self.pad_ic = nd.interconnects.Interconnect(width=100, radius=10, xs='pad_xs')
        self.post_pad_ic = nd.interconnects.Interconnect(width=20, radius=10, xs='pad_xs')
        self.trench_ic = nd.interconnects.Interconnect(width=10, radius=10, xs='trench_xs')
        self.oxide_open_test_ic = nd.interconnects.Interconnect(width=10, radius=10, xs='oxide_open_xs')
        self.oxide_open_small_ic = nd.interconnects.Interconnect(width=50, radius=10, xs='oxide_open_xs')
        self.oxide_open_large_ic = nd.interconnects.Interconnect(width=100, radius=10, xs='oxide_open_xs')

        self.grating = self._grating()

    # Polimi grating (Maziyar Milanizadeh)
    def _grating(self, pitch: float = 0.315, first_radius: float = 24.350,
                 num_ribs: int = 26, waveguide_w: float = 0.5, theta: float = 22.0,
                 grat_l: float = 29.245, grat_w: float = 23.274, last_rib_w: float = 0.55):
        waveguide_ic, slab_ic, grating_ic = self.waveguide_ic, self.slab_ic, self.grating_ic
        with nd.Cell(name='grating') as grating:
            # Local variables
            taper_cord = first_radius * (1 - np.cos(np.radians(theta / 2)))
            taper_offset = waveguide_w / 2 / np.tan(np.radians(theta / 2))  # Offset from center of circles
            final_taper_w = 2 * first_radius * np.sin(np.radians(theta / 2))
            taper_l = first_radius - taper_cord - taper_offset

            grat_sep = 3 * taper_l / 4  # Separation between gratin mask and input port

            # Geometric definition
            # Create taper with circular head
            in_tpr = waveguide_ic.taper(length=taper_l,
                                        width1=waveguide_w,
                                        width2=final_taper_w).put(0, 0)
            tpr_cap = nd.Polygon(layer=10,
                                 points=nd.geometries.pie(radius=first_radius,
                                                          angle=theta))

            tpr_cap.put(-taper_offset, 0, -90 + theta / 2)

            current_radius = first_radius + pitch * 2

            # Create grating
            for i in range(num_ribs):
                rib = nd.Polygon(layer=10,
                                 points=nd.geometries.arc(radius=current_radius,
                                                          angle=theta,
                                                          width=pitch))
                rib.put(-taper_offset, 0, 90 - theta / 2)
                previous_radius = current_radius
                current_radius = previous_radius + 2 * pitch

            # Create last rib
            current_radius = current_radius + last_rib_w / 2
            rib = nd.Polygon(layer=10,
                             points=nd.geometries.arc(radius=current_radius,
                                                      angle=theta,
                                                      width=last_rib_w))
            rib.put(-taper_offset, 0, 90 - theta / 2)

            # Create rib and slab regions
            grating_ic.strt(length=grat_l, width=grat_w).put(grat_sep)
            slab_ic.strt(length=grat_l, width=grat_w).put(grat_sep)
            # Export nodes
            nd.Pin(f'g').put(in_tpr.pin['a0'])

            nd.put_stub([], length=0)

        return grating

    # Polimi heater (Maziyar Milanizadeh)
    @nd.hashme('heater', 'heater_l')
    def heater(self, heater_l: float, via_w: float = 2):
        heater_ic, via_ic = self.heater_ic, self.via_heater_ic
        with nd.Cell(hashme=True) as heater:
            htr = heater_ic.strt(length=heater_l).put(0, 0)
            via_in = via_ic.strt(width=via_w, length=via_w).put(htr.pin['a0'])
            via_out = via_ic.strt(width=via_w, length=via_w).put(htr.pin['b0'])

            # add pins to the building block
            nd.Pin('hl').put(htr.pin['a0'])
            nd.Pin('hr').put(htr.pin['b0'])
            nd.Pin('pl').put(via_in.pin['a0'])
            nd.Pin('pr').put(via_out.pin['b0'])

            nd.put_stub([], length=0)
        return heater


    # Polimi bond pad (Maziyar Milanizadeh)
    def bond_pad(self, pad_w: float = 100, pad_l: float = 100):
        with nd.Cell(name='bond_pad') as bond_pad:
            pad = self.pad_ic.strt(length=pad_l, width=pad_w).put()
            # add pins to the building block
            nd.Pin('a0', pin=pad.pin['a0']).put()
            nd.Pin('b0', pin=pad.pin['b0']).put()
        return bond_pad

    # Polimi 1d bond pad array (Maziyar Milanizadeh)
    @nd.hashme('bond_pad_array', 'n_pads', 'pitch')
    def bond_pad_array(self, n_pads: int, pitch: float = 200,
                       pad_w: float = 100, pad_l: float = 100):
        lattice_bond_pads = []
        with nd.Cell(hashme=True) as bond_pad_array:
            pad = self.bond_pad(pad_w=pad_w, pad_l=pad_l)
            for i in range(n_pads):
                lattice_bond_pads.append(pad.put(i * pitch, 0, 270))
                nd.Pin(f'pdt{i}').put(lattice_bond_pads[i].pin['a0'])
                nd.Pin(f'pdb{i}').put(lattice_bond_pads[i].pin['b0'])
                message = nd.text(text=f'{i}', align='cc', layer=10, height=50)
                message.put(pitch / 2 + i * pitch, -pad_l / 2)
            nd.put_stub([], length=0)
        return bond_pad_array

    def coupler_path(self, angle, interaction_l, radius):
        input_waveguide = self.waveguide_ic.bend(radius=radius, angle=angle).put()
        self.waveguide_ic.bend(radius=radius, angle=-angle).put()
        self.waveguide_ic.strt(length=interaction_l).put()
        self.waveguide_ic.bend(radius=radius, angle=-angle).put()
        output_waveguide = self.waveguide_ic.bend(radius=radius, angle=angle).put()
        return input_waveguide.pin['a0'], output_waveguide.pin['b0']

    def mzi_path(self, angle, arm_l, interaction_l, radius, trench_gap,
                 heater: Optional[nd.Cell] = None, trench: Optional[nd.Cell] = None):
        def put_heater():
            x, y = nd.cp.x(), nd.cp.y()
            if trench:
                trench.put(x, y + trench_gap)
                # trench.put(x, y - trench_gap)
            if heater:
                heater.put(x, y)
        put_heater()
        self.waveguide_ic.strt(length=arm_l).put()
        i_node, l_node = self.coupler_path(angle, interaction_l, radius)
        put_heater()
        self.waveguide_ic.strt(length=arm_l).put()
        r_node, o_node = self.coupler_path(angle, interaction_l, radius)
        return i_node, l_node, r_node, o_node

    @nd.hashme('mzi', 'n_pads', 'pitch')
    def mzi(self, gap_w, interaction_l, mzi_w, arm_l, radius, trench_gap, with_grating_taps=True):
        heater = self.heater(heater_l=arm_l)
        trench = self.trench_ic.strt(length=arm_l)
        with nd.Cell(hashme=True) as mzi:
            angle = np.arccos(1 - (mzi_w - gap_w - self.waveguide_w) / 4 / radius) * 180 / np.pi

            # upper path
            self.grating.put(0, 0, 180)
            self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, 0, 0)
            self.mzi_path(angle, arm_l, interaction_l, radius,
                          trench_gap, heater=heater, trench=trench)
            self.waveguide_ic.strt(length=arm_l).put()
            self.grating.put()

            # lower path
            self.grating.put(0, mzi_w, 180)
            self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, mzi_w, 0)
            self.mzi_path(-angle, arm_l, interaction_l, radius,
                          trench_gap, heater=heater, trench=trench)
            self.waveguide_ic.strt(length=arm_l).put()
            self.grating.put()
        return mzi
    
    # Nate's Small Oxide Opening
    @nd.hashme('oxide_open_small', 'opening_l')
    def oxide_open_small(self, opening_l: float):
        oxide_open_small_ic=self.oxide_open_small_ic
        post_pads_ic=self.post_pad_ic
        offset=oxide_open_small_ic.width+post_pads_ic.width
        with nd.Cell(hashme=True) as oxide_open_small:
            oos = oxide_open_small_ic.strt(length=opening_l).put(0,0)
            pp1 = post_pads_ic.strt(length=0.5*(opening_l-5)).put(0,0.5*offset+5)
            pp2 = post_pads_ic.strt(length=0.5*(opening_l-5)).put(0,-0.5*offset-5)
            pp3 = post_pads_ic.strt(length=0.5*(opening_l-5)).put(opening_l,0.5*offset+5,180)
            pp4 = post_pads_ic.strt(length=0.5*(opening_l-5)).put(opening_l,-0.5*offset-5,180)
            # via_in = via_ic.strt(width=via_w, length=via_w).put(htr.pin['a0'])
            # via_out = via_ic.strt(width=via_w, length=via_w).put(htr.pin['b0'])

            # add pins to the building block
            nd.Pin('hl').put(oos.pin['a0'])
            nd.Pin('hr').put(oos.pin['b0'])
            # nd.Pin('pl').put(via_in.pin['a0'])
            # nd.Pin('pr').put(via_out.pin['b0'])

            nd.put_stub([], length=0)
        return oxide_open_small
    
    @nd.hashme('oxide_open_small', 'opening_l', 'width')
    def oxide_open_large(self, opening_l: float, width: float):
        self.oxide_open_large_ic = nd.interconnects.Interconnect(width=width, radius=10, xs='oxide_open_xs')
        oxide_open_large_ic=self.oxide_open_large_ic
        post_pads_ic=self.post_pad_ic
        offset=oxide_open_large_ic.width+post_pads_ic.width
        with nd.Cell(hashme=True) as oxide_open_large:
            oos = oxide_open_large_ic.strt(length=opening_l).put(0,0)
            pp1 = post_pads_ic.strt(length=0.5*(opening_l-5)).put(0,0.5*offset+5)
            pp2 = post_pads_ic.strt(length=0.5*(opening_l-5)).put(0,-0.5*offset-5)
            pp3 = post_pads_ic.strt(length=0.5*(opening_l-5)).put(opening_l,0.5*offset+5,180)
            pp4 = post_pads_ic.strt(length=0.5*(opening_l-5)).put(opening_l,-0.5*offset-5,180)
            # via_in = via_ic.strt(width=via_w, length=via_w).put(htr.pin['a0'])
            # via_out = via_ic.strt(width=via_w, length=via_w).put(htr.pin['b0'])

            # add pins to the building block
            nd.Pin('hl').put(oos.pin['a0'])
            nd.Pin('hr').put(oos.pin['b0'])
            # nd.Pin('pl').put(via_in.pin['a0'])
            # nd.Pin('pr').put(via_out.pin['b0'])

            nd.put_stub([], length=0)
        return oxide_open_large

    #### Nate's Modification to have a non heater mzi ####
    def mzi_n_path(self, gap_w, interaction_l, port_in_w, port_out_w, grating_arm_l, arm_l, radius, radius_in, radius_out, sign=1,
                 oxide_open_small: Optional[nd.Cell] = None):
        
        del_y_in = (port_in_w - gap_w - self.waveguide_w)/2 
        del_y_out = (port_out_w - gap_w - self.waveguide_w)/2 
        angle_in = sign*np.arccos(1 - (del_y_in / (radius+radius_in))) * 180 / np.pi      
        angle_out = sign*np.arccos(1 - (del_y_out / (radius+radius_in))) * 180 / np.pi
        def put_oxide_opening():
            x, y = nd.cp.x(), nd.cp.y()
            if oxide_open_small:
                oxide_open_small.put(x, y)
        # put_oxide_opening()
        # self.waveguide_ic.strt(length=arm_l).put()
        x, y = nd.cp.x(), nd.cp.y()
        i_node, l_node = self.coupler_path_n(angle_in, interaction_l, angle_out, radius, radius_in, radius_out,x,y)
        put_oxide_opening()
        self.waveguide_ic.strt(length=arm_l).put()
        x, y = nd.cp.x(), nd.cp.y()
        r_node, o_node = self.coupler_path_n(angle_out, interaction_l, angle_in, radius, radius_out, radius_in,x,y)
        return i_node, l_node, r_node, o_node

    @nd.hashme('mzi_n', 'n_pads', 'pitch')
    def mzi_n(self, gap_w, interaction_l, port_in_w, port_out_w, grating_arm_l,arm_l, radius, radius_in, radius_out):
        # oxide_open_small = self.oxide_open_small(opening_l=arm_l)
        with nd.Cell(hashme=True) as mzi_n:
            # sep_mid=162.572 #unfortunately a quick impirical param
            # width=(port_in_w+5)
            oxide_open_small = self.oxide_open_small(opening_l=arm_l)
            # oxide_open_small = self.oxide_open_large(opening_l=arm_l, width=width)

            # upper path
            self.grating.put(0, 0, 180)
            self.waveguide_ic.strt(width=self.waveguide_w, length=grating_arm_l).put(0, 0, 0)
            self.mzi_n_path(gap_w, interaction_l, port_in_w, port_out_w, grating_arm_l, arm_l, radius, radius_in, radius_out, sign=1,
                 oxide_open_small=oxide_open_small)
            self.waveguide_ic.strt(length=grating_arm_l).put()
            self.grating.put()

            # lower path
            self.grating.put(0, port_in_w, 180)
            self.waveguide_ic.strt(width=self.waveguide_w, length=grating_arm_l).put(0, port_in_w, 0)
            self.mzi_n_path(gap_w, interaction_l, port_in_w, port_out_w, grating_arm_l, arm_l, radius, radius_in, radius_out, sign=-1,
                 oxide_open_small=oxide_open_small)
            self.waveguide_ic.strt(length=grating_arm_l).put()
            self.grating.put()
        return mzi_n

    def coupler_path_n(self, angle_in, interaction_l, angle_out, radius, radius_in, radius_out, x, y):
        input_waveguide = self.waveguide_ic.bend(radius=radius_in, angle=angle_in).put(x,y)
        self.waveguide_ic.bend(radius=radius, angle=-angle_in).put()
        self.waveguide_ic.strt(length=interaction_l).put()
        self.waveguide_ic.bend(radius=radius, angle=-angle_out).put()
        output_waveguide = self.waveguide_ic.bend(radius=radius_out, angle=angle_out).put()
        
        return input_waveguide.pin['a0'], output_waveguide.pin['b0']

    @nd.hashme('tunable_region','gap_w', 'interaction_l')
    def tunable_region(self, gap_w, interaction_l, port_in_w, port_out_w, grating_arm_l, radius, radius_in, radius_out, extra_w=10):
        oxide_open_w = max([port_in_w,port_out_w])+extra_w
        oxide_open_ts_ic = nd.interconnects.Interconnect(width=oxide_open_w, radius=10, xs='oxide_open_xs')
        post_pads_ic=self.post_pad_ic

        offset=oxide_open_ts_ic.width+post_pads_ic.width

        del_y_in = (port_in_w - gap_w - self.waveguide_w)/2 
        del_y_out = (port_out_w - gap_w - self.waveguide_w)/2 
        angle_in = np.arccos(1 - (del_y_in / (radius+radius_in))) * 180 / np.pi      
        angle_out = np.arccos(1 - (del_y_out / (radius+radius_in))) * 180 / np.pi
        
        bend_in_x = (radius+radius_in)*np.absolute(np.sin(angle_in*(np.pi/180)))
        bend_out_x = (radius+radius_out)*np.absolute(np.sin(angle_out*(np.pi/180)))
        opening_l=interaction_l+bend_in_x+bend_out_x

        ## Notes for arcs: delx= r*sin(angle), dely= r*(1-cos(angle))
        with nd.Cell(hashme=True) as tunable_region:
            oos = oxide_open_ts_ic.strt(length=opening_l).put(0,0)
            # x, y = nd.cp.x(), nd.cp.y()
            u_in_pin, u_out_pin = self.coupler_path_n(angle_in, interaction_l, angle_out, radius, radius_in, radius_out,0,-port_in_w/2)
            l_in_pin, l_out_pin = self.coupler_path_n(-angle_in, interaction_l, -angle_out, radius, radius_in, radius_out,0,port_in_w/2)

            pp1 = post_pads_ic.strt(length=0.5*(opening_l-5)).put(0,0.5*offset+5)
            pp2 = post_pads_ic.strt(length=0.5*(opening_l-5)).put(0,-0.5*offset-5)
            pp3 = post_pads_ic.strt(length=0.5*(opening_l-5)).put(opening_l,0.5*offset+5,180)
            pp4 = post_pads_ic.strt(length=0.5*(opening_l-5)).put(opening_l,-0.5*offset-5,180)
            
            
            # add pins to the building block
            # nd.Pin('u_wg_in', pin=u_in_pin).put(0,port_in_w/2,180)
            # nd.Pin('u_wg_out', pin=u_out_pin).put(opening_l,port_out_w/2)
            # nd.Pin('l_wg_in', pin=l_in_pin).put(0,-port_in_w/2,180)
            # nd.Pin('l_wg_out', pin=l_out_pin).put(opening_l,-port_out_w/2)

            # nd.Pin('a0', pin=l_in_pin).put(l_in_pin)
            # nd.Pin('b0', pin=l_out_pin).put(l_out_pin)
            # nd.Pin('a1', pin=u_in_pin).put(u_in_pin)
            # nd.Pin('b1', pin=u_out_pin).put(u_out_pin)

            nd.Pin('a0').put(l_in_pin)
            nd.Pin('b0').put(l_out_pin)
            nd.Pin('a1').put(u_in_pin)
            nd.Pin('b1').put(u_out_pin)

            nd.put_stub([], length=0)
        return tunable_region



    @nd.hashme('tunable_splitter', 'gap_w', 'interaction_l')
    def tunable_splitter(self, gap_w, interaction_l, port_in_w, port_out_w, grating_arm_l, radius, radius_in, radius_out, extra_w=10):
        with nd.Cell(hashme=True) as tunable_splitter:
            # upper path
            grating_u_in=self.grating.put(0, 0, 180)
            wg_u_in=self.waveguide_ic.strt(width=self.waveguide_w, length=grating_arm_l).put(grating_u_in.pin['a0'])
            # insert opening here
            tunable_region=self.tunable_region(gap_w, interaction_l, port_in_w, port_out_w, grating_arm_l, radius, radius_in, radius_out, extra_w).put(wg_u_in.pin['b0'])
            a0=tunable_region.pin['a0']
            a1=tunable_region.pin['a1']
            b0=tunable_region.pin['b0']
            b1=tunable_region.pin['b1']
            # finish upper path
            wg_u_out=self.waveguide_ic.strt(length=grating_arm_l).put(b0)
            self.grating.put(wg_u_out.pin['b0'])

          
            # lower path            
            wg_l_in=self.waveguide_ic.strt(width=self.waveguide_w, length=grating_arm_l).put(a1,180)
            self.grating.put(wg_l_in.pin['b0'],180)
            wg_l_out=self.waveguide_ic.strt(width=self.waveguide_w, length=grating_arm_l).put(b1)
            self.grating.put(wg_l_out.pin['b0'])
            

        return tunable_splitter

    
    @nd.hashme('directional_coupler', 'gap_w', 'interaction_l')
    def directional_coupler(self, gap_w, interaction_l, port_in_w, port_out_w, grating_arm_l, radius, radius_in, radius_out, extra_w=10):
        with nd.Cell(hashme=True) as directional_coupler:
            
            del_y_in = (port_in_w - gap_w - self.waveguide_w)/2 
            del_y_out = (port_out_w - gap_w - self.waveguide_w)/2 

            angle_in = np.arccos(1 - (del_y_in / (radius+radius_in))) * 180 / np.pi
            
            angle_out = np.arccos(1 - (del_y_out / (radius+radius_in))) * 180 / np.pi

            # lower path
            self.grating.put(0, 0, 180)
            self.waveguide_ic.strt(width=self.waveguide_w, length=grating_arm_l).put(0, 0, 0)
            x, y = nd.cp.x(), nd.cp.y()
            self.coupler_path_n(angle_in, interaction_l, angle_out, radius, radius_in, radius_out, x, y)
            self.waveguide_ic.strt(length=grating_arm_l).put()
            self.grating.put()

            # # upper path
            offset_in=port_in_w
            offset_out=port_out_w
            self.grating.put(0, offset_in, 180)
            self.waveguide_ic.strt(width=self.waveguide_w, length=grating_arm_l).put(0, offset_in, 0)
            x, y = nd.cp.x(), nd.cp.y()
            self.coupler_path_n(-angle_in, interaction_l, -angle_out, radius, radius_in, radius_out, x, y)
            self.waveguide_ic.strt(length=grating_arm_l).put()
            self.grating.put()

        return directional_coupler

    @nd.hashme('binary_tree_4', 'n_pads', 'pitch', 'sensor')
    def splitter_tree_4(self, gap_w, interaction_l, mzi_w, arm_l, radius, trench_gap, right_facing=True,
                        sensor=0):
        directions = np.asarray([(1, 1), (1, -1), (-1, 1), (-1, -1)]) if right_facing else \
            np.asarray([(1, 1), (-1, 1), (1, -1), (-1, -1)])

        with nd.Cell(hashme=True) as binary_tree_4:
            heater = self.heater(heater_l=arm_l)
            trench = self.trench_ic.strt(length=arm_l)
            angle = np.arccos(1 - (mzi_w - gap_w - self.waveguide_w) / 4 / radius) * 180 / np.pi
            for idx, direction in enumerate(directions):
                self.grating.put(0, mzi_w * idx, 180)
                self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, mzi_w * idx, 0)
                self.mzi_path(direction[0] * angle, arm_l, interaction_l, radius,
                              trench_gap, heater=heater, trench=trench)
                self.mzi_path(direction[1] * angle, arm_l, interaction_l, radius,
                              trench_gap, heater=heater, trench=trench)
                self.waveguide_ic.strt(length=arm_l).put()
                if sensor != 0:
                    c = idx if sensor < 0 else 4 - idx
                    self.waveguide_ic.strt(length=arm_l * (c + 0.25 * (1 + sensor))).put()
                    self.waveguide_ic.bend(radius=radius, angle=sensor * 90).put()
                    if sensor < 0:
                        self.waveguide_ic.strt(length=((4.5 - idx) * mzi_w - 2 * radius)).put()
                    else:
                        self.waveguide_ic.strt(length=((idx + 1.5) * mzi_w - 2 * radius)).put()
                    self.waveguide_ic.bend(radius=radius, angle=-sensor * 90).put()
                    self.waveguide_ic.strt(length=arm_l * (4.25 - c - 0.25 * sensor)).put()
                    self.mzi_path(sensor * angle, arm_l, interaction_l, radius,
                                  trench_gap, heater=heater, trench=trench)
                self.grating.put()

        return binary_tree_4

    @nd.hashme('triangular_mesh', 'n', 'arm_l', 'gap_w', 'interaction_l')
    def triangular_mesh(self, n, arm_l, gap_w, interaction_l, mzi_w, radius, trench_gap,
                        with_grating_taps=True):

        num_straight = (n - 1) - (np.hstack([np.arange(1, n), np.arange(n - 2, 0, -1)]) + 1)
        bend_angle = np.arccos(1 - (mzi_w - gap_w - self.waveguide_w) / 4 / radius) * 180 / np.pi

        angles, pos_i, pos_l, pos_r, pos_o = [], [], [], [], []

        with nd.Cell(hashme=True) as triangular_mesh:
            heater = self.heater(heater_l=arm_l)
            trench = self.trench_ic.strt(length=arm_l)
            # mesh
            for idx in range(n):
                self.grating.put(0, mzi_w * idx, 180)
                self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, mzi_w * idx, 0)
                # heater.put(arm_l, mzi_w * idx)
                # trench.put(arm_l, mzi_w * idx + trench_gap)
                # trench.put(arm_l, mzi_w * idx - trench_gap)
                # self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(arm_l, mzi_w * idx)
                for layer in range(2 * n - 3):
                    angle = -bend_angle if idx - layer % 2 < n and idx >= num_straight[layer] and (
                            idx + layer) % 2 else bend_angle
                    angle = -angle if idx < num_straight[layer] else angle
                    angles.append(angle)
                    i_node, l_node, r_node, o_node = self.mzi_path(angle, arm_l, interaction_l, radius,
                                                                   trench_gap, heater=heater, trench=trench)

                    nd.Pin(f'i{idx}{layer}').put(i_node)
                    nd.Pin(f'l{idx}{layer}').put(l_node)
                    nd.Pin(f'r{idx}{layer}').put(r_node)
                    nd.Pin(f'o{idx}{layer}').put(o_node)

                    y_offset = (self.waveguide_w + gap_w) * np.sign(angle)

                    pos_i.append((i_node.x, i_node.y - y_offset))
                    pos_l.append((l_node.x, l_node.y - y_offset))
                    pos_r.append((r_node.x, r_node.y - y_offset))
                    pos_o.append((o_node.x, o_node.y - y_offset))
                output = self.waveguide_ic.strt(length=2 * arm_l).put()
                self.grating.put()
                o_node = output.pin['a0']
                heater.put(o_node.x, mzi_w * idx)
                trench.put(o_node.x, mzi_w * idx + trench_gap)

            # grating taps
            if with_grating_taps:
                for (angle, pi, pl, pr, po) in zip(angles, pos_i, pos_l, pos_r, pos_o):
                    for p in (pi, pr):
                        self.waveguide_ic.bend(width=self.waveguide_w, radius=radius, angle=-angle / 4).put(p[0], p[1])
                        self.waveguide_ic.bend(radius=radius, angle=angle / 2).put()
                        self.waveguide_ic.bend(radius=radius, angle=-angle / 4).put()
                        self.waveguide_ic.strt(length=15).put()
                        self.grating.put(nd.cp.x(), nd.cp.y())

                    for p in (pl, po):
                        self.waveguide_ic.bend(width=self.waveguide_w, radius=radius, angle=angle / 4).put(p[0], p[1], -180)
                        self.waveguide_ic.bend(radius=radius, angle=-angle / 2).put()
                        self.waveguide_ic.bend(radius=radius, angle=angle / 4).put()
                        if angle < 0:
                            self.waveguide_ic.bend(radius=57 / 4, angle=np.sign(angle) * 90).put()
                            self.waveguide_ic.strt(length=5).put()
                            self.waveguide_ic.bend(radius=57 / 4, angle=np.sign(angle) * 90).put()
                            self.waveguide_ic.strt(length=45).put()
                            self.grating.put(nd.cp.x(), nd.cp.y())

            nd.put_stub([], length=0)
        return triangular_mesh

    @nd.hashme('test1_wg', 'length')
    def test1_waveguide(self,length):
        with nd.Cell(hashme=True) as test1:
            self.grating.put(0, 0, 180)
            self.waveguide_ic.strt(length=length).put(0,0,0)
            self.grating.put()
        return test1
    
    @nd.hashme('oxide_open_test', 'opening_l')
    def oxide_open_test(self, opening_l: float):
        oxide_open_test_ic=self.oxide_open_test_ic
        with nd.Cell(hashme=True) as oxide_open_small:
            oos = oxide_open_test_ic.strt(length=opening_l).put(0,0)
            
            nd.put_stub([], length=0)
        return oxide_open_small

    @nd.hashme('test2_trench', 'length')
    def test2_trench(self,length):
        with nd.Cell(hashme=True) as test2:
            self.grating.put(0, 0, 180)
            self.waveguide_ic.strt(length=length/3).put(0,0,0)
            self.oxide_open_test(opening_l=length/3).put()
            self.waveguide_ic.strt(length=length*(2/3)).put()
            self.grating.put()
        return test2

    @nd.hashme('test3_bends', 'length', 'radius', 'radius_in','radius_out')
    def test3_bends(self, grating_arm_l, length, radius, radius_in, radius_out,gap_w, port_in_w, port_out_w):
        del_y_in = (port_in_w - gap_w - self.waveguide_w)/2 
        del_y_out = (port_out_w - gap_w - self.waveguide_w)/2 

        angle_in = np.arccos(1 - (del_y_in / (radius+radius_in))) * 180 / np.pi
        
        angle_out = np.arccos(1 - (del_y_out / (radius+radius_in))) * 180 / np.pi
        with nd.Cell(hashme=True) as test3:
            self.grating.put(0, 0, 180)
            self.waveguide_ic.strt(length=grating_arm_l).put(0,0,0)
            self.waveguide_ic.bend(radius=radius_in, angle=angle_in).put()
            self.waveguide_ic.bend(radius=radius, angle=-angle_in).put()
            self.waveguide_ic.strt(length=length).put()
            self.waveguide_ic.bend(radius=radius, angle=-angle_out).put()
            self.waveguide_ic.bend(radius=radius_out, angle=angle_out).put()
            self.waveguide_ic.strt(length=grating_arm_l).put()
            self.grating.put()
        return test3
    


print('Running Test Script')

#################################### The Chip Layout ###################### 
# filename ='amf_post_processing_basic_structures.gds'
# filename ='amf_post_processing_test_structures.gds'
filename= 'amf_post_processing_chiplet.gds'

waveguide_w = 0.5
radius_milan=35
interaction_l_milan=40
port_in_w=70
port_out_w=120
# fudge_factor=9-0.108
# port_out_w=(fudge_factor+262.572-55)/2 #decided not to go with this scheme
padding=50
grating_arm_l=80

ps_l= 800 # can be 400 for a pi phaseshift
ps_l_pi= 400
interaction_l_ts =185

port_out_w_ts=port_in_w

length_test=1400/8

mzi_kwargs = {
    'arm_l': ps_l,
    'gap_w': 0.3,
    'interaction_l': interaction_l_milan,
    'port_in_w': port_in_w,
    'port_out_w': port_out_w,
    'grating_arm_l': grating_arm_l,
    'radius_in':port_in_w/2,
    'radius_out':port_out_w/2,
    'radius': radius_milan,
}

mzi2_kwargs = {
    'arm_l': ps_l_pi,
    'gap_w': 0.3,
    'interaction_l': interaction_l_milan,
    'port_in_w': port_in_w,
    'port_out_w': port_out_w,
    'grating_arm_l': grating_arm_l,
    'radius_in':port_in_w/2,
    'radius_out':port_out_w/2,
    'radius': radius_milan,
}

dc_kwargs = {
    'gap_w': 0.3,
    'interaction_l': interaction_l_milan,
    'port_in_w': port_in_w,
    'port_out_w': port_out_w,
    'grating_arm_l': grating_arm_l,
    'radius_in':port_in_w/2,
    'radius_out':port_out_w/2,
    'radius': radius_milan,
    # 'radius': 35,
}

dc_milan_kwargs = {
    'gap_w': 0.3,
    'interaction_l': interaction_l_milan,
    'port_in_w': port_in_w,
    'port_out_w': port_in_w,
    'grating_arm_l': grating_arm_l,
    'radius_in':radius_milan,
    'radius_out':radius_milan,
    'radius': radius_milan,
    # 'radius': 35,
}

ts_kwargs = {
    'gap_w': 0.3,
    'interaction_l': interaction_l_ts,
    'port_in_w': port_in_w,
    'port_out_w': port_out_w_ts,
    'grating_arm_l': grating_arm_l,
    'radius_in':port_in_w/2,
    'radius_out':port_out_w_ts/2,
    'radius': radius_milan,
    'extra_w':50
}

chip0 = PhotonicChip(AMF_STACK, waveguide_w)
mzi=chip0.mzi_n(**mzi_kwargs).put(0,0)
mzi2=chip0.mzi_n(**mzi2_kwargs).put(0,0)
dc=chip0.directional_coupler(**dc_kwargs).put(0,0)
ts=chip0.tunable_splitter(**ts_kwargs).put(0, 0)
bp=chip0.bond_pad_array(n_pads=10).put(0,0)

mzi_x_sep=mzi.bbox[2]-mzi.bbox[0] + padding
mzi_y_sep=mzi.bbox[3]-mzi.bbox[1] + padding

mzi2_x_sep=mzi2.bbox[2]-mzi2.bbox[0] + padding
mzi2_y_sep=mzi2.bbox[3]-mzi2.bbox[1] + padding

dc_x_sep=dc.bbox[2]-dc.bbox[0] + padding
dc_y_sep=dc.bbox[3]-dc.bbox[1] + padding

ts_x_sep=ts.bbox[2]-ts.bbox[0] + padding
ts_y_sep=ts.bbox[3]-ts.bbox[1] + padding

bp_x_sep=bp.bbox[2]-bp.bbox[0] + padding
bp_y_sep=bp.bbox[3]-bp.bbox[1] + padding

mzi_x_0 = mzi.bbox[0] 
mzi_y_0 = mzi.bbox[1] 
mzi2_x_0 = mzi2.bbox[0] 
mzi2_y_0 = mzi2.bbox[1] 
dc_x_0 = dc.bbox[0]
dc_y_0 = dc.bbox[1]
ts_x_0=ts.bbox[0]
ts_y_0=ts.bbox[1]
bp_x_0=bp.bbox[0]
bp_y_0=bp.bbox[1]

print(mzi_x_sep,mzi_y_sep)
print(dc_x_sep,dc_y_sep)
print(ts_x_sep,ts_y_sep)


nd.clear_layout()

chip = PhotonicChip(AMF_STACK, waveguide_w)
# mzi=chip.mzi_n(**mzi_kwargs).put(-mzi_x_0,-mzi_y_0)
# mzi2=chip.mzi_n(**mzi2_kwargs).put(-mzi2_x_0,-mzi2_y_0- mzi2_y_sep)
# dc=chip.directional_coupler(**dc_kwargs).put(-dc_x_0,-dc_y_0 + mzi_y_sep )
# ts=chip.tunable_splitter(**ts_kwargs).put(-ts_x_0,-ts_y_0 + mzi_y_sep + dc_y_sep)

test1c=chip.test1_waveguide(length=16*length_test).put(0,-350)
test1c=chip.test1_waveguide(length=32*length_test).put(0,-300)
test1c=chip.test1_waveguide(length=64*length_test).put(0,-250)

test1=chip.test1_waveguide(length=length_test).put(0,-400)
test2=chip.test2_trench(length=length_test).put(0,-600)

test1a=chip.test1_waveguide(length=2*length_test).put(0,-450)
test2a=chip.test2_trench(length=2*length_test).put(0,-650)

test1b=chip.test1_waveguide(length=4*length_test).put(0,-500)
test2b=chip.test2_trench(length=4*length_test).put(0,-700)

test1c=chip.test1_waveguide(length=8*length_test).put(0,-550)
test2c=chip.test2_trench(length=8*length_test).put(0,-750)

test_dc=chip.directional_coupler(**dc_kwargs).put(0,-dc_y_0-dc_y_sep-750)
test_dc_milan=chip.directional_coupler(**dc_milan_kwargs).put(0,-dc_y_0-2*dc_y_sep-750)

test3_bends=chip.test3_bends(grating_arm_l=grating_arm_l, length=interaction_l_milan, 
                        radius=radius_milan, radius_in=port_in_w/2, radius_out = port_out_w/2,
                        gap_w=0.3, port_in_w=port_in_w, port_out_w=port_out_w).put(0,-dc_y_0-2.5*dc_y_sep-750)
test3_milan_bends=chip.test3_bends(grating_arm_l=grating_arm_l, length=interaction_l_milan, 
                        radius=radius_milan, radius_in=port_in_w/2, radius_out = port_in_w/2,
                        gap_w=0.3, port_in_w=port_in_w, port_out_w=port_in_w).put(0,-dc_y_0-3*dc_y_sep-750)
print(mzi_x_sep,mzi_y_sep)
print(dc_x_sep,dc_y_sep)
print(ts_x_sep,ts_y_sep)

nd.clear_layout()

dice_lane=100
chipx=4000-dice_lane
chipy0=2800-dice_lane
chipy=chipy0-6*bp_y_sep
print(chipy)

## Exp 1: Tunable DCs

Ny=int(np.floor((chipy-padding)/ts_y_sep))

# bond_pad_array = chip.bond_pad_array(n_pads=Ny)
# bond_pad_array.put(-bp_x_0-100, -bp_y_0)
# bond_pad_array.put(-bp_x_0, -bp_y_0+bp_y_sep)
# bond_pad_array.put(-bp_x_0-100, chipy-bp_y_0-bp_y_sep)
# bond_pad_array.put(-bp_x_0, chipy-bp_y_0-2*bp_y_sep)

x0 = 0.5*(chipx-(2* ts_x_sep + mzi_x_sep + mzi2_x_sep))

y0=3*bp_y_sep + padding + 0.5*((chipy-padding)-(Ny*ts_y_sep))
for ny in range(Ny):
    x=x0
    y=y0+ny*ts_y_sep
    print(y)    
    chip.tunable_splitter(**ts_kwargs).put(x-ts_x_0, y-ts_y_0)

## Exp 2: Phase Shifting Arms

Ny2=int(np.floor((chipy-padding)/mzi_y_sep))

fudge_factor2 = (496.75-489.475) #The shift in the designs b/c of the bends
fudge_factor2 = 0
x0=x0+ts_x_sep+fudge_factor2
y0=3*bp_y_sep + padding + 0.5*((chipy-padding)-(Ny2*mzi_y_sep))
for ny in range(Ny2):
    x=x0
    y=y0+ny*mzi_y_sep
    print(y)    
    chip.mzi_n(**mzi_kwargs).put(x-mzi_x_0,y-mzi_y_0)


## Exp4: MZI 2

fudge_factor3 = (691.084-676.535)
fudge_factor3 = 0
x0=x0+mzi_x_sep+fudge_factor3
y0=3*bp_y_sep + padding + 0.5*((chipy-padding)-(Ny2*mzi2_y_sep))
for ny in range(Ny2):
    x=x0
    y=y0+ny*mzi2_y_sep
    print(y)    
    chip.mzi_n(**mzi2_kwargs).put(x-mzi2_x_0,y-mzi2_y_0)

## Exp 3: tunable DC 2

fudge_factor4 = (2884.584-2855.486)
fudge_factor4 = 0
x0=x0+mzi2_x_sep+fudge_factor4
y0=3*bp_y_sep + padding + 0.5*((chipy-padding)-(Ny*ts_y_sep))
for ny in range(Ny):
    x=x0
    y=y0+ny*ts_y_sep
    print(y)    
    chip.tunable_splitter(**ts_kwargs).put(x-ts_x_0, y-ts_y_0)



#Bond Pads

num_exp1=2
num_exp2=2
N=int(np.ceil((num_exp1*(Ny*2+2) + num_exp2*(Ny2*4+2))))
n_row =19
(N % 20)
bond_pad_array = chip.bond_pad_array(n_pads=N)
i=0
for n in range((N // (2*n_row) )):
    # n=(N // 40)-n
    chip.bond_pad_array(n_pads=n_row).put(-bp_x_0+i,n*bp_y_sep - bp_y_0)
    chip.bond_pad_array(n_pads=n_row).put(-bp_x_0+i, chipy0-bp_y_0-(n+1)*bp_y_sep)
    if i > 0:
        i=i-100
    else:
        i=100
n_r=int((N % (2*n_row)/2))
# n=0
bp_shift = 0.5*(chipx-n_r*200)-50
chip.bond_pad_array(n_pads=n_r).put(-bp_x_0+bp_shift,(n+1)*bp_y_sep - bp_y_0)
chip.bond_pad_array(n_pads=n_r).put(-bp_x_0+bp_shift, chipy0-bp_y_0-(n+2)*bp_y_sep)



nd.export_gds(filename=filename)

