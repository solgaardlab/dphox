#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:19:16 2020

@author: YuMiao
"""

import nazca as nd
import numpy as np

passive = nd.load_gds('../aim_lib/APSUNY_v35a_passive.gds',asdict=True, topcellsonly=False)
waveguides = nd.load_gds('../aim_lib/APSUNY_v35_waveguides.gds',asdict=True, topcellsonly=False)

#print(passive.keys(),'\n')


# Define Layers
nd.add_layer(name='ZLAM', layer=(701,727),overwrite=True)
nd.add_layer(name='REAM', layer=(702,727),overwrite=True)
nd.add_layer(name='SEAM', layer=(709,727),overwrite=True)

nd.add_layer(name='FNAM', layer=(733,727),overwrite=True)
nd.add_layer(name='SNAM', layer=(735,727),overwrite=True)

nd.add_layer(name='TZAM', layer=(737,727),overwrite=True)
nd.add_layer(name='DIAM', layer=(726,727),overwrite=True)

nd.add_layer(name='BSEAMFILL', layer=(727,727),overwrite=True)
nd.add_layer(name='BFNAMFILL', layer=(734,727),overwrite=True)
nd.add_layer(name='BSNAMFILL', layer=(736,727),overwrite=True)

nd.add_layer(name='WGKOAM', layer=(802,727),overwrite=True)
nd.add_layer(name='METKOAM', layer=(803, 727),overwrite=True)
nd.add_layer(name='ABSTRACTAM', layer=(804, 727),overwrite=True)




#Add pins
# silicon level device
# 1. silicon low loss waveguide 
    # 1mm
with nd.Cell(name='nazca_cl_band_low_loss_wg_1_mm') as cl_band_low_loss_wg_1_mm:
    passive['cl_band_low_loss_wg_1_mm'].put()
    passive['cl_band_low_loss_wg_1_mm'].pin['a0']=nd.Pin('a0').put(0,0,180)
    passive['cl_band_low_loss_wg_1_mm'].pin['b0']=nd.Pin('b0').put(1000,0,0)
    # 2mm
with nd.Cell(name='nazca_cl_band_low_loss_wg_2_mm') as cl_band_low_loss_wg_2_mm:
    passive['cl_band_low_loss_wg_2_mm'].put()
    passive['cl_band_low_loss_wg_2_mm'].pin['a0']=nd.Pin('a0').put(0,0,180)
    passive['cl_band_low_loss_wg_2_mm'].pin['b0']=nd.Pin('b0').put(2000,0,0)
    # 0.5mm
with nd.Cell(name='nazca_cl_band_low_loss_wg_0p5_mm') as cl_band_low_loss_wg_0p5_mm:
    passive['cl_band_low_loss_wg_0p5_mm'].put()
    passive['cl_band_low_loss_wg_0p5_mm'].pin['a0']=nd.Pin('a0').put(0,0,180)
    passive['cl_band_low_loss_wg_0p5_mm'].pin['b0']=nd.Pin('b0').put(500,0,0)
    
    
# 2. silicon coupler
    # grating coupler (vertical couple)
with nd.Cell(name='nazca_cl_band_vertical_coupler_si') as cl_band_vertical_coupler_si:
    passive['cl_band_vertical_coupler_si'].put()  
    passive['cl_band_vertical_coupler_si'].pin['b0']=nd.Pin('b0').put(0,0,-90)
    nd.put_stub()
    # edge coupler
    # Notice: for edge coupler, x=0 of the cell has to put at the edge of the design area
with nd.Cell(name='nazca_cl_band_edge_coupler_si') as cl_band_edge_coupler_si:
    passive['cl_band_edge_coupler_si'].put()  
    passive['cl_band_edge_coupler_si'].pin['a0']=nd.Pin('a0').put(0,0,180)
    passive['cl_band_edge_coupler_si'].pin['b0']=nd.Pin('b0').put(400,0,0)


# 3. silicon 4 port 50/50 splitter
with nd.Cell(name='nazca_cl_band_splitter_4port_si') as cl_band_splitter_4port_si:
    passive['cl_band_splitter_4port_si'].put()  
    passive['cl_band_splitter_4port_si'].pin['b0']=nd.Pin('a0').put(0,5,180)
    passive['cl_band_splitter_4port_si'].pin['a1']=nd.Pin('a1').put(0,-5,180)
    passive['cl_band_splitter_4port_si'].pin['b0']=nd.Pin('b0').put(200,5,0)
    passive['cl_band_splitter_4port_si'].pin['b1']=nd.Pin('b1').put(200,-5,0)



# 4. silicon Y junction
with nd.Cell(name='nazca_cl_band_splitter_3port_si') as cl_band_splitter_3port_si:
    passive['cl_band_splitter_3port_si'].put()
    passive['cl_band_splitter_3port_si'].pin['a0']=nd.Pin('a0').put(0,0,180)
    passive['cl_band_splitter_3port_si'].pin['b0']=nd.Pin('b0').put(100,5,0)
    passive['cl_band_splitter_3port_si'].pin['b1']=nd.Pin('b1').put(100,-5,0)
    



# silicon nitride hybride escalator device
# 5. escalator
    #Nitride waveguide to silicon waveguide
with nd.Cell(name='nazca_cl_band_escalator_FN_SE') as cl_band_escalator_FN_SE:
    passive['cl_band_escalator_FN_SE'].put()
    passive['cl_band_escalator_FN_SE'].pin['a0']=nd.Pin('a0').put(0,0,180)
    passive['cl_band_escalator_FN_SE'].pin['b0']=nd.Pin('b0').put(40,0,0)
    #Nitride waveguide to nitride slot waveguide
with nd.Cell(name='nazca_cl_band_escalator_FN_FNSN') as cl_band_escalator_FN_FNSN:
    passive['cl_band_escalator_FN_FNSN'].put()
    passive['cl_band_escalator_FN_FNSN'].pin['a0']=nd.Pin('a0').put(0,0,180)
    passive['cl_band_escalator_FN_FNSN'].pin['b0']=nd.Pin('b0').put(40,0,0)




# Nitride layer device
# 2b. nitride coupler
    # grating coupler (vertical couple)
with nd.Cell(name='nazca_cl_band_vertical_coupler_FN') as cl_band_vertical_coupler_FN:
    passive['cl_band_vertical_coupler_FN'].put()  
    passive['cl_band_vertical_coupler_FN'].pin['b0']=nd.Pin('b0').put(0,0,-90)
    # edge coupler
    # Notice: for edge coupler, x=0 of the cell has to put at the edge of the design area
with nd.Cell(name='nazca_cl_band_edge_coupler_FN') as cl_band_edge_coupler_FN:
    passive['cl_band_edge_coupler_FN'].put()  
    passive['cl_band_edge_coupler_FN'].pin['a0']=nd.Pin('a0').put(0,0,180)
    passive['cl_band_edge_coupler_FN'].pin['b0']=nd.Pin('b0').put(300,0,0)


# 3b. nitride 4 port 50/50 splitter
with nd.Cell(name='nazca_cl_band_splitter_4port_si') as cl_band_splitter_4port_FN:
    passive['cl_band_splitter_4port_FN'].put()  
    passive['cl_band_splitter_4port_FN'].pin['b0']=nd.Pin('a0').put(0,5,180)
    passive['cl_band_splitter_4port_FN'].pin['a1']=nd.Pin('a1').put(0,-5,180)
    passive['cl_band_splitter_4port_FN'].pin['b0']=nd.Pin('b0').put(400,5,0)
    passive['cl_band_splitter_4port_FN'].pin['b1']=nd.Pin('b1').put(400,-5,0)


# 4b. nitride Y junction (first nitride layer)
with nd.Cell(name='nazca_cl_band_splitter_3port_FN') as cl_band_splitter_3port_FN:
    passive['cl_band_splitter_3port_FN'].put()
    passive['cl_band_splitter_3port_FN'].pin['a0']=nd.Pin('a0').put(0,0,180)
    passive['cl_band_splitter_3port_FN'].pin['b0']=nd.Pin('b0').put(200,5,0)
    passive['cl_band_splitter_3port_FN'].pin['b1']=nd.Pin('b1').put(200,-5,0)

#ADD WAVEGUIDES GDS
with nd.Cell(name='nazca_si_480nm_offset_30um') as si_480nm_offset_30um:
    waveguides['si_480nm_offset_30um'].put()
    waveguides['si_480nm_offset_30um'].pin['a0']=nd.Pin('a0').put(0,0,180)
    waveguides['si_480nm_offset_30um'].pin['b0']=nd.Pin('b0').put(50,-30,0)
    nd.put_stub()

# parameterized single mode waveguide for silicon and nitride
# 6a. silicon single mode waveguide
def cl_band_waveguide_si(length=30, turn = False, angle = 90,width=0.48, radius = None):
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
        ic = nd.interconnects.Interconnect(xs='xs_si', radius=radius, width=width)
        
        if turn:
            i1 = ic.bend(angle=angle, arrow=False).put()
        else:
            i1 = ic.strt(length=length, arrow=False).put()
        
        #add pin
        nd.Pin('a0', pin=i1.pin['a0']).put()
        nd.Pin('b0', pin=i1.pin['b0']).put()
    return C

# 6b. first layer nitride single mode waveguide
def cl_band_waveguide_FN(length=30, turn = False, angle = 90, radius = None):
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
    if radius == None:
        radius = 100
    
    with nd.Cell(name='nazca_cl_band_waveguide_FN') as C:
        nd.add_xsection('xs_FN')
        nd.add_layer2xsection(xsection='xs_FN', layer='FNAM')
        ic = nd.interconnects.Interconnect(xs='xs_FN', radius=radius, width=1.5)
        
        if turn:
            i1 = ic.bend(angle=angle, arrow=False).put()
        else:
            i1 = ic.strt(length=length, arrow=False).put()
        
        #add pin
        nd.Pin('a0', pin=i1.pin['a0']).put()
        nd.Pin('b0', pin=i1.pin['b0']).put()
    return C

# 6c. first layer nitride single mode waveguide
def cl_band_waveguide_FNSN(length=30, turn = False, angle = 90, radius=None):
    """Create a length parameterized silicon SM WG

    Args:
        length (float): length of the SM WG

    Returns:
        Cell: SM WG element
        
    Note: PDK specify radius should be larger than 100um
        We choose:
            radius = 40um
            width = 1.1um
    """
    
    if radius == None:
        radius = 40
        
    with nd.Cell(name='nazca_cl_band_waveguide_FNSN') as C:
        nd.add_xsection('xs_FNSN')
        nd.add_layer2xsection(xsection='xs_FNSN', layer='FNAM')
        nd.add_layer2xsection(xsection='xs_FNSN', layer='SNAM')
        ic = nd.interconnects.Interconnect(xs='xs1', radius=radius, width=1.1)
        
        if turn:
            i1 = ic.bend(angle=angle, arrow=False).put()
        else:
            i1 = ic.strt(length=length, arrow=False).put()
        
        #add pin
        nd.Pin('a0', pin=i1.pin['a0']).put()
        nd.Pin('b0', pin=i1.pin['b0']).put()
    return C

#Define taper length
def taper_length(start_width,end_width,wavelength):
    if start_width>end_width:
        start_width,end_width = end_width,start_width
    return (end_width**2 - start_width**2 ) / ( 2 * wavelength )

#Define horn taper
def horntaper_si(start_width,end_width,wavelength, n=500, name = None,xya = None):
    if name is None:
        name = 'cl_band_horntaper_si'

    if xya is None:
        xya = [0,0,0]

    with nd.Cell(name='{}'.format(name)) as taperBB:
        if start_width<end_width:
             small_width, large_width = start_width,end_width 
        else:
             small_width, large_width = end_width, start_width

        xpoints1 = np.linspace(-0.5 * large_width,-0.5 * small_width, n//2)
        xpoints2 = np.linspace(0.5 * small_width, 0.5 * large_width, n//2)
        xpoints = np.concatenate((xpoints1,xpoints2))

        ypoints = (4 * xpoints**2 - end_width**2 ) / ( 2 * wavelength )
        
        taper_points = list(map(lambda x, y:(x,y),xpoints,ypoints))
             
        nd.Polygon(points = taper_points,layer='SEAM').put(*xya) 
        
        if start_width<end_width:
            nd.Pin('a0').put(0,-np.abs(ypoints).max(),-90)
            nd.Pin('b0').put(0,0,90)
            
        else:
            nd.Pin('b0').put(0,0,-90)
            nd.Pin('a0').put(0,ypoints.max(),90)
            
        nd.put_stub()
    
    return taperBB


#cl_band_waveguide_FN(turn=True, angle=90).put()
#si_480nm_offset_30um.put()

#nd.export_gds(filename='test_component')
