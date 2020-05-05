from simphox.design.aim import *
import nazca as nd
chip = AIMPhotonicChip(
    passive_filepath='/mnt/c/Users/nsabe/Nate/Research/Solgaard_lab/20200501_aim_run/aim_lib/APSUNY_v35a_passive.gds',
    waveguides_filepath='/mnt/c/Users/nsabe/Nate/Research/Solgaard_lab/20200501_aim_run/aim_lib/APSUNY_v35_waveguides.gds'
)


################ Nate's gds code ##########################
waveguide_w = 0.5
interport_w = 70
arm_l = 150
end_l = 50
tdc_interaction_w = 250
mzi_interation_w = 45
gap_w = 0.3
gap_w_id = 0.6
cp_radius = 35
trench_gap = 12

pad_l=200
pad_w=150-20

dc_kwargs = {
    'gap_w': gap_w,
    'interaction_l': tdc_interaction_w,
    'interport_w': interport_w,
    'end_l': end_l,
    'radius': cp_radius
}
mzi_kwargs = {
    'gap_w': gap_w,
    'interaction_l': mzi_interation_w,
    'interport_w': interport_w,
    'end_l': end_l,
    'arm_l': arm_l,
    'radius': cp_radius
}



def phase_shifter_base(l_ps):
    with nd.Cell('phase_shifter_base') as phase_shifter_base:
        w1=0.75
        w2=0
        offset1=0
        offset2=0
        length=l_ps+10
        l_gc_sep=5

        dc_l = chip.cl_band_splitter_4port_si.put(0, 0)
        #Phase Shifter

        if l_ps <= 1:
            upper_arm = chip.cl_band_waveguide_si(length=length).put(dc_l.pin['b0'])
        else:
            upper_arm = chip.static_ps_simple(w1=w1, w2=w2, offset1=offset1, offset2=offset2, length=l_ps,
                        length_taper=5).put(dc_l.pin['b0'])


        lower_arm = chip.cl_band_waveguide_si(length=length).put(dc_l.pin['b1'])
    
        dc_r = chip.cl_band_splitter_4port_si.put(upper_arm.pin['b0'])

        chip.cl_band_waveguide_si(angle=-90).put(dc_r.pin['b1'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()

        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        chip.cl_band_waveguide_si(angle=90).put(dc_r.pin['b0'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()

        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        chip.cl_band_waveguide_si(angle=-90).put(dc_l.pin['a0'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(),90)
        chip.cl_band_waveguide_si(angle=90).put(dc_l.pin['a1'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(),90) 
    return(phase_shifter_base)

def phase_shifter_opt(l_ps):
    with nd.Cell('phase_shifter_opt') as phase_shifter_opt:
        w1=0.828
        w2=1.406
        offset1=-.008
        offset2=0.539
        length=l_ps+10
        l_gc_sep=5

        dc_l = chip.cl_band_splitter_4port_si.put(0, 0)
        #Phase Shifter

        if l_ps <= 1:
            upper_arm = chip.cl_band_waveguide_si(length=length).put(dc_l.pin['b0'])
        else:
            upper_arm = chip.static_ps_simple(w1=w1, w2=w2, offset1=offset1, offset2=offset2, length=l_ps,
                        length_taper=5).put(dc_l.pin['b0'])
        


        lower_arm = chip.cl_band_waveguide_si(length=length).put(dc_l.pin['b1'])
    
        dc_r = chip.cl_band_splitter_4port_si.put(upper_arm.pin['b0'])

        chip.cl_band_waveguide_si(angle=-90).put(dc_r.pin['b1'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()

        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        chip.cl_band_waveguide_si(angle=90).put(dc_r.pin['b0'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()

        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        chip.cl_band_waveguide_si(angle=-90).put(dc_l.pin['a0'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(),90)
        chip.cl_band_waveguide_si(angle=90).put(dc_l.pin['a1'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(),90) 
    return phase_shifter_opt

def phase_shifter_3(l_ps):
    with nd.Cell('phase_shifter_3') as phase_shifter_3:
        offset1=0
        offset2=0
        length=l_ps+10
        l_gc_sep=5

        dc_l = chip.cl_band_splitter_4port_si.put(0, 0)
        #Phase Shifter
        if l_ps <= 1:
            upper_arm = chip.cl_band_waveguide_si(length=length).put(dc_l.pin['b0'])
        else:
            upper_arm = chip.static_ps_3(offset1=offset1, offset2=offset2, length=l_ps,
                        length_taper=5).put(dc_l.pin['b0'])


        lower_arm = chip.cl_band_waveguide_si(length=length).put(dc_l.pin['b1'])

        dc_r = chip.cl_band_splitter_4port_si.put(upper_arm.pin['b0'])

        chip.cl_band_waveguide_si(angle=-90).put(dc_r.pin['b1'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()

        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        chip.cl_band_waveguide_si(angle=90).put(dc_r.pin['b0'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()

        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        chip.cl_band_waveguide_si(angle=-90).put(dc_l.pin['a0'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(),90)
        chip.cl_band_waveguide_si(angle=90).put(dc_l.pin['a1'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(),90) 
    return phase_shifter_3

def phase_shifter_3_MEMS(l_ps):
    with nd.Cell('phase_shifter_3') as phase_shifter_3:
        offset1=0
        offset2=0
        length=l_ps+10
        l_gc_sep=0
        l_x_sep = 30

        dc_l = chip.cl_band_splitter_4port_si.put(0, 0)

        # Including bends to si crossings for contacts
        chip.cl_band_waveguide_si(angle=90).put(dc_l.pin['b0'])
        chip.cl_band_waveguide_si(length=l_x_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()
        ul_x=chip.cl_band_crossing.put()

        chip.cl_band_waveguide_si(angle=-90).put(dc_l.pin['b1'])
        chip.cl_band_waveguide_si(length=l_x_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()
        ll_x=chip.cl_band_crossing.put()


        #Phase Shifter
        if l_ps <= 1:
            upper_arm = chip.cl_band_waveguide_si(length=length).put(ul_x.pin['b0'])
            lower_arm = chip.cl_band_waveguide_si(length=length).put(ll_x.pin['b0'])
        else:
            upper_arm = chip.static_ps_3(offset1=offset1, offset2=offset2, length=l_ps,
                        length_taper=5).put(ul_x.pin['b0'])
            lower_arm = chip.static_ps_3(offset1=offset1, offset2=offset2, length=l_ps,
                        length_taper=5).put(ll_x.pin['b0'])

        # Crossings after Phase Shifters
        
        lr_x=chip.cl_band_crossing.put(lower_arm.pin['b0'])
        chip.cl_band_waveguide_si(angle=90).put(lr_x.pin['b0'])
        chip.cl_band_waveguide_si(length=l_x_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()

        # Add taps and gratings to taps
        tap_lower = chip.cl_band_1p_tap_si.put()

        ur_x=chip.cl_band_crossing.put(upper_arm.pin['b0'])
        chip.cl_band_waveguide_si(angle=-90).put(ur_x.pin['b0'])
        chip.cl_band_waveguide_si(length=l_x_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()
        
        tap_upper = chip.cl_band_1p_tap_si.put(flip=True)
        
        dc_r = chip.cl_band_splitter_4port_si.put()

        chip.cl_band_waveguide_si(angle=-90).put(dc_r.pin['b1'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()

        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        chip.cl_band_waveguide_si(angle=90).put(dc_r.pin['b0'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()

        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        chip.cl_band_waveguide_si(angle=-90).put(dc_l.pin['a0'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(),90)
        chip.cl_band_waveguide_si(angle=90).put(dc_l.pin['a1'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(),90) 


        # gratings from the taps

        chip.cl_band_waveguide_si(angle=90).put(tap_upper.pin['b1'])
        chip.cl_band_waveguide_si(length=25).put()
        chip.cl_band_waveguide_si(angle=-90).put()
        chip.cl_band_waveguide_si(length=200).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)
        chip.cl_band_waveguide_si(angle=-90).put(tap_lower.pin['b1'])
        chip.cl_band_waveguide_si(length=25).put()
        chip.cl_band_waveguide_si(angle=90).put()
        chip.cl_band_waveguide_si(length=200).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        # Adding contacts
        chip.si_contact_pad(length=l_ps+120,width=pad_w).put(ul_x.pin['a1'].x-5,ul_x.pin['a1'].y+pad_w/2)
        chip.si_contact_pad(length=l_ps+120,width=pad_w).put(ll_x.pin['b1'].x-5,ll_x.pin['b1'].y-pad_w/2)

    return phase_shifter_3

def phase_shifter_1_MEMS(l_ps):
    with nd.Cell('phase_shifter_1') as phase_shifter_1:
        offset1=0
        offset2=0
        length=l_ps+10
        l_gc_sep=0
        l_x_sep = 30

        dc_l = chip.cl_band_splitter_4port_si.put(0, 0)

        # Including bends to si crossings for contacts
        chip.cl_band_waveguide_si(angle=90).put(dc_l.pin['b0'])
        chip.cl_band_waveguide_si(length=l_x_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()
        ul_x=chip.cl_band_crossing.put()

        chip.cl_band_waveguide_si(angle=-90).put(dc_l.pin['b1'])
        chip.cl_band_waveguide_si(length=l_x_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()
        ll_x=chip.cl_band_crossing.put()


        #Phase Shifter
        if l_ps <= 1:
            upper_arm = chip.cl_band_waveguide_si(length=length).put(ul_x.pin['b0'])
            lower_arm = chip.cl_band_waveguide_si(length=length).put(ll_x.pin['b0'])
        else:
            upper_arm = chip.static_ps_simple(w1=0.75, w2=0, offset1=0, offset2=0, length=l_ps,
                        length_taper=5).put(ul_x.pin['b0'])
            lower_arm = chip.static_ps_simple(w1=0.75, w2=0, offset1=0, offset2=0, length=l_ps,
                        length_taper=5).put(ll_x.pin['b0'])

        # Crossings after Phase Shifters
        
        lr_x=chip.cl_band_crossing.put(lower_arm.pin['b0'])
        chip.cl_band_waveguide_si(angle=90).put(lr_x.pin['b0'])
        chip.cl_band_waveguide_si(length=l_x_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()

        # Add taps and gratings to taps
        tap_lower = chip.cl_band_1p_tap_si.put()
        

        # Crossing

        ur_x=chip.cl_band_crossing.put(upper_arm.pin['b0'])
        chip.cl_band_waveguide_si(angle=-90).put(ur_x.pin['b0'])
        chip.cl_band_waveguide_si(length=l_x_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()


        # Add taps and gratings to taps

        tap_upper = chip.cl_band_1p_tap_si.put(flip=True)
        
        
        dc_r = chip.cl_band_splitter_4port_si.put()

        chip.cl_band_waveguide_si(angle=-90).put(dc_r.pin['b1'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()

        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        chip.cl_band_waveguide_si(angle=90).put(dc_r.pin['b0'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()

        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        chip.cl_band_waveguide_si(angle=-90).put(dc_l.pin['a0'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(),90)
        chip.cl_band_waveguide_si(angle=90).put(dc_l.pin['a1'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(),90) 


        # gratings from the taps

        chip.cl_band_waveguide_si(angle=90).put(tap_upper.pin['b1'])
        chip.cl_band_waveguide_si(length=25).put()
        chip.cl_band_waveguide_si(angle=-90).put()
        chip.cl_band_waveguide_si(length=200).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)
        chip.cl_band_waveguide_si(angle=-90).put(tap_lower.pin['b1'])
        chip.cl_band_waveguide_si(length=25).put()
        chip.cl_band_waveguide_si(angle=90).put()
        chip.cl_band_waveguide_si(length=200).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        # Adding contacts
        chip.si_contact_pad(length=l_ps+120,width=pad_w).put(ul_x.pin['a1'].x-5,ul_x.pin['a1'].y+pad_w/2)
        chip.si_contact_pad(length=l_ps+120,width=pad_w).put(ll_x.pin['b1'].x-5,ll_x.pin['b1'].y-pad_w/2)

    return phase_shifter_1



def test1(l_ps):
    with nd.Cell('test1') as test1:
        offset1=0
        offset2=0
        length=l_ps+10
        l_gc_sep=0
        chip.cl_band_vertical_coupler_si.put(0,0,-90) 
        chip.cl_band_waveguide_si(length=length).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(),90) 
    return test1

def test2(l_ps):
    with nd.Cell('test2') as test2:
        w1=0.75
        w2=0
        offset1=0
        offset2=0
        length=l_ps+10
        l_gc_sep=5

        dc_l = chip.cl_band_splitter_4port_si.put(0, 0)
        # #Phase Shifter

        # if l_ps <= 1:
        #     upper_arm = chip.cl_band_waveguide_si(length=length).put(dc_l.pin['b0'])
        # else:
        #     upper_arm = chip.static_ps_simple(w1=w1, w2=w2, offset1=offset1, offset2=offset2, length=l_ps,
        #                 length_taper=5).put(dc_l.pin['b0'])


        # lower_arm = chip.cl_band_waveguide_si(length=length).put(dc_l.pin['b1'])
    
        # dc_r = chip.cl_band_splitter_4port_si.put(upper_arm.pin['b0'])

        chip.cl_band_waveguide_si(angle=-90).put(dc_l.pin['b1'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()

        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        chip.cl_band_waveguide_si(angle=90).put(dc_l.pin['b0'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()

        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

        chip.cl_band_waveguide_si(angle=-90).put(dc_l.pin['a0'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(),90)
        chip.cl_band_waveguide_si(angle=90).put(dc_l.pin['a1'])
        chip.cl_band_waveguide_si(length=l_gc_sep).put()
        chip.cl_band_waveguide_si(angle=-90).put()
        chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(),90) 
    return test2



### Trying out waveguide crossings

# first_x=chip.cl_band_crossing.put(0,-1000)
# chip.cl_band_waveguide_si(length=100).put()
# second_x=chip.cl_band_crossing.put()
# chip.cl_band_waveguide_si(length=100).put(first_x.pin['a1'])
# chip.cl_band_waveguide_si(length=100).put(second_x.pin['b1'])

### Changing the whole chip ###
ps_sep=360-20
l_ps=250-75
phase_shifter_3_MEMS(l_ps=l_ps).put(0,0)
# chip.shallow_trench(length=210, width=500).put()
phase_shifter_1_MEMS(l_ps=l_ps).put(0,ps_sep)


# test1(l_3[0]).put(250,(3.5+2*N)*chip_sep)
# phase_shifter_opt.put(0, 140)
# phase_shifter_3.put(0, 280)
nd.export_gds(filename='static.gds')



##################################################### Rebecca's GDS code ##############################################################

length_taper = 10
y_offset=195-20

### TDC #1
hat_width = 0.75
edge_width = 0.25
interport_w = 70
arm_l = 150
end_l = 200
end_l = 10
tdc_interaction_w = 60
gap_w = .52 #gapw = 2, gap between edges = 1.23, g= 0.77, get that brims are touching
cp_radius = 35
dc_kwargs = {
    'gap_w': gap_w,
    'interaction_l': tdc_interaction_w,
    'interport_w': interport_w,
    'end_l': end_l,
    'radius': cp_radius,
    'hat_width': hat_width,
    'edge_width': edge_width
}

### TDC #2
hat_width2 = 0.75
edge_width2 = 0.25
interport_w2 = 70
arm_l2 = 150
end_l2 = 200
end_l2 = 10
tdc_interaction_w2 = 55
gap_w2 = .52 #gapw = 2, gap between edges = 1.23, g= 0.77, get that brims are touching
cp_radius2 = 35
dc_kwargs2 = {
    'gap_w': gap_w2,
    'interaction_l': tdc_interaction_w2,
    'interport_w': interport_w2,
    'end_l': end_l2,
    'radius': cp_radius2,
    'hat_width': hat_width2,
    'edge_width': edge_width2
}


### 1st bridge for MEMS
bridge_w = 5
bridge_l = 75
tether_l = tdc_interaction_w
tether_w = 8
block_w = 0.48
block_l = tdc_interaction_w
ring_shape = False
mb_kwargs = {
    'bridge_w': bridge_w,
    'bridge_l': bridge_l,
    'tether_l': tether_l,
    'tether_w': tether_w,
    'block_w': block_w,
    'block_l': block_l,
    'ring_shape': ring_shape
}

### 2nd bridge for MEMS
bridge_w2 = 5
bridge_l2 = 75
tether_l2 = tdc_interaction_w2
tether_w2 = 8
block_w2 = 0.48
block_l2 = tdc_interaction_w2
ring_shape2 = False
mb_kwargs2 = {
    'bridge_w': bridge_w2,
    'bridge_l': bridge_l2,
    'tether_l': tether_l2,
    'tether_w': tether_w2,
    'block_w': block_w2,
    'block_l': block_l2,
    'ring_shape': ring_shape2
}

x = 125;
### TDC WITH MEMS #1
with nd.Cell('mems_tdc_rib_chiplet1') as mems_tdc_rib_chiplet1:
    dc_rib = chip.dc_rib(**dc_kwargs).put(0, 0)
    mtdc = chip.microbridge_pshack(**mb_kwargs)
    mtdc.put(dc_rib.pin['c1'], flip=True, flop=True)
    #add tapers
    taper_ina=geom.taper(length=length_taper,width1=0.75,width2=0.48)
    taper_outa=geom.taper(length=length_taper,width1=0.48,width2=0.75)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['a0'], dc_rib.pin['a0'].y)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['a1'], dc_rib.pin['a1'].y)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['b0'], dc_rib.pin['b0'].y, -90)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['b1'], dc_rib.pin['b1'].y, -90)
    #add gratings
    # Adding Si contacts 

    ul_x=chip.cl_band_crossing.put(dc_rib.pin['a1'].x - length_taper, dc_rib.pin['a1'].y,180)
    ur_x=chip.cl_band_crossing.put(dc_rib.pin['b1'].x + length_taper, dc_rib.pin['b1'].y)
    # ul_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['a1'].x - length_taper, dc_rib.pin['a1'].y,180)
    # ur_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['b1'].x + length_taper, dc_rib.pin['b1'].y)
    
    # ll_x=chip.cl_band_crossing.put(dc_rib.pin['a0'].x - length_taper, dc_rib.pin['a0'].y,180)
    # lr_x=chip.cl_band_crossing.put(dc_rib.pin['b0'].x + length_taper, dc_rib.pin['b0'].y)

    ll_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['a0'].x - length_taper, dc_rib.pin['a0'].y,180)
    lr_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['b0'].x + length_taper, dc_rib.pin['b0'].y)

    gratea = chip.cl_band_vertical_coupler_si.put(ll_x.pin['b0'].x, ll_x.pin['b0'].y , 90)
    grateb = chip.cl_band_vertical_coupler_si.put(ul_x.pin['b0'].x, ul_x.pin['b0'].y, 90)
    gratec = chip.cl_band_vertical_coupler_si.put(lr_x.pin['b0'].x, lr_x.pin['b0'].y, -90)
    grated = chip.cl_band_vertical_coupler_si.put(ur_x.pin['b0'].x, ur_x.pin['b0'].y, -90)

    # Adding contact Pads
    pad_l=abs(ul_x.pin['b1'].x-ur_x.pin['b1'].x)+10
    chip.si_contact_pad(length=pad_l,width=pad_w).put(ul_x.pin['b1'].x-5,ul_x.pin['b1'].y+pad_w/2)
    # chip.si_contact_pad(length=arm_l+120,width=pad_w).put(ll_x.pin['b1'].x-5,ll_x.pin['b1'].y-pad_w/2)

mems_tdc_rib_chiplet1.put(0, 0)

### TDC WITH MEMS #2
with nd.Cell('mems_tdc_rib_chiplet2') as mems_tdc_rib_chiplet2:
    dc_rib = chip.dc_rib(**dc_kwargs2).put(0, 0)
    mtdc = chip.microbridge_pshack(**mb_kwargs2)
    mtdc.put(dc_rib.pin['c1'], flip=True, flop=True)
    #add tapers and grating couplers
    taper_ina=geom.taper(length=length_taper,width1=0.75,width2=0.48)
    taper_outa=geom.taper(length=length_taper,width1=0.48,width2=0.75)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['a0'], dc_rib.pin['a0'].y)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['a1'], dc_rib.pin['a1'].y)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['b0'], dc_rib.pin['b0'].y, -90)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['b1'], dc_rib.pin['b1'].y, -90)
    
    # Adding Si contacts 

    # ul_x=chip.cl_band_crossing.put(dc_rib.pin['a1'].x - length_taper, dc_rib.pin['a1'].y,180)
    # ur_x=chip.cl_band_crossing.put(dc_rib.pin['b1'].x + length_taper, dc_rib.pin['b1'].y)
    ul_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['a1'].x - length_taper, dc_rib.pin['a1'].y,180)
    ur_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['b1'].x + length_taper, dc_rib.pin['b1'].y)
    
    ll_x=chip.cl_band_crossing.put(dc_rib.pin['a0'].x - length_taper, dc_rib.pin['a0'].y,180)
    lr_x=chip.cl_band_crossing.put(dc_rib.pin['b0'].x + length_taper, dc_rib.pin['b0'].y)

    gratea = chip.cl_band_vertical_coupler_si.put(ll_x.pin['b0'].x, ll_x.pin['b0'].y , 90)
    grateb = chip.cl_band_vertical_coupler_si.put(ul_x.pin['b0'].x, ul_x.pin['b0'].y, 90)
    gratec = chip.cl_band_vertical_coupler_si.put(lr_x.pin['b0'].x, lr_x.pin['b0'].y, -90)
    grated = chip.cl_band_vertical_coupler_si.put(ur_x.pin['b0'].x, ur_x.pin['b0'].y, -90)
mems_tdc_rib_chiplet2.put(0, x+y_offset)

### TDC #1 ONLY
with nd.Cell('tdc_rib_chiplet1') as tdc_rib_chiplet1:
    dc_rib = chip.dc_rib(**dc_kwargs).put(0, 0)
    #mtdc = chip.microbridge_pshack(**mb_kwargs)
    #mtdc.put(dc_rib.pin['c1'], flip=True, flop=True)
    #add grating couplers
    taper_ina=geom.taper(length=length_taper,width1=0.75,width2=0.48)
    taper_outa=geom.taper(length=length_taper,width1=0.48,width2=0.75)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['a0'], dc_rib.pin['a0'].y)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['a1'], dc_rib.pin['a1'].y)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['b0'], dc_rib.pin['b0'].y, -90)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['b1'], dc_rib.pin['b1'].y, -90)
    
    
    # ul_x=chip.cl_band_crossing.put(dc_rib.pin['a1'].x - length_taper, dc_rib.pin['a1'].y,180)
    # ur_x=chip.cl_band_crossing.put(dc_rib.pin['b1'].x + length_taper, dc_rib.pin['b1'].y)
    ul_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['a1'].x - length_taper, dc_rib.pin['a1'].y,180)
    ur_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['b1'].x + length_taper, dc_rib.pin['b1'].y)
    
    # ll_x=chip.cl_band_crossing.put(dc_rib.pin['a0'].x - length_taper, dc_rib.pin['a0'].y,180)
    # lr_x=chip.cl_band_crossing.put(dc_rib.pin['b0'].x + length_taper, dc_rib.pin['b0'].y)

    ll_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['a0'].x - length_taper, dc_rib.pin['a0'].y,180)
    lr_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['b0'].x + length_taper, dc_rib.pin['b0'].y)

    gratea = chip.cl_band_vertical_coupler_si.put(ll_x.pin['b0'].x, ll_x.pin['b0'].y , 90)
    grateb = chip.cl_band_vertical_coupler_si.put(ul_x.pin['b0'].x, ul_x.pin['b0'].y, 90)
    gratec = chip.cl_band_vertical_coupler_si.put(lr_x.pin['b0'].x, lr_x.pin['b0'].y, -90)
    grated = chip.cl_band_vertical_coupler_si.put(ur_x.pin['b0'].x, ur_x.pin['b0'].y, -90)
tdc_rib_chiplet1.put(0, 2*x+y_offset)


### TDC #2 ONLY
with nd.Cell('tdc_rib_chiplet2') as tdc_rib_chiplet2:
    dc_rib = chip.dc_rib(**dc_kwargs2).put(0, 0)
    #mtdc = chip.microbridge_pshack(**mb_kwargs)
    #mtdc.put(dc_rib.pin['c1'], flip=True, flop=True)
    #add grating couplers
    taper_ina=geom.taper(length=length_taper,width1=0.75,width2=0.48)
    taper_outa=geom.taper(length=length_taper,width1=0.48,width2=0.75)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['a0'], dc_rib.pin['a0'].y)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['a1'], dc_rib.pin['a1'].y)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['b0'], dc_rib.pin['b0'].y, -90)
    nd.Polygon(points=taper_ina, layer='SEAM').put(dc_rib.pin['b1'], dc_rib.pin['b1'].y, -90)
    
    # ul_x=chip.cl_band_crossing.put(dc_rib.pin['a1'].x - length_taper, dc_rib.pin['a1'].y,180)
    # ur_x=chip.cl_band_crossing.put(dc_rib.pin['b1'].x + length_taper, dc_rib.pin['b1'].y)
    ul_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['a1'].x - length_taper, dc_rib.pin['a1'].y,180)
    ur_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['b1'].x + length_taper, dc_rib.pin['b1'].y)
    
    # ll_x=chip.cl_band_crossing.put(dc_rib.pin['a0'].x - length_taper, dc_rib.pin['a0'].y,180)
    # lr_x=chip.cl_band_crossing.put(dc_rib.pin['b0'].x + length_taper, dc_rib.pin['b0'].y)

    ll_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['a0'].x - length_taper, dc_rib.pin['a0'].y,180)
    lr_x=chip.cl_band_waveguide_si(length=100).put(dc_rib.pin['b0'].x + length_taper, dc_rib.pin['b0'].y)

    gratea = chip.cl_band_vertical_coupler_si.put(ll_x.pin['b0'].x, ll_x.pin['b0'].y , 90)
    grateb = chip.cl_band_vertical_coupler_si.put(ul_x.pin['b0'].x, ul_x.pin['b0'].y, 90)
    gratec = chip.cl_band_vertical_coupler_si.put(lr_x.pin['b0'].x, lr_x.pin['b0'].y, -90)
    grated = chip.cl_band_vertical_coupler_si.put(ur_x.pin['b0'].x, ur_x.pin['b0'].y, -90)
tdc_rib_chiplet2.put(0, 3*x+y_offset)

nd.export_gds(filename = 'rib_dc_layout.gds')



####################################################### Sunil's gds code ###################################################

waveguide_w = 0.5
waveguide_w = 0.48 # making consistent with aim.py
interport_w = 70
arm_l = 150
arm_l = 210
end_l = 202
end_l_dc = 10
tdc_interaction_w = 100
mzi_interation_w = 45
gap_w = 0.3
gap_w_id = 0.6
cp_radius = 35
trench_gap = 12
l_x_sep = 30

dc_kwargs = {
    'gap_w': gap_w,
    'interaction_l': tdc_interaction_w,
    'interport_w': interport_w+40,
    'end_l': end_l_dc,
    'radius': cp_radius
}

mzi_kwargs = {
    'gap_w': gap_w,
    'interaction_l': mzi_interation_w,
    'interport_w': interport_w,
    'end_l': end_l - 120-80,
    'arm_l': arm_l+10,
    'radius': cp_radius
}

with nd.Cell('mems_phase_shifter_chiplet') as mems_phase_shifter_chiplet:

    ### Placing the topmost element ###
    # Left edge is zeroed now
    dc_l = chip.cl_band_splitter_4port_si.put(20+200, 1700)
    #Insert si crossing for contact
    chip.cl_band_waveguide_si(angle=90).put(dc_l.pin['b0'])
    chip.cl_band_waveguide_si(length=l_x_sep).put()
    chip.cl_band_waveguide_si(angle=-90).put()
    ul_x=chip.cl_band_crossing.put()


    upper_arm = chip.cl_band_waveguide_si(length=arm_l+20).put(ul_x.pin['b0'])
    #Insert si crossing for contact

    ur_x=chip.cl_band_crossing.put()
    chip.cl_band_waveguide_si(angle=-90).put(ur_x.pin['b0'])
    chip.cl_band_waveguide_si(length=l_x_sep).put()
    chip.cl_band_waveguide_si(angle=90).put()


    tap_upper = chip.cl_band_1p_tap_si.put(flip=True)
    #Insert si crossing for contact
    chip.cl_band_waveguide_si(angle=-90).put(dc_l.pin['b1'])
    chip.cl_band_waveguide_si(length=l_x_sep).put()
    chip.cl_band_waveguide_si(angle=90).put()
    ll_x=chip.cl_band_crossing.put()

    lower_arm = chip.cl_band_waveguide_si(length=arm_l+20).put(ll_x.pin['b0'])
    #Insert si crossing for contact
    lr_x=chip.cl_band_crossing.put()
    chip.cl_band_waveguide_si(angle=90).put(lr_x.pin['b0'])
    chip.cl_band_waveguide_si(length=l_x_sep).put()
    chip.cl_band_waveguide_si(angle=-90).put()
    
    tap_lower = chip.cl_band_1p_tap_si.put()


    chip.cl_band_waveguide_si(angle=90).put(tap_upper.pin['b1'])
    chip.cl_band_waveguide_si(length=25).put()
    chip.cl_band_waveguide_si(angle=-90).put()
    chip.cl_band_waveguide_si(length=200).put()
    chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)
    chip.cl_band_waveguide_si(angle=-90).put(tap_lower.pin['b1'])
    chip.cl_band_waveguide_si(length=25).put()
    chip.cl_band_waveguide_si(angle=90).put()
    chip.cl_band_waveguide_si(length=200).put()
    chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)
    dc_r = chip.cl_band_splitter_4port_si.put(tap_upper.pin['b0'])
    chip.cl_band_waveguide_si(angle=-90).put(dc_l.pin['a0'])
    chip.cl_band_waveguide_si(angle=90).put()
    chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), 90)
    chip.cl_band_waveguide_si(angle=90).put(dc_l.pin['a1'])
    chip.cl_band_waveguide_si(angle=-90).put()
    chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), 90)
    chip.cl_band_waveguide_si(angle=90).put(dc_r.pin['b0'])
    chip.cl_band_waveguide_si(angle=-90).put()
    chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)
    chip.cl_band_waveguide_si(angle=-90).put(dc_r.pin['b1'])
    chip.cl_band_waveguide_si(angle=90).put()
    chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)

    # Adding contacts
    chip.si_contact_pad(length=arm_l+10+120,width=pad_w).put(ul_x.pin['a1'].x-5,ul_x.pin['a1'].y+pad_w/2)
    chip.si_contact_pad(length=arm_l+10+120,width=pad_w).put(ll_x.pin['b1'].x-5,ll_x.pin['b1'].y-pad_w/2)



    ###TDC SUNIL ###

    ##left edge is zeroed
    dc = chip.dc(**dc_kwargs).put(100+200, 1200)
    #Insert Si X for contacts
    ul_x=chip.cl_band_crossing.put(dc.pin['a1'])
    ur_x=chip.cl_band_crossing.put(dc.pin['b1'])
    ll_x=chip.cl_band_crossing.put(dc.pin['a0'])
    lr_x=chip.cl_band_crossing.put(dc.pin['b0'])

    chip.cl_band_vertical_coupler_si.put(ul_x.pin['b0'].x, ul_x.pin['b0'].y, 90)
    chip.cl_band_vertical_coupler_si.put(ll_x.pin['b0'].x, ll_x.pin['b0'].y, 90)
    chip.cl_band_vertical_coupler_si.put(ur_x.pin['b0'].x, ur_x.pin['b0'].y, -90)
    chip.cl_band_vertical_coupler_si.put(lr_x.pin['b0'].x, lr_x.pin['b0'].y, -90)


    ### MZI SUNIL ###
    # mzi = chip.mzi(**mzi_kwargs).put(0, -120)
    # Left edge is zeroed now
    mzi = chip.mzi_x_contacts(**mzi_kwargs).put(200, 800)
    
    chip.cl_band_vertical_coupler_si.put(mzi.pin['a0'].x, mzi.pin['a0'].y, 90)
    chip.cl_band_vertical_coupler_si.put(mzi.pin['a1'].x, mzi.pin['a1'].y, 90)
    chip.cl_band_vertical_coupler_si.put(mzi.pin['b0'].x, mzi.pin['b0'].y, -90)
    chip.cl_band_vertical_coupler_si.put(mzi.pin['b1'].x, mzi.pin['b1'].y, -90)
    mps = chip.microbridge_ps(bridge_w=5, bridge_l=100,
                                tether_l=10, tether_w=5,
                                block_w=1, block_l=arm_l, radius=2)
    mps.put(mzi.pin['c1'].x+5, mzi.pin['c1'].y, flip=True)
    mps.put(mzi.pin['c0'].x+5, mzi.pin['c0'].y)

    mps.put(upper_arm.pin['a0'].x + 10, upper_arm.pin['a0'].y) # edit refs for this
    mps.put(lower_arm.pin['a0'].x + 10, lower_arm.pin['a0'].y, flip=True) #edit refs for this

    bridge_l = 100
    mtdc = chip.microbridge_ps(bridge_w=5, bridge_l=bridge_l,
                                tether_l=10, tether_w=5,
                                block_w=0.48, block_l=tdc_interaction_w, radius=1)
    mtdc.put(dc.pin['c1'], flip=True, flop=True)
    mtdc.put(dc.pin['c0'], flop=True)

    

    use_mps = [False, False, False]
    use_bus = [True, True, False]
    # use_bus = [False, False, False]
    use_crossing = [False, False, True]
    positions=[(200,0),(200,100),(200,200)]
    flip=[False, True, False]
    for idx, racetrack_l in enumerate((100, 500, 300)):
        if use_crossing[idx]: 
            interaction_l = 5
            radius = 30
            waveguide = chip.cl_band_waveguide_si(length=300).put(positions[idx][0],positions[idx][1])
            chip.cl_band_vertical_coupler_si.put(waveguide.pin['a0'].x, waveguide.pin['a0'].y, 90)
            rr = chip.ring_resonator_x(radius=radius, gap_w=0.2, racetrack_l=racetrack_l+200, interaction_l=interaction_l,
                                        interaction_angle=30).put(waveguide.pin['b0'])
            waveguide = chip.cl_band_waveguide_si(length=300).put(rr.pin['b0'])
            chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)
        else:
            interaction_l = 5
            radius = 30
            waveguide = chip.cl_band_waveguide_si(length=300).put(positions[idx][0],positions[idx][1])
            chip.cl_band_vertical_coupler_si.put(waveguide.pin['a0'].x, waveguide.pin['a0'].y, 90)
            rr = chip.ring_resonator(radius=radius, gap_w=0.2, racetrack_l=racetrack_l, interaction_l=interaction_l,
                                        interaction_angle=30).put(waveguide.pin['b0'])
            waveguide = chip.cl_band_waveguide_si(length=300).put(rr.pin['b0'])
            chip.cl_band_vertical_coupler_si.put(nd.cp.x(), nd.cp.y(), -90)
        if use_mps[idx]:
            # phase shifter
            mps.put(rr.pin['c0'].x + interaction_l / 2 - arm_l / 2,
                    rr.pin['c0'].y + waveguide_w + 2 * radius + gap_w / 2, flip=True)
        if use_bus[idx]:
            # bus
            gap_w=0.2
            chip.cl_band_waveguide_si(length=5).put(rr.pin['c0'].x + interaction_l / 2 + 0.35*racetrack_l,
                                                        rr.pin['c0'].y + 2 * waveguide_w + 2*gap_w)


    # shallow_trench = chip.shallow_trench(length=500, width=900)
    # shallow_trench2 = chip.shallow_trench(length=500, width=600)
    # shallow_trench.put(65, shallow_trench.bbox[3] / 2 + 100)
    # shallow_trench2.put(35, -shallow_trench.bbox[3] / 2 - 480)

    # tdc = nd.load_gds('tdc_v2.gds')  # insert rebecca's filepath (or scripted cell) here
    tdc= nd.load_gds('rib_dc_layout.gds')
    tdc.put(0, -1550, flip=True)
    static = nd.load_gds('static.gds')  # insert nate's filepath (or scripted cell) here
    static.put(220, -300, flip=True)
    

mems_phase_shifter_chiplet.put(0+1500, 0)
## Drawing the bounding box
dx=2150
dy=1850
chip.cl_band_waveguide_si(angle=90, radius=5).put(-dx/2,-dy/2+5,-90)
chip.cl_band_waveguide_si(length=dx-10).put()
chip.cl_band_waveguide_si(angle=90,radius=5).put()
chip.cl_band_waveguide_si(length=dy-10).put()
chip.cl_band_waveguide_si(angle=90, radius=5).put()
chip.cl_band_waveguide_si(length=dx-10).put()
chip.cl_band_waveguide_si(angle=90,radius=5).put()
chip.cl_band_waveguide_si(length=dy-10).put()

nd.export_gds(filename='mems_phase_shifter_chiplet.gds')