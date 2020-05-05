from simphox.design.aim import *
import nazca as nd
chip = AIMPhotonicChip(
    passive_filepath='/mnt/c/Users/nsabe/Nate/Research/Solgaard_lab/20200501_aim_run/aim_lib/APSUNY_v35a_passive.gds',
    waveguides_filepath='/mnt/c/Users/nsabe/Nate/Research/Solgaard_lab/20200501_aim_run/aim_lib/APSUNY_v35_waveguides.gds'
)
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

### Pararmeters and placing on the layout ###

N=6
phis=np.linspace(0,2*np.pi,N)
amps=np.linspace(-1,1,N)

l_2p_0=130 #get from sims
l_2p_1=128
l_2p_3=266

chip_sep=130

l_bases=l_2p_0*np.arccos(amps)/(2*np.pi)
l_opt=l_2p_1*np.arccos(amps)/(2*np.pi)
l_3=l_2p_3*np.arccos(amps)/(2*np.pi)

i1=0
# for l in l_bases:
#     phase_shifter_base(l_ps=l).put(0, i1*chip_sep)
#     i1+=1

i2=i1
for l in l_opt:
    phase_shifter_opt(l_ps=l).put(0, i2*chip_sep)
    i2+=1
i3=i2
for l in l_3:
    phase_shifter_3(l_ps=l).put(0, i3*chip_sep)
    i3+=1

### Grating test structures
test1(l_bases[0]).put(250,3.5*chip_sep)
test1(l_opt[0]).put(250,(3.5+N)*chip_sep)
test2(10).put(0,i3*chip_sep)

# test1(l_3[0]).put(250,(3.5+2*N)*chip_sep)
# phase_shifter_opt.put(0, 140)
# phase_shifter_3.put(0, 280)
nd.export_gds(filename='phase_shifter_chiplet.gds')



start_width_d = 1.25
end_width_d = 0.75
wavelength = 1.55
waveguide_w = 0.75

start_widthxd = 0.75
end_widthxd = 0.48


with nd.Cell('rib_taper') as rib_taper:
    taper = chip.horntaper_si_ream(start_width_d, end_width_d, wavelength)
    taper_cell = taper.put(0,0)
    hat = chip.cl_band_waveguide_si_thick(length = 0.323).put(taper_cell.pin['b0'], flop = True)
    taper1 = chip.horntaper_si(start_widthxd, end_widthxd, wavelength)
    taper1_cell = taper1.put(taper_cell.pin['b0'])
    #thin = chip.cl_band_waveguide_si(length=1).put(taper1_cell.pin['b0'])
#rib_taper.put(0,0)
#rib_taper.put(5, 0, flop = True)
#nd.export_plt()

#coupler
waveguide= 0.75
interport_w = 70
arm_l = 150
end_l = 200
tdc_interaction_w = 60
gap_w = 0.5
cp_radius = 35
dc_kwargs = {
    'gap_w': gap_w,
    'interaction_l': tdc_interaction_w,
    'interport_w': interport_w,
    'end_l': end_l,
    'radius': cp_radius,
    'waveguide': waveguide
}


interport_w1 = 70
arm_l1 = 150
end_l1 = 200
tdc_interaction_w1 = 50
gap_w1 = 0.5
cp_radius1 = 35
dc_kwargs1 = {
    'gap_w': gap_w1,
    'interaction_l': tdc_interaction_w1,
    'interport_w': interport_w1,
    'end_l': end_l1,
    'radius': cp_radius1,
    'waveguide': waveguide
}
#bridge1
bridge_w = 5
bridge_l = 75 
tether_l = 15
tether_w = 5
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

#bridge2
bridge_w2 = 5
bridge_l2 = 75 
tether_l2 = 25
tether_w2 = 10
block_w2 = 0.8
block_l2 = tdc_interaction_w
ring_shape = False
mb_kwargs2 = {
    'bridge_w': bridge_w2,
    'bridge_l': bridge_l2,
    'tether_l': tether_l2,
    'tether_w': tether_w2,
    'block_w': block_w2,
    'block_l': block_l2,
    'ring_shape': ring_shape
}

#bridge3
bridge_w3 = 5
bridge_l3 = 85 
tether_l3 = 25
tether_w3 = 10
block_w3 = 1
block_l3 = tdc_interaction_w
ring_shape = False
mb_kwargs3 = {
    'bridge_w': bridge_w3,
    'bridge_l': bridge_l3,
    'tether_l': tether_l3,
    'tether_w': tether_w3,
    'block_w': block_w3,
    'block_l': block_l3,
    'ring_shape': ring_shape
}



#sunil bridge
bridge_wps = 5
bridge_lps = 75 
tether_lps = 15
tether_wps = 5
block_wps = 0.48
block_lps = tdc_interaction_w
ring_shapeps = True
mb_kwargsps = {
    'bridge_w': bridge_wps,
    'bridge_l': bridge_lps,
    'tether_l': tether_lps,
    'tether_w': tether_wps,
    'block_w': block_wps,
    'block_l': block_lps,
    'ring_shape': ring_shapeps
}

x = 125; 
with nd.Cell('2mems_tdc_rib_chiplet') as mems_tdc_rib_chiplet:
    dc_rib = chip.dc_rib(**dc_kwargs).put(0, 0)
    mtdc = chip.microbridge_ps(**mb_kwargs)
    mtdc.put(dc_rib.pin['c1'], flip=True, flop=True)
    mtdc.put(dc_rib.pin['c0'], flop=True)
    taperc = rib_taper.put(dc_rib.pin['b0'])
    taperd = rib_taper.put(dc_rib.pin['b1'])
    tapera = rib_taper.put(dc_rib.pin['a0'], flip = True)
    taperb = rib_taper.put(dc_rib.pin['a1'], flip = True)
    #add grating couplers 
    gratea = chip.cl_band_vertical_coupler_si.put(tapera.pin['b0'].x - 0.43, tapera.pin['b0'].y, 90)
    grateb = chip.cl_band_vertical_coupler_si.put(taperb.pin['b0'].x -0.43, taperb.pin['b0'].y, 90)
    gratec = chip.cl_band_vertical_coupler_si.put(taperc.pin['a0'].x + 0.43, taperc.pin['a0'].y, -90)
    grated = chip.cl_band_vertical_coupler_si.put(taperd.pin['a0'].x + 0.43, taperd.pin['a0'].y, -90)
mems_tdc_rib_chiplet.put(0, 0)

with nd.Cell('1mems_tdc_rib_chiplet') as mems_tdc_rib_chiplet:
    dc_rib = chip.dc_rib(**dc_kwargs).put(0, 0)
    mtdc = chip.microbridge_ps(**mb_kwargs)
    #mtdc.put(dc_rib.pin['c1'], flip=True, flop=True)
    mtdc.put(dc_rib.pin['c0'], flop=True)
    taperc = rib_taper.put(dc_rib.pin['b0'])
    taperd = rib_taper.put(dc_rib.pin['b1'])
    tapera = rib_taper.put(dc_rib.pin['a0'], flip = True)
    taperb = rib_taper.put(dc_rib.pin['a1'], flip = True)
    #add grating couplers 
    gratea = chip.cl_band_vertical_coupler_si.put(tapera.pin['b0'].x - 0.43, tapera.pin['b0'].y, 90)
    grateb = chip.cl_band_vertical_coupler_si.put(taperb.pin['b0'].x -0.43, taperb.pin['b0'].y, 90)
    gratec = chip.cl_band_vertical_coupler_si.put(taperc.pin['a0'].x + 0.43, taperc.pin['a0'].y, -90)
    grated = chip.cl_band_vertical_coupler_si.put(taperd.pin['a0'].x + 0.43, taperd.pin['a0'].y, -90)
mems_tdc_rib_chiplet.put(0, x)

with nd.Cell('1mems_tdc_rib_chiplet2') as mems_tdc_rib_chiplet2:
    dc_rib = chip.dc_rib(**dc_kwargs).put(0, 0)
    mtdc = chip.microbridge_ps(**mb_kwargs2)
    mtdc.put(dc_rib.pin['c1'], flip=True, flop=True)
    #mtdc.put(dc_rib.pin['c0'], flop=True)
    taperc = rib_taper.put(dc_rib.pin['b0'])
    taperd = rib_taper.put(dc_rib.pin['b1'])
    tapera = rib_taper.put(dc_rib.pin['a0'], flip = True)
    taperb = rib_taper.put(dc_rib.pin['a1'], flip = True)
    #add grating couplers 
    gratea = chip.cl_band_vertical_coupler_si.put(tapera.pin['b0'].x - 0.43, tapera.pin['b0'].y, 90)
    grateb = chip.cl_band_vertical_coupler_si.put(taperb.pin['b0'].x -0.43, taperb.pin['b0'].y, 90)
    gratec = chip.cl_band_vertical_coupler_si.put(taperc.pin['a0'].x + 0.43, taperc.pin['a0'].y, -90)
    grated = chip.cl_band_vertical_coupler_si.put(taperd.pin['a0'].x + 0.43, taperd.pin['a0'].y, -90)
mems_tdc_rib_chiplet2.put(0, 2*x)

with nd.Cell('1mems_tdc_rib_chiplet3') as mems_tdc_rib_chiplet3:
    dc_rib = chip.dc_rib(**dc_kwargs).put(0, 0)
    mtdc = chip.microbridge_ps(**mb_kwargs3)
    mtdc.put(dc_rib.pin['c1'], flip=True, flop=True)
    #mtdc.put(dc_rib.pin['c0'], flop=True)
    taperc = rib_taper.put(dc_rib.pin['b0'])
    taperd = rib_taper.put(dc_rib.pin['b1'])
    tapera = rib_taper.put(dc_rib.pin['a0'], flip = True)
    taperb = rib_taper.put(dc_rib.pin['a1'], flip = True)
    #add grating couplers 
    gratea = chip.cl_band_vertical_coupler_si.put(tapera.pin['b0'].x - 0.43, tapera.pin['b0'].y, 90)
    grateb = chip.cl_band_vertical_coupler_si.put(taperb.pin['b0'].x -0.43, taperb.pin['b0'].y, 90)
    gratec = chip.cl_band_vertical_coupler_si.put(taperc.pin['a0'].x + 0.43, taperc.pin['a0'].y, -90)
    grated = chip.cl_band_vertical_coupler_si.put(taperd.pin['a0'].x + 0.43, taperd.pin['a0'].y, -90)
mems_tdc_rib_chiplet2.put(0, 3*x)

with nd.Cell('2memsps_tdc_rib_chiplet') as memsps_tdc_rib_chiplet:
    dc_rib = chip.dc_rib(**dc_kwargs).put(0, 0)
    mtdc = chip.microbridge_ps(**mb_kwargsps)
    mtdc.put(dc_rib.pin['c1'], flip=True, flop=True)
    mtdc.put(dc_rib.pin['c0'], flop=True)
    taperc = rib_taper.put(dc_rib.pin['b0'])
    taperd = rib_taper.put(dc_rib.pin['b1'])
    tapera = rib_taper.put(dc_rib.pin['a0'], flip = True)
    taperb = rib_taper.put(dc_rib.pin['a1'], flip = True)
    #add grating couplers 
    gratea = chip.cl_band_vertical_coupler_si.put(tapera.pin['b0'].x - 0.43, tapera.pin['b0'].y, 90)
    grateb = chip.cl_band_vertical_coupler_si.put(taperb.pin['b0'].x -0.43, taperb.pin['b0'].y, 90)
    gratec = chip.cl_band_vertical_coupler_si.put(taperc.pin['a0'].x + 0.43, taperc.pin['a0'].y, -90)
    grated = chip.cl_band_vertical_coupler_si.put(taperd.pin['a0'].x + 0.43, taperd.pin['a0'].y, -90)
memsps_tdc_rib_chiplet.put(0, -x)


with nd.Cell('tdc_rib_chiplet') as tdc_rib_chiplet:
    dc_rib = chip.dc_rib(**dc_kwargs).put(0, 0)
    #tapers
    taperc = rib_taper.put(dc_rib.pin['b0'])
    taperd = rib_taper.put(dc_rib.pin['b1'])
    tapera = rib_taper.put(dc_rib.pin['a0'], flip = True)
    taperb = rib_taper.put(dc_rib.pin['a1'], flip = True)
    #add grating couplers 
    gratea = chip.cl_band_vertical_coupler_si.put(tapera.pin['b0'].x - 0.43, tapera.pin['b0'].y, 90)
    grateb = chip.cl_band_vertical_coupler_si.put(taperb.pin['b0'].x -0.43, taperb.pin['b0'].y, 90)
    gratec = chip.cl_band_vertical_coupler_si.put(taperc.pin['a0'].x + 0.43, taperc.pin['a0'].y, -90)
    grated = chip.cl_band_vertical_coupler_si.put(taperd.pin['a0'].x + 0.43, taperd.pin['a0'].y, -90)
tdc_rib_chiplet.put(0,-2*x)


with nd.Cell('tdc_rib_chiplet1') as tdc_rib_chiplet1:
    dc_rib = chip.dc_rib(**dc_kwargs1).put(0, 0)
    #tapers
    taperc = rib_taper.put(dc_rib.pin['b0'])
    taperd = rib_taper.put(dc_rib.pin['b1'])
    tapera = rib_taper.put(dc_rib.pin['a0'], flip = True)
    taperb = rib_taper.put(dc_rib.pin['a1'], flip = True)
    #add grating couplers 
    gratea = chip.cl_band_vertical_coupler_si.put(tapera.pin['b0'].x - 0.43, tapera.pin['b0'].y, 90)
    grateb = chip.cl_band_vertical_coupler_si.put(taperb.pin['b0'].x -0.43, taperb.pin['b0'].y, 90)
    gratec = chip.cl_band_vertical_coupler_si.put(taperc.pin['a0'].x + 0.43, taperc.pin['a0'].y, -90)
    grated = chip.cl_band_vertical_coupler_si.put(taperd.pin['a0'].x + 0.43, taperd.pin['a0'].y, -90)
tdc_rib_chiplet1.put(0,-3*x)



nd.export_gds(filename='mems_tdc_rib_test_xxx.gds')