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