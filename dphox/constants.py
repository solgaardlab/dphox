import numpy as np

AMF_STACK = {
    'layers': {
        'waveguide': 10,
        'grating': 11,
        'via': 100,
        'via_heater': 120,
        'mt_heater': 125,
        'heater': 115,
        'slab': 12,
        'pad': 150,
        'trench': 160
    },
    'cross_sections': {
        'heater_xs': [
            {
                'layer': 115,  # heater
                'growx': 0.755,  # (waveguide_w - heater_w) / 2 + 0.005
                'growy': 0.005
            },
            {
                'layer': 10,  # waveguide
                'growy': 0.001
            }
        ],
        'metal_xs': [
            {
                'layer': 125,  # mt_heater
            }
        ],
        'via_heater_xs': [
            {
                'layer': 120  # via_heater
            },
            {
                'layer': 125,  # mt_heater
                'growx': 1.5,
                'growy': 1.5
            },
            {
                'layer': 115,  # heater
                'growx': 1.5,
                'growy': 1.5
            }
        ],
        'grating_xs': [
            {
                'layer': 11  # grating
            }
        ],
        'waveguide_xs': [
            {
                'layer': 10,  # waveguide
                'growy': 0.004
            }
        ],
        'pad_xs': [
            {
                'layer': 125  # mt_heater
            },
            {
                'layer': 150,  # pad
                'growx': -2,
                'growy': -2
            }
        ],
        'trench_xs': [
            {
                'layer': 160  # trench
            }
        ],
        'slab_xs': [
            {
                'layer': 12  # slab
            }
        ]
    },
}


AIM_STACK = {
    'layers': {
        'ream': (702, 727),  # ream, ridge etch
        'seam': (709, 727),  # seam, silicon etch
        'fnam': (733, 727),  # fnam, nitride waveguide
        'snam': (735, 727),  # snam, nitride waveguide
        'ndam': (791, 727),  # ndam, n implant
        'nnam': (792, 727),  # nnam, nn implant
        'nnnam': (793, 727),  # nnnam, nnn implant
        'pdam': (794, 727),  # pdam, p implant
        'ppam': (795, 727),  # ppam, pp implant
        'pppam': (796, 727),  # pppam, ppp implant
        'tram': (718, 727),  # tram, detector trench
        'ngam': (776, 727),  # ngam, n-type ion
        'esam': (720, 727),  # esam, etch nitride etch stop
        'caam': (721, 727),  # detector contact
        'cbam': (722, 727),  # contact to Si Level
        'm1am': (710, 727),  # metal 1 contact to caam/cbam
        'v1am': (715, 727),  # via to m1am
        'm2am': (725, 727),  # metal 2 level
        'vaam': (771, 727),  # aluminum via to m2am
        'tzam': (737, 727),  # tzam
        'diam': (726, 727),  # diam, dicing channel
        'paam': (779, 727),  # metal passivation
        'mlam': (780, 727)
    },
    'cross_sections': {
        'v1_xs': [
            {
                'layer': 'm2am'
            },
            {
                'layer': 'v1am',
                'growx': -0.05,
                'growy': -0.05
            },
            {
                'layer': 'm1am'
            },
        ],
        'pad_xs': [
            {
                'layer': 'm2am'
            },
            {
                'layer': 'mlam'  # aluminum pads
            },
            {
                'layer': 'vaam',  # via to aluminum layer
                'growx': -2,
                'growy': -2
            }
        ],
        'waveguide_xs': [
            {
                'layer': 'seam'
            }
        ],
        'dice_xs': [
            {
                'layer': 'diam'
            }
        ],
        'm1_xs': [
            {
                'layer': 'm1am'
            }
        ],
        'm2_xs': [
            {
                'layer': 'm2am'
            }
        ],
        'ml_xs': [
            {
                'layer': 'mlam'
            }
        ],
        'va_xs': [
            {
                'layer': 'mlam'
            },
            {
                'layer': 'vaam',
                'growx': -0.75,
                'growy': -0.75
            },
            {
                'layer': 'm2am'
            }
        ]
    }
}

AIM_PDK = {
    'cl_band_1p_tap_si': {
        'a1': (0, 5, 180),
        'a0': (0, -5, 180),
        'b1': (40, 5, 0),
        'b0': (40, -5, 0)
    },
    'cl_band_vertical_coupler_si': {
        'b0': (0, 0, -90),
    },
    'cl_band_splitter_4port_si': {
        'a1': (0, 5, 180),
        'a0': (0, -5, 180),
        'b1': (200, 5, 0),
        'b0': (200, -5, 0)
    },
    'cl_band_thermo_optic_switch': {
        'a1': (0, 5, 180),
        'a0': (0, -5, 180),
        'b1': (550, 5, 0),
        'b0': (550, -5, 0),
        'p': (272.3, 62.5, 0),
        'n': (277.3, 62.5, 0)
    },
    'cl_band_photodetector_analog': {
        'a0': (0, 0, 180),
        'p': (169.15, 26, -90),
        'n': (173.55, 26, -90)
    },
    'cl_band_photodetector_digital': {
        'a0': (0, 0, 180),
        'p': (16, 15, 0),
        'n': (16, 5, 0)
    },
    'cl_band_thermo_optic_phase_shifter': {
        'a0': (0, 0, 180),
        'b0': (0, 100, 0),
        'p': (47.3, 12.5, 90),
        'n': (52.3, 12.5, 90),
    }
}


AIM_PDK_PASSIVE_PATH = '../../aim_lib/APSUNY_v35a_passive.design'
AIM_PDK_WAVEGUIDE_PATH = '../../aim_lib/APSUNY_v35_waveguides.design'
AIM_PDK_ACTIVE_PATH = '../../aim_lib/APSUNY_v35_actives.design'
