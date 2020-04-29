import numpy as np

EPS_0 = 8.85418782e-12           # vacuum permittivity
MU_0 = 1.25663706e-6             # vacuum permeability
C_0 = 1 / np.sqrt(EPS_0 * MU_0)  # speed of light in vacuum
ETA_0 = np.sqrt(MU_0 / EPS_0)    # vacuum impedance

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
    }
}


AIM_STACK = {
    'layers': {
        'si_ridge': [707, 727],  # ream
        'si': [709, 727],  # seam
        'sin_bot': [733, 727],  # fnam
        'sin_top': [735, 727],  # snam
        'sin_ox_open': [737, 727],  # tzam
        'trench': [726, 727]  # diam
    },
    'cross_sections': {
        'si_ridge_xs': [
            {
                'layer': [707, 727],  # rib etch
                'growy': 0.004
            }
        ],
        'waveguide_xs': [
            {
                'layer': [709, 727],  # waveguide
                'growy': 0.004
            }
        ],
        'sin_bot_xs': [
            {
                'layer': [733, 727],  # sin_bot
            }
        ],
        'sin_top_xs': [
            {
                'layer': [735, 727],  # sin_top
            }
        ],
        'trench_xs': [
            {
                'layer': [726, 727]  # trench
            }
        ]
    }
}

AIM_PDK_PASSIVE_PATH = '../../aim_lib/APSUNY_v35a_passive.gds'
AIM_PDK_WAVEGUIDE_PATH = '../../aim_lib/APSUNY_v35_waveguides.gds'
