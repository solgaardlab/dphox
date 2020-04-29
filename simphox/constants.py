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
        'si': 1,  # ream
        'sin_1': 2,
        'sin_2': 3
    },
    'cross_sections': {
        'si_xs': [
            {
                'layer': 10,  # waveguide
                'growy': 0.004
            }
        ],
        'sin_xs': [
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
