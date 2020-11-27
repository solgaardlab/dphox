try:
    import meep as mp
    from meep import mpb
    MEEP_IMPORTED = True
except ImportError:
    MEEP_IMPORTED = False
    pass


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
                'layer': 115,    # heater
                'growx': 0.755,  # (waveguide_w - heater_w) / 2 + 0.005
                'growy': 0.005
            },
            {
                'layer': 10,   # waveguide
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
                'layer': 120   # via_heater
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
                'layer': 11    # grating
            }
        ],
        'waveguide_xs': [
            {
                'layer': 10,   # waveguide
                'growy': 0.004
            }
        ],
        'pad_xs': [
            {
                'layer': 125   # mt_heater
            },
            {
                'layer': 150,  # pad
                'growx': -2,
                'growy': -2
            }
        ],
        'trench_xs': [
            {
                'layer': 160   # trench
            }
        ],
        'slab_xs': [
            {
                'layer': 12    # slab
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
        'mlam': (780, 727),  # aluminum metal layer
        'oxide': (998, 1),  # oxide fill
        'clearout': (999, 1),  # pseudo clearout for nems devices
    },
    'drc': {  # design rules for AIM PDKv3.5
        # format int (layer * 100 + dr_index): (rule str, special str, relation, float)
        # ream
        70201: [('width', '', '>=', 0.5)],
        70202: [('length', '', '>=', 2)],
        70211: [('width', 'space on wafer', '>=', 0.5)],
        70203: [('space', '', '>=', 0.2)],
        70204: [('area', '', '>=', 2)],
        70205: [('space', 'to tapered waveguide', '>=', 0.13)],
        70206: [('outside', 'seam', '>=', 0.25)],
        # seam
        70901: [('width', '', '>=', 0.15)],
        70902: [('length', '', '>=', 0.6)],
        70903: [('space', '', '>=', 0.1)],
        70905: [('area', '', '>=', 0.09)],
        70906: [('ring', '', '<=', 0.05)],
        70907: [('outside', 'diam', '>=', 6)],
        # fnam
        73301: [('width', '', '>=', 0.15)],
        73302: [('length', '', '>=', 0.15)],
        73303: [('space', '', '>=', 0.1)],
        73304: [('ring', '', '<=', 0.05)],
        73305: [('space', 'pdam', '>=', 0.15),
                ('space', 'ppam', '>=', 0.15),
                ('space', 'pppam', '>=', 0.15),
                ('space', 'ngam', '>=', 0.15),
                ('space', 'ndam', '>=', 0.15),
                ('space', 'nnam', '>=', 0.15),
                ('space', 'nnnam', '>=', 0.15)],
        # snam
        73501: [('width', '', '>=', 0.15)],
        73502: [('length', '', '>=', 0.15)],
        73503: [('space', '', '>=', 0.1)],
        73504: [('ring', '', '<=', 0.05)],
        # ndam
        79101: [('width', '', '>=', 0.5)],
        79102: [('length', '', '>=', 0.8)],
        79103: [('space', '', '>=', 0.5)],
        79104: [('area', '', '>=', 0.4)],
        79108: [('foreshortening', '', 'approx', 0.1)],
        79109: [('cornerrounding', '', 'approx', 0.1)],
        79110: [('outside', 'seam', 'approx', 0.1)],
        # nnam
        79201: [('width', '', '>=', 0.5)],
        79202: [('length', '', '>=', 0.8)],
        79203: [('space', '', '>=', 0.5)],
        79204: [('area', '', '>=', 0.4)],
        79208: [('foreshortening', '', 'approx', 0.1)],
        79209: [('cornerrounding', '', 'approx', 0.1)],
        79210: [('outside', 'seam', 'approx', 0.1)],
        # nnnam
        79301: [('width', '', '>=', 0.5)],
        79302: [('length', '', '>=', 2)],
        79303: [('space', '', '>=', 0.5)],
        79304: [('area', '', '>=', 0.4)],
        79308: [('foreshortening', '', 'approx', 0.1)],
        79309: [('cornerrounding', '', 'approx', 0.1)],
        79310: [('outside', 'seam', 'approx', 0.1)],
        # pdam
        79401: [('width', '', '>=', 0.5)],
        79402: [('length', '', '>=', 0.8)],
        79403: [('space', '', '>=', 0.5)],
        79404: [('area', '', '>=', 0.4)],
        79408: [('foreshortening', '', 'approx', 0.1)],
        79409: [('cornerrounding', '', 'approx', 0.1)],
        79410: [('outside', 'seam', 'approx', 0.1)],
        # ppam
        79501: [('width', '', '>=', 0.5)],
        79502: [('length', '', '>=', 0.8)],
        79503: [('space', '', '>=', 0.5)],
        79504: [('area', '', '>=', 0.4)],
        79508: [('foreshortening', '', 'approx', 0.1)],
        79509: [('cornerrounding', '', 'approx', 0.1)],
        79510: [('outside', 'seam', 'approx', 0.1)],
        # pppam
        79601: [('width', '', '>=', 0.5)],
        79602: [('length', '', '>=', 2)],
        79603: [('space', '', '>=', 0.5)],
        79604: [('area', '', '>=', 0.4)],
        79608: [('foreshortening', '', 'approx', 0.1)],
        79609: [('cornerrounding', '', 'approx', 0.1)],
        79610: [('outside', 'seam', 'approx', 0.1)],
        # tram
        71801: [('width', '', '==', 4)],
        71802: [('length', '', '>=', 8)],
        71811: [('length', '', '<=', 25)],
        71803: [('space', '', '>=', 0.5)],
        71804: [('area', '', '>=', 32)],
        71805: [('within', 'seam', '>=', 0.25)],
        71806: [('density', 'per 100um tile', '<=', 0.02)],
        # ngam
        77601: [('width', '', '>=', 0.9)],
        77602: [('length', '', '>=', 6)],
        77603: [('space', '', '>=', 0.5)],
        77604: [('area', '', '>=', 5.4)],
        77605: [('outside', 'caam', '>=', 0.25)],
        # esam
        72001: [('width', '', '>=', 4.7)],
        72002: [('length', '', '>=', 8.7)],
        72003: [('space', '', '>=', 1.5)],
        72004: [('area', '', '>=', 40.9)],
        72005: [('outside', 'tram', '>=', 0.35)],
        # cabar
        72104: [('within', 'tram, length', '>=', 1.8)],
        72114: [('within', 'ngam', '>=', 0.25)],
        72105: [('area', 'diam', '<=', 0)],
        72106: [('width', '', '==', 0.4),
                ('length', '', '<=', 22.5),
                ('length', '', '>=', 5.5)],
        72107: [('within', 'tram, width', '>=', 1.25)],
        72109: [('space', '', '>=', 0.8)],
        72110: [('within', 'm1am', '>=', 0.06)],
        72111: [('space', 'cbam', '>=', 2.5)],
        # cbam
        72201: [('width', '', '==', 0.4)],  # slightly different from actual rule
        72203: [('space', '', '>=', 0.4)],
        72213: [('outside', 'tram', '>=', 0.7)],
        72204: [('within', 'seam', '>=', 0.1)],
        72205: [('outside', 'diam', '>=', 0)],
        72214: [('within', 'm1am', '>=', 0.04)],
        72215: [('outside', 'esam', '>=', 0.35)],
        72216: [('outside', 'ream', '>=', 0.1)],
        72217: [('over', 'seam', '>=', 0)],
        # m1am
        71001: [('width', '', '>=', 0.48)],
        71002: [('length', '', '>=', 0.48)],
        71003: [('space', '', '>=', 0.48)],
        71004: [('area', '', '>=', 0.2304)],
        71006: [('outside', 'caam', '>=', 0.04),
                ('outside', 'cbam', '>=', 0.04)],
        71008: [('space', 'diam', '>=', 5)],
        # v1am
        71501: [('width', '', '==', 0.4)],
        71502: [('length', '', '==', 0.4)],
        71503: [('space', '', '>=', 0.4)],
        # m2am
        72501: [('width', '', '>=', 0.48)],
        72502: [('length', '', '>=', 3.8)],
        72503: [('space', '', '>=', 0.48)],
        72504: [('area', '', '>=', 1.824)],
        72505: [('outside', 'v1am', '>=', 0.1)],
        72508: [('space', 'diam', '>=', 5)],
        # vaam
        77101: [('width', '', '==', 3.6)],
        77102: [('length', '', '>=', 3.6)],
        77103: [('space', '', '>=', 2.4)],
        77105: [('within', 'mlam', '>=', 1.5)],
        77106: [('within', 'm2am', '>=', 1)],
        77107: [('space', 'v1am', '>=', 6)],
        # mlam
        78001: [('width', '', '>=', 1)],
        78002: [('length', '', '>=', 4)],
        78003: [('within', 'mlam', '>=', 1.5)],
        78005: [('area', '', '>=', 4)],
        # diam
        72601: [('width', '', '>=', 100)],
        72602: [('length', '', '>=', 1000)],
        72603: [('space', '', '>=', 1000)],
        # paam
        77901: [('width', '', '==', 40)],
        77902: [('length', '', '==', 40)],
        77903: [('space', '', '>=', 60)],
        77904: [('area', '', '>=', 1600)],
        77905: [('within', 'm2am', '>=', 10)],
        # wgkoam
        80201: [('outside', 'ream', '>=', 0)],
        80202: [('outside', 'seam', '>=', 0)],
        80203: [('outside', 'ndam', '>=', 0)],
        80204: [('outside', 'nnam', '>=', 0)],
        80205: [('outside', 'nnnam', '>=', 0)],
        80206: [('outside', 'pdam', '>=', 0)],
        80207: [('outside', 'ppam', '>=', 0)],
        80208: [('outside', 'pppam', '>=', 0)],
        80209: [('outside', 'snam', '>=', 0)],
        80210: [('outside', 'ppam', '>=', 0)],
        80211: [('outside', 'pppam', '>=', 0)],
        # metkoam
        80301: [('outside', 'caam', '>=', 0)],
        80302: [('outside', 'cbam', '>=', 0)],
        80303: [('outside', 'm1am', '>=', 0)],
        80304: [('outside', 'm2am', '>=', 0)],
        80305: [('outside', 'mlam', '>=', 0)],
        80306: [('outside', 'vaam', '>=', 0)],
        80307: [('outside', 'v1am', '>=', 0)]
    },
    'zranges': {
        # 'ream': (0.11, 0.22),  # ream, ridge etch
        'ream': (0.00, 0.11),  # ream, si ridge remaining
        'seam': (0.00, 0.22),  # seam, silicon
        'fnam': (0.32, 0.54),  # fnam, nitride waveguide
        'snam': (0.64, 0.86),  # snam, nitride waveguide
        'ndam': (0.00, 0.11),  # ndam, n implant #this is a depth estimate
        'nnam': (0.00, 0.11),  # nnam, nn implant #this is a depth estimate
        'nnnam': (0.00, 0.22),  # nnnam, nnn implant #this is a depth estimate
        'pdam': (0.11, 0.22),  # pdam, p implant #this is a depth estimate
        'ppam': (0.11, 0.22),  # ppam, pp implant #this is a depth estimate
        'pppam': (0.11, 0.22),  # pppam, ppp implant #this is a depth estimate
        'tram': (0.22, 1.00),  # tram, detector trench # guessing
        'ngam': (0.78, 1.00),  # ngam, n-type ion # guessing
        'esam': (1.00, 1.100),  # esam, etch nitride etch stop # guessing
        'caam': (0.93, 1.135),  # detector contact # guessing
        'cbam': (0.22, 1.135),  # contact to Si Level # guessing
        'm1am': (1.135, 1.24),  # metal 1 contact to caam/cbam # guessing
        'v1am': (1.24, 1.74),  # via to m1am # guessing
        'm2am': (1.74, 1.83),  # metal 2 level # guessing
        'vaam': (1.83, 2.23),  # aluminum via to m2am # guessing
        'tzam': (0.64, 2.00),  # tzam # guessing
        'diam': (-2.00, 2.00),  # diam, dicing channel # guessing
        'paam': (2.23, 2.33),  # metal passivation # guessing
        'mlam': (2.23, 2.33),  # mteal pad layer # guessing
        'oxide': (-2.00, 2.50),  # oxide fill
        'clearout': (-2.00, 2.50),  # pseudo clearout for nems devices
    },
    'process_extrusion': {
        # ream, si ridge remaining
        'etch_re': [('ream', 'seam', 'intersection'), ('ream', 'diam', 'difference')],
        'etch_se': [('seam', 'ream', 'difference'), ('seam', 'diam', 'difference')],
        'dope_nd': [('ndam', 'seam', 'intersection'), ('ndam', 'ream', 'intersection')],
        'dope_nn': [('nnam', 'seam', 'intersection'), ('nnam', 'ream', 'intersection')],
        'dope_nnn': [('nnnam', 'seam', 'intersection'), ('nnnam', 'ream', 'intersection')],
        'dope_pd': [('pdam', 'seam', 'intersection'), ('pdam', 'ream', 'intersection')],
        'dope_pp': [('ppam', 'seam', 'intersection'), ('ppam', 'ream', 'intersection')],
        'dope_ppp': [('pppam', 'seam', 'intersection'), ('pppam', 'ream', 'intersection')],
        # fnam, nitride waveguide
        'etch_fn': [('fnam', 'diam', 'difference')],
        # snam, nitride waveguide
        'etch_sn': [('snam', 'tzam', 'difference'), ('snam', 'diam', 'difference'), ('snam', 'clearout', 'difference')],
        # tram, detector trench # guessing
        'etch_tr': [('tram', 'diam', 'difference')],
        # ngam, n-type ion # guessing
        'dope_ng': [('ngam', 'tram', 'intersection'), ('ngam', 'diam', 'difference')],
        # esam, etch nitride etch stop # guessing
        'etch_es': [('esam', 'caam', 'difference'), ('esam', 'diam', 'difference')],
        # detector contact # guessing
        'via_ca': [('caam', 'diam', 'difference')],
        # contact to Si Level # guessing
        'via_cb': [('cbam', 'diam', 'difference')],
        # metal 1 contact to caam/cbam # guessing
        'metal_m1': [('m1am', 'diam', 'difference')],
        # via to m1am # guessing
        'via_v1': [('v1am', 'diam', 'difference')],
        # metal 2 level # guessing
        'metal_m2': [('m2am', 'diam', 'difference')],
        # aluminum via to m2am # guessing
        'via_va': [('vaam', 'diam', 'difference')],
        # mteal pad layer # guessing
        'metal_ml': [('mlam', 'diam', 'difference')],
        # metal passivation # guessing
        'fill_pa': [('paam', 'diam', 'difference')],
        # oxide fill
        'etch_ox': [('oxide', 'clearout', 'difference'), ('oxide', 'tzam', 'difference'), ('oxide', 'diam', 'difference')],
        # air in case it's needed for
        'etch_air': [('clearout', 'diam', 'difference')],
    },
    'layer_to_color': {
        'seam': (0.5, 0.5, 0.5, 1),
        'ream': (0.5, 0.5, 0.5, 1),
        'oxide': (0.8, 0, 0, 0.5),
        'cbam': (0.9, 0.5, 0.0, 1),
        'v1am': (0.9, 0.5, 0.0, 1),
        'vaam': (0.9, 0.5, 0.0, 1),
        'm1am': (0.1, 0.5, 0.3, 1),
        'm2am': (0.1, 0.2, 0.6, 1),
        'mlam': (0.0, 0.7, 0.8, 1),
        'pdam': (0.4, 0.2, 0.6, 0.5),
        'ppam': (0.4, 0.2, 0.6, 0.7),
        'pppam': (0.4, 0.2, 0.6, 0.9),
        'ndam': (1.0, 0.3, 0.3, 0.5),
        'nnam': (1.0, 0.3, 0.3, 0.7),
        'nnnam': (1.0, 0.3, 0.3, 0.9),
        'fnam': (0.8, 0, 0, 0.5),
        'snam': (0.8, 0.5, 0, 0.5),
        'clearout': (0.7, 0.7, 0.2, 0.5),
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
        'p': (47.3, 12.499, 90),
        'n': (52.3, 12.499, 90),
    }
}


AIM_PDK_PASSIVE_PATH = '../../aim_lib/APSUNY_v35a_passive.design'
AIM_PDK_WAVEGUIDE_PATH = '../../aim_lib/APSUNY_v35_waveguides.design'
AIM_PDK_ACTIVE_PATH = '../../aim_lib/APSUNY_v35_actives.design'


def get_sinx280_stack(gap, mechanical_thickness, metal_thickness):
    sinx280_STACK = {
        'layers': {  # gotta makeup a mapping scheme (material, 0 = real layer / 1 = dummy layer)
            'handle': (99, 0),  # si handle that is probabaly only useful for generating full sims
            'box': (98, 0),  # bottom oxide, proabably most useful for grating simulations
            'nit1': (1, 0),  # nitride waveguide layer
            'tox': (2, 0),  # top oxide etch mask (data clear), most useful for gratings
            # TODO: Add internal process check/warning for no etch stop
            'asi1': (3, 0),  # etch of sacricificial layer of amorphous (data clear)
            # TODO:ADD layers
            'nit2': (4, 0),  # nitride mechanical layer
            'metal1': (5, 0),  # nitride mechanical layer
        },


        'zranges': {
            'handle': (-3, -2.16),  # si handle that is probabaly only useful for generating full sims
            'box': (-2.16, 0),  # bottom oxide, proabably most useful for grating simulations
            'nit1': (0, 0.28),  # nitride waveguide layer
            'tox': (0.28, 0.97),  # top oxide, 690nm thick 0.97=0.69+0.28
            'asi1': (0.97, 0.97 + gap),  # sacricificial layer of amorphous si
            'nit2': (0.97 + gap, 0.97 + gap + mechanical_thickness),  # nitride mechanical layer
            'metal1': (0.97 + gap + mechanical_thickness, 0.97 + gap + mechanical_thickness + metal_thickness),  # top metal layer
            'dice': (-3, 3)  # dice lines
        },

        'meep': {
            # 'handle': mp.Medium(index=3.47),  # si handle that is probabaly only useful for generating full sims
            'box': mp.Medium(index=1.45),  # bottom oxide, proabably most useful for grating simulations
            'nit1': mp.Medium(index=1.95),  # nitride waveguide layer
            'tox': mp.Medium(index=1.45),  # top oxide, 690nm thick 0.97=0.69+0.28
            # 'asi1': mp.Medium(index=3.47),
            'nit2': mp.Medium(index=1.95),  # nitride mechanical layer
        },


        'process_extrusion': {
            'directional_shape_box': [('box', 'dice', 'difference')],
            'conformal_shape_nit1': [('nit1', 'dice', 'difference')],
            'conformal_etch_tox': [('tox', 'dice', 'difference')],
            'conformal_shape_asi1': [('asi1', 'dice', 'difference')],
            'conformal_shape_nit2': [('nit2', 'dice', 'difference')],
            'directional_shape_metal1': [('metal1', 'dice', 'difference')],
        },

        'cross_sections': {
            'waveguide_xs': [
                {
                    'layer': 'nit1'
                }
            ],
            'ml_xs': [
                {
                    'layer': 'metal1'
                }
            ],

        }
    }
    return sinx280_STACK


GCMESHMSDLA_PDK = {
    'dc_exp': {
        'a0': (-7600, -1800, 180)
    },
    'mmi_exp': {
        'a0': (-7600, -1800, 180)
    },
    'gc_exp': {
        'a0': (-7600, -1800, 180)
    },
    'mmi_mzi': {
        'a0': (-7600, -1800, 180)
    }
}
