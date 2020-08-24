# the Solgaard Lab PDK!
from .component import *


mems_monitor = MemsMonitorCoupler(waveguide_w=0.48, interaction_l=40, gap_w=0.1, end_l=5, bend_radius=5,
                                  detector_wg_l=30, pad_dim=(30, 20))

nems_miller = NemsMillerNode(waveguide_w=0.48, upper_interaction_l=30, lower_interaction_l=50,
                             gap_w=0.1, bend_radius=5, bend_extension=20, lr_nanofin_w=0.2,
                             ud_nanofin_w=0.2, lr_gap_w=0.5, ud_gap_w=0.3, lr_pad_dim=(10, 20),
                             ud_pad_dim=(50, 20), lr_connector_dim=(2, 0.1), ud_connector_dim=(0.1, 2))
