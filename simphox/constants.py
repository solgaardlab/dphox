import numpy as np

EPS_0 = 8.85418782e-12           # vacuum permittivity
MU_0 = 1.25663706e-6             # vacuum permeability
C_0 = 1 / np.sqrt(EPS_0 * MU_0)  # speed of light in vacuum
ETA_0 = np.sqrt(MU_0 / EPS_0)    # vacuum impedance
