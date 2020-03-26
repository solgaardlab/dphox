import numpy as np

EPS_0 = 8.85418782e-12           # vacuum permittivity
MU_0 = 1.25663706e-6             # vacuum permeability
C_0 = 1 / np.sqrt(EPS_0 * MU_0)  # speed of light in vacuum
ETA_0 = np.sqrt(MU_0 / EPS_0)    # vacuum impedance

# nd.add_layer(name='Wvg',  layer=10, accuracy=0.001)
# nd.add_layer(name='Grat', layer=11, accuracy=0.001)
# nd.add_layer(name='Slab', layer=12, accuracy=0.001)
# nd.add_layer(name='PIM', layer=23, accuracy=0.001)
# nd.add_layer(name='NIM', layer=24, accuracy=0.001)
# nd.add_layer(name='Excl', layer=90, accuracy=0.001)
# nd.add_layer(name='Via1', layer=100, accuracy=0.001)
# nd.add_layer(name='Mt1', layer=105, accuracy=0.001)
# nd.add_layer(name='Htr', layer=115, accuracy=0.001)
# nd.add_layer(name='Via2', layer=120, accuracy=0.001)
# nd.add_layer(name='Mt2', layer=125, accuracy=0.001)
# nd.add_layer(name='Pad', layer=150, accuracy=0.001)
# nd.add_layer(name='DT', layer=160, accuracy=0.001)
# nd.add_layer(name='LBL', layer=80, accuracy=0.001)
# nd.add_layer(name='BND', layer=82, accuracy=0.001)
# nd.add_layer(name='DRCExcl', layer=1000, accuracy=0.001)