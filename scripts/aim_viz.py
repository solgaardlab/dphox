from dphox.aim import *
from dphox.constants import AIM_STACK

# specify the component you want to visualize in 3d here:
miller_node.show(
    layer_to_zrange=AIM_STACK['zranges'],
    process_extrusion=AIM_STACK['process_extrusion'],
    layer_to_color=AIM_STACK['layer_to_color'],
    # ignore_layers=['snam'],
    engine='blender'
)
