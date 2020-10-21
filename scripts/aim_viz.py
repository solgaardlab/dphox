from dphox.aim import *
from dphox.constants import AIM_STACK

# specify the component you want to visualize in 3d here:
# miller_node.show(
#     layer_to_zrange=AIM_STACK['zranges'],
#     process_extrusion=AIM_STACK['process_extrusion'],
#     layer_to_color=AIM_STACK['layer_to_color'],
#     # ignore_layers=['snam'],
#     engine='blender'
# )

# pull_in_full_ps.show(
#     layer_to_zrange=AIM_STACK['zranges'],
#     process_extrusion=AIM_STACK['process_extrusion'],
#     layer_to_color=AIM_STACK['layer_to_color'],
#     # ignore_layers=['snam'],
#     engine='blender'
# )

# pull_apart_full_ps.show(
#     layer_to_zrange=AIM_STACK['zranges'],
#     process_extrusion=AIM_STACK['process_extrusion'],
#     layer_to_color=AIM_STACK['layer_to_color'],
#     # ignore_layers=['snam'],
#     engine='blender'
# )

pull_in_full_tdc.show(
    layer_to_zrange=AIM_STACK['zranges'],
    process_extrusion=AIM_STACK['process_extrusion'],
    layer_to_color=AIM_STACK['layer_to_color'],
    # ignore_layers=['snam'],
    engine='blender'
)
#
# tether_full_tdc.show(
#     layer_to_zrange=AIM_STACK['zranges'],
#     process_extrusion=AIM_STACK['process_extrusion'],
#     layer_to_color=AIM_STACK['layer_to_color'],
#     # ignore_layers=['snam'],
#     engine='blender'
# )

# tether_full_ps.show(
#     layer_to_zrange=AIM_STACK['zranges'],
#     process_extrusion=AIM_STACK['process_extrusion'],
#     layer_to_color=AIM_STACK['layer_to_color'],
#     # ignore_layers=['snam'],
#     engine='blender'
# )

# make the stl files
# miller_node.to_stls(
#     prefix='miller_node',
#     layer_to_zrange=AIM_STACK['zranges'],
#     process_extrusion=AIM_STACK['process_extrusion'],
#     layer_to_color=AIM_STACK['layer_to_color'],
#     layers=['oxide', 'clearout', 'seam', 'ream'],
#     engine='blender'
# )
