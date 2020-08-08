from typing import List, Dict
from shapely.geometry import Polygon, LineString
from copy import deepcopy

from .materialblock import MaterialBlock
from .layerprofile import LayerProfile


class Fabricator:
    def __init__(self, name: str, substrate_block: MaterialBlock, base_blocks: List[MaterialBlock], span: float,
                 axis: int):
        self.name = name
        self.substrate_block = substrate_block
        self.substrate_layer = self.substrate_block.substrate_deposit(height=-substrate_block.dim[2] / 2, axis=axis)
        self.base_blocks = base_blocks
        self.span = span
        self.axis = axis
        self.current_layer_profile = LayerProfile(
            span=self.span,
            base_blocks=self.base_blocks,
            axis=axis
        )
        self.blocks: Dict[str, MaterialBlock] = {
            block.name: block for block in self.base_blocks
        }
        self.layers: Dict[str, Polygon] = {
            base_block.name: base_block.substrate_deposit(height=base_block.dim[2] / 2, axis=axis)
            for base_block in base_blocks
        }
        self.current_layers = [self.substrate_layer]
        self.current_layers.extend(self.layers.values())
        self.layer_to_profile: Dict[str, LineString] = {
            base_block.name: LineString([(-self.span / 2, 0), (self.span / 2, 0)])
            for base_block in base_blocks
        }

    def update_current_layers(self):
        self.current_layers = [self.substrate_layer]
        self.current_layers.extend(self.layers.values())

    def deposit_block(self, material_block: MaterialBlock, include_edges: bool = False):
        material_layer = material_block.deposit(self.current_layer_profile.path, self.current_layers, self.axis)
        self.layer_to_profile[material_block.name] = deepcopy(self.current_layer_profile.path)
        self.current_layer_profile.add_block(
            block=material_block,
            include_edges=include_edges
        )
        self.blocks[material_block.name] = material_block
        self.layers[material_block.name] = material_layer

    def directional_etch(self, etch_block: MaterialBlock, include_edges: bool = False):
        etch_shape = self.current_layer_profile.etch(etch_block, include_edges=include_edges)
        self.layers = {layer_name: self.layers[layer_name] - etch_shape for layer_name in self.layers}

    def fabricate(self, material_blocks: List[MaterialBlock]):
        prev_block = self.substrate_block
        for material_block in material_blocks:
            if material_block.material.name == 'Etch':
                self.directional_etch(etch_block=material_block,
                                      include_edges=(prev_block.dim[self.axis] >= self.span))
            else:
                self.deposit_block(material_block=material_block,
                                   include_edges=(prev_block.dim[self.axis] >= self.span) or material_block.full_length)
            prev_block = material_block
            self.update_current_layers()
