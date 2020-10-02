from .pattern import Path, Pattern, Port
from .passive import Box
from ...typing import *

from collections import defaultdict

import numpy as np
import gdspy as gy
import nazca as nd
from shapely.geometry import Polygon, MultiPolygon
from descartes import PolygonPatch
import trimesh
from trimesh import creation, visual
from copy import deepcopy as copy

try:
    import plotly.graph_objects as go
except ImportError:
    pass


class Multilayer:
    def __init__(self, pattern_to_layer: List[Tuple[Union[Pattern, Path, gy.Polygon, gy.FlexPath, Polygon],
                                                    Union[int, str]]]):
        self.pattern_to_layer = pattern_to_layer
        self._pattern_to_layer = {comp: layer if isinstance(comp, Pattern) else Pattern(comp)
                                  for comp, layer in pattern_to_layer}
        self.layer_to_pattern = self._layer_to_pattern()
        # TODO: temporary way to assign ports
        self.port = dict(sum([list(pattern.port.items()) for pattern, _ in pattern_to_layer], []))

    @classmethod
    def from_nazca_cell(cls, cell: nd.Cell):
        # a glimpse into cell_iter()
        # code from https://nazca-design.org/forums/topic/clipping-check-distance-and-length-for-interconnects/
        multilayers = defaultdict(list)
        for named_tuple in nd.cell_iter(cell, flat=True):
            if named_tuple.cell_start:
                for i, (polygon, points, bbox) in enumerate(named_tuple.iters['polygon']):
                    if polygon.layer == 'bb_pin':
                        continue
                    # fixing point definitions from mask to 1nm prcesision,
                    # kinda hacky but is physical and prevents false polygons
                    points = np.around(points, decimals=3)
                    multilayers[polygon.layer].append(Pattern(Polygon(points)))
        return cls([(Pattern(*pattern_list), layer) for layer, pattern_list in multilayers.items()])

    @property
    def bounds(self) -> Dim4:
        bbox = self.gdspy_cell().get_bounding_box()
        return bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]

    def gdspy_cell(self, cell_name: str = 'dummy') -> gy.Cell:
        cell = gy.Cell(cell_name, exclude_from_current=(cell_name == 'dummy'))
        for pattern, layer in self._pattern_to_layer.items():
            for poly in pattern.polys:
                cell.add(gy.Polygon(np.asarray(poly.exterior.coords.xy).T, layer=layer))
        return cell

    def nazca_cell(self, cell_name: str) -> nd.Cell:
        with nd.Cell(cell_name) as cell:
            for pattern, layer in self._pattern_to_layer.items():
                for poly in pattern.polys:
                    nd.Polygon(points=np.asarray(poly.exterior.coords.xy).T, layer=layer).put()
            for name, port in self.port.items():
                nd.Pin(name).put(*port.xya_nazca)
            nd.put_stub()
        return cell

    def _layer_to_pattern(self) -> Dict[Union[int, str], MultiPolygon]:
        layer_to_polys = defaultdict(list)
        for component, layer in self._pattern_to_layer.items():
            layer_to_polys[layer].extend(component.polys)
        pattern_dict = {layer: MultiPolygon(polys) for layer, polys in layer_to_polys.items()}
        return pattern_dict

    def plot(self, ax, layer_to_color: Dict[Union[int, str], Union[Dim3, str]], alpha: float = 0.5):
        for layer, pattern in self.layer_to_pattern.items():
            ax.add_patch(PolygonPatch(pattern, facecolor=layer_to_color[layer], edgecolor='none', alpha=alpha))
        b = self.bounds
        ax.set_xlim((b[0], b[2]))
        ax.set_ylim((b[1], b[3]))
        ax.set_aspect('equal')

    def to_trimesh_dict(self, layer_to_zrange: Dict[str, Tuple[float, float]],
                        process_extrusion: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
                        layer_to_color: Optional[Dict[str, str]] = None, engine: str = 'scad'):
        meshes = {}
        if process_extrusion is not None:
            layer_to_extrusion = self.build_layers(layer_to_zrange, process_extrusion)
            for layer, pattern_zrange in layer_to_extrusion.items():
                try:
                    zmin, zmax = pattern_zrange[1]
                    layer_meshes = [
                        trimesh.creation.extrude_polygon(poly, height=zmax - zmin).apply_translation((0, 0, zmin))
                        for poly in pattern_zrange[0]]
                    mesh = trimesh.Trimesh().union(layer_meshes, engine=engine)
                    mesh.visual.vertex_colors = visual.random_color() \
                        if layer_to_color is None else layer_to_color[layer]
                    meshes[layer] = mesh
                except KeyError:
                    print(f"No zranges given for the layer {layer}")
            return meshes
        else:
            for layer, pattern in self.layer_to_pattern.items():
                try:
                    zmin, zmax = layer_to_zrange[layer]
                    layer_meshes = [
                        trimesh.creation.extrude_polygon(poly, height=zmax - zmin).apply_translation((0, 0, zmin))
                        for poly in pattern]
                    mesh = trimesh.Trimesh().union(layer_meshes, engine=engine)
                    mesh.visual.vertex_colors = visual.random_color() if layer_to_color is None else layer_to_color[layer]
                    meshes[layer] = mesh
                except KeyError:
                    print(f"No zranges given for the layer {layer}")
            return meshes

    def to_trimesh_scene(self, layer_to_zrange: Dict[str, Tuple[float, float]],
                         process_extrusion: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
                         layer_to_color: Optional[Dict[str, str]] = None, engine: str = 'scad'):
        meshes = self.to_trimesh_dict(layer_to_zrange, process_extrusion, layer_to_color, engine)
        return trimesh.Scene(meshes.values())

    def build_layers(self, layer_to_zrange: Dict[str, Tuple[float, float]],
                     process_extrusion: Dict[str, List[Tuple[str, str, str]]]):
        layer_to_extrusion = {}
        layers = self.layer_to_pattern.keys()
        layer_to_pattern_processed = self.layer_to_pattern.copy()
        for step, operations in process_extrusion.items():
            for layer_relation in operations:
                layer, other_layer, operation = layer_relation
                if 'DOPE' in step and operation == 'intersection':
                    # make a new layer for each doping intersection
                    new_layer = layer + '_' + other_layer
                    zmin, zmax = layer_to_zrange[other_layer]
                    z0, z1 = layer_to_zrange[layer]
                    # TODO(): how to deal with different depth doping currently not addressed
                    new_zrange = (max(zmax - (z1 - z0), zmin), zmax)
                else:
                    new_layer = layer
                    new_zrange = layer_to_zrange[layer]
                if layer in layers:
                    if other_layer in layers:
                        pattern = Pattern(layer_to_pattern_processed[layer]).boolean_operation(
                            Pattern(layer_to_pattern_processed[other_layer]), operation
                        ).shapely
                    else:
                        pattern = layer_to_pattern_processed[layer]
                    if pattern.geoms:
                        layer_to_pattern_processed[new_layer] = pattern
                        layer_to_extrusion[new_layer] = (pattern, new_zrange)
        return layer_to_extrusion

    def fill_material(self, layer_name: str, growth: float, centered_layer: str = None):
        all_patterns = [Pattern(poly) for layer, poly in self.layer_to_pattern.items()]
        all_patterns = Pattern(*all_patterns)
        minx, miny, maxx, maxy = all_patterns.bounds
        centered_pattern = all_patterns if centered_layer is None else Pattern(self.layer_to_pattern[centered_layer])
        fill = Pattern(gy.Polygon(
            [(minx - growth / 2, miny - growth / 2), (minx - growth / 2, maxy + growth / 2),
             (maxx + growth / 2, maxy + growth / 2), (maxx + growth / 2, miny - growth / 2)]
        )).align(centered_pattern)
        self.pattern_to_layer.append((fill, layer_name))
        self._pattern_to_layer = {comp: layer if isinstance(comp, Pattern) else Pattern(comp)
                                  for comp, layer in self.pattern_to_layer}
        self.layer_to_pattern = self._layer_to_pattern()
        return [(fill, layer_name)]


class Via(Multilayer):
    def __init__(self, via_dim: Dim2, boundary_grow: float, top_metal: str, bot_metal: str, via: str,
                 pitch: float = 0, shape: Optional[Shape2] = None):
        self.via_dim = via_dim
        self.boundary_grow = boundary_grow
        self.top_metal = top_metal
        self.bot_metal = bot_metal
        self.via = via
        self.pitch = pitch
        self.shape = shape
        self.config = self.__dict__

        via_pattern = Box(via_dim)
        if pitch > 0 and shape is not None:
            patterns = []
            x, y = np.meshgrid(np.arange(shape[0]) * pitch, np.arange(shape[1]) * pitch)
            for x, y in zip(x.flatten(), y.flatten()):
                patterns.append(copy(via_pattern).translate(x, y))
            via_pattern = Pattern(*patterns)
        boundary = Box((via_pattern.size[0] + 2 * boundary_grow,
                        via_pattern.size[1] + 2 * boundary_grow)).align((0, 0)).halign(0)
        via_pattern.align(boundary)
        layers = [(via_pattern, via)]
        layers += [(boundary, top_metal)] if top_metal is not None else []
        layers += [(copy(boundary), bot_metal)] if bot_metal is not None else []
        super(Via, self).__init__(layers)
        self.port['a0'] = Port(self.bounds[0], 0, np.pi)
        self.port['b0'] = Port(self.bounds[2], 0)
