from collections import defaultdict

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

from ...typing import *
from .pattern import Pattern, GroupedPattern, Path
from .passive import Box


class Multilayer:
    def __init__(self, pattern_to_layer: List[Tuple[Union[Pattern, Path, gy.Polygon, gy.FlexPath, Polygon], Union[int, str]]]):
        self.pattern_to_layer = pattern_to_layer
        self._pattern_to_layer = {comp: layer if isinstance(comp, Pattern) else Pattern(comp)
                                  for comp, layer in pattern_to_layer}
        self.layer_to_pattern = self._layer_to_pattern()
        self.port = dict(sum([list(pattern.port.items()) for pattern, _ in pattern_to_layer], []))

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

    def to_trimesh(self, layer_to_zrange: Dict[str, Tuple[float, float]],
                   layer_to_color: Optional[Dict[str, str]] = None, engine: str = 'scad'):
        meshes = []
        for layer, zrange in layer_to_zrange.items():
            zmin, zmax = zrange
            layer_meshes = [
                trimesh.creation.extrude_polygon(poly, height=zmax - zmin).apply_translation((0, 0, zmin))
                for poly in self.layer_to_pattern[layer]]
            mesh = trimesh.Trimesh().union(layer_meshes, engine=engine)
            mesh.visual.vertex_colors = visual.random_color() if layer_to_color is None else layer_to_color[layer]
            meshes.append(mesh)
        return trimesh.Scene(meshes)

    # TODO(Nate): change  above to_tirmesh_scene and create a function for the zrange enumeration
    def to_trimesh_dict(self, layer_to_zrange: Dict[str, Tuple[float, float]],
                   layer_to_color: Optional[Dict[str, str]] = None, engine: str = 'scad'):
        meshes = {}
        for layer, zrange in layer_to_zrange.items():
            zmin, zmax = zrange
            layer_meshes = [
                trimesh.creation.extrude_polygon(poly, height=zmax - zmin).apply_translation((0, 0, zmin))
                for poly in self.layer_to_pattern[layer]]
            mesh = trimesh.Trimesh().union(layer_meshes, engine=engine)
            mesh.visual.vertex_colors = visual.random_color() if layer_to_color is None else layer_to_color[layer]
            meshes[layer] =(mesh)
        return meshes


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
            via_pattern = GroupedPattern(*patterns)
        boundary = Box((via_pattern.size[0] + 2 * boundary_grow,
                        via_pattern.size[1] + 2 * boundary_grow)).align((0, 0)).halign(0)
        via_pattern.align(boundary)
        super(Via, self).__init__([(via_pattern, via), (boundary, top_metal), (copy(boundary), bot_metal)])
        self.port['a0'] = boundary.port['a0']
        self.port['b0'] = boundary.port['b0']
