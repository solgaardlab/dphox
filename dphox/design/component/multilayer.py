from collections import defaultdict

import gdspy as gy
import nazca as nd
from shapely.geometry import Polygon, MultiPolygon
from descartes import PolygonPatch
import trimesh
from trimesh import creation, visual

try:
    import plotly.graph_objects as go
except ImportError:
    pass

from ...typing import *
from .pattern import Pattern, Path


class Multilayer:
    def __init__(self, pattern_to_layer: List[Tuple[Union[Pattern, Path, gy.Polygon, gy.FlexPath, Polygon], Union[int, str]]]):
        self.pattern_to_layer = {comp: layer if isinstance(comp, Pattern) else Pattern(comp)
                                 for comp, layer in pattern_to_layer}
        self.layer_to_pattern = self._layer_to_pattern()

    @property
    def input_ports(self) -> np.ndarray:
        all_input_ports = [c.input_ports for c in self.pattern_to_layer.keys() if c.input_ports.size > 0]
        return np.vstack(all_input_ports) if len(all_input_ports) > 0 else np.asarray([])

    @property
    def output_ports(self) -> np.ndarray:
        all_output_ports = [c.output_ports for c in self.pattern_to_layer.keys() if c.output_ports.size > 0]
        return np.vstack(all_output_ports) if len(all_output_ports) > 0 else np.asarray([])

    @property
    def contact_ports(self) -> np.ndarray:
        contact_ports = [c.contact_ports for c in self.pattern_to_layer.keys() if c.contact_ports.size > 0]
        return np.vstack(contact_ports) if len(contact_ports) > 0 else np.asarray([])

    @property
    def attachment_ports(self) -> np.ndarray:
        attachment_ports = [c.attachment_ports for c in self.pattern_to_layer.keys() if c.attachment_ports.size > 0]
        return np.vstack(attachment_ports) if len(attachment_ports) > 0 else np.asarray([])

    @property
    def bounds(self) -> Dim4:
        return self.gdspy_cell().get_bounding_box()

    def gdspy_cell(self, cell_name: str = 'dummy') -> gy.Cell:
        cell = gy.Cell(cell_name, exclude_from_current=(cell_name == 'dummy'))
        for pattern, layer in self.pattern_to_layer.items():
            for poly in pattern.polys:
                cell.add(gy.Polygon(np.asarray(poly.exterior.coords.xy).T, layer=layer))
        return cell

    def nazca_cell(self, cell_name: str) -> nd.Cell:
        with nd.Cell(cell_name) as cell:
            for pattern, layer in self.pattern_to_layer.items():
                for poly in pattern.polys:
                    nd.Polygon(points=np.asarray(poly.exterior.coords.xy).T, layer=layer).put()
            for idx, port in enumerate(self.input_ports):
                nd.Pin(f'a{idx}').put(*port, 180)
            for idx, port in enumerate(self.output_ports):
                nd.Pin(f'b{idx}').put(*port)
            for idx, port in enumerate(self.contact_ports):
                nd.Pin(f'c{idx}').put(*port)
            for idx, port in enumerate(self.attachment_ports):
                nd.Pin(f't{idx}').put(*port)
            nd.put_stub()
        return cell

    def _layer_to_pattern(self) -> Dict[Union[int, str], MultiPolygon]:
        layer_to_polys = defaultdict(list)
        for component, layer in self.pattern_to_layer.items():
            layer_to_polys[layer].extend(component.polys)
        pattern_dict = {layer: MultiPolygon(polys) for layer, polys in layer_to_polys.items()}
        # pattern_dict = {layer: (pattern if isinstance(pattern, MultiPolygon) else MultiPolygon([pattern]))
        #                 for layer, pattern in pattern_dict.items()}
        return pattern_dict

    def plot(self, ax, layer_to_color: Dict[Union[int, str], Union[Dim3, str]], alpha: float = 0.5):
        for layer, pattern in self.layer_to_pattern:
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

#### Nate simplifying the above command to output multiple stls ############
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
###################################################################################
#
#
# def multilayer(waveguide_pattern: Pattern, pads: List[Pattern],
#                waveguide_layer: str, metal_stack_layers: Tuple[str, ...],
#                via_stack_layers: Tuple[str, ...], clearout_layer: str, clearout_etch_stop_layer: str,
#                rib_pattern: Optional[Pattern] = None,
#                clearout_box_dim: Dim2 = None, clearouts: Tuple[Union[Dim2, Pattern], ...] = (),
#                doped_elems: List[Pattern] = None, contact_box_dim: Dim2 = None,
#                doping_stack_layer: Optional[str] = None,
#                clearout_etch_stop_grow: float = 0, via_shrink: float = 1, doping_grow: float = 0.25) -> Multilayer:
#     pattern_to_layer = {}
#     if rib_pattern is not None:
#         pattern_to_layer[GroupedPattern(rib_pattern, waveguide_pattern)] = waveguide_layer
#         pattern_to_layer[Pattern(waveguide_pattern.pattern - rib_pattern)] = rib_pattern
#     else:
#         pattern_to_layer[waveguide_pattern] = waveguide_layer
#     if len(pads) > 0:
#         sizes = [pad.size if contact_box_dim is None else contact_box_dim for pad in pads]
#         pattern_to_layer.update(
#             {GroupedPattern(*[Box(size).center_align(pad) for pad, size in zip(pads, sizes)]): layer
#              for layer in metal_stack_layers})
#         pattern_to_layer.update(
#             {GroupedPattern(*[Box(size).center_align(pad).grow(-via_shrink) for pad, size in zip(pads, sizes)]): layer
#              for layer in via_stack_layers})
#     if doping_stack_layer is not None:
#         dope_patterns = pads if doped_elems is None else doped_elems
#         if len(dope_patterns) > 0:
#             pattern_to_layer[GroupedPattern(*dope_patterns).grow(doping_grow)] = doping_stack_layer
#         # pattern_to_layer[GroupedPattern(*[Box(pad.size).center_align(pad).grow(doping_grow)
#         #                                   for pad in pads])] = doping_stack_layer
#
#     if clearout_box_dim is not None:
#         for region in clearouts:
#             clearout = Box(clearout_box_dim).center_align(region if isinstance(region, tuple) else region.center)
#             pattern_to_layer[clearout] = clearout_layer
#             if clearout_etch_stop_grow > 0 and clearout_etch_stop_layer is not None:
#                 pattern_to_layer[clearout.grow(clearout_etch_stop_grow)] = clearout_etch_stop_layer
#
#     return Multilayer(pattern_to_layer)
