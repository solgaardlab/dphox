from .pattern import Path, Pattern, Port
from .passive import Box
from ..typing import *

from collections import defaultdict

import numpy as np
import gdspy as gy
import nazca as nd
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.affinity import rotate
from shapely.ops import cascaded_union
from descartes import PolygonPatch
import trimesh
from trimesh import creation, visual
from copy import deepcopy as copy

try:
    import plotly.graph_objects as go
except ImportError:
    pass
try:
    import meep as mp
except ImportError:
    print("Unable to import pymeep")


class Multilayer:
    def __init__(self, pattern_to_layer: List[Tuple[Union[Pattern, Path, gy.Polygon, gy.FlexPath, Polygon],
                                                    Union[int, str]]] = None):
        self.pattern_to_layer = [] if pattern_to_layer is None else pattern_to_layer
        self.layer_to_pattern, self.port = self._init_multilayer()

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
        """

        Returns: Bounding box for the component

        """
        bbox = self.gdspy_cell().get_bounding_box()
        return bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]

    @property
    def center(self) -> Dim2:
        """

        Returns:
            Center for the component

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return (b[2] + b[0]) / 2, (b[3] + b[1]) / 2  # (avgx, avgy)

    def align(self, c: Union["Pattern", Tuple[float, float]]) -> "Multilayer":
        """Align center of pattern

        Args:
            c: A pattern (align to the pattern's center) or a center point for alignment

        Returns:
            Aligned pattern

        """
        old_x, old_y = self.center
        center = c if isinstance(c, tuple) else c.center
        self.translate(center[0] - old_x, center[1] - old_y)
        return self

    def translate(self, dx: float = 0, dy: float = 0) -> "Multilayer":
        """Translate the multilayer by translating all of the patterns within it individually

        Args:
            dx: translation in x
            dy: translation in y

        Returns:

        """
        for pattern, _ in self.pattern_to_layer:
            pattern.translate(dx, dy)
        self.layer_to_pattern, self.port = self._init_multilayer()
        return self

    def rotate(self, angle: float, origin: str = (0, 0)) -> "Multilayer":
        """Rotate the multilayer by rotating all of the patterns within it individually

        Args:
            angle: rotation angle
            origin: origin of rotation

        Returns:
            Rotated pattern

        """
        for pattern, _ in self.pattern_to_layer:
            pattern.rotate(angle, origin)
        self.layer_to_pattern, _ = self._init_multilayer()
        port_to_point = {name: rotate(Point(*port.xy), angle, origin) for name, port in self.port.items()}
        self.port = {name: Port(float(point.x), float(point.y), self.port[name].a + angle / 180 * np.pi)
                     for name, point in port_to_point.items()}
        return self

    def flip(self, center: Tuple[float, float] = (0, 0), horiz: bool = False) -> "Multilayer":
        """Flip the multilayer about center (vertical, or about x-axis, by default)

        Args:
            center: center about which to flip
            horiz: flip horizontally instead of vertically

        Returns:
            Flipped pattern

        """
        for pattern, _ in self.pattern_to_layer:
            pattern.flip(center, horiz)
        self.layer_to_pattern, _ = self._init_multilayer()
        return self

    def to(self, port: Port):
        return self.rotate(port.a_deg).translate(port.x, port.y)

    @property
    def copy(self) -> "Multilayer":
        """Return a copy of this layer for repeated use

        Returns:
            A deep copy of this layer

        """
        return copy(self)

    def gdspy_cell(self, cell_name: str = 'dummy') -> gy.Cell:
        """

        Args:
            cell_name: Cell name

        Returns:
            A GDSPY cell

        """
        cell = gy.Cell(cell_name, exclude_from_current=(cell_name == 'dummy'))
        for pattern, layer in self._pattern_to_layer.items():
            for poly in pattern.polys:
                cell.add(gy.Polygon(np.asarray(poly.exterior.coords.xy).T, layer=layer))
        return cell

    def nazca_cell(self, cell_name: str, callback: Optional[Callable] = None) -> nd.Cell:
        """Turn this multilayer into a Nazca cell

        Args:
            cell_name: Cell name
            callback: Callback function to call using Nazca (adding pins, other structures)

        Returns:
            A Nazca cell
        """
        with nd.Cell(cell_name) as cell:
            for pattern, layer in self._pattern_to_layer.items():
                for poly in pattern.polys:
                    nd.Polygon(points=np.around(np.asarray(poly.exterior.coords.xy).T, decimals=3),
                               layer=layer).put()
            for name, port in self.port.items():
                nd.Pin(name).put(*port.xya_deg)
            if callback is not None:
                callback()
            nd.put_stub()
        return cell

    def meep_geometry(self, layer_to_zrange: Dict[str, Tuple[float, float]], process_extrusion: Dict[str, List[Tuple[str, str, str]]] = None, layer_to_material: Dict = None):
        """Turn this multilayer into a  dictionary of meep geometry objects
        Args:

        Returns:
            A dictionary mapping layers to lists of meep geomtery objects
        """
        # probably a simple way to do this
        # def _xy_to_vector3(pt, dim=[1, 2, 0]):
        #     i = 0
        #     coord = [0, 0, 0]
        #     while min(dim) < 100 and i < len(pt):
        #         ind = dim.index(min(dim))
        #         if dim[ind] == 0:
        #             coord[ind] = 0
        #         else:
        #             coord[ind] = pt[dim[ind] - 1]
        #             i += 1
        #         dim[ind] = 100
        #         # print(coord)
        #         # i+=1
        #     return(mp.Vector3(*coord))

        def _pattern_to_prisms(pattern, material, zrange):
            zmin, zmax = zrange
            meep_geom_list = []
            for poly in pattern:
                prism_pts = [mp.Vector3(x, y, 0) for x, y in np.asarray(poly.exterior.coords.xy).T[1:, :]]  # normally is returned as a row of xs and ten a row of ys
                meep_geom_list.append(mp.Prism(prism_pts, height=0, axis=mp.Vector3(0, 0, 1), material=material).shift(mp.Vector3(0, 0, zmin)))
                # skip the first pt, meep doesn't like repeated pts like shapely

            # Convert into a list of vector3 objects and have a specified axis for the prism
            return meep_geom_list

        layer_to_extrusion = self.build_layers(layer_to_zrange=layer_to_zrange,
                                               process_extrusion=process_extrusion)
        meep_geometries = {}
        for layer, pattern_zrange in layer_to_extrusion.items():
            layer = layer.split('_')[0]
            if layer not in layer_to_material.keys():
                continue
            pattern, zrange = pattern_zrange
            material = layer_to_material[layer]

            if layer in meep_geometries.keys():
                meep_geometries[layer].append(_pattern_to_prisms(pattern, material, zrange))
            else:
                meep_geometries[layer] = _pattern_to_prisms(pattern, material, zrange)

        return meep_geometries

    def _init_multilayer(self) -> Tuple[Dict[Union[int, str], MultiPolygon], Dict[str, Port]]:
        self._pattern_to_layer = {comp: layer if isinstance(comp, Pattern) else Pattern(comp)
                                  for comp, layer in self.pattern_to_layer}
        layer_to_polys = defaultdict(list)
        for component, layer in self._pattern_to_layer.items():
            layer_to_polys[layer].extend(component.polys)
        pattern_dict = {layer: MultiPolygon(polys) for layer, polys in layer_to_polys.items()}
        # TODO: temporary way to assign ports
        port = dict(sum([list(pattern.port.items()) for pattern, _ in self.pattern_to_layer], []))
        return pattern_dict, port

    def add(self, pattern: Pattern, layer: str):
        self.pattern_to_layer.append((pattern, layer))
        self.layer_to_pattern, self.port = self._init_multilayer()

    def plot(self, ax, layer_to_color: Dict[Union[int, str], Union[Dim3, str]], alpha: float = 0.5):
        for layer, pattern in self.layer_to_pattern.items():
            ax.add_patch(PolygonPatch(pattern, facecolor=layer_to_color[layer], edgecolor='none', alpha=alpha))
        b = self.bounds
        ax.set_xlim((b[0], b[2]))
        ax.set_ylim((b[1], b[3]))
        ax.set_aspect('equal')

    @ property
    def size(self) -> Dim2:
        """Size of the pattern

        Returns:
            Tuple of the form :code:`(sizex, sizey)`

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return b[2] - b[0], b[3] - b[1]  # (maxx - minx, maxy - miny)

    def to_trimesh_dict(self, layer_to_zrange: Dict[str, Tuple[float, float]],
                        process_extrusion: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
                        layer_to_color: Optional[Dict[str, str]] = None, engine: str = 'scad',
                        include_oxide: bool = True):
        if include_oxide and 'oxide' not in self.layer_to_pattern:
            print('WARNING: oxide not included, so adding to multilayer')
            self.pattern_to_layer.append((Box(self.size).align(self.center), 'oxide'))
            self.layer_to_pattern, self.port = self._init_multilayer()
        meshes = {}

        # TODO(sunil): start using logging rather than printing
        def _add_trimesh_layer(pattern, zrange, layer):
            if layer_to_color is not None:  # TODO(Nate): remove repeated code
                if layer in layer_to_color:
                    zmin, zmax = zrange
                    layer_meshes = []
                    for poly in pattern:
                        try:
                            layer_meshes.append(
                                trimesh.creation.extrude_polygon(poly, height=zmax - zmin).apply_translation((0, 0, zmin))
                            )
                        except IndexError:
                            print('WARNING: bad polygon, skipping')
                            print(poly)
                    mesh = trimesh.Trimesh().union(layer_meshes, engine=engine)
                    mesh.visual.face_colors = visual.random_color() if layer_to_color is None else layer_to_color[layer]
                    meshes[layer] = mesh
                else:
                    print(f'WARNING: layer {layer} does not have a color, skipping...')
            else:
                zmin, zmax = zrange
                layer_meshes = []
                for poly in pattern:
                    try:
                        layer_meshes.append(
                            trimesh.creation.extrude_polygon(poly, height=zmax - zmin).apply_translation((0, 0, zmin))
                        )
                    except IndexError:
                        print('WARNING: bad polygon, skipping')
                        print(poly)
                mesh = trimesh.Trimesh().union(layer_meshes, engine=engine)
                layer = layer.split('_')[0]
                if layer in meshes.keys():
                    mesh = trimesh.Trimesh().union([mesh, meshes[layer]], engine=engine)
                mesh.visual.face_colors = visual.random_color() if layer_to_color is None else layer_to_color[layer]
                meshes[layer] = mesh

        if process_extrusion is not None:
            layer_to_extrusion = self.build_layers(layer_to_zrange, process_extrusion)
            for layer, pattern_zrange in layer_to_extrusion.items():
                pattern, zrange = pattern_zrange
                _add_trimesh_layer(pattern, zrange, layer)
                print('adding trimesh layer', layer)
            return meshes
        else:
            for layer, pattern in self.layer_to_pattern.items():
                try:
                    _add_trimesh_layer(pattern, layer_to_zrange[layer], layer)
                except KeyError:
                    print(f"No zranges given for the layer {layer}")
            return meshes

    def to_stls(self, prefix: str, layer_to_zrange: Dict[str, Tuple[float, float]],
                process_extrusion: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
                layer_to_color: Optional[Dict[str, str]] = None, engine: str = 'scad',
                layers: Optional[List[str]] = None,
                include_oxide: bool = True,
                all_layers: bool = False):  # addded this boolean to handle conformal maaping where we don't know the layer names immediately
        print('Building extrusion')
        meshes = self.to_trimesh_dict(layer_to_zrange, process_extrusion, layer_to_color, engine, include_oxide)
        print('Have meshes', meshes.keys())
        for layer, mesh in meshes.items():
            if all_layers:
                print('trying to export stl')
                mesh.export(f'{prefix}_{layer}.stl')
            elif layers and layer in layers:
                print('trying to export stl')
                mesh.export(f'{prefix}_{layer}.stl')

    def to_trimesh_scene(self, layer_to_zrange: Dict[str, Tuple[float, float]],
                         process_extrusion: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
                         layer_to_color: Optional[Dict[str, str]] = None, ignore_layers: Optional[List[str]] = None,
                         engine: str = 'scad'):
        meshes = self.to_trimesh_dict(layer_to_zrange, process_extrusion, layer_to_color, engine)
        scene = trimesh.Scene()
        ignore_layers = [] if ignore_layers is None else ignore_layers
        for mesh_name, mesh in meshes.items():
            if mesh_name not in ignore_layers:
                scene.add_geometry(mesh, mesh_name)
        return scene

    def show(self, layer_to_zrange: Dict[str, Tuple[float, float]],
             process_extrusion: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
             layer_to_color: Optional[Dict[str, str]] = None, ignore_layers: Optional[List[str]] = None,
             engine: str = 'scad'):
        self.to_trimesh_scene(layer_to_zrange, process_extrusion, layer_to_color, ignore_layers, engine).show()

    def build_layers(self, layer_to_zrange: Dict[str, Tuple[float, float]],
                     process_extrusion: Dict[str, List[Tuple[str, str, str]]]):
        layer_to_extrusion = {}
        layers = self.layer_to_pattern.keys()
        layer_to_pattern_processed = self.layer_to_pattern.copy()

        # TODO(Nate):Add Conformal deposition logic
        # build topographic map
        minx, miny, maxx, maxy = self.bounds
        starting_slate = Pattern(gy.Polygon(
            [(minx, miny), (minx, maxy),
             (maxx, maxy), (maxx, miny)]))
        topo_map_dict = {str(0.0): starting_slate}  # zero is the starting height

        # TODO: sort out the Pattern instance creations, so they aren't necessary

        for step, operations in process_extrusion.items():
            for layer_relation in operations:
                layer, other_layer, operation = layer_relation
                print(layer)
                if 'dope' in step and operation == 'intersection':
                    # make a new layer for each doping intersection
                    zmin, zmax = layer_to_zrange[other_layer]
                    z0, z1 = layer_to_zrange[layer]
                    # TODO(): how to deal with different depth doping currently not addressed
                    new_zrange = (max(zmax - (z1 - z0), zmin), zmax)
                else:
                    new_zrange = layer_to_zrange[layer]
                if layer in layers:
                    if other_layer in layers:
                        pattern = Pattern(layer_to_pattern_processed[layer]).boolean_operation(
                            Pattern(layer_to_pattern_processed[other_layer]), operation
                        ).shapely
                        layer_to_pattern_processed[layer] = pattern
                    else:
                        pattern = layer_to_pattern_processed[layer]
                    # insert conformal stuff here

                    # building the topographic map
                    zmin, zmax = new_zrange
                    thickness = zmax - zmin
                    heights = list(topo_map_dict.keys())
                    old_topo_map_dict = topo_map_dict.copy()
                    sorted_heights = list(map(float, heights))
                    sorted_heights.sort(reverse=True)
                    if 'conformal' in step:
                        if 'etch' in step:
                            raising_pattern = starting_slate.boolean_operation(Pattern(pattern), 'difference')
                        else:
                            raising_pattern = Pattern(pattern)
                        for count, infr_h in enumerate(sorted_heights):
                            for supr_h in sorted_heights:
                                if count == 0:
                                    # dilation, should only occur in not etched regions, need to filter with raisd_pattern and union
                                    dialated_pattern = topo_map_dict[str(supr_h)].offset(grow_d=thickness).boolean_operation(raising_pattern, 'intersection')
                                    topo_map_dict[str(supr_h)] = dialated_pattern.boolean_operation(topo_map_dict[str(supr_h)], 'union')
                                else:
                                    if supr_h > infr_h:
                                        #  and moving boudaries
                                        topo_map_dict[str(infr_h)] = Pattern(topo_map_dict[str(infr_h)]).boolean_operation(
                                            Pattern(topo_map_dict[str(supr_h)]), 'difference')
                                    else:
                                        continue
                            topo_map_dict[str(infr_h)] = starting_slate.boolean_operation(topo_map_dict[str(infr_h)], 'intersection')
                    else:
                        raising_pattern = pattern
                    # move these actions into the sorted list? actually no because dialation should be finished first
                    for elevation in heights:
                        # material added
                        raised = Pattern(raising_pattern).boolean_operation(
                            Pattern(topo_map_dict[elevation]), 'intersection'
                        ).shapely
                        # new matrial fully etched
                        not_raised = Pattern(topo_map_dict[elevation]).boolean_operation(
                            (Pattern(raised)), 'difference'
                        ).shapely

                        topo_map_dict[str(elevation)] = Pattern(not_raised)
                        topo_map_dict[str(float(elevation) + (zmax - zmin))] = Pattern(raised)

                    # use the current topographic map and the previous topographic map to make extrusions
                    # needs to be adjusted to only deal with added materials
                    for old_elevation in heights:
                        for new_elevation in topo_map_dict.keys():
                            if float(old_elevation) == float(new_elevation):
                                continue
                            new_zrange = (float(old_elevation), float(new_elevation))
                            # there may be coincedent edges whcih cause intersection problems
                            pattern = Pattern(topo_map_dict[new_elevation]).boolean_operation(Pattern(old_topo_map_dict[old_elevation]), 'intersection').shapely

                            new_layer = layer + '_' + f'{np.around(float(old_elevation),3)*1000:.0f}' + '_' + f'{np.around(float(new_elevation),3)*1000:.0f}'
                            if pattern.geoms:
                                pattern_shapely = cascaded_union(pattern.geoms)
                                pattern_shapely = MultiPolygon([pattern_shapely]) if isinstance(pattern_shapely, Polygon) else pattern_shapely
                                layer_to_extrusion[new_layer] = (pattern_shapely, new_zrange)
                            # pass

                    # if pattern.geoms:
                    #     pattern_shapely = cascaded_union(pattern.geoms)
                    #     pattern_shapely = MultiPolygon([pattern_shapely]) if isinstance(pattern_shapely, Polygon) else pattern_shapely
                    #     layer_to_pattern_processed[layer] = pattern
                    #     layer_to_extrusion[layer] = (pattern_shapely, new_zrange)

        topo_list = []
        for layer, pattern in topo_map_dict.items():
            topo_list.append((pattern, layer))

        # WORKS, now I need to add dialation and fit this into the extrusion flow
        # I think the strategy is dialate everything, then difference by height when tallest is unobstructed and then apply raisig pattern
        # use buffer for dialtion and use default rounding
        # with the new topo, compare/intersect with the old topo and make the zranges for all height combinations
        self.topo = Multilayer(topo_list)
        print(layer_to_extrusion.keys())
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
        self.layer_to_pattern, self.port = self._init_multilayer()
        return [(fill, layer_name)]


class Via(Multilayer):
    def __init__(self, via_dim: Dim2, boundary_grow: Union[float, Tuple[float]], metal: Union[str, List[str]],
                 via: Union[str, List[str]], pitch: float = 0, shape: Optional[Shape2] = None):
        """Via / metal multilayer stack (currently all params should be specified to 2 decimal places)

        Args:
            via_dim: dimensions of the via (or each via in via array)
            boundary_grow: boundary growth around the via or via array
            metal: Metal layers
            via: Via layers
            pitch: Pitch of the vias (center to center)
            shape: Shape of the array (rows, cols)
        """
        self.via_dim = via_dim
        metal = (metal,) if isinstance(metal, str) else metal
        boundary_grow = boundary_grow if isinstance(boundary_grow, tuple) else tuple([boundary_grow] * len(metal))
        self.boundary_grow = boundary_grow
        self.metal = metal
        self.via = via
        self.pitch = pitch
        self.shape = shape
        self.config = copy(self.__dict__)

        max_boundary_grow = max(boundary_grow)
        via_pattern = Box(via_dim, decimal_places=2)
        if pitch > 0 and shape is not None:
            patterns = []
            x, y = np.meshgrid(np.arange(shape[0]) * pitch, np.arange(shape[1]) * pitch)
            for x, y in zip(x.flatten(), y.flatten()):
                patterns.append(via_pattern.copy.translate(x, y))
            via_pattern = Pattern(*patterns, decimal_places=2)
        boundary = Box((via_pattern.size[0] + 2 * max_boundary_grow,
                        via_pattern.size[1] + 2 * max_boundary_grow), decimal_places=2).align((0, 0)).halign(0)
        via_pattern.align(boundary)
        layers = []
        if isinstance(via, list):
            layers += [(via_pattern.copy, layer) for layer in via]
        elif isinstance(via, str):
            layers += [(via_pattern, via)]
        layers += [(Box((via_pattern.size[0] + 2 * bg,
                         via_pattern.size[1] + 2 * bg), decimal_places=2).align(boundary), layer)
                   for layer, bg in zip(metal, boundary_grow)]
        super(Via, self).__init__(layers)
        self.port['a0'] = Port(self.bounds[0], 0, np.pi)
        self.port['b0'] = Port(self.bounds[2], 0)
