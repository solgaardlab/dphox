import datetime
from collections import defaultdict
from copy import deepcopy as copy
from typing import BinaryIO

import gdspy as gy
import klamath
import matplotlib.pyplot as plt
import numpy as np
from descartes import PolygonPatch
from klamath.library import FileHeader
from pydantic.dataclasses import dataclass
from shapely.affinity import rotate
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from .foundry import CommonLayer, FABLESS, fabricate, Foundry
from .pattern import Box, Pattern, Port
from .typing import Dict, Float2, Float4, Int2, List, Optional, Tuple, Union
from .utils import fix_dataclass_init_docs, poly_points, PORT_LABEL_LAYER, PORT_LAYER, split_holes

try:
    import plotly.graph_objects as go
except ImportError:
    pass
try:
    MEEP_IMPORTED = True
    import meep as mp
except ImportError:
    MEEP_IMPORTED = False
try:
    NAZCA_IMPORTED = True
    import nazca as nd
except ImportError:
    NAZCA_IMPORTED = False

try:
    HOLOVIEWS_IMPORTED = True
    import holoviews as hv
    from holoviews.streams import Pipe
    from holoviews import opts
    import panel as pn
    from bokeh.models import Range1d, LinearAxis
    from bokeh.models.renderers import GlyphRenderer
    from bokeh.plotting.figure import Figure
except ImportError:
    HOLOVIEWS_IMPORTED = False


class Device:
    """A :code:`Device` defines a device (active or passive) in a GDS.

    A :code:`Device` is a core object in DPhox, which enables composition of multiple :code:`Pattern` or
    :code:`Polygon` mapped to process stack layers into a single :code:`Device`. Once the process is defined,
    :code:`Trimesh` may be used to show a full 3D model of the device. Additionally, a :code:`Device` can be mapped
    to a GDSPY or Nazca cell.

    Attributes:
        name: Name of the device. This name can be anything you choose, ideally not repeating
            any previously defined :code:`Device`.
        pattern_to_layer: A list of tuples, each consisting of a :code:`Pattern` or :code:`PolygonLike` followed by
            a layer label (integer or string).
    """

    def __init__(self, name: str, pattern_to_layer: List[Tuple[Pattern, Union[int, str]]] = None):
        self.name = name
        self.pattern_to_layer = [] if pattern_to_layer is None else pattern_to_layer
        self.pattern_to_layer = [(Pattern(comp), layer) for comp, layer in self.pattern_to_layer]
        self.layer_to_geoms, self.port = self._init_multilayer()

    def _init_multilayer(self) -> Tuple[Dict[Union[int, str], MultiPolygon], Dict[str, Port]]:
        layer_to_polys = defaultdict(list)
        for component, layer in self.pattern_to_layer:
            layer_to_polys[layer].extend(component.polys)
        pattern_dict = {layer: MultiPolygon(polys) for layer, polys in layer_to_polys.items()}
        # TODO: temporary way to assign ports
        port = dict(sum([list(pattern.port.items()) for pattern, _ in self.pattern_to_layer], []))
        return pattern_dict, port

    @classmethod
    def from_pattern(cls, pattern: Pattern, name: str, layer: str):
        """A class method to convert a :code:`Pattern` into a :code:`Device`.

        Args:
            pattern: The pattern that is being used to generate the device.
            name: Name for the component.
            layer: The layer for the pattern.

        Returns:
            The :code:`Device` containing the :code:`Pattern` at the specified :code:`layer`.

        """
        return cls(name, [(pattern, layer)])

    @classmethod
    def from_nazca_cell(cls, cell: "nd.Cell"):
        """Get the Device from a nazca cell (assumes nazca is installed).

        See Also:
            https://nazca-design.org/forums/topic/clipping-check-distance-and-length-for-interconnects/

        Args:
            cell: Nazca cell to get Multilayer

        Returns:
            The :code:`Device` with the :code:`nazca` cell.

        """
        # a glimpse into cell_iter()
        # code from
        multilayers = defaultdict(list)
        for named_tuple in nd.cell_iter(cell, flat=True):
            if named_tuple.cell_start:
                for i, (polygon, points, bbox) in enumerate(named_tuple.iters['polygon']):
                    if polygon.layer == 'bb_pin':
                        # TODO(sunil): actually extract the pins from this layer.
                        continue
                    # fixing point definitions from mask to 1nm precision,
                    # kinda hacky but is physical and prevents false polygons
                    points = np.around(points, decimals=3)
                    multilayers[polygon.layer].append(Pattern(Polygon(points)))
        return cls(cell.name, [(Pattern(*pattern_list), layer) for layer, pattern_list in multilayers.items()])

    @property
    def bounds(self) -> Float4:
        """Bounding box of the form (minx, maxx, miny, maxy)

        Returns:
            Bounding box tuple.

        """
        bound_list = np.array([p.bounds for p, _ in self.pattern_to_layer]).T
        return np.min(bound_list[0]), np.min(bound_list[1]), np.max(bound_list[2]), np.max(bound_list[3])

    @property
    def center(self) -> Float2:
        """Center (x, y) for the device.

        Returns:
            Center for the component.

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return (b[2] + b[0]) / 2, (b[3] + b[1]) / 2  # (avgx, avgy)

    def align(self, c: Union[Pattern, "Device", Float2]) -> "Device":
        """Align center of pattern

        Args:
            c: A pattern (align to the pattern's center) or a center point for alignment.

        Returns:
            Aligned pattern

        """
        old_x, old_y = self.center
        center = c.center if isinstance(c, Device) or isinstance(c, Pattern) else c
        self.translate(center[0] - old_x, center[1] - old_y)
        return self

    def halign(self, c: Union[Pattern, "Device", float], left: bool = True, opposite: bool = False) -> "Device":
        """Horizontal alignment of device

        Args:
            c: A device (horizontal align to the device's boundary) or a center x for alignment.
            left: (if :code:`c` is pattern) Align to left boundary of component, otherwise right boundary.
            opposite: (if :code:`c` is pattern) Align opposite faces (left-right, right-left).

        Returns:
            Horizontally aligned device

        """
        x = self.bounds[0] if left else self.bounds[2]
        p = c if isinstance(c, float) or isinstance(c, int) \
            else (c.bounds[0] if left and not opposite or opposite and not left else c.bounds[2])
        self.translate(dx=p - x)
        return self

    def valign(self, c: Union[Pattern, "Device", float], bottom: bool = True, opposite: bool = False) -> "Device":
        """Vertical alignment of devie

        Args:
            c: A pattern (vertical align to the pattern's boundary) or a center y for alignment.
            bottom: (if :code:`c` is pattern) Align to upper boundary of component, otherwise lower boundary.
            opposite: (if :code:`c` is pattern) Align opposite faces (upper-lower, lower-upper).

        Returns:
            Vertically aligned device

        """
        y = self.bounds[1] if bottom else self.bounds[3]
        p = c if isinstance(c, float) or isinstance(c, int) \
            else (c.bounds[1] if bottom and not opposite or opposite and not bottom else c.bounds[3])
        self.translate(dy=p - y)
        return self

    def translate(self, dx: float = 0, dy: float = 0) -> "Device":
        """Translate the device by translating all of the patterns within it.

        Args:
            dx: translation in x.
            dy: translation in y.

        Returns:
            The translated device.

        """
        for pattern, _ in self.pattern_to_layer:
            pattern.translate(dx, dy)
        self.layer_to_geoms, _ = self._init_multilayer()
        for name, port in self.port.items():
            self.port[name] = Port(port.x + dx, port.y + dy, port.a, port.w)
        return self

    def rotate(self, angle: float, origin: Float2 = (0, 0)) -> "Device":
        """Rotate the device by rotating all of the patterns within it.

        Args:
            angle: rotation angle in degrees.
            origin: origin of rotation.

        Returns:
            The rotated device.

        """
        for pattern, _ in self.pattern_to_layer:
            pattern.rotate(angle, origin)
        self.layer_to_geoms, _ = self._init_multilayer()
        port_to_point = {name: rotate(Point(*port.xy), angle, origin) for name, port in self.port.items()}
        self.port = {name: Port(float(point.x), float(point.y), np.mod(self.port[name].a + angle, 360))
                     for name, point in port_to_point.items()}
        return self

    def reflect(self, center: Float2 = (0, 0), horiz: bool = False) -> "Device":
        """Reflected the multilayer about center (vertical, or about x-axis, by default)

        Args:
            center: center about which to reflect.
            horiz: reflect horizontally instead of vertically.

        Returns:
            Reflected pattern.

        """
        for pattern, _ in self.pattern_to_layer:
            pattern.reflect(center, horiz)
        self.layer_to_geoms, _ = self._init_multilayer()
        return self

    def to(self, port: Port, from_port: Optional[str] = None):
        """Lego-connect this device's :code:`from_port` (origin if not specified) to another device's port.

        Args:
            port: The port to which the device should be connected.
            from_port: The port name corresponding to this device's port that should be connected to :code:`port`.

        Returns:
            This device, translated and rotated after connection.

        """
        if from_port is None:
            return self.rotate(port.a).translate(port.x, port.y)
        else:
            return self.rotate(port.a - self.port[from_port].a + 180, origin=self.port[from_port].xy).translate(
                port.x - self.port[from_port].x, port.y - self.port[from_port].y
            )

    @property
    def copy(self) -> "Device":
        """Return a copy of this device for repeated use

        Returns:
            A deep copy of this device.

        """
        return copy(self)

    @property
    def gdspy_cell(self, foundry: Foundry = FABLESS) -> gy.Cell:
        """Turn this multilayer into a gdspy cell.

        Args:
            Foundry for creating the gdspy cell (provide the layer map).

        Returns:
            A GDSPY cell.

        """
        cell = gy.Cell(self.name)
        for pattern, layer in self.pattern_to_layer:
            for poly in pattern.polys:
                cell.add(
                    gy.Polygon(poly_points(poly),
                               layer=foundry.layer_to_gds_label[layer][0],
                               datatype=foundry.layer_to_gds_label[layer][1])
                )
        return cell

    def nazca_cell(self, foundry: Foundry = FABLESS) -> "nd.Cell":
        """Turn this multilayer into a nazca cell (need to install nazca for this to work).

        Args:
            Foundry for creating the nazca cell (provide the layer map).

        Returns:
            A Nazca cell.

        """
        if not NAZCA_IMPORTED:
            raise ImportError('Nazca not installed! Please install nazca prior to running nazca_cell().')
        with nd.Cell(self.name) as cell:
            for pattern, layer in self.pattern_to_layer:
                for poly in pattern.polys:
                    nd.Polygon(points=list(np.around(poly_points(poly), decimals=3)),
                               layer=foundry.layer_to_gds_label[layer]).put()
            for name, port in self.port.items():
                nd.Pin(name).put(*port.xya)
            nd.put_stub()
        return cell

    def add(self, pattern: Pattern, layer: str):
        """Add pattern, layer to Multilayer

        Args:
            pattern: :code:`Pattern` to add.
            layer: Layer to incorporate :code:`Pattern`.

        Returns:

        """
        self.pattern_to_layer.append((pattern, layer))
        self.layer_to_geoms, self.port = self._init_multilayer()

    def plot(self, ax: Optional = None, foundry: Foundry = FABLESS,
             exclude_layer: Optional[List[CommonLayer]] = None, alpha: float = 0.5):
        """Plot this device on a matplotlib plot.

        Args:
            ax: Matplotlib axis handle to plot the device.
            foundry: :code:`Foundry` object for matplotlib plotting.
            exclude_layer: Exclude all layers in this list when plotting.
            alpha: The transparency factor for the plot (to see overlay of structures from many layers).

        Returns:

        """
        ax = plt.gca() if ax is None else ax
        for layer, pattern in self.layer_to_geoms.items():
            if layer in exclude_layer:
                continue
            color = foundry.color(layer)
            if color is None:
                raise ValueError(f"The layer {layer} does not exist in the foundry stack, so could not find a color.")
            ax.add_patch(PolygonPatch(unary_union(pattern.geoms),  # union avoids some weird plotting with alpha < 1
                                      facecolor=color, edgecolor='none', alpha=alpha))
            for name, port in self.port.items():
                port_xy = port.xy - port.normal(port.w)
                ax.add_patch(PolygonPatch(port.shapely,
                                          facecolor='red', edgecolor='none', alpha=alpha))
                ax.text(*port_xy, name)
        b = self.bounds
        ax.set_xlim((b[0], b[2]))
        ax.set_ylim((b[1], b[3]))
        ax.set_xlabel(r'$x$ ($\mu$m)')
        ax.set_ylabel(r'$y$ ($\mu$m)')
        ax.set_aspect('equal')

    def hvplot(self, foundry: Foundry = FABLESS, exclude_layer: Optional[List[CommonLayer]] = None, alpha: float = 0.5):
        """Plot this device on a matplotlib plot.

        Args:
            foundry: :code:`Foundry` object for holoviews plotting.
            exclude_layer: Exclude all layers in this list when plotting.
            alpha: The transparency factor for the plot (to see overlay of structures from many layers).

        Returns:
            The holoviews Overlay for displaying all of the polygons.

        """
        if not HOLOVIEWS_IMPORTED:
            raise ImportError('Holoviews, Panel, and/or Bokeh not yet installed. Check your installation...')
        plots_to_overlay = []
        exclude_layer = [] if exclude_layer is None else exclude_layer
        b = self.bounds
        for layer, pattern in self.layer_to_geoms.items():
            if layer in exclude_layer:
                continue
            geom = unary_union(pattern.geoms)

            def _holoviews_poly(shapely_poly):
                x, y = poly_points(shapely_poly).T
                holes = [[np.array(hole.coords.xy).T for hole in shapely_poly.interiors]]
                return {'x': x, 'y': y, 'holes': holes}

            polys = [_holoviews_poly(poly) for poly in geom] if isinstance(geom, MultiPolygon) else [
                _holoviews_poly(geom)]
            color = foundry.color(layer)
            if color is None:
                raise ValueError(f"The layer {layer} does not exist in the foundry stack, so could not find a color.")
            plots_to_overlay.append(
                hv.Polygons(polys, name=layer).opts(data_aspect=1, frame_height=200, fill_alpha=alpha,
                                                    ylim=(b[1], b[3]), xlim=(b[0], b[2]),
                                                    color=color, line_alpha=0))
        for name, port in self.port.items():
            x, y = port.shapely.exterior.coords.xy
            port_xy = port.xy - port.normal(port.w)
            plots_to_overlay.append(
                hv.Polygons([{'x': x, 'y': y}]).opts(
                    data_aspect=1, frame_height=200, ylim=(b[1], b[3]), xlim=(b[0], b[2]), color='red', line_alpha=0
                ) * hv.Text(*port_xy, name)
            )
        return hv.Overlay(plots_to_overlay).opts(show_legend=True)

    @property
    def size(self) -> Float2:
        """Size of the pattern.

        Returns:
            Tuple of the form :code:`(sizex, sizey)`.

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return b[2] - b[0], b[3] - b[1]  # (maxx - minx, maxy - miny)

    def trimesh(self, foundry: Foundry = FABLESS, exclude_layer: Optional[List[CommonLayer]] = None):
        """Fabricate this device based on a :code:`Foundry`.

        This method is fairly rudimentary and will not implement things like conformal deposition. At the moment,
        you can implement things like rib etches which can be determined using 2d shape operations. Depositions in
        layers above etched layers will just start from the maximum z extent of the previous layer. This is specified
        by the :code:`Foundry` stack.

        Args:
            foundry: The foundry for each layer.
            exclude_layer: Exclude all layers in this list.

        Returns:
            The device :code:`Scene` to visualize.

        """
        return fabricate(self.layer_to_geoms, foundry, exclude_layer=exclude_layer)

    @classmethod
    def aggregate(cls, devices: List["Device"], name: Optional[str] = None):
        """Aggregate many devices in the list into a single device (non-annotated)

        Args:
            devices: List of devices.
            name: Name of the new aggregated device.

        Returns:
            The aggregated :code:`Device`.

        """
        name = '|'.join([m.name for m in devices]) if name is None else name
        return cls(name, sum([m.pattern_to_layer for m in devices], []))

    @classmethod
    def from_gds(cls, filepath: str, foundry: Foundry = FABLESS) -> List["Device"]:
        """Generate non-annotated device from GDS file based on the provided layer map in :code:`foundry`.

        Args:
            filepath: The filepath to read the GDS device.
            foundry: The foundry to get the device.

        Returns:
            The non-annotated Device generated from reading a GDS file.

        """
        with open(filepath, 'rb') as stream:
            header = klamath.library.FileHeader.read(stream)
            structs = {}
            struct = klamath.library.try_read_struct(stream)
            while struct is not None:
                name, elements = struct
                structs[name] = elements
                struct = klamath.library.try_read_struct(stream)
            devices = []
            for key in structs.keys():
                pattern_to_layer = []
                port = {}
                for obj in structs[key]:
                    if isinstance(obj, klamath.elements.Boundary):
                        if obj.layer in foundry.gds_label_to_layer:
                            layer = foundry.gds_label_to_layer[obj.layer]  # klamath layer is gds_label in dphox
                            pattern = Pattern(header.user_units_per_db_unit * obj.xy)
                            pattern_to_layer.append((pattern, layer))
                device = cls(str(key), pattern_to_layer)
                device.port = port
                devices.append(device)
        return devices

    def gds_elements(self, foundry: Foundry = FABLESS, user_units_per_db_unit: float = 0.001):
        """Use `klamath <https://mpxd.net/code/jan/klamath/src/branch/master/klamath>`_ to convert to GDS elements
        using a foundry object and user units.

        Args:
            foundry: The foundry used for the layer map.
            user_units_per_db_unit: User units per unit (to convert from nm to um, need to use 0.001 to convert).

        Returns:
            The `klamath` GDS elements for the device.
        """
        elements = []
        for layer, geom in self.layer_to_geoms.items():
            elements += [
                klamath.elements.Boundary(
                    layer=foundry.layer_to_gds_label[layer],
                    xy=(poly_points(poly) / user_units_per_db_unit).astype(np.int32),
                    properties={})
                for poly in split_holes(geom)
            ]
        for name, port in self.port.items():
            elements += [
                klamath.elements.Text(layer=(PORT_LABEL_LAYER, 0),
                                      xy=(port.xy - port.normal(port.w)) / user_units_per_db_unit,
                                      string=name.encode('utf-8'), properties={},
                                      presentation=0, angle_deg=0, invert_y=False, width=0, path_type=0, mag=1),
                klamath.elements.Boundary(
                    layer=(PORT_LAYER, 0),
                    xy=(poly_points(port.shapely) / user_units_per_db_unit).astype(np.int32),
                    properties={})
            ]
        return elements

    def to_gds_stream(self, stream: BinaryIO, foundry: Foundry = FABLESS, user_units_per_db_unit: float = 0.001):
        """Use `klamath <https://mpxd.net/code/jan/klamath/src/branch/master/klamath>`_ to add device struct to GDS.

        Args:
            stream: Stream for a GDS file.
            foundry: The foundry used for the layer map.
            user_units_per_db_unit: User units per unit (to convert from nm to um, need to use 0.001 to convert).

        Returns:

        """
        klamath.library.write_struct(stream, self.name.encode('utf-8'),
                                     self.gds_elements(foundry, user_units_per_db_unit))

    def to_gds(self, filepath: str, foundry: Foundry = FABLESS, user_units_per_db_unit: float = 0.001,
               meters_per_db_unit: float = 1e-9):
        """Use `klamath <https://mpxd.net/code/jan/klamath/src/branch/master/klamath>`_ to convert to GDS
        using a foundry object and a provided :code:`filepath`.

        Args:
            filepath: The filepath to output the GDS.
            foundry: The foundry used for the layer map.
            user_units_per_db_unit: User units per unit (to convert from nm to um, need to use 0.001 to convert).
            meters_per_db_unit: Meters per unit (usually nanometers, hence 1e-9).

        Returns:

        """
        with open(filepath, 'wb') as stream:
            header = FileHeader(name=b'dphox', user_units_per_db_unit=user_units_per_db_unit,
                                meters_per_db_unit=meters_per_db_unit,
                                mod_time=datetime.datetime.now(), acc_time=datetime.datetime.now())
            header.write(stream)
            self.to_gds_stream(stream, foundry, user_units_per_db_unit)
            klamath.records.ENDLIB.write(stream, None)


@fix_dataclass_init_docs
@dataclass
class Via(Device):
    """Via / metal multilayer stack (currently all params should be specified to 2 decimal places)

    Attributes:
        via_extent: Dimensions of the via (or each via in via array).
        boundary_grow: Boundary growth around the via or via array.
        metal: Metal layer labels for the via (the thin metal layers).
        via: Via layer labels for the via (the actual tall vias).
        pitch: Pitch of the vias (center to center).
        shape: Shape of the array (rows, cols).
        name: Name of the via.
    """

    via_extent: Float2
    boundary_grow: Union[float, Float2]
    metal: Union[str, List[str]] = (CommonLayer.METAL_1, CommonLayer.METAL_2)
    via: Union[str, List[str]] = CommonLayer.VIA_1_2
    pitch: float = 0
    shape: Optional[Int2] = None
    name: str = 'via'

    def __post_init_post_parse__(self):
        self.metal = (self.metal,) if isinstance(self.metal, str) else self.metal
        self.boundary_grow = self.boundary_grow if isinstance(self.boundary_grow, tuple) \
            else tuple([self.boundary_grow] * len(self.metal))

        max_boundary_grow = max(self.boundary_grow)
        via_pattern = Box(self.via_extent, decimal_places=2)
        if self.pitch > 0 and self.shape is not None:
            patterns = []
            x, y = np.meshgrid(np.arange(self.shape[0]) * self.pitch, np.arange(self.shape[1]) * self.pitch)
            for x, y in zip(x.flatten(), y.flatten()):
                patterns.append(via_pattern.copy.translate(x, y))
            via_pattern = Pattern(*patterns, decimal_places=2)
        boundary = Box((via_pattern.size[0] + 2 * max_boundary_grow,
                        via_pattern.size[1] + 2 * max_boundary_grow), decimal_places=2).align((0, 0)).halign(0)
        via_pattern.align(boundary)
        layers = []
        if isinstance(self.via, list):
            layers += [(via_pattern.copy, layer) for layer in self.via]
        elif isinstance(self.via, str):
            layers += [(via_pattern, self.via)]
        layers += [(Box((via_pattern.size[0] + 2 * bg,
                         via_pattern.size[1] + 2 * bg), decimal_places=2).align(boundary), layer)
                   for layer, bg in zip(self.metal, self.boundary_grow)]
        super(Via, self).__init__(self.name, layers)
        self.port['w'] = Port(self.bounds[0], 0, -180)
        self.port['e'] = Port(self.bounds[2], 0)
        self.port['s'] = Port(0, self.bounds[1], -90)
        self.port['n'] = Port(0, self.bounds[3], 90)
        self.port['c'] = Port(0, 0)

