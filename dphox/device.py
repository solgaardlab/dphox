from collections import defaultdict
from copy import deepcopy as copy

import gdspy as gy
import numpy as np
from descartes import PolygonPatch
from pydantic.dataclasses import dataclass
from shapely.affinity import rotate
from shapely.geometry import MultiPolygon, Point, Polygon

from .foundry import Foundry, fabricate, CommonLayer, FABLESS, ProcessOp
from .pattern import Box, Pattern, Port
from .typing import Dict, List, Optional, Int2, Float2, Float3, Float4, Tuple, Union
from .utils import fix_dataclass_init_docs

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
        """Bounding box

        Returns:
            Bounding box for the component of the form (minx, maxx, miny, maxy)

        """
        bound_list = np.array([p.bounds for p, _ in self.pattern_to_layer]).T
        return np.min(bound_list[0]), np.min(bound_list[1]), np.max(bound_list[2]), np.max(bound_list[3])

    @property
    def center(self) -> Float2:
        """

        Returns:
            Center for the component

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return (b[2] + b[0]) / 2, (b[3] + b[1]) / 2  # (avgx, avgy)

    def align(self, c: Union["Pattern", Float2]) -> "Device":
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
            self.port[name] = Port(port.x + dx, port.y + dy, port.a)
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
        self.port = {name: Port(float(point.x), float(point.y), self.port[name].a + angle)
                     for name, point in port_to_point.items()}
        return self

    def reflect(self, center: Float2 = (0, 0), horiz: bool = False) -> "Device":
        """Reflected the multilayer about center (vertical, or about x-axis, by default)

        Args:
            center: center about which to reflect
            horiz: reflect horizontally instead of vertically

        Returns:
            Reflected pattern.

        """
        for pattern, _ in self.pattern_to_layer:
            pattern.reflect(center, horiz)
        self.layer_to_geoms, _ = self._init_multilayer()
        return self

    def to(self, port: Port, port_name: Optional[str] = None):
        """Translate the device so the origin (or port :code:`port_name`) is matched with that of :code:`port`.

        Args:
            port: The port to which the device should be moved
            port_name: The port name corresponding to this device's port that should be connected to :code:`port`.

        Returns:
            The resultant device.

        """
        if port_name is None:
            return self.rotate(port.a).translate(port.x, port.y)
        else:
            return self.rotate(port.a - self.port[port_name].a + 180, origin=self.port[port_name].xy).translate(
                port.x - self.port[port_name].x, port.y - self.port[port_name].y
            )

    @property
    def copy(self) -> "Device":
        """Return a copy of this layer for repeated use

        Returns:
            A deep copy of this layer

        """
        return copy(self)

    @property
    def gdspy_cell(self, foundry: Foundry = FABLESS) -> gy.Cell:
        """Turn this multilayer into a GDSPY cell.

        Returns:
            A GDSPY cell.

        """
        cell = gy.Cell(self.name)
        for pattern, layer in self.pattern_to_layer:
            for poly in pattern.polys:
                cell.add(gy.Polygon(np.asarray(poly.exterior.coords.xy).T, layer=foundry.layer_to_gds_label[layer]))
        return cell

    @property
    def nazca_cell(self) -> "nd.Cell":
        """Turn this multilayer into a Nazca cell

        Args:
            callback: Callback function to call using Nazca (adding pins, other structures)

        Returns:
            A Nazca cell
        """
        if not NAZCA_IMPORTED:
            raise ImportError('Nazca not installed! Please install nazca prior to running nazca_cell().')
        with nd.Cell(self.name) as cell:
            for pattern, layer in self.pattern_to_layer:
                for poly in pattern.polys:
                    nd.Polygon(points=list(np.around(np.asarray(poly.exterior.coords.xy).T, decimals=3)),
                               layer=layer).put()
            for name, port in self.port.items():
                nd.Pin(name).put(*port.xya)
            nd.put_stub()
        return cell

    def add(self, pattern: Pattern, layer: str):
        """Add pattern, layer to Multilayer

        Args:
            pattern: :code:`Pattern` to add
            layer: Layer to incorporate :code:`Pattern`

        Returns:

        """
        self.pattern_to_layer.append((pattern, layer))
        self.layer_to_geoms, self.port = self._init_multilayer()

    def plot(self, ax, foundry: Foundry = FABLESS, alpha: float = 0.5):
        """Plot this device on a matplotlib plot.

        Args:
            ax: Matplotlib axis handle to plot the device.
            alpha: The transparency factor for the plot (to see overlay of structures from many layers).

        Returns:

        """
        for layer, pattern in self.layer_to_geoms.items():
            color = None
            for step in foundry.stack:
                if step.layer == layer:
                    color = step.mat.color
                    break
            if color is None:
                raise ValueError("The layer does not exist in the foundry stack, so could not find a color.")
            ax.add_patch(PolygonPatch(pattern, facecolor=color, edgecolor='none', alpha=alpha))
        b = self.bounds
        ax.set_xlim((b[0], b[2]))
        ax.set_ylim((b[1], b[3]))
        ax.set_aspect('equal')

    @property
    def size(self) -> Float2:
        """Size of the pattern

        Returns:
            Tuple of the form :code:`(sizex, sizey)`

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return b[2] - b[0], b[3] - b[1]  # (maxx - minx, maxy - miny)

    def trimesh(self, foundry: Foundry = FABLESS, exclude_layer: Optional[List[CommonLayer]] = None):
        return fabricate(self.layer_to_geoms, foundry, exclude_layer=exclude_layer)

    @classmethod
    def aggregate(cls, devices: List["Device"], name: Optional[str] = None):
        name = '|'.join([m.name for m in devices]) if name is None else name
        return cls(name, sum([m.pattern_to_layer for m in devices], []))


@fix_dataclass_init_docs
@dataclass
class Grating(Device):
    """Grating with partial etch.

    extent: Dimension of the extent of the grating
    pitch: float
    duty_cycle: The fill factor for the grating

    """
    extent: Float2
    pitch: float
    duty_cycle: float
    name: str = 'grating'

    def __post_init__(self):
        self.stripe_w = self.pitch * self.duty_cycle
        self.pitch = self.pitch
        self.duty_cycle = self.duty_cycle

        super(Grating, self).__init__(self.name,
                                      [(Box(self.extent), CommonLayer.RIB_SI),
                                       (Box(self.extent).striped(self.stripe_w, (self.pitch, 0)), CommonLayer.RIDGE_SI)])


@fix_dataclass_init_docs
@dataclass
class Via(Device):
    """Via / metal multilayer stack (currently all params should be specified to 2 decimal places)

    Attributes:
        via_extent: Dimensions of the via (or each via in via array).
        boundary_grow: Boundary growth around the via or via array.
        metal: Metal layer labels for the via (the thin metal layers)
        via: Via layer labels for the via (the actual tall vias)
        pitch: Pitch of the vias (center to center)
        shape: Shape of the array (rows, cols)
        name: Name of the via
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
        self.boundary_grow = self.boundary_grow if isinstance(self.boundary_grow, tuple)\
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
