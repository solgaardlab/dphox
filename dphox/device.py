import datetime
import hashlib
from collections import defaultdict
from copy import deepcopy, deepcopy as copy
from dataclasses import dataclass
from typing import BinaryIO, List, Optional, Dict

import numpy as np

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
import networkx as nx

from .foundry import CommonLayer, DEFAULT_FOUNDRY, Foundry

from .geometry import Geometry
from .pattern import Box, Pattern, Port
from .transform import GDSTransform
from .typing import Float2, Int2, Tuple, Union
from .utils import fix_dataclass_init_docs, min_aspect_bounds, poly_bounds, poly_points, PORT_GDS_LABEL, PORT_LAYER, \
    shapely_patch

try:
    NAZCA_IMPORTED = True
    import nazca as nd
except ImportError:
    NAZCA_IMPORTED = False

KLAMATH_IMPORTED = True
try:
    import klamath
    from klamath.library import FileHeader
except ImportError:
    KLAMATH_IMPORTED = False

GDSTransformOrTuple = Union[GDSTransform, Tuple, np.ndarray, Port, List[Port]]


class Device:
    """A :code:`Device` defines a device (active or passive) in a GDS.

    A :code:`Device` is a core object in DPhox, which enables composition of multiple :code:`Pattern` or
    :code:`Polygon` mapped to process stack layers into a single :code:`Device`. Once the process is defined,
    :code:`Trimesh` may be used to show a full 3D model of the device. Additionally, a :code:`Device` can be mapped
    to a GDSPY or Nazca cell.

    A :code:`Device` is more or less the same as a GDS cell, including hierarchical design and transform data for
    subcells in the hierarchy.


    Attributes:
        name: Name of the device. This name can be anything you choose, ideally not repeating
            any previously defined :code:`Device`.
        devices: A list of devices, either a :code:`Device` OR a :code:`Pattern` or :code:`PolygonLike` followed by
            a layer label (integer or string). None of these devices should have existing children (will throw
            and error if they do, since these should be added via `place` not this contructor).
        child_to_device: A map between child names and actual devices (cells)
        child_to_transform: A map between child names and the GDS-formatted transforms (used for fast GDS generation)
    """

    def __init__(self, name: str, devices: List[Union[Tuple[Pattern, Union[int, str]], "Device"]] = None):
        self.name = name
        self.pattern_to_layer = [] if devices is None else devices
        self.pattern_to_layer = sum(
            (p.pattern_to_layer if isinstance(p, Device) else [p] for p in self.pattern_to_layer), [])

        self.layer_to_polys = self._update_layers()
        self.child_to_device: Dict[str, Device] = {}  # children in this dictionary
        self.child_to_transform: Dict[str, GDSTransform] = {}  # dictionary from child name to transform

        # check to see if the original devices
        if devices is not None:
            for p in devices:
                if isinstance(p, Device) and p.child_to_device != {}:
                    raise AttributeError(f'Device {p.name} has child devices {p.child_to_device}'
                                         f'and should be added via `place` method.')
        self.port = {}

    def _update_layers(self) -> Dict[Union[int, str], List[np.ndarray]]:
        layer_to_polys = defaultdict(list)
        for pattern, layer in self.pattern_to_layer:
            layer_to_polys[layer].extend(pattern.geoms)
        return layer_to_polys

    def merge_patterns(self):
        self.pattern_to_layer = [(Pattern(*self.layer_to_polys[layer]), layer) for layer in self.layer_to_polys]
        return self

    @classmethod
    def from_pattern(cls, pattern: Pattern, name: str, layer: str):
        """A class method to convert a :code:`Pattern` into a :code:`Device`.

        Note:
            The ports from pattern is "raised" to the pattern in ths device.

        Args:
            pattern: The pattern that is being used to generate the device.
            name: Name for the component.
            layer: The layer for the pattern.

        Returns:
            The :code:`Device` containing the :code:`Pattern` at the specified :code:`layer`.

        """
        device = cls(name, [(pattern, layer)])
        device.port = pattern.port
        return device

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
        multilayers = defaultdict(list)
        for named_tuple in nd.cell_iter(cell, flat=True):
            if named_tuple.cell_start:
                for polygon, points, bbox in named_tuple.iters['polygon']:
                    if polygon.layer == 'bb_pin':
                        # TODO(sunil): actually extract the pins from this layer.
                        continue
                    # fixing point definitions from mask to 1nm precision,
                    # kinda hacky but is physical and prevents false polygons
                    multilayers[polygon.layer].append(Pattern(Polygon(np.around(points, decimals=3))))
        return cls(cell.name, [(Pattern(*pattern_list), layer) for layer, pattern_list in multilayers.items()])

    @property
    def bounds(self) -> np.ndarray:
        """Bounding box of the form :code:`(minx, miny, maxx, maxy)`

        Returns:
            Bounding box ndarray.

        """
        bound_list = [np.reshape(p.bounds, (2, 2)).T for p, _ in self.pattern_to_layer]
        bound_list = np.hstack(bound_list) if bound_list else np.array([[], []])
        child_bboxes = []
        for child_name, child in self.child_to_device.items():
            child_bboxes.extend(self.child_to_transform[child_name][0].transform_points(child.bbox))
        child_bboxes = np.hstack(child_bboxes) if child_bboxes else np.array([[], []])
        bound_list = np.hstack([bound_list, child_bboxes])
        if bound_list.size > 0:
            return np.array((np.min(bound_list[0]), np.min(bound_list[1]),
                             np.max(bound_list[0]), np.max(bound_list[1])))
        else:
            return np.array((0, 0, 0, 0))

    @property
    def layer_to_pattern(self):
        return {layer: Pattern(p) for layer, p in self.layer_to_polys.items()}

    @property
    def full_layer_to_pattern(self):
        return {layer: Pattern(p) for layer, p in self.full_layer_to_polys.items()}

    @property
    def tree(self):
        """Tree / hierarchical representation of the device.

        Returns:
            A dictionary containing the tree representation of the device (ignoring transforms).

        """
        return {
            self.name: {
                name: child.tree for name, child in self.child_to_device
            }
        }

    def __hash__(self):
        return hashlib.sha256(
            {
                self.name: {
                    layer: p.__hash__() for layer, p in self.layer_to_pattern
                },
                f'{self.name}_children': {
                    name: child.__hash__() + hashlib.sha256(self.child_to_transform[name][0])
                    for name, child in self.child_to_device
                }
            }
        )

    @property
    def hash(self):
        return self.__hash__()

    @property
    def bbox(self) -> np.ndarray:
        """

        Returns:
            The linestring along the diagonal of the bbox

        """
        return np.reshape(self.bounds, (2, 2)).T

    @property
    def bbox_pattern(self) -> Box:
        """

        Returns:
            The linestring along the diagonal of the bbox

        """
        bbox = Box(self.size).align(self.center)
        bbox.port = self.port_copy
        return bbox

    @property
    def dummy_port_pattern(self) -> Box:
        """

        Returns:
            The linestring along the diagonal of the bbox

        """
        return Pattern().set_port(self.port_copy)

    @property
    def center(self) -> Float2:
        """Center (x, y) for the device.

        Returns:
            Center for the component.

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return (b[2] + b[0]) / 2, (b[3] + b[1]) / 2  # (avgx, avgy)

    def align(self, c: Union[Geometry, "Device", Float2] = (0, 0)) -> "Device":
        """Align center of pattern

        Args:
            c: A pattern (align to the pattern's center) or a center point for alignment.

        Returns:
            Aligned pattern

        """
        old_x, old_y = self.center
        center = c.center if isinstance(c, (Device, Pattern)) else c
        self.translate(center[0] - old_x, center[1] - old_y)
        return self

    def halign(self, c: Union[Geometry, "Device", float] = 0, left: bool = True, opposite: bool = False) -> "Device":
        """Horizontal alignment of device

        Args:
            c: A device (horizontal align to the device's boundary) or a center x for alignment.
            left: (if :code:`c` is pattern) Align to left boundary of component, otherwise right boundary.
            opposite: (if :code:`c` is pattern) Align opposite faces (left-right, right-left).

        Returns:
            Horizontally aligned device

        """
        x = self.bounds[0] if left else self.bounds[2]
        p = c if np.isscalar(c) \
            else (c.bounds[0] if left and not opposite or opposite and not left else c.bounds[2])
        self.translate(dx=p - x)
        return self

    def valign(self, c: Union[Geometry, "Device", float] = 0, bottom: bool = True, opposite: bool = False) -> "Device":
        """Vertical alignment of device.

        Args:
            c: A pattern (vertical align to the pattern's boundary) or a center y for alignment.
            bottom: (if :code:`c` is pattern) Align to upper boundary of component, otherwise lower boundary.
            opposite: (if :code:`c` is pattern) Align opposite faces (upper-lower, lower-upper).

        Returns:
            Vertically aligned device

        """
        y = self.bounds[1] if bottom else self.bounds[3]
        p = c if np.isscalar(c) \
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
        if self.child_to_device:
            raise NotImplementedError("We do not yet support transforming cells with children."
                                      "This should in principle not be required though: you can place this cell"
                                      "in a parent cell, i.e. dp.Device('transformed').place(device, transform).")
        for pattern, _ in self.pattern_to_layer:
            pattern.translate(dx, dy)
        self.layer_to_polys = self._update_layers()
        for name, port in self.port.items():
            self.port[name] = Port(port.x + dx, port.y + dy, port.a, port.w)
        return self

    def rotate(self, angle: float, origin: Float2 = (0, 0)) -> "Device":
        """Rotate the device by rotating all the patterns within it.

        Args:
            angle: rotation angle in degrees.
            origin: origin of rotation.

        Returns:
            The rotated device.

        """
        if self.child_to_device:
            raise NotImplementedError("We do not yet support transforming cells with children."
                                      "This should in principle not be required though: you can place this cell"
                                      "in a parent cell, i.e. dp.Device('transformed').place(device, transform).")
        if angle % 360 != 0:  # to save time, only rotate if the angle is nonzero
            for pattern, _ in self.pattern_to_layer:
                pattern.rotate(angle, origin)
            self.layer_to_polys = self._update_layers()
            self.port = {name: port.rotate(angle, origin) for name, port in self.port.items()}
        return self

    def reflect(self, center: Float2 = (0, 0), horiz: bool = False) -> "Device":
        """Reflected the multilayer about center (vertical, or about x-axis, by default)

        Args:
            center: center about which to reflect.
            horiz: reflect horizontally instead of vertically.

        Returns:
            Reflected pattern.

        """
        if self.child_to_device:
            raise NotImplementedError("We do not yet support transforming cells with children."
                                      "This should in principle not be required though: you can place this cell"
                                      "in a parent cell, i.e. dp.Device('transformed').place(device, transform).")
        for pattern, _ in self.pattern_to_layer:
            pattern.reflect(center, horiz)
        self.layer_to_polys = self._update_layers()
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

    def place(self, device: "Device", placement: Union[GDSTransformOrTuple],
              from_port: Optional[Union[str, Port, tuple]] = None, flip_y: bool = False, return_ports: bool = False):
        """Place another device into this device at a specified :code:`placement` (location and angle) or list of them.

        Args:
            device: The child device to place.
            placement: The transform to apply OR the port to which the device should be connected.
                Note that the transform to apply can be the form of an xya (x, y, angle) tuple or a port.
            from_port: The port name corresponding to child device's port that should be connected according to
                :code:`placement`. This effectively acts like a reference position/orientation for placement.
            flip_y: Whether to flip the design vertically (to be applied to all GDS transforms,
                UNLESS placement is a GDS transform, in which case this arg is ignored)
            return_ports: Return the ports of the placed device according to the format of placement.

        Returns:
            The ports of the placed device, according to the specification of :code:`return_ports`.

        """
        if device.name not in self.child_to_device:
            self.child_to_device[device.name] = device

        if not isinstance(placement, list) and not isinstance(placement, tuple) and not isinstance(placement,
                                                                                                   np.ndarray):
            placement = [placement]

        from_port = Port(0, 0, 180) if from_port is None else from_port

        if isinstance(from_port, str):
            from_port = device.port[from_port]
        elif isinstance(from_port, tuple):
            from_port = Port(*from_port)
        elif not isinstance(from_port, Port):
            raise TypeError(f"from_port must be str, tuple or Port type but got {type(from_port)}.")

        # synthesize placement and from_port into parseable GDS transform table
        transform = []
        for p in placement:
            if isinstance(p, Port):
                transform.append(from_port.orient_xyaf(p.xya, flip_y=flip_y))
            elif isinstance(p, GDSTransform):
                p.set_xyaf(from_port.orient_xyaf(p.xya))  # if GDS transform is supplied, ignore the flip_y argument
                transform.append(p)
            else:
                transform.append(from_port.orient_xyaf(p, flip_y=flip_y))
        self.child_to_transform[device.name] = GDSTransform.parse(transform, self.child_to_transform.get(device.name))
        if return_ports:
            placed_ports = [{pname: port.copy.transform_xyaf(t) for pname, port in device.port.items()} for t in
                            transform]

            return placed_ports[0] if len(placed_ports) == 1 else placed_ports

    def clear(self, device: Union[str, "Device"]):
        """Clear the device or its name from the children of this device. Do nothing if the device isn't present.

        Args:
            device: Child device to remove

        Returns:
            This device.

        """
        name = device.name if isinstance(device, Device) else device
        if name in self.child_to_device:
            del self.child_to_device[name]
            del self.child_to_transform[name]
        return self

    @property
    def copy(self) -> "Device":
        """Return a copy of this device for repeated use.

        Returns:
            A deep copy of this device.

        """
        return copy(self)

    @property
    def gdspy_cell(self, foundry: Foundry = DEFAULT_FOUNDRY):
        """Turn this multilayer into a gdspy cell.

        Args:
            Foundry for creating the gdspy cell (provide the layer map).

        Returns:
            A GDSPY cell.

        """
        import gdspy as gy
        cell = gy.Cell(self.name)
        for pattern, layer in self.pattern_to_layer:
            for poly in self.layer_to_polys[layer]:
                cell.add(
                    gy.Polygon(poly,
                               layer=foundry.layer_to_gds_label[layer][0],
                               datatype=foundry.layer_to_gds_label[layer][1])
                )
        return cell

    def nazca_cell(self, foundry: Foundry = DEFAULT_FOUNDRY) -> "nd.Cell":
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
                for poly in pattern.geoms:
                    nd.Polygon(points=poly,
                               layer=foundry.layer_to_gds_label[layer]).to()
            for name, port in self.port.items():
                nd.Pin(name).to(*port.xya)
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
        self.layer_to_polys = self._update_layers()

    def plot(self, ax: Optional = None, foundry: Foundry = DEFAULT_FOUNDRY,
             exclude_layer: Optional[List[CommonLayer]] = None, alpha: float = 0.5,
             plot_ports: bool = False):
        """Plot this device on a matplotlib plot.

        Args:
            ax: Matplotlib axis handle to plot the device.
            foundry: :code:`Foundry` object for matplotlib plotting.
            exclude_layer: Exclude all layers in this list when plotting.
            alpha: The transparency factor for the plot (to see overlay of structures from many layers).

        Returns:

        """
        # import locally since this import takes some time.
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        exclude_layer = [] if exclude_layer is None else exclude_layer
        for layer, multipoly in self.full_layer_to_polys.items():
            if layer in exclude_layer:
                continue
            color = foundry.color(layer)
            if color is None:
                raise ValueError(f"The layer {layer} does not exist in the foundry stack,"
                                 f"so could not find a color. Make sure you specify the correct foundry.")
            pattern = Pattern(*multipoly)
            if len(pattern.geoms) > 0:
                ax.add_patch(
                    shapely_patch(unary_union(MultiPolygon(pattern.shapely)),
                                  # union avoids some weird plotting with alpha < 1
                                  facecolor=color, edgecolor='none', alpha=alpha))
            if plot_ports:
                for name, port in self.port.items():
                    port_xy = port.xy - port.tangent(port.w)
                    ax.add_patch(shapely_patch(port.shapely,
                                               facecolor='red', edgecolor='none', alpha=alpha))
                    ax.text(*port_xy, name)
        b = min_aspect_bounds(self.bounds)
        ax.set_xlim((b[0], b[2]))
        ax.set_ylim((b[1], b[3]))
        ax.set_xlabel(r'$x$ ($\mu$m)')
        ax.set_ylabel(r'$y$ ($\mu$m)')
        ax.set_aspect('equal')

    def hvplot(self, foundry: Foundry = DEFAULT_FOUNDRY, exclude_layer: Optional[List[CommonLayer]] = None, alpha: float = 0.5):
        """Plot this device on a holoviews plot.

        Args:
            foundry: :code:`Foundry` object for holoviews plotting.
            exclude_layer: Exclude all layers in this list when plotting.
            alpha: The transparency factor for the plot (to see overlay of structures from many layers).

        Returns:
            The holoviews Overlay for displaying all of the polygons.

        """
        # import locally since this import takes a while to import globally.
        import holoviews as hv
        b = min_aspect_bounds(self.bounds)
        plots_to_overlay = []
        exclude_layer = [] if exclude_layer is None else exclude_layer
        for layer, multipoly in self.full_layer_to_polys.items():
            if layer in exclude_layer:
                continue
            color = foundry.color(layer)
            if color is None:
                raise ValueError(f"The layer {layer} does not exist in the foundry stack, so could not find a color."
                                 f"Make sure you specify the correct foundry.")
            pattern = Pattern(*multipoly)
            plots_to_overlay.append(pattern.hvplot(color, layer, alpha, plot_ports=False))
        plots_to_overlay.extend(port.hvplot(name) for name, port in self.port.items())
        return hv.Overlay(plots_to_overlay).opts(show_legend=True, shared_axes=False,
                                                 ylim=(b[1], b[3]), xlim=(b[0], b[2]))

    @property
    def full_layer_to_polys(self):
        """Using breadth-first search of children, get all polygons in this device (useful for plotting purposes).

        Returns:
            The full layer to polygon representation of this device hierarchy (call this on-demand).

        """
        layer_to_polys = deepcopy(self.layer_to_polys)
        for child_name, child in self.child_to_device.items():
            # this recursively goes down the child tree
            for layer, multipoly in child.full_layer_to_polys.items():
                if len(multipoly) > 0:
                    polygons = self.child_to_transform[child_name][0].transform_geoms(multipoly)
                    new_polys = sum((np.split(p, p.shape[0]) if p.ndim == 3 else [p] for p in polygons), [])

                    new_polys = [np.squeeze(p) for p in new_polys]
                    layer_to_polys[layer].extend(new_polys)
        return layer_to_polys

    @property
    def size(self) -> Float2:
        """Size of the pattern.

        Returns:
            Tuple of the form :code:`(sizex, sizey)`.

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return b[2] - b[0], b[3] - b[1]  # (maxx - minx, maxy - miny)

    def trimesh(self, foundry: Foundry = DEFAULT_FOUNDRY, exclude_layer: Optional[List[CommonLayer]] = None):
        """Fabricate this device based on a :code:`Foundry`.

        This method is fairly rudimentary and will not implement things like conformal deposition. At the moment,
        you can implement things like rib etches which can be determined using 2d shape operations. Depositions in
        layers above etched layers will just start from the maximum z extent of the previous layer. This is specified
        by the :code:`Foundry` stack.

        Args:
            foundry: The foundry for each layer which converts it into a trimesh
            exclude_layer: Exclude all layers in this list.

        Returns:
            The device :code:`Scene` to visualize.

        """
        return foundry.fabricate(
            layer_to_geom={layer: MultiPolygon([Polygon(p.T) for p in polys])
                           for layer, polys in self.full_layer_to_polys.items()},
            exclude_layer=exclude_layer
        )

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
        return cls(name, sum((m.pattern_to_layer for m in devices), []))

    def smooth_layer(self, distance: float, layer: str):
        """Smooth a layer in the device (useful for sharp corners that may appear)

        Args:
            distance: Smooth the device by some distance.
            layer: Layer to smooth

        Returns:
            Device with smoothed layer.

        """
        for pattern, _layer in self.pattern_to_layer:
            if _layer == layer:
                pattern.smooth(distance)
        self.layer_to_polys = self._update_layers()
        return self

    @property
    def port_copy(self):
        """The copy of ports in this device.

        Note:
            Whenever using the ports of a given geometry in another geometry, it is highly recommended
            to extract :code:`port_copy`, which creates fresh copies of the ports.

        Returns:
            The port dictionary copy.

        """
        return {name: p.copy for name, p in self.port.items()}

    def set_port(self, port: Dict[str, Port]):
        self.port = port
        return self

    def _pdk_ports(self, labels: List["klamath.elements.Text"], foundry: Foundry,
                   user_units_per_db_unit: float):
        """A helper function for reading a standard PDK layout's labelled ports from a GDS file."""
        port = {}
        port_boxes = {port_layer: poly_bounds(self.layer_to_polys[port_layer])
                      for port_layer in foundry.port_layers if self.layer_to_polys[port_layer]}
        overall_bounds = self.bounds
        port_angle_options = (180, -90, 0, 90)
        # find the port closest to the labels
        for label in labels:
            layer = foundry.gds_label_to_layer.get(label.layer, CommonLayer.PORT)
            if foundry.use_port_boxes:
                bbox = None
                for port_layer in foundry.port_layers:
                    if label.layer[0] == foundry.layer_to_gds_label[port_layer][0] and port_layer in port_boxes:
                        bbox = min(port_boxes[port_layer],
                                   key=lambda p: np.linalg.norm(
                                       (p[:2] + p[2:]) / 2 / user_units_per_db_unit - label.xy))
                if bbox is not None:
                    side = np.argmin(np.abs(np.array(bbox) - np.array(overall_bounds)))
                    side = side[0] if isinstance(side, np.ndarray) else side
                    angle = port_angle_options[side]
                    width = bbox[2] - bbox[0] if side % 2 else bbox[3] - bbox[1]
                    x = (bbox[0] + bbox[2]) / 2 if side % 2 else bbox[side]
                    y = bbox[side] if side % 2 else (bbox[1] + bbox[3]) / 2
                    port[label.string.decode("utf-8")] = Port(x, y, angle, width, layer=layer)
            else:
                x, y = label.xy.T.squeeze() * user_units_per_db_unit
                side = np.argmin(np.abs(np.array((x, y, x, y)) - np.array(overall_bounds)))
                side = int(side[0]) if isinstance(side, np.ndarray) else side
                angle = port_angle_options[side]
                port[label.string.decode("utf-8")] = Port(x, y, angle, layer=layer)
        return port

    @classmethod
    def from_gds(cls, filepath: str, foundry: Foundry = DEFAULT_FOUNDRY) -> Dict[str, "Device"]:
        """Generate non-annotated device from GDS file based on the provided layer map in :code:`foundry`.

        Args:
            filepath: The filepath to read the GDS device.
            foundry: The foundry to get the device.
            name: The name of the device (top cell) from the GDS file, if None return a dictionary

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
            cells = {}
            references = {}

            for key, value in structs.items():
                labels = []
                name = key.decode("utf-8")
                references[name] = []
                pattern_to_layer = []
                for obj in value:
                    if isinstance(obj, klamath.elements.Boundary):
                        if obj.layer in foundry.gds_label_to_layer:
                            layer = foundry.gds_label_to_layer[obj.layer]  # klamath layer is gds_label in dphox
                            poly = obj.xy.T
                            if poly.size > 0:
                                pattern = Pattern(header.user_units_per_db_unit * poly)
                                pattern_to_layer.append((pattern, layer))
                    elif isinstance(obj, klamath.elements.Text):
                        labels.append(obj)
                    elif isinstance(obj, klamath.elements.Reference):
                        x, y = obj.xy[0] * header.user_units_per_db_unit
                        references[name].append((obj.struct_name.decode("utf-8"),
                                                 GDSTransform(x, y, obj.angle_deg, obj.invert_y, obj.mag)))
                cell = cls(name, pattern_to_layer)
                cell.port = cell._pdk_ports(labels, foundry, header.user_units_per_db_unit)
                cells[name] = cell
            net = nx.DiGraph([(r, s[0]) for r in references for s in references[r]])
            topo_sorted_cells = list(nx.topological_sort(net))
            for node in topo_sorted_cells[::-1]:
                for cell_name, ref in references[node]:
                    cells[node].place(cells[cell_name], ref)
        return cells

    def gds_elements(self, foundry: Foundry = DEFAULT_FOUNDRY, user_units_per_db_unit: float = 0.001):
        """Use `klamath <https://mpxd.net/code/jan/klamath/src/branch/master/klamath>`_ to convert to GDS elements
        using a foundry object and user units.

        Args:
            foundry: The foundry used for the layer map.
            user_units_per_db_unit: User units per unit (to convert from nm to um, need to use 0.001 to convert).

        Returns:
            The `klamath` GDS elements for the device.
        """
        if not KLAMATH_IMPORTED:
            raise ImportError('Klamath not imported, need klamath for GDS export.')
        elements = []
        for layer, geom in self.layer_to_polys.items():
            elements += [
                klamath.elements.Boundary(
                    layer=foundry.layer_to_gds_label[layer],
                    xy=(poly.T / user_units_per_db_unit).astype(np.int32),
                    properties={}) for poly in geom
            ]
        for name, port in self.port.items():
            elements += [
                klamath.elements.Text(layer=foundry.layer_to_gds_label.get(port.layer, PORT_GDS_LABEL),
                                      xy=port.xy / user_units_per_db_unit,
                                      string=name.encode('utf-8'), properties={},
                                      presentation=0, angle_deg=0, invert_y=False, width=0, path_type=0, mag=1),
                klamath.elements.Boundary(
                    layer=(PORT_LAYER, 0),
                    xy=(poly_points(port.shapely) / user_units_per_db_unit).astype(np.int32),
                    properties={})
            ]
        return elements

    def to_gds_stream(self, stream: BinaryIO, foundry: Foundry = DEFAULT_FOUNDRY, user_units_per_db_unit: float = 0.001):
        """Use `klamath <https://mpxd.net/code/jan/klamath/src/branch/master/klamath>`_ to add device struct to GDS.

        Args:
            stream: Stream for a GDS file.
            foundry: The foundry used for the layer map.
            user_units_per_db_unit: User units per unit (to convert from nm to um, need to use 0.001 to convert).

        Returns:

        """
        elements = self.gds_elements(foundry, user_units_per_db_unit)
        for child_name, child in self.child_to_device.items():
            child.to_gds_stream(stream, foundry, user_units_per_db_unit)
            for gds_transform in self.child_to_transform[child_name][1]:
                xy = (np.array([[gds_transform.x, gds_transform.y]]) / user_units_per_db_unit).astype(np.int32)
                elements.append(
                    klamath.elements.Reference(
                        struct_name=child.name.encode('utf-8'),
                        xy=xy, angle_deg=gds_transform.angle,
                        mag=gds_transform.mag,
                        invert_y=gds_transform.flip_y,
                        colrow=None,
                        properties={}
                    )
                )
        klamath.library.write_struct(stream, self.name.encode('utf-8'), elements)

    def to_gds(self, filepath: str, foundry: Foundry = DEFAULT_FOUNDRY, user_units_per_db_unit: float = 0.001,
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
        if not KLAMATH_IMPORTED:
            raise ImportError('Klamath not imported, need klamath for GDS export.')
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
    decimals: int = 2
    name: str = 'via'

    def __post_init__(self):
        self.metal = (self.metal,) if isinstance(self.metal, str) else self.metal
        self.boundary_grow = self.boundary_grow if isinstance(self.boundary_grow, tuple) \
            else tuple([self.boundary_grow] * len(self.metal))

        max_boundary_grow = max(self.boundary_grow)
        via_pattern = Box(self.via_extent, decimals=self.decimals)
        if self.pitch > 0 and self.shape is not None:
            x, y = np.meshgrid(np.arange(self.shape[0]) * self.pitch, np.arange(self.shape[1]) * self.pitch)
            patterns = [via_pattern.copy.translate(x, y) for x, y in zip(x.flatten(), y.flatten())]

            via_pattern = Pattern(*patterns, decimals=self.decimals)
        boundary = Box((via_pattern.size[0] + 2 * max_boundary_grow,
                        via_pattern.size[1] + 2 * max_boundary_grow), decimals=2).align((0, 0)).halign(0)
        via_pattern.align(boundary)
        layers = []
        if isinstance(self.via, list):
            layers += [(via_pattern.copy, layer) for layer in self.via]
        elif isinstance(self.via, str):
            layers += [(via_pattern, self.via)]
        layers += [(Box((via_pattern.size[0] + 2 * bg,
                         via_pattern.size[1] + 2 * bg), decimals=2).align(boundary), layer)
                   for layer, bg in zip(self.metal, self.boundary_grow)]
        super(Via, self).__init__(self.name, layers)
        self.port = {
            'w': Port(self.bounds[0], self.center[1], -180),
            'e': Port(self.bounds[2], self.center[1]),
            's': Port(self.center[0], self.bounds[1], -90),
            'n': Port(self.center[0], self.bounds[3], 90),
            'c': Port(*self.center)
        }
