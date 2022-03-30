from copy import deepcopy as copy

from .port import Port
from .transform import AffineTransform, rotate2d, translate2d, reflect2d, skew2d, scale2d
from .typing import Union, Float4, Float2, List, Dict, Optional, Tuple

import numpy as np


class Geometry:
    def __init__(self, geoms: List[np.ndarray], port: Dict[str, Port], refs: List["Geometry"],
                 tangents: List[np.ndarray] = None):
        self.geoms = geoms
        self.port = port
        self.refs = refs
        self.curve = None  # reserved for paths.
        self.tangents = [] if tangents is None else tangents

    @property
    def points(self) -> np.ndarray:
        return np.hstack(self.geoms) if len(self.geoms) > 0 else np.zeros((2, 0))

    @property
    def num_geoms(self) -> int:
        return len(self.geoms)

    @property
    def num_points(self) -> int:
        return len(self.points)

    @property
    def bounds(self) -> Float4:
        """Bounds of the geom.

        Returns:
            Tuple of the form :code:`(minx, miny, maxx, maxy)`

        """
        p = self.points
        if p.shape[1] > 0:
            return np.min(p[0]), np.min(p[1]), np.max(p[0]), np.max(p[1])
        else:
            return 0, 0, 0, 0

    @property
    def size(self) -> Float2:
        """Size of the geom.

        Returns:
            Tuple of the form :code:`(sizex, sizey)`.

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return b[2] - b[0], b[3] - b[1]  # (maxx - minx, maxy - miny)

    @property
    def bbox(self):
        """Opposing diagonal corners of the bounding box (bbox).

        Returns:
            A numpy array of bottom left and top right corners of bounding box (bbox).

        """
        return np.reshape(self.bounds, (2, 2)).T

    @property
    def center(self) -> Float2:
        """Center of the geom.

        Returns:
            Center for the component.

        """
        b = self.bounds  # (minx, miny, maxx, maxy)
        return (b[2] + b[0]) / 2, (b[3] + b[1]) / 2  # (avgx, avgy)

    def transform(self, transform: Union[AffineTransform, np.ndarray]):
        transformer = transform if isinstance(transform, AffineTransform) else AffineTransform(transform)
        if self.geoms:
            self.geoms = transformer.transform_geoms(self.geoms)
        if self.tangents:
            self.tangents = transformer.transform_geoms(self.tangents, tangents=True)
        self.port = {name: port.transform(transform) for name, port in self.port.items()}
        for geom in self.refs:
            geom.transform(transform)
        if self.curve:
            self.curve.transform(transform)
        return self

    @property
    def shapely(self):
        raise NotImplementedError("This needs to be implemented by subclasses of Geometry.")

    def set_port(self, port: Dict[str, Port]):
        self.port = port
        return self

    def translate(self, dx: float = 0, dy: float = 0) -> "Geometry":
        """Translate patter

        Args:
            dx: Displacement in x
            dy: Displacement in y

        Returns:
            The translated geom

        """
        return self.transform(translate2d((dx, dy)))

    def reflect(self, center: Float2 = (0, 0), horiz: bool = False) -> "Geometry":
        """Reflect the component across a center point (default (0, 0))

        Args:
            center: The center point about which to flip
            horiz: do horizontal flip, otherwise vertical flip

        Returns:
            Flipped geom

        """
        self.transform(reflect2d(center, horiz))
        for name, port in self.port.items():
            port.a = np.mod(port.a + 180, 360)  # temp fix... need to change how ports are represented
        return self

    def rotate(self, angle: float, origin: Union[Float2, np.ndarray] = (0, 0)) -> "Geometry":
        """Runs Shapely's rotate operation on the geometry about :code:`origin`.

        Args:
            angle: rotation angle in degrees.
            origin: origin of rotation.

        Returns:
            Rotated geom by :code:`angle` about :code:`origin`

        """
        if angle % 360 != 0:  # to save time, only rotate if the angle is nonzero
            a = angle * np.pi / 180
            self.transform(rotate2d(a, origin))
        return self

    def skew(self, xs: float = 0, ys: float = 0, origin: Optional[Float2] = None) -> "Geometry":
        """Affine skew operation on the geometry about :code:`origin`.

        Args:
            xs: x skew factor.
            ys: y skew factor.
            origin: origin of rotation (uses center of geom if :code:`None`).

        Returns:
            Rotated geom by :code:`angle` about :code:`origin`

        """
        return self.transform(skew2d((xs, ys), self.center if origin is None else origin))

    def scale(self, xfact: float = 1, yfact: float = None, origin: Optional[Float2] = None) -> "Geometry":
        """Affine scale operation on the geometry about :code:`origin`.

        Args:
            xfact: x scale factor.
            yfact: y scale factor (same as x scale factor if not specified).
            origin: origin of rotation (uses center of geom if :code:`None`).

        Returns:
            Rotated geom by :code:`angle` about :code:`origin`

        """
        yfact = xfact if yfact is None else yfact
        return self.transform(scale2d((xfact, yfact), self.center if origin is None else origin))

    def to(self, port: Union[Tuple[float, ...], Port] = (0, 0), from_port: Optional[Union[str, Port]] = None):
        port = Port(*port) if isinstance(port, tuple) or isinstance(port, np.ndarray) else port
        from_port = Port(*from_port) if isinstance(from_port, tuple) else from_port
        if from_port is None:
            return self.rotate(port.a).translate(port.x, port.y)
        else:
            fp = self.port[from_port] if isinstance(from_port, str) else from_port
            return self.rotate(port.a - fp.a + 180, origin=fp.xy).translate(port.x - fp.x, port.y - fp.y)

    def align(self, geom_or_center: Union["Geometry", Float2] = (0, 0),
              other: Union["Geometry", Float2] = None) -> "Geometry":
        """Align center of geom

        Args:
            geom_or_center: A geom (align to the geom's center) or a center point for alignment.
            other: If specified, instead of aligning based on this geom's center,
                align based on another geom's center and translate accordingly.

        Returns:
            Aligned geom

        """
        if other is None:
            old_x, old_y = self.center
        else:
            old_x, old_y = other.center if isinstance(other, Geometry) else other
        center = geom_or_center.center if isinstance(geom_or_center, Geometry) else geom_or_center
        self.translate(center[0] - old_x, center[1] - old_y)
        return self

    def halign(self, c: Union["Geometry", float], left: bool = True, opposite: bool = False) -> "Geometry":
        """Horizontal alignment of geom

        Args:
            c: A geom (horizontal align to the geom's boundary) or a center x for alignment
            left: (if :code:`c` is geom) Align to left boundary of component, otherwise right boundary
            opposite: (if :code:`c` is geom) Align opposite faces (left-right, right-left)

        Returns:
            Horizontally aligned geom

        """
        x = self.bounds[0] if left else self.bounds[2]
        p = c if np.isscalar(c) \
            else (c.bounds[0] if left and not opposite or opposite and not left else c.bounds[2])
        self.translate(dx=p - x)
        return self

    def valign(self, c: Union["Geometry", float], bottom: bool = True, opposite: bool = False) -> "Geometry":
        """Vertical alignment of geom

        Args:
            c: A geom (vertical align to the geom's boundary) or a center y for alignment
            bottom: (if :code:`c` is geom) Align to upper boundary of component, otherwise lower boundary
            opposite: (if :code:`c` is geom) Align opposite faces (upper-lower, lower-upper)

        Returns:
            Vertically aligned geom

        """
        y = self.bounds[1] if bottom else self.bounds[3]
        p = c if np.isscalar(c) \
            else (c.bounds[1] if bottom and not opposite or opposite and not bottom else c.bounds[3])
        self.translate(dy=p - y)
        return self

    def vstack(self, other_geom: "Geometry", bottom: bool = False) -> "Geometry":
        return self.align(other_geom).valign(other_geom, bottom=bottom, opposite=True)

    def hstack(self, other_geom: "Geometry", left: bool = False) -> "Geometry":
        return self.align(other_geom).halign(other_geom, left=left, opposite=True)

    @property
    def copy(self) -> "Geometry":
        """Copies the pattern using deepcopy.

        Returns:
            A copy of the Pattern so that changes do not propagate to the original :code:`Pattern`.

        """
        return Geometry(self.geoms, self.port, self.refs, self.tangents)

    def reverse(self) -> "Geometry":
        """Reverse the geometry and the direction of the tangents along the path.

        Returns:
            The reversed geometry.

        """
        self.geoms = [g[:, ::-1] for g in self.geoms[::-1]]
        self.tangents = [-t[:, ::-1] for t in self.tangents[::-1]]
        self.refs = [r.reverse() for r in self.refs]
        return self.flip_ends()

    def symmetrized(self, front_port: str = 'b0') -> "Geometry":
        """Symmetrize this curve across a mirror plane decided by one of the curves in the curve set.

        Args:
            front_port: Front port label.
            back_port: Back port label.

        Returns:
            The symmetrized curve

        """
        d = self.port[front_port].tangent()
        final_angle = np.arctan2(*d[::-1])
        mirror = AffineTransform([reflect2d(self.port[front_port].xy),
                                  rotate2d(2 * final_angle + np.pi, self.port[front_port].xy)]).transform
        mirrored = self.copy.transform(mirror).reverse()
        forward = self.copy
        forward.geoms += mirrored.geoms
        forward.tangents += mirrored.tangents
        forward.port[front_port] = mirrored.port[front_port].flip().copy
        forward.refs.extend(mirrored.refs)
        forward.curve = self.curve.symmetrized() if self.curve is not None else None
        return forward

    def flip_ends(self, front_port: str = 'b0', back_port: str = 'a0'):
        self.port = {
            front_port: self.port[back_port],
            back_port: self.port[front_port]
        }
        return self

    def __hash__(self):
        return hash(id(self.shapely.wkt))

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
