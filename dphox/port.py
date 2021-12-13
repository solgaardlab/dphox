import pickle

import numpy as np
from pydantic.dataclasses import dataclass
from shapely.geometry import LineString

from .typing import Polygon
from .utils import fix_dataclass_init_docs, poly_points
from .transform import AffineTransform, rotate2d, translate2d, GDSTransform

try:
    HOLOVIEWS_IMPORTED = True
    import holoviews as hv
    from holoviews import opts
    import panel as pn
except ImportError:
    HOLOVIEWS_IMPORTED = False


@fix_dataclass_init_docs
@dataclass
class Port:
    """Port used in components in DPhox

    A port defines the center, width and angle/orientation of a port in a design. Note that ports always are considered
    to have a width, and if a width is not provided the width is assumed to be 1 in the units of the file.

    Attributes:
        x: x position of the port.
        y: y position of the port.
        w: width of the port
        a: Angle (orientation) of the port (in degrees).
        z: z position of the port (optional, not specified in design, mostly used for simulation).
        h: the height / thickness of the port (optional, not specified in design, mostly used for simulation).
    """
    x: float = 0
    y: float = 0
    a: float = 0
    w: float = 1
    z: float = 0
    h: float = 0

    def __post_init_post_parse__(self):
        self.xy = np.array((self.x, self.y))
        self.xya = np.array((self.x, self.y, self.a))
        self.center = np.array((self.x, self.y, self.z))

    @property
    def size(self):
        """Get the size of the :code:`Port` for simulation-related applications.

        Returns:
            The size of the Port in 3D space, i.e., (x, y, z).

        """
        if np.mod(self.a, 90) != 0:
            raise ValueError(f"Require angle to be a multiple a multiple of 90 but got {self.a}")
        return np.array((self.w, 0, self.h)) if np.mod(self.a, 180) != 0 else np.array((0, self.w, self.h))

    @property
    def shapely(self) -> Polygon:
        """Return the :code:`Polygon` triangle corresponding to the :code:`Port`.

        Based on center and orientation of the :code:`Port`, return the corresponding Shapely triangle.
        This is effectively the inverse of the :code:`from_shapely` classmethod of this class.

        Returns:
            The shapely :code:`Polygon` triangle represented by the :code:`Port`.

        """
        dx, dy = np.sin(self.a * np.pi / 180) * self.w / 2, np.cos(self.a * np.pi / 180) * self.w / 2
        return Polygon(
            [(self.x - dx - dy * 0.75, self.y - dy - dx * 0.75), (self.x + dx - dy * 0.75, self.y + dy - dx * 0.75),
             (self.x, self.y)])

    @classmethod
    def from_shapely(cls, triangle: Polygon, z: float = 0, h: float = 0) -> "Port":
        """Initialize a :code:`Port` using a :code:`LineString` in Shapely.

        The port can be unambiguously defined using a line. The Shapely :code:`Polygon` triangle defines the
        center :math:`x, y` of the line as well as the width :math:`w` of the port. This is effectively the
        inverse of the :code:`shapely` property of this class.

        Args:
            triangle: Triangle representing the port.
            z: The z position of the port.
            h: The height / thickness of the port.

        Returns:
            The :code:`Port` represented by the shapely :code:`Polygon` triangle.

        """
        if not isinstance(triangle, Polygon):
            raise TypeError(f'Input line must be a shapely Polygon but got {type(triangle)}')

        points = poly_points(triangle)
        first, second, port_point, _ = points
        x, y = port_point
        c = (second[1] - first[1]) + (second[0] - first[0]) * 1j
        a = np.angle(c) * 180 / np.pi
        return cls(x, y, a, np.abs(c), z, h)

    @classmethod
    def from_linestring(cls, linestring: LineString, z: float = 0, h: float = 0) -> "Port":
        """Initialize a :code:`Port` using a :code:`LineString` in Shapely.

        The port can be unambiguously defined using a line. The Shapely :code:`Polygon` triangle defines the
        center :math:`x, y` of the line as well as the width :math:`w` of the port. This is effectively the
        inverse of the :code:`shapely` property of this class.

        Args:
            linestring: Shapely :code:`LineString` representing the port.
            z: The z position of the port.
            h: The height / thickness of the port.

        Returns:
            The :code:`Port` represented by the shapely :code:`Polygon` triangle.

        """
        if not isinstance(linestring, LineString):
            raise TypeError(f'Input line must be a shapely LineString but got {type(linestring)}')

        first, second = linestring.coords
        c, = linestring.centroid.coords
        d = (second[1] - first[1]) + (second[0] - first[0]) * 1j
        a = -np.angle(d) * 180 / np.pi
        return cls(c[0], c[1], a, np.abs(d), z, h)

    def normal(self, scale: float = 1):
        """The vector normal to the port

        Args:
            scale: The magnitude of the normal vector

        Returns:
            Return the vector normal to the port

        """
        return np.array([np.cos(self.a * np.pi / 180), np.sin(self.a * np.pi / 180)]) * scale

    def transformed_line(self, transform: np.ndarray):
        dx, dy = -np.sin(self.a * np.pi / 180) * self.w / 2, np.cos(self.a * np.pi / 180) * self.w / 2
        x, y = self.xy
        return np.array(transform[..., :2, :]) @ np.array([[x - dx, y - dy, 1], [x + dx, y + dy, 1]]).T

    def transform(self, transform: np.ndarray):
        line = self.transformed_line(transform).T
        c = (line[0] + line[1]) / 2
        d = (line[1, 1] - line[0, 1]) + (line[1, 0] - line[0, 0]) * 1j
        self.a = -np.angle(d) * 180 / np.pi
        self.x = c[0]
        self.y = c[1]
        self.w = np.abs(d)
        self.xy = np.array((self.x, self.y))
        self.xya = np.array((self.x, self.y, self.a))
        self.center = np.array((self.x, self.y, self.z))
        return self

    def hvplot(self, name: str = 'port'):
        x, y = self.shapely.exterior.coords.xy
        port_xy = self.xy - self.normal(self.w)
        return hv.Polygons([{'x': x, 'y': y}]).opts(
            data_aspect=1, frame_height=200, color='red', line_alpha=0) * hv.Text(*port_xy, name)

    @property
    def copy(self) -> "Port":
        """Return a copy of this port for repeated use.

        Returns:
            A deep copy of this port.

        """
        return pickle.loads(pickle.dumps(self))


def port_transform(to_port: Port, from_port: Port = None) -> AffineTransform:
    if from_port is None:
        return AffineTransform((rotate2d(to_port.a), translate2d(to_port.xy)))
    else:
        return AffineTransform((rotate2d(to_port.a - from_port.a + 180, origin=from_port.xy),
                                translate2d(to_port.xy - from_port.xy)))


def port_gds_transform(to_port: Port, from_port: Port = None):
    if from_port is None:
        return GDSTransform(*to_port.xy, to_port.a)
    else:
        xy = to_port.xy - from_port.xy
        angle = to_port.a - from_port.a + 180
        xy -= AffineTransform(rotate2d(angle)).transform_points(from_port.xy)
        return GDSTransform(*xy, angle)
