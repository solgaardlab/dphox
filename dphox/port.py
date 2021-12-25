from copy import deepcopy as copy
from dataclasses import dataclass

import numpy as np

from .transform import AffineTransform, GDSTransform, rotate2d, translate2d
from .typing import Polygon
from .utils import DECIMALS, fix_dataclass_init_docs


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

    def __post_init__(self):
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
        dx, dy = -np.sin(self.a * np.pi / 180) * self.w / 2, np.cos(self.a * np.pi / 180) * self.w / 2
        n = -self.tangent(self.w / 2)
        return Polygon(
            [(self.x - dx + n[0], self.y - dy + n[1]), (self.x + dx + n[0], self.y + dy + n[1]),
             (self.x, self.y)])

    def flip(self):
        self.a = np.mod(self.a + 180, 360)
        self.xya = np.array((self.x, self.y, self.a))
        return self

    @classmethod
    def from_points(cls, points: np.ndarray, z: float = 0, h: float = 0, decimals: float = DECIMALS) -> "Port":
        """Initialize a :code:`Port` using a :code:`LineString` in Shapely.

        The port can be unambiguously defined using a tangent whose port
        faces 90 degrees counterclockwise (normal/perpendicular direction) from that tangent.
        The width of the port is the magnitude of the vector and the location of the port is the
        centroid of the vector.

        Args:
            points: Points representing the vector.
            z: The z position of the port.
            h: The height / thickness of the port.
            decimals: decimal precision for the points

        Returns:
            The :code:`Port` represented by the shapely :code:`Polygon` triangle.

        """

        first, second = points
        c = (first + second) / 2
        d = (second[1] - first[1]) + (second[0] - first[0]) * 1j
        a = -np.angle(d) * 180 / np.pi
        return cls(*np.around(c, DECIMALS), a, np.abs(d), z, h)

    def tangent(self, scale: float = 1):
        """The vector tangent (parallel) to the direction of the port

        Args:
            scale: The magnitude of the normal vector

        Returns:
            Return the vector normal to the port

        """
        return np.array([np.cos(self.a * np.pi / 180), np.sin(self.a * np.pi / 180)]) * scale

    def normal(self, scale: float = 1):
        """The normal vector perpendicular to the direction of the port (e.g. useful for turns)

        Args:
            scale: The magnitude of the normal vector

        Returns:
            Return the vector normal to the port

        """
        return np.array([np.sin(self.a * np.pi / 180), -np.cos(self.a * np.pi / 180)]) * scale

    @property
    def line(self):
        dx, dy = -self.normal(self.w / 2)
        x, y = self.xy
        return np.array([[x - dx, y - dy], [x + dx, y + dy]]).T

    def transformed_line(self, transform: np.ndarray):
        return np.array(transform[..., :2, :]) @ np.vstack((self.line, np.array([(1, 1)])))

    def transform(self, transform: np.ndarray, decimals: float = DECIMALS):
        line = self.transformed_line(transform).T
        c = (line[0] + line[1]) / 2
        d = (line[1, 1] - line[0, 1]) + (line[1, 0] - line[0, 0]) * 1j
        self.a = -np.angle(d) * 180 / np.pi
        self.x = np.around(c[0], DECIMALS)
        self.y = np.around(c[1], DECIMALS)
        self.w = np.abs(d)
        self.xy = np.array((self.x, self.y))
        self.xya = np.array((self.x, self.y, self.a))
        self.center = np.array((self.x, self.y, self.z))
        return self

    def hvplot(self, name: str = 'port'):
        # import locally since this import takes a while to import globally.
        import holoviews as hv
        x, y = self.shapely.exterior.coords.xy
        px, py = self.shapely.centroid.xy
        return hv.Polygons([{'x': x, 'y': y}]).opts(
            data_aspect=1, frame_height=200, color='red', line_alpha=0) * hv.Text(px[0], py[0], name)

    @property
    def copy(self) -> "Port":
        """Return a copy of this port for repeated use.

        Returns:
            A deep copy of this port.

        """
        return Port(*self.xya, self.w, self.z, self.h)
