from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .transform import reflect2d, rotate2d, translate2d
from .foundry import CommonLayer
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
    layer: str = CommonLayer.PORT

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
        """Transform.

        Args:
            transform: affine transform tensor whose final two dimensions transform the port
            decimals: decimal precision for port x, y after the rotation.

        Returns:

        """
        line = self.transformed_line(transform).T
        c = (line[0] + line[1]) / 2
        d = (line[1, 1] - line[0, 1]) + (line[1, 0] - line[0, 0]) * 1j
        self.a = -np.angle(d) * 180 / np.pi
        self.x = np.around(c[0], decimals)
        self.y = np.around(c[1], decimals)
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

    def translate(self, dx: float = 0, dy: float = 0) -> "Port":
        """Translate port.

        Args:
            dx: Displacement in x
            dy: Displacement in y

        Returns:
            The translated port

        """
        return self.transform(translate2d((dx, dy)))

    def rotate(self, angle: float, origin: Tuple[float, float] = (0, 0)) -> "Port":
        """Rotate the geometry about :code:`origin`.

        Args:
            angle: Angle of rotation in degrees
            origin: Rotation origin

        Returns:
            The rotated port

        """
        return self.transform(rotate2d(np.radians(angle), origin))

    @property
    def copy(self) -> "Port":
        """Return a copy of this port for repeated use.

        Returns:
            A deep copy of this port.

        """
        return Port(*self.xya, self.w, self.z, self.h)

    def orient_xyaf(self, xyaf: np.ndarray, flip_y: bool = False):
        """Orient xyaf (x, y , angle, flip) based on this port.

        Note:
            The orientation is only modified if the port specified is not the default port.

        Args:
            xyaf: The x, y, angle, and flip objects.
            flip_y: If only xya is provided, specify the flip.

        Returns:
            The new xyaf after orienting based on this port.

        """
        if not isinstance(xyaf, np.ndarray) or len(xyaf) > 4 or len(xyaf) < 2:
            raise TypeError(f"Require xya to be ndarray but got {type(xyaf)}")
        elif len(xyaf) == 2:
            transform_array = np.array((*xyaf, 0, flip_y))
        elif len(xyaf) == 3:
            transform_array = np.array((*xyaf, flip_y))
        else:
            transform_array = np.array(xyaf)

        rotated_translate = -rotate2d(np.radians(xyaf[-1] - self.a + 180))[:2, :2] @ self.xy
        return transform_array + np.array((*rotated_translate, -self.a + 180, 0))

    def transform_xyaf(self, xyaf: Tuple[float, float, float, bool]):
        """Transform this port given some xyaf (x, y , angle, flip) specification.

        Args:
            xyaf: The x, y, angle, and flip objects.

        Returns:

        """
        x, y, a, f = xyaf
        return self.transform(translate2d((x, y)) @ rotate2d(np.radians(a)) @ reflect2d(flip=bool(f)))
