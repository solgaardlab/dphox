from dataclasses import dataclass

from .typing import Float2, Tuple, List, Union, Iterable, Optional
from .utils import fix_dataclass_init_docs

import numpy as np


def rotate2d(angle: float, origin: Float2 = (0, 0)):
    c, s = np.cos(angle), np.sin(angle)
    x0, y0 = origin
    xoff = x0 - x0 * c + y0 * s
    yoff = y0 - x0 * s - y0 * c
    return np.array([[c, -s, xoff],
                     [s, c, yoff],
                     [0, 0, 1]])


def translate2d(shift: Float2 = (0, 0)):
    dx, dy = shift
    return np.array([[1, 0, dx],
                     [0, 1, dy],
                     [0, 0, 1]])


def scale2d(scale: Float2 = (0, 0), origin: Float2 = (0, 0)):
    x0, y0 = origin
    xs, ys = scale
    xoff = x0 * (1 - xs)
    yoff = y0 * (1 - ys)
    return np.array([[xs, 0, xoff],
                     [0, ys, yoff],
                     [0, 0, 1]])


def skew2d(skew: Float2 = (0, 0), origin: Float2 = (0, 0)):
    tx, ty = np.tan(skew)
    x0, y0 = origin
    return np.array([[1, tx, -y0 * tx],
                     [ty, 1, -x0 * ty],
                     [0, 0, 1]])


def reflect2d(origin: Float2 = (0, 0), horiz: bool = False):
    x0, y0 = origin
    return np.array([[1 - 2 * horiz, 0, -2 * x0 * horiz],
                     [0, 1 - 2 * (1 - horiz), -2 * y0 * (1 - horiz)],
                     [0, 0, 1]])


class AffineTransform:
    def __init__(self, transform: Union[np.ndarray, Iterable[np.ndarray]]):
        self.transform = transform
        if isinstance(self.transform, list) or isinstance(self.transform, tuple):
            self.transform = np.linalg.multi_dot(self.transform[::-1])

    @property
    def ndim(self):
        return self.transform.ndim

    def transform_points(self, points: np.ndarray):
        return self.transform[..., :2, :] @ np.vstack((points, np.ones(points.shape[1])))

    def transform_polygons(self, polygons: List[np.ndarray]):
        all_points = np.hstack(polygons) if len(polygons) > 1 else polygons[0]
        split = np.cumsum([p.shape[1] for p in polygons])
        return np.split(self.transform_points(all_points), split[:-1], axis=-1)


@fix_dataclass_init_docs
@dataclass
class GDSTransform(AffineTransform):
    x: float
    y: float
    angle: float = 0
    mag: float = 1

    def __post_init__(self):
        super(GDSTransform, self).__init__((scale2d((self.mag, self.mag)),
                                            rotate2d(self.angle),
                                            translate2d((self.x, self.y))))

    @classmethod
    def from_array(cls, transform: Optional[Union["GDSTransform", Tuple, np.ndarray]]) -> \
            Tuple[AffineTransform, List["GDSTransform"]]:
        """Turns representations like :code:`(x, y, angle)` or a numpy array into convenient GDS/affine transforms.

        Args:
            transform: The transform array in the order :code:`(x, y, angle, mag)` or just a GDS transform.

        Returns:
            A tuple of :code:`AffineTransform` representation and a list of GDS transforms for gds output.

        """
        if transform is None:
            transform = (0, 0, 0)
        if isinstance(transform, tuple):
            transform = np.array(transform)
        if transform.ndim == 1:
            gds_transforms = [cls(*transform)]
        elif transform.ndim == 2:  # efficiently represent the transformation in 2D
            gds_transforms = [cls(*t) for t in transform]
        elif isinstance(transform, GDSTransform):
            gds_transforms = [transform]
        else:
            raise TypeError("Expected transform to be of type GDSTransform, or tuple or ndarray representing GDS"
                            "transforms, but got a malformed input.")
        return AffineTransform(np.array([t.transform for t in gds_transforms])), gds_transforms
