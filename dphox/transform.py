from dataclasses import dataclass
from typing import Iterable, List, Optional

from .typing import Float2, Tuple, Union
from .utils import DECIMALS, fix_dataclass_init_docs

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


def reflect2d(origin: Float2 = (0, 0), horiz: bool = False, flip: bool = True):
    if not flip:
        return np.eye(3)
    x0, y0 = origin
    return np.array([[1 - 2 * horiz, 0, 2 * x0 * horiz],
                     [0, 1 - 2 * (1 - horiz), 2 * y0 * (1 - horiz)],
                     [0, 0, 1]])


class AffineTransform:
    """Affine transformer.

    An affine transform

    Attributes:
        transform: The `...x3x3` transform or list of sequential 3x3 transforms that are multiplied together.
    """

    def __init__(self, transform: Union[np.ndarray, Iterable[np.ndarray]]):
        self.transform = transform
        if isinstance(self.transform, (list, tuple)):
            self.transform = np.linalg.multi_dot(self.transform[::-1])

    @property
    def ndim(self):
        return self.transform.ndim

    def transform_points(self, points: np.ndarray, tangents_only: bool = False, decimals: int = DECIMALS):
        """Transform a list of points (or tangents).

        This method runs the transformation :math:`AP`, where :math:`A \\in M \\times 2 \\times 3` and
        :math:`P \\in 3 \\times N`, where a list of ones is concatenated in the final dimension.

        In the case we need to transform tangents, the third dimension is irrelevant since translations do not change
        the tangent and normal vectors. So now we have :math:`A \\in M \\times 2 \\times 2` and
        :math:`P \\in 2 \\times N` where now we don't concatenate any 1's for the transformation.

        Args:
            points: The :code:`2xN` array of points to be transformed.
            tangents_only: Ignore the third dimension of the transform (for tangents).
            decimals: The number of decimal places of precision for the transform.

        Returns:
            The :code:`2xN` transformed points

        """
        points = points if tangents_only else np.vstack((points, np.ones(points.shape[1])))
        return np.around(self.transform[..., :2, :3 - tangents_only] @ points, decimals=decimals)

    def transform_geoms(self, geoms: List[np.ndarray], tangents: bool = False, decimals: int = DECIMALS):
        all_points = np.hstack(geoms) if len(geoms) > 1 else geoms[0]
        split = np.cumsum([p.shape[1] for p in geoms])
        return np.split(self.transform_points(all_points, tangents_only=tangents,
                                              decimals=decimals), split[:-1], axis=-1)


@fix_dataclass_init_docs
@dataclass
class GDSTransform(AffineTransform):
    """GDS transform class

    Attributes:
        x: x translation
        y: y translation
        angle: rotation angle
        flip_y: Whether to flip the design about the x-axis (in y direction)
        mag: scale magnification

    """
    x: float = 0
    y: float = 0
    angle: float = 0
    flip_y: bool = False
    mag: float = 1

    def __post_init__(self):
        super(GDSTransform, self).__init__((scale2d((self.mag, self.mag)),
                                            reflect2d(flip=self.flip_y),
                                            rotate2d(np.radians(self.angle)),
                                            translate2d((self.x, self.y))))

    @property
    def xya(self):
        return np.array((self.x, self.y, self.angle))

    @property
    def xyaf(self):
        return np.array((self.x, self.y, self.angle, self.flip_y))

    def set_xya(self, xya: Union[np.ndarray, Tuple[float, float, float]]):
        self.x, self.y, self.angle = xya

    def set_xyaf(self, xyaf: Union[np.ndarray, Tuple[float, float, float, bool]]):
        self.x, self.y, self.angle = xyaf[:3]
        self.flip_y = bool(xyaf[-1])

    @classmethod
    def parse(cls, transform: Optional[Union["GDSTransform", Tuple, np.ndarray]],
              existing_transform: Optional[Tuple[AffineTransform, List["GDSTransform"]]] = None) -> \
            Tuple[AffineTransform, List["GDSTransform"]]:
        """Turns representations like :code:`(x, y, angle)` or a numpy array into convenient GDS/affine transforms.

        Args:
            transform: The transform array in the order :code:`(x, y, angle, mag)` or just a GDS transform.
            existing_transform: Given an existing transform(s), apply the new transform on top of the existing one.

        Returns:
            A tuple of :code:`AffineTransform` representation and a list of GDS transforms for gds output.

        """
        gds_transforms = _parse_gds_transform(transform)
        parallel_transform = np.array([t.transform for t in gds_transforms])
        if existing_transform is not None:
            return AffineTransform(np.vstack((existing_transform[0].transform, parallel_transform))), existing_transform[1] + gds_transforms
        else:
            return AffineTransform(parallel_transform), gds_transforms


def _parse_gds_transform(transform):
    """Recursive helper function for parsing GDS transform objects flexibly."""
    if transform is None:
        gds_transforms = [GDSTransform()]
    elif isinstance(transform, GDSTransform):
        gds_transforms = [transform]
    elif isinstance(transform, tuple) or isinstance(transform, list) or isinstance(transform, np.ndarray):
        gds_transforms = [GDSTransform(*transform)] if len(transform) <= 5 and all(np.isscalar(v) for v in transform)\
            else sum((_parse_gds_transform(t) for t in transform), [])

    else:
        raise TypeError("Expected transform to be of type GDSTransform, or tuple or ndarray representing GDS"
                        f"transforms, but got a malformed input of type: {type(transform)}")

    return gds_transforms
