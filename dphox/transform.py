from dataclasses import dataclass

from .typing import Float2, Float4, Tuple, List, Union, Iterable, Optional
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


def reflect2d(origin: Float2 = (0, 0), horiz: bool = False):
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
        if isinstance(self.transform, list) or isinstance(self.transform, tuple):
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
    x: float
    y: float
    angle: float = 0
    mag: float = 1

    def __post_init__(self):
        super(GDSTransform, self).__init__((scale2d((self.mag, self.mag)),
                                            rotate2d(np.radians(self.angle)),
                                            translate2d((self.x, self.y))))

    @classmethod
    def parse(cls, transform: Optional[Union["GDSTransform", Tuple, np.ndarray]],
              existing_transform: Optional[Tuple[AffineTransform, List["GDSTransform"]]] = None) -> \
            Tuple[AffineTransform, List["GDSTransform"]]:
        """Turns representations like :code:`(x, y, angle)` or a numpy array into convenient GDS/affine transforms.

        Args:
            transform: The transform array in the order :code:`(x, y, angle, mag)` or just a GDS transform.

        Returns:
            A tuple of :code:`AffineTransform` representation and a list of GDS transforms for gds output.

        """
        if transform is None:
            transform = (0, 0)
        if isinstance(transform, tuple) or isinstance(transform, list):
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
        parallel_transform = np.array([t.transform for t in gds_transforms])
        if existing_transform is not None:
            return AffineTransform(np.vstack((existing_transform[0].transform, parallel_transform))),\
                   existing_transform[1] + gds_transforms
        else:
            return AffineTransform(parallel_transform), gds_transforms
