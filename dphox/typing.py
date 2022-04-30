from collections import namedtuple
from typing import Callable, Tuple, Union
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString

CurveTuple = namedtuple("CurveTuple", "points tangents")

Int2 = Tuple[int, int]
Shape3 = Tuple[int, int, int]
Float2 = Tuple[float, float]
Float3 = Tuple[float, float, float]
Float4 = Tuple[float, float, float, float]
Size5 = Tuple[float, float, float, float, float]
Shape = Union[Int2, Shape3]
Dim = Union[Float2, Float3]
Spacing = Union[float, Tuple[float, float, float]]
PolygonLike = Union[Polygon, MultiPolygon, np.ndarray]
CurveLike = Union[LineString, np.ndarray, CurveTuple]
LayerLabel = Union[int, str, Int2]
PathWidth = Union[Callable, float, Tuple[float, ...], np.ndarray]
