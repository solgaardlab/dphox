from typing import Union, Tuple, List, Optional, Dict, Callable
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import gdspy as gy

Int2 = Tuple[int, int]
Shape3 = Tuple[int, int, int]
Float2 = Tuple[float, float]
Float3 = Tuple[float, float, float]
Float4 = Tuple[float, float, float, float]
Size5 = Tuple[float, float, float, float, float]
Shape = Union[Int2, Shape3]
Dim = Union[Float2, Float3]
Spacing = Union[float, Tuple[float, float, float]]
PolygonLike = Union[gy.Polygon, gy.FlexPath, Polygon, MultiPolygon, np.ndarray]
LayerLabel = Union[int, str, Int2]
