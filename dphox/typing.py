from typing import Union, Tuple, List, Optional, Dict, Callable
import numpy as np
import scipy.sparse as sp
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
Op = Callable[[np.ndarray], np.ndarray]
SpSolve = Callable[[sp.spmatrix, np.ndarray], np.ndarray]
Source = Union[Callable[[float], Tuple[np.ndarray, np.ndarray]], np.ndarray]
State = Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]], Optional[List[np.ndarray]]]
PolygonLike = Union[gy.Polygon, gy.FlexPath, Polygon, MultiPolygon, np.ndarray]
LayerLabel = Union[int, str, Int2]
