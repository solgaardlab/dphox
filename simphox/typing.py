from typing import Union, Tuple, List, Optional

Shape2 = Tuple[int, int]
Shape3 = Tuple[int, int, int]
Dim2 = Tuple[float, float]
Dim3 = Tuple[float, float, float]
GridShape = Union[Shape2, Shape3]
GridSpacing = Union[float, Tuple[float, float, float]]
