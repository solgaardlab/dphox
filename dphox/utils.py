import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split

from .typing import Union, Optional

MAX_GDS_POINTS = 199

PORT_LAYER = 1
PORT_LABEL_LAYER = 1002


def poly_points(geom: Polygon, decimal_places: Optional[int] = None):
    """Get the `exterior` points of a given polygon geometry

    Args:
        geom: Geometry from which to get exterior points
        decimal_places: Number of decimal places to round the points

    Returns:
        The numpy array representing the points of the polygon.

    """
    coords = geom.exterior.coords
    if len(coords) == 0:
        raise ValueError('There are no coordinates in this geometry.')
    points = np.array(coords.xy).T
    return points if decimal_places is None else np.around(points, decimal_places)


def fix_dataclass_init_docs(cls):
    """Fix the ``__init__`` documentation for a :class:`dataclasses.dataclass`.

    Args:
        cls: The class whose docstring needs fixing

    Returns:
        The class that was passed so this function can be used as a decorator

    See Also:
        https://github.com/agronholm/sphinx-autodoc-typehints/issues/123
    """
    cls.__init__.__qualname__ = f'{cls.__name__}.__init__'
    return cls


def split_holes(geom: Union[Polygon, MultiPolygon], along_y: bool = True) -> MultiPolygon:
    """Fracture a shapely geometry into polygon along x or y direction (depending on :code:`along_x`).

    If there are no holes, just return the input geometry :code:`geom`. Otherwise, recursively run the algorithm on the
    split geometries.

    Args:
        geom: Geometry that potentially has holes
        along_y: Split the geometry along y

    Returns:
        MultiPolygon representing the fractured geometry.

    """
    polys = []
    if isinstance(geom, Polygon):
        geom = MultiPolygon([geom])
    for geom_poly in geom:
        if len(geom_poly.interiors):
            c = geom_poly.interiors[0].centroid
            minx, miny, maxx, maxy = geom_poly.bounds
            splitter = LineString([(c.x, miny), (c.x, maxy)]) if along_y else LineString([(minx, c.y), (maxx, c.y)])
            for g in split(geom_poly, splitter):
                polys.extend(split_holes(g, along_y))
        else:
            polys.append(geom_poly)
    return MultiPolygon(polys)


def split_max_points(geom: Union[Polygon, MultiPolygon], max_num_points: int = MAX_GDS_POINTS, along_y: bool = True):
    """Fracture shapely geometry into polygons that possess a specified maximum number of points :code:`max_num_points`.

    If the number of points in yhe geometry is already less than or equal to :code:`max_num_points`,
    just return the input geometry :code:`geom`. Otherwise, recursively run the algorithm on the split geometries.

    Args:
        geom: Geometry that potentially has holes
        max_num_points:
        along_y: Split the geometry along y

    Returns:
        MultiPolygon representing the fractured geometry.

    """
    points = poly_points(geom)
    num_points = points.shape[0]
    polys = []

    if num_points > max_num_points:
        minx, miny, maxx, maxy = geom.bounds
        c = geom.centroid
        splitter = LineString([(c[0], miny), (c[0], maxy)]) if along_y else LineString([(minx, c[1]), (miny, c[1])])
        for g in split(geom, splitter):
            polys.append(split_max_points(g, along_y))
    else:
        return MultiPolygon([geom]) if isinstance(geom, Polygon) else geom
