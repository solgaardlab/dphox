import numpy as np
from typing import List, Optional, Union
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import split

from .typing import Float4

MAX_GDS_POINTS = 8096
DEFAULT_RESOLUTION = 99
DECIMALS = 6

PORT_LAYER = 1
PORT_GDS_LABEL = (1002, 0)


def poly_points(geom: Polygon, decimals: Optional[int] = None):
    """Get the `exterior` points of a given polygon geometry

    Args:
        geom: Geometry from which to get exterior points
        decimals: Number of decimal places to round the points

    Returns:
        The numpy array representing the points of the polygon.

    """
    coords = geom.exterior.coords
    if len(coords) == 0:
        raise ValueError('There are no coordinates in this geometry.')
    points = np.array(coords.xy).T
    return points if decimals is None else np.around(points, decimals)


def linestring_points(geom: LineString, decimals: Optional[int] = None):
    """Get the `exterior` points of a given polygon geometry

    Args:
        geom: Geometry from which to get exterior points
        decimals: Number of decimal places to round the points

    Returns:
        The numpy array representing the points of the polygon.

    """
    coords = geom.coords
    if len(coords) == 0:
        raise ValueError('There are no coordinates in this geometry.')
    points = np.array(coords.xy).T
    return points if decimals is None else np.around(points, decimals)


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


def split_holes(geom: Union[Polygon, MultiPolygon, GeometryCollection, np.ndarray],
                along_y: bool = True) -> MultiPolygon:
    """Fracture a shapely geometry into polygon along x or y direction (depending on :code:`along_x`).

    If there are no holes, just return the input geometry :code:`geom`. Otherwise, recursively run the algorithm on the
    split geometries.

    Args:
        geom: Geometry that potentially has holes.
        along_y: Split the geometry along y (flips the split direction on each recursive call).

    Returns:
        MultiPolygon representing the fractured geometry.

    """
    polys = []
    if isinstance(geom, Polygon):
        geom = MultiPolygon([geom])
    elif isinstance(geom, np.ndarray):
        geom = MultiPolygon([Polygon(geom)])
    elif not isinstance(geom, MultiPolygon) and not isinstance(geom, GeometryCollection):
        raise TypeError("The input geometry is not either a shapely Polygon, MultiPolygon or ndarray.")
    for geom_poly in geom.geoms:
        if geom_poly.interiors:
            c = geom_poly.interiors[0].centroid
            minx, miny, maxx, maxy = geom_poly.bounds
            splitter = LineString([(c.x, miny), (c.x, maxy)]) if along_y else LineString([(minx, c.y), (maxx, c.y)])
            for g in split(geom_poly, splitter).geoms:
                polys.extend(split_holes(g, not along_y))
        else:
            polys.append(geom_poly)
    return MultiPolygon(polys)


def split_max_points(geom: Union[Polygon, MultiPolygon, np.ndarray], max_num_points: int = MAX_GDS_POINTS,
                     along_y: bool = True):
    """Fracture shapely geometry into polygons that possess a specified maximum number of points :code:`max_num_points`.

    If the number of points in yhe geometry is already less than or equal to :code:`max_num_points`,
    just return the input geometry :code:`geom`. Otherwise, recursively run the algorithm on the split geometries.

    Args:
        geom: Geometry that potentially has holes.
        max_num_points: Maximum number of points.
        along_y: Split the geometry along y.

    Returns:
        MultiPolygon representing the fractured geometry.

    """
    polys = []
    if isinstance(geom, Polygon):
        geom = MultiPolygon([geom])
    elif isinstance(geom, np.ndarray):
        geom = MultiPolygon([Polygon(geom)])
    elif not isinstance(geom, MultiPolygon):
        raise TypeError("The input geometry is not either a shapely Polygon, MultiPolygon or ndarray.")

    for geom_poly in geom:
        if poly_points(geom_poly).shape[0] > max_num_points:
            minx, miny, maxx, maxy = geom_poly.bounds
            c = geom_poly.centroid
            splitter = LineString([(c.x, miny), (c.x, maxy)]) if along_y else LineString([(minx, c.y), (maxx, c.y)])
            for g in split(geom_poly, splitter).geoms:
                polys.extend(split_max_points(g, along_y=along_y))
        else:
            polys.append(geom_poly)
    return MultiPolygon(polys)


def bounds(points: np.ndarray):
    """Bounding box of points of the form :code:`(minx, miny, maxx, maxy)`

    Args:
        polygons: The ndarray polygon representation.
        overall: Get the overall bounds rather than a list of bounds for each polygon

    Returns:
        Bounding box tuple.

    """
    return np.array((np.min(points[0]), np.min(points[1]), np.max(points[0]), np.max(points[1])))


def poly_bounds(polygons: List[np.ndarray], overall: bool = False):
    """Bounding box of polygons of the form :code:`(minx, miny, maxx, maxy)`

    Args:
        polygons: The ndarray polygon representation.
        overall: Get the overall bounds rather than a list of bounds for each polygon

    Returns:
        Bounding box tuple.

    """

    return bounds(np.hstack(polygons)) if overall else [bounds(p) for p in polygons]


def min_aspect_bounds(b: Union[np.ndarray, Float4], min_aspect: float = 0.25):
    """Minimum aspect ratio (needed for plotting)

    We adjust the bounds of the smaller dimension to achieve the min aspect ratio if it is not already exceeded.
    An aspect ratio of 0.25 indicates that one dimension cannot exceed 4 times the other.

    Args:
        b: bounds.
        min_aspect: minimum aspect ratio.

    Returns:
        The new bounds.

    """
    b = np.array(b)
    center = np.array((np.sum(b[::2]) / 2, np.sum(b[1::2]) / 2))
    size = np.array((np.abs(b[2] - b[0]), np.abs(b[3] - b[1])))
    dx = np.array((np.maximum(min_aspect * size[1], size[0]) / 2, np.maximum(size[1], min_aspect * size[0]) / 2))
    return np.hstack((center - dx, center + dx))


def shapely_patch(geom: Union[MultiPolygon, Polygon], **kwargs):
    """Get the shapely patch for plotting in matplotlib (remove descartes dependency).

    Args:
        geom: geometry
        kwargs: keyword arguments for matplotlib's PathPatch

    Returns:
        The Matplotlib `PathPatch` for plotting in matplotlib.

    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    if geom.geom_type == 'Polygon':
        polygon = [Polygon(geom)]
    elif geom.geom_type in ['MultiPolygon', 'GeometryCollection']:
        polygon = [Polygon(p) for p in geom]
    else:
        raise ValueError("A polygon or multi-polygon representation is required")

    if len(polygon) == 0:
        return None

    vertices = np.vstack(
        [np.vstack([np.array(poly.exterior)[:, :2]] + [np.array(hole)[:, :2] for hole in poly.interiors])
         for poly in polygon]).squeeze()
    codes = sum([
        ([Path.MOVETO] + [Path.LINETO] * (len(poly.exterior.coords) - 1) + sum(([Path.MOVETO] + [Path.LINETO] * (len(hole.coords) - 1)
                                                                                for hole in poly.interiors), []))
        for poly in polygon], [])

    return PathPatch(Path(vertices, codes), **kwargs)
