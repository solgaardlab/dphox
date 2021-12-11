from typing import Callable, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split
from scipy.special import fresnel

from .typing import Union, Optional, Float2

MAX_GDS_POINTS = 8096

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
        geom: Geometry that potentially has holes.
        along_y: Split the geometry along y (flips the split direction on each recursive call).

    Returns:
        MultiPolygon representing the fractured geometry.

    """
    polys = []
    if isinstance(geom, Polygon):
        geom = MultiPolygon([geom])
    for geom_poly in geom.geoms:
        if len(geom_poly.interiors):
            c = geom_poly.interiors[0].centroid
            minx, miny, maxx, maxy = geom_poly.bounds
            splitter = LineString([(c.x, miny), (c.x, maxy)]) if along_y else LineString([(minx, c.y), (maxx, c.y)])
            for g in split(geom_poly, splitter).geoms:
                polys.extend(split_holes(g, not along_y))
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
    polys = []
    if isinstance(geom, Polygon):
        geom = MultiPolygon([geom])

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


def parametric_fn(path: Union[float, Callable],
                  width: Union[Callable, float, Tuple[float, float]],
                  num_evaluations: int = 99, max_num_points: int = MAX_GDS_POINTS,
                  decimal_places: int = 5):
    u = np.linspace(0, 1, num_evaluations)[:, np.newaxis]
    if isinstance(path, float) or isinstance(path, int):
        path = np.hstack([u * path, np.zeros_like(u)])
        path_diff = np.hstack([np.ones_like(u) / num_evaluations * path, np.zeros_like(u)]).T
    else:
        path = path(u)
        if isinstance(path, tuple):
            path, path_diff = path
            path_diff = path_diff.T
        else:
            path_diff = np.diff(path, axis=0)
            path_diff = np.vstack((path_diff[0], path_diff))
            path_diff = path_diff.T

    widths = width
    if not isinstance(width, float) and not isinstance(width, int):
        widths = taper(width)(u) if isinstance(width, tuple) else width(u)

    angles = np.arctan2(path_diff[1], path_diff[0])
    width_translation = np.vstack((-np.sin(angles) * widths, np.cos(angles) * widths)).T / 2
    top_path = np.around(path + width_translation, decimal_places)
    bottom_path = np.around(path - width_translation, decimal_places)

    back_port = LineString([bottom_path[-1], top_path[-1]])
    front_port = LineString([top_path[0], bottom_path[0]])

    # clean path fracture
    num_splitters = num_evaluations * 2 // max_num_points
    splitters = [LineString([top_path[(i + 1) * max_num_points // 2], bottom_path[(i + 1) * max_num_points // 2]])
                 for i in range(num_splitters)]
    polygon = Polygon(np.vstack((top_path, bottom_path[::-1])))
    polys = []
    for splitter in splitters:
        geoms = split(polygon, splitter).geoms
        poly, polygon = geoms[0], geoms[1]
        polys.append(poly)
    polys.append(polygon)
    return MultiPolygon(polys), back_port, front_port


def bezier_sbend_fn(bend_l, bend_h, inverted=False):
    pole_1 = np.asarray((bend_l / 2, 0))
    pole_2 = np.asarray((bend_l / 2, (-1) ** inverted * bend_h))
    pole_3 = np.asarray((bend_l, (-1) ** inverted * bend_h))

    def _sbend(t: np.ndarray):
        path = 3 * (1 - t) ** 2 * t * pole_1 + 3 * (1 - t) * t ** 2 * pole_2 + t ** 3 * pole_3
        derivative = 3 * (1 - t) ** 2 * pole_1 + 6 * (1 - t) * t * (pole_2 - pole_1) + 3 * t ** 2 * (pole_3 - pole_2)
        return path, derivative

    return _sbend


def taper(taper_params: Union[np.ndarray, Tuple[float]]):
    poly_exp = np.arange(len(taper_params), dtype=float)
    return lambda u: np.sum(taper_params * u ** poly_exp, axis=1)


def euler_bend_fn(radius: float, angle: float = 90):
    angle = angle / 180 * np.pi

    def _bend(t: np.ndarray):
        z = np.sqrt(angle * t)
        y, x = fresnel(z / np.sqrt(np.pi / 2))
        return radius * np.hstack((x, y)), radius * np.hstack((y, -x))

    return _bend


def circular_bend_fn(radius: float, angle: float = 90):
    angle = angle / 180 * np.pi

    def _bend(t: np.ndarray):
        x = radius * np.sin(angle * t)
        y = radius * (1 - np.cos(angle * t))
        return np.hstack((x, y)), radius * np.hstack((np.cos(angle * t), np.sin(angle * t)))

    return _bend


def spiral_fn(rotations: int, scale: float = 5, separation_scale: float = 1):
    def _spiral(t: np.ndarray):
        theta = t * rotations * np.pi + 2 * np.pi
        radius = (theta - 2 * np.pi) * separation_scale / scale + 2 * np.pi
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        return scale / np.pi * np.hstack((x, y)), scale / np.pi * np.hstack((y, -x))

    return _spiral
