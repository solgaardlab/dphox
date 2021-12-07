from typing import Callable, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split
from scipy.special import fresnel

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
            for g in split(geom_poly, splitter).geoms:
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


def parametric(path: Union[float, Callable], width: Union[Callable, float, Tuple[float, float]],
               path_derivative: Optional[Callable] = None, num_evaluations: int = 99):
    u = np.linspace(0, 1, num_evaluations)[:, np.newaxis]
    path = np.hstack([u * path, np.zeros_like(u)]) if isinstance(path, float) or isinstance(path, int) else path(u)
    widths = width
    if not isinstance(widths, float) and not isinstance(widths, int):
        widths = np.linspace(widths[0], widths[1], num_evaluations) if isinstance(widths, tuple) else widths(u)

    if path_derivative:
        path_diff = path_derivative(u)
    else:
        path_diff = np.diff(path, axis=0)
        path_diff = np.vstack((path_diff[0], path_diff))
        path_diff = path_diff.T
    angles = np.arctan2(path_diff[1], path_diff[0])
    width_translation = np.vstack((-np.sin(angles) * widths, np.cos(angles) * widths)).T / 2
    top_path = path + width_translation
    bottom_path = path - width_translation
    return Polygon(np.vstack((top_path, bottom_path[::-1])))


def cubic_bezier(bend_l, bend_h, inverted=False):
    pole_1 = np.asarray((bend_l / 2, 0))
    pole_2 = np.asarray((bend_l / 2, (-1) ** inverted * bend_h))
    pole_3 = np.asarray((bend_l, (-1) ** inverted * bend_h))
    return lambda t: 3 * (1 - t) ** 2 * t * pole_1 + 3 * (1 - t) * t ** 2 * pole_2 + t ** 3 * pole_3


def cubic_bezier_derivative(bend_l, bend_h, inverted=False):
    pole_1 = np.asarray((bend_l / 2, 0))
    pole_2 = np.asarray((bend_l / 2, (-1) ** inverted * bend_h))
    pole_3 = np.asarray((bend_l, (-1) ** inverted * bend_h))
    return lambda t: 3 * (1 - t) ** 2 * pole_1 + 6 * (1 - t) * t * (pole_2 - pole_1) + 3 * t ** 2 * (pole_3 - pole_2)


def taper(w: float, taper_params: np.ndarray, inverted: bool = False):
    poly_exp = np.arange(len(taper_params), dtype=float)
    if inverted:
        return lambda u: w - np.sum(taper_params) + np.sum(taper_params * (1 - u) ** poly_exp, axis=1)
    else:
        return lambda u: w + np.sum(taper_params * u ** poly_exp, axis=1)


def bend(radius: float, angle: float = 90, euler: bool = False):
    angle = angle / 180 * np.pi
    if euler > 0:
        def _bend(t: np.ndarray):
            z = np.sqrt(angle * t)
            y, x = fresnel(z / np.sqrt(np.pi / 2))
            return radius * np.hstack((x, y))
    else:
        def _bend(t: np.ndarray):
            x = radius * np.sin(angle * t)
            y = radius * (1 - np.cos(angle * t))
            return np.hstack((x, y))
    return _bend


def spiral(rotations: int, scale: float):
    def _spiral(t: np.ndarray):
        theta = t * rotations * 2 * np.pi
        x, y = theta * np.cos(theta), theta * np.sin(theta)
        return scale * np.hstack((x, y))
    return _spiral

