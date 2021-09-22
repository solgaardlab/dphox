import numpy as np
import pytest
from typing import Optional, Tuple, List

from dphox.pattern import Pattern, Box, Circle
from dphox.utils import poly_points, split_holes

UNIT_BOX = Box()


@pytest.mark.parametrize(
    "pattern, poly_list",
    [
        [UNIT_BOX, [[[0.5, 0.5, -0.5, -0.5, 0.5],
                     [-0.5, 0.5, 0.5, -0.5, -0.5]]]],
        [UNIT_BOX.hollow(0.1), [[[0.4, 0.5, 0.5, 0.4, 0.4],
                                 [0.5, 0.5, -0.5, -0.5, 0.5]],
                                [[-0.4, -0.5, -0.5, -0.4, -0.4],
                                 [-0.5, -0.5, 0.5, 0.5, -0.5]],
                                [[0.5, 0.5, -0.5, -0.5, 0.5],
                                 [-0.4, -0.5, -0.5, -0.4, -0.4]],
                                [[-0.5, -0.5, 0.5, 0.5, -0.5],
                                 [0.4, 0.5, 0.5, 0.4, 0.4]]
                                ]],
        # Demonstration of regular polygons of `resolution * 2` sides!
        [Circle(resolution=2), [[[1., 0.70711, 0., -0.70711, -1., -0.70711, 0., 0.70711, 1.],
                                 [0., -0.70711, -1., -0.70711, 0., 0.70711, 1., 0.70711, 0.]]]],
        [Circle(resolution=4), [[[1., 0.92388, 0.70711, 0.38268, 0., -0.38268,
                                  -0.70711, -0.92388, -1., -0.92388, -0.70711, -0.38268,
                                  0., 0.38268, 0.70711, 0.92388, 1.],
                                 [0., -0.38268, -0.70711, -0.92388, -1., -0.92388,
                                  -0.70711, -0.38268, 0., 0.38268, 0.70711, 0.92388,
                                  1., 0.92388, 0.70711, 0.38268, 0.]]]]
    ],
)
def test_poly(pattern: Pattern, poly_list: List[np.ndarray]):
    for i, poly in enumerate(pattern.polys):
        np.testing.assert_allclose(poly_points(poly).T, poly_list[i])


@pytest.mark.parametrize(
    "pattern, poly_list",
    [
        [UNIT_BOX - UNIT_BOX.copy.scale(0.5, 0.5), [[[0.5, 0., 0., 0.25, 0.25, 0., 0., 0.5, 0.5],
                                                     [-0.5, -0.5, -0.25, -0.25, 0.25, 0.25, 0.5, 0.5, -0.5]],
                                                    [[0., -0.5, -0.5, 0., 0., -0.25, -0.25, 0., 0.],
                                                     [-0.5, -0.5, 0.5, 0.5, 0.25, 0.25, -0.25, -0.25, -0.5]]]],
        [Circle(resolution=3) - Circle(resolution=3).scale(0.5, 0.5), [[[1, 8.66030000e-01, 5.0e-01, 0, 0, 2.5e-01,
                                                                         4.33015000e-01, 5.0e-01, 4.33015000e-01,
                                                                         2.5e-01, 0, 0, 5.0e-01, 8.66030000e-01, 1],
                                                                        [0, -5.0e-01, -8.66030000e-01, -1, -5.0e-01,
                                                                         -4.33015000e-01, -2.5e-01, 0, 2.5e-01,
                                                                         4.33015000e-01, 5.0e-01, 1, 8.66030000e-01,
                                                                         5.0e-01, 0]],
                                                                       [[0, 0, -0.5, -8.66030000e-01, -1,
                                                                         -8.66030000e-01, -0.5, 0, 0, 0, 0,
                                                                         -0.25, -4.33015000e-01, -0.5,
                                                                         -4.33015000e-01, -0.25, 0, 0, 0],
                                                                        [-1, -1, -8.66030000e-01, -0.5, 0, 0.5,
                                                                         8.66030000e-01, 1.00000000e+00, 1.00000000e+00,
                                                                         0.5, 0.5, 4.33015000e-01, 0.25, 0,
                                                                         -0.25, -4.33015000e-01, -0.5, -0.5, -1]]]],
    ],
)
def test_poly_with_hole(pattern: Pattern, poly_list: List[np.ndarray]):
    for i, poly in enumerate(split_holes(pattern.shapely_union())):
        np.testing.assert_allclose(poly_points(poly).T, poly_list[i], atol=1e-10)
