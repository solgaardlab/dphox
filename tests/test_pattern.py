from typing import List

import numpy as np
import pytest

from dphox.pattern import Box, Circle, Pattern
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
        [Circle(resolution=2), [[[1, 7.07106781e-01, 0, -7.07106781e-01, -1, -7.07106781e-01, 0, 7.07106781e-01, 1],
                                 [0, -7.07106781e-01, -1, -7.07106781e-01, 0, 7.07106781e-01, 1, 7.07106781e-01, 0]]]],
        [Circle(resolution=4), [[[1, 9.23879533e-01, 7.07106781e-01, 3.82683432e-01, 0, -3.82683432e-01,
                                  -7.07106781e-01, -9.23879533e-01, -1, -9.23879533e-01, -7.07106781e-01,
                                  -3.82683432e-01, 0, 3.82683432e-01, 7.07106781e-01, 9.23879533e-01, 1],
                                 [0, -3.82683432e-01, -7.07106781e-01, -9.23879533e-01, -1, -9.23879533e-01,
                                  -7.07106781e-01, -3.82683432e-01, 0, 3.82683432e-01, 7.07106781e-01,
                                  9.23879533e-01, 1, 9.23879533e-01, 7.07106781e-01, 3.82683432e-01, 0]]]]
    ],
)
def test_poly(pattern: Pattern, poly_list: List[np.ndarray]):
    for i, poly in enumerate(pattern.geoms):
        np.testing.assert_allclose(poly, poly_list[i], atol=1e-6)


@pytest.mark.parametrize(
    "pattern, poly_list",
    [
        [UNIT_BOX - UNIT_BOX.copy.scale(0.5, 0.5), [[[0.5, 0., 0., 0.25, 0.25, 0., 0., 0.5, 0.5],
                                                     [-0.5, -0.5, -0.25, -0.25, 0.25, 0.25, 0.5, 0.5, -0.5]],
                                                    [[0., -0.5, -0.5, 0., 0., -0.25, -0.25, 0., 0.],
                                                     [-0.5, -0.5, 0.5, 0.5, 0.25, 0.25, -0.25, -0.25, -0.5]]]],
        [Circle(resolution=3) - Circle(resolution=3).scale(0.5, 0.5), [[[1., 0.866025, 0.5, 0., 0., 0.25,
                                                                         0.433012, 0.5, 0.433012, 0.25, 0., -0.,
                                                                         0.5, 0.866025, 1.],
                                                                        [0., -0.5, -0.866025, -1., -0.5, -0.433012,
                                                                         -0.25, 0., 0.25, 0.433012, 0.5, 1.,
                                                                         0.866025, 0.5, 0.]],
                                                                       [[0., -0.5, -0.866025, -1., -0.866025, -0.5,
                                                                         -0., 0., -0.25, -0.433012, -0.5, -0.433012,
                                                                         -0.25, 0., 0.],
                                                                        [-1., -0.866025, -0.5, -0., 0.5, 0.866025,
                                                                         1., 0.5, 0.433012, 0.25, 0., -0.25,
                                                                         -0.433012, -0.5, -1.]]]],
    ],
)
def test_poly_with_hole(pattern: Pattern, poly_list: List[np.ndarray]):
    for i, poly in enumerate(split_holes(pattern.shapely_union())):
        np.testing.assert_allclose(poly_points(poly).T, poly_list[i], atol=1e-5)
