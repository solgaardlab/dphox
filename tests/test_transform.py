import numpy as np
import pytest
from typing import Optional, Tuple

from dphox.pattern import Pattern, Box

BOX = Box((1, 1))


@pytest.mark.parametrize(
    "pattern, origin, horiz, expected_polygon",
    [
        [BOX, (1, 1), True, [[1.5, 1.5, 2.5, 2.5, 1.5], [-0.5, 0.5, 0.5, -0.5, -0.5]]],
        [BOX, (-1, 0.5), True, [[-2.5, -2.5, -1.5, -1.5, -2.5], [-0.5, 0.5, 0.5, -0.5, -0.5]]],
        [BOX, (1, 1), False, [[0.5, 0.5, -0.5, -0.5, 0.5], [2.5, 1.5, 1.5, 2.5, 2.5]]],
        [BOX, (-1, 0.5), False, [[0.5, 0.5, -0.5, -0.5, 0.5], [1.5, 0.5, 0.5, 1.5, 1.5]]]
    ],
)
def test_reflect(pattern: Pattern, origin: Optional[Tuple[int, int]], horiz: bool, expected_polygon: np.ndarray):
    np.testing.assert_allclose(np.asarray(pattern.copy.reflect(origin, horiz).geoms[0]),
                               expected_polygon)


@pytest.mark.parametrize(
    "pattern, origin, angle, expected_polygon",
    [
        [BOX, (0, 0), 90, [[0.5, -0.5, -0.5, 0.5, 0.5],
                           [0.5, 0.5, -0.5, -0.5, 0.5]]],
        [BOX, (1, 1), 90, [[2.5, 1.5, 1.5, 2.5, 2.5], [0.5, 0.5, -0.5, -0.5, 0.5]]],
        [BOX, (1, 1), 70, [[2.238529, 1.298836, 0.956816, 1.896509, 2.238529],
                           [0.017123, 0.359144, -0.580549, -0.922569, 0.017123]]],
        [BOX, (-1, 0.5), 20, [[0.751559, 0.409539, -0.530154, -0.188134, 0.751559],
                              [0.073338, 1.01303, 0.67101, -0.268683, 0.073338]]],
        [BOX, (1, 1), 45, [[1.707107, 1., 0.292893, 1., 1.707107],
                           [-0.414214, 0.292893, -0.414214, -1.12132, -0.414214]]],
        [BOX, (-1, 0.5), 90, [[0., -1., -1., 0., 0.], [2., 2., 1., 1., 2.]]]
    ],
)
def test_rotate(pattern: Pattern, origin: Optional[Tuple[int, int]], angle: float, expected_polygon: np.ndarray):
    np.testing.assert_allclose(np.asarray(pattern.copy.rotate(angle, origin).geoms[0]),
                               expected_polygon, rtol=3e-5)


@pytest.mark.parametrize(
    "pattern, xfact, yfact, origin, expected_polygon",
    [
        [BOX, 2, 1, None, [[1., 1., -1., -1., 1.], [-0.5, 0.5, 0.5, -0.5, -0.5]]],
        [BOX, 1, 2, (3, 0), [[0.5, 0.5, -0.5, -0.5, 0.5], [-1., 1., 1., -1., -1.]]],
        [BOX, 2, 2, (1, 1), [[0., 0., -2., -2., 0.], [-2., 0., 0., -2., -2.]]],
        [BOX, 3, 1, (1, 1), [[-0.5, -0.5, -3.5, -3.5, -0.5], [-0.5, 0.5, 0.5, -0.5, -0.5]]]
    ],
)
def test_scale(pattern: Pattern, xfact: int, yfact: int, origin: Optional[Tuple[int, int]],
               expected_polygon: np.ndarray):
    np.testing.assert_allclose(np.asarray(pattern.copy.scale(xfact, yfact, origin=origin).geoms[0]),
                               expected_polygon, rtol=3e-5)
