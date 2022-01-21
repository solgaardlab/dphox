from typing import Union

import numpy as np

from .parametric import arc, circular_bend, link, spiral, turn
from .path import Curve
from .pattern import Pattern
from .port import Port
from .typing import Optional
from .utils import DEFAULT_RESOLUTION


def _turn_connect_angle_solve(start: Port, end: Port, start_r: float, end_r: float):
    """Solve the turn connect problem with the least possible bending (try 16 options) using nazca strategy."""
    # Curve up or down at start or end ports for a total of four circle orientations
    circle_orientations = np.array([(1, 1), (1, -1), (-1, 1), (-1, -1)])

    # Get the list of possible angles; we eventually want to find the shortest possible path between the ports.
    possible_angles = []
    possible_lengths = []

    # edge case for circular bend connection
    # if start_r == end_r and np.linalg.norm(start.xy - end.xy) == 2 * start_r * np.cos(start.a - end.a):
    #     return (-90, -90), 0

    for co in circle_orientations:
        # define the center of the circles
        start_c = start.xy - co[0] * start.normal(start_r)
        end_c = end.xy - co[1] * end.normal(end_r)
        end_a = np.radians(end.a)
        start_a = np.radians(start.a - 180)
        diff_c = end_c - start_c
        s = np.linalg.norm(diff_c)
        rr = co @ np.array((start_r, end_r))
        # angle through circle centers (for aligning the circles along axis)
        circle_a = np.arctan2(*diff_c[::-1])
        if np.abs(rr) <= s:
            # angle due to the difference in radii when circles are horizontally aligned)
            corr_a = np.pi / 2 if rr == s == 0 else np.arcsin(rr / s)
            angles = (circle_a - corr_a - np.array((start_a, end_a))) * np.array((-1, 1))
            angles = angles % (2 * np.pi) - (1 + np.array((-1, 1)) * co) * np.pi
            possible_angles.append(-angles)
            possible_lengths.append(np.abs(s * np.cos(corr_a)))
    if not possible_angles:
        raise ValueError(
            "No solution found for the turn connector. Make sure the radii of the turns are sufficiently"
            f"small to connect {start} to {end}. One possible solution: reduce the turn radius.")
    idx = np.argmin(np.sum(np.abs(np.array(possible_angles)), axis=1))
    return np.degrees(possible_angles[idx]), possible_lengths[idx]


def turn_connect(start: Port, end: Port, radius: float, radius_end: Optional[float] = None, euler: float = 0,
                 resolution: int = DEFAULT_RESOLUTION, include_width: bool = True) -> Union[Pattern, Curve]:
    """Turn connect.

    Args:
        start: Start port
        end: End port
        radius: Start turn effective radius
        radius_end: End turn effective radius (use :code:`radius` if :code:`None`)
        euler: Euler path contribution (see :code:`turn` method).
        resolution: Number of evaluations for the turns.
        include_width: Whether to include the width (cubic taper) of the turn connect

    Returns:
        A path connecting the start and end port using a turn-straight-turn approach

    """
    start_r = radius
    end_r = start_r if radius_end is None else radius_end
    angles, length = _turn_connect_angle_solve(start.copy.flip(), end.copy.flip(), start_r, end_r)
    curve = link(turn(start_r, angles[0], euler, resolution) if angles[0] % 360 != 0 else None,
                 length, turn(end_r, angles[1], euler, resolution) if angles[1] % 360 != 0 else None).to(start)

    return curve.path(start.w) if include_width else curve


def manhattan_route(start: Port, lengths: np.ndarray, include_width: bool = True):
    """Manhattan route (intended for metal routing).

    Starting with horizontal and switching back and forth with vertical, make alternating turns.
    A positive length indicates moving in a +x or +y direction, whereas a negative length
    indicates moving in a -x or -y direction.

    Args:
        start: Start port for the route.
        lengths: List of dx and dy segments (alternating left and right turns).
        include_width: Include width returns a path instead of a curve.

    Returns:
        The manhattan route path.

    """
    lengths = np.array(lengths) if not isinstance(lengths, np.ndarray) else lengths
    xs = np.hstack((0, np.tile(np.cumsum(lengths[::2]), 2).reshape((2, -1)).T.flatten()))[:lengths.size + 1]
    ys = np.hstack(((0, 0), np.tile(np.cumsum(lengths[1::2]), 2).reshape((2, -1)).T.flatten()))[:lengths.size + 1]
    points = np.vstack((xs, ys)).T
    path = Curve(points)
    if include_width:
        path = Pattern(path.shapely.buffer(start.w, join_style=2, cap_style=2))
    return path.to(start)


def spiral_delay(n_turns: int, min_radius: float, separation: float,
                 resolution: int = 1000, turn_resolution: int = DEFAULT_RESOLUTION):
    """Spiral delay (large waveguide delay in minimal area).

    Args:
        n_turns: Number of turns in the spiral
        min_radius: Minimum radius of the spiral (affects the inner part of the design)
        separation: Separation of waveguides in the spiral.
        resolution: Number of evaluations for the spiral.
        turn_resolution: Number of evaluations for the turns.

    Returns:
        The spiral delay waveguide.
    """
    spiral_ = spiral(n_turns, min_radius, theta_offset=2 * np.pi,
                     separation_scale=separation, resolution=resolution)
    bend = circular_bend(min_radius, 180, resolution=turn_resolution)
    curve = link(spiral_.copy.reverse(), bend.copy.reflect(), bend, spiral_)
    start, end = curve.port['a0'], curve.port['b0']
    start_section = turn_connect(start, Port(start.x - 3 * min_radius, start.y, 0), min_radius,
                                 resolution=turn_resolution, include_width=False)
    end_section = turn_connect(end, Port(end.x + 3 * min_radius, end.y, 180), min_radius,
                               resolution=turn_resolution, include_width=False)
    return link(start_section.reverse(), curve.reflect(), end_section)


def loopify(curve: Curve, radius: float, euler: float = 0, resolution: int = DEFAULT_RESOLUTION):
    """Automatically create a loop by connecting the ends of a curve

    Note:
        This only works in some cases, as there are no self-intersection checks.

    Args:
        curve: Curve to loopify.
        radius: Radius of the turns.
        euler: Euler parameter for the turn.
        resolution: Number of evaluations for the curves in the loop.

    Returns:
        The loopified curve.

    """
    return link(curve, turn_connect(curve.port['b0'], curve.port['a0'], radius, euler=euler, resolution=resolution))

