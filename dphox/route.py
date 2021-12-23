import numpy as np
from pydantic.dataclasses import dataclass
from shapely.geometry import JOIN_STYLE, LineString

from .path import Curve
from .port import Port
import scipy.interpolate as interpolate
from .parametric import loopback, link, trombone, straight, turn, spiral, circular_bend
from .transform import rotate2d
from .pattern import Pattern
from .typing import Optional
from .utils import fix_dataclass_init_docs, NUM_EVALUATIONS


def _turn_connect_angle_solve(start: Port, end: Port, start_r: float, end_r: float):
    """Solve the turn connect problem with the least possible bending (try 16 options) using nazca strategy."""
    # Curve up or down at start or end ports for a total of four circle orientations
    circle_orientations = np.array([(1, 1), (1, -1), (-1, 1), (-1, -1)])

    # Get the list of possible angles; we eventually want to find the shortest possible path between the ports.
    possible_angles = []
    possible_lengths = []
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
            corr_a = np.arcsin(rr / s)
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
                 include_width: bool = True):
    """Turn connect.

    Args:
        start: Start port
        end: End port
        radius: Start turn effective radius
        radius_end: End turn effective radius (use :code:`radius` if :code:`None`)
        euler: Euler path contribution (see :code:`turn` method).
        include_width: Whether to include the width (cubic taper) of the turn connect

    Returns:
        A path connecting the start and end port using a turn-straight-turn approach

    """
    start_r = radius
    end_r = start_r if radius_end is None else radius_end
    angles, length = _turn_connect_angle_solve(start.copy.flip(), end.copy.flip(), start_r, end_r)
    curve = link(turn(start_r, angles[0], euler),
                 straight(length),
                 turn(end_r, angles[1], euler)).to(start)

    return curve.path(start.w) if include_width else curve


def manhattan_route(start: Port, lengths: np.ndarray, bezier_evaluations: int = 0, include_width: bool = True):
    """Manhattan route.

    Args:
        start: Start port.
        lengths: List of dx and dy segments (alternating).
        bezier_evaluations: If zero, the path is not smoothed, otherwise, specify the number of bezier evaluations
            for each segment.
        include_width: Include width returns a path instead of a curve.

    Returns:
        The manhattan route path.

    """
    lengths = np.array(lengths) if not isinstance(lengths, np.ndarray) else lengths
    xs = np.hstack((0, np.tile(np.cumsum(lengths[::2]), 2).reshape((2, -1)).T.flatten()))[:lengths.size + 1]
    ys = np.hstack(((0, 0), np.tile(np.cumsum(lengths[1::2]), 2).reshape((2, -1)).T.flatten()))[:lengths.size + 1]
    points = np.vstack((xs, ys)).T
    if bezier_evaluations:
        midpoints = np.vstack((points[0], (points[1:] + points[:-1]) / 2, points[-1]))
        tck = interpolate.splprep(midpoints, s=0, per=True)[0]
        # create interpolated lists of points
        xint, yint = interpolate.splev(np.linspace(0, 1, bezier_evaluations * lengths.size), tck)
        segments.append(points)
    else:
        path = Curve(points)
        if include_width:
            path = Pattern(path.shapely.buffer(start.w, join_style=2, cap_style=2))

    return path.to(start)


def spiral_delay(n_turns: int, min_radius: float, separation: float, num_evaluations: int = NUM_EVALUATIONS):
    spiral_ = spiral(n_turns, min_radius, theta_offset=2 * np.pi,
                     separation_scale=separation, num_evaluations=num_evaluations)
    bend = circular_bend(min_radius, 180)
    curve = link(spiral_.copy.reverse().flip_ends(), bend.copy.reflect(), bend, spiral_)
    start, end = curve.port['a0'], curve.port['b0']
    start_section = turn_connect(start, Port(start.x - 3 * min_radius, start.y, 0), min_radius, include_width=False)
    end_section = turn_connect(end, Port(end.x + 3 * min_radius, end.y, 180), min_radius, include_width=False)
    return link(start_section.reverse().flip_ends(), curve.reflect(), end_section)


@fix_dataclass_init_docs
@dataclass
class Interposer(Pattern):
    """Pitch-changing array of waveguides with path length correction.

    Args:
        waveguide_w: The waveguide width.
        n: The number of I/O (waveguides) for interposer.
        init_pitch: The initial pitch (distance between successive waveguides) entering the interposer.
        radius: The radius of bends for the interposer.
        trombone_radius: The trombone bend radius for path equalization.
        final_pitch: The final pitch (distance between successive waveguides) for the interposer.
        self_coupling_extension_extent: The self coupling for alignment, which is useful since a major use case of
            the interposer is for fiber array coupling.
        additional_x: The additional horizontal distance (useful in fiber array coupling for wirebond clearance).
        num_trombones: The number of trombones for path equalization.
        trombone_at_end: Whether to use a path-equalizing trombone near the waveguides spaced at :code:`final_period`.
    """
    waveguide_w: float
    n: int
    init_pitch: float
    radius: float
    euler: float = 0
    trombone_radius: Optional[float] = None
    final_pitch: Optional[float] = None
    self_coupling_final: bool = True
    self_coupling_init: bool = False
    self_coupling_radius: float = 5
    self_coupling_extension: float = 0
    additional_x: float = 0
    num_trombones: int = 1
    trombone_at_end: bool = True

    def __init__(self):
        w = self.waveguide_w
        self.trombone_radius = self.radius if self.trombone_radius is None else self.trombone_radius
        self.final_pitch = self.init_pitch if self.final_pitch is None else self.final_pitch
        pitch_diff = self.final_pitch - self.init_pitch
        paths = []
        init_pos = np.zeros((self.n, 2))
        final_pos = np.zeros_like(init_pos)
        radius = pitch_diff / 2 if not self.radius else self.radius
        angle_r = np.sign(pitch_diff) * np.arccos(1 - np.abs(pitch_diff) / 4 / radius)
        angled_length = np.abs(pitch_diff / np.sin(angle_r))
        x_length = np.abs(pitch_diff / np.tan(angle_r))
        mid = int(np.ceil(self.n / 2))
        max_length_diff = (angled_length - x_length) * (mid - 1)
        self.num_trombones = int(np.ceil(max_length_diff / 2 / (self.final_pitch - 3 * radius))) \
            if not self.num_trombones else self.num_trombones
        for idx in range(self.n):
            init_pos[idx] = np.asarray((0, self.init_pitch * idx))
            length_diff = (angled_length - x_length) * idx if idx < mid else (angled_length - x_length) * (
                    self.n - 1 - idx)
            segments = []
            trombone_section = [trombone(w, self.trombone_radius,
                                         length_diff / 2 / self.num_trombones, self.euler)] * self.num_trombones
            if not self.trombone_at_end:
                segments += trombone_section
            segments.append(straight(self.waveguide_w, self.additional_x))
            if idx < mid:
                segments += [turn(w, radius, -angle_r, self.euler), straight(w, angled_length * (mid - idx - 1)),
                             turn(w, radius, angle_r, self.euler), straight(w, x_length * (idx + 1))]
            else:
                segments += [turn(w, radius, angle_r, self.euler), straight(w, angled_length * (mid - self.n + idx)),
                             turn(w, radius, -angle_r, self.euler), straight(w, x_length * (self.n - idx))]
            if self.trombone_at_end:
                segments += trombone_section
            paths.append(Path(segments))
            final_pos[idx] = np.asarray((paths[-1].x(), paths[-1].y()))

        port = {
            **{f'a{idx}': Port(*init_pos[idx], -180, w=self.waveguide_w) for idx in range(self.n)},
            **{f'b{idx}': Port(*final_pos[idx], w=self.waveguide_w) for idx in range(self.n)}
        }

        if self.self_coupling_final:
            dx, dy = final_pos[0, 0], final_pos[0, 1]
            p = self.final_pitch
            port = Port(dx, dy - p)
            extension = (self.self_coupling_extension, p * (self.n + 1) - 6 * radius)
            paths.append(loopback(w, radius, self.euler, extension).to())
        if self.self_coupling_init:
            dx, dy = init_pos[0, 0], init_pos[0, 1]
            p = self.init_pitch
            port = Port(dx, dy - p)
            extension = (self.self_coupling_extension, p * (self.n + 1) - 6 * radius)
            paths.append(loopback(w, radius, self.euler, extension).to())

        super().__init__(*paths)
        self.self_coupling_path = None if self.self_coupling_extension_extent is None else paths[-1]
        self.paths = paths
        self.init_pos = init_pos
        self.final_pos = final_pos
        self.port = port
