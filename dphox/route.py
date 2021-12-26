from dataclasses import dataclass
from typing import Union

import numpy as np

from .foundry import CommonLayer
from .device import Device
from .passive import FocusingGrating
from .path import Curve
from .pattern import Pattern
from .port import Port
from .prefab import arc, circular_bend, link, loopback, spiral, trombone, turn
from .typing import Optional
from .utils import DEFAULT_RESOLUTION, fix_dataclass_init_docs


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


def semicircle(radius: float, resolution: int = DEFAULT_RESOLUTION):
    """A semicircle pattern.

    Args:
        radius: The radius of the semicircle.
        resolution: Number of evaluations for each turn.

    Returns:
        The semicircle pattern.

    """
    return arc(radius, 180, resolution=resolution).pattern


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
    final_pitch: float
    radius: Optional[float] = None
    euler: float = 0
    trombone_radius: float = 5
    self_coupling_final: bool = True
    self_coupling_init: bool = False
    self_coupling_radius: float = None
    self_coupling_extension: float = 0
    additional_x: float = 0
    num_trombones: int = 1
    trombone_at_end: bool = True

    def __post_init__(self):
        w = self.waveguide_w
        pitch_diff = self.final_pitch - self.init_pitch
        self.radius = np.abs(pitch_diff) / 2 if self.radius is None else self.radius
        r = self.radius

        paths = []
        init_pos = np.zeros((2, self.n))
        init_pos[1] = self.init_pitch * np.arange(self.n)
        init_pos = init_pos.T
        final_pos = np.zeros_like(init_pos)

        if np.abs(1 - np.abs(pitch_diff) / 4 / r) > 1:
            raise ValueError(f"Radius {r} needs to be at least abs(pitch_diff) / 2 = {np.abs(pitch_diff) / 2}.")
        angle_r = np.sign(pitch_diff) * np.arccos(1 - np.abs(pitch_diff) / 4 / r)
        angled_length = np.abs(pitch_diff / np.sin(angle_r))
        x_length = np.abs(pitch_diff / np.tan(angle_r))
        mid = int(np.ceil(self.n / 2))
        angle = np.degrees(angle_r)

        for idx in range(self.n):
            init_pos[idx] = np.asarray((0, self.init_pitch * idx))
            length_diff = (angled_length - x_length) * idx if idx < mid else (angled_length - x_length) * (
                    self.n - 1 - idx)
            segments = []
            trombone_section = [trombone(self.trombone_radius,
                                         length_diff / 2 / self.num_trombones, self.euler)] * self.num_trombones
            if not self.trombone_at_end:
                segments += trombone_section
            segments.append(self.additional_x)
            if idx < mid:
                segments += [turn(r, -angle, self.euler), angled_length * (mid - idx - 1),
                             turn(r, angle, self.euler), x_length * (idx + 1)]
            else:
                segments += [turn(r, angle, self.euler), angled_length * (mid - self.n + idx),
                             turn(r, -angle, self.euler), x_length * (self.n - idx)]
            if self.trombone_at_end:
                segments += trombone_section
            paths.append(link(*segments).path(w).to(init_pos[idx]))
            final_pos[idx] = paths[-1].port['b0'].xy

        if self.self_coupling_final:
            scr = self.final_pitch / 4 if self.self_coupling_radius is None else self.self_coupling_radius
            dx, dy = final_pos[0, 0], final_pos[0, 1]
            p = self.final_pitch
            port = Port(dx, dy - p, -180)
            extension = (self.self_coupling_extension, p * (self.n + 1) - 6 * scr)
            paths.append(loopback(scr, self.euler, extension).path(w).to(port))
        if self.self_coupling_init:
            scr = self.init_pitch / 4 if self.self_coupling_radius is None else self.self_coupling_radius
            dx, dy = init_pos[0, 0], init_pos[0, 1]
            p = self.init_pitch
            port = Port(dx, dy - p)
            extension = (self.self_coupling_extension, p * (self.n + 1) - 6 * scr)
            paths.append(loopback(scr, self.euler, extension).path(w).to(port))

        port = {**{f'a{idx}': Port(*init_pos[idx], -180, w=self.waveguide_w) for idx in range(self.n)},
                **{f'b{idx}': Port(*final_pos[idx], w=self.waveguide_w) for idx in range(self.n)},
                'l0': paths[-1].port['a0'], 'l1': paths[-1].port['b0']}

        super().__init__(*paths)
        self.self_coupling_path = None if self.self_coupling_extension is None else paths[-1]
        self.paths = paths
        self.init_pos = init_pos
        self.final_pos = final_pos
        self.port = port

    def device(self, layer: str = CommonLayer.RIDGE_SI):
        return Device('interposer', [(self, layer)]).set_port(self.port_copy)

    def with_gratings(self, grating: FocusingGrating, layer: str = CommonLayer.RIDGE_SI):
        interposer = self.device(layer)
        interposer.port = self.port
        for idx in range(6):
            interposer.place(grating, self.port[f'b{idx}'], from_port=grating.port['a0'])
        interposer.place(grating, self.port['l0'], from_port=grating.port['a0'])
        interposer.place(grating, self.port['l1'], from_port=grating.port['a0'])
        return interposer
