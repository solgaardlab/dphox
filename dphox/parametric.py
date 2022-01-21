import numpy as np
from scipy.special import fresnel

from .path import Curve, link, straight
from .transform import translate2d
from .typing import Callable, CurveTuple, Float2, Optional, Tuple, Union
from .utils import DEFAULT_RESOLUTION


def parametric_curve(curve_fn: Callable, resolution: int):
    """Define curve (parametric function) given :math:`t \\in [0, 1]` input.

    Since the tangent direction is dependent on a tangents along the path, the curve definition is much more accurate
    when the curve parametric function returns both the path points and the tangent directions.
    Therefore, when possible, definitely ensure that the path returns its points and its tangent;
    the tangent only needs to be defined by its direction (the angle matters, not the magnitude).

    The parametric path is the core to all path operations in dphox.

    Args:
        curve_fn: The function returning either the points in the path or a tuple of points and tangents.
        resolution: The number of evaluations from 0 to 1 to evaluate the parametric function.

    Returns:
        The curve represented by the curve function.

    """
    t = np.linspace(0, 1, resolution)[:, np.newaxis]
    points = curve_fn(t)
    if isinstance(points, tuple):
        points, tangents = points
    else:
        tangents = np.gradient(points, axis=0)
    return Curve(CurveTuple(points=points.T, tangents=tangents.T))


def taper(length: float, resolution: int = DEFAULT_RESOLUTION):
    """Basically the same as a straight line along the x axis, but here we make extra evaluations by default.

    Note:
        This is the same code as a straight segment, but we use this boilerplate function to implement
        a default number of evaluations more than 2, which is compatible with a tapered straight segment.

    Args:
        length: Length of the taper.
        resolution: Number of evaluations along the taper.

    Returns:
        A straight segment that can be tapered.

    """

    def _linear(t: np.ndarray):
        return np.hstack((t * length, np.zeros_like(t))), np.hstack((np.ones_like(t), np.zeros_like(t)))

    return parametric_curve(_linear, resolution=resolution)


def cubic_bezier(pole_1: np.ndarray, pole_2: np.ndarray, pole_3: np.ndarray,
                 resolution: int = DEFAULT_RESOLUTION):
    """The cubic bezier function, which is very useful for both smooth paths and sbends.

    The formula for a cubic bezier function follows the expression:

    .. math::
        \\boldsymbol{p}(t) = 3t(1 - t)^2 * \\boldsymbol{c}_1 + 3 * (1 - t) * t ** 2 * \\boldsymbol{c}_2 +
        t ** 3 * \\boldsymbol{c}_3

    Args:
        pole_1: Pole 1 :math:`\\boldsymbol{c}_1`
        pole_2: Pole 2 :math:`\\boldsymbol{c}_2`
        pole_3: Pole 3 :math:`\\boldsymbol{c}_3`
        resolution: Number of evaluations along the cubic bezier.

    Returns:

    """

    def _bezier(t: np.ndarray):
        path = 3 * (1 - t) ** 2 * t * pole_1 + 3 * (1 - t) * t ** 2 * pole_2 + t ** 3 * pole_3
        tangents = 3 * (1 - t) ** 2 * pole_1 + 6 * (1 - t) * t * (pole_2 - pole_1) + 3 * t ** 2 * (pole_3 - pole_2)
        return path, tangents

    return parametric_curve(_bezier, resolution=resolution)


def bezier_sbend(bend_x: float, bend_y: float, resolution: int = DEFAULT_RESOLUTION):
    """Bezier sbend.

    The formula for a cubic bezier function follows the expression:

    .. math::
        \\boldsymbol{p}(t) = 3t(1 - t)^2 * \\boldsymbol{c}_1 + 3 * (1 - t) * t ** 2 * \\boldsymbol{c}_2 +
        t ** 3 * \\boldsymbol{c}_3

    Here, we just choose a specific set of poles :math:`\\boldsymbol{c}` to construct the bezier sbend.

    Args:
        bend_x: Change in :math:`x` due to the bend.
        bend_y: Change in :math:`y` due to the bend.
        resolution: Number of evaluations along the bezier sbend.

    Returns:
        A function mapping 0 to 1 to the width of the taper along the path linestring.

    """
    pole_1 = np.asarray((bend_x / 2, 0))
    pole_2 = np.asarray((bend_x / 2, bend_y))
    pole_3 = np.asarray((bend_x, bend_y))
    return cubic_bezier(pole_1, pole_2, pole_3, resolution=resolution)


def euler_bend(radius: float, angle: float = 90, resolution: int = DEFAULT_RESOLUTION):
    """Euler bend.

    The formula for an euler bend function follows the expression given radius :math:`r` and angle :math:`\\alpha`:

    .. math::
        \\boldsymbol{p}(t) &= r F(\\sqrt{2\\alpha t / \\pi}) \\
        F(t) &= (\\int_{0}^{t} \\sin(t^2) dt, \\int_{0}^{t} \\cos(t^2) dt)

    Args:
        radius: Radius of the euler bend.
        angle: Angle change of the euler bend arc.
        resolution: Number of evaluations along the euler bend.

    Returns:
        The euler bend curve.

    """
    sign = np.sign(angle)
    angle = np.abs(angle / 180 * np.pi)

    def _bend(t: np.ndarray):
        z = np.sqrt(angle * t)
        y, x = fresnel(z / np.sqrt(np.pi / 2))
        return radius * np.hstack((x, y * sign)), np.hstack((np.cos(angle * t), np.sin(angle * t) * sign))

    return parametric_curve(_bend, resolution=resolution)


def circular_bend(radius: float, angle: float = 90, resolution: int = DEFAULT_RESOLUTION):
    """Circular bend functional.

    Args:
        radius: Radius of the circular bend.
        angle: Angle change of the circular arc.
        resolution: Number of evaluations along the circular bend.

    Returns:
        The circular bend curve.

    """
    sign = np.sign(angle)
    angle = np.abs(angle / 180 * np.pi)

    def _bend(t: np.ndarray):
        x = radius * np.sin(angle * t)
        y = radius * (1 - np.cos(angle * t))
        return np.hstack((x, y * sign)), np.hstack((np.cos(angle * t), np.sin(angle * t) * sign))

    return parametric_curve(_bend, resolution=resolution)


def elliptic_bend(radius_x: float, radius_y: float, angle: float):
    """An elliptic bend of specified x and y radii.

    Args:
        radius_x: The x radius of the ellipse.
        radius_y: The y radius of the ellipse.
        angle: The final angle of the elliptic bend.

    Returns:
        The elliptic bend curve.

    """
    return circular_bend(1, np.arctan2(np.sin(angle) / radius_y, np.cos(angle) / radius_x)).scale(radius_x, radius_y)


def turn(radius: float, angle: float = 90, euler: float = 0, resolution: int = DEFAULT_RESOLUTION):
    """Turn (partial Euler) functional for angles between -90 and 90 degrees.

    Args:
        radius: Effective radius of the turn (for an equivalent circular bend).
        angle: Angle change for the turn in degrees.
        euler: Fraction of the bend :math:`p` that is an Euler bend, must satisfy :math:`0 \\leq p < 1`.
        resolution: Number of evaluations for the path.

    Returns:
        The turn curve.

    """
    if not 0 <= euler < 1:
        raise ValueError(f"Expected euler parameter to be 0 <= euler < 1 but got {euler}.")
    if euler > 0:
        euler_angle = euler * angle / 2
        circular_angle = (1 - euler) * angle / 2

        euler_curve = euler_bend(1, euler_angle, resolution=int(euler / 2 * resolution))
        circular_curve = circular_bend(1 / np.sqrt(2 * np.pi * np.radians(np.abs(euler_angle))), circular_angle,
                                       resolution=int((1 - euler) / 2 * resolution))

        curve = Curve(euler_curve, circular_curve.to(euler_curve.port['b0'])).symmetrized()
        scale = radius * 2 * np.sin(np.radians(np.abs(angle)) / 2) / np.linalg.norm(curve.points.T[-1])
        return curve.scale(scale, scale, origin=(0, 0))
    else:
        return circular_bend(radius, angle, resolution)


def arc(angle: float, radius: float, start_angle: float = None, resolution: int = DEFAULT_RESOLUTION):
    """Arc.

    Notes:
        Unlike the turn, the arc starts out at a radius away from :code:`(0, 0)` and shouldn't be used in paths.

    Args:
        angle: The angle of the arc in degrees.
        radius: The radius.
        start_angle: Start angle for the arc (default to -angle / 2 if None).
        resolution: Number of evaluations for the curve.

    Returns:
        The function mapping 0 to 1 to the curve/tangents for the arc.

    """
    angle = np.abs(angle / 180 * np.pi)
    start_angle = -angle / 2 if start_angle is None else start_angle

    def _arc(t: np.ndarray):
        angles = angle * t + start_angle
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.hstack((x, y)), np.hstack((-np.sin(angles), np.cos(angles)))

    return parametric_curve(_arc, resolution)


def grating_arc(angle: float, duty_cycle: float, n_core: float, n_clad: float,
                fiber_angle: float, wavelength: float, m: float,
                resolution: int = DEFAULT_RESOLUTION, include_width: bool = True):
    """Grating arc.

    See Also:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7407772/

    Args:
        angle: The opening angle of the grating in degrees.
        duty_cycle: duty cycle for the grating
        n_clad: clad material index of refraction.
        n_core: core material index of refraction.
        fiber_angle: angle of the fiber in degrees from horizontal (not NOT vertical).
        wavelength: wavelength accepted by the grating.
        m: grating index.
        resolution: Number of evaluations for the curve.
        include_width: Include the width (paths in the grating arc).

    Returns:
        The function mapping 0 to 1 to the curve/tangents for the provided grating parameters.

    """

    angle = np.abs(np.radians(angle))
    fiber_angle = np.abs(np.radians(fiber_angle))
    n_eff = np.sqrt(duty_cycle * n_core ** 2 + (1 - duty_cycle) * n_clad ** 2)

    def _grating_arc(t: np.ndarray):
        angles = angle * t - angle / 2
        radius = m * wavelength / (n_eff - n_clad * np.cos(fiber_angle) * np.cos(angles))
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.hstack((x, y))

    width = duty_cycle * wavelength / (n_eff - n_clad * np.cos(fiber_angle))

    curve = parametric_curve(_grating_arc, resolution)
    return curve.path(width) if include_width else curve


def turn_sbend(height: float, radius: float, euler: float = 0, resolution: int = DEFAULT_RESOLUTION):
    """Turn-based sbend (as opposed to bezier-based sbend).

    Args:
        height: Height of the sbend.
        radius: Radius of the sbend.
        euler: Fraction of the turns that should be Euler turns.
        resolution: Number of evaluations for the curve.

    Returns:
        The turn sbend curve.

    """
    h = np.abs(height)
    sign = np.sign(height)
    if h >= 2 * radius:
        angle = 90 * sign
        turn_up = turn(radius, angle, euler, resolution)
        turn_down = turn(radius, -angle, euler, resolution).to(turn_up.port['b0'])
        return Curve(turn_up, turn_down.transform(translate2d((0, (h - 2 * radius) * sign)))).coalesce()
    else:
        angle = 180 / np.pi * np.arccos(1 - h / (2 * radius)) * sign
        turn_up = turn(radius, angle, euler, resolution)
        turn_down = turn(radius, -angle, euler, resolution).to(turn_up.port['b0'])
        return Curve(turn_up, turn_down).coalesce()


def spiral(turns: int, scale: float, separation_scale: float = 1, theta_offset: float = 0,
           resolution: int = 1000):
    """Spiral (Archimedian).

    Args:
        turns: Number of 180 degree turns in the spiral function
        scale: The scale of the spiral function (maps to minimum radius in final implementation relative to scale 1).
        separation_scale: The separation scale for the spiral function (how fast to spiral out relative to scale 1)
        resolution: Number of evaluations for the curve.

    Returns:
        The spiral curve.

    """

    def _spiral(t: np.ndarray):
        theta = t * turns * np.pi + theta_offset
        radius = (theta - theta_offset) * separation_scale / scale + theta_offset
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        return scale / np.pi * np.hstack((x, y)), np.hstack((-np.sin(theta), np.cos(theta)))

    return parametric_curve(_spiral, resolution)


def bezier_dc(dx: float, dy: float, interaction_l: float, resolution: int = DEFAULT_RESOLUTION):
    """Bezier curve-based directional coupler

    Args:
        dx: Bend length (specify positive value for bend to go backwards).
        dy: Bend height (specify negative value for bend to go down).
        interaction_l: Interaction length for the directional coupler.
        resolution: Number of points to evaluate in each of the four bezier paths.

    Returns:
        The bezier dc path.

    """
    return link(
        bezier_sbend(dx, dy, resolution),
        straight(interaction_l),
        bezier_sbend(dx, -dy, resolution)
    )


def dc_path(radius: float, dy: float, interaction_l: float, euler: float = 0, resolution: int = DEFAULT_RESOLUTION):
    """Turn curve-based directional coupler path (just the bottom waveguide curve).

    Args:
        radius: Bend radius (specify positive value).
        dy: Bend height (specify negative value for bend to go down).
        interaction_l: Interaction length for the directional coupler.
        euler: Euler contribution to the directional coupler's bends.
        resolution: Number of points to evaluate in each of the four bezier paths.

    Returns:
        The turn sbend-based dc path.

    """
    return link(
        turn_sbend(dy, radius, euler, resolution),
        straight(interaction_l),
        turn_sbend(-dy, radius, euler, resolution)
    )


def mzi_path(radius: float, dy: float, interaction_l: float, arm_l: float,
             euler: float = 0, resolution: int = DEFAULT_RESOLUTION):
    """Turn curve-based MZI path (just the bottom waveguide curve).

    Args:
        radius: Bend radius (specify positive value).
        dy: Bend height (specify negative value for bend to go down).
        interaction_l: Interaction length for the directional couplers.
        arm_l: Arm length for the MZI.
        euler: Euler contribution to the directional coupler's bends.
        resolution: Number of points to evaluate in each of the four bezier paths.

    Returns:
        The turn sbend-based dc path.

    """
    return link(
        turn_sbend(dy, radius, euler, resolution),
        interaction_l,
        turn_sbend(-dy, radius, euler, resolution),
        arm_l,
        turn_sbend(dy, radius, euler, resolution),
        interaction_l,
        turn_sbend(-dy, radius, euler, resolution),
    )


def loopback(radius: float, euler: float, extension: Float2, loop_ccw: bool = True,
             resolution: int = DEFAULT_RESOLUTION):
    """Loopback path.

    Args:
        radius: Radius of the loopback turns.
        euler: Euler parameter for the turns (recommended: 0.2).
        extension: Extension x, y for the horizontal, vertical stretches of the loopback respectively.
        loop_ccw: Loop counterclockwise
        resolution: Number of evaluations for each turn.

    Returns:

    """
    s = 2 * loop_ccw - 1
    return link(
        turn(radius, -180 * s, euler, resolution=resolution),
        extension[0],
        turn(radius, 90 * s, euler, resolution=resolution),
        extension[1],
        turn(radius, 90 * s, euler, resolution=resolution),
        extension[0],
        turn(radius, -180 * s, euler, resolution=resolution)
    )


def trombone(radius: float, length: float, euler: float = 0, resolution: int = DEFAULT_RESOLUTION):
    """Trombone path.

    Args:
        radius: Radius of the trombone turns.
        length: Length / height of the trombone (minus the turn contributions).
        euler: Euler parameter for the turns (recommended: 0.2)
        resolution: Number of evaluations for each turn.

    Returns:

    """
    return link(
        turn_sbend(length + 2 * radius, radius, euler, resolution),
        turn_sbend(-length - 2 * radius, radius, euler, resolution)
    )


def bent_trombone(radius: float, length: float, euler: float = 0, resolution: int = DEFAULT_RESOLUTION):
    """Bent trombone path.

    Args:
        radius: Radius of the trombone turns.
        length: Length / height of the trombone (minus the turn contributions).
        euler: Euler parameter for the turns (recommended: 0.2)
        resolution: Number of evaluations for each turn.

    Returns:
        A bent trombone path for path balancing with fewer bends

    """
    return link(
        turn_sbend(4 * radius, radius, euler, resolution),
        straight(length + radius),
        turn(radius, -180, euler, resolution=resolution),
        straight(length),
        turn(radius, 180, euler, resolution=resolution)
    )


def right_turn(radius: float, euler: float = 0, resolution: int = DEFAULT_RESOLUTION):
    """Right turn curve from the perspective of the output port of the curve.

    Args:
        radius: Radius of the curve.
        euler: Euler bend contribution.
        resolution: Number of evaluations for each turn.

    Returns:
        The right turn curve.

    """
    return turn(radius, -90, euler=euler, resolution=resolution)


def left_turn(radius: float, euler: float = 0, resolution: int = DEFAULT_RESOLUTION):
    """Left turn curve from the perspective of the output port of the curve.

    Args:
        radius: Radius of the curve.
        euler: Euler bend contribution.
        resolution: Number of evaluations for each turn.

    Returns:
        The left turn curve.

    """
    return turn(radius, euler=euler, resolution=resolution)


def left_uturn(radius: float, length: float = 0, euler: float = 0, resolution: int = DEFAULT_RESOLUTION):
    """Left uturn from the perspective of the output port of the curve.

    Args:
        radius: Radius of the uturn.
        length: Length between the two 90 degree turns.
        euler: Euler bend contribution.
        resolution: Number of evaluations for each turn.

    Returns:
        The left uturn curve.

    """
    return link(left_turn(radius, euler, resolution), length, left_turn(radius, euler, resolution)).coalesce()


def right_uturn(radius: float, length: float = 0, euler: float = 0, resolution: int = DEFAULT_RESOLUTION):
    """Right uturn from the perspective of the output port of the curve.

    Args:
        radius: Radius of the uturn.
        length: Length between the two 90 degree turns.
        euler: Euler bend contribution.
        resolution: Number of evaluations for each turn.

    Returns:
        The right uturn curve.

    """
    return link(right_turn(radius, euler, resolution), length,
                right_turn(radius, euler, resolution)).coalesce()


def racetrack(radius: float, length: float, euler: float, resolution: int = DEFAULT_RESOLUTION):
    """A circle or ring curve of specified radius.

    Args:
        radius: The radius of the uturn bends of the racetrack.
        length: The length of the straight section of the racetrack.
        euler: The Euler turn parameter for the uturns.
        resolution: Number of evaluations for each turn.

    Returns:
        The racetrack curve.

    """
    return link(left_uturn(radius, euler, resolution), length, left_uturn(radius, euler, resolution), length)


def polytaper_fn(taper_params: Union[np.ndarray, Tuple[float, ...]]):
    """Polynomial taper functional.

    Args:
        taper_params: Polynomial taper parameter function of the form :math:`f(t; \\mathbf{x}) = \\sum_{n = 1}^N x_nt^n`

    Returns:
         A function mapping 0 to 1 to the width of the taper along the path linestring.

    """
    poly_exp = np.arange(len(taper_params), dtype=float)
    return lambda u: np.sum(taper_params * u ** poly_exp, axis=1)


def linear_taper_fn(init_w: float, final_w: float):
    """Linear taper function for parametric width/offset.

    Args:
        init_w: Initial width.
        final_w: Final width.

    Returns:
        The linear taper function.

    """
    change_w = final_w - init_w
    return polytaper_fn((init_w, change_w))


def quad_taper_fn(init_w: float, final_w: float):
    """Quadratic taper function for parametric width/offset.

    Args:
        init_w: Initial width.
        final_w: Final width.

    Returns:
        The quadratic taper function.

    """
    change_w = final_w - init_w
    return polytaper_fn((init_w, 2 * change_w, -1 * change_w))


def cubic_taper_fn(init_w: float, final_w: float):
    """Cubic taper function for parametric width/offset.

    Args:
        init_w: Initial width.
        final_w: Final width.

    Returns:
        The cubic taper function.

    """
    change_w = final_w - init_w
    return polytaper_fn((init_w, 0., 3 * change_w, -2 * change_w))


def cubic_taper(init_w: float, change_w: float, length: float, taper_length: float,
                symmetric: bool = True, taper_first: bool = True, resolution: int = DEFAULT_RESOLUTION):
    """A cubic taper waveguide pattern.

    Args:
        init_w: Initial width of the waveguide
        change_w: Change in the waveguide width.
        length: Length of the overall waveguide.
        taper_length: Length of just the waveguide portion.
        symmetric: Whether to symmetrize the waveguide.
        taper_first: Whether to taper first.
        resolution: Number of evaluations for each turn.

    Returns:
        The cubic taper waveguide.

    """
    if 2 * taper_length > length:
        raise ValueError(f"Require 2 * taper_length <= length, but got {2 * taper_length} >= {length}.")
    straight_length = length / (1 + symmetric) - taper_length
    if taper_first:
        path = link(taper(taper_length, resolution), straight_length).path((cubic_taper_fn(init_w, init_w + change_w),
                                                                            init_w + change_w))
    else:
        path = link(straight_length, taper(taper_length, resolution)).path(
            (init_w, cubic_taper_fn(init_w, init_w + change_w)))
    return path.symmetrized() if symmetric else path


def ring(radius: float, resolution: int = DEFAULT_RESOLUTION):
    """A circle or ring curve of specified radius.

    Args:
        radius: The radius of the circle.
        resolution: Number of evaluations for each turn.

    Returns:
        The circle or ring curve.

    """
    return arc(radius, 360, resolution=resolution)


def semicircle(radius: float, resolution: int = DEFAULT_RESOLUTION):
    """A semicircle pattern.

    Args:
        radius: The radius of the semicircle.
        resolution: Number of evaluations for each turn.

    Returns:
        The semicircle pattern.

    """
    return arc(radius, 180, resolution=resolution).pattern


def circle(radius: float, resolution: int = DEFAULT_RESOLUTION):
    """A circle of specified radius.

    Args:
        radius: The radius of the circle.
        resolution: Number of evaluations for each turn.

    Returns:
        The circle pattern.

    """
    return ring(radius, resolution).pattern


def ellipse(radius_x: float, radius_y: float, resolution: int = DEFAULT_RESOLUTION):
    """An ellipse of specified x and y radii.

    Args:
        radius_x: The x radius of the circle.
        radius_y: The y radius of the circle.
        resolution: Number of evaluations for each turn.

    Returns:
        The ellipse pattern.

    """
    return circle(1, resolution).scale(radius_x, radius_y).pattern
