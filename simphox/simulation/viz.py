import k3d
import numpy as np
from typing import Tuple, Optional

from matplotlib import colors as mcolors


def get_extent_2d(ax, shape, spacing: Optional[float] = None):
    """

    Args:
        ax: Matplotlib axis handle
        shape: shape of the elements to plot
        spacing: spacing between grid points (assumed to be isotropic)

    Returns:

    """
    if spacing:  # in microns!
        ax.set_ylabel(r'$y$ ($\mu$m)')
        ax.set_xlabel(r'$x$ ($\mu$m)')
    return (0, shape[0] * spacing, 0, shape[1] * spacing) if spacing else (0, shape[0], 0, shape[1])


def plot_eps_2d(ax, eps: np.ndarray, spacing: Optional[float] = None, cmap: str = 'nipy_spectral'):
    """

    Args:
        ax: Matplotlib axis handle
        eps: epsilon permittivity
        spacing: spacing between grid points (assumed to be isotropic)
        cmap: colormap for field array (we highly recommend RdBu)

    Returns:

    """
    extent = get_extent_2d(ax, eps.shape, spacing)
    ax.imshow(eps.T, cmap=cmap, origin='lower left', alpha=1, extent=extent)


def plot_field_2d(ax, field: np.ndarray, eps: Optional[np.ndarray] = None, spacing: Optional[float] = None,
                  cmap: str = 'RdBu', alpha: float = 0.8):
    """

    Args:
        ax: Matplotlib axis handle
        field: field to plot
        eps: epsilon permittivity for overlaying field onto materials
        spacing: spacing between grid points (assumed to be isotropic)
        cmap: colormap for field array (we highly recommend RdBu)
        alpha: transparency of the plots to visualize overlay

    Returns:

    """
    extent = get_extent_2d(ax, field.shape, spacing)
    if eps is not None:
        plot_eps_2d(ax, eps, spacing)
    im_val = field * np.sign(field.flat[np.abs(field).argmax()])
    norm = mcolors.DivergingNorm(vcenter=0, vmin=-im_val.max(), vmax=im_val.max())
    ax.imshow(im_val.T, cmap=cmap, origin='lower left', alpha=alpha, extent=extent, norm=norm)


def plot_power_2d(ax, power: np.ndarray, eps: Optional[np.ndarray] = None, spacing: Optional[float] = None,
                  cmap: str = 'hot', alpha: float = 0.8):
    """

    Args:
        ax: Matplotlib axis handle
        power: power array of size (X, Y)
        eps: epsilon for overlay with materials
        spacing: spacing between grid points (assumed to be isotropic)
        cmap: colormap for power array
        alpha: transparency of the plots to visualize overlay

    Returns:

    """
    extent = get_extent_2d(ax, power.shape, spacing)
    if eps is not None:
        plot_eps_2d(ax, eps, spacing)
    ax.imshow(power.T, cmap=cmap, origin='lower left', alpha=alpha, extent=extent)


def plot_power_3d(plot: k3d.Plot, power: np.ndarray, axis: int = 0, eps: Optional[np.ndarray] = None,
                  spacing: float = None, color_range: Tuple[float, float] = None, alpha: float = 100,
                  samples: float = 1200):
    """Plot the 3d power in a notebook given the fields :math:`E` and :math:`H`.

    Args:
        plot: K3D plot handle (NOTE: this is for plotting in a Jupyter notebook)
        power: power (either Poynting field of size (3, X, Y, Z) or power of size (X, Y, Z))
        axis: pick the correct axis if inputting power in Poynting field
        eps: permittivity (if specified, plot with default options)
        spacing: spacing between grid points (assumed to be isotropic)
        color_range: color range for visualization (if none, use half maximum value of field)
        alpha: alpha for k3d plot
        samples: samples for k3d plot rendering

    Returns:

    """
    power = power[axis] if power.ndim == 4 else power
    power = power.transpose((2, 1, 0))
    color_range = (0, np.max(power[axis]) / 2) if color_range is None else color_range

    bounds = [0, power.shape[0] * spacing, 0, power.shape[1] * spacing, 0, power.shape[2] * spacing]

    if eps is not None:
        plot_eps_3d(plot, eps)  # use defaults for now

    power_volume = k3d.volume(
        power[axis],
        alpha_coef=alpha,
        samples=samples,
        color_range=color_range,
        color_map=(np.array(k3d.colormaps.matplotlib_color_maps.hot).reshape(-1, 4)).astype(np.float32),
        compression_level=8,
        name='power'
    )
    power_volume.transform.bounds = bounds
    plot += power_volume


def plot_field_3d(plot: k3d.Plot, field: np.ndarray, axis: int = 1,
                  eps: Optional[np.ndarray] = None, imag: bool = False, spacing: Optional[float] = None,
                  alpha: float = 100, samples: float = 1200, color_range: Tuple[float, float] = (0, 0.01)):
    """

    Args:
        plot: K3D plot handle (NOTE: this is for plotting in a Jupyter notebook)
        field: field to plot
        axis: pick the correct axis for power in Poynting vector form
        eps: permittivity (if specified, plot with default options)
        imag: whether to use the imaginary (instead of real) component of the field
        spacing: spacing between grid points (assumed to be isotropic)
        color_range: color range for visualization (if none, use half maximum value of field)
        alpha: alpha for k3d plot
        samples: samples for k3d plot rendering

    Returns:

    """
    field = field[axis] if field.ndim == 4 else field
    field = field.transpose((2, 1, 0))
    field = field.imag if imag else field.real
    color_range = np.asarray((0, np.max(field)) if color_range is None else color_range)

    bounds = [0, field.shape[0] * spacing, 0, field.shape[1] * spacing, 0, field.shape[2] * spacing]

    if eps is not None:
        plot_eps_3d(plot, eps)  # use defaults for now

    pos_e_volume = k3d.volume(
        volume=field,
        alpha_coef=alpha,
        samples=samples,
        color_range=color_range,
        color_map=(np.array(k3d.colormaps.matplotlib_color_maps.RdBu).reshape(-1, 4)).astype(np.float32),
        compression_level=8,
        name='pos'
    )

    neg_e_volume = k3d.volume(
        volume=-field,
        alpha_coef=alpha,
        samples=1200,
        color_range=-color_range,
        color_map=(np.array(k3d.colormaps.matplotlib_color_maps.RdBu_r).reshape(-1, 4)).astype(np.float32),
        compression_level=8,
        name='neg'
    )
    neg_e_volume.transform.bounds = bounds
    pos_e_volume.transform.bounds = bounds

    plot += neg_e_volume
    plot += pos_e_volume


def plot_eps_3d(plot: k3d.Plot, eps: Optional[np.ndarray] = None, color_range: Tuple[float, float] = None,
                alpha: float = 100, samples: float = 1200):

    color_range = (1, np.max(eps)) if color_range is None else color_range

    eps_volume = k3d.volume(
        eps,
        alpha_coef=alpha,
        samples=samples,
        color_map=(np.array(k3d.colormaps.matplotlib_color_maps.RdBu).reshape(-1, 4)).astype(np.float32),
        compression_level=8,
        color_range=color_range,
        name='epsilon'
    )

    bounds = [0, eps.shape[2] / 20, 0, eps.shape[1] / 20, 0, eps.shape[0] / 20]

    eps_volume.transform.bounds = bounds

    plot += eps_volume
