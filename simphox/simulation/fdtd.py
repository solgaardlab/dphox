from functools import lru_cache

import numpy as np
from typing import Tuple, Callable

from .grid import SimGrid
from ..typing import Shape, Dim, GridSpacing, Optional, Union
from ..utils import pml_params, d2curl


class FDTD(SimGrid):
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray] = 1,
                 pml: Optional[Union[Shape, Dim]] = None):
        super(FDTD, self).__init__(shape, spacing, eps, pml=pml)
        self.dt = 1 / np.sqrt(np.sum(1 / self.spacing ** 2))  # includes courant condition!

        # pml (internal to the grid, so specified here!)
        if self.pml_shape is not None:
            b, c = zip(*[self._cpml(ax) for ax in range(3)])
            b_e, c_e = [b[ax][0] for ax in range(3)], [c[ax][0] for ax in range(3)]
            b_h, c_h = [b[ax][1] for ax in range(3)], [c[ax][1] for ax in range(3)]
            b_e, c_e = np.asarray(np.meshgrid(*b_e, indexing='ij')), np.asarray(np.meshgrid(*c_e, indexing='ij'))
            b_h, c_h = np.asarray(np.meshgrid(*b_h, indexing='ij')), np.asarray(np.meshgrid(*c_h, indexing='ij'))
            # for memory and time purposes, we only update the pml slices, NOT the full field
            self.pml_regions = ((slice(None), slice(None, self.pml_shape[0]), slice(None), slice(None)),
                                (slice(None), slice(-self.pml_shape[0], None), slice(None), slice(None)),
                                (slice(None), slice(None), slice(None, self.pml_shape[1]), slice(None)),
                                (slice(None), slice(None), slice(-self.pml_shape[1], None), slice(None)),
                                (slice(None), slice(None), slice(None), slice(None, self.pml_shape[2])),
                                (slice(None), slice(None), slice(None), slice(-self.pml_shape[2], None)))
            self.cpml_be, self.cpml_bh = [b_e[s] for s in self.pml_regions], [b_h[s] for s in self.pml_regions]
            self.cpml_ce, self.cpml_ch = [c_e[s] for s in self.pml_regions], [c_h[s] for s in self.pml_regions]

    def initial_state(self, e_init: Optional[np.ndarray] = None, h_init: Optional[np.ndarray] = None,
                      psi_e_init: Optional[Tuple[np.ndarray, ...]] = None,
                      psi_h_init: Optional[Tuple[np.ndarray, ...]] = None):
        """Initial state

        Notes:
            Initial values are typically not specified unless starting from a midpoint of a simulation.

        Args:
            e_init: initial :math:`\mathbf{E}`
            h_init: initial :math:`\mathbf{H}`
            psi_e_init: :math:`\\boldsymbol{\\Psi}_E` for CPML updates (zero if :code:`None`)
            psi_h_init: :math:`\\boldsymbol{\\Psi}_E` for CPML updates (zero if :code:`None`)

        Returns: Hidden state of the form:
            e: current :math:`\mathbf{E}`
            h: current :math:`\mathbf{H}`
            psi_e: current :math:`\\boldsymbol{\\Psi}_E` for CPML updates (otherwise :code:`None`)
            psi_h: current :math:`\\boldsymbol{\\Psi}_H` for CPML updates (otherwise :code:`None`)

        """
        # stored fields for fdtd
        e = e_init if e_init is not None else np.zeros(self.field_shape, dtype=np.float64)
        h = h_init if h_init is not None else np.zeros_like(e)
        # for pml updates
        psi_e = psi_e_init if psi_e_init is not None else tuple([np.zeros_like(e)] * 6)
        psi_h = psi_h_init if psi_h_init is not None else tuple([np.zeros_like(e)] * 6)
        return e, h, psi_e, psi_h

    def step(self, state: Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, ...]],
                                Optional[Tuple[np.ndarray, ...]]],
                   src: np.ndarray, src_idx: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray,
                                                                            Optional[Tuple[np.ndarray, ...]],
                                                                            Optional[Tuple[np.ndarray, ...]]]:
        """FDTD step (in the form of an RNNCell)

        Notes:
            The FDTD update consists of updating the fields and auxiliary vectors that comprise the system "state."

            The updates are of the form:

            .. math::
                \mathbf{E}(t + \mathrm{d}t) &= \mathbf{E}(t) + \mathrm{d}t \\frac{\mathrm{d}\mathbf{E}}{\mathrm{d}t} \\
                \mathbf{H}(t + \mathrm{d}t) &= \mathbf{H}(t) + \mathrm{d}t \\frac{\mathrm{d}\mathbf{H}}{\mathrm{d}t}

            From Maxwell's equations, we have (for current source :math:`\mathbf{J}(t)`):

            .. math::
                \\frac{\mathrm{d}\mathbf{E}}{\mathrm{d}t} = \\frac{1}{\\epsilon} \\nabla \\times \mathbf{H}(t) + \mathbf{J}(t) \\
                \\frac{\mathrm{d}\mathbf{H}}{\mathrm{d}t} = -\\frac{1}{\mu} \\nabla \\times \mathbf{E}(t) + \mathbf{M}(t)

            The recurrent update assumes that :math:`\mu = c = 1, \mathbf{M}(t) = \mathbf{0}` and factors in
            perfectly-matched layers (PML), which requires storing two additional PML arrays in the system's state
            vector, namely :math:`\\boldsymbol{\\Psi}_E(t)` and :math:`\\boldsymbol{\\Psi}_H(t)`.

            .. math::
                \mathbf{\Psi_E}^{(t+1/2)} = \mathbf{b} \mathbf{\Psi_E}^{(t-1/2)} + \\nabla_{\mathbf{c}} \\times \mathbf{H}^{(t)}\\
                \mathbf{\Psi_H}^{(t + 1)} = \mathbf{b} \mathbf{\Psi_H}^{(t)} + \\nabla_{\mathbf{c}} \\times \mathbf{E}^{(t)} \\
                \mathbf{E}^{(t+1/2)} = \mathbf{E}^{(t-1/2)} + \\frac{\\Delta t}{\\epsilon} \\left(\\nabla \\times \mathbf{H}^{(t)} + \mathbf{J}^{(t)} + \mathbf{\Psi_E}^{(t+1/2)}\\right)\\
                \mathbf{H}^{(t + 1)} = \mathbf{H}^{(t)} - \\Delta t \\left(\\nabla \\times \mathbf{E}^{(t+1/2)} + \mathbf{\Psi_H}^{(t + 1)}\\right)


            Note, in Einstein notation, the weighted curl operator is given by:
            :math:`\\nabla_{\mathbf{c}} \\times \mathbf{v} := \epsilon_{ijk} c_j \partial_j v_k`.

        Args:
            state: current state of the form :code:`(e, h, psi_e, psi_h)`
                -:code:`e` refers to electric field :math:`\mathbf{E}(t)`
                -:code:`h` refers to magnetic field :math:`\mathbf{H}(t)`
                -:code:`psi_e` refers to :math:`\\boldsymbol{\\Psi}_E(t)`
                -:code:`psi_h` refers to :math:`\\boldsymbol{\\Psi}_H(t)`
            src: is the source :math:`\mathbf{J}(t)`, the "input" to the system.
            src_idx: slice of the added source to be added to E in the update (assume same shape as :code:`e`
                if :code:`None`)

        Returns:
            e_next: refers to :math:`\mathbf{E}(t)`
            h_next: refers to :math:`\mathbf{H}(t)`
            psi_e_next: next :code:`psi_e` (if PML)
            psi_h_next: next :code:`psi_h` (if PML)

        """
        e, h, psi_e, psi_h = state
        src_idx = tuple([slice(None)] * 4) if src_idx is None else src_idx

        # add pml in pml regions if specified
        if self.pml_shape is not None:
            for pml_idx, region in enumerate(self.pml_regions):
                psi_e[pml_idx] = self.cpml_be[pml_idx] * psi_e[pml_idx] + self._curl_h_pml(h, pml_idx)
                psi_h[pml_idx] = self.cpml_bh[pml_idx] * psi_h[pml_idx] - self._curl_e_pml(e, pml_idx)
                e[region] += psi_e / self.eps_t[region] * self.dt
                h[region] += psi_h * self.dt

        # add source
        e[src_idx] += src * self.dt / self.eps_t[src_idx]

        # update e, h fields
        e += (self.curl_h(h) / self.eps_t) * self.dt
        h += -self.curl_e(e) * self.dt

        return e, h, psi_e, psi_h

    def _cpml(self, ax: int, alpha_max: float = 0, exp_scale: float = 3.5,
             kappa: float = 1, log_reflection: float = 1.6) -> Tuple[np.ndarray, np.ndarray]:
        if self.cell_sizes[ax].size == 1:
            return np.ones(2), np.ones(2)
        sigma, alpha = pml_params(self.pos[ax], t=self.pml_shape[ax], exp_scale=exp_scale,
                                  log_reflection=log_reflection, absorption_corr=1)
        alpha *= alpha_max  # alpha_max recommended to be np.pi * central_wavelength / 5 if nonzero
        b = np.exp(-(alpha + sigma / kappa) * self.dt)
        return b, (b - 1) * (sigma / (sigma * kappa + alpha * kappa ** 2))

    def _curl_e_pml(self, e: np.ndarray, pml_idx: int) -> np.ndarray:
        dx, _ = self._dxes
        c, s = self.cpml_ch[pml_idx], self.pml_regions[pml_idx]

        def de(e_, ax):
            return (np.roll(e_, -1, axis=ax) - e_) / dx[ax][s] * c[ax]

        return d2curl(e, de)

    def _curl_h_pml(self, h: np.ndarray, pml_idx: int) -> np.ndarray:
        _, dx = self._dxes
        c, s = self.cpml_ce[pml_idx], self.pml_regions[pml_idx]

        def dh(h_, ax):
            return (h_ - np.roll(h_, 1, axis=ax)) / dx[ax][s] * c[ax]

        return d2curl(h, dh)

    def run(self, src_func: Callable[[float], Tuple[np.ndarray, np.ndarray]], time: float):
        """

        Args:
            src_func: a function that provides the input source
            time: total time to run the simulation

        Returns:
            state: final state of the form :code:`(e, h, psi_e, psi_h)`
                -:code:`e` refers to electric field :math:`\mathbf{E}(t)`
                -:code:`h` refers to magnetic field :math:`\mathbf{H}(t)`
                -:code:`psi_e` refers to :math:`\\boldsymbol{\\Psi}_E(t)` (for debugging PML)
                -:code:`psi_h` refers to :math:`\\boldsymbol{\\Psi}_H(t)` (for debugging PML)

        """
        state = self.initial_state()
        for step in range(int(time // self.dt)):
            state = self.step(state, *src_func(step * self.dt))
        return state
