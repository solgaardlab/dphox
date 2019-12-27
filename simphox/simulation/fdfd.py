from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, spsolve

from .grid import Grid
from ..constants import C_0, EPS_0, MU_0
from ..ops import d2curl_op, d2curl_fn
from ..typing import Shape, Dim, GridSpacing, Optional, Tuple, List, Union, SpSolve, Op


class FDFD(Grid):
    def __init__(self, grid_shape: Shape, grid_spacing: GridSpacing, eps: Union[float, np.ndarray] = 1,
                 pml_shape: Optional[Union[Shape, Dim]] = None, bloch_phase: Union[Dim, float] = 0.0,
                 wavelength: float = 1.55, pml_eps: float = 1.0):
        super(FDFD, self).__init__(grid_shape, grid_spacing, eps)
        self.bloch = np.ones_like(self.shape) * np.exp(1j * np.asarray(bloch_phase)) if isinstance(bloch_phase, float) \
            else np.exp(1j * np.asarray(bloch_phase))
        self.pml_shape = pml_shape
        self.pml_eps = pml_eps
        self.omega = 2 * np.pi * C_0 / (wavelength * 1e-6)
        self.wavelength = wavelength * 1e-6
        if pml_shape is not None and not len(pml_shape) == len(self.shape):
            raise ValueError(f'Need len(pml_shape) == len(grid_shape) but got ({len(pml_shape)}, {len(self.shape)}).')
        if not len(self.bloch) == len(self.shape):
            raise ValueError(
                f'Need len(bloch_phase) == len(grid_shape) but got ({len(self.bloch)}, {len(self.shape)}).')

    @property
    def mat(self) -> sp.spmatrix:
        """Build the discrete Maxwell operator :math:`A(\omega)` acting on the electric field :math:`e`.

        The discretized version of Maxwell's equations in frequency domain is:
        .. math::
            \nabla \times \mu^{-1} \nabla \times e - \omega^2 \epsilon e = i \omega J,
        which can be written in the form :math:`A e = b`, where:
        .. math::
            A = \nabla \times \mu^{-1} \nabla \times  - \omega^2 \epsilon \\
            b = i \omega J
        is an operator representing the discretized EM wave operator at frequency :math:`omega`.

        Returns:
            Electric field operator :math:`A` for solving Maxwell's equations at frequency :math:`omega`.
        """
        curl_curl = self.curl_b @ sp.diags(1 / self.mu_t.flatten()) @ self.curl_f / MU_0
        return curl_curl - self.omega ** 2 * EPS_0 * sp.diags(self.eps_t.flat)

    A = mat  # alias A (common symbol for FDFD matrix) to mat

    @property
    def wgm(self) -> sp.spmatrix:
        """Build the WaveGuide Mode (WGM) operator :math:`C(\omega)` acting on the magnetic field
        :math:`\mathbf{h}` of the form `(hx, hy)`, which assumes cross-sectional symmetry:
        .. math::
            C \mathbf{h}_m = \lambda_m \mathbf{h}_m,
        where :math:`0 \leq m < M` for the :math:`M` modes with the largest wavenumbers
        (:math:`\beta_m = \sqrt{\lambda_m}`).

        Returns:
            Magnetic field operator :math:`C`.
        """

        df, db = self.df, self.db
        eps = [e.flatten() for e in self.eps_t]
        mu = [m.flatten() for m in self.mu_t]
        eps_10 = sp.diags(np.hstack((eps[1], eps[0])))

        m1 = self.omega ** 2 * MU_0 * EPS_0 * sp.diags(np.hstack((mu[1], mu[0]))) @ eps_10
        m2 = eps_10 @ sp.vstack([-df[1], df[0]]) @ sp.diags(1 / eps[2]) @ sp.hstack([-db[1], db[0]])
        m3 = sp.vstack(db[:2]) @ sp.hstack(df[:2])

        return m1 + m2 + m3

    Aw = wgm  # Aw is the A for the guided mode eigensolver

    @property
    def e2h_op(self) -> sp.spmatrix:
        """
        Convert magnetic field :math:`\mathbf{e}` to electric field :math:`\mathbf{h}` (op).

        Usage is: `h = fdfd.e2h @ e`

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            -i \omega h = \nabla \times e

        Returns:

        """
        return self.curl_f * C_0 / (1j * self.omega) @ sp.diags(1 / self.mu_t.flatten())

    @property
    def h2e_op(self) -> sp.spmatrix:
        """
        Convert magnetic field :math:`\mathbf{h}` to electric field :math:`\mathbf{e}` (op).

        Usage is: `e = fdfd.h2e @ h`

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            -i \omega \epsilon \mathbf{e} = \nabla \times \mathbf{h}

        Returns:

        """
        return self.curl_b * C_0 / (-1j * self.omega) @ sp.diags(1 / self.eps_t.flatten())

    @property
    def e2h(self) -> Op:
        """
        Convert magnetic field :math:`\mathbf{e}` to electric field :math:`\mathbf{h}` (functional)

        Usage is: `h = fdfd.e2h(e)`

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            i \omega \mu \mathbf{h} = \nabla \times \mathbf{e}

        Returns:
            Function to convert e-field to h-field

        """
        return lambda e: self.curl_e(e) * C_0 / (1j * self.omega * self.mu_t.flatten())

    @property
    def h2e(self) -> Op:
        """
        Convert magnetic field :math:`\mathbf{h}` to electric field :math:`\mathbf{e}` (functional)

        Usage is: `e = fdfd.h2e(h)`

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            -i \omega \epsilon \mathbf{e} = \nabla \times \mathbf{h}

        Returns:
            Function to convert h-field to e-field

        """
        return lambda h: self.curl_h(h) * C_0 / (-1j * self.omega * self.eps_t.flatten())

    def solve(self, src: np.ndarray, solver_fn: Optional[SpSolve] = None) -> np.ndarray:
        return solver_fn(self.mat, src) if solver_fn else spsolve(self.mat, src)

    def wgm_solve(self, num_modes: int = 1, beta_guess: Optional[float] = None,
                  tol: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
        """Solve for waveguide modes (z-translational symmetry) by finding the eigenvalues of :math:`C`.
        .. math::
            C \mathbf{h}_m = \lambda_m \mathbf{h}_m,
        where :math:`0 \leq m < M` for the :math:`M` (`num_modes`) modes with the largest wavenumbers
        (:math:`\beta_m = \sqrt{\lambda_m}`).

        The wavenumber or :math:`\beta` corresponds to the square root of the eigenvalues, i.e.

        Args:
            num_modes: Number of modes
            beta_guess: Guess for propagation constant :math:`\beta`

        Returns:
            `num_modes` (:math:`M`) largest propagation constants (:math:`\sqrt{\lambda_m(C)}`)
            and corresponding modes (:math:`\mathbf{h}_m`).

        """

        if not self.dim <= 2:
            raise AttributeError("Dim must be 1 or 2")
        db = self.db
        mu = [m.flatten() for m in self.mu_t]
        sigma = beta_guess ** 2 if beta_guess else (2 * np.pi * np.sqrt(np.max(self.eps * EPS_0)) / self.wavelength) ** 2
        eigvals, eigvecs = eigs(self.wgm, k=num_modes, sigma=sigma, tol=tol)
        inds_sorted = np.asarray(np.argsort(np.real(np.sqrt(eigvals)))[::-1])
        hz = 1 / (1j * np.sqrt(eigvals)) * sp.diags(1 / mu[2]) @ sp.vstack(db[:2]) @ sp.diags(mu[:2]) @ eigvecs
        h = np.hstack((eigvecs, hz))
        return np.sqrt(eigvals[inds_sorted]), h[..., inds_sorted].T

    @lru_cache()
    def deriv(self, back: bool = False) -> List[sp.spmatrix]:
        """Calculate directional derivative (cached, since this does not depend on any params)

        Args:
            back: Return backward derivative

        Returns:
            Discrete directional derivative `d` of the form `(d_x, d_y, d_z)`

        """

        # account for 1d and 2d cases
        b = np.hstack((self.bloch, np.ones((3 - self.dim,), dtype=self.bloch.dtype)))
        s = np.hstack((self.shape, np.ones((3 - self.dim,), dtype=self.shape.dtype)))

        # define grid cell sizes (including pml if necessary)
        dx_f, dx_b = self._dxes

        if back:
            # get backward derivative
            dx = np.meshgrid(*dx_b, indexing='ij')
            d = [sp.diags([1, -1, -np.conj(b[ax])], [0, -1, n - 1], shape=(n, n), dtype=np.complex128)
                  if n > 1 else 0 for ax, n in enumerate(s)]  # get single axis back-derivs
        else:
            # get forward derivative
            dx = np.meshgrid(*dx_f, indexing='ij')
            d = [sp.diags([-1, 1, b[ax]], [0, 1, -n + 1], shape=(n, n), dtype=np.complex128)
                  if n > 1 else 0 for ax, n in enumerate(s)]  # get single axis forward-derivs
        d = [sp.kron(d[0], sp.eye(s[1] * s[2])),
             sp.kron(sp.kron(sp.eye(s[0]), d[1]), sp.eye(s[2])),
             sp.kron(sp.eye(s[0] * s[1]), d[2])]  # tile over the other axes using sp.kron
        d = [sp.diags(1 / dx[ax].ravel()) @ d[ax] for ax in range(len(s))]  # scale by dx (incl pml)

        return d

    @lru_cache()
    @property
    def df(self):
        return self.deriv()

    @lru_cache()
    @property
    def db(self):
        return self.deriv(back=True)

    @lru_cache()
    @property
    def curl_f(self):
        return d2curl_op(self.df)

    @lru_cache()
    @property
    def curl_b(self):
        return d2curl_op(self.db)

    @lru_cache()
    @property
    def curl_e(self) -> Op:
        dx_f, dx_b = self._dxes
        dx = np.meshgrid(*dx_f, indexing='ij')

        def de(e, d):
            return (np.roll(e[d], -1, axis=d) - e[d]) / dx[d]
        return lambda e: d2curl_fn(e, de)

    @lru_cache()
    @property
    def curl_h(self) -> Op:
        dx_f, dx_b = self._dxes
        dx = np.meshgrid(*dx_b, indexing='ij')

        def dh(h, d):
            return (h[d] - np.roll(h[d], 1, axis=d)) / dx[d]
        return lambda h: d2curl_fn(h, dh)

    @lru_cache()
    @property
    def _dxes(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Conditional transformation of self.dxes based on stretched-coordinated perfectly matched layers (SC-PML)

        Returns:
            SC-PML transformation of dxes

        """

        if self.pml_shape is None:
            return self.dxes, self.dxes
        else:
            dxes_pml_e, dxes_pml_h = ([], [])
            for ax, p in enumerate(self.pos):
                if self.dxes[ax].size == 1:
                    dxes_pml_e.append(self.dxes[ax])
                    dxes_pml_h.append(self.dxes[ax])
                else:
                    pe, ph = (p[:-1] + p[1:]) / 2, p[:-1]
                    dxes_pml_e.append(self.dxes[ax] * self._scpml(pe, ax))
                    dxes_pml_h.append(self.dxes[ax] * self._scpml(ph, ax))
            return dxes_pml_e, dxes_pml_h

    def _scpml(self, d: np.ndarray, ax: int, exp_scale: float = 4, log_reflection: float = -16):
        absorption_corr = self.omega * self.pml_eps
        t = self.pml_shape[ax]
        d_pml = np.hstack((
            (d[:t] - np.min(d)) / (d[t] - np.min(d)),
            np.zeros_like(d[t:-t]),
            (np.max(d) - d[-t:]) / (np.max(d) - d[-t])
        ))
        return 1 + 1j * (exp_scale + 1) * log_reflection / 2 * d_pml ** exp_scale / absorption_corr