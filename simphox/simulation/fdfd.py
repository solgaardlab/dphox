from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, spsolve

from .grid import SimGrid
from ..constants import C_0
from ..typing import Shape, Dim, GridSpacing, Optional, Tuple, List, Union, SpSolve, Op


class FDFD(SimGrid):
    def __init__(self, grid_shape: Shape, grid_spacing: GridSpacing, eps: Union[float, np.ndarray] = 1,
                 pml_shape: Optional[Union[Shape, Dim]] = None, bloch_phase: Union[Dim, float] = 0.0,
                 wavelength: float = 1.55, pml_eps: float = 1.0):

        self.wavelength = wavelength
        self.k0 = 2 * np.pi / self.wavelength  # defines the units for the simulation!
        self.omega = C_0 * self.k0

        super(FDFD, self).__init__(grid_shape, grid_spacing, eps, pml_shape, pml_eps)

        self.bloch = np.ones_like(self.shape) * np.exp(1j * np.asarray(bloch_phase)) if isinstance(bloch_phase, float) \
            else np.exp(1j * np.asarray(bloch_phase))
        if not len(self.bloch) == len(self.shape):
            raise AttributeError(f'Need len(bloch_phase) == len(grid_shape),'
                                 f'got ({len(self.bloch)}, {len(self.shape)}).')

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
        curl_curl = self.curl_b @ sp.diags(1 / self.mu_t.flatten()) @ self.curl_f
        return curl_curl - self.k0 ** 2 * sp.diags(self.eps_t.flat)

    A = mat  # alias A (common symbol for FDFD matrix) to mat

    @property
    def wgm(self) -> sp.spmatrix:
        """Build the WaveGuide Mode (WGM) operator (for 1D or 2D grid only)

        The WGM operator :math:`C(\omega)` acts on the magnetic field
        :math:`\mathbf{h}` of the form `(hx, hy)`, which assumes cross-section translational x-symmetry:
        .. math::
            C \mathbf{h}_m = \lambda_m \mathbf{h}_m,
        where :math:`0 \leq m < M` for the :math:`M` modes with the largest wavenumbers
        (:math:`\beta_m = \sqrt{\lambda_m}`).

        Returns:
            Magnetic field operator :math:`C`.
        """

        if not self.dim <= 2:
            raise AttributeError("Grid dimension must be 1 or 2")

        df, db = self.df, self.db
        eps = [e.flatten() for e in self.eps_t]
        mu = [m.flatten() for m in self.mu_t]

        if self.dim == 2:
            eps_10 = sp.diags(np.hstack((eps[1], eps[0])))
            m1 = sp.diags(np.hstack((mu[1], mu[0]))) @ eps_10 * self.k0 ** 2
            m2 = eps_10 @ sp.vstack([-df[1], df[0]]) @ sp.diags(1 / eps[2]) @ sp.hstack([-db[1], db[0]])
            m3 = sp.vstack(db[:2]) @ sp.hstack(df[:2])
            return m1 + m2 + m3
        else:
            return sp.diags(self.eps_t[0].flatten()) * self.k0 ** 2 + df[0].dot(db[0])

    C = wgm  # C is the matrix for the guided mode eigensolver

    @property
    def e2h_op(self) -> sp.spmatrix:
        """
        Convert magnetic field :math:`\mathbf{e}` to electric field :math:`\mathbf{h}` (op).

        Usage is: `h = fdfd.e2h @ e`, where `e` is flattened (not grid-shaped)

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            -i \omega h = \nabla \times e

        Returns:

        """
        return self.curl_f @ sp.diags(1 / (self.k0 * self.mu_t.flatten()))

    @property
    def h2e_op(self) -> sp.spmatrix:
        """
        Convert magnetic field :math:`\mathbf{h}` to electric field :math:`\mathbf{e}` (op).

        Usage is: `e = fdfd.h2e @ h`, where `h` is flattened (not grid-shaped)

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            -i \omega \epsilon \mathbf{e} = \nabla \times \mathbf{h}

        Returns:

        """
        return self.curl_b @ sp.diags(1 / (self.k0 * self.eps_t.flatten()))

    @property
    def e2h(self) -> Op:
        """
        Convert magnetic field :math:`\mathbf{e}` to electric field :math:`\mathbf{h}` (functional)

        Usage is: `h = fdfd.e2h(e)`, where `e` is grid-shaped (not flattened)

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            i \omega \mu \mathbf{h} = \nabla \times \mathbf{e}

        Returns:
            Function to convert e-field to h-field

        """
        return lambda e: self.curl_e(e) / (self.k0 * self.mu_t.flatten())

    @property
    def h2e(self) -> Op:
        """
        Convert magnetic field :math:`\mathbf{h}` to electric field :math:`\mathbf{e}` (functional)

        Usage is: `e = fdfd.h2e(h)`, where `h` is grid-shaped (not flattened)

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            -i \omega \epsilon \mathbf{e} = \nabla \times \mathbf{h}

        Returns:
            Function to convert h-field to e-field

        """
        return lambda h: self.curl_h(h) / (self.k0 * self.eps_t.flatten())

    def solve(self, src: np.ndarray, solver_fn: Optional[SpSolve] = None, reshaped: bool = True) -> np.ndarray:
        """FDFD e-field Solver

        Args:
            src: normalized source (can be wgm or tfsf)
            solver_fn: any function that performs a sparse linalg solve
            reshaped: reshape into the grid shape (instead of vectorized/flattened form)

        Returns:
            Electric fields that solve the problem :math:`A\mathbf{e} = \mathbf{b} = i \omega \mathbf{j}`

        """
        b = -1j * self.omega * src  # do we need to do this??
        e = solver_fn(self.mat, b) if solver_fn else spsolve(self.mat, b)
        return self.reshape(e) if reshaped else e

    def wgm_solve(self, num_modes: int = 1, beta_guess: Optional[float] = None,
                  tol: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
        """FDFD waveguide mode (WGM) solver

        Solve for waveguide modes (x-translational symmetry) by finding the eigenvalues of :math:`C`.

        .. math::
            C \mathbf{h}_m = \lambda_m \mathbf{h}_m,
        where :math:`0 \leq m < M` for the :math:`M` (`num_modes`) modes with the largest wavenumbers
        (:math:`\beta_m = \sqrt{\lambda_m}`).

        The wavenumber or :math:`\beta` corresponds to the square root of the eigenvalues, i.e.

        Args:
            num_modes: Number of modes
            beta_guess: Guess for propagation constant :math:`\beta`
            tol: Tolerance of the mode solver

        Returns:
            `num_modes` (:math:`M`) largest propagation constants (:math:`\sqrt{\lambda_m(C)}`)
            and corresponding modes (:math:`\mathbf{h}_m`).

        """

        db = self.db
        mu = [m.flatten() for m in self.mu_t]
        sigma = beta_guess ** 2 if beta_guess else (self.k0 * np.sqrt(np.max(self.eps))) ** 2
        eigvals, eigvecs = eigs(self.wgm, k=num_modes, sigma=sigma, tol=tol)
        inds_sorted = np.asarray(np.argsort(np.real(np.sqrt(eigvals)))[::-1])
        hz = 1 / (1j * np.sqrt(eigvals)) * sp.diags(1 / mu[2]) @ sp.vstack(db[:2]) @ sp.diags(mu[:2]) @ eigvecs
        h = np.hstack((eigvecs, hz))
        return np.sqrt(eigvals[inds_sorted]), h[..., inds_sorted].T

    def wgm_src(self, axis: int = 0, mode_num: int = 0, negative: bool = False,
                beta_guess: Optional[float] = None, tol: float = 1e-5) -> np.ndarray:
        """Define waveguide mode source using waveguide mode solver (incl. pml if part of the mode solver!)

        Args:
            axis: Axis of propagation
            mode_num: Mode index to use (0 is fundamental mode)
            negative: Propagate in the -ve direction (else +ve)
            beta_guess: Guess for propagation constant :math:`\beta`
            tol: Tolerance of the mode solver

        Returns:
            Grid-shaped waveguide mode (wgm) source (normalized h-mode for 1d, spins-b source for 2d)
        """

        beta, h = self.wgm_solve(mode_num + 1, beta_guess, tol)
        h = self.reshape(h[..., -1])  # get the last mode and shape it

        if self.dim == 2 and self.pml_shape:
            idx = np.roll(np.arange(3), -axis)
            direction = 1 - 2 * negative
            _, dx = self._dxes
            dx = np.meshgrid(*dx)
            mu = self.reshape(self.mu_t.flatten())
            phasor = np.exp(1j * direction * beta * dx[axis])

            e = np.roll(self.h2e(h), shift=-1, axis=axis)  # get shifted e-field
            # define current sources
            j = np.stack((np.zeros(self.shape), -h[idx[2]], h[idx[1]])) * phasor[np.newaxis, ...]
            m = np.stack((np.zeros(self.shape), -e[idx[2]], e[idx[1]]))
            # need (3, nx, ny, nz) array for functional curl to work, hence the newaxis
            jm = (self.curl_h(m[..., np.newaxis] / mu[..., np.newaxis]) / (1j * self.omega)).squeeze()

            return (j + jm) / dx[axis] * direction
        else:
            if self.dim == 1 and self.pml_shape:
                raise NotImplementedError("PML for 1d wgm source must be None.")
            return h

    @property
    @lru_cache()
    def _dxes(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Conditional transformation of self.dxes based on stretched-coordinated perfectly matched layers (SC-PML)

        Returns:
            SC-PML transformation of dxes for the e-fields and h-fields, respectively
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
