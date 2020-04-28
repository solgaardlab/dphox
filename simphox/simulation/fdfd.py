from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs


from .grid import SimGrid
from ..typing import Shape, Dim, GridSpacing, Optional, Tuple, List, Union, SpSolve, Op

try:  # pardiso (using Intel MKL) is much faster than scipy's solver
    from .mkl import spsolve, feast_eigs
except OSError:  # if mkl isn't installed
    from scipy.sparse.linalg import spsolve


class FDFD(SimGrid):
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray] = 1,
                 wavelength: float = 1.55, bloch_phase: Union[Dim, float] = 0.0,
                 pml: Optional[Union[Shape, Dim]] = None, pml_eps: float = 1.0,
                 grid_avg: bool = True, no_grad: bool = True):

        self.wavelength = wavelength
        self.no_grad = no_grad

        super(FDFD, self).__init__(
            shape=shape,
            spacing=spacing,
            eps=eps,
            bloch_phase=bloch_phase,
            pml=pml,
            pml_eps=pml_eps,
            grid_avg=grid_avg
        )

    @property
    def k0(self):
        return 2 * np.pi / self.wavelength  # defines the units for the simulation!

    @property
    def mat(self) -> Union[sp.spmatrix, Tuple[np.ndarray, np.ndarray]]:
        """Build the discrete Maxwell operator :math:`A(k_0)` acting on :math:`\mathbf{e}`.

        The discretized version of Maxwell's equations in frequency domain is:
        .. math::
            \nabla \times \mu^{-1} \nabla \times \mathbf{e} - k_0^2 \epsilon \mathbf{e} = k_0 \mathbf{j},
        which can be written in the form :math:`A \mathbf{e} = \mathbf{b}`, where:
        .. math::
            A = \nabla \times \mu^{-1} \nabla \times - k_0^2 \epsilon \\
            b = k_0 \mathbf{j}
        is an operator representing the discretized EM wave operator at frequency :math:`omega`.

        Returns:
            Electric field operator :math:`A` for solving Maxwell's equations at frequency :math:`omega`.
        """
        if self.no_grad:
            mat = self.curl_curl - self.k0 ** 2 * sp.diags(self.eps_t.flatten())
            return mat
        else:
            data, rc = self.curl_curl_coo
            data_param = self.k0 ** 2 * self.eps_t.flatten()
            rc_param = np.hstack((np.arange(self.n * 3), np.arange(self.n * 3)))
            # TODO(sunil): change this to bd.hstack for autograd, torch, tf
            return np.hstack((data_param, data)), np.hstack((rc_param, rc))

    A = mat  # alias A (common symbol for FDFD matrix) to mat

    @property
    def matz(self) -> Union[sp.spmatrix, Tuple[np.ndarray, np.ndarray]]:
        """Build the discrete Maxwell operator :math:`A_z(k_0)` acting on :math:`\mathbf{e}_z`.

        The discretized version of Maxwell's equations in frequency domain is:
        .. math::
            \nabla \times \mu^{-1} \nabla \times \mathbf{e} - k_0^2 \epsilon \mathbf{e} = k_0 \mathbf{j},
        which can be written in the form :math:`A \mathbf{e} = \mathbf{b}`, where:
        .. math::
            A = (\nabla \times \mu^{-1} \nabla \times) - k_0^2 \epsilon \\
            \mathbf{b} = k_0 \mathbf{j}
        is an operator representing the discretized EM wave operator at frequency :math:`omega`.

        But when only :math:`\mathbf{e}_z` is non-zero, then we can solve a smaller problem to improve the efficiency.
        The form of this problem is :math:`A_z \mathbf{e}_z = \mathbf{b}_z`, where:
        .. math::
            A = (\nabla \times \mu^{-1} \nabla \times)_z + k_0^2 \epsilon_z \\
            \mathbf{b}_z = k_0 \mathbf{j}_z \\

        Returns:
            Electric field operator :math:`A_z` for a source with z-polarized e-field.
        """

        if self.no_grad:
            mat = self.ddz - self.k0 ** 2 * sp.diags(self.eps_t[2].flatten())
            return mat
        else:
            dd: sp.coo_matrix = self.ddz.tocoo()
            data_param = self.k0 ** 2 * self.eps_t[2].flatten()
            rc_param = np.hstack((np.arange(self.n), np.arange(self.n)))
            # TODO(sunil): change this to bd.hstack for autograd, torch, tf
            return np.hstack((data_param, dd.data)), np.hstack((rc_param, np.vstack((dd.row, dd.col))))

    Az = matz

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

        if not self.ndim <= 2:
            raise AttributeError("Grid dimension must be 1 or 2")

        df, db = self.df, self.db
        eps = [e.flatten() for e in self.eps_t]

        if self.ndim == 2:
            eps_10 = sp.diags(np.hstack((eps[1], eps[0])))
            m1 = eps_10 * self.k0 ** 2
            m2 = eps_10 @ sp.vstack([-df[1], df[0]]) @ sp.diags(1 / eps[2]) @ sp.hstack([-db[1], db[0]])
            m3 = sp.vstack(db[:2]) @ sp.hstack(df[:2])
            return m1 + m2 + m3
        else:
            return sp.diags(self.eps_t[0].flatten()) * self.k0 ** 2 + df[0].dot(db[0])

    C = wgm  # C is the matrix for the guided mode eigensolver

    def e2h(self, e: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """
        Convert magnetic field :math:`\mathbf{e}` to electric field :math:`\mathbf{h}`.

        Usage is: `h = fdfd.e2h(e)`, where `e` is grid-shaped (not flattened)

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            i \omega \mu \mathbf{h} = \nabla \times \mathbf{e}

        Returns:
            The h-field converted from the e-field

        """
        e = self.reshape(e) if e.ndim == 2 else e
        return self.curl_e(e, beta) / (1j * self.k0)

    def h2e(self, h: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """
        Convert magnetic field :math:`\mathbf{h}` to electric field :math:`\mathbf{e}`.

        Usage is: `e = fdfd.h2e(h)`, where `h` is grid-shaped (not flattened)

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            -i \omega \epsilon \mathbf{e} = \nabla \times \mathbf{h}

        Returns:
            Function to convert h-field to e-field

        """
        h = self.reshape(h) if h.ndim == 2 else h
        return self.curl_h(h, beta) / (1j * self.k0 * self.eps_t)

    def solve(self, src: np.ndarray, solver_fn: Optional[SpSolve] = None, reshaped: bool = True) -> np.ndarray:
        """FDFD e-field Solver

        Args:
            src: normalized source (can be wgm or tfsf)
            solver_fn: any function that performs a sparse linalg solve
            reshaped: reshape into the grid shape (instead of vectorized/flattened form)

        Returns:
            Electric fields that solve the problem :math:`A\mathbf{e} = \mathbf{b} = i \omega \mathbf{j}`

        """
        b = self.k0 * src.flatten()
        if b.size == self.n * 3:
            e = solver_fn(self.mat, b) if solver_fn else spsolve(self.mat, b)
        elif b.size == self.n:  # assume only the z component
            ez = solver_fn(self.matz, b) if solver_fn else spsolve(self.matz, b)
            o = np.zeros_like(ez)
            e = np.vstack((o, o, ez))
        else:
            raise ValueError(f'Expected src.size == {self.n * 3} or {self.n}, but got {b.size}.')
        return self.reshape(e) if reshaped else e

    def wgm_solve(self, num_modes: int = 6, beta_guess: Optional[Union[float, Tuple[float, float]]] = None,
                  tol: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
        """FDFD waveguide mode (WGM) solver

        Solve for waveguide modes (x-translational symmetry) by finding the eigenvalues of :math:`C`.

        .. math::
            C \mathbf{h}_m = \lambda_m \mathbf{h}_m,
        where :math:`0 \leq m < M` for the :math:`M` (`num_modes`) modes with the largest wavenumbers
        (:math:`\beta_m = \sqrt{\lambda_m}`).

        Args:
            num_modes: Number of modes
            beta_guess: Guess for propagation constant :math:`\beta`
            tol: Tolerance of the mode solver

        Returns:
            `num_modes` (:math:`M`) largest propagation constants (:math:`\sqrt{\lambda_m(C)}`)
            and corresponding modes (:math:`\mathbf{h}_m`) of shape `(num_modes, n)`.

        """

        db = self.db
        if isinstance(beta_guess, float) or beta_guess is None:
            sigma = beta_guess ** 2 if beta_guess else (self.k0 * np.sqrt(np.max(self.eps))) ** 2
            eigvals, eigvecs = eigs(self.wgm, k=num_modes, sigma=sigma, tol=tol)
        elif isinstance(beta_guess, tuple):
            erange = beta_guess[0] ** 2, beta_guess[1] ** 2
            eigvals, eigvecs, _, _, _, _ = feast_eigs(self.wgm, erange=erange, k=num_modes)
        else:
            raise TypeError(f'Expected beta_guess to be None, float, or Tuple[float, float] but got {type(beta_guess)}')
        inds_sorted = np.asarray(np.argsort(np.sqrt(eigvals.real))[::-1])
        if self.ndim > 1:
            hz = sp.hstack(db[:2]) @ eigvecs / (1j * np.sqrt(eigvals))
            h = np.vstack((eigvecs, hz))
        else:
            h = eigvecs

        factor = np.exp(-1j * np.angle(h[:1, :])) if self.dtype == np.complex128 else np.sign(h[:1, :])
        h *= factor  # divide by global phase or set polarity (set reference plane)
        return np.sqrt(eigvals[inds_sorted]), h[:, inds_sorted].T

    def src(self, axis: int = 0, mode_idx: int = 0, power: float = 1,
            beta_guess: Optional[float] = None, tol: float = 1e-5) -> np.ndarray:
        """Define waveguide mode source using waveguide mode solver (incl. pml if part of the mode solver!)

        Args:
            axis: Axis of propagation
            mode_idx: Mode index to use (default is 0, the fundamental mode)
            power: Power to scale the source (default is 1, a normalized mode in arb units),
            and if negative, the source moves in opposite direction (polarity is encoded in sign of power).
            beta_guess: Guess for propagation constant :math:`\beta`
            tol: Tolerance of the mode solver

        Returns:
            Grid-shaped waveguide mode (wgm) source (normalized h-mode for 1d, spins-b source for 2d)
        """

        polarity = np.sign(power)
        p = np.abs(power)

        beta, h = self.wgm_solve(min(mode_idx + 1, 6), beta_guess, tol)

        if self.ndim == 2 and self.pml_shape:
            h = self.reshape(h[mode_idx])  # get the last mode and shape it
            idx = np.roll(np.arange(3, dtype=np.int), -axis)
            _, dx = self._dxes
            phasor = np.exp(1j * polarity * beta * dx[axis])

            # get shifted e-field
            e = np.roll(self.h2e(h), shift=-1, axis=axis)
            # define current sources
            j = np.stack((np.zeros(self.shape), -h[idx[2]], h[idx[1]])) * phasor[np.newaxis, ...]
            m = np.stack((np.zeros(self.shape), -e[idx[2]], e[idx[1]]))
            jm = self.curl_h(m) / self.k0

            return (j + jm) / dx[axis] * polarity * np.sqrt(p)
        else:
            if self.ndim == 1 and self.pml_shape:
                raise NotImplementedError("PML for 1d wgm source must be None.")
            return h * polarity * np.sqrt(p)

    @property
    @lru_cache()
    def _dxes(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Conditional transformation of self.dxes based on stretched-coordinated perfectly matched layers (SC-PML)

        Returns:
            SC-PML transformation of dxes for the e-fields and h-fields, respectively
        """

        if self.pml_shape is None:
            return np.meshgrid(*self.cell_sizes, indexing='ij'), np.meshgrid(*self.cell_sizes, indexing='ij')
        else:
            dxes_pml_e, dxes_pml_h = [], []
            for ax, p in enumerate(self.pos):
                scpml_e, scpml_h = self.scpml(ax)
                dxes_pml_e.append(self.cell_sizes[ax] * scpml_e)
                dxes_pml_h.append(self.cell_sizes[ax] * scpml_h)
            return np.meshgrid(*dxes_pml_e, indexing='ij'), np.meshgrid(*dxes_pml_h, indexing='ij')

    def scpml(self, ax: int, exp_scale: float = 4, log_reflection: float = -16) -> Tuple[np.ndarray, np.ndarray]:
        if self.cell_sizes[ax].size == 1:
            return np.ones(1), np.ones(1)
        p = self.pos[ax]
        pe, ph = (p[:-1] + p[1:]) / 2, p[:-1]
        absorption_corr = self.k0 * self.pml_eps
        t = self.pml_shape[ax]

        def _scpml(d: np.ndarray):
            d_pml = np.hstack((
                (d[t] - d[:t]) / (d[t] - p[0]),
                np.zeros_like(d[t:-t]),
                (d[-t:] - d[-t]) / (p[-1] - d[-t])
            ))
            return 1 + 1j * (exp_scale + 1) * (d_pml ** exp_scale) * log_reflection / (2 * absorption_corr)
        return _scpml(pe), _scpml(ph)

    @property
    @lru_cache()
    def curl_curl(self) -> sp.spmatrix:
        curl_curl: sp.spmatrix = self.curl_b @ self.curl_f
        curl_curl.sort_indices()  # for the solver
        return curl_curl

    @property
    @lru_cache()
    def ddz(self) -> sp.spmatrix:
        df, db = self.df, self.db
        ddz = -db[0] @ df[0] - db[1] @ df[1]
        ddz.sort_indices()  # for the solver
        return ddz

    @property
    @lru_cache()
    def curl_curl_coo(self) -> Tuple[np.ndarray, np.ndarray]:
        curl_curl: sp.coo_matrix = self.curl_curl.tocoo()
        rc_curl_curl = np.vstack((curl_curl.row, curl_curl.col))
        return curl_curl.data, rc_curl_curl

    @property
    @lru_cache()
    def ddz_coo(self) -> Tuple[np.ndarray, np.ndarray]:
        ddz: sp.coo_matrix = self.ddz.tocoo()
        rc_curl_curl = np.vstack((ddz.row, ddz.col))
        return ddz.data, rc_curl_curl

