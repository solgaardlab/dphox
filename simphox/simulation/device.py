import numpy as np
from ..typing import Tuple, Dim2, Optional, Callable
from .fdfd import FDFD
from ..utils import poynting_z
from functools import lru_cache
import copy


class Material:
    def __init__(self, name: str, facecolor: Tuple[float, float, float] = None, eps: float = None):
        self.name = name
        self.eps = eps
        self.facecolor = facecolor

    def __str__(self):
        return self.name


# SILICON = Material('Silicon', (0.3, 0.3, 0.3), 3.4784 ** 2)
# POLYSILICON = Material('Poly-Si', (0.5, 0.5, 0.5), 3.4784 ** 2)
# OXIDE = Material('Oxide', (0.6, 0, 0), 1.4442 ** 2)
# NITRIDE = Material('Nitride', (0, 0, 0.7), 1.996 ** 2)

SILICON = Material('Silicon', (0.3, 0.3, 0.3), 3.45 ** 2)
POLYSILICON = Material('Poly-Si', (0.5, 0.5, 0.5), 3.45 ** 2)
OXIDE = Material('Oxide', (0.6, 0, 0), 1.45 ** 2)
NITRIDE = Material('Nitride', (0, 0, 0.7), 2 ** 2)
LS_NITRIDE = Material('Low-Stress Nitride', (0, 0.4, 1))
# LT_OXIDE = Material('Low-Temp Oxide', (0.8, 0.2, 0.2), 1.4442 ** 2)
LT_OXIDE = Material('Low-Temp Oxide', (0.8, 0.2, 0.2), 1.45 ** 2)
ALUMINUM = Material('Aluminum', (0, 0.5, 0))
ALUMINA = Material('Alumina', (0.2, 0, 0.2), 1.75)
ETCH = Material('Etch', (0, 0, 0))


class ModeBlock:
    def __init__(self, dim: Dim2, material: Material):
        self.dim = dim
        self.material = material
        self.x = dim[0]
        self.y = dim[1]
        self.eps = self.material.eps


class Modes:
    def __init__(self, betas: np.ndarray, modes: np.ndarray, fdfd: FDFD):
        self.betas = betas.real
        self.modes = modes
        self.fdfd = copy.deepcopy(fdfd)
        self.eps = fdfd.eps

    @property
    @lru_cache()
    def h(self):
        return self.fdfd.reshape(self.modes[0])

    @property
    @lru_cache()
    def e(self):
        return self.fdfd.h2e(self.h, self.betas[0])

    @property
    @lru_cache()
    def sz(self):
        return poynting_z(self.e, self.h)

    @property
    @lru_cache()
    def beta(self):
        return self.betas[0]

    @property
    @lru_cache()
    def n(self):
        return self.betas[0] / self.fdfd.k0

    @property
    @lru_cache()
    def hs(self):
        hs = []
        for mode in self.modes:
            hs.append(self.fdfd.reshape(mode))
        return np.stack(hs).squeeze()

    @property
    @lru_cache()
    def es(self):
        es = []
        for beta, h in zip(self.betas, self.hs):
            es.append(self.fdfd.h2e(h[..., np.newaxis], beta))
        return np.stack(es).squeeze()

    @property
    @lru_cache()
    def szs(self):
        szs = []
        for beta, e, h in zip(self.betas, self.es, self.hs):
            szs.append(poynting_z(e[..., np.newaxis], h[..., np.newaxis]))
        return np.stack(szs).squeeze()

    @property
    @lru_cache()
    def ns(self):
        return self.betas / self.fdfd.k0

    @property
    def dbeta(self):
        return self.betas[0] - self.betas[1]

    @property
    def dn(self):
        return (self.betas[0] - self.betas[1]) / self.fdfd.k0

    @property
    def te_ratios(self):
        te_ratios = []
        for h in self.hs:
            habs = np.abs(h.squeeze())
            norms = np.asarray((np.linalg.norm(habs[0].flatten()), np.linalg.norm(habs[1].flatten())))
            te_ratios.append(norms[0] ** 2 / np.sum(norms ** 2))
        return np.asarray(te_ratios)


class ModeDevice:
    def __init__(self, wg: ModeBlock, sub: ModeBlock, size: Tuple[float, float], wg_height: float,
                 wavelength: float = 1.55, spacing: float = 0.01):
        self.size = size
        self.spacing = spacing
        self.nx = int(self.size[0] / spacing)
        self.ny = int(self.size[1] / spacing)
        self.fdfd = FDFD(
            shape=(self.nx, self.ny),
            spacing=spacing,
            wavelength=wavelength
        )
        self.wg_height = wg_height
        self.wg = wg
        self.sub = sub

        self.modes_list = []

    def solve(self, eps: np.ndarray, save: bool = False, m: int = 6):
        self.fdfd.eps = eps
        beta, modes = self.fdfd.wgm_solve(num_modes=m, beta_guess=self.fdfd.k0 * np.sqrt(self.wg.material.eps))
        solution = Modes(beta, modes, self.fdfd)
        if save:
            self.modes_list.append(solution)
        return solution

    def single(self, ps: Optional[ModeBlock] = None, sep: float = 0):
        nx, ny = self.nx, self.ny
        center = nx // 2
        wg, sub, dx = self.wg, self.sub, self.fdfd.spacing[0]
        wg_y = (self.wg_height, self.wg_height + wg.y)
        xr_wg = (center - int(wg.x / 2 / dx), center + int(wg.x / 2 / dx))
        yr_wg = (int(wg_y[0] / dx), int(wg_y[1] / dx))
        eps = np.ones((nx, ny))
        eps[:, :int(self.sub.y / dx)] = sub.material.eps
        eps[xr_wg[0]:xr_wg[1], yr_wg[0]:yr_wg[1]] = wg.material.eps

        if ps is not None:
            ps_y = (self.wg.y + self.wg_height + sep, self.wg.y + self.wg_height + sep + ps.y)
            xr_ps = (center - int(ps.x / 2 / dx), center + int(ps.x / 2 / dx))
            yr_ps = (int(ps_y[0] / dx), int(ps_y[1] / dx))
            eps[xr_ps[0]:xr_ps[1], yr_ps[0]:yr_ps[1]] = ps.material.eps

        return eps

    def double_ps(self, ps: Optional[ModeBlock] = None, sep: float = 0):
        nx, ny = self.nx, self.ny
        center = nx // 2
        wg, sub, dx = self.wg, self.sub, self.fdfd.spacing[0]
        wg_y = (self.wg_height, self.wg_height + wg.y)
        xr_wg = (center - int(wg.x / 2 / dx), center + int(wg.x / 2 / dx))
        yr_wg = (int(wg_y[0] / dx), int(wg_y[1] / dx))
        eps = np.ones((nx, ny))
        eps[:, :int(self.sub.y / dx)] = sub.material.eps
        eps[xr_wg[0]:xr_wg[1], yr_wg[0]:yr_wg[1]] = wg.material.eps

        if ps is not None:
            ps_y = (self.wg.y + self.wg_height + sep, self.wg.y + self.wg_height + sep + ps.y)
            xr_ps = (center - int(ps.x / 2 / dx), center + int(ps.x / 2 / dx))
            yr_ps = (int(ps_y[0] / dx), int(ps_y[1] / dx))
            yr_ps_2 = (yr_ps[1] + int(sep / dx), yr_ps[1] + int((sep + ps.y) / dx))
            eps[xr_ps[0]:xr_ps[1], yr_ps[0]:yr_ps[1]] = ps.material.eps
            eps[xr_ps[0]:xr_ps[1], yr_ps_2[0]:yr_ps_2[1]] = ps.material.eps

        return eps

    def coupled(self, gap: float, ps: Optional[ModeBlock] = None, seps: Tuple[float, float] = (0, 0)):
        nx, ny = self.nx, self.ny
        center = nx // 2
        wg, sub, dx = self.wg, self.sub, self.fdfd.spacing[0]
        wg_y = (self.wg_height, self.wg_height + wg.y)

        xr_l = (center - int((gap / 2 + wg.x) / dx), center - int(gap / 2 / dx))
        xr_r = (center + int((gap / 2) / dx), center + int((gap / 2 + wg.x) / dx))
        yr = (int(wg_y[0] / dx), int(wg_y[1] / dx))

        eps = np.ones((nx, ny))
        eps[:, :int(sub.y / dx)] = sub.eps
        eps[xr_l[0]:xr_l[1], yr[0]:yr[1]] = wg.eps
        eps[xr_r[0]:xr_r[1], yr[0]:yr[1]] = wg.eps

        if ps is not None:
            ps_y = (self.wg.y + self.wg_height + seps[0], self.wg.y + self.wg_height + seps[0] + ps.y)
            ps_y_2 = (self.wg.y + self.wg_height + seps[1], self.wg.y + self.wg_height + seps[1] + ps.y)

            wg_l, wg_r = (xr_l[0] + xr_l[1]) / 2, (xr_r[0] + xr_r[1]) / 2
            xrps_l = (int(wg_l - ps.x / dx / 2), int(wg_l + ps.x / dx / 2))
            xrps_r = (int(wg_r - ps.x / dx / 2), int(wg_r + ps.x / dx / 2))
            # xrps_l = (xr_l[0], xr_l[0] + int(ps.x / dx))
            # xrps_r = (xr_r[1] - int(ps.x / dx), xr_r[1])
            yr_ps = (int(ps_y[0] / dx), int(ps_y[1] / dx))
            yr_ps2 = (int(ps_y_2[0] / dx), int(ps_y_2[1] / dx))
            eps[xrps_l[0]:xrps_l[1], yr_ps[0]:yr_ps[1]] = ps.eps
            eps[xrps_r[0]:xrps_r[1], yr_ps2[0]:yr_ps2[1]] = ps.eps

        return eps

    def dc_grid(self, seps: np.ndarray, pbar: Callable, gap: float, ps: Optional[ModeBlock] = None, store: bool = True,
                m: int = 6):
        for sep_1 in pbar(seps):
            for sep_2 in pbar(seps):
                ps_height_1 = self.wg.y + self.wg_height + sep_1
                ps_height_2 = self.wg.y + self.wg_height + sep_2
                eps = self.coupled(gap, ps, seps=(ps_height_1, ps_height_2))
                self.solve(eps, store, m)

    def ps_sweep(self, seps: np.ndarray, pbar: Callable, ps: Optional[ModeBlock] = None, store: bool = True,
                 m: int = 6):
        for sep in pbar(seps):
            eps = self.single(ps, sep=self.wg.y + self.wg_height + sep)
            self.solve(eps, store, m)


def dispersion_sweep(device: ModeDevice, lmbdas: np.ndarray, pbar: Callable):
    solutions = []
    fdfd = device.fdfd
    for lmbda in pbar(lmbdas):
        fdfd.wavelength = lmbda
        beta, modes = fdfd.wgm_solve(num_modes=6)
        solutions.append(copy.deepcopy(Modes(beta, modes, device.fdfd)))
    return solutions

