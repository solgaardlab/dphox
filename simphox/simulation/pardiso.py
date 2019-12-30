import sys
from ctypes import CDLL, byref, c_int, c_longlong, POINTER

import numpy as np
import scipy.sparse as sp
import hashlib


libname = {'linux': 'libmkl_rt.so',  # python3
           'linux2': 'libmkl_rt.so',  # python2
           'darwin': 'libmkl_rt.dylib',
           'win32': 'mkl_rt.dll'}
mkl = CDLL(libname[sys.platform])

pardisoinit = mkl.pardisoinit
pardisoinit.argtypes = [POINTER(c_longlong),
                        POINTER(c_int),
                        POINTER(c_int)]
pardisoinit.restype = None

pardiso = mkl.pardiso
pardiso.argtypes = [POINTER(c_longlong),  # pt
                    POINTER(c_int),  # maxfct
                    POINTER(c_int),  # mnum
                    POINTER(c_int),  # mtype
                    POINTER(c_int),  # phase
                    POINTER(c_int),  # n
                    POINTER(None),  # a
                    POINTER(c_int),  # ia
                    POINTER(c_int),  # ja
                    POINTER(c_int),  # perm
                    POINTER(c_int),  # nrhs
                    POINTER(c_int),  # iparm
                    POINTER(c_int),  # msglvl
                    POINTER(None),   # rhs
                    POINTER(None),   # x
                    POINTER(c_int)]  # error
pardiso.restype = None

from ..typing import Optional

PARDISO_FREEFACTOR = -1
PARDISO_FREEALL = 0
PARDISO_SOLVE = 33
PARDISO_FACTORIZE = 12
PARDISO_FULLSOLVE = 13


# a cleaner version of pyMKL with some additional factorization optimizations
class Pardiso:
    def __init__(self, mtype: int = 13):
        self.mtype = mtype
        if mtype in (1, 3):
            raise NotImplementedError(f"mtype = {mtype} - structurally symmetric not supported")
        if self.is_complex:
            self.dtype = np.complex128
        elif self.is_real:
            self.dtype = np.float64
        else:
            raise ValueError(f"mtype = {mtype} - invalid mtype, need (2, -2, 4, -4, 6, 11, 13)")
        self.ctypes_dtype = np.ctypeslib.ndpointer(self.dtype)

        self.pt = np.zeros(64, np.int64)
        self.pt_ = self.pt.ctypes.data_as(POINTER(c_longlong))

        self.iparm = np.zeros(64, dtype=np.int32)
        self.iparm_ = self.iparm.ctypes.data_as(POINTER(c_int))

        pardisoinit(self.pt_, byref(c_int(self.mtype)), self.iparm_)

        # from pyMKL
        self.iparm[1] = 3   # Parallel nested dissection for reordering
        self.iparm[23] = 1  # Parallel factorization
        self.iparm[34] = 1  # Zero-indexing

        self.phase = PARDISO_FULLSOLVE
        self._mat_hash = 0  # no matrix has been factorized yet, and this is an unlikely hash for it to be assigned

    @property
    def is_complex(self) -> bool:
        return self.mtype in (4, -4, 6, 13)

    @property
    def is_real(self) -> bool:
        return self.mtype in (2, -2, 11)

    def _set_mat(self, mat: sp.csr_matrix):
        # If mat is symmetric, store only the upper triangular portion
        if self.mtype in [2, -2, 4, -4, 6]:
            mat = sp.triu(mat, format='csr')

        if mat.dtype != self.dtype:
            raise ValueError(f"Expected mat.dtype to match chosen mtype but got {mat.dtype} != {self.dtype}")
        if mat.shape[0] != mat.shape[1] or mat.ndim > 2:
            raise ValueError(f'Expected mat square (i.e., shape (n, n)), but has shape: {mat.shape}')

        if not mat.has_sorted_indices:
            mat.sort_indices()

        self.mat: sp.csr_matrix = mat

        self.a = self.mat.data
        self.a_ = self.a.ctypes.data_as(self.ctypes_dtype)

        self.ia = self.mat.indptr
        self.ia_ = self.ia.ctypes.data_as(POINTER(c_int))

        self.ja = self.mat.indices
        self.ja_ = self.ja.ctypes.data_as(POINTER(c_int))

        self.n = mat.shape[0]

    def free(self, complete: bool = True):
        self.phase = PARDISO_FREEALL if complete else PARDISO_FREEFACTOR
        self.pardiso()

    def factor(self, mat: sp.csr_matrix):
        self.phase = PARDISO_FACTORIZE
        self._set_mat(mat)
        return self.pardiso()

    def solve(self, mat: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        if spm_hash(mat) != self._mat_hash:
            self._set_mat(mat)
            self._mat_hash = spm_hash(mat)
            self.phase = PARDISO_FULLSOLVE
        else:
            self.phase = PARDISO_SOLVE
        return self.pardiso(rhs)

    def pardiso(self, rhs: Optional[np.ndarray] = None) -> np.ndarray:
        if self._mat_hash == 0:
            raise RuntimeError('Mat information not stored in Pardiso.')
        if rhs is not None and rhs.shape[0] != self.n:
            raise RuntimeError(f'Expected rhs.shape[0] == {self.n}, but got {rhs.shape[0]}')

        nrhs = 0 if rhs is None else (np.prod(rhs.shape[1:]) if rhs.ndim > 1 else 1)
        rhs = np.zeros(1) if rhs is None else rhs.astype(self.dtype).flatten(order='f')
        x = np.zeros(1) if rhs is None else np.zeros(nrhs * self.n, dtype=self.dtype)
        rhs_ = rhs.ctypes.data_as(self.ctypes_dtype)
        x_ = x.ctypes.data_as(self.ctypes_dtype)

        err_c = c_int(0)

        mkl.pardiso(
            self.pt_,                  # pt
            byref(c_int(1)),           # maxfct
            byref(c_int(1)),           # mnum
            byref(c_int(self.mtype)),  # mtype
            byref(c_int(self.phase)),  # phase
            byref(c_int(self.n)),      # n
            self.a_,                   # a
            self.ia_,                  # ia
            self.ja_,                  # ja
            byref(c_int(0)),           # perm
            byref(c_int(nrhs)),        # nrhs
            self.iparm_,               # iparm
            byref(c_int(0)),           # msglvl
            rhs_,                      # rhs
            x_,                        # x
            byref(err_c)               # error
        )

        if self.iparm[13] > 0:
            raise RuntimeError(f"Pardiso - Number of perturbed pivot elements = {repr(self.iparm[13])}. "
                               f"This could mean that the matrix is singular.")

        if err_c.value != 0:
            raise RuntimeError(f"Pardiso returned an error with code {err_c.value}. "
                               f"Check error codes in manual: https://pardiso-project.org/manual/manual.pdf")

        return x.reshape(rhs.shape, order='f') if nrhs > 1 else x


pardiso = Pardiso()


def spsolve(mat: sp.spmatrix, rhs: np.ndarray):
    if not isinstance(mat, sp.spmatrix):
        raise TypeError(f'mat must be an instance of spmatrix but got {type(mat)}')
    if not isinstance(rhs, np.ndarray):
        raise TypeError(f'mat must be an instance of ndarray but got {type(rhs)}')
    return pardiso.solve(mat.tocsr(), rhs)


def spm_hash(mat: sp.csr_matrix):
    return hashlib.sha1(mat.indices).hexdigest() + hashlib.sha1(mat.data).hexdigest()
