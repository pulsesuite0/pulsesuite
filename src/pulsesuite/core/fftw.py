from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import scipy.fft as _sfft
    from scipy.special import jn_zeros as _jn_zeros, jv as _Jv
except Exception as _e:  # pragma: no cover
    import numpy.fft as _sfft  # fallback

    _jn_zeros = None  # type: ignore
    _Jv = None  # type: ignore

try:  # optional pyFFTW acceleration
    import pyfftw  # type: ignore

    _HAS_PYFFTW = True
    # Enable pyFFTW multithreading and wisdom caching
    pyfftw.interfaces.cache.enable()
    _FFTW_THREADS = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
except Exception:
    _HAS_PYFFTW = False
    _FFTW_THREADS = 1

_dp = np.float64
_dc = np.complex128

__all__ = [
    # 1D/2D/3D FFTs
    "fft_1D",
    "ifft_1D",
    "fftc_1D",
    "ifftc_1D",
    "fft_2D",
    "ifft_2D",
    "fftc_2D",
    "ifftc_2D",
    "fft_3D",
    "ifft_3D",
    "fftc_3D",
    "ifftc_3D",
    # lowercase aliases
    "fft_1d",
    "ifft_1d",
    "fftc_1d",
    "ifftc_1d",
    "fft_2d",
    "ifft_2d",
    "fftc_2d",
    "ifftc_2d",
    "fft_3d",
    "ifft_3d",
    "fftc_3d",
    "ifftc_3d",
    # Nyquist helpers
    "nyquist_1D",
    "nyquist_2D",
    "nyquist_3D",
    # Init no-ops for API compatibility
    "fftw_initialize_2D",
    "fftw_initialize_3D",
    # Hankel transform plumbing
    "Transform",
    "iTransform",
    "HankelTransform",
    "CreateHT",
]


# Utility helpers
def _asF(a: NDArray, dtype) -> NDArray:
    """Return Fortran-contiguous view/copy of `a` with dtype, no-copy if possible."""
    b = np.asarray(a, dtype=dtype, order="F")
    if not b.flags.f_contiguous:
        b = np.asfortranarray(b, dtype=dtype)
    return b


def _fft1(a: NDArray[_dc], inverse: bool = False) -> NDArray[_dc]:
    if _HAS_PYFFTW:  # fastest path
        # Use empty_aligned to guarantee alignment and preserve F-order.
        # Plan first (FFTW_MEASURE may destroy input), then copy data in.
        ain = pyfftw.empty_aligned(a.shape, dtype=_dc, order="F")
        out = pyfftw.empty_aligned(a.shape, dtype=_dc, order="F")
        fft_obj = pyfftw.FFTW(
            ain,
            out,
            direction="FFTW_BACKWARD" if inverse else "FFTW_FORWARD",
            threads=_FFTW_THREADS,
        )
        ain[...] = np.asfortranarray(a.astype(_dc, copy=False))
        fft_obj(ain, out)
        # pyfftw normalise_idft=True (default) already handles 1/N for inverse
        return out
    else:
        assert _sfft is not None, "SciPy is required when pyFFTW is unavailable."
        out = _sfft.ifft(a, axis=-1) if inverse else _sfft.fft(a, axis=-1)
        # SciPy normalizes inverse by N automatically; Fortran code also divides by size(Z)
        return out.astype(_dc, copy=False)


def _fftn(
    a: NDArray[_dc], axes: Tuple[int, ...], inverse: bool = False
) -> NDArray[_dc]:
    if _HAS_PYFFTW:
        # Use empty_aligned to guarantee alignment and preserve F-order.
        # Plan first (FFTW_MEASURE may destroy input), then copy data in.
        ain = pyfftw.empty_aligned(a.shape, dtype=_dc, order="F")
        out = pyfftw.empty_aligned(a.shape, dtype=_dc, order="F")
        fft_obj = pyfftw.FFTW(
            ain,
            out,
            axes=axes,
            direction="FFTW_BACKWARD" if inverse else "FFTW_FORWARD",
            threads=_FFTW_THREADS,
        )
        ain[...] = np.asfortranarray(a.astype(_dc, copy=False))
        fft_obj(ain, out)
        # pyfftw normalise_idft=True (default) already handles 1/N for inverse
        return out
    else:
        assert _sfft is not None, "SciPy is required when pyFFTW is unavailable."
        if inverse:
            out = _sfft.ifftn(a, axes=axes)
        else:
            out = _sfft.fftn(a, axes=axes)
        return out.astype(_dc, copy=False)


# Nyquist helpers (phase flip to center the FFT)
def nyquist_1D(Z: NDArray[_dc]) -> None:
    s = (1.0 - 2.0 * (np.arange(Z.shape[0]) % 2)).astype(_dp)
    Z *= s


def nyquist_2D(Z: NDArray[_dc]) -> None:
    i = np.arange(Z.shape[0])[:, None]
    j = np.arange(Z.shape[1])[None, :]
    s = (1.0 - 2.0 * ((i + j) % 2)).astype(_dp)
    Z *= s


def nyquist_3D(Z: NDArray[_dc]) -> None:
    i = np.arange(Z.shape[0])[:, None, None]
    j = np.arange(Z.shape[1])[None, :, None]
    k = np.arange(Z.shape[2])[None, None, :]
    s = (1.0 - 2.0 * ((i + j + k) % 2)).astype(_dp)
    Z *= s


# 1D FFTs (in-place semantics)
def fft_1D(Z: NDArray[_dc]) -> None:
    Zf = _asF(Z, _dc)
    out = _fft1(Zf, inverse=False)
    Z[...] = out


def ifft_1D(Z: NDArray[_dc]) -> None:
    Zf = _asF(Z, _dc)
    out = _fft1(Zf, inverse=True)
    Z[...] = out


def fftc_1D(Z: NDArray[_dc]) -> None:
    nyquist_1D(Z)
    fft_1D(Z)
    nyquist_1D(Z)


def ifftc_1D(Z: NDArray[_dc]) -> None:
    nyquist_1D(Z)
    ifft_1D(Z)
    nyquist_1D(Z)


# 2D FFTs
def fft_2D(Z: NDArray[_dc]) -> None:
    Zf = _asF(Z, _dc)
    out = _fftn(Zf, axes=(0, 1), inverse=False)
    Z[...] = out


def ifft_2D(Z: NDArray[_dc]) -> None:
    Zf = _asF(Z, _dc)
    out = _fftn(Zf, axes=(0, 1), inverse=True)
    Z[...] = out


def fftc_2D(Z: NDArray[_dc]) -> None:
    nyquist_2D(Z)
    fft_2D(Z)
    nyquist_2D(Z)


def ifftc_2D(Z: NDArray[_dc]) -> None:
    nyquist_2D(Z)
    ifft_2D(Z)
    nyquist_2D(Z)


# 3D FFTs
def fftw_initialize_2D(Z: NDArray[_dc]) -> None:
    """API-compat stub: planning handled by SciPy/pyFFTW internally."""
    return


def fftw_initialize_3D(Z: NDArray[_dc]) -> None:
    """API-compat stub: planning handled by SciPy/pyFFTW internally."""
    return


# 3D FFTs
def fft_3D(Z: NDArray[_dc]) -> None:
    Zf = _asF(Z, _dc)
    out = _fftn(Zf, axes=(0, 1, 2), inverse=False)
    Z[...] = out


def ifft_3D(Z: NDArray[_dc]) -> None:
    Zf = _asF(Z, _dc)
    out = _fftn(Zf, axes=(0, 1, 2), inverse=True)
    Z[...] = out


def fftc_3D(Z: NDArray[_dc]) -> None:
    nyquist_3D(Z)
    fft_3D(Z)
    nyquist_3D(Z)


def ifftc_3D(Z: NDArray[_dc]) -> None:
    nyquist_3D(Z)
    ifft_3D(Z)
    nyquist_3D(Z)


# Hankel transform (radial mix with FFT)
_HT: Optional[NDArray[_dp]] = None  # (Nr, Nr)
_a_zeros: Optional[NDArray[_dp]] = None  # J0 zeros, length Nr+1


def CreateHT(Nr: int) -> None:
    """Precompute Hankel transform matrix HT(Nr, Nr) from J0 zeros.

    Fortran reference:
      HT(m,n) = (2/a(Nr+1)) * J0( a(m)*a(n) / a(Nr+1) ) / J1(a(n))**2,   m,n=1..Nr
    where a(:) are the first Nr+1 zeros of J0.

    We store a length-(Nr+1) array and a (Nr,Nr) HT in double precision.
    """
    global _HT, _a_zeros
    if _HT is not None and _HT.shape == (Nr, Nr):
        return
    assert _jn_zeros is not None and _Jv is not None, (
        "scipy.special is required for CreateHT"
    )

    # J0 zeros; SciPy returns 1..Nr+1 zeros for order 0
    a = _jn_zeros(0, Nr + 1).astype(_dp)
    aN1 = a[-1]

    # Build HT
    mgrid, ngrid = np.meshgrid(a[:Nr], a[:Nr], indexing="ij")
    J0_arg = (mgrid * ngrid) / aN1
    J0 = _Jv(0, J0_arg).astype(_dp)
    J1 = _Jv(1, ngrid).astype(_dp)
    HT = (2.0 / aN1) * (J0 / (J1**2))

    _HT = np.asfortranarray(HT, dtype=_dp)
    _a_zeros = a


def HankelTransform(f: NDArray[_dc]) -> None:
    """Apply HT @ f in-place (f length must match HT dimension)."""
    if _HT is None:
        raise RuntimeError("CreateHT must be called before HankelTransform")
    if f.shape[0] != _HT.shape[0]:
        raise ValueError(
            f"HankelTransform: size mismatch {f.shape[0]} vs HT {_HT.shape}"
        )
    # Real-valued matrix multiply on complex vector; cast via real multiplication
    # Using einsum for cache-friendly matmul
    f[...] = (_HT @ f).astype(_dc, copy=False)


# Mixed transforms on 3D fields (r, y, t) with special 1×Nr×Nt radial case
def Transform(Z: NDArray[_dc]) -> None:
    """If Z has shape (1, Nr, Nt), apply Hankel along r and FFT along t; else FFT3D.

    Fortran logic:
        if size(Z,1) == 1:
            for k: HankelTransform(Z(1,:,k))
            for j: FFT(Z(1,j,:))
        else:
            FFT(Z)
    """
    if Z.shape[0] == 1:
        Nr = Z.shape[1]
        if _HT is None or _HT.shape[0] != Nr:
            CreateHT(Nr)
        # Hankel along axis=1 for each time slice
        for k in range(Z.shape[2]):
            HankelTransform(Z[0, :, k])
        # FFT along time axis for each angular index j
        for j in range(Z.shape[1]):
            fft_1D(Z[0, j, :])
    else:
        fft_3D(Z)


def iTransform(Z: NDArray[_dc]) -> None:
    """Inverse of Transform using ifft along t; Hankel is its own inverse for this discretization.

    Fortran logic called HankelTransform both ways; we mirror that.
    """
    if Z.shape[0] == 1:
        Nr = Z.shape[1]
        if _HT is None or _HT.shape[0] != Nr:
            CreateHT(Nr)
        for k in range(Z.shape[2]):
            HankelTransform(Z[0, :, k])
        for j in range(Z.shape[1]):
            ifft_1D(Z[0, j, :])
    else:
        ifft_3D(Z)


# Aliases with lowercase _1d/_2d/_3d to match some imports in the codebase
fft_1d = fft_1D
ifft_1d = ifft_1D
fftc_1d = fftc_1D
ifftc_1d = ifftc_1D

fft_2d = fft_2D
ifft_2d = ifft_2D
fftc_2d = fftc_2D
ifftc_2d = ifftc_2D

fft_3d = fft_3D
ifft_3d = ifft_3D
fftc_3d = fftc_3D
ifftc_3d = ifftc_3D
