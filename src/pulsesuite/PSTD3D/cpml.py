"""cpml — 3D Convolutional PML for PSTD Maxwell solver.

Port of Fortran ``cpml.f90`.

CFS-PML Theory (Roden & Gedney 2000, Chen & Wang 2013):

    Coordinate stretching function:

    .. math:: s = \\kappa + \\sigma / (\\alpha + j\\omega\\varepsilon_0)

    Recursive convolution coefficients (same for E and H on collocated grid):

    .. math::
        b = \\exp\\!\\left(-\\left(\\frac{\\sigma}{\\kappa}+\\alpha\\right)
                          \\frac{\\Delta t}{\\varepsilon_0}\\right)
        \\qquad
        c = \\frac{\\sigma}{\\sigma\\kappa + \\alpha\\kappa^2}\\,(b-1)

    Auxiliary field update:

    .. math:: \\psi^{n+1} = b\\,\\psi^n + c\\,(\\partial F/\\partial x)

    Modified derivative:

    .. math::
        \\frac{1}{s}\\frac{\\partial F}{\\partial x}
        \\to \\frac{1}{\\kappa}\\frac{\\partial F}{\\partial x} + \\psi

Architecture
------------
- CPML corrections are **additive** to the standard PSTD spectral update
  (not a replacement FDTD-style update).
- Spectral derivatives are computed via ``IFFT(i k Field_k)``.
- 12 auxiliary ψ fields (6 for B, 6 for E) carry the convolution state between
  time steps.
- The inner PML loop is JIT-compiled with Numba ``parallel=True``
  (mirrors Fortran OpenMP).
- Module-level functions delegate to a default ``CPML`` instance
  for Fortran API parity.

Notes
-----
- Not thread-safe when using the module-level default instance (same as Fortran).
  For concurrent/multi-config use, create separate ``CPML`` instances directly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit, prange  # type: ignore

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore  # noqa: N807
        def _dec(f):
            return f

        if args and callable(args[0]):
            return args[0]
        return _dec

    prange = range  # type: ignore

try:
    from scipy.constants import c as _c0, epsilon_0 as _eps0
except ImportError:  # pragma: no cover
    _c0: float = 2.99792458e8
    _eps0: float = 8.8541878128e-12

from ..core.fftw import fft_3D, ifft_3D

_dp = np.float64
_dc = np.complex128

# ──────────────────────────────────────────────────────────────────────────────
# Grading constants (mirror Fortran module parameters)
# ──────────────────────────────────────────────────────────────────────────────
_M_PROFILE: int = 4         # polynomial grading order
_KAPPA_MAX: float = 8.0     # coordinate stretching maximum
_ALPHA_MAX: float = 0.05    # CFS shift parameter
_R_TARGET: float = 1.0e-8   # target reflection coefficient

# ──────────────────────────────────────────────────────────────────────────────
# Pure helper functions (module-level, no state — Fortran "pure" subroutines)
# ──────────────────────────────────────────────────────────────────────────────


def CalcNPML(N: int) -> int:
    """Compute optimal PML thickness for a grid dimension N.

    Uses 5 % of the grid per side, clamped to at least 6 cells.
    Returns 0 (no PML, periodic) when the grid is too small to
    support a stable PML layer (< 32 cells).

    Parameters
    ----------
    N : int
        Grid size along one axis.

    Returns
    -------
    int
        Number of PML cells per side (0 = periodic, no PML).
    """
    if N < 32:
        # Grid too small for stable PML — treat as periodic
        return 0
    npml = max(6, int(0.05 * N))
    if 2 * npml >= N:
        npml = max(6, N // 4)
    return npml


def CalcSigmaMax(L: float) -> float:
    r"""Compute maximum conductivity for a PML of physical thickness L.

    .. math::
        \sigma_{\max} = -\frac{(m+1)\,\varepsilon_0\,c_0\,\ln R_{\rm target}}{2L}

    Parameters
    ----------
    L : float
        Physical PML thickness (m).

    Returns
    -------
    float
        :math:`\sigma_{\max}` (S/m).
    """
    return -(_M_PROFILE + 1) * _eps0 * _c0 * np.log(_R_TARGET) / (2.0 * L)


def CalcCoefficients1D(
    sigma: NDArray[_dp],
    kappa: NDArray[_dp],
    alpha: NDArray[_dp],
    dt: float,
) -> tuple[NDArray[_dp], NDArray[_dp]]:
    r"""Compute CFS-PML recursive convolution coefficients b and c.

    .. math::
        b = \exp\!\left(-\left(\frac{\sigma}{\kappa}+\alpha\right)
                        \frac{\Delta t}{\varepsilon_0}\right)
        \qquad
        c = \frac{\sigma}{\sigma\kappa + \alpha\kappa^2}\,(b-1)

    Parameters
    ----------
    sigma, kappa, alpha : ndarray, shape (N,)
        CFS-PML 1-D profiles along one axis.
    dt : float
        Time step (s).

    Returns
    -------
    b, c : ndarray, shape (N,), dtype float64
        Exponential decay and scaling coefficients.

    Notes
    -----
    Both E and H use :math:`\varepsilon_0` (not :math:`\mu_0` for H) as per
    CFS-PML on a collocated PSTD grid (Roden & Gedney 2000).
    Interior points where ``sigma = 0`` yield ``b = 1``, ``c = 0``;
    the denominator guard prevents 0/0.
    """
    b = np.exp(-(sigma / kappa + alpha) * (dt / _eps0))
    denom = sigma * kappa + alpha * kappa**2
    safe = np.abs(denom) > 1.0e-20
    c = np.where(safe, sigma / np.where(safe, denom, 1.0) * (b - 1.0), 0.0)
    return b.astype(_dp), c.astype(_dp)


# ──────────────────────────────────────────────────────────────────────────────
# Numba JIT kernels (must be module-level; Numba cannot JIT class methods)
# fastmath=False: preserves FP order, required for correctness in field solvers.
# ──────────────────────────────────────────────────────────────────────────────


@njit(parallel=True, cache=True)
def _cpml_B_kernel(
    Bx, By, Bz,
    dEz_dy, dEy_dz,   # for Bx: (∂Ez/∂y, ∂Ey/∂z)
    dEx_dz, dEz_dx,   # for By: (∂Ex/∂z, ∂Ez/∂x)
    dEy_dx, dEx_dy,   # for Bz: (∂Ey/∂x, ∂Ex/∂y)
    psi_Bxy, psi_Bxz,
    psi_Byx, psi_Byz,
    psi_Bzx, psi_Bzy,
    bx, cx, by, cy, bz, cz,
    kappa_x, kappa_y, kappa_z,
    npml_x, npml_y, npml_z,
    Nx, Ny, Nz,
    dt,
):
    r"""B-field CPML correction kernel (Numba parallel, in-place).

    Applies the additive correction after the standard PSTD Faraday update:

    .. math::
        \Delta B_x = -dt \bigl[
          (1/\kappa_y - 1)\,\partial_y E_z + \psi_{Bxy}
          - (1/\kappa_z - 1)\,\partial_z E_y - \psi_{Bxz}
        \bigr]

    and cyclic permutations for :math:`B_y`, :math:`B_z`.

    Arrays Bx/By/Bz are complex128 (real space after IFFT); corrections are real.
    Arrays dE* are complex128; only the real part is used (imaginary ≈ 0 for
    physical fields with conjugate-symmetric k-space representation).
    """
    for i in prange(Nx):  # noqa: E741  (prange replaces OpenMP parallel do)
        in_x = (i < npml_x) or (i >= Nx - npml_x)
        for j in range(Ny):
            in_y = (j < npml_y) or (j >= Ny - npml_y)
            for k in range(Nz):
                in_z = (k < npml_z) or (k >= Nz - npml_z)
                if not (in_x or in_y or in_z):
                    continue

                # Extract real parts of spectral derivatives
                dEzdy = dEz_dy[i, j, k].real
                dEydz = dEy_dz[i, j, k].real
                dExdz = dEx_dz[i, j, k].real
                dEzdx = dEz_dx[i, j, k].real
                dEydx = dEy_dx[i, j, k].real
                dExdy = dEx_dy[i, j, k].real

                # Recursive convolution: ψ^(n+1) = b·ψ^n + c·(∂F/∂x)
                psi_Bxy[i, j, k] = by[j] * psi_Bxy[i, j, k] + cy[j] * dEzdy
                psi_Bxz[i, j, k] = bz[k] * psi_Bxz[i, j, k] + cz[k] * dEydz
                psi_Byx[i, j, k] = bx[i] * psi_Byx[i, j, k] + cx[i] * dEzdx
                psi_Byz[i, j, k] = bz[k] * psi_Byz[i, j, k] + cz[k] * dExdz
                psi_Bzx[i, j, k] = bx[i] * psi_Bzx[i, j, k] + cx[i] * dEydx
                psi_Bzy[i, j, k] = by[j] * psi_Bzy[i, j, k] + cy[j] * dExdy

                # Additive CPML corrections to B (real-space complex128)
                Bx[i, j, k] -= dt * (
                    (1.0 / kappa_y[j] - 1.0) * dEzdy + psi_Bxy[i, j, k]
                    - (1.0 / kappa_z[k] - 1.0) * dEydz - psi_Bxz[i, j, k]
                )
                By[i, j, k] -= dt * (
                    (1.0 / kappa_z[k] - 1.0) * dExdz + psi_Byz[i, j, k]
                    - (1.0 / kappa_x[i] - 1.0) * dEzdx - psi_Byx[i, j, k]
                )
                Bz[i, j, k] -= dt * (
                    (1.0 / kappa_x[i] - 1.0) * dEydx + psi_Bzx[i, j, k]
                    - (1.0 / kappa_y[j] - 1.0) * dExdy - psi_Bzy[i, j, k]
                )


@njit(parallel=True, cache=True)
def _cpml_E_kernel(
    Ex, Ey, Ez,
    dBz_dy, dBy_dz,   # for Ex: (∂Bz/∂y, ∂By/∂z)
    dBx_dz, dBz_dx,   # for Ey: (∂Bx/∂z, ∂Bz/∂x)
    dBy_dx, dBx_dy,   # for Ez: (∂By/∂x, ∂Bx/∂y)
    psi_Exy, psi_Exz,
    psi_Eyx, psi_Eyz,
    psi_Ezx, psi_Ezy,
    bx, cx, by, cy, bz, cz,
    kappa_x, kappa_y, kappa_z,
    npml_x, npml_y, npml_z,
    Nx, Ny, Nz,
    v2, dt,
):
    r"""E-field CPML correction kernel (Numba parallel, in-place).

    Applies the additive correction after the standard PSTD Ampere update:

    .. math::
        \Delta E_x = v^2 dt \bigl[
          (1/\kappa_y - 1)\,\partial_y B_z + \psi_{Exy}
          - (1/\kappa_z - 1)\,\partial_z B_y - \psi_{Exz}
        \bigr]

    and cyclic permutations for :math:`E_y`, :math:`E_z`.

    ``v2 = c_0^2 / eps_r`` is the phase velocity squared of the background medium.
    """
    for i in prange(Nx):
        in_x = (i < npml_x) or (i >= Nx - npml_x)
        for j in range(Ny):
            in_y = (j < npml_y) or (j >= Ny - npml_y)
            for k in range(Nz):
                in_z = (k < npml_z) or (k >= Nz - npml_z)
                if not (in_x or in_y or in_z):
                    continue

                dBzdy = dBz_dy[i, j, k].real
                dBydz = dBy_dz[i, j, k].real
                dBxdz = dBx_dz[i, j, k].real
                dBzdx = dBz_dx[i, j, k].real
                dBydx = dBy_dx[i, j, k].real
                dBxdy = dBx_dy[i, j, k].real

                # Recursive convolution
                psi_Exy[i, j, k] = by[j] * psi_Exy[i, j, k] + cy[j] * dBzdy
                psi_Exz[i, j, k] = bz[k] * psi_Exz[i, j, k] + cz[k] * dBydz
                psi_Eyx[i, j, k] = bx[i] * psi_Eyx[i, j, k] + cx[i] * dBzdx
                psi_Eyz[i, j, k] = bz[k] * psi_Eyz[i, j, k] + cz[k] * dBxdz
                psi_Ezx[i, j, k] = bx[i] * psi_Ezx[i, j, k] + cx[i] * dBydx
                psi_Ezy[i, j, k] = by[j] * psi_Ezy[i, j, k] + cy[j] * dBxdy

                # Additive CPML corrections to E
                Ex[i, j, k] += v2 * dt * (
                    (1.0 / kappa_y[j] - 1.0) * dBzdy + psi_Exy[i, j, k]
                    - (1.0 / kappa_z[k] - 1.0) * dBydz - psi_Exz[i, j, k]
                )
                Ey[i, j, k] += v2 * dt * (
                    (1.0 / kappa_z[k] - 1.0) * dBxdz + psi_Eyz[i, j, k]
                    - (1.0 / kappa_x[i] - 1.0) * dBzdx - psi_Eyx[i, j, k]
                )
                Ez[i, j, k] += v2 * dt * (
                    (1.0 / kappa_x[i] - 1.0) * dBydx + psi_Ezx[i, j, k]
                    - (1.0 / kappa_y[j] - 1.0) * dBxdy - psi_Ezy[i, j, k]
                )


# ──────────────────────────────────────────────────────────────────────────────
# CPML class — encapsulates all CPML state for one grid configuration
# ──────────────────────────────────────────────────────────────────────────────


class CPML:
    """3D Convolutional PML state for a PSTD Maxwell solver.

    Holds all CPML data: 1-D profiles (σ, κ, α), recursive convolution
    coefficients (b, c), 12 auxiliary ψ fields, and pre-allocated scratch
    buffers for spectral derivative computation.

    Multiple independent instances may coexist in the same process, unlike
    the Fortran module singleton.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Grid dimensions.
    dx, dy, dz : float
        Grid spacings (m).
    dt : float
        Time step (s).
    epsr : float, optional
        Background relative permittivity.  Accepted for API parity with
        Fortran ``InitCPML``; not used in the CFS-PML coefficient formulae
        (which use :math:`\\varepsilon_0` regardless of the medium).

    Notes
    -----
    Field arrays are expected in k-space (complex128, shape (Nx, Ny, Nz),
    C-order) on entry to ``ApplyCPML_B`` / ``ApplyCPML_E``.  They are
    temporarily IFFT'd to real space, corrected, and FFT'd back.
    """

    def __init__(
        self,
        Nx: int,
        Ny: int,
        Nz: int,
        dx: float,
        dy: float,
        dz: float,
        dt: float,
        epsr: float = 1.0,  # noqa: ARG002  API parity; unused in coefficients
    ) -> None:
        self.Nx, self.Ny, self.Nz = int(Nx), int(Ny), int(Nz)
        self.dx, self.dy, self.dz = float(dx), float(dy), float(dz)
        self.dt = float(dt)

        self.npml_x = CalcNPML(self.Nx)
        self.npml_y = CalcNPML(self.Ny)
        self.npml_z = CalcNPML(self.Nz)

        print("=== CPML Initialization ===")
        print(f"  Grid: {Nx}x{Ny}x{Nz}")
        print(
            f"  PML thickness:  npml_x={self.npml_x}"
            f"  npml_y={self.npml_y}"
            f"  npml_z={self.npml_z}"
        )

        # 1-D profiles: σ(d), κ(d), α(d) graded from outer PML edge inward
        self.sigma_x, self.kappa_x, self.alpha_x = self._build_profile_1d(
            self.Nx, self.dx, self.npml_x
        )
        self.sigma_y, self.kappa_y, self.alpha_y = self._build_profile_1d(
            self.Ny, self.dy, self.npml_y
        )
        self.sigma_z, self.kappa_z, self.alpha_z = self._build_profile_1d(
            self.Nz, self.dz, self.npml_z
        )

        # Recursive convolution coefficients  b, c  (same for E and H)
        self.bx, self.cx = CalcCoefficients1D(
            self.sigma_x, self.kappa_x, self.alpha_x, dt
        )
        self.by, self.cy = CalcCoefficients1D(
            self.sigma_y, self.kappa_y, self.alpha_y, dt
        )
        self.bz, self.cz = CalcCoefficients1D(
            self.sigma_z, self.kappa_z, self.alpha_z, dt
        )

        print(f"  sigma_max_x = {self.sigma_x.max():.6g}")
        print(f"  sigma_max_y = {self.sigma_y.max():.6g}")
        print(f"  sigma_max_z = {self.sigma_z.max():.6g}")

        # Auxiliary ψ fields — real-valued, full 3-D, C-order (cache-friendly
        # for the (i, j, k) Numba kernel with k as the innermost loop)
        shape = (self.Nx, self.Ny, self.Nz)
        self.psi_Bxy: NDArray[_dp] = np.zeros(shape, dtype=_dp)
        self.psi_Bxz: NDArray[_dp] = np.zeros(shape, dtype=_dp)
        self.psi_Byx: NDArray[_dp] = np.zeros(shape, dtype=_dp)
        self.psi_Byz: NDArray[_dp] = np.zeros(shape, dtype=_dp)
        self.psi_Bzx: NDArray[_dp] = np.zeros(shape, dtype=_dp)
        self.psi_Bzy: NDArray[_dp] = np.zeros(shape, dtype=_dp)
        self.psi_Exy: NDArray[_dp] = np.zeros(shape, dtype=_dp)
        self.psi_Exz: NDArray[_dp] = np.zeros(shape, dtype=_dp)
        self.psi_Eyx: NDArray[_dp] = np.zeros(shape, dtype=_dp)
        self.psi_Eyz: NDArray[_dp] = np.zeros(shape, dtype=_dp)
        self.psi_Ezx: NDArray[_dp] = np.zeros(shape, dtype=_dp)
        self.psi_Ezy: NDArray[_dp] = np.zeros(shape, dtype=_dp)

        # Scratch buffers for spectral derivatives — Fortran-contiguous so that
        # ifft_3D (which uses _asF internally) avoids an extra copy per call.
        # 6 buffers shared between ApplyCPML_B and ApplyCPML_E (called sequentially).
        self._d: list[NDArray[_dc]] = [
            np.zeros(shape, dtype=_dc, order="F") for _ in range(6)
        ]

        print("=== CPML Initialization Complete ===")

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_profile_1d(
        N: int, h: float, npml: int
    ) -> tuple[NDArray[_dp], NDArray[_dp], NDArray[_dp]]:
        r"""Build σ, κ, α profiles graded from the outer PML edge inward.

        .. math::
            \sigma(d) = \sigma_{\max}(d/L)^m
            \quad
            \kappa(d) = 1 + (\kappa_{\max}-1)(d/L)^m
            \quad
            \alpha(d) = \alpha_{\max}(1 - d/L)

        where :math:`d` is the fractional depth into the PML
        (0 = interior edge, 1 = outer wall) and :math:`L = \text{npml} \cdot h`.

        Index conversion (Fortran 1-based → Python 0-based):
        - Left PML  (``i < npml``):  ``pos = (npml - 1 - i + 0.5) / npml``
        - Right PML (``i >= N-npml``): ``pos = (i - (N - npml) + 0.5) / npml``
        The half-cell offset (``+0.5``) preserves the Fortran grading profile.
        """
        if npml == 0:
            # No PML on this axis — return flat identity profiles
            sigma = np.zeros(N, dtype=_dp)
            kappa = np.ones(N, dtype=_dp)
            alpha = np.zeros(N, dtype=_dp)
            return sigma, kappa, alpha

        L = npml * h
        sigma_max = CalcSigmaMax(L)
        m = float(_M_PROFILE)

        i_arr = np.arange(N, dtype=_dp)
        pos = np.zeros(N, dtype=_dp)

        left = i_arr < npml
        right = i_arr >= (N - npml)

        pos[left] = (npml - 1.0 - i_arr[left] + 0.5) / npml
        pos[right] = (i_arr[right] - (N - npml) + 0.5) / npml

        sigma = (sigma_max * pos**m).astype(_dp)
        kappa = (1.0 + (_KAPPA_MAX - 1.0) * pos**m).astype(_dp)
        alpha = (_ALPHA_MAX * (1.0 - pos)).astype(_dp)
        return sigma, kappa, alpha

    def _spectral_deriv(
        self,
        field_kspace: NDArray[_dc],
        k_1d: NDArray[_dp],
        axis: int,
        out: NDArray[_dc],
    ) -> None:
        r"""Compute :math:`\text{IFFT}(i\,k\,\hat{F})` into ``out`` (in-place).

        Parameters
        ----------
        field_kspace : complex128 ndarray, shape (Nx, Ny, Nz)
            Field in k-space (read-only).
        k_1d : float64 ndarray
            Wavenumber array along ``axis``; length must equal
            ``field_kspace.shape[axis]``.
        axis : int
            Derivative axis: 0 → x, 1 → y, 2 → z.
        out : complex128 ndarray, shape (Nx, Ny, Nz), Fortran-contiguous
            Pre-allocated output buffer; overwritten in-place.

        Notes
        -----
        ``out`` is Fortran-contiguous so that ``ifft_3D`` (which uses
        ``_asF`` internally) avoids an extra array copy on every call.
        """
        shape = [1, 1, 1]
        shape[axis] = -1
        np.multiply(1j * k_1d.reshape(shape), field_kspace, out=out)
        ifft_3D(out)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — Fortran name parity
    # ──────────────────────────────────────────────────────────────────────────

    def ApplyCPML_B(
        self,
        Bx: NDArray[_dc],
        By: NDArray[_dc],
        Bz: NDArray[_dc],
        Ex: NDArray[_dc],
        Ey: NDArray[_dc],
        Ez: NDArray[_dc],
        kx: NDArray[_dp],
        ky: NDArray[_dp],
        kz: NDArray[_dp],
        dt: float,
    ) -> None:
        r"""Apply CPML correction to B fields after standard PSTD update.

        The standard PSTD Faraday update computes
        ``B = B - dt·(∇×E)`` spectrally.  This method adds:

        .. math::
            \Delta B_x = -dt \bigl[
              (1/\kappa_y - 1)\,\partial_y E_z + \psi_{Bxy}
              - (1/\kappa_z - 1)\,\partial_z E_y - \psi_{Bxz}
            \bigr]

        (and cyclic permutations for :math:`B_y`, :math:`B_z`).

        Parameters
        ----------
        Bx, By, Bz : complex128 ndarray, shape (Nx, Ny, Nz)
            B fields in k-space.  **Modified in-place.**
        Ex, Ey, Ez : complex128 ndarray, shape (Nx, Ny, Nz)
            E fields in k-space.  Read-only.
        kx, ky, kz : float64 ndarray
            Wavenumber arrays (lengths Nx, Ny, Nz).
        dt : float
            Time step (s).

        Notes
        -----
        B fields are temporarily IFFT'd to real space, corrected, then FFT'd
        back.  E fields are never transformed.
        """
        d = self._d

        # Step 1: spectral derivatives of E → real space  ∂E_α/∂β = IFFT(i kβ Eα)
        self._spectral_deriv(Ez, ky, 1, d[0])  # ∂Ez/∂y
        self._spectral_deriv(Ey, kz, 2, d[1])  # ∂Ey/∂z
        self._spectral_deriv(Ex, kz, 2, d[2])  # ∂Ex/∂z
        self._spectral_deriv(Ez, kx, 0, d[3])  # ∂Ez/∂x
        self._spectral_deriv(Ey, kx, 0, d[4])  # ∂Ey/∂x
        self._spectral_deriv(Ex, ky, 1, d[5])  # ∂Ex/∂y

        # Step 2: IFFT B fields → real space
        ifft_3D(Bx)
        ifft_3D(By)
        ifft_3D(Bz)

        # Step 3: update ψ fields and apply corrections (Numba kernel)
        _cpml_B_kernel(
            Bx, By, Bz,
            d[0], d[1], d[2], d[3], d[4], d[5],
            self.psi_Bxy, self.psi_Bxz,
            self.psi_Byx, self.psi_Byz,
            self.psi_Bzx, self.psi_Bzy,
            self.bx, self.cx,
            self.by, self.cy,
            self.bz, self.cz,
            self.kappa_x, self.kappa_y, self.kappa_z,
            self.npml_x, self.npml_y, self.npml_z,
            self.Nx, self.Ny, self.Nz,
            dt,
        )

        # Step 4: FFT B fields → k-space
        fft_3D(Bx)
        fft_3D(By)
        fft_3D(Bz)

    def ApplyCPML_E(
        self,
        Ex: NDArray[_dc],
        Ey: NDArray[_dc],
        Ez: NDArray[_dc],
        Bx: NDArray[_dc],
        By: NDArray[_dc],
        Bz: NDArray[_dc],
        kx: NDArray[_dp],
        ky: NDArray[_dp],
        kz: NDArray[_dp],
        v2: float,
        dt: float,
    ) -> None:
        r"""Apply CPML correction to E fields after standard PSTD update.

        The standard PSTD Ampere update computes
        ``E = E + v²·dt·(∇×B)`` spectrally.  This method adds:

        .. math::
            \Delta E_x = v^2 dt \bigl[
              (1/\kappa_y - 1)\,\partial_y B_z + \psi_{Exy}
              - (1/\kappa_z - 1)\,\partial_z B_y - \psi_{Exz}
            \bigr]

        (and cyclic permutations for :math:`E_y`, :math:`E_z`).

        Parameters
        ----------
        Ex, Ey, Ez : complex128 ndarray, shape (Nx, Ny, Nz)
            E fields in k-space.  **Modified in-place.**
        Bx, By, Bz : complex128 ndarray, shape (Nx, Ny, Nz)
            B fields in k-space.  Read-only.
        kx, ky, kz : float64 ndarray
            Wavenumber arrays.
        v2 : float
            Phase velocity squared :math:`c_0^2 / \varepsilon_r` (m²/s²).
        dt : float
            Time step (s).
        """
        d = self._d

        # Step 1: spectral derivatives of B → real space
        self._spectral_deriv(Bz, ky, 1, d[0])  # ∂Bz/∂y
        self._spectral_deriv(By, kz, 2, d[1])  # ∂By/∂z
        self._spectral_deriv(Bx, kz, 2, d[2])  # ∂Bx/∂z
        self._spectral_deriv(Bz, kx, 0, d[3])  # ∂Bz/∂x
        self._spectral_deriv(By, kx, 0, d[4])  # ∂By/∂x
        self._spectral_deriv(Bx, ky, 1, d[5])  # ∂Bx/∂y

        # Step 2: IFFT E fields → real space
        ifft_3D(Ex)
        ifft_3D(Ey)
        ifft_3D(Ez)

        # Step 3: update ψ fields and apply corrections
        _cpml_E_kernel(
            Ex, Ey, Ez,
            d[0], d[1], d[2], d[3], d[4], d[5],
            self.psi_Exy, self.psi_Exz,
            self.psi_Eyx, self.psi_Eyz,
            self.psi_Ezx, self.psi_Ezy,
            self.bx, self.cx,
            self.by, self.cy,
            self.bz, self.cz,
            self.kappa_x, self.kappa_y, self.kappa_z,
            self.npml_x, self.npml_y, self.npml_z,
            self.Nx, self.Ny, self.Nz,
            v2, dt,
        )

        # Step 4: FFT E fields → k-space
        fft_3D(Ex)
        fft_3D(Ey)
        fft_3D(Ez)

    def GetCPMLInfo(self) -> tuple[int, int, int]:
        """Return PML cell counts per side for diagnostics.

        Returns
        -------
        npml_x, npml_y, npml_z : int
            Number of PML cells per side along x, y, z.
        """
        return self.npml_x, self.npml_y, self.npml_z

    def InPML(self, i: int, j: int, k: int) -> bool:
        """Check whether 0-based grid point ``(i, j, k)`` lies in any PML region.

        Parameters
        ----------
        i, j, k : int
            0-based grid indices (x, y, z).

        Returns
        -------
        bool
        """
        return (
            i < self.npml_x
            or i >= self.Nx - self.npml_x
            or j < self.npml_y
            or j >= self.Ny - self.npml_y
            or k < self.npml_z
            or k >= self.Nz - self.npml_z
        )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level default instance — Fortran subroutine API parity
#
# NOT thread-safe (mirrors Fortran module singleton).
# For concurrent use or multiple CPML configurations, instantiate CPML directly.
# ──────────────────────────────────────────────────────────────────────────────

_default: CPML | None = None  # NOT thread-safe


def InitCPML(
    Nx: int,
    Ny: int,
    Nz: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    epsr: float = 1.0,
) -> None:
    """Create (or replace) the module-level default ``CPML`` instance.

    Fortran signature parity: ``InitCPML(Nx, Ny, Nz, dx, dy, dz, dt, epsr)``.

    For multiple independent CPML configurations use ``CPML(...)`` directly.
    """
    global _default
    _default = CPML(Nx, Ny, Nz, dx, dy, dz, dt, epsr)


def ApplyCPML_B(
    Bx: NDArray[_dc],
    By: NDArray[_dc],
    Bz: NDArray[_dc],
    Ex: NDArray[_dc],
    Ey: NDArray[_dc],
    Ez: NDArray[_dc],
    kx: NDArray[_dp],
    ky: NDArray[_dp],
    kz: NDArray[_dp],
    dt: float,
) -> None:
    """Apply B-field CPML correction via the module-level default instance.

    Raises ``RuntimeError`` if ``InitCPML`` has not been called.
    """
    if _default is None:
        raise RuntimeError("Call InitCPML before ApplyCPML_B.")
    _default.ApplyCPML_B(Bx, By, Bz, Ex, Ey, Ez, kx, ky, kz, dt)


def ApplyCPML_E(
    Ex: NDArray[_dc],
    Ey: NDArray[_dc],
    Ez: NDArray[_dc],
    Bx: NDArray[_dc],
    By: NDArray[_dc],
    Bz: NDArray[_dc],
    kx: NDArray[_dp],
    ky: NDArray[_dp],
    kz: NDArray[_dp],
    v2: float,
    dt: float,
) -> None:
    """Apply E-field CPML correction via the module-level default instance.

    Raises ``RuntimeError`` if ``InitCPML`` has not been called.
    """
    if _default is None:
        raise RuntimeError("Call InitCPML before ApplyCPML_E.")
    _default.ApplyCPML_E(Ex, Ey, Ez, Bx, By, Bz, kx, ky, kz, v2, dt)


def GetCPMLInfo() -> tuple[int, int, int]:
    """Return PML thicknesses from the module-level default instance.

    Returns
    -------
    npml_x, npml_y, npml_z : int
    """
    if _default is None:
        raise RuntimeError("Call InitCPML before GetCPMLInfo.")
    return _default.GetCPMLInfo()
