"""Additive soft-source injection for PSTD3D Maxwell solver.

Adds incident field via smooth Gaussian profile to avoid Gibbs oscillations.
Unidirectional +x injection via impedance-matched Ey and Bz.

Author: Emily S. Hatten
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.constants import c as _c0
except ImportError:  # pragma: no cover
    _c0: float = 2.99792458e8

from ..core.fftw import fft_3D, ifft_3D

_dp = np.float64
_dc = np.complex128
_twopi = 2.0 * np.pi


def InitializeTFSF(space, pulse) -> NDArray[_dp]:
    """Compute the 1-D normalised Gaussian source profile.

    Uses a narrow Gaussian with sigma = 5*dx, normalised so that
    sum(S * dx) ~ 1. This avoids Gibbs ringing from the spectral derivatives.

    Parameters
    ----------
    space : ss
        Spatial grid structure.
    pulse : ps
        Pulse parameter structure.

    Returns
    -------
    ndarray, shape (Nx,), dtype float64
        Normalised Gaussian source profile.
    """
    from .typespace import GetDx, GetNx, GetXArray

    Nx = GetNx(space)
    x = GetXArray(space)
    dx = GetDx(space)

    # Source location: 25 % into the x-array (away from left PML)
    xp = x[int(round(Nx * 0.25))]

    # Narrow Gaussian: sigma = 5*dx
    sigma_src = 5.0 * dx

    print(f"  TFSF source: xp = {xp:.6e} m, sigma = {sigma_src:.6e} m")

    # Normalised Gaussian: integral(S * dx) = 1
    tfsf = np.exp(-0.5 * ((x - xp) / sigma_src) ** 2) / (sigma_src * np.sqrt(_twopi))
    return tfsf


def UpdateTFSC(
    E: NDArray[_dc],
    tfsf: NDArray[_dp],
    space,
    time,
    pulse,
    Emax_amp: float,
) -> None:
    """Additive soft-source injection (in-place).

    Adds the analytical incident field to the propagated field, weighted by
    the source profile. The dx factor makes the emitted amplitude independent
    of grid spacing.

    Parameters
    ----------
    E : ndarray, shape (Nx, Ny, Nz), complex128
        Field component in k-space. Modified in-place.
    tfsf : ndarray, shape (Nx,), float64
        Source profile from ``InitializeTFSF``.
    space : ss
        Spatial grid structure.
    time : ts
        Time grid structure.
    pulse : ps
        Pulse parameter structure.
    Emax_amp : float
        Amplitude scale factor for the incident field (V/m for E, T for B).
    """
    from .typepulse import GetChirp, GetTp, GetW0
    from .typespace import GetDx, GetEpsr, GetXArray, GetYArray, GetZArray
    from .typetime import GetT

    omega0 = pulse.CalcOmega0()
    chirp_val = GetChirp(pulse)
    w0 = GetW0(pulse)
    tau_G = pulse.CalcTau()
    v = _c0 / np.sqrt(GetEpsr(space))
    dx = GetDx(space)

    x = GetXArray(space)
    yy = GetYArray(space)
    zz = GetZArray(space)
    t_now = GetT(time)
    Tp = GetTp(pulse)

    # Retarded time: tau(x) = t - x/v - Tp   — shape (Nx,)
    tau = t_now - x / v - Tp

    # IFFT field to real space
    E_real = E.copy()
    ifft_3D(E_real)

    # Vectorised additive injection over (Nx, Ny, Nz)
    # tau -> (Nx,1,1),  yy -> (1,Ny,1),  zz -> (1,1,Nz)
    tau_3 = tau[:, np.newaxis, np.newaxis]
    yy_3 = yy[np.newaxis, :, np.newaxis]
    zz_3 = zz[np.newaxis, np.newaxis, :]

    # Transverse Gaussian beam profile (paraxial approximation)
    gauss_yz = np.exp(-(yy_3**2 + zz_3**2) / w0**2)

    # Analytical incident field: chirped Gaussian pulse
    # propagating in +x (retarded time tau = t - x/v - Tp)
    E_inc = (
        Emax_amp
        * gauss_yz
        * np.exp(-(tau_3**2) / tau_G**2)
        * np.cos(omega0 * tau_3 + chirp_val * tau_3**2)
    )

    # Source profile -> (Nx,1,1) for broadcasting
    S = tfsf[:, np.newaxis, np.newaxis]

    # Additive injection: E += S(x) * E_inc * dx
    E_real += S * E_inc * dx

    # FFT back to k-space
    fft_3D(E_real)
    E[...] = E_real
