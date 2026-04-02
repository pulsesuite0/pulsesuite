"""
Initial condition (IC) module for PSTD3D.

Seeds the electromagnetic field with a propagating Gaussian pulse at t=0.
The pulse uses a complex carrier exp(+ikx) for unidirectional +x propagation
(cos(kx) would create a standing wave). The impedance relation Bz = Ey/v
ensures only the forward-propagating mode is excited.

Also provides transverse grid validation (warns if the grid is too coarse
to resolve the carrier wavelength in the transverse direction).

Extracted from PSTD3D.py for modularity and independent testing.

Author: Emily S. Hatten
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass

try:
    from scipy.constants import c as _c0
except ImportError:
    _c0: float = 2.99792458e8

_dp = np.float64
_dc = np.complex128
_twopi = 2.0 * np.pi


def ValidateTransverseGrid(space, pulse) -> None:
    """Warn if transverse grid resolution is too coarse for the carrier.

    When dk_y = 2*pi / (Ny * dy) is much larger than the carrier
    k_med = 2*pi*n / lambda, spectral aliasing destroys propagation
    (standing-wave artifact). Rule of thumb: dk / k_carrier < 10 is safe.

    Fortran parity: PSTD3D.f90 lines 191-243.

    Parameters
    ----------
    space : ss
        Spatial grid structure.
    pulse : ps
        Pulse parameter structure.
    """
    from .typepulse import GetLambda
    from .typespace import GetDy, GetDz, GetEpsr, GetNy, GetNz

    Ny = GetNy(space)
    Nz = GetNz(space)
    epsr = GetEpsr(space)
    k_carrier = _twopi * np.sqrt(epsr) / GetLambda(pulse)

    if Ny > 1:
        dky = _twopi / (Ny * GetDy(space))
        ratio_y = dky / k_carrier
        if ratio_y > 10.0:
            Ny_min = math.ceil(_twopi / (k_carrier * GetDy(space)))
            warnings.warn(
                f"Too few transverse points in y: dk_y / k_carrier = {ratio_y:.1f}. "
                f"Ny = {Ny} needs >= {Ny_min} for dk_y ~ k_carrier. "
                f"This causes spectral aliasing that prevents pulse propagation "
                f"(standing-wave artifact). Fix: use Ny >= {Ny_min} or Ny = 1 for 1D.",
                stacklevel=2,
            )

    if Nz > 1:
        dkz = _twopi / (Nz * GetDz(space))
        ratio_z = dkz / k_carrier
        if ratio_z > 10.0:
            Nz_min = math.ceil(_twopi / (k_carrier * GetDz(space)))
            warnings.warn(
                f"Too few transverse points in z: dk_z / k_carrier = {ratio_z:.1f}. "
                f"Nz = {Nz} needs >= {Nz_min} for dk_z ~ k_carrier. "
                f"This causes spectral aliasing that prevents pulse propagation "
                f"(standing-wave artifact). Fix: use Nz >= {Nz_min} or Nz = 1 for 1D.",
                stacklevel=2,
            )


def SeedInitialCondition(
    space,
    time,
    pulse,
    Ey: NDArray[_dc],
    Bz: NDArray[_dc],
    npml_x: int = 0,
) -> None:
    """Seed Ey and Bz with the full pulse at t=t0 (IC mode).

    Seeds a +x propagating pulse using complex carrier exp(+i*k*xi) for
    unidirectional propagation (cos would create standing waves). Gaussian
    envelope with spatial width sigma_x = v * tau_G and carrier wavenumber
    k_med = omega0 / v. Impedance relation Bz = Ey / v ensures +x only.

    After seeding in real space, both fields are FFT'd to k-space.

    Parameters
    ----------
    space : ss
        Spatial grid structure.
    time : ts
        Time grid structure.
    pulse : ps
        Pulse parameter structure.
    Ey : ndarray (Nx, Ny, Nz), complex128
        Electric field (y-component).  Modified in-place.
    Bz : ndarray (Nx, Ny, Nz), complex128
        Magnetic field (z-component).  Modified in-place.
    npml_x : int, optional
        Number of PML cells per side on x-axis.  Field is zeroed in these
        regions (CPML/absorber assumes zero initial field there).
    """
    from ..core.fftw import fft_3D
    from .typepulse import GetAmp, GetChirp, GetW0
    from .typespace import GetEpsr, GetXArray, GetYArray, GetZArray

    Nx, Ny, Nz = Ey.shape

    x = GetXArray(space)
    yy = GetYArray(space)
    zz = GetZArray(space)

    omega0 = pulse.CalcOmega0()
    chirp_val = GetChirp(pulse)
    w0 = GetW0(pulse)
    tau_G = pulse.CalcTau()
    Emax = GetAmp(pulse)
    v = _c0 / np.sqrt(GetEpsr(space))

    # Spatial pulse parameters
    sigma_x = v * tau_G
    k_med = omega0 / v
    x_center = x[Nx // 4]  # 25% from left — gives 75% for +x propagation

    # Vectorised seeding over (Nx, Ny, Nz)
    xi = x - x_center  # (Nx,)
    xi_3 = xi[:, np.newaxis, np.newaxis]
    yy_3 = yy[np.newaxis, :, np.newaxis]
    zz_3 = zz[np.newaxis, np.newaxis, :]

    gauss_yz = np.exp(-(yy_3**2 + zz_3**2) / w0**2)

    # Complex carrier exp(+ikx) for unidirectional +x propagation.
    # exp(+ikx) populates ONLY +k modes (traveling wave in +x).
    Ey_vals = (
        Emax
        * gauss_yz
        * np.exp(-(xi_3**2) / sigma_x**2)
        * np.exp(1j * (k_med * xi_3 + (chirp_val / v**2) * xi_3**2))
    )

    # Smooth cosine taper in PML regions instead of hard zero.
    # This avoids Gibbs ringing from a sharp field discontinuity
    # at the PML boundary after FFT.
    # taper = 0 at outermost cell, 1 at interior edge.
    if npml_x > 0:
        taper = np.ones(Nx, dtype=_dp)
        for i in range(npml_x):
            taper[i] = 0.5 * (1.0 - np.cos(np.pi * i / npml_x))
        for i in range(Nx - npml_x, Nx):
            taper[i] = 0.5 * (1.0 - np.cos(np.pi * (Nx - 1 - i) / npml_x))
        Ey_vals = Ey_vals * taper[:, np.newaxis, np.newaxis]

    Ey[...] = Ey_vals
    Bz[...] = Ey_vals / v

    # Transform to k-space for the spectral time-stepper
    fft_3D(Ey)
    fft_3D(Bz)

    print(f"  IC seed: pulse center  x = {x_center * 1e6:.4f} um")
    print(f"  IC seed: spatial 1/e   sigma_x = {sigma_x * 1e6:.4f} um")
    print(f"  IC seed: spatial FWHM  = {sigma_x * 2.355 * 1e6:.4f} um")
    print(f"  IC seed: carrier lam_med = {_twopi / k_med * 1e9:.2f} nm")
    print(f"  IC seed: grid length   = {(x[-1] - x[0]) * 1e6:.4f} um")
