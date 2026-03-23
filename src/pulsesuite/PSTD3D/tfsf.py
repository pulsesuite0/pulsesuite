r"""tfsf — Additive soft-source injection for PSTD3D Maxwell solver.

Port of the ``InitializeTFSF`` and ``UpdateTFSC`` subroutines contained
in Fortran ``PSTD3D.F90``.

Theory & Motivation
-------------------
In PSTD, spatial derivatives are computed via FFT (Liu 1997, *Microwave
Opt. Technol. Lett.* 15:158).  The FFT treats the computational domain as
spatially periodic and each derivative depends on **all** grid points
simultaneously.  This has two consequences for source injection:

(a) A replacement / blend source that forces field values in a localised
    region creates discontinuities in the field's spatial derivatives at
    the blend boundary.  The spectral method amplifies these into Gibbs
    oscillations that wrap around the periodic domain (Munro *et al.*
    2015, *J. Biomed. Opt.* 20:095007; Jerri 1998, *The Gibbs Phenomenon
    in Fourier Analysis*).

(b) Even smooth blend windows (super-Gaussian) still create a
    near-discontinuity in :math:`d^n E/dx^n` at the transition zone,
    producing high-*k* spectral content that aliases on the discrete grid.

**Fix**: Additive soft source (Schneider 2010, *Understanding the FDTD
Method*, Sec. 5.5; Taflove & Hagness 2005, *Computational
Electrodynamics*, Ch. 5).  Instead of replacing field values, we **add**
the incident field at each time step:

.. math::

    E(\mathbf{r}) = E_{\text{prop}}(\mathbf{r})
                   + S(x)\,E_{\text{inc}}(\mathbf{r}, t)

where :math:`S(x)` is a narrow, smooth (band-limited) normalised Gaussian
source profile.  This generates outgoing waves from the source location
without creating derivative discontinuities.

Unidirectional injection
~~~~~~~~~~~~~~~~~~~~~~~~
A soft source alone radiates in both :math:`+x` and :math:`-x` directions.
To produce a unidirectional :math:`+x` wave, the caller injects both
:math:`E_y` and :math:`B_z` with the plane-wave impedance ratio
:math:`E_y / B_z = v` (Taflove & Hagness 2005, Sec. 5.2.1).

Incident field
~~~~~~~~~~~~~~
.. math::

    E_{\text{inc}}(x,t) = A\,G(y,z)\,
        \exp\!\left(-\tau^2 / \tau_G^2\right)\,
        \cos(\omega_0 \tau + \chi \tau^2)

    \tau = t - x/v - T_p  \quad (\text{retarded time})

Cost: 1 IFFT + 1 FFT per call (2 FFTs total).

Architecture
------------
Standalone functions (no class needed — the only state is the 1-D
source profile array, returned by ``InitializeTFSF``).
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


# ──────────────────────────────────────────────────────────────────────────────
# InitializeTFSF
# ──────────────────────────────────────────────────────────────────────────────


def InitializeTFSF(space, pulse) -> NDArray[_dp]:
    r"""Compute the 1-D normalised Gaussian source profile.

    The previous implementation used a wide super-Gaussian window (order 6,
    width up to 10 % of the grid).  This caused Gibbs ringing and
    wraparound in the spectral derivatives because the near-discontinuous
    transition at the window edges introduced high-*k* content (Liu 1997;
    Munro *et al.* 2015).

    **Fix**: a narrow Gaussian with :math:`\sigma = 5\,\Delta x`, normalised
    so that :math:`\sum S(x)\,\Delta x \approx 1`.

    Why :math:`\sigma = 5\,\Delta x`:

    - The Gaussian is sampled by >10 points across its :math:`2\sigma`
      width, satisfying Nyquist for its own spectral content (Shannon 1949,
      *Proc. IRE* 37:10).  Its Fourier transform has spectral width
      :math:`1/\sigma`, which equals :math:`\sim 6\%` of
      :math:`k_{\text{Nyquist}} = \pi/\Delta x`.
    - The profile decays to :math:`e^{-12.5} \approx 3.7 \times 10^{-6}` at
      5 cells from centre and to machine zero at ~16 cells.
    - Normalisation makes the emitted amplitude proportional to
      ``Emax_amp``, independent of :math:`\sigma` or :math:`\Delta x`.

    Parameters
    ----------
    space : ss
        Spatial grid structure (``typespace.ss``).
    pulse : ps
        Pulse parameter structure (``typepulse.ps``).

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
    tfsf = np.exp(-0.5 * ((x - xp) / sigma_src) ** 2) / (
        sigma_src * np.sqrt(_twopi)
    )
    return tfsf


# ──────────────────────────────────────────────────────────────────────────────
# UpdateTFSC
# ──────────────────────────────────────────────────────────────────────────────


def UpdateTFSC(
    E: NDArray[_dc],
    tfsf: NDArray[_dp],
    space,
    time,
    pulse,
    Emax_amp: float,
) -> None:
    r"""Additive soft-source injection (in-place).

    At each time step the analytical incident field is **added** to the
    propagated field, weighted by the narrow source profile:

    .. math::

        E(\mathbf{r}) = E_{\text{prop}}(\mathbf{r})
                       + S(x)\,E_{\text{inc}}(\mathbf{r}, t)\,\Delta x

    This is the "soft source" approach (Schneider 2010, Ch. 5): the field
    is not replaced but augmented, so Maxwell's equations remain
    self-consistent everywhere.  The propagated field passes through the
    source region undisturbed, while the additive term generates new
    outgoing waves.

    The factor :math:`\Delta x` converts the normalised profile
    (:math:`\int S\,dx = 1`) to a dimensionally correct discrete source,
    making the emitted amplitude proportional to ``Emax_amp`` regardless
    of grid spacing or source width :math:`\sigma`.

    Parameters
    ----------
    E : ndarray, shape (Nx, Ny, Nz), complex128
        Field component in k-space.  Modified in-place.
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
