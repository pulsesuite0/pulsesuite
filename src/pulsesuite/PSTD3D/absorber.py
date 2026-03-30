r"""absorber — Masking absorber boundary conditions for PSTD Maxwell solver.

Port of Fortran ``absorber.f90``.

Implements absorbing boundary conditions via multiplicative masking
(Kosloff & Kosloff 1986).  After each field update, the field is
multiplied by a pre-computed mask:

.. math::

    F^{n+1}(\mathbf{r}) = F^{n+1}_{\text{Maxwell}}(\mathbf{r})
                          \times \text{mask}(\mathbf{r})

where :math:`\text{mask} = 1` in the interior and :math:`\text{mask} < 1`
in the absorbing layer.  This is equivalent to adding a negative imaginary
potential (NIP):

.. math::

    \frac{\partial F}{\partial t} = [\text{Maxwell terms}]
                                    - \gamma(\mathbf{r})\,F

which causes exponential decay in the absorbing region.

Why this is stable
~~~~~~~~~~~~~~~~~~
The mask satisfies :math:`|\text{mask}| \le 1` everywhere, so the field
amplitude can only decrease or stay the same at each step.  There is no
recursive state (unlike CPML's :math:`\psi` fields), so there is **no
mechanism for energy growth**.  This is **unconditionally stable** for any
:math:`\Delta t`, any grid aspect ratio, any CFL number.

Absorption profile
~~~~~~~~~~~~~~~~~~
Polynomial grading:

.. math::

    \gamma(\rho) = \gamma_{\max}\,\rho^m

where :math:`\rho` is the normalised distance from the interior edge
(0 at interior, 1 at outermost cell) and :math:`m = 3` is the grading
order (Taflove & Hagness 2005, Sec. 7.6).

The mask value at each point:

.. math::

    \text{mask}(x) = \exp(-\gamma(x)\,\Delta t)

Using :math:`\exp` instead of :math:`(1 - \gamma \Delta t)` avoids the
need for :math:`\gamma \Delta t < 1` (Chen 2024, Eq. 7).

Architecture
------------
Module-level state (Fortran ``SAVE`` variables).  Not thread-safe.

References
----------
- Kosloff R, Kosloff D (1986) "Absorbing boundaries for wave propagation
  problems." *J Comput Phys* 63:363-376.
- Chen K (2024) "A General PSTD Method to Solve Quantum Scattering..."
  arXiv:2403.04053v2.  Eqs. 2-7.
- Taflove A, Hagness S (2005) *Computational Electrodynamics*, 3rd ed.,
  Artech House.  Sec. 7.6 (polynomial grading).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_dp = np.float64
_dc = np.complex128

# === Tunable parameters (module-level constants) ===
#
# U0_per_step: fraction of field absorbed per time step at the outermost
#   PML cell.  mask_edge = 1 - U0_per_step.  Over N_cross steps
#   (PML crossing time), total attenuation is (1 - U0_per_step)^N_cross.
#   For U0_per_step=0.5 and N_cross=100: 0.5^100 ~ 7.9e-31 = -301 dB.
U0_PER_STEP: float = 0.5

# m_grade: polynomial grading order for the absorption profile.
#   gamma(rho) = gamma_max * rho^m_grade.
#   Higher order = more gradual ramp near interior, sharper at edge.
#   m=3 is standard (Taflove & Hagness 2005, Sec. 7.6).
M_GRADE: int = 3

# frac_pml: fraction of grid per side used for absorbing layer.
FRAC_PML: float = 0.05

# NOT thread-safe — same as Fortran SAVE variables.
_state = {
    "Nx": 0,
    "Ny": 0,
    "Nz": 0,
    "npml_x": 0,
    "npml_y": 0,
    "npml_z": 0,
    "mask": None,
}


def CalcNPML(N: int) -> int:
    """Compute absorbing layer thickness for a given grid dimension.

    For axes with N <= 4 (effectively 1D simulation), no absorption
    is applied (npml = 0, mask = 1).

    Parameters
    ----------
    N : int
        Number of grid points along one axis.

    Returns
    -------
    int
        Number of PML cells per side.
    """
    if N <= 4:
        return 0

    npml = max(6, int(FRAC_PML * N))
    if npml > N // 10:
        npml = N // 10
    if 2 * npml >= N:
        npml = max(1, N // 4)
    return npml


def _build_1d_mask(N: int, npml: int, dt: float) -> NDArray[_dp]:
    """Construct the 1-D mask profile for one axis.

    The absorbing layer occupies cells ``[0, npml-1]`` at the left and
    ``[N-npml, N-1]`` at the right.  Interior cells have ``mask = 1``.

    Parameters
    ----------
    N : int
        Number of grid points.
    npml : int
        Number of PML cells per side.
    dt : float
        Time step (s).

    Returns
    -------
    ndarray, shape (N,), dtype float64
        1-D mask profile.
    """
    mask1d = np.ones(N, dtype=_dp)

    if npml <= 0:
        return mask1d

    # gamma_max such that mask at outermost cell = 1 - U0_PER_STEP
    gamma_max = -np.log(1.0 - U0_PER_STEP) / dt

    # Left absorbing layer: cells 0 to npml-1
    # rho = 0 at interior edge (cell npml-1), rho = 1 at outermost (cell 0)
    for i in range(npml):
        rho = (npml - 1 - i) / max(npml - 1, 1)
        gamma_val = gamma_max * rho**M_GRADE
        mask1d[i] = np.exp(-gamma_val * dt)

    # Right absorbing layer: cells N-npml to N-1
    # rho = 0 at interior edge, rho = 1 at outermost cell
    for i in range(N - npml, N):
        rho = (i - (N - npml)) / max(npml, 1)
        gamma_val = gamma_max * rho**M_GRADE
        mask1d[i] = np.exp(-gamma_val * dt)

    return mask1d


def InitAbsorber(Nx: int, Ny: int, Nz: int, dt: float) -> None:
    """Pre-compute the 3-D absorbing mask.  Called once before the time loop.

    The mask is separable:
    ``mask(i,j,k) = mask_x(i) * mask_y(j) * mask_z(k)``.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Grid dimensions.
    dt : float
        Time step (s).
    """
    _state["Nx"] = Nx
    _state["Ny"] = Ny
    _state["Nz"] = Nz

    npml_x = CalcNPML(Nx)
    npml_y = CalcNPML(Ny)
    npml_z = CalcNPML(Nz)

    _state["npml_x"] = npml_x
    _state["npml_y"] = npml_y
    _state["npml_z"] = npml_z

    mask_x = _build_1d_mask(Nx, npml_x, dt)
    mask_y = _build_1d_mask(Ny, npml_y, dt)
    mask_z = _build_1d_mask(Nz, npml_z, dt)

    # Separable 3D mask via broadcasting:
    #   mask_x -> (Nx,1,1),  mask_y -> (1,Ny,1),  mask_z -> (1,1,Nz)
    _state["mask"] = (
        mask_x[:, np.newaxis, np.newaxis]
        * mask_y[np.newaxis, :, np.newaxis]
        * mask_z[np.newaxis, np.newaxis, :]
    )

    print("=== Absorber Initialization ===")
    print("  Type: Masking (polynomial grading, Kosloff & Kosloff 1986)")
    print(f"  Grid: {Nx} x {Ny} x {Nz}")
    print(f"  PML cells: npml_x={npml_x}  npml_y={npml_y}  npml_z={npml_z}")
    print(f"  U0_per_step = {U0_PER_STEP}  m_grade = {M_GRADE}")
    print(f"  Mask range: [{_state['mask'].min():.6e}, {_state['mask'].max():.6e}]")
    print("=== Absorber Initialization Complete ===")


def ApplyAbsorber_E(
    Ex: NDArray[_dc],
    Ey: NDArray[_dc],
    Ez: NDArray[_dc],
) -> None:
    """Apply absorbing mask to E fields (in-place).

    Called **after** the spectral Maxwell update for E.

    Algorithm:
        1. IFFT E fields to real space
        2. Multiply by pre-computed mask (element-wise)
        3. FFT back to k-space

    Parameters
    ----------
    Ex, Ey, Ez : ndarray (Nx, Ny, Nz), complex128
        Electric field components in k-space.  Modified in-place.
    """
    from ..core.fftw import fft_3D, ifft_3D

    mask = _state["mask"]

    ifft_3D(Ex)
    ifft_3D(Ey)
    ifft_3D(Ez)

    Ex *= mask
    Ey *= mask
    Ez *= mask

    fft_3D(Ex)
    fft_3D(Ey)
    fft_3D(Ez)


def ApplyAbsorber_B(
    Bx: NDArray[_dc],
    By: NDArray[_dc],
    Bz: NDArray[_dc],
) -> None:
    """Apply absorbing mask to B fields (in-place).

    Called **after** the spectral Maxwell update for B.
    Same algorithm as ``ApplyAbsorber_E``.

    Parameters
    ----------
    Bx, By, Bz : ndarray (Nx, Ny, Nz), complex128
        Magnetic field components in k-space.  Modified in-place.
    """
    from ..core.fftw import fft_3D, ifft_3D

    mask = _state["mask"]

    ifft_3D(Bx)
    ifft_3D(By)
    ifft_3D(Bz)

    Bx *= mask
    By *= mask
    Bz *= mask

    fft_3D(Bx)
    fft_3D(By)
    fft_3D(Bz)


def GetAbsorberInfo() -> tuple[int, int, int]:
    """Return PML thicknesses for diagnostics.

    Returns
    -------
    npml_x, npml_y, npml_z : int
        Number of absorbing-layer cells per side on each axis.
    """
    return _state["npml_x"], _state["npml_y"], _state["npml_z"]
