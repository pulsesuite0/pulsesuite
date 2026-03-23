r"""PSTD3D — 3-D pseudo-spectral time-domain Maxwell propagator.

Port of Fortran ``PSTD3D.F90``.

Solves the 3-D Maxwell equations in Fourier space using a leapfrog
time-stepping scheme with absorbing boundaries and source injection.

Time-stepping equations
~~~~~~~~~~~~~~~~~~~~~~~
.. math::

    \mathbf{E}^{n+1} = \mathbf{E}^n
        + i\,v^2 \Delta t\,(\mathbf{k} \times \mathbf{B}^n)
        - \mu_0 v^2 \Delta t\,\mathbf{J}^n

    \mathbf{B}^{n+1} = \mathbf{B}^n
        - i\,\Delta t\,(\mathbf{k} \times \mathbf{E}^{n+1})

where :math:`v = c_0 / \sqrt{\varepsilon_r}` and the spectral curl
:math:`\mathbf{k} \times` replaces spatial derivatives exactly
(no truncation error — only time discretisation is approximate).

Architecture
------------
Hybrid: a ``PSTD3DPropagator`` class holds simulation state (fields,
source profile, output directory) while the spectral update kernels
(``UpdateE3D``, ``UpdateB3D``) are standalone vectorised functions.
This mirrors the Fortran layout where the ``program`` block holds state
and the ``contains`` subroutines do the math.

Source modes
~~~~~~~~~~~~
- ``'soft'`` — Additive soft source (Schneider 2010, Ch. 5; Taflove &
  Hagness 2005, Ch. 5).  Injects Ey and Bz each step via a narrow
  normalised Gaussian profile.
- ``'ic'`` — Initial condition.  Seeds the full pulse waveform directly
  into Ey and Bz at t=t0 (Yee 1966; Munro *et al.* 2014).

Boundary types
~~~~~~~~~~~~~~
- ``'mask'`` — Multiplicative masking absorber (Kosloff & Kosloff 1986).
  Unconditionally stable.  Default.
- ``'cpml'`` — Convolutional PML (Chen & Wang 2013).  Higher absorption
  but may be unstable on anisotropic grids.

Notes
-----
- Not thread-safe when using the module-level ``_state`` dict or the
  ``_default`` propagator (same as Fortran).
- ``calc_j`` callback defaults to a no-op (Jx = Jy = Jz = 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.constants import c as _c0, mu_0 as _mu0
except ImportError:  # pragma: no cover
    _c0: float = 2.99792458e8
    _mu0: float = 1.25663706212e-6

_dp = np.float64
_dc = np.complex128
_ii = 1j
_twopi = 2.0 * np.pi


# ──────────────────────────────────────────────────────────────────────────────
# Standalone spectral update kernels (pure functions — no state)
# ──────────────────────────────────────────────────────────────────────────────


def UpdateE3D(
    space,
    time,
    Bx: NDArray[_dc],
    By: NDArray[_dc],
    Bz: NDArray[_dc],
    Jx: NDArray[_dc],
    Jy: NDArray[_dc],
    Jz: NDArray[_dc],
    Ex: NDArray[_dc],
    Ey: NDArray[_dc],
    Ez: NDArray[_dc],
) -> None:
    r"""Spectral E-field update (Maxwell–Ampere law, in-place).

    .. math::

        \mathbf{E}^{n+1} = \mathbf{E}^n
            + i\,v^2 \Delta t\,(\mathbf{k} \times \mathbf{B})
            - \mu_0\,v^2 \Delta t\,\mathbf{J}

    Parameters
    ----------
    space : ss
        Spatial grid structure.
    time : ts
        Time grid structure.
    Bx, By, Bz : ndarray (Nx, Ny, Nz), complex128
        Magnetic field components in k-space.
    Jx, Jy, Jz : ndarray (Nx, Ny, Nz), complex128
        Current density components in k-space.
    Ex, Ey, Ez : ndarray (Nx, Ny, Nz), complex128
        Electric field components in k-space.  Modified in-place.
    """
    from .typespace import GetEpsr, GetKxArray, GetKyArray, GetKzArray
    from .typetime import GetDt

    v2 = _c0**2 / GetEpsr(space)
    dt = GetDt(time)

    qx = GetKxArray(space)[:, np.newaxis, np.newaxis]
    qy = GetKyArray(space)[np.newaxis, :, np.newaxis]
    qz = GetKzArray(space)[np.newaxis, np.newaxis, :]

    coeff = v2 * dt

    # Ex += i*v2*dt*(ky*Bz - kz*By) - mu0*v2*dt*Jx
    Ex += _ii * (qy * Bz - qz * By) * coeff - _mu0 * Jx * coeff

    # Ey += i*v2*dt*(kz*Bx - kx*Bz) - mu0*v2*dt*Jy
    Ey += _ii * (qz * Bx - qx * Bz) * coeff - _mu0 * Jy * coeff

    # Ez += i*v2*dt*(kx*By - ky*Bx) - mu0*v2*dt*Jz
    Ez += _ii * (qx * By - qy * Bx) * coeff - _mu0 * Jz * coeff


def UpdateB3D(
    space,
    time,
    Ex: NDArray[_dc],
    Ey: NDArray[_dc],
    Ez: NDArray[_dc],
    Bx: NDArray[_dc],
    By: NDArray[_dc],
    Bz: NDArray[_dc],
) -> None:
    r"""Spectral B-field update (Faraday's law, in-place).

    .. math::

        \mathbf{B}^{n+1} = \mathbf{B}^n
            - i\,\Delta t\,(\mathbf{k} \times \mathbf{E})

    Parameters
    ----------
    space : ss
        Spatial grid structure.
    time : ts
        Time grid structure.
    Ex, Ey, Ez : ndarray (Nx, Ny, Nz), complex128
        Electric field components in k-space.
    Bx, By, Bz : ndarray (Nx, Ny, Nz), complex128
        Magnetic field components in k-space.  Modified in-place.
    """
    from .typespace import GetKxArray, GetKyArray, GetKzArray
    from .typetime import GetDt

    dt = GetDt(time)

    qx = GetKxArray(space)[:, np.newaxis, np.newaxis]
    qy = GetKyArray(space)[np.newaxis, :, np.newaxis]
    qz = GetKzArray(space)[np.newaxis, np.newaxis, :]

    Bx -= _ii * (qy * Ez - qz * Ey) * dt
    By -= _ii * (qz * Ex - qx * Ez) * dt
    Bz -= _ii * (qx * Ey - qy * Ex) * dt


def InitializeFields(
    Nx: int, Ny: int, Nz: int
) -> tuple[
    NDArray[_dc], NDArray[_dc], NDArray[_dc],
    NDArray[_dc], NDArray[_dc], NDArray[_dc],
    NDArray[_dc], NDArray[_dc], NDArray[_dc],
]:
    """Allocate and zero-fill all nine field component arrays.

    Returns
    -------
    Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz : ndarray (Nx, Ny, Nz), complex128
    """
    shape = (Nx, Ny, Nz)
    return tuple(np.zeros(shape, dtype=_dc) for _ in range(9))


def SeedInitialCondition(
    space, time, pulse,
    Ey: NDArray[_dc],
    Bz: NDArray[_dc],
    npml_x: int = 0,
) -> None:
    r"""Seed Ey and Bz with the full pulse at t=t0 (IC mode).

    A +x propagating plane wave uses a **complex carrier**
    :math:`\exp(+i\,k\,\xi)` to populate only positive-k Fourier modes,
    producing unidirectional +x propagation.  Using :math:`\cos(k\xi)`
    would create both +k and -k components (standing wave).

    .. math::

        E_y(x) = E_0\,G(y,z)\,
            \exp\!\left(-\xi^2 / \sigma_x^2\right)\,
            \exp\!\left[+i\,(k_{\text{med}}\,\xi
                         + C_{\text{spatial}}\,\xi^2)\right]

        B_z(x) = E_y(x) / v

    where :math:`\xi = x - x_{\text{center}}`,
    :math:`\sigma_x = v\,\tau_G` is the spatial Gaussian half-width,
    :math:`k_{\text{med}} = \omega_0 / v` is the carrier wavenumber in
    the medium, and :math:`C_{\text{spatial}} = C / v^2` converts temporal
    chirp to spatial chirp.

    The physical field is :math:`\operatorname{Re}(E_y)`.  The imaginary
    part is the Hilbert transform (analytic signal representation).

    The impedance relation :math:`B_z = E_y / v` ensures unidirectional
    +x propagation (Taflove & Hagness 2005, Sec. 5.2.1).

    After seeding in real space, both fields are FFT'd to k-space for the
    spectral time-stepper.

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
    x_center = x[Nx // 2]  # center of grid

    # Vectorised seeding over (Nx, Ny, Nz)
    xi = x - x_center  # (Nx,)
    xi_3 = xi[:, np.newaxis, np.newaxis]
    yy_3 = yy[np.newaxis, :, np.newaxis]
    zz_3 = zz[np.newaxis, np.newaxis, :]

    gauss_yz = np.exp(-(yy_3**2 + zz_3**2) / w0**2)

    # Complex carrier exp(+ikx) for unidirectional +x propagation.
    # cos(kx) would create BOTH +k and -k modes (standing wave).
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


# ──────────────────────────────────────────────────────────────────────────────
# Null CalcJ callback (default when no SBE coupling)
# ──────────────────────────────────────────────────────────────────────────────


def _calc_j_noop(space, time, Ex, Ey, Ez, Jx, Jy, Jz) -> None:  # noqa: N802
    """No-op current density source (Jx = Jy = Jz = 0)."""


# ──────────────────────────────────────────────────────────────────────────────
# PSTD3DPropagator
# ──────────────────────────────────────────────────────────────────────────────


class PSTD3DPropagator:
    """3-D PSTD Maxwell propagator with absorbing boundaries and source.

    Parameters
    ----------
    space : ss
        Spatial grid structure (``typespace.ss``).
    time : ts
        Time grid structure (``typetime.ts``).
    pulse : ps
        Pulse parameter structure (``typepulse.ps``).
    source_type : ``'ic'`` or ``'soft'``, optional
        Source injection method.  Default ``'ic'``.

        - ``'ic'`` — Seeds the full pulse into Ey/Bz before the time loop.
          Best for pulse propagation studies.
        - ``'soft'`` — Adds field at a source plane each step.  Best for
          CW-like excitation and material response studies.
    boundary_type : ``'mask'``, ``'cpml'``, or ``'none'``, optional
        Absorbing boundary method.  Default ``'mask'``.

        - ``'mask'`` — Masking absorber (unconditionally stable).
        - ``'cpml'`` — CPML (experimental, may be unstable on anisotropic
          grids).
        - ``'none'`` — Periodic FFT boundaries (no absorption).
    calc_j : callable, optional
        Current density callback with signature
        ``calc_j(space, time, Ex, Ey, Ez, Jx, Jy, Jz)``.
        Defaults to a no-op (free-space propagation).
    output_dir : str or Path, optional
        Base output directory.  Default ``"output"``.
    snapshot_interval : int, optional
        Save field snapshots every N steps.  Default 500.
    """

    def __init__(
        self,
        space,
        time,
        pulse,
        *,
        source_type: str = "ic",
        boundary_type: str = "mask",
        calc_j: Callable | None = None,
        output_dir: str | Path = "output",
        snapshot_interval: int = 500,
    ):
        from .typespace import (
            GetDx,
            GetDy,
            GetDz,
            GetEpsr,
            GetKxArray,
            GetKyArray,
            GetKzArray,
            GetNx,
            GetNy,
            GetNz,
        )
        from .typetime import GetDt

        self.space = space
        self.time = time
        self.pulse = pulse
        self.source_type = source_type
        self.boundary_type = boundary_type
        self.calc_j = calc_j if calc_j is not None else _calc_j_noop
        self.snapshot_interval = snapshot_interval

        self._Nx = GetNx(space)
        self._Ny = GetNy(space)
        self._Nz = GetNz(space)

        # Derived constants
        self._v = _c0 / np.sqrt(GetEpsr(space))
        self._v2 = _c0**2 / GetEpsr(space)
        self._dt = GetDt(time)

        # Wavenumber arrays (extracted once, reused every step)
        self._kx = GetKxArray(space)
        self._ky = GetKyArray(space)
        self._kz = GetKzArray(space)

        # Allocate fields
        (
            self.Ex, self.Ey, self.Ez,
            self.Bx, self.By, self.Bz,
            self.Jx, self.Jy, self.Jz,
        ) = InitializeFields(self._Nx, self._Ny, self._Nz)

        # Real-space scratch arrays for diagnostics
        self._Ex_r = np.zeros((self._Nx, self._Ny, self._Nz), dtype=_dc)
        self._Ey_r = np.zeros((self._Nx, self._Ny, self._Nz), dtype=_dc)
        self._Ez_r = np.zeros((self._Nx, self._Ny, self._Nz), dtype=_dc)

        # Initialize absorbing boundary
        if boundary_type == "cpml":
            from .cpml import InitCPML

            InitCPML(
                self._Nx, self._Ny, self._Nz,
                GetDx(space), GetDy(space), GetDz(space),
                self._dt, GetEpsr(space),
            )
        elif boundary_type == "mask":
            from .absorber import InitAbsorber

            InitAbsorber(self._Nx, self._Ny, self._Nz, self._dt)
        else:
            print("  Boundary: none (periodic FFT boundaries)")

        # Warn if PML is thinner than one medium wavelength
        if boundary_type in ("cpml", "mask"):
            from .typepulse import GetLambda

            npml_x = self._get_npml_x()
            lambda_med = GetLambda(pulse) / np.sqrt(GetEpsr(space))
            if npml_x > 0:
                pml_thickness = npml_x * GetDx(space)
                if pml_thickness < lambda_med:
                    print()
                    print(
                        f"*** WARNING: PML x-thickness ({pml_thickness * 1e9:.2f} nm)"
                        f" < lambda_med ({lambda_med * 1e9:.2f} nm)"
                    )
                    print(
                        "*** Absorption will be poor. Use a larger grid"
                        " or increase frac_pml."
                    )
                    print()

        # TFSF source window (soft mode only)
        self._tfsf = None
        if source_type == "soft":
            from .tfsf import InitializeTFSF

            self._tfsf = InitializeTFSF(space, pulse)

        # IC mode: seed pulse into fields
        if source_type == "ic":
            npml_x = self._get_npml_x()
            SeedInitialCondition(
                space, time, pulse, self.Ey, self.Bz, npml_x
            )
            print("  Source type: Initial Condition (IC)")
        else:
            print("  Source type: Additive Soft Source")

        # Output directory (auto-incremented sim001, sim002, ...)
        self.simdir = self._create_simdir(output_dir)

    def _get_npml_x(self) -> int:
        """Get the x-axis PML thickness from the active boundary module."""
        if self.boundary_type == "cpml":
            from .cpml import GetCPMLInfo

            npx, _, _ = GetCPMLInfo()
            return npx
        elif self.boundary_type == "mask":
            from .absorber import GetAbsorberInfo

            npx, _, _ = GetAbsorberInfo()
            return npx
        return 0

    # ── output directory management ──────────────────────────────────────

    @staticmethod
    def _create_simdir(base: str | Path) -> Path:
        """Find the next available simNNN directory and create it."""
        base = Path(base)
        for sim_id in range(1, 10000):
            d = base / f"sim{sim_id:03d}"
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                print(f"Output directory: {d}")
                return d
        raise RuntimeError("Exhausted sim directory slots (sim001–sim9999).")

    # ── main entry point ─────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full time-stepping loop.

        Mirrors Fortran ``PSTD_3D_Propagator(space, time, pulse)``.

        Time-stepping order (leapfrog):

        1. Diagnostics (IFFT, Emax, snapshots)
        2. Spectral E update (``UpdateE3D``)
        3. Absorbing boundary correction for E
        4. Soft-source injection for Ey (soft mode only)
        5. Half-step time advance
        6. Current density callback (``calc_j``)
        7. Spectral B update (``UpdateB3D``)
        8. Absorbing boundary correction for B
        9. Soft-source injection for Bz (soft mode only)
        10. Half-step time advance + counter increment
        """
        from ..core.fftw import ifft_3D
        from .typepulse import GetAmp
        from .typespace import GetDx, GetDy, GetDz
        from .typetime import CalcNt, GetN

        n1 = GetN(self.time)
        Nt = CalcNt(self.time)
        dt = self._dt
        v = self._v
        v2 = self._v2
        Amp = GetAmp(self.pulse)

        # CFL stability check
        dx_min = min(GetDx(self.space), GetDy(self.space), GetDz(self.space))
        dt_max = dx_min / (v * np.sqrt(3.0))
        cfl_number = v * np.sqrt(3.0) * dt / dx_min

        print("=========================================")
        print("PSTD3D Simulation")
        print("=========================================")
        print(f"  Boundary type: {self.boundary_type}")
        print(f"  Source type:   {self.source_type}")
        print(f"  Phase velocity v = {v:.6e} m/s")
        print(f"  v^2 = {v2:.6e}")
        print(f"  Max stable dt  = {dt_max:.6e} s")
        print(f"  Current dt     = {dt:.6e} s")
        print(f"  CFL number     = {cfl_number:.6f}")
        if cfl_number > 0.95:
            print("  WARNING: CFL condition may be violated!")
        else:
            print("  CFL condition satisfied.")
        print("=========================================")

        self._write_summary(Nt, cfl_number, dt_max)
        self._write_grid()

        # Import boundary-specific routines
        if self.boundary_type == "cpml":
            from .cpml import ApplyCPML_B, ApplyCPML_E
        elif self.boundary_type == "mask":
            from .absorber import ApplyAbsorber_B, ApplyAbsorber_E

        # Import source injection (soft mode only)
        if self.source_type == "soft":
            from .tfsf import UpdateTFSC

        # Emax time-series file
        emax_path = self.simdir / "Emax.dat"
        with open(emax_path, "w", encoding="utf-8") as f_emax:
            f_emax.write("# step   Emax (V/m)\n")

            # ── TIME LOOP ────────────────────────────────────────────
            for n in range(n1, Nt + 1):

                # --- Diagnostics (IFFT to real space) ---
                self._Ex_r[...] = self.Ex
                self._Ey_r[...] = self.Ey
                self._Ez_r[...] = self.Ez
                ifft_3D(self._Ex_r)
                ifft_3D(self._Ey_r)
                ifft_3D(self._Ez_r)

                Emax = np.sqrt(
                    np.max(
                        np.abs(self._Ex_r) ** 2
                        + np.abs(self._Ey_r) ** 2
                        + np.abs(self._Ez_r) ** 2
                    )
                )

                # NaN / overflow detection
                if np.isnan(Emax) or Emax > 1.0e20:
                    raise RuntimeError(
                        f"NaN or overflow at step {n}, Emax = {Emax}"
                    )

                print(f"  {n}  {Nt}  {Emax:.6e}")
                f_emax.write(f"{n:8d} {Emax:15.7e}\n")

                # --- Field snapshots ---
                if n % self.snapshot_interval == 0:
                    self._snapshot(self._Ex_r, "Ex", n)
                    self._snapshot(self._Ey_r, "Ey", n)
                    self._snapshot(self._Ez_r, "Ez", n)

                # ── UPDATE E-FIELDS ──────────────────────────────────
                # 1. Spectral curl update
                UpdateE3D(
                    self.space, self.time,
                    self.Bx, self.By, self.Bz,
                    self.Jx, self.Jy, self.Jz,
                    self.Ex, self.Ey, self.Ez,
                )

                # 2. Absorbing boundary correction
                if self.boundary_type == "cpml":
                    ApplyCPML_E(
                        self.Ex, self.Ey, self.Ez,
                        self.Bx, self.By, self.Bz,
                        self._kx, self._ky, self._kz,
                        v2, dt,
                    )
                elif self.boundary_type == "mask":
                    ApplyAbsorber_E(self.Ex, self.Ey, self.Ez)

                # 3. Soft-source injection for Ey (soft mode only)
                if self.source_type == "soft":
                    UpdateTFSC(
                        self.Ey, self._tfsf, self.space,
                        self.time, self.pulse, Amp,
                    )

                # 4. Advance time by half-step
                self.time.UpdateT(dt / 2.0)

                # 5. Current density source
                self.calc_j(
                    self.space, self.time,
                    self.Ex, self.Ey, self.Ez,
                    self.Jx, self.Jy, self.Jz,
                )

                # ── UPDATE B-FIELDS ──────────────────────────────────
                # 6. Spectral curl update
                UpdateB3D(
                    self.space, self.time,
                    self.Ex, self.Ey, self.Ez,
                    self.Bx, self.By, self.Bz,
                )

                # 7. Absorbing boundary correction
                if self.boundary_type == "cpml":
                    ApplyCPML_B(
                        self.Bx, self.By, self.Bz,
                        self.Ex, self.Ey, self.Ez,
                        self._kx, self._ky, self._kz,
                        dt,
                    )
                elif self.boundary_type == "mask":
                    ApplyAbsorber_B(self.Bx, self.By, self.Bz)

                # 8. Soft-source injection for Bz (soft mode only)
                #    Amplitude = E0/v for impedance ratio Ey/Bz = v
                if self.source_type == "soft":
                    UpdateTFSC(
                        self.Bz, self._tfsf, self.space,
                        self.time, self.pulse, Amp / v,
                    )

                # 9. Advance time by half-step, update counter
                self.time.UpdateT(dt / 2.0)
                self.time.n = n + 1

        print(f"Simulation complete. Output in: {self.simdir}")

    # ── I/O helpers ──────────────────────────────────────────────────────

    def _snapshot(self, F: NDArray[_dc], fieldname: str, n: int) -> None:
        """Save a real-space field snapshot as ``.npy`` (float32)."""
        fname = self.simdir / f"{fieldname}_{n:06d}.npy"
        np.save(fname, F.real.astype(np.float32))

    def _write_summary(self, Nt: int, cfl: float, dt_max: float) -> None:
        """Write simulation parameters to ``summary.txt``.

        Mirrors Fortran ``WriteSummary`` with additional diagnostics:
        memory estimate, refractive index, points-per-wavelength,
        k0/k_Nyquist ratio, boundary details, and references.
        """
        from .typepulse import (
            CalcDeltaOmega,
            CalcRayleigh,
            CalcTau,
            GetAmp,
            GetChirp,
            GetLambda,
            GetTp,
            GetTw,
            GetW0,
        )
        from .typespace import (
            GetDx,
            GetDy,
            GetDz,
            GetEpsr,
            GetNx,
            GetNy,
            GetNz,
            GetXWidth,
            GetYWidth,
            GetZWidth,
        )
        from .typetime import GetDt, GetT, GetTf

        space = self.space
        time = self.time
        pulse = self.pulse
        v = self._v
        w0 = GetW0(pulse)
        Nx, Ny, Nz = GetNx(space), GetNy(space), GetNz(space)
        epsr = GetEpsr(space)

        lam_med = GetLambda(pulse) / np.sqrt(epsr)
        k0_med = _twopi / lam_med
        ppw_x = lam_med / GetDx(space)
        n_cycles = GetTw(pulse) / (GetLambda(pulse) / _c0)

        # PML info
        npx = npy = npz = 0
        if self.boundary_type == "cpml":
            from .cpml import GetCPMLInfo
            npx, npy, npz = GetCPMLInfo()
        elif self.boundary_type == "mask":
            from .absorber import GetAbsorberInfo
            npx, npy, npz = GetAbsorberInfo()

        # Memory estimate (MB)
        # 9 complex field arrays + 3 diagnostic copies = 12 x Nx*Ny*Nz x 16 bytes
        n_cells = float(Nx) * Ny * Nz
        mem_fields = 9.0 * n_cells * 16.0 / 1.0e6
        mem_aux = 3.0 * n_cells * 16.0 / 1.0e6
        if self.boundary_type == "cpml":
            mem_aux += 12.0 * n_cells * 8.0 / 1.0e6  # 12 psi fields (float64)
        elif self.boundary_type == "mask":
            mem_aux += n_cells * 8.0 / 1.0e6  # 1 mask array (float64)
        mem_total = mem_fields + mem_aux

        with open(self.simdir / "summary.txt", "w", encoding="utf-8") as u:
            u.write("============================================================\n")
            u.write("PSTD3D Simulation Summary\n")
            u.write("============================================================\n\n")

            u.write("--- Method ---\n")
            u.write("  Solver           = PSTD (Pseudospectral Time-Domain)\n")
            u.write("  Spatial derivs   = FFT (spectrally exact, 2 cells/wavelength)\n")
            u.write("  Time integration = Leapfrog (2nd order, explicit)\n")
            u.write("  FFT library      = pyFFTW / scipy.fft\n")
            u.write(f"  Source type      = {self.source_type}\n")
            u.write(f"  Boundary type    = {self.boundary_type}\n")
            u.write("  Language         = Python (NumPy vectorised)\n")
            u.write(f"  Memory estimate  = {mem_total:.1f} MB\n\n")

            u.write("--- Grid ---\n")
            u.write(f"  Nx x Ny x Nz  = {Nx} x {Ny} x {Nz}\n")
            u.write(f"  Total cells    = {Nx * Ny * Nz}\n")
            u.write(f"  dx             = {GetDx(space):12.4e} m\n")
            u.write(f"  dy             = {GetDy(space):12.4e} m\n")
            u.write(f"  dz             = {GetDz(space):12.4e} m\n")
            u.write(f"  Lx             = {GetXWidth(space):12.4e} m\n")
            u.write(f"  Ly             = {GetYWidth(space):12.4e} m\n")
            u.write(f"  Lz             = {GetZWidth(space):12.4e} m\n")
            u.write(f"  epsr           = {epsr:12.4e}\n")
            u.write(f"  Phase speed v  = {v:12.4e} m/s\n")
            u.write(f"  Refr. index n  = {np.sqrt(epsr):12.4e}\n")
            u.write(f"  lambda_med     = {lam_med:12.4e} m\n")
            u.write(f"  ppw_x          = {ppw_x:8.1f}\n")
            u.write(f"  k0_med         = {k0_med:12.4e} rad/m\n")
            u.write(f"  k_Nyquist_x    = {np.pi / GetDx(space):12.4e} rad/m\n")
            u.write(f"  k0/k_Nyquist   = {k0_med / (np.pi / GetDx(space)):8.4f}\n\n")

            u.write("--- Time ---\n")
            u.write(f"  Nt             = {Nt}\n")
            u.write(f"  t0             = {GetT(time):12.4e} s\n")
            u.write(f"  dt             = {GetDt(time):12.4e} s\n")
            u.write(f"  tf             = {GetTf(time):12.4e} s\n")
            u.write(f"  Duration       = {GetTf(time) - GetT(time):12.4e} s\n")
            u.write(f"  CFL number     = {cfl:8.4f}\n")
            u.write(f"  dt_stable_max  = {dt_max:12.4e} s\n")
            u.write(f"  Safety margin  = {(1.0 - cfl) * 100.0:.1f}%\n")
            if cfl > 0.95:
                u.write("  *** WARNING: CFL may be violated! ***\n")
            u.write("\n")

            u.write("--- Pulse ---\n")
            u.write(f"  lambda         = {GetLambda(pulse):12.4e} m\n")
            u.write(f"  Amplitude      = {GetAmp(pulse):12.4e} V/m\n")
            u.write(f"  Pulsewidth Tw  = {GetTw(pulse):12.4e} s  (intensity FWHM)\n")
            u.write(f"  tau_G          = {CalcTau(pulse):12.4e} s  (1/e half-width)\n")
            u.write(f"  Peak time Tp   = {GetTp(pulse):12.4e} s\n")
            u.write(f"  Chirp          = {GetChirp(pulse):12.4e} rad/s^2\n")
            omega0 = pulse.CalcOmega0()
            u.write(f"  omega0         = {omega0:12.4e} rad/s\n")
            u.write(f"  Delta_omega    = {CalcDeltaOmega(pulse):12.4e} rad/s  (spectral FWHM)\n")
            u.write(f"  w0 (beam waist)= {w0:12.4e} m\n")
            u.write(f"  Optical cycles = {n_cycles:.1f}\n")
            u.write(f"  T_optical      = {GetLambda(pulse) / _c0:12.4e} s\n")
            if w0 < float("inf"):
                u.write(f"  Rayleigh range = {CalcRayleigh(pulse, w0):12.4e} m\n")
            else:
                u.write("  Rayleigh range = Inf (plane-wave mode)\n")
            u.write("\n")

            u.write("--- Boundary Conditions ---\n")
            u.write(f"  Type           = {self.boundary_type}\n")
            u.write(f"  PML cells      = {npx} x {npy} x {npz}\n")
            u.write(
                f"  Interior cells = {Nx - 2 * npx} x {Ny - 2 * npy}"
                f" x {Nz - 2 * npz}\n"
            )
            if self.boundary_type == "mask":
                u.write("  Algorithm      = Multiplicative masking (Kosloff & Kosloff 1986)\n")
                u.write("  Profile        = Polynomial grading, exp(-gamma*dt) per step\n")
                u.write("  Stability      = Unconditionally stable (|mask| <= 1)\n")
            elif self.boundary_type == "cpml":
                u.write("  Algorithm      = CPML (Chen & Wang 2013, embedded formulation)\n")
                u.write("  Profile        = CFS polynomial grading (Roden & Gedney 2000)\n")
                u.write("  Stability      = Conditional (see Gedney Ch. 7)\n")
            else:
                u.write("  Algorithm      = None (periodic FFT boundaries)\n")
                u.write("  Note           = Pulse wraps around via FFT periodicity\n")
            u.write("\n")

            u.write("--- Source Injection ---\n")
            u.write(f"  Type           = {self.source_type}\n")
            if self.source_type == "ic":
                u.write("  Method         = Initial condition: full E_y(x) + B_z(x) seeded at t=t0\n")
                u.write("  Spatial profile = Gaussian envelope * exp(+ik*x) carrier\n")
                u.write("  B_z coupling   = B_z = E_y / v  (forward-propagating +x wave)\n")
                u.write("  Reference      = Taflove & Hagness 2005, Ch. 5\n")
            else:
                u.write("  Method         = Soft source: additive field injection each step\n")
                u.write("  TFSF window    = Normalised Gaussian (sigma = 5*dx)\n")
                u.write("  Reference      = Schneider 2010, Ch. 5; Capoglu et al. 2012\n")
            u.write("\n")

            u.write("--- Output Format ---\n")
            u.write("  Emax.dat            ASCII, columns: step  Emax (V/m)\n")
            u.write("  grid_{x,y,z}.npy    Coordinate arrays (numpy binary)\n")
            u.write(f"  [F]_NNNNNN.npy      Field snapshots every {self.snapshot_interval} steps\n")
            u.write("    Shape: (Nx, Ny, Nz), dtype float32 (real part)\n\n")

            u.write("--- Reading .npy snapshots in Python ---\n")
            u.write("  import numpy as np\n")
            u.write('  data = np.load("Ey_000500.npy")  # shape (Nx, Ny, Nz), float32\n')
            u.write("  # Mid-plane slice (y-normal):\n")
            u.write("  # slice_xz = data[:, Ny//2, :]\n\n")

            u.write("--- References ---\n")
            u.write('  [1] Liu QH (1997) "The PSTD algorithm." Microw Opt Technol Lett 15:158\n')
            u.write('  [2] Berenger JP (1994) "A perfectly matched layer." J Comput Phys 114:185\n')
            u.write('  [3] Chen J, Wang J (2013) "CPML for PSTD." ACES J 28(8):680\n')
            u.write('  [4] Gedney SD (2005) "PML ABC." Ch.7 in Taflove & Hagness\n')
            u.write('  [5] Kosloff R, Kosloff D (1986) "Absorbing boundaries." J Comput Phys 63:363\n')
            u.write('  [6] Roden JA, Gedney SD (2000) "CPML." Microw Opt Technol Lett 27:334\n')

    def _write_grid(self) -> None:
        """Save spatial coordinate arrays as ``.npy`` files."""
        from .typespace import GetXArray, GetYArray, GetZArray

        np.save(self.simdir / "grid_x.npy", GetXArray(self.space))
        np.save(self.simdir / "grid_y.npy", GetYArray(self.space))
        np.save(self.simdir / "grid_z.npy", GetZArray(self.space))


# ──────────────────────────────────────────────────────────────────────────────
# Module-level Fortran API wrapper
# ──────────────────────────────────────────────────────────────────────────────

_default: PSTD3DPropagator | None = None  # NOT thread-safe


def PSTD_3D_Propagator(
    space,
    time,
    pulse,
    *,
    source_type: str = "ic",
    boundary_type: str = "mask",
    calc_j: Callable | None = None,
    output_dir: str | Path = "output",
    snapshot_interval: int = 500,
) -> None:
    """Run a full PSTD3D simulation.

    Fortran signature parity: ``PSTD_3D_Propagator(space, time, pulse)``.

    Parameters
    ----------
    space : ss
        Spatial grid structure.
    time : ts
        Time grid structure.
    pulse : ps
        Pulse parameter structure.
    source_type : ``'ic'`` or ``'soft'``, optional
        Source injection method.  Default ``'ic'``.
    boundary_type : ``'mask'``, ``'cpml'``, or ``'none'``, optional
        Absorbing boundary method.  Default ``'mask'``.
    calc_j : callable, optional
        Current density callback.  Defaults to no-op.
    output_dir : str or Path, optional
        Base output directory.
    snapshot_interval : int, optional
        Field snapshot interval in time steps.
    """
    global _default
    _default = PSTD3DPropagator(
        space, time, pulse,
        source_type=source_type,
        boundary_type=boundary_type,
        calc_j=calc_j,
        output_dir=output_dir,
        snapshot_interval=snapshot_interval,
    )
    _default.run()
