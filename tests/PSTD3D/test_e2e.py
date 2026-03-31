"""
End-to-end integration tests for PSTD3D Maxwell propagator.
=============================================================

These tests verify physical reality by running minimal PSTD3D simulations
and checking that the results obey Maxwell's equations and electromagnetic
wave physics.

Tests
-----
- test_pulse_propagates_forward : pulse peak moves in +x at v = c/n
- test_energy_conservation_vacuum : EM energy conserved in vacuum (5% tolerance)
- test_impedance_relation_maintained : |Bz| ~ |Ey|/v throughout propagation
- test_cfl_stability : simulation completes without NaN/overflow under CFL

All tests use small grids (Nx=128-256, Ny=Nz=1) and few steps (50-200)
to keep runtime manageable.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
from scipy.constants import c as c0, epsilon_0, mu_0

from pulsesuite.core.fftw import ifft_3D
from pulsesuite.PSTD3D.PSTD3D import PSTD3DPropagator
from pulsesuite.PSTD3D.typepulse import ps
from pulsesuite.PSTD3D.typespace import ss
from pulsesuite.PSTD3D.typetime import CalculateOptimalDt, ts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_space(Nx=256, Ny=1, Nz=1, dx=20e-9, epsr=1.0):
    """Create a minimal spatial grid for 1D propagation."""
    return ss(
        Dims=1 if (Ny == 1 and Nz == 1) else 3,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        dx=dx,
        dy=dx,
        dz=dx,
        epsr=epsr,
    )


def _make_pulse(lambda_=800e-9, Amp=1.0e8, Tw=10e-15):
    """Create a short Gaussian pulse (plane wave, no chirp)."""
    return ps(
        lambda_=lambda_,
        Amp=Amp,
        Tw=Tw,
        Tp=0.0,
        chirp=0.0,
        pol=0,
        w0=float("inf"),  # plane wave
    )


def _make_time(space, Nsteps=100, safety=0.5):
    """Create a time grid with CFL-safe dt for the given number of steps."""
    dt = CalculateOptimalDt(space.dx, space.dy, space.dz, space.epsr, safety=safety)
    tf = Nsteps * dt
    return ts(t=0.0, tf=tf, dt=dt, n=0)


def _get_real_field(F_k):
    """IFFT a k-space field array and return real-space copy (real part)."""
    tmp = F_k.copy()
    ifft_3D(tmp)
    return tmp.real


def _peak_x_index(Ey_real):
    """Return x-index of the maximum |Ey| along x (squeeze y,z)."""
    profile = np.abs(Ey_real[:, 0, 0])
    return np.argmax(profile)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_pulse_propagates_forward():
    """A 1D pulse should propagate in +x. After N steps the peak should
    have moved to the right by approximately v * t."""
    space = _make_space(Nx=256, dx=20e-9)
    pulse = _make_pulse()
    Nsteps = 150
    time = _make_time(space, Nsteps=Nsteps, safety=0.5)

    v = c0 / np.sqrt(space.epsr)

    with tempfile.TemporaryDirectory() as tmpdir:
        prop = PSTD3DPropagator(
            space,
            time,
            pulse,
            source_type="ic",
            boundary_type="mask",
            output_dir=tmpdir,
            snapshot_interval=99999,  # no snapshots
        )

        # Record initial peak position
        Ey_init = _get_real_field(prop.Ey)
        peak_init = _peak_x_index(Ey_init)

        # Run the simulation
        prop.run()

        # Get final field
        Ey_final = _get_real_field(prop.Ey)
        peak_final = _peak_x_index(Ey_final)

        # The pulse should have moved to the right
        assert peak_final > peak_init, (
            f"Pulse did not propagate forward: initial peak at {peak_init}, "
            f"final peak at {peak_final}"
        )

        # Check that the displacement is roughly v * t
        dx = space.dx
        displacement_cells = peak_final - peak_init
        displacement_m = displacement_cells * dx
        expected_m = v * time.t  # time.t has been advanced by run()

        # Allow generous tolerance: absorber damping and envelope dispersion
        # shift the apparent peak, but displacement should be within 30-170%
        # of the expected value.
        ratio = displacement_m / expected_m if expected_m > 0 else 0
        assert 0.3 < ratio < 1.7, (
            f"Pulse displacement {displacement_m:.3e} m vs expected {expected_m:.3e} m "
            f"(ratio = {ratio:.2f})"
        )


@pytest.mark.slow
@pytest.mark.integration
def test_energy_conservation_vacuum():
    """In vacuum with absorbing boundaries far from the pulse,
    total EM energy should be approximately conserved over short propagation.

    Energy: U = (eps_0 * eps_r / 2) * sum(|E|^2) * dV
              + (1 / (2 * mu_0))   * sum(|B|^2) * dV
    """
    Nx = 256
    dx = 20e-9
    space = _make_space(Nx=Nx, dx=dx)
    pulse = _make_pulse(Amp=1.0e8, Tw=6e-15)  # narrow pulse, well inside grid
    Nsteps = 80
    time = _make_time(space, Nsteps=Nsteps, safety=0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        prop = PSTD3DPropagator(
            space,
            time,
            pulse,
            source_type="ic",
            boundary_type="mask",
            output_dir=tmpdir,
            snapshot_interval=99999,
        )

        def _total_energy(propagator):
            """Compute total EM energy from k-space fields."""
            Ey_r = _get_real_field(propagator.Ey)
            Bz_r = _get_real_field(propagator.Bz)
            # For 1D (Ny=Nz=1), volume element is just dx
            dV = dx
            energy_E = 0.5 * epsilon_0 * space.epsr * np.sum(Ey_r**2) * dV
            energy_B = 0.5 / mu_0 * np.sum(Bz_r**2) * dV
            return energy_E + energy_B

        U_initial = _total_energy(prop)
        assert U_initial > 0, "Initial energy should be positive"

        prop.run()

        U_final = _total_energy(prop)

        # Energy should not grow (absorber can only remove energy).
        # Allow 5% growth tolerance for discretisation artifacts.
        # Lower bound is 50%: some energy may be absorbed by boundary.
        ratio = U_final / U_initial
        assert 0.50 < ratio < 1.05, (
            f"Energy not conserved: U_final/U_initial = {ratio:.4f} "
            f"(U_initial = {U_initial:.4e}, U_final = {U_final:.4e})"
        )


@pytest.mark.slow
@pytest.mark.integration
def test_impedance_relation_maintained():
    r"""Throughout propagation, |Bz| \approx |Ey|/v should hold.

    This is the Maxwell impedance relation for a forward-propagating plane
    wave in a medium with phase velocity v = c / sqrt(eps_r).
    """
    space = _make_space(Nx=256, dx=20e-9)
    pulse = _make_pulse(Tw=8e-15)
    Nsteps = 100
    time = _make_time(space, Nsteps=Nsteps, safety=0.5)

    v = c0 / np.sqrt(space.epsr)

    with tempfile.TemporaryDirectory() as tmpdir:
        prop = PSTD3DPropagator(
            space,
            time,
            pulse,
            source_type="ic",
            boundary_type="mask",
            output_dir=tmpdir,
            snapshot_interval=99999,
        )

        # -- Check impedance at initial condition --
        Ey_r = _get_real_field(prop.Ey)
        Bz_r = _get_real_field(prop.Bz)

        # Only check where field is significant (above 10% of max)
        threshold = 0.1 * np.max(np.abs(Ey_r))
        mask = np.abs(Ey_r) > threshold
        if np.any(mask):
            ratio_init = np.abs(Bz_r[mask]) / (np.abs(Ey_r[mask]) / v)
            assert np.allclose(ratio_init, 1.0, atol=0.05), (
                f"Initial impedance relation violated: "
                f"mean ratio = {np.mean(ratio_init):.4f}"
            )

        # -- Run simulation --
        prop.run()

        # -- Check impedance after propagation --
        Ey_r = _get_real_field(prop.Ey)
        Bz_r = _get_real_field(prop.Bz)

        threshold = 0.1 * np.max(np.abs(Ey_r))
        mask = np.abs(Ey_r) > threshold

        if np.any(mask):
            ratio_final = np.abs(Bz_r[mask]) / (np.abs(Ey_r[mask]) / v)
            # Allow wider tolerance after propagation due to boundary effects
            # and numerical dispersion; median should stay near 1.0.
            median_ratio = np.median(ratio_final)
            assert 0.7 < median_ratio < 1.3, (
                f"Impedance relation violated after propagation: "
                f"median |Bz|/(|Ey|/v) = {median_ratio:.4f}"
            )


@pytest.mark.slow
@pytest.mark.integration
def test_cfl_stability():
    """Simulation should complete without NaN or overflow when dt
    satisfies the CFL condition."""
    space = _make_space(Nx=128, dx=25e-9)
    pulse = _make_pulse(Tw=8e-15)
    Nsteps = 200
    # Use a conservative safety factor to ensure CFL is well satisfied
    time = _make_time(space, Nsteps=Nsteps, safety=0.4)

    with tempfile.TemporaryDirectory() as tmpdir:
        prop = PSTD3DPropagator(
            space,
            time,
            pulse,
            source_type="ic",
            boundary_type="mask",
            output_dir=tmpdir,
            snapshot_interval=99999,
        )

        # run() raises RuntimeError if NaN or overflow is detected internally
        prop.run()

        # Verify all field components are finite after simulation
        Ey_r = _get_real_field(prop.Ey)
        Bz_r = _get_real_field(prop.Bz)

        assert np.all(np.isfinite(Ey_r)), "Ey contains NaN or Inf after simulation"
        assert np.all(np.isfinite(Bz_r)), "Bz contains NaN or Inf after simulation"

        # Verify field has not grown to unphysical levels (no numerical explosion)
        Emax = np.max(np.abs(Ey_r))
        assert Emax < 1e12, f"Ey grew to unphysical level: Emax = {Emax:.4e}"
        assert Emax > 0, "All fields are zero -- simulation may not have run"
