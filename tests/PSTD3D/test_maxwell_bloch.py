"""
Maxwell-Bloch coupling end-to-end tests.

Tests the coupling between the PSTD3D Maxwell solver and external current
density sources via the calc_j callback mechanism.
"""

import tempfile

import numpy as np
import pytest
from scipy.constants import c as c0, epsilon_0, mu_0

from pulsesuite.core.fftw import fft_3D, ifft_3D
from pulsesuite.PSTD3D.PSTD3D import PSTD3DPropagator
from pulsesuite.PSTD3D.typepulse import ps
from pulsesuite.PSTD3D.typespace import ss
from pulsesuite.PSTD3D.typetime import CalculateOptimalDt, ts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_space(Nx=128, Ny=1, Nz=1, dx=25e-9, epsr=1.0):
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


def _make_pulse(lambda_=800e-9, Amp=1e8, Tw=8e-15):
    return ps(
        lambda_=lambda_,
        Amp=Amp,
        Tw=Tw,
        Tp=0.0,
        chirp=0.0,
        pol=0,
        w0=float("inf"),
    )


def _make_time(space, Nsteps=100, safety=0.5):
    dt = CalculateOptimalDt(space.dx, space.dy, space.dz, space.epsr, safety=safety)
    return ts(t=0.0, tf=Nsteps * dt, dt=dt, n=0)


def _get_real_field(F_k):
    tmp = F_k.copy()
    ifft_3D(tmp)
    return tmp.real


def _total_em_energy(prop, dx):
    """Total EM energy from propagator fields."""
    Ey_r = _get_real_field(prop.Ey)
    Bz_r = _get_real_field(prop.Bz)
    energy_E = 0.5 * epsilon_0 * np.sum(Ey_r**2) * dx
    energy_B = 0.5 / mu_0 * np.sum(Bz_r**2) * dx
    return energy_E + energy_B


def _make_oscillating_calc_j(J0, omega):
    """Return calc_j that adds uniform oscillating Jy = J0*sin(omega*t)."""

    def calc_j(space, time, Ex, Ey, Ez, Jx, Jy, Jz):
        Nx, Ny, Nz = Jy.shape
        J_real = np.full((Nx, Ny, Nz), J0 * np.sin(omega * time.t), dtype=np.complex128)
        fft_3D(J_real)
        Jy += J_real

    return calc_j


def _make_absorbing_calc_j(sigma):
    """Return calc_j that does Ohmic absorption: J = sigma * E."""

    def calc_j(space, time, Ex, Ey, Ez, Jx, Jy, Jz):
        tmp = Ey.copy()
        ifft_3D(tmp)
        J_real = sigma * tmp
        fft_3D(J_real)
        Jy += J_real

    return calc_j


def _make_recording_absorbing_calc_j(sigma):
    """Return (calc_j, j_maxes) that does J=sigma*E and records max |Jy|."""
    j_maxes = []

    def calc_j(space, time, Ex, Ey, Ez, Jx, Jy, Jz):
        tmp = Ey.copy()
        ifft_3D(tmp)
        J_real = sigma * tmp
        j_maxes.append(np.max(np.abs(J_real.real)))
        fft_3D(J_real)
        Jy += J_real

    return calc_j, j_maxes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_current_source_modifies_fields():
    """An oscillating current Jy should modify Ey via Ampere's law."""
    space = _make_space(Nx=128, dx=25e-9)
    pulse = _make_pulse()
    Nsteps = 100
    omega_drive = 2.0 * np.pi * c0 / pulse.lambda_

    # Run WITHOUT current source
    time_no_j = _make_time(space, Nsteps=Nsteps)
    with tempfile.TemporaryDirectory() as tmpdir:
        prop_no_j = PSTD3DPropagator(
            space,
            time_no_j,
            pulse,
            source_type="ic",
            boundary_type="mask",
            output_dir=tmpdir,
            snapshot_interval=99999,
        )
        prop_no_j.run()
        Ey_no_j = _get_real_field(prop_no_j.Ey)

    # Run WITH oscillating current source
    calc_j = _make_oscillating_calc_j(J0=1e10, omega=omega_drive)
    time_with_j = _make_time(_make_space(Nx=128, dx=25e-9), Nsteps=Nsteps)
    with tempfile.TemporaryDirectory() as tmpdir:
        prop_with_j = PSTD3DPropagator(
            _make_space(Nx=128, dx=25e-9),
            time_with_j,
            pulse,
            source_type="ic",
            boundary_type="mask",
            calc_j=calc_j,
            output_dir=tmpdir,
            snapshot_interval=99999,
        )
        prop_with_j.run()
        Ey_with_j = _get_real_field(prop_with_j.Ey)

    # Fields should differ — current source injects energy
    diff = np.max(np.abs(Ey_with_j - Ey_no_j))
    assert diff > 0, "Current source had no effect on E-field"

    # Both should remain finite
    assert np.all(np.isfinite(Ey_no_j))
    assert np.all(np.isfinite(Ey_with_j))


@pytest.mark.slow
@pytest.mark.integration
def test_ohmic_absorption_reduces_energy():
    """Ohmic current J=sigma*E should extract energy from the field."""
    dx = 25e-9
    Nsteps = 100
    sigma = 1e3  # moderate conductivity

    space = _make_space(Nx=128, dx=dx)
    pulse = _make_pulse()

    # Run WITHOUT absorption
    time_ref = _make_time(space, Nsteps=Nsteps)
    with tempfile.TemporaryDirectory() as tmpdir:
        prop_ref = PSTD3DPropagator(
            space,
            time_ref,
            pulse,
            source_type="ic",
            boundary_type="mask",
            output_dir=tmpdir,
            snapshot_interval=99999,
        )
        U_init = _total_em_energy(prop_ref, dx)
        prop_ref.run()
        U_ref = _total_em_energy(prop_ref, dx)

    # Run WITH Ohmic absorption
    calc_j = _make_absorbing_calc_j(sigma)
    time_abs = _make_time(_make_space(Nx=128, dx=dx), Nsteps=Nsteps)
    with tempfile.TemporaryDirectory() as tmpdir:
        prop_abs = PSTD3DPropagator(
            _make_space(Nx=128, dx=dx),
            time_abs,
            pulse,
            source_type="ic",
            boundary_type="mask",
            calc_j=calc_j,
            output_dir=tmpdir,
            snapshot_interval=99999,
        )
        prop_abs.run()
        U_abs = _total_em_energy(prop_abs, dx)

    # Absorption should reduce energy more than reference (which only has boundary loss)
    assert U_abs < U_ref, (
        f"Ohmic absorption did not reduce energy: U_abs={U_abs:.4e}, U_ref={U_ref:.4e}"
    )

    # Should be a measurable difference
    assert U_abs < 0.95 * U_ref, (
        f"Absorption too weak: U_abs/U_ref = {U_abs / U_ref:.4f}"
    )

    # Fields should remain finite
    assert np.all(np.isfinite(prop_abs.Ey))


@pytest.mark.slow
@pytest.mark.integration
def test_linear_response_proportionality():
    """For J=sigma*E, doubling E should double the peak current."""
    dx = 25e-9
    Nsteps = 50
    sigma = 1e2

    Amp_1 = 1e6  # weak field
    Amp_2 = 2e6  # doubled

    # Run with Amp_1
    calc_j_1, j_maxes_1 = _make_recording_absorbing_calc_j(sigma)
    space_1 = _make_space(Nx=128, dx=dx)
    time_1 = _make_time(space_1, Nsteps=Nsteps)
    pulse_1 = _make_pulse(Amp=Amp_1)
    with tempfile.TemporaryDirectory() as tmpdir:
        prop_1 = PSTD3DPropagator(
            space_1,
            time_1,
            pulse_1,
            source_type="ic",
            boundary_type="mask",
            calc_j=calc_j_1,
            output_dir=tmpdir,
            snapshot_interval=99999,
        )
        prop_1.run()

    # Run with Amp_2
    calc_j_2, j_maxes_2 = _make_recording_absorbing_calc_j(sigma)
    space_2 = _make_space(Nx=128, dx=dx)
    time_2 = _make_time(space_2, Nsteps=Nsteps)
    pulse_2 = _make_pulse(Amp=Amp_2)
    with tempfile.TemporaryDirectory() as tmpdir:
        prop_2 = PSTD3DPropagator(
            space_2,
            time_2,
            pulse_2,
            source_type="ic",
            boundary_type="mask",
            calc_j=calc_j_2,
            output_dir=tmpdir,
            snapshot_interval=99999,
        )
        prop_2.run()

    # Peak current should scale linearly with field amplitude
    peak_j_1 = max(j_maxes_1) if j_maxes_1 else 0
    peak_j_2 = max(j_maxes_2) if j_maxes_2 else 0

    if peak_j_1 == 0:
        pytest.skip("No current recorded for Amp_1")

    ratio = peak_j_2 / peak_j_1
    expected_ratio = Amp_2 / Amp_1  # should be 2.0

    assert abs(ratio - expected_ratio) / expected_ratio < 0.10, (
        f"Linear response violated: J ratio = {ratio:.3f}, "
        f"expected {expected_ratio:.1f} (10% tolerance)"
    )
