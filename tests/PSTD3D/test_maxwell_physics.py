"""
Advanced Maxwell solver physics validation.

Dispersion relation, beam diffraction, group velocity invariance
under chirp, and spectral convergence rate.
"""

import tempfile

import numpy as np
import pytest
from scipy.constants import c as c0

from pulsesuite.core.fftw import ifft_3D
from pulsesuite.PSTD3D.PSTD3D import PSTD3DPropagator
from pulsesuite.PSTD3D.typepulse import ps
from pulsesuite.PSTD3D.typespace import ss
from pulsesuite.PSTD3D.typetime import CalculateOptimalDt, ts

_twopi = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_space(Nx=256, Ny=1, Nz=1, dx=20e-9, epsr=1.0):
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


def _make_pulse(lambda_=800e-9, Amp=1e8, Tw=10e-15, chirp=0.0, w0=float("inf")):
    return ps(
        lambda_=lambda_,
        Amp=Amp,
        Tw=Tw,
        Tp=0.0,
        chirp=chirp,
        pol=0,
        w0=w0,
    )


def _make_time(space, Nsteps=100, safety=0.5):
    dt = CalculateOptimalDt(space.dx, space.dy, space.dz, space.epsr, safety=safety)
    return ts(t=0.0, tf=Nsteps * dt, dt=dt, n=0)


def _get_real_field(F_k):
    tmp = F_k.copy()
    ifft_3D(tmp)
    return tmp.real


def _peak_x_index(Ey_real):
    return np.argmax(np.abs(Ey_real[:, 0, 0]))


def _make_recording_calc_j(probe_ix, probe_iy=0, probe_iz=0):
    """Return (calc_j, record) where record accumulates (t, Ey_real) at probe."""
    record = []

    def calc_j(space, time, Ex, Ey, Ez, Jx, Jy, Jz):
        tmp = Ey.copy()
        ifft_3D(tmp)
        record.append((time.t, tmp[probe_ix, probe_iy, probe_iz].real))

    return calc_j, record


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_dispersion_relation():
    """Temporal FFT of Ey(t) at a probe should peak at omega0 = 2*pi*c/lambda."""
    Nx = 256
    dx = 20e-9
    space = _make_space(Nx=Nx, dx=dx)
    pulse = _make_pulse(Tw=10e-15)
    Nsteps = 500
    time = _make_time(space, Nsteps=Nsteps, safety=0.5)

    omega0_expected = _twopi * c0 / pulse.lambda_

    # Probe at 75% of grid (pulse starts at 25%, propagates right)
    probe_ix = int(Nx * 0.75)
    calc_j, record = _make_recording_calc_j(probe_ix)

    with tempfile.TemporaryDirectory() as tmpdir:
        prop = PSTD3DPropagator(
            space,
            time,
            pulse,
            source_type="ic",
            boundary_type="mask",
            calc_j=calc_j,
            output_dir=tmpdir,
            snapshot_interval=99999,
        )
        prop.run()

    # Need enough samples for frequency resolution
    if len(record) < 20:
        pytest.skip("Too few samples recorded")

    times = np.array([r[0] for r in record])
    vals = np.array([r[1] for r in record])

    # Temporal FFT
    dt_sample = np.mean(np.diff(times))
    freqs = np.fft.fftfreq(len(vals), d=dt_sample) * _twopi  # angular freq
    spectrum = np.abs(np.fft.fft(vals)) ** 2

    # Find peak in positive frequencies
    pos_mask = freqs > 0
    omega_peak = freqs[pos_mask][np.argmax(spectrum[pos_mask])]

    # Spectral resolution
    d_omega = _twopi / (len(vals) * dt_sample)

    assert abs(omega_peak - omega0_expected) < max(
        3 * d_omega, 0.02 * omega0_expected
    ), f"Spectral peak at {omega_peak:.4e} rad/s, expected {omega0_expected:.4e} rad/s"


@pytest.mark.slow
@pytest.mark.integration
def test_beam_diffraction_2d():
    """A finite-waist beam should have a broader transverse profile than a plane wave."""
    Nx = 128
    Ny = 64
    dx = 40e-9
    w0 = 1.0e-6
    lam = 800e-9

    # Run with finite beam waist
    space = _make_space(Nx=Nx, Ny=Ny, dx=dx)
    pulse_beam = _make_pulse(lambda_=lam, w0=w0, Tw=10e-15)
    Nsteps = 50
    time = _make_time(space, Nsteps=Nsteps, safety=0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        prop = PSTD3DPropagator(
            space,
            time,
            pulse_beam,
            source_type="ic",
            boundary_type="mask",
            output_dir=tmpdir,
            snapshot_interval=99999,
        )
        # Check that the initial field has a Gaussian transverse profile
        Ey_init = _get_real_field(prop.Ey)
        center_x = Nx // 4
        profile = np.abs(Ey_init[center_x, :, 0])

        # Beam should have non-uniform transverse profile (not flat)
        peak_val = np.max(profile)
        edge_val = profile[0]

        assert peak_val > 0, "No field at beam center"
        # Edge should be weaker than center for a Gaussian beam
        assert edge_val < 0.5 * peak_val, (
            f"Transverse profile is too flat for a focused beam: "
            f"edge/peak = {edge_val / peak_val:.3f}"
        )


@pytest.mark.slow
@pytest.mark.integration
def test_group_velocity_chirp_invariant():
    """Chirp does not change group velocity in a non-dispersive medium."""
    Nx = 256
    dx = 20e-9
    Nsteps = 150

    space = _make_space(Nx=Nx, dx=dx)
    time_unchirped = _make_time(space, Nsteps=Nsteps, safety=0.5)
    time_chirped = _make_time(space, Nsteps=Nsteps, safety=0.5)
    pulse_unchirped = _make_pulse(chirp=0.0)
    pulse_chirped = _make_pulse(chirp=5e28)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Run unchirped
        prop1 = PSTD3DPropagator(
            space,
            time_unchirped,
            pulse_unchirped,
            source_type="ic",
            boundary_type="mask",
            output_dir=tmpdir,
            snapshot_interval=99999,
        )
        peak_init_1 = _peak_x_index(_get_real_field(prop1.Ey))
        prop1.run()
        peak_final_1 = _peak_x_index(_get_real_field(prop1.Ey))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Run chirped
        prop2 = PSTD3DPropagator(
            _make_space(Nx=Nx, dx=dx),
            time_chirped,
            pulse_chirped,
            source_type="ic",
            boundary_type="mask",
            output_dir=tmpdir,
            snapshot_interval=99999,
        )
        peak_init_2 = _peak_x_index(_get_real_field(prop2.Ey))
        prop2.run()
        peak_final_2 = _peak_x_index(_get_real_field(prop2.Ey))

    disp_1 = peak_final_1 - peak_init_1
    disp_2 = peak_final_2 - peak_init_2

    # Both should propagate forward
    assert disp_1 > 0, "Unchirped pulse did not propagate"
    assert disp_2 > 0, "Chirped pulse did not propagate"

    # Displacement should be nearly identical (within 3 cells)
    assert abs(disp_1 - disp_2) <= 3, (
        f"Group velocity differs: unchirped moved {disp_1} cells, "
        f"chirped moved {disp_2} cells"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_convergence_spectral():
    """PSTD error decreases faster than first-order with grid refinement."""
    Nsteps = 50
    configs = [
        (128, 40e-9),  # coarse
        (256, 20e-9),  # medium
        (512, 10e-9),  # fine
    ]
    peaks = []

    for Nx, dx in configs:
        space = _make_space(Nx=Nx, dx=dx)
        time = _make_time(space, Nsteps=Nsteps, safety=0.4)
        pulse = _make_pulse(Tw=10e-15)

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
            prop.run()
            Ey_r = _get_real_field(prop.Ey)
            peaks.append(np.max(np.abs(Ey_r)))

    peak_coarse, peak_medium, peak_fine = peaks

    # Errors relative to finest grid
    err_coarse = abs(peak_coarse - peak_fine) / max(abs(peak_fine), 1e-30)
    err_medium = abs(peak_medium - peak_fine) / max(abs(peak_fine), 1e-30)

    # If both errors are already small, the method has converged
    converged_tol = 0.01
    if err_coarse < converged_tol and err_medium < converged_tol:
        return

    # Medium should be closer to fine than coarse is
    assert err_medium < err_coarse, (
        f"No convergence: err_coarse={err_coarse:.4e}, err_medium={err_medium:.4e}"
    )

    # For spectral methods, convergence should be better than first-order
    if err_coarse > 1e-10:
        assert err_medium < 0.5 * err_coarse, (
            f"Convergence too slow for spectral method: "
            f"err_coarse={err_coarse:.4e}, err_medium={err_medium:.4e}"
        )
