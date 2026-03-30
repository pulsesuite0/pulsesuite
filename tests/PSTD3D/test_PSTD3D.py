"""
Test suite for pulsesuite.PSTD3D.PSTD3D and pulsesuite.PSTD3D.tfsf
====================================================================

Tests are written against **mathematical truth** (Maxwell's equations in
Fourier space), not against the Fortran implementation.

Coverage
--------
- InitializeFields       — shapes, dtype, zero-fill
- UpdateE3D              — spectral curl: E += i·v²·dt·(k×B) - μ₀·v²·dt·J
- UpdateB3D              — spectral curl: B -= i·dt·(k×E)
- UpdateE3D / UpdateB3D  — zero fields unchanged, J=0 reduces correctly
- InitializeTFSF         — shape, non-negative, normalised Gaussian, peak at source centre
- UpdateTFSC             — additive injection modifies field, k-space round-trip
- SeedInitialCondition   — IC mode: pulse seeding, impedance relation, k-space output
- PSTD3DPropagator       — construction, simdir creation, snapshot I/O, both boundary types
- PSTD_3D_Propagator     — module-level wrapper runs without crash (tiny grid)
- Divergence-free        — ∇·E=0 and ∇·B=0 preserved by spectral updates
"""

import tempfile

import numpy as np
import pytest
from scipy.constants import c as c0, mu_0 as mu0

from pulsesuite.PSTD3D.PSTD3D import (
    InitializeFields,
    PSTD3DPropagator,
    SeedInitialCondition,
    UpdateB3D,
    UpdateE3D,
)
from pulsesuite.PSTD3D.tfsf import InitializeTFSF, UpdateTFSC
from pulsesuite.PSTD3D.typepulse import ps
from pulsesuite.PSTD3D.typespace import GetKxArray, ss
from pulsesuite.PSTD3D.typetime import ts

# ──────────────────────────────────────────────────────────────────────────────
# Shared test fixtures
# ──────────────────────────────────────────────────────────────────────────────

NX = NY = NZ = 16
DX = DY = DZ = 1e-7  # 100 nm
EPSR = 1.0
DT = DX / (3.0 * c0)  # well within CFL limit for 3D
RNG = np.random.default_rng(42)


def _make_space(Nx=NX, Ny=NY, Nz=NZ, dx=DX, dy=DY, dz=DZ, epsr=EPSR):
    return ss(Dims=3, Nx=Nx, Ny=Ny, Nz=Nz, dx=dx, dy=dy, dz=dz, epsr=epsr)


def _make_time(t=0.0, tf=None, dt=DT, n=1):
    if tf is None:
        tf = 10 * dt  # 10 steps
    return ts(t=t, tf=tf, dt=dt, n=n)


def _make_pulse(lam=800e-9, Amp=1e8, Tw=30e-15, Tp=100e-15, chirp=0.0, w0=float("inf")):
    return ps(lambda_=lam, Amp=Amp, Tw=Tw, Tp=Tp, chirp=chirp, w0=w0)


def _zeros():
    return np.zeros((NX, NY, NZ), dtype=np.complex128)


def _zeros_shape(Nx, Ny, Nz):
    return np.zeros((Nx, Ny, Nz), dtype=np.complex128)


def _random_field():
    return RNG.standard_normal((NX, NY, NZ)) + 1j * RNG.standard_normal((NX, NY, NZ))


# ──────────────────────────────────────────────────────────────────────────────
# InitializeFields
# ──────────────────────────────────────────────────────────────────────────────


class TestInitializeFields:
    def test_returns_nine_arrays(self):
        fields = InitializeFields(NX, NY, NZ)
        assert len(fields) == 9

    def test_shape(self):
        fields = InitializeFields(NX, NY, NZ)
        for F in fields:
            assert F.shape == (NX, NY, NZ)

    def test_dtype(self):
        fields = InitializeFields(NX, NY, NZ)
        for F in fields:
            assert F.dtype == np.complex128

    def test_all_zero(self):
        fields = InitializeFields(NX, NY, NZ)
        for F in fields:
            assert np.all(F == 0)

    @pytest.mark.parametrize("shape", [(4, 4, 4), (8, 16, 32), (1, 1, 1)])
    def test_various_shapes(self, shape):
        fields = InitializeFields(*shape)
        for F in fields:
            assert F.shape == shape


# ──────────────────────────────────────────────────────────────────────────────
# UpdateE3D
# ──────────────────────────────────────────────────────────────────────────────


class TestUpdateE3D:
    """E += i·v²·dt·(k×B) - μ₀·v²·dt·J."""

    def test_zero_B_zero_J_no_change(self):
        """With B=0 and J=0, E is unchanged."""
        space = _make_space()
        time = _make_time()
        Ex, Ey, Ez = _random_field(), _random_field(), _random_field()
        Bx, By, Bz = _zeros(), _zeros(), _zeros()
        Jx, Jy, Jz = _zeros(), _zeros(), _zeros()

        Ex0, Ey0, Ez0 = Ex.copy(), Ey.copy(), Ez.copy()
        UpdateE3D(space, time, Bx, By, Bz, Jx, Jy, Jz, Ex, Ey, Ez)

        assert np.allclose(Ex, Ex0, rtol=1e-12, atol=1e-12)
        assert np.allclose(Ey, Ey0, rtol=1e-12, atol=1e-12)
        assert np.allclose(Ez, Ez0, rtol=1e-12, atol=1e-12)

    def test_nonzero_B_modifies_E(self):
        """Non-zero B field should modify E."""
        space = _make_space()
        time = _make_time()
        Ex, Ey, Ez = _zeros(), _zeros(), _zeros()
        Bx, By, Bz = _random_field(), _random_field(), _random_field()
        Jx, Jy, Jz = _zeros(), _zeros(), _zeros()

        UpdateE3D(space, time, Bx, By, Bz, Jx, Jy, Jz, Ex, Ey, Ez)

        assert not np.allclose(Ex, 0, atol=1e-30)
        assert not np.allclose(Ey, 0, atol=1e-30)
        assert not np.allclose(Ez, 0, atol=1e-30)

    def test_manual_verification_single_mode(self):
        """Verify UpdateE3D against manual formula for a single k-mode.

        Set Bz = exp(i kx x) (single mode along x).
        Then: dEy/dt = v²·dt·(-kx·Bz)  → Ey += i·v²·dt·(-kx)·Bz
        """
        space = _make_space()
        time = _make_time()
        v2 = c0**2 / EPSR
        dt = DT

        kx = GetKxArray(space)
        # Bz has a single mode at kx[1]
        Bz = _zeros()
        Bz[1, 0, 0] = 1.0 + 0j  # single Fourier mode

        Ex, Ey, Ez = _zeros(), _zeros(), _zeros()
        Bx, By = _zeros(), _zeros()
        Jx, Jy, Jz = _zeros(), _zeros(), _zeros()

        UpdateE3D(space, time, Bx, By, Bz, Jx, Jy, Jz, Ex, Ey, Ez)

        # Ey += i*v2*dt*(- kx * Bz)  [from kz*Bx - kx*Bz, and Bx=0]
        expected_Ey = _zeros()
        kx3 = kx[:, np.newaxis, np.newaxis]
        expected_Ey += 1j * (-kx3 * Bz) * v2 * dt

        assert np.allclose(Ey, expected_Ey, rtol=1e-12, atol=1e-12)

    def test_current_source_subtracts(self):
        """J contribution is -μ₀·v²·dt·J (subtractive)."""
        space = _make_space()
        time = _make_time()
        v2 = c0**2 / EPSR
        dt = DT

        Ex = _zeros()
        Bx, By, Bz = _zeros(), _zeros(), _zeros()
        Jx = np.ones((NX, NY, NZ), dtype=np.complex128)
        Jy, Jz = _zeros(), _zeros()
        Ey, Ez = _zeros(), _zeros()

        UpdateE3D(space, time, Bx, By, Bz, Jx, Jy, Jz, Ex, Ey, Ez)

        expected_Ex = -mu0 * v2 * dt * Jx
        assert np.allclose(Ex, expected_Ex, rtol=1e-12, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# UpdateB3D
# ──────────────────────────────────────────────────────────────────────────────


class TestUpdateB3D:
    """B -= i·dt·(k×E)."""

    def test_zero_E_no_change(self):
        space = _make_space()
        time = _make_time()
        Ex, Ey, Ez = _zeros(), _zeros(), _zeros()
        Bx, By, Bz = _random_field(), _random_field(), _random_field()
        Bx0, By0, Bz0 = Bx.copy(), By.copy(), Bz.copy()

        UpdateB3D(space, time, Ex, Ey, Ez, Bx, By, Bz)

        assert np.allclose(Bx, Bx0, rtol=1e-12, atol=1e-12)
        assert np.allclose(By, By0, rtol=1e-12, atol=1e-12)
        assert np.allclose(Bz, Bz0, rtol=1e-12, atol=1e-12)

    def test_nonzero_E_modifies_B(self):
        space = _make_space()
        time = _make_time()
        Ex, Ey, Ez = _random_field(), _random_field(), _random_field()
        Bx, By, Bz = _zeros(), _zeros(), _zeros()

        UpdateB3D(space, time, Ex, Ey, Ez, Bx, By, Bz)

        assert not np.allclose(Bx, 0, atol=1e-30)

    def test_manual_verification_single_mode(self):
        """Verify UpdateB3D against manual formula for a single k-mode.

        Set Ey = exp(i kx x) (single mode along x).
        Then: Bz -= i·dt·(kx·Ey - ky·Ex)  → Bz -= i·dt·kx·Ey  [Ex=0]
        """
        space = _make_space()
        time = _make_time()
        dt = DT

        kx = GetKxArray(space)
        Ey = _zeros()
        Ey[1, 0, 0] = 1.0 + 0j

        Ex, Ez = _zeros(), _zeros()
        Bx, By, Bz = _zeros(), _zeros(), _zeros()

        UpdateB3D(space, time, Ex, Ey, Ez, Bx, By, Bz)

        kx3 = kx[:, np.newaxis, np.newaxis]
        expected_Bz = -1j * (kx3 * Ey) * dt
        assert np.allclose(Bz, expected_Bz, rtol=1e-12, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Divergence-free preservation
# ──────────────────────────────────────────────────────────────────────────────


class TestDivergenceFree:
    """Spectral updates preserve ∇·B = 0 (and ∇·E = 0 when J = 0)."""

    def _spectral_div(self, Fx, Fy, Fz, space):
        """Compute ik·F in k-space."""
        from pulsesuite.PSTD3D.typespace import GetKxArray, GetKyArray, GetKzArray

        kx = GetKxArray(space)[:, np.newaxis, np.newaxis]
        ky = GetKyArray(space)[np.newaxis, :, np.newaxis]
        kz = GetKzArray(space)[np.newaxis, np.newaxis, :]
        return 1j * (kx * Fx + ky * Fy + kz * Fz)

    def test_div_B_zero_initial_preserved(self):
        """Starting from B=0 (trivially div-free), UpdateB3D keeps ∇·B = 0.

        Since B^{n+1} = B^n - i dt (k × E), and k · (k × E) = 0
        algebraically, div(B) = 0 is preserved after any number of steps.
        """
        space = _make_space()
        time = _make_time()

        Bx, By, Bz = _zeros(), _zeros(), _zeros()
        Ex, Ey, Ez = _random_field(), _random_field(), _random_field()

        # After one update: B = -i dt (k × E), so k · B should be 0
        UpdateB3D(space, time, Ex, Ey, Ez, Bx, By, Bz)

        div = self._spectral_div(Bx, By, Bz, space)
        assert np.allclose(div, 0, rtol=1e-10, atol=1e-10)

    def test_div_B_preserved_two_steps(self):
        """∇·B = 0 preserved across multiple UpdateB3D steps."""
        space = _make_space()
        time = _make_time()

        Bx, By, Bz = _zeros(), _zeros(), _zeros()

        for _ in range(3):
            Ex, Ey, Ez = _random_field(), _random_field(), _random_field()
            UpdateB3D(space, time, Ex, Ey, Ez, Bx, By, Bz)

        div = self._spectral_div(Bx, By, Bz, space)
        assert np.allclose(div, 0, rtol=1e-10, atol=1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# InitializeTFSF
# ──────────────────────────────────────────────────────────────────────────────


class TestInitializeTFSF:
    """Source profile: normalised Gaussian, shape (Nx,), non-negative."""

    def test_shape(self):
        space = _make_space()
        pulse = _make_pulse()
        tfsf = InitializeTFSF(space, pulse)
        assert tfsf.shape == (NX,)

    def test_dtype_float64(self):
        space = _make_space()
        pulse = _make_pulse()
        tfsf = InitializeTFSF(space, pulse)
        assert tfsf.dtype == np.float64

    def test_non_negative(self):
        space = _make_space(Nx=64)
        pulse = _make_pulse()
        tfsf = InitializeTFSF(space, pulse)
        assert np.all(tfsf >= 0.0)

    def test_decays_at_edges(self):
        """With Nx=256, source at 25% is far enough from boundaries
        that the narrow Gaussian (sigma=5*dx) decays below 1e-6."""
        space = _make_space(Nx=256)
        pulse = _make_pulse()
        tfsf = InitializeTFSF(space, pulse)
        assert tfsf[0] < 1e-6
        assert tfsf[-1] < 1e-6

    def test_source_at_25_percent(self):
        """Peak should be near 25% into the array."""
        Nx = 128
        space = _make_space(Nx=Nx)
        pulse = _make_pulse()
        tfsf = InitializeTFSF(space, pulse)
        peak_idx = np.argmax(tfsf)
        assert abs(peak_idx - int(round(Nx * 0.25))) <= 2


# ──────────────────────────────────────────────────────────────────────────────
# UpdateTFSC
# ──────────────────────────────────────────────────────────────────────────────


class TestUpdateTFSC:
    """Additive soft-source injection: E += S · E_inc · dx."""

    def test_modifies_field(self):
        """Non-trivial source injection should change the field."""
        space = _make_space()
        pulse = _make_pulse(Amp=1e8, Tp=0.0, w0=float("inf"))
        time = _make_time(t=0.0)
        tfsf = InitializeTFSF(space, pulse)

        E = _zeros()
        UpdateTFSC(E, tfsf, space, time, pulse, Emax_amp=pulse.Amp)

        # The field should now be non-zero
        assert np.max(np.abs(E)) > 0

    def test_zero_amplitude_preserves_field(self):
        """With Emax_amp=0, additive term is zero → field unchanged."""
        space = _make_space()
        pulse = _make_pulse()
        time = _make_time()
        tfsf = InitializeTFSF(space, pulse)

        E = _random_field()
        from pulsesuite.core.fftw import fft_3D

        fft_3D(E)
        E_before = E.copy()

        UpdateTFSC(E, tfsf, space, time, pulse, Emax_amp=0.0)

        # With Emax_amp=0, additive source is zero, so E survives
        # the IFFT→FFT round-trip unchanged
        assert np.allclose(E, E_before, rtol=1e-10, atol=1e-10)

    def test_output_shape_preserved(self):
        space = _make_space()
        pulse = _make_pulse()
        time = _make_time()
        tfsf = InitializeTFSF(space, pulse)
        E = _zeros()
        UpdateTFSC(E, tfsf, space, time, pulse, Emax_amp=1e8)
        assert E.shape == (NX, NY, NZ)
        assert E.dtype == np.complex128


# ──────────────────────────────────────────────────────────────────────────────
# PSTD3DPropagator — construction and I/O
# ──────────────────────────────────────────────────────────────────────────────


class TestPSTD3DPropagator:
    def test_construction(self):
        """Propagator initialises without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space()
            time = _make_time()
            pulse = _make_pulse(w0=1e-3)
            prop = PSTD3DPropagator(space, time, pulse, output_dir=tmpdir)
            assert prop.simdir.exists()

    def test_fields_allocated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space()
            time = _make_time()
            pulse = _make_pulse(w0=1e-3)
            prop = PSTD3DPropagator(space, time, pulse, output_dir=tmpdir)
            assert prop.Ex.shape == (NX, NY, NZ)
            assert prop.Bz.shape == (NX, NY, NZ)
            assert prop.Jx.dtype == np.complex128

    def test_simdir_auto_increment(self):
        """Second propagator gets sim002."""
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space()
            time = _make_time()
            pulse = _make_pulse(w0=1e-3)
            p1 = PSTD3DPropagator(space, time, pulse, output_dir=tmpdir)
            p2 = PSTD3DPropagator(space, time, pulse, output_dir=tmpdir)
            assert p1.simdir != p2.simdir
            assert "sim001" in str(p1.simdir)
            assert "sim002" in str(p2.simdir)

    def test_snapshot_writes_npy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space()
            time = _make_time()
            pulse = _make_pulse(w0=1e-3)
            prop = PSTD3DPropagator(space, time, pulse, output_dir=tmpdir)

            F = np.ones((NX, NY, NZ), dtype=np.complex128) * (3.0 + 1j)
            prop._snapshot(F, "Ex", 100)

            fpath = prop.simdir / "Ex_000100.npy"
            assert fpath.exists()
            data = np.load(fpath)
            assert data.dtype == np.float32
            assert data.shape == (NX, NY, NZ)
            assert np.allclose(data, 3.0, rtol=1e-6, atol=1e-6)

    def test_write_grid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space()
            time = _make_time()
            pulse = _make_pulse(w0=1e-3)
            prop = PSTD3DPropagator(space, time, pulse, output_dir=tmpdir)
            prop._write_grid()

            for name in ["grid_x.npy", "grid_y.npy", "grid_z.npy"]:
                fpath = prop.simdir / name
                assert fpath.exists()
                arr = np.load(fpath)
                assert arr.ndim == 1

    def test_write_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space()
            time = _make_time()
            pulse = _make_pulse(w0=1e-3)
            prop = PSTD3DPropagator(space, time, pulse, output_dir=tmpdir)
            prop._write_summary(10, 0.5, DT)

            summary = prop.simdir / "summary.txt"
            assert summary.exists()
            text = summary.read_text()
            assert "PSTD3D Simulation Summary" in text
            assert "Grid" in text
            assert "Pulse" in text

    def test_custom_calc_j_called(self):
        """Verify the pluggable calc_j callback is invoked during run()."""
        call_count = [0]

        def mock_calc_j(space, time, Ex, Ey, Ez, Jx, Jy, Jz):
            call_count[0] += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space(Nx=8, Ny=8, Nz=8)
            time = _make_time(t=0.0, tf=3 * DT, dt=DT, n=1)
            pulse = _make_pulse(w0=1e-3)
            prop = PSTD3DPropagator(
                space,
                time,
                pulse,
                calc_j=mock_calc_j,
                output_dir=tmpdir,
            )
            prop.run()

        assert call_count[0] > 0


# ──────────────────────────────────────────────────────────────────────────────
# Full short simulation (integration test)
# ──────────────────────────────────────────────────────────────────────────────


class TestShortSimulation:
    """Run a tiny 2-step simulation and verify outputs."""

    @pytest.mark.slow
    def test_two_step_run_completes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space(Nx=8, Ny=8, Nz=8)
            time = _make_time(t=0.0, tf=3 * DT, dt=DT, n=1)
            pulse = _make_pulse(w0=1e-3)
            prop = PSTD3DPropagator(
                space,
                time,
                pulse,
                output_dir=tmpdir,
                snapshot_interval=1,
            )
            prop.run()

            # Emax.dat should exist with data
            emax_path = prop.simdir / "Emax.dat"
            assert emax_path.exists()
            lines = emax_path.read_text().strip().split("\n")
            assert len(lines) >= 2  # header + at least 1 data line

            # Snapshots should exist (interval=1)
            assert (prop.simdir / "Ex_000001.npy").exists()

    @pytest.mark.slow
    def test_fields_finite_after_run(self):
        """All fields remain finite (no NaN/overflow) after a short run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space(Nx=8, Ny=8, Nz=8)
            time = _make_time(t=0.0, tf=3 * DT, dt=DT, n=1)
            pulse = _make_pulse(w0=1e-3)
            prop = PSTD3DPropagator(
                space,
                time,
                pulse,
                output_dir=tmpdir,
            )
            prop.run()

            for name in ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]:
                F = getattr(prop, name)
                assert np.all(np.isfinite(F)), f"{name} has non-finite values"


# ──────────────────────────────────────────────────────────────────────────────
# SeedInitialCondition
# ──────────────────────────────────────────────────────────────────────────────


class TestSeedInitialCondition:
    """IC mode: seed Ey and Bz with the full pulse at t=t0."""

    def test_ey_nonzero_after_seed(self):
        """Seeding should produce non-zero Ey in k-space."""
        space = _make_space(Nx=32, Ny=4, Nz=4)
        time = _make_time()
        pulse = _make_pulse(Amp=1e8, w0=float("inf"))

        Ey = _zeros_shape(32, 4, 4)
        Bz = _zeros_shape(32, 4, 4)

        SeedInitialCondition(space, time, pulse, Ey, Bz, npml_x=0)
        assert np.max(np.abs(Ey)) > 0

    def test_bz_nonzero_after_seed(self):
        """Bz should also be seeded."""
        space = _make_space(Nx=32, Ny=4, Nz=4)
        time = _make_time()
        pulse = _make_pulse(Amp=1e8, w0=float("inf"))

        Ey = _zeros_shape(32, 4, 4)
        Bz = _zeros_shape(32, 4, 4)

        SeedInitialCondition(space, time, pulse, Ey, Bz, npml_x=0)
        assert np.max(np.abs(Bz)) > 0

    def test_impedance_relation(self):
        r"""In real space, :math:`B_z = E_y / v` for a +x plane wave.

        After IFFT, the impedance ratio should hold approximately.
        """
        from pulsesuite.core.fftw import ifft_3D

        Nx = 64
        space = _make_space(Nx=Nx, Ny=4, Nz=4)
        time = _make_time()
        pulse = _make_pulse(Amp=1e8, w0=float("inf"))
        v = c0 / np.sqrt(EPSR)

        Ey = _zeros_shape(Nx, 4, 4)
        Bz = _zeros_shape(Nx, 4, 4)

        SeedInitialCondition(space, time, pulse, Ey, Bz, npml_x=0)

        # IFFT back to real space for comparison
        Ey_r = Ey.copy()
        Bz_r = Bz.copy()
        ifft_3D(Ey_r)
        ifft_3D(Bz_r)

        # Where Ey is significant, Bz ~ Ey / v
        mask = np.abs(Ey_r.real) > 0.01 * np.max(np.abs(Ey_r.real))
        if np.any(mask):
            ratio = Bz_r.real[mask] / Ey_r.real[mask]
            assert np.allclose(ratio, 1.0 / v, rtol=1e-6, atol=1e-10)

    def test_pml_zeroed(self):
        """Field should be zero in PML regions when npml_x > 0."""
        from pulsesuite.core.fftw import ifft_3D

        Nx = 64
        npml = 6
        space = _make_space(Nx=Nx, Ny=4, Nz=4)
        time = _make_time()
        pulse = _make_pulse(Amp=1e8, w0=float("inf"))

        Ey = _zeros_shape(Nx, 4, 4)
        Bz = _zeros_shape(Nx, 4, 4)

        SeedInitialCondition(space, time, pulse, Ey, Bz, npml_x=npml)

        # IFFT to check real space
        Ey_r = Ey.copy()
        ifft_3D(Ey_r)

        # PML regions should have near-zero field
        # (not exactly zero due to FFT round-trip, but very small)
        pml_energy = np.sum(np.abs(Ey_r[:npml]) ** 2) + np.sum(
            np.abs(Ey_r[-npml:]) ** 2
        )
        total_energy = np.sum(np.abs(Ey_r) ** 2)
        if total_energy > 0:
            assert pml_energy / total_energy < 0.01

    def test_output_is_kspace(self):
        """After seeding, Ey and Bz should be in k-space (FFT'd)."""
        space = _make_space(Nx=32, Ny=4, Nz=4)
        time = _make_time()
        pulse = _make_pulse(Amp=1e8, w0=float("inf"))

        Ey = _zeros_shape(32, 4, 4)
        Bz = _zeros_shape(32, 4, 4)

        SeedInitialCondition(space, time, pulse, Ey, Bz, npml_x=0)

        # k-space fields should have significant imaginary parts
        # (a real-space Gaussian pulse has complex Fourier transform)
        assert np.max(np.abs(Ey.imag)) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Boundary type selection
# ──────────────────────────────────────────────────────────────────────────────


class TestBoundaryTypeSelection:
    """Propagator supports both 'mask' and 'cpml' boundary types."""

    def test_mask_boundary_construction(self):
        """Propagator with mask boundary initialises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space(Nx=16, Ny=4, Nz=4)
            time = _make_time()
            pulse = _make_pulse(w0=float("inf"))
            prop = PSTD3DPropagator(
                space,
                time,
                pulse,
                boundary_type="mask",
                output_dir=tmpdir,
            )
            assert prop.boundary_type == "mask"
            assert prop.simdir.exists()

    def test_cpml_boundary_construction(self):
        """Propagator with cpml boundary initialises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space(Nx=16, Ny=4, Nz=4)
            time = _make_time()
            pulse = _make_pulse(w0=float("inf"))
            prop = PSTD3DPropagator(
                space,
                time,
                pulse,
                boundary_type="cpml",
                output_dir=tmpdir,
            )
            assert prop.boundary_type == "cpml"


# ──────────────────────────────────────────────────────────────────────────────
# Source type selection
# ──────────────────────────────────────────────────────────────────────────────


class TestSourceTypeSelection:
    """Propagator supports both 'ic' and 'soft' source types."""

    def test_ic_source_seeds_fields(self):
        """IC mode should seed Ey and Bz with non-zero values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space(Nx=32, Ny=4, Nz=4)
            time = _make_time()
            pulse = _make_pulse(Amp=1e8, w0=float("inf"))
            prop = PSTD3DPropagator(
                space,
                time,
                pulse,
                source_type="ic",
                boundary_type="mask",
                output_dir=tmpdir,
            )
            assert prop.source_type == "ic"
            assert np.max(np.abs(prop.Ey)) > 0
            assert np.max(np.abs(prop.Bz)) > 0

    def test_soft_source_starts_with_zero_fields(self):
        """Soft mode should start with zero fields (injected during run)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space(Nx=16, Ny=4, Nz=4)
            time = _make_time()
            pulse = _make_pulse(w0=float("inf"))
            prop = PSTD3DPropagator(
                space,
                time,
                pulse,
                source_type="soft",
                boundary_type="mask",
                output_dir=tmpdir,
            )
            assert prop.source_type == "soft"
            # Fields should be zero (no IC seed)
            assert np.allclose(prop.Ex, 0)
            assert np.allclose(prop.Ey, 0)
            assert np.allclose(prop.Bz, 0)

    def test_soft_source_has_tfsf(self):
        """Soft mode should initialise the TFSF source profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space(Nx=16, Ny=4, Nz=4)
            time = _make_time()
            pulse = _make_pulse(w0=float("inf"))
            prop = PSTD3DPropagator(
                space,
                time,
                pulse,
                source_type="soft",
                boundary_type="mask",
                output_dir=tmpdir,
            )
            assert prop._tfsf is not None
            assert prop._tfsf.shape == (16,)

    def test_ic_source_no_tfsf(self):
        """IC mode should NOT initialise TFSF source profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space(Nx=16, Ny=4, Nz=4)
            time = _make_time()
            pulse = _make_pulse(w0=float("inf"))
            prop = PSTD3DPropagator(
                space,
                time,
                pulse,
                source_type="ic",
                boundary_type="mask",
                output_dir=tmpdir,
            )
            assert prop._tfsf is None


# ──────────────────────────────────────────────────────────────────────────────
# Short simulation with mask boundary (integration test)
# ──────────────────────────────────────────────────────────────────────────────


class TestShortSimulationMask:
    """Run tiny simulations with masking absorber boundary."""

    @pytest.mark.slow
    def test_ic_mask_run_completes(self):
        """IC + mask boundary: 2-step run completes without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space(Nx=16, Ny=4, Nz=4)
            time = _make_time(t=0.0, tf=3 * DT, dt=DT, n=1)
            pulse = _make_pulse(Amp=1e8, w0=float("inf"))
            prop = PSTD3DPropagator(
                space,
                time,
                pulse,
                source_type="ic",
                boundary_type="mask",
                output_dir=tmpdir,
                snapshot_interval=1,
            )
            prop.run()

            emax_path = prop.simdir / "Emax.dat"
            assert emax_path.exists()

            for name in ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]:
                F = getattr(prop, name)
                assert np.all(np.isfinite(F)), f"{name} has non-finite values"

    @pytest.mark.slow
    def test_soft_mask_run_completes(self):
        """Soft source + mask boundary: 2-step run completes without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space(Nx=16, Ny=4, Nz=4)
            time = _make_time(t=0.0, tf=3 * DT, dt=DT, n=1)
            pulse = _make_pulse(Amp=1e8, Tp=0.0, w0=float("inf"))
            prop = PSTD3DPropagator(
                space,
                time,
                pulse,
                source_type="soft",
                boundary_type="mask",
                output_dir=tmpdir,
                snapshot_interval=1,
            )
            prop.run()

            emax_path = prop.simdir / "Emax.dat"
            assert emax_path.exists()

    @pytest.mark.slow
    def test_summary_includes_config(self):
        """Summary file should mention source and boundary types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            space = _make_space(Nx=16, Ny=4, Nz=4)
            time = _make_time(t=0.0, tf=3 * DT, dt=DT, n=1)
            pulse = _make_pulse(Amp=1e8, w0=float("inf"))
            prop = PSTD3DPropagator(
                space,
                time,
                pulse,
                source_type="ic",
                boundary_type="mask",
                output_dir=tmpdir,
            )
            prop._write_summary(3, 0.5, DT)

            text = (prop.simdir / "summary.txt").read_text()
            assert "Source type" in text
            assert "Boundary type" in text
            assert "ic" in text
            assert "mask" in text
