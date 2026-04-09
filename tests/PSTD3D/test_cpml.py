"""
Test suite for pulsesuite.PSTD3D.tfsf
=======================================

Tests are written against **mathematical truth** (normalised Gaussian
profile, additive soft-source injection), not against the Fortran
implementation.

Coverage
--------
InitializeTFSF
~~~~~~~~~~~~~~
- Shape, dtype, non-negative
- Peak at 25 % of the x-axis
- Normalised Gaussian profile: integral(S * dx) ≈ 1
- sigma = 5 * dx
- Decays to near-zero at edges
- Different grid sizes (16, 32, 64, 128)
- Dielectric media (epsr > 1)

UpdateTFSC
~~~~~~~~~~
- Additive formula: E_out = E_prop + S · E_inc · dx
- Zero field → output equals S · E_inc · dx (after IFFT)
- Existing field preserved (additive, not replacement)
- Plane-wave mode (w0 = ∞) — uniform transverse profile
- Gaussian beam mode (finite w0) — transverse decay
- E_inc peak at τ=0 (retarded time)
- Chirped pulse: carrier frequency modulation
- FFT / IFFT round-trip preserves shape and dtype
- Zero amplitude → field unchanged
"""

import numpy as np
import pytest
from scipy.constants import c as c0

from pulsesuite.core.fftw import fft_3D, ifft_3D
from pulsesuite.PSTD3D.tfsf import InitializeTFSF, UpdateTFSC
from pulsesuite.PSTD3D.typepulse import ps
from pulsesuite.PSTD3D.typespace import GetDx, GetXArray, GetYArray, GetZArray, ss
from pulsesuite.PSTD3D.typetime import ts

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

DX = DY = DZ = 1e-7  # 100 nm
EPSR = 1.0
RNG = np.random.default_rng(42)


def _space(Nx=32, Ny=8, Nz=8, dx=DX, dy=DY, dz=DZ, epsr=EPSR):
    return ss(Dims=3, Nx=Nx, Ny=Ny, Nz=Nz, dx=dx, dy=dy, dz=dz, epsr=epsr)


def _pulse(lam=800e-9, Amp=1e8, Tw=30e-15, Tp=100e-15, chirp=0.0, w0=float("inf")):
    return ps(lambda_=lam, Amp=Amp, Tw=Tw, Tp=Tp, chirp=chirp, w0=w0)


def _time(t=0.0, tf=1e-12, dt=1e-15, n=1):
    return ts(t=t, tf=tf, dt=dt, n=n)


def _zeros(Nx=32, Ny=8, Nz=8):
    return np.zeros((Nx, Ny, Nz), dtype=np.complex128)


# ──────────────────────────────────────────────────────────────────────────────
# InitializeTFSF — basics
# ──────────────────────────────────────────────────────────────────────────────


class TestInitializeTFSFBasics:
    """Shape, dtype, non-negative, peak location."""

    @pytest.mark.parametrize("Nx", [16, 32, 64, 128])
    def test_shape_matches_nx(self, Nx):
        tfsf = InitializeTFSF(_space(Nx=Nx), _pulse())
        assert tfsf.shape == (Nx,)

    def test_dtype_float64(self):
        tfsf = InitializeTFSF(_space(), _pulse())
        assert tfsf.dtype == np.float64

    def test_non_negative(self):
        tfsf = InitializeTFSF(_space(Nx=128), _pulse())
        assert np.all(tfsf >= 0.0)

    def test_edges_near_zero(self):
        """With Nx=256, source at 25% is far enough from boundaries
        that the narrow Gaussian (sigma=5*dx) decays below 1e-6."""
        tfsf = InitializeTFSF(_space(Nx=256), _pulse())
        assert tfsf[0] < 1e-6
        assert tfsf[-1] < 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# InitializeTFSF — source location
# ──────────────────────────────────────────────────────────────────────────────


class TestInitializeTFSFLocation:
    """Source is placed at 25 % of the x-axis."""

    @pytest.mark.parametrize("Nx", [32, 64, 128, 256])
    def test_peak_at_25_percent(self, Nx):
        space = _space(Nx=Nx)
        tfsf = InitializeTFSF(space, _pulse())
        peak_idx = np.argmax(tfsf)
        expected_idx = int(round(Nx * 0.25))
        assert abs(peak_idx - expected_idx) <= 1

    def test_peak_position_matches_x_array(self):
        """The x-coordinate of the peak matches x[round(Nx*0.25)]."""
        Nx = 128
        space = _space(Nx=Nx)
        x = GetXArray(space)
        tfsf = InitializeTFSF(space, _pulse())
        xp_actual = x[np.argmax(tfsf)]
        xp_expected = x[int(round(Nx * 0.25))]
        assert np.isclose(xp_actual, xp_expected, rtol=1e-12, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# InitializeTFSF — normalised Gaussian profile
# ──────────────────────────────────────────────────────────────────────────────


class TestInitializeTFSFProfile:
    """Profile is a normalised Gaussian with sigma = 5*dx."""

    def test_gaussian_profile_matches_formula(self):
        """Verify the profile matches the analytical normalised Gaussian."""
        Nx = 128
        space = _space(Nx=Nx)
        pulse = _pulse(lam=800e-9)
        tfsf = InitializeTFSF(space, pulse)

        x = GetXArray(space)
        dx = GetDx(space)
        sigma = 5.0 * dx
        xp = x[int(round(Nx * 0.25))]
        expected = np.exp(-0.5 * ((x - xp) / sigma) ** 2) / (
            sigma * np.sqrt(2.0 * np.pi)
        )

        assert np.allclose(tfsf, expected, rtol=1e-12, atol=1e-12)

    def test_integral_approximately_one(self):
        """Normalised Gaussian: sum(S) * dx ≈ 1."""
        Nx = 256
        space = _space(Nx=Nx)
        pulse = _pulse()
        tfsf = InitializeTFSF(space, pulse)
        dx = GetDx(space)
        integral = np.sum(tfsf) * dx
        # Discrete sum of a Gaussian with sigma=5*dx should be very
        # close to the analytical integral of 1
        assert np.isclose(integral, 1.0, rtol=1e-6, atol=1e-6)

    def test_sigma_is_5_dx(self):
        """The source width is 5*dx regardless of wavelength."""
        for dx_val in [1e-7, 2e-7, 5e-8]:
            Nx = 128
            space = _space(Nx=Nx, dx=dx_val)
            pulse = _pulse()
            tfsf = InitializeTFSF(space, pulse)

            x = GetXArray(space)
            xp = x[np.argmax(tfsf)]
            sigma = 5.0 * dx_val

            expected = np.exp(-0.5 * ((x - xp) / sigma) ** 2) / (
                sigma * np.sqrt(2.0 * np.pi)
            )
            assert np.allclose(tfsf, expected, rtol=1e-12, atol=1e-12)

    def test_dielectric_does_not_change_width(self):
        """Unlike the old super-Gaussian, the new profile width depends
        only on dx, not on the medium wavelength."""
        Nx = 128
        pulse = _pulse(lam=800e-9)

        space_vac = _space(Nx=Nx, epsr=1.0)
        space_die = _space(Nx=Nx, epsr=4.0)

        tfsf_vac = InitializeTFSF(space_vac, pulse)
        tfsf_die = InitializeTFSF(space_die, pulse)

        # Same dx → same profile (independent of epsr)
        assert np.allclose(tfsf_vac, tfsf_die, rtol=1e-12, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# UpdateTFSC — additive injection
# ──────────────────────────────────────────────────────────────────────────────


class TestUpdateTFSCAdditive:
    r"""E_out = E_prop + S · E_inc · dx."""

    def _compute_E_inc(self, space, time, pulse, Emax_amp):
        """Compute the analytical incident field (Nx, Ny, Nz)."""
        x = GetXArray(space)
        yy = GetYArray(space)
        zz = GetZArray(space)
        v = c0 / np.sqrt(space.epsr)
        tau = time.t - x / v - pulse.Tp
        omega0 = pulse.CalcOmega0()
        tau_G = pulse.CalcTau()

        tau_3 = tau[:, np.newaxis, np.newaxis]
        yy_3 = yy[np.newaxis, :, np.newaxis]
        zz_3 = zz[np.newaxis, np.newaxis, :]
        gauss_yz = np.exp(-(yy_3**2 + zz_3**2) / pulse.w0**2)
        E_inc = (
            Emax_amp
            * gauss_yz
            * np.exp(-(tau_3**2) / tau_G**2)
            * np.cos(omega0 * tau_3 + pulse.chirp * tau_3**2)
        )
        return E_inc

    def test_zero_field_gets_additive_source(self):
        """Starting from E=0, output is S · E_inc · dx after IFFT."""
        Nx, Ny, Nz = 32, 8, 8
        space = _space(Nx=Nx, Ny=Ny, Nz=Nz)
        pulse = _pulse(Amp=1e8, Tp=0.0, w0=float("inf"))
        time = _time(t=0.0)
        tfsf = InitializeTFSF(space, pulse)

        E = _zeros(Nx, Ny, Nz)
        UpdateTFSC(E, tfsf, space, time, pulse, Emax_amp=pulse.Amp)

        # IFFT the result back to real space
        E_out = E.copy()
        ifft_3D(E_out)

        # Compute expected: S · E_inc · dx
        E_inc = self._compute_E_inc(space, time, pulse, pulse.Amp)
        S = tfsf[:, np.newaxis, np.newaxis]
        dx = GetDx(space)
        expected = S * E_inc * dx

        # FFT round-trip tolerance (spectral operations)
        assert np.allclose(E_out.real, expected, rtol=1e-4, atol=1e-4)

    def test_existing_field_preserved_outside_source(self):
        """Where S ≈ 0, the additive source does not modify the field."""
        Nx, Ny, Nz = 64, 8, 8
        space = _space(Nx=Nx, Ny=Ny, Nz=Nz)
        pulse = _pulse()
        time = _time()
        tfsf = InitializeTFSF(space, pulse)

        # Create a known real-space field, FFT to k-space
        E_real_orig = RNG.standard_normal((Nx, Ny, Nz)).astype(np.complex128)
        E = E_real_orig.copy()
        fft_3D(E)

        E_before = E_real_orig.copy()

        UpdateTFSC(E, tfsf, space, time, pulse, Emax_amp=pulse.Amp)

        # IFFT back to check
        E_after = E.copy()
        ifft_3D(E_after)

        # Where S ≈ 0 (far from source), field should be unchanged
        S = tfsf[:, np.newaxis, np.newaxis] * np.ones((1, Ny, Nz))
        far_mask = S < 1e-10
        if np.any(far_mask):
            assert np.allclose(
                E_after.real[far_mask],
                E_before.real[far_mask],
                rtol=1e-6,
                atol=1e-10,
            )

    def test_additive_formula_pointwise(self):
        """Verify the additive formula at every grid point."""
        Nx, Ny, Nz = 32, 4, 4
        space = _space(Nx=Nx, Ny=Ny, Nz=Nz)
        pulse = _pulse(Tp=0.0, w0=float("inf"))
        time = _time(t=0.0)
        tfsf = InitializeTFSF(space, pulse)

        # Known real-space field
        E_real_orig = RNG.standard_normal((Nx, Ny, Nz)).astype(np.complex128)
        E = E_real_orig.copy()
        fft_3D(E)

        UpdateTFSC(E, tfsf, space, time, pulse, Emax_amp=pulse.Amp)

        # IFFT result
        E_out = E.copy()
        ifft_3D(E_out)

        # Compute expected: E_prop + S * E_inc * dx
        E_inc = self._compute_E_inc(space, time, pulse, pulse.Amp)
        S = tfsf[:, np.newaxis, np.newaxis]
        dx = GetDx(space)
        expected = E_real_orig.real + S * E_inc * dx

        # FFT round-trip introduces small numerical error
        assert np.allclose(E_out.real, expected, rtol=1e-6, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# UpdateTFSC — incident field physics
# ──────────────────────────────────────────────────────────────────────────────


class TestUpdateTFSCIncidentField:
    r"""E_inc = Amp · G(y,z) · exp(-τ²/τ_G²) · cos(ω₀τ + χτ²)."""

    def test_plane_wave_uniform_transverse(self):
        """With w0=∞, G(y,z)=1 everywhere (no transverse decay)."""
        Nx, Ny, Nz = 32, 16, 16
        space = _space(Nx=Nx, Ny=Ny, Nz=Nz)
        pulse = _pulse(Amp=1e8, Tp=0.0, w0=float("inf"), chirp=0.0)
        time = _time(t=0.0)
        tfsf = InitializeTFSF(space, pulse)

        E = _zeros(Nx, Ny, Nz)
        UpdateTFSC(E, tfsf, space, time, pulse, Emax_amp=pulse.Amp)

        E_out = E.copy()
        ifft_3D(E_out)

        # At the peak of the source profile, all (y,z) slices should be identical
        peak_x = np.argmax(tfsf)
        slice_yz = E_out.real[peak_x, :, :]
        # All y,z values should be the same (plane wave)
        assert np.allclose(slice_yz, slice_yz[0, 0], rtol=1e-10, atol=1e-10)

    def test_gaussian_beam_transverse_decay(self):
        """With finite w0, the field decays away from y=z=0."""
        Nx, Ny, Nz = 32, 32, 32
        space = _space(Nx=Nx, Ny=Ny, Nz=Nz)
        w0 = 5e-7  # 500 nm beam waist
        pulse = _pulse(Amp=1e8, Tp=0.0, w0=w0, chirp=0.0)
        time = _time(t=0.0)
        tfsf = InitializeTFSF(space, pulse)

        E = _zeros(Nx, Ny, Nz)
        UpdateTFSC(E, tfsf, space, time, pulse, Emax_amp=pulse.Amp)

        E_out = E.copy()
        ifft_3D(E_out)

        # At the source peak, centre (y≈0,z≈0) should be stronger than edges
        peak_x = np.argmax(tfsf)
        centre_val = abs(E_out.real[peak_x, Ny // 2, Nz // 2])
        edge_val = abs(E_out.real[peak_x, 0, 0])
        if centre_val > 1e-15:
            assert edge_val < centre_val

    def test_chirped_modifies_carrier(self):
        """Chirp changes the instantaneous frequency but not envelope peak."""
        Nx, Ny, Nz = 64, 4, 4
        space = _space(Nx=Nx, Ny=Ny, Nz=Nz)
        time = _time(t=0.0)
        tfsf = InitializeTFSF(space, _pulse())

        pulse_unchirped = _pulse(chirp=0.0, Tp=0.0, w0=float("inf"))
        pulse_chirped = _pulse(chirp=1e26, Tp=0.0, w0=float("inf"))

        E1 = _zeros(Nx, Ny, Nz)
        E2 = _zeros(Nx, Ny, Nz)
        UpdateTFSC(E1, tfsf, space, time, pulse_unchirped, Emax_amp=1e8)
        UpdateTFSC(E2, tfsf, space, time, pulse_chirped, Emax_amp=1e8)

        # The fields should differ (chirp changes the carrier phase)
        assert not np.allclose(E1, E2, rtol=1e-6, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# UpdateTFSC — round-trip and invariants
# ──────────────────────────────────────────────────────────────────────────────


class TestUpdateTFSCInvariants:
    """Shape, dtype, and FFT round-trip properties."""

    def test_output_shape_dtype(self):
        Nx, Ny, Nz = 32, 8, 8
        space = _space(Nx=Nx, Ny=Ny, Nz=Nz)
        pulse = _pulse()
        time = _time()
        tfsf = InitializeTFSF(space, pulse)
        E = _zeros(Nx, Ny, Nz)
        UpdateTFSC(E, tfsf, space, time, pulse, Emax_amp=1e8)
        assert E.shape == (Nx, Ny, Nz)
        assert E.dtype == np.complex128

    def test_zero_amplitude_preserves_field(self):
        """With Emax_amp=0, E_inc=0, so E is unchanged (additive with 0)."""
        Nx, Ny, Nz = 32, 8, 8
        space = _space(Nx=Nx, Ny=Ny, Nz=Nz)
        pulse = _pulse()
        time = _time()
        tfsf = InitializeTFSF(space, pulse)

        # Create known field
        E_real_orig = RNG.standard_normal((Nx, Ny, Nz)).astype(np.complex128)
        E = E_real_orig.copy()
        fft_3D(E)
        E_before = E.copy()

        UpdateTFSC(E, tfsf, space, time, pulse, Emax_amp=0.0)

        # With Emax_amp=0, the additive term is 0, so E should survive
        # the IFFT→FFT round-trip unchanged
        assert np.allclose(E, E_before, rtol=1e-10, atol=1e-10)

    def test_multiple_calls_stable(self):
        """Calling UpdateTFSC multiple times doesn't produce NaN or overflow."""
        Nx, Ny, Nz = 32, 8, 8
        space = _space(Nx=Nx, Ny=Ny, Nz=Nz)
        pulse = _pulse(Tp=0.0, w0=float("inf"))
        time = _time(t=0.0)
        tfsf = InitializeTFSF(space, pulse)

        E = _zeros(Nx, Ny, Nz)
        for _ in range(5):
            UpdateTFSC(E, tfsf, space, time, pulse, Emax_amp=1e8)

        assert np.all(np.isfinite(E))

    @pytest.mark.parametrize("Nx,Ny,Nz", [(16, 8, 8), (32, 16, 16), (64, 4, 4)])
    def test_various_grid_sizes(self, Nx, Ny, Nz):
        """UpdateTFSC works for different grid dimensions."""
        space = _space(Nx=Nx, Ny=Ny, Nz=Nz)
        pulse = _pulse(w0=float("inf"))
        time = _time()
        tfsf = InitializeTFSF(space, pulse)
        E = _zeros(Nx, Ny, Nz)
        UpdateTFSC(E, tfsf, space, time, pulse, Emax_amp=1e8)
        assert E.shape == (Nx, Ny, Nz)
        assert np.all(np.isfinite(E))
