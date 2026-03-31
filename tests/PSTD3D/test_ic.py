"""
Test suite for pulsesuite.PSTD3D.ic
====================================

Tests verify **physical correctness** of the initial-condition seeding,
not implementation details.

Coverage
--------
ValidateTransverseGrid
~~~~~~~~~~~~~~~~~~~~~~
- No warning when grid is fine (Ny large enough)
- Warning when Ny is too small (dk_y / k_carrier > 10)
- No warning when Ny = 1 (1D mode, transverse dimension collapsed)
- No warning when Nz = 1 (2D mode, z-dimension collapsed)
- Warning when Nz is too small (dk_z / k_carrier > 10)

SeedInitialCondition
~~~~~~~~~~~~~~~~~~~~
- Impedance relation: Bz = Ey / v everywhere (forward-propagating mode)
- Peak location: maximum field near x_center = x[Nx // 4]
- Envelope shape: Gaussian with spatial width sigma_x = v * tau_G
- Carrier wavelength: FFT of Ey peaks at k_med = omega0 / v
- PML tapering: with npml_x > 0, field is near zero at boundaries
- Unidirectional propagation: +k modes dominate over -k modes
"""

import warnings

import numpy as np
import pytest
from scipy.constants import c as c0

from pulsesuite.core.fftw import ifft_3D
from pulsesuite.PSTD3D.ic import SeedInitialCondition, ValidateTransverseGrid
from pulsesuite.PSTD3D.typepulse import ps
from pulsesuite.PSTD3D.typespace import GetXArray, ss
from pulsesuite.PSTD3D.typetime import ts

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

DX = DY = DZ = 1e-7  # 100 nm pixel size
EPSR = 1.0
LAMBDA = 800e-9  # 800 nm carrier wavelength
AMP = 1e8  # V/m
TW = 30e-15  # 30 fs FWHM pulse width
TP = 100e-15  # peak crossing time
TWOPI = 2.0 * np.pi


def _space(Nx=128, Ny=1, Nz=1, dx=DX, dy=DY, dz=DZ, epsr=EPSR):
    return ss(Dims=3, Nx=Nx, Ny=Ny, Nz=Nz, dx=dx, dy=dy, dz=dz, epsr=epsr)


def _pulse(lam=LAMBDA, Amp=AMP, Tw=TW, Tp=TP, chirp=0.0, w0=float("inf")):
    return ps(lambda_=lam, Amp=Amp, Tw=Tw, Tp=Tp, chirp=chirp, w0=w0)


def _time(t=0.0, tf=1e-12, dt=1e-15, n=1):
    return ts(t=t, tf=tf, dt=dt, n=n)


def _zeros(Nx=128, Ny=1, Nz=1):
    return np.zeros((Nx, Ny, Nz), dtype=np.complex128)


def _seed_and_get_real(space, time, pulse, npml_x=0):
    """Seed fields, IFFT back to real space, return (Ey_real, Bz_real, x)."""
    Nx, Ny, Nz = space.Nx, space.Ny, space.Nz
    Ey = _zeros(Nx, Ny, Nz)
    Bz = _zeros(Nx, Ny, Nz)
    SeedInitialCondition(space, time, pulse, Ey, Bz, npml_x=npml_x)
    # SeedInitialCondition leaves fields in k-space; transform back
    ifft_3D(Ey)
    ifft_3D(Bz)
    x = GetXArray(space)
    return Ey, Bz, x


# ──────────────────────────────────────────────────────────────────────────────
# ValidateTransverseGrid
# ──────────────────────────────────────────────────────────────────────────────


class TestValidateTransverseGrid:
    """Spectral aliasing warnings for transverse grid resolution."""

    def test_no_warning_when_grid_is_fine(self):
        """A grid with enough transverse points should produce no warning."""
        # Ny = 128 with dy = 100 nm for lambda = 800 nm is well-resolved
        space = _space(Ny=128, Nz=1)
        pulse = _pulse()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidateTransverseGrid(space, pulse)
        assert len(w) == 0, f"Unexpected warning: {[str(x.message) for x in w]}"

    def test_warning_when_ny_too_small(self):
        """When Ny is small enough that dk_y / k_carrier > 10, warn."""
        # Need Ny * dy < lambda / (10 * sqrt(epsr))
        # lambda / 10 = 80e-9, so Ny=2, dy=20e-9 => Ny*dy = 40e-9 < 80e-9
        space = _space(Ny=2, dy=20e-9)
        pulse = _pulse()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidateTransverseGrid(space, pulse)
        assert len(w) >= 1
        assert "transverse points in y" in str(w[0].message).lower()

    def test_no_warning_ny_1_mode(self):
        """1D mode (Ny = 1) should never warn about transverse resolution."""
        space = _space(Ny=1, Nz=1)
        pulse = _pulse()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidateTransverseGrid(space, pulse)
        assert len(w) == 0

    def test_no_warning_nz_1_mode(self):
        """2D mode (Nz = 1) should not warn about z resolution."""
        space = _space(Ny=128, Nz=1)
        pulse = _pulse()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidateTransverseGrid(space, pulse)
        assert len(w) == 0

    def test_warning_when_nz_too_small(self):
        """When Nz is small and dk_z / k_carrier > 10, warn about z."""
        space = _space(Ny=1, Nz=2, dz=20e-9)
        pulse = _pulse()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidateTransverseGrid(space, pulse)
        assert len(w) >= 1
        assert "transverse points in z" in str(w[0].message).lower()

    def test_both_y_and_z_warn_independently(self):
        """If both y and z are under-resolved, both warnings should fire."""
        space = _space(Ny=2, dy=20e-9, Nz=2, dz=20e-9)
        pulse = _pulse()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidateTransverseGrid(space, pulse)
        assert len(w) == 2
        messages = [str(x.message).lower() for x in w]
        assert any("in y" in m for m in messages)
        assert any("in z" in m for m in messages)

    @pytest.mark.parametrize("epsr", [1.0, 2.25, 12.0])
    def test_accounts_for_refractive_index(self, epsr):
        """k_carrier = 2*pi*n / lambda depends on epsr; check no false alarm."""
        # Higher epsr => larger k_carrier => easier to resolve
        space = _space(Ny=4, dy=DY, epsr=epsr)
        pulse = _pulse()
        k_carrier = TWOPI * np.sqrt(epsr) / LAMBDA
        dk_y = TWOPI / (4 * DY)
        ratio = dk_y / k_carrier
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidateTransverseGrid(space, pulse)
        if ratio > 10.0:
            assert len(w) >= 1
        else:
            assert len(w) == 0


# ──────────────────────────────────────────────────────────────────────────────
# SeedInitialCondition — impedance relation
# ──────────────────────────────────────────────────────────────────────────────


class TestImpedanceRelation:
    """For a +x propagating pulse, Bz = Ey / v everywhere."""

    def test_impedance_1d(self):
        """In 1D (Ny=Nz=1), Bz = Ey / v at every grid point."""
        space = _space(Nx=256)
        pulse = _pulse()
        time = _time()
        v = c0 / np.sqrt(space.epsr)
        Ey, Bz, _ = _seed_and_get_real(space, time, pulse)
        # Compare where field is non-negligible to avoid 0/0
        mask = np.abs(Ey) > 1e-6 * AMP
        np.testing.assert_allclose(
            Bz[mask],
            Ey[mask] / v,
            rtol=1e-10,
            err_msg="Impedance relation Bz = Ey/v violated",
        )

    @pytest.mark.parametrize("epsr", [1.0, 2.25])
    def test_impedance_with_dielectric(self, epsr):
        """Impedance relation holds in dielectric media (v = c0 / n).

        Note: GetEpsr forces epsr=1.0 when Nz==1 (1D mode). Use Nz>1
        to actually test dielectric propagation.
        """
        space = _space(Nx=256, Ny=4, Nz=4, epsr=epsr)
        pulse = _pulse(w0=1e-5)
        time = _time()
        from pulsesuite.PSTD3D.typespace import GetEpsr

        effective_epsr = GetEpsr(space)
        v = c0 / np.sqrt(effective_epsr)
        Ey, Bz, _ = _seed_and_get_real(space, time, pulse)
        mask = np.abs(Ey) > 1e-6 * AMP
        np.testing.assert_allclose(Bz[mask], Ey[mask] / v, rtol=1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# SeedInitialCondition — peak location
# ──────────────────────────────────────────────────────────────────────────────


class TestPeakLocation:
    """Maximum field amplitude should be near x_center = x[Nx // 4]."""

    def test_peak_at_quarter_grid(self):
        """Envelope peak is at 25% of the x-axis (Nx // 4)."""
        space = _space(Nx=256)
        pulse = _pulse()
        time = _time()
        Ey, _, x = _seed_and_get_real(space, time, pulse)
        envelope = np.abs(Ey[:, 0, 0])
        peak_idx = np.argmax(envelope)
        expected_idx = space.Nx // 4
        # Allow +/- 2 grid cells tolerance for discretization
        assert abs(peak_idx - expected_idx) <= 2, (
            f"Peak at index {peak_idx}, expected near {expected_idx}"
        )

    @pytest.mark.parametrize("Nx", [128, 256, 512])
    def test_peak_location_various_grids(self, Nx):
        """Peak at Nx // 4 for different grid sizes."""
        space = _space(Nx=Nx)
        pulse = _pulse()
        time = _time()
        Ey, _, x = _seed_and_get_real(space, time, pulse)
        envelope = np.abs(Ey[:, 0, 0])
        peak_idx = np.argmax(envelope)
        expected_idx = Nx // 4
        assert abs(peak_idx - expected_idx) <= 2


# ──────────────────────────────────────────────────────────────────────────────
# SeedInitialCondition — Gaussian envelope shape
# ──────────────────────────────────────────────────────────────────────────────


class TestEnvelopeShape:
    """The spatial envelope is Gaussian with width sigma_x = v * tau_G."""

    def test_envelope_is_gaussian(self):
        """Envelope decays as exp(-xi^2 / sigma_x^2)."""
        space = _space(Nx=512)
        pulse = _pulse()
        time = _time()
        v = c0 / np.sqrt(space.epsr)
        tau_G = pulse.CalcTau()
        sigma_x = v * tau_G

        Ey, _, x = _seed_and_get_real(space, time, pulse)
        envelope = np.abs(Ey[:, 0, 0])
        x_center = x[space.Nx // 4]
        xi = x - x_center

        # Normalize to compare shape
        peak_val = np.max(envelope)
        if peak_val == 0:
            pytest.skip("Zero field, cannot test envelope shape")
        envelope_norm = envelope / peak_val
        expected_norm = np.exp(-(xi**2) / sigma_x**2)

        # Compare in the region where the envelope is significant (> 5% of peak)
        mask = expected_norm > 0.05
        np.testing.assert_allclose(
            envelope_norm[mask],
            expected_norm[mask],
            atol=0.05,
            err_msg="Envelope shape deviates from Gaussian",
        )

    def test_sigma_x_equals_v_times_tau(self):
        """1/e half-width of the spatial envelope matches v * tau_G."""
        space = _space(Nx=1024, dx=50e-9)
        pulse = _pulse()
        time = _time()
        v = c0 / np.sqrt(space.epsr)
        tau_G = pulse.CalcTau()
        sigma_x_expected = v * tau_G

        Ey, _, x = _seed_and_get_real(space, time, pulse)
        envelope = np.abs(Ey[:, 0, 0])
        peak_val = np.max(envelope)
        peak_idx = np.argmax(envelope)

        # Find where envelope drops to 1/e of peak on the right side
        target = peak_val / np.e
        right_half = envelope[peak_idx:]
        # Find first crossing below 1/e
        crossings = np.where(right_half < target)[0]
        if len(crossings) == 0:
            pytest.skip("Envelope does not decay to 1/e within grid")
        cross_idx = crossings[0] + peak_idx
        sigma_x_measured = abs(x[cross_idx] - x[peak_idx])

        # Allow 10% tolerance for discrete sampling
        np.testing.assert_allclose(
            sigma_x_measured,
            sigma_x_expected,
            rtol=0.10,
            err_msg="Spatial 1/e width does not match v * tau_G",
        )


# ──────────────────────────────────────────────────────────────────────────────
# SeedInitialCondition — carrier wavelength via FFT
# ──────────────────────────────────────────────────────────────────────────────


class TestCarrierWavelength:
    """FFT of the seeded field should peak at k_med = omega0 / v."""

    def test_spectral_peak_at_k_med(self):
        """Dominant spatial frequency corresponds to carrier wavelength."""
        Nx = 1024
        dx = 50e-9
        space = _space(Nx=Nx, dx=dx)
        pulse = _pulse()
        time = _time()
        v = c0 / np.sqrt(space.epsr)
        omega0 = pulse.CalcOmega0()
        k_med_expected = omega0 / v

        Ey, _, x = _seed_and_get_real(space, time, pulse)
        ey_line = Ey[:, 0, 0]

        # Spatial FFT to find dominant wavenumber
        Ek = np.fft.fft(ey_line)
        kx = np.fft.fftfreq(Nx, d=dx) * TWOPI
        power = np.abs(Ek) ** 2

        # Find peak in positive-k half (unidirectional pulse)
        pos_mask = kx > 0
        k_pos = kx[pos_mask]
        power_pos = power[pos_mask]
        k_peak = k_pos[np.argmax(power_pos)]

        # dk resolution = 2*pi / (Nx * dx)
        dk = TWOPI / (Nx * dx)
        assert abs(k_peak - k_med_expected) < 3 * dk, (
            f"Spectral peak at k = {k_peak:.4e}, "
            f"expected k_med = {k_med_expected:.4e} "
            f"(dk = {dk:.4e})"
        )

    @pytest.mark.parametrize("lam", [400e-9, 800e-9, 1500e-9])
    def test_carrier_scales_with_wavelength(self, lam):
        """Spectral peak shifts correctly with carrier wavelength."""
        Nx = 1024
        dx = 50e-9
        space = _space(Nx=Nx, dx=dx)
        pulse = _pulse(lam=lam)
        time = _time()
        v = c0 / np.sqrt(space.epsr)
        k_med_expected = TWOPI / lam  # in vacuum (epsr=1)

        Ey, _, _ = _seed_and_get_real(space, time, pulse)
        ey_line = Ey[:, 0, 0]

        Ek = np.fft.fft(ey_line)
        kx = np.fft.fftfreq(Nx, d=dx) * TWOPI
        power = np.abs(Ek) ** 2

        pos_mask = kx > 0
        k_peak = kx[pos_mask][np.argmax(power[pos_mask])]
        dk = TWOPI / (Nx * dx)

        assert abs(k_peak - k_med_expected) < 3 * dk


# ──────────────────────────────────────────────────────────────────────────────
# SeedInitialCondition — PML tapering
# ──────────────────────────────────────────────────────────────────────────────


class TestPMLTapering:
    """With npml_x > 0, field is tapered to near zero at boundaries."""

    def test_boundary_field_suppressed(self):
        """Field at the outermost cells should be near zero with PML."""
        npml = 16
        space = _space(Nx=256)
        pulse = _pulse()
        time = _time()
        Ey, Bz, _ = _seed_and_get_real(space, time, pulse, npml_x=npml)
        peak = np.max(np.abs(Ey))
        if peak == 0:
            pytest.skip("Zero field")

        # The outermost cell (index 0 and Nx-1) should be essentially zero
        # because taper[0] = 0.5*(1 - cos(0)) = 0
        left_edge = np.abs(Ey[0, 0, 0]) / peak
        right_edge = np.abs(Ey[-1, 0, 0]) / peak
        assert left_edge < 1e-6, f"Left boundary not suppressed: {left_edge}"
        assert right_edge < 1e-6, f"Right boundary not suppressed: {right_edge}"

    def test_pml_region_weaker_than_interior(self):
        """Average field in PML region should be much weaker than interior."""
        npml = 16
        space = _space(Nx=256)
        pulse = _pulse()
        time = _time()
        Ey, _, _ = _seed_and_get_real(space, time, pulse, npml_x=npml)

        env = np.abs(Ey[:, 0, 0])
        # The pulse is centered near Nx//4 = 64, so left PML (0:16) overlaps
        # the tail of the Gaussian anyway. Check the taper is applied
        # by comparing first cell to next interior cell.
        # taper[0] = 0, taper[npml] = 1
        assert env[0] < env[npml] or env[npml] < 1e-10

    def test_no_pml_leaves_boundaries_untouched(self):
        """Without PML, the boundary field depends only on the Gaussian."""
        space = _space(Nx=256)
        pulse = _pulse()
        time = _time()
        Ey_no_pml, _, _ = _seed_and_get_real(space, time, pulse, npml_x=0)
        Ey_pml, _, _ = _seed_and_get_real(space, time, pulse, npml_x=16)

        # With PML, boundaries should be more suppressed
        edge_no_pml = np.abs(Ey_no_pml[0, 0, 0])
        edge_pml = np.abs(Ey_pml[0, 0, 0])
        assert edge_pml <= edge_no_pml + 1e-15


# ──────────────────────────────────────────────────────────────────────────────
# SeedInitialCondition — unidirectional propagation
# ──────────────────────────────────────────────────────────────────────────────


class TestUnidirectionalPropagation:
    """Complex carrier exp(+ikx) ensures +k modes dominate over -k modes."""

    def test_positive_k_dominates(self):
        """Spectral power in +k should vastly exceed power in -k."""
        Nx = 512
        space = _space(Nx=Nx)
        pulse = _pulse()
        time = _time()

        Ey, _, _ = _seed_and_get_real(space, time, pulse)
        ey_line = Ey[:, 0, 0]

        Ek = np.fft.fft(ey_line)
        kx = np.fft.fftfreq(Nx, d=space.dx) * TWOPI
        power = np.abs(Ek) ** 2

        pos_power = np.sum(power[kx > 0])
        neg_power = np.sum(power[kx < 0])

        # For exp(+ikx), positive k should have > 99% of total power
        total = pos_power + neg_power
        if total == 0:
            pytest.skip("Zero field")
        ratio = pos_power / total
        assert ratio > 0.99, (
            f"Only {ratio * 100:.1f}% power in +k modes; "
            "expected > 99% for unidirectional propagation"
        )

    def test_not_a_standing_wave(self):
        """A standing wave (cos(kx)) would split power 50/50; ours does not."""
        Nx = 512
        space = _space(Nx=Nx)
        pulse = _pulse()
        time = _time()

        Ey, _, _ = _seed_and_get_real(space, time, pulse)
        ey_line = Ey[:, 0, 0]

        Ek = np.fft.fft(ey_line)
        kx = np.fft.fftfreq(Nx, d=space.dx) * TWOPI
        power = np.abs(Ek) ** 2

        pos_power = np.sum(power[kx > 0])
        neg_power = np.sum(power[kx < 0])

        # Standing wave would give ratio ~ 1.0; traveling wave gives ratio >> 1
        if neg_power == 0:
            return  # perfectly unidirectional, test passes
        asymmetry = pos_power / neg_power
        assert asymmetry > 100, (
            f"+k/-k power ratio = {asymmetry:.1f}; "
            "expected >> 1 for traveling wave (not standing wave)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# SeedInitialCondition — field amplitude and dtype
# ──────────────────────────────────────────────────────────────────────────────


class TestFieldBasics:
    """Basic sanity checks on seeded fields."""

    def test_fields_are_complex128(self):
        """Seeded fields should remain complex128 after FFT."""
        space = _space()
        Ey = _zeros()
        Bz = _zeros()
        SeedInitialCondition(space, _time(), _pulse(), Ey, Bz)
        assert Ey.dtype == np.complex128
        assert Bz.dtype == np.complex128

    def test_peak_amplitude_matches_emax(self):
        """Real-space peak of |Ey| should equal Amp (plane-wave, no chirp)."""
        space = _space(Nx=256)
        pulse = _pulse(chirp=0.0)
        time = _time()
        Ey, _, _ = _seed_and_get_real(space, time, pulse)
        peak = np.max(np.abs(Ey))
        # For plane wave (w0=inf) unchirped pulse, peak should be Amp
        np.testing.assert_allclose(peak, AMP, rtol=1e-10)

    def test_fields_not_all_zero(self):
        """Seeded fields should be non-trivial."""
        space = _space()
        Ey = _zeros()
        Bz = _zeros()
        SeedInitialCondition(space, _time(), _pulse(), Ey, Bz)
        assert np.any(Ey != 0), "Ey is all zeros after seeding"
        assert np.any(Bz != 0), "Bz is all zeros after seeding"

    def test_shape_preserved(self):
        """Seeding should not change array shape."""
        Nx, Ny, Nz = 64, 4, 4
        space = _space(Nx=Nx, Ny=Ny, Nz=Nz)
        Ey = _zeros(Nx, Ny, Nz)
        Bz = _zeros(Nx, Ny, Nz)
        SeedInitialCondition(space, _time(), _pulse(w0=1e-5), Ey, Bz)
        assert Ey.shape == (Nx, Ny, Nz)
        assert Bz.shape == (Nx, Ny, Nz)
