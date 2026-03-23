"""
Test suite for pulsesuite.PSTD3D.typepulse
============================================

Tests are written against **mathematical truth** (optics formulas), not
against the Fortran implementation.

Coverage
--------
- ps dataclass         — construction, fields, defaults (pol=0, w0=inf)
- CalcK0               — k₀ = 2π / λ
- CalcFreq0            — ν₀ = c₀ / λ
- CalcOmega0           — ω₀ = 2π c₀ / λ
- CalcTau              — τ = Tw / (2√ln2)
- CalcDeltaOmega       — δω = 0.44 / τ
- CalcTime_BandWidth   — TBP = τ · δω = 0.44
- CalcRayleigh         — z_R = π w₀² / λ
- CalcCurvature        — R(x) = x [1 + (z_R/x)²],  R(0) = ∞
- CalcGouyPhase        — φ_G = arctan(x / z_R)
- PulseFieldXT         — peak at (x=0, t=Tp), envelope decay, carrier frequency
- w0 field             — default inf, GetW0 / SetW0 accessors
- Module-level API     — Get*/Set*/Calc* wrappers
- File I/O             — WritePulseParams / ReadPulseParams round-trip
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as c0

from pulsesuite.PSTD3D.typepulse import (
    CalcDeltaOmega,
    CalcFreq0,
    CalcGouyPhase,
    CalcK0,
    CalcOmega0,
    CalcRayleigh,
    CalcTau,
    CalcTime_BandWidth,
    GetAmp,
    GetChirp,
    GetLambda,
    GetPol,
    GetTp,
    GetTw,
    GetW0,
    PulseFieldXT,
    SetAmp,
    SetChirp,
    SetLambda,
    SetTp,
    SetTw,
    SetW0,
    ps,
)

_twopi = 2.0 * np.pi


def _make_pulse(**overrides) -> ps:
    """Create a default 800 nm Ti:Sapphire-like pulse."""
    defaults = dict(
        lambda_=800e-9,
        Amp=1e8,
        Tw=30e-15,
        Tp=100e-15,
        chirp=0.0,
    )
    defaults.update(overrides)
    return ps(**defaults)


# ──────────────────────────────────────────────────────────────────────────────
# ps dataclass construction
# ──────────────────────────────────────────────────────────────────────────────


class TestPsConstruction:

    def test_fields_stored(self):
        p = _make_pulse()
        assert p.lambda_ == 800e-9
        assert p.Amp == 1e8
        assert p.Tw == 30e-15
        assert p.Tp == 100e-15
        assert p.chirp == 0.0

    def test_default_pol(self):
        p = _make_pulse()
        assert p.pol == 0

    def test_default_w0_infinite(self):
        p = _make_pulse()
        assert p.w0 == float("inf")

    def test_custom_w0(self):
        p = _make_pulse(w0=10e-6)
        assert p.w0 == 10e-6


# ──────────────────────────────────────────────────────────────────────────────
# Carrier wavenumber / frequency / angular frequency
# ──────────────────────────────────────────────────────────────────────────────


class TestCarrierQuantities:
    """k₀ = 2π/λ,  ν₀ = c₀/λ,  ω₀ = 2πc₀/λ."""

    @pytest.mark.parametrize("lam", [400e-9, 800e-9, 1550e-9, 10.6e-6])
    def test_calc_k0(self, lam):
        p = _make_pulse(lambda_=lam)
        expected = _twopi / lam
        assert np.isclose(p.CalcK0(), expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("lam", [400e-9, 800e-9, 1550e-9])
    def test_calc_freq0(self, lam):
        p = _make_pulse(lambda_=lam)
        expected = c0 / lam
        assert np.isclose(p.CalcFreq0(), expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("lam", [400e-9, 800e-9, 1550e-9])
    def test_calc_omega0(self, lam):
        p = _make_pulse(lambda_=lam)
        expected = _twopi * c0 / lam
        assert np.isclose(p.CalcOmega0(), expected, rtol=1e-12, atol=1e-12)

    def test_consistency_omega0_equals_twopi_freq0(self):
        p = _make_pulse()
        assert np.isclose(
            p.CalcOmega0(), _twopi * p.CalcFreq0(), rtol=1e-12, atol=1e-12
        )

    def test_module_wrappers(self):
        p = _make_pulse()
        assert CalcK0(p) == p.CalcK0()
        assert CalcFreq0(p) == p.CalcFreq0()
        assert CalcOmega0(p) == p.CalcOmega0()


# ──────────────────────────────────────────────────────────────────────────────
# CalcTau / CalcDeltaOmega / CalcTime_BandWidth
# ──────────────────────────────────────────────────────────────────────────────


class TestTemporalSpectral:
    r"""τ_G = Tw/√(2ln2),  Δω = √(8ln2(1+a²))/τ_G."""

    @pytest.mark.parametrize("Tw", [10e-15, 30e-15, 100e-15, 1e-12])
    def test_calc_tau(self, Tw):
        """CalcTau = Tw / sqrt(2*ln2) for exp(-t²/τ_G²) convention."""
        p = _make_pulse(Tw=Tw)
        expected = Tw / np.sqrt(2.0 * np.log(2.0))
        assert np.isclose(p.CalcTau(), expected, rtol=1e-12, atol=1e-12)

    def test_calc_delta_omega_unchirped(self):
        """For unchirped pulse: Δω = sqrt(8*ln2) / τ_G."""
        p = _make_pulse(chirp=0.0)
        tau_G = p.CalcTau()
        expected = np.sqrt(8.0 * np.log(2.0)) / tau_G
        assert np.isclose(
            p.CalcDeltaOmega(), expected, rtol=1e-12, atol=1e-12
        )

    def test_calc_delta_omega_chirped(self):
        """Chirp broadens the spectrum: Δω = sqrt(8*ln2*(1+a²)) / τ_G."""
        p = _make_pulse(chirp=1e28)
        tau_G = p.CalcTau()
        a = p.chirp * tau_G**2
        expected = np.sqrt(8.0 * np.log(2.0) * (1.0 + a**2)) / tau_G
        assert np.isclose(
            p.CalcDeltaOmega(), expected, rtol=1e-12, atol=1e-12
        )
        # Chirped should be broader than unchirped
        p_unchirped = _make_pulse(chirp=0.0, Tw=p.Tw)
        assert p.CalcDeltaOmega() > p_unchirped.CalcDeltaOmega()

    def test_time_bandwidth_product_unchirped(self):
        """Unchirped TBP = τ_G·Δω = sqrt(8*ln2) ≈ 2.355."""
        expected_tbp = np.sqrt(8.0 * np.log(2.0))
        for Tw in [10e-15, 30e-15, 100e-15, 1e-12]:
            p = _make_pulse(Tw=Tw, chirp=0.0)
            assert np.isclose(
                p.CalcTime_BandWidth(), expected_tbp, rtol=1e-12, atol=1e-12
            )

    def test_module_wrappers(self):
        p = _make_pulse()
        assert CalcTau(p) == p.CalcTau()
        assert CalcDeltaOmega(p) == p.CalcDeltaOmega()
        assert CalcTime_BandWidth(p) == p.CalcTime_BandWidth()


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian beam: Rayleigh range, curvature, Gouy phase
# ──────────────────────────────────────────────────────────────────────────────


class TestGaussianBeam:
    r"""z_R = πw₀²/λ,  R(x) = x[1+(z_R/x)²],  φ_G = arctan(x/z_R)."""

    @pytest.mark.parametrize("w0,lam", [
        (10e-6, 800e-9),
        (50e-6, 1550e-9),
        (1e-3, 10.6e-6),
    ])
    def test_rayleigh_range(self, w0, lam):
        p = _make_pulse(lambda_=lam)
        expected = np.pi * w0**2 / lam
        assert np.isclose(
            p.CalcRayleigh(w0), expected, rtol=1e-12, atol=1e-12
        )

    def test_curvature_at_waist_is_inf(self):
        p = _make_pulse()
        assert p.CalcCurvature(0.0, 10e-6) == float("inf")

    def test_curvature_far_field_approaches_x(self):
        """For x >> z_R, R(x) → x."""
        p = _make_pulse(lambda_=800e-9)
        w0 = 10e-6
        z_R = p.CalcRayleigh(w0)
        x_far = 1000 * z_R
        R = p.CalcCurvature(x_far, w0)
        assert np.isclose(R / x_far, 1.0, rtol=1e-6, atol=1e-12)

    def test_curvature_at_rayleigh_range(self):
        """R(z_R) = 2·z_R."""
        p = _make_pulse(lambda_=800e-9)
        w0 = 10e-6
        z_R = p.CalcRayleigh(w0)
        R = p.CalcCurvature(z_R, w0)
        assert np.isclose(R, 2.0 * z_R, rtol=1e-12, atol=1e-12)

    def test_gouy_phase_at_zero(self):
        p = _make_pulse()
        assert np.isclose(
            p.CalcGouyPhase(0.0, 10e-6), 0.0, rtol=1e-12, atol=1e-12
        )

    def test_gouy_phase_at_rayleigh_is_pi_over_4(self):
        p = _make_pulse(lambda_=800e-9)
        w0 = 10e-6
        z_R = p.CalcRayleigh(w0)
        assert np.isclose(
            p.CalcGouyPhase(z_R, w0), np.pi / 4.0, rtol=1e-12, atol=1e-12
        )

    def test_module_wrappers(self):
        p = _make_pulse(lambda_=800e-9)
        w0 = 50e-6
        assert CalcRayleigh(p, w0) == p.CalcRayleigh(w0)
        assert CalcGouyPhase(p, 1.0, w0) == p.CalcGouyPhase(1.0, w0)


# ──────────────────────────────────────────────────────────────────────────────
# PulseFieldXT
# ──────────────────────────────────────────────────────────────────────────────


class TestPulseFieldXT:
    r"""E(x,t) = Amp·exp(-τ²/τ_G²)·exp(i(ω₀τ + χτ²)),  τ = t - Tp - x/c₀."""

    def test_peak_amplitude_at_origin(self):
        """At x=0, t=Tp, τ=0: E = Amp."""
        p = _make_pulse(chirp=0.0)
        E = p.PulseFieldXT(0.0, p.Tp)
        assert np.isclose(abs(E), p.Amp, rtol=1e-12, atol=1e-12)

    def test_envelope_decay_away_from_peak(self):
        """Far from the peak, |E| < Amp."""
        p = _make_pulse()
        E_far = p.PulseFieldXT(0.0, p.Tp + 10 * p.Tw)
        assert abs(E_far) < 1e-10 * p.Amp

    def test_unchirped_carrier_real_at_peak(self):
        """With chirp=0 at τ=0, phase=0 → E is real and positive."""
        p = _make_pulse(chirp=0.0)
        E = p.PulseFieldXT(0.0, p.Tp)
        assert np.isclose(E.real, p.Amp, rtol=1e-12, atol=1e-12)
        assert np.isclose(E.imag, 0.0, rtol=1e-12, atol=1e-12)

    def test_retarded_time_shift(self):
        """Pulse at x=d arrives delayed by d/c₀."""
        p = _make_pulse(chirp=0.0)
        d = 1e-3  # 1 mm
        t_delayed = p.Tp + d / c0
        E = p.PulseFieldXT(d, t_delayed)
        assert np.isclose(abs(E), p.Amp, rtol=1e-10, atol=1e-12)

    def test_chirped_pulse_peak_still_at_tp(self):
        """Chirp doesn't shift the envelope peak."""
        p = _make_pulse(chirp=1e26)
        E = p.PulseFieldXT(0.0, p.Tp)
        assert np.isclose(abs(E), p.Amp, rtol=1e-12, atol=1e-12)

    def test_module_level_wrapper_arg_order(self):
        """Module-level PulseFieldXT(x, t, pulse) — pulse is last."""
        p = _make_pulse()
        x, t = 0.0, p.Tp
        assert PulseFieldXT(x, t, p) == p.PulseFieldXT(x, t)


# ──────────────────────────────────────────────────────────────────────────────
# w0 field + GetW0 / SetW0
# ──────────────────────────────────────────────────────────────────────────────


class TestW0:
    """w0 beam waist: default inf (plane wave), GetW0, SetW0."""

    def test_default_is_inf(self):
        p = _make_pulse()
        assert GetW0(p) == float("inf")

    def test_set_w0(self):
        p = _make_pulse()
        SetW0(p, 25e-6)
        assert GetW0(p) == 25e-6

    def test_construction_with_w0(self):
        p = ps(lambda_=800e-9, Amp=1e8, Tw=30e-15, Tp=100e-15, chirp=0.0, w0=5e-6)
        assert p.w0 == 5e-6


# ──────────────────────────────────────────────────────────────────────────────
# Module-level Get/Set accessors
# ──────────────────────────────────────────────────────────────────────────────


class TestModuleLevelAccessors:

    def test_get_lambda(self):
        p = _make_pulse(lambda_=1064e-9)
        assert GetLambda(p) == 1064e-9

    def test_get_amp(self):
        p = _make_pulse(Amp=5e7)
        assert GetAmp(p) == 5e7

    def test_get_tw(self):
        p = _make_pulse(Tw=50e-15)
        assert GetTw(p) == 50e-15

    def test_get_tp(self):
        p = _make_pulse(Tp=200e-15)
        assert GetTp(p) == 200e-15

    def test_get_chirp(self):
        p = _make_pulse(chirp=1e25)
        assert GetChirp(p) == 1e25

    def test_get_pol(self):
        p = _make_pulse()
        assert GetPol(p) == 0

    def test_set_lambda(self):
        p = _make_pulse()
        SetLambda(p, 400e-9)
        assert p.lambda_ == 400e-9

    def test_set_amp(self):
        p = _make_pulse()
        SetAmp(p, 2e8)
        assert p.Amp == 2e8

    def test_set_tw(self):
        p = _make_pulse()
        SetTw(p, 15e-15)
        assert p.Tw == 15e-15

    def test_set_tp(self):
        p = _make_pulse()
        SetTp(p, 50e-15)
        assert p.Tp == 50e-15

    def test_set_chirp(self):
        p = _make_pulse()
        SetChirp(p, 5e25)
        assert p.chirp == 5e25


# ──────────────────────────────────────────────────────────────────────────────
# File I/O round-trip
# ──────────────────────────────────────────────────────────────────────────────


class TestFileIO:

    def test_write_read_round_trip(self):
        original = _make_pulse(lambda_=1550e-9, Amp=3e7, Tw=50e-15,
                               Tp=200e-15, chirp=1e24)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pulse.params")
            from pulsesuite.PSTD3D.typepulse import (
                WritePulseParams,
                readpulseparams_sub,
            )
            WritePulseParams(path, original)

            loaded = _make_pulse()  # dummy to overwrite
            with open(path, "r") as f:
                readpulseparams_sub(f, loaded)

            assert np.isclose(loaded.lambda_, original.lambda_, rtol=1e-12, atol=1e-12)
            assert np.isclose(loaded.Amp, original.Amp, rtol=1e-12, atol=1e-12)
            assert np.isclose(loaded.Tw, original.Tw, rtol=1e-12, atol=1e-12)
            assert np.isclose(loaded.Tp, original.Tp, rtol=1e-12, atol=1e-12)
            assert np.isclose(loaded.chirp, original.chirp, rtol=1e-12, atol=1e-12)

    def test_round_trip_preserves_small_wavelength(self):
        """EUV / X-ray wavelength (~13 nm)."""
        original = _make_pulse(lambda_=13.5e-9)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pulse.params")
            from pulsesuite.PSTD3D.typepulse import (
                WritePulseParams,
                readpulseparams_sub,
            )
            WritePulseParams(path, original)
            loaded = _make_pulse()
            with open(path, "r") as f:
                readpulseparams_sub(f, loaded)
            assert np.isclose(loaded.lambda_, 13.5e-9, rtol=1e-10, atol=1e-25)
