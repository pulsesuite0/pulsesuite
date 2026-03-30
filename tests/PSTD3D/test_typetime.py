"""
Test suite for pulsesuite.PSTD3D.typetime
==========================================

Tests are written against **mathematical truth**, not against the Fortran
implementation.

Coverage
--------
- ts dataclass         — construction, field storage
- CalcNt               — floor((tf - t) / dt), edge cases
- UpdateT / UpdateN    — in-place mutation
- GetTArray            — t[i] = t + i·dt,  special Nt=1 case
- GetOmegaArray        — matches numpy.fft.fftfreq * 2π
- GetdOmega            — 2π / (Nt·dt)
- Module-level API     — Get*/Set* accessors, CalcNt, UpdateT, UpdateN wrappers
- File I/O             — ReadTimeParams / writetimeparams round-trip
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pulsesuite.PSTD3D.typetime import (
    CalcNt,
    GetdOmega,
    GetDt,
    GetN,
    GetOmegaArray,
    GetT,
    GetTArray,
    GetTf,
    SetDt,
    SetN,
    SetT,
    SetTf,
    UpdateN,
    UpdateT,
    ts,
    writetimeparams,
)

_twopi = 2.0 * np.pi


# ──────────────────────────────────────────────────────────────────────────────
# ts dataclass construction
# ──────────────────────────────────────────────────────────────────────────────


class TestTsConstruction:
    """Verify dataclass field storage and access."""

    def test_fields_stored_correctly(self):
        time = ts(t=1.0, tf=10.0, dt=0.5, n=0)
        assert time.t == 1.0
        assert time.tf == 10.0
        assert time.dt == 0.5
        assert time.n == 0

    def test_fields_mutable(self):
        time = ts(t=0.0, tf=1.0, dt=0.1, n=0)
        time.t = 5.0
        assert time.t == 5.0


# ──────────────────────────────────────────────────────────────────────────────
# CalcNt
# ──────────────────────────────────────────────────────────────────────────────


class TestCalcNt:
    """CalcNt: floor((tf - t) / dt)."""

    def test_exact_division(self):
        time = ts(t=0.0, tf=1.0, dt=0.1, n=0)
        assert time.CalcNt() == 10

    def test_non_exact_division_floors(self):
        time = ts(t=0.0, tf=1.0, dt=0.3, n=0)
        # (1.0 - 0.0) / 0.3 = 3.333... -> floor = 3
        assert time.CalcNt() == 3

    def test_single_step(self):
        time = ts(t=0.0, tf=0.5, dt=0.5, n=0)
        assert time.CalcNt() == 1

    def test_zero_remaining(self):
        time = ts(t=1.0, tf=1.0, dt=0.1, n=0)
        assert time.CalcNt() == 0

    def test_module_level_wrapper(self):
        time = ts(t=0.0, tf=2.0, dt=0.25, n=0)
        assert CalcNt(time) == time.CalcNt()

    @pytest.mark.parametrize(
        "tf,dt,expected",
        [
            (1e-12, 1e-14, 100),
            (2.0, 0.002, 1000),
            (5.0, 0.01, 500),
        ],
    )
    def test_various_scales(self, tf, dt, expected):
        time = ts(t=0.0, tf=tf, dt=dt, n=0)
        assert time.CalcNt() == expected


# ──────────────────────────────────────────────────────────────────────────────
# UpdateT / UpdateN
# ──────────────────────────────────────────────────────────────────────────────


class TestUpdateT:
    """UpdateT: t ← t + dt (in-place)."""

    def test_single_advance(self):
        time = ts(t=0.0, tf=10.0, dt=0.5, n=0)
        time.UpdateT(0.5)
        assert np.isclose(time.t, 0.5, rtol=1e-12, atol=1e-12)

    def test_multiple_advances(self):
        time = ts(t=0.0, tf=10.0, dt=0.1, n=0)
        for _ in range(10):
            time.UpdateT(0.1)
        assert np.isclose(time.t, 1.0, rtol=1e-12, atol=1e-12)

    def test_half_step(self):
        time = ts(t=0.0, tf=10.0, dt=1.0, n=0)
        time.UpdateT(0.5)
        assert np.isclose(time.t, 0.5, rtol=1e-12, atol=1e-12)

    def test_module_level_wrapper(self):
        time = ts(t=0.0, tf=10.0, dt=1.0, n=0)
        UpdateT(time, 2.0)
        assert np.isclose(time.t, 2.0, rtol=1e-12, atol=1e-12)


class TestUpdateN:
    """UpdateN: n ← n + dn (in-place)."""

    def test_increment_by_one(self):
        time = ts(t=0.0, tf=10.0, dt=1.0, n=0)
        time.UpdateN(1)
        assert time.n == 1

    def test_increment_by_arbitrary(self):
        time = ts(t=0.0, tf=10.0, dt=1.0, n=5)
        time.UpdateN(3)
        assert time.n == 8

    def test_module_level_wrapper(self):
        time = ts(t=0.0, tf=10.0, dt=1.0, n=0)
        UpdateN(time, 7)
        assert time.n == 7


# ──────────────────────────────────────────────────────────────────────────────
# GetTArray
# ──────────────────────────────────────────────────────────────────────────────


class TestGetTArray:
    """GetTArray: t[i] = t + i·dt for i = 0…Nt-1."""

    def test_values(self):
        time = ts(t=0.0, tf=1.0, dt=0.25, n=0)
        tarr = time.GetTArray()
        expected = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)
        assert np.allclose(tarr, expected, rtol=1e-12, atol=1e-12)

    def test_length_matches_calcnt(self):
        time = ts(t=0.0, tf=2.0, dt=0.1, n=0)
        assert len(time.GetTArray()) == time.CalcNt()

    def test_starts_at_current_time(self):
        time = ts(t=5.0, tf=6.0, dt=0.2, n=0)
        tarr = time.GetTArray()
        assert np.isclose(tarr[0], 5.0, rtol=1e-12, atol=1e-12)

    def test_nt_equals_one_returns_zero(self):
        """Special case: CalcNt=1 → returns [0.0] (mirrors Fortran)."""
        time = ts(t=0.0, tf=0.1, dt=0.1, n=0)
        assert time.CalcNt() == 1
        tarr = time.GetTArray()
        assert len(tarr) == 1
        assert tarr[0] == 0.0

    def test_dtype_float64(self):
        time = ts(t=0.0, tf=1.0, dt=0.1, n=0)
        assert time.GetTArray().dtype == np.float64

    def test_module_level_wrapper(self):
        time = ts(t=0.0, tf=1.0, dt=0.25, n=0)
        assert np.allclose(GetTArray(time), time.GetTArray(), rtol=1e-12, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# GetOmegaArray
# ──────────────────────────────────────────────────────────────────────────────


class TestGetOmegaArray:
    """GetOmegaArray: matches numpy.fft.fftfreq(Nt, d=dt) * 2π."""

    @pytest.mark.parametrize(
        "Nt,dt",
        [
            (16, 1e-15),
            (64, 0.5e-14),
            (100, 1e-12),
            (128, 1.0),
        ],
    )
    def test_matches_numpy_fftfreq(self, Nt, dt):
        tf = Nt * dt
        time = ts(t=0.0, tf=tf, dt=dt, n=0)
        omega = time.GetOmegaArray()
        expected = np.fft.fftfreq(Nt, d=dt).astype(np.float64) * _twopi
        assert np.allclose(omega, expected, rtol=1e-12, atol=1e-12)

    def test_length(self):
        time = ts(t=0.0, tf=1.0, dt=0.01, n=0)
        assert len(time.GetOmegaArray()) == time.CalcNt()

    def test_dc_component_zero(self):
        time = ts(t=0.0, tf=1.0, dt=0.1, n=0)
        omega = time.GetOmegaArray()
        assert omega[0] == 0.0

    def test_dtype_float64(self):
        time = ts(t=0.0, tf=1.0, dt=0.1, n=0)
        assert time.GetOmegaArray().dtype == np.float64

    def test_module_level_wrapper(self):
        time = ts(t=0.0, tf=1.0, dt=0.1, n=0)
        assert np.allclose(
            GetOmegaArray(time), time.GetOmegaArray(), rtol=1e-12, atol=1e-12
        )


# ──────────────────────────────────────────────────────────────────────────────
# GetdOmega
# ──────────────────────────────────────────────────────────────────────────────


class TestGetdOmega:
    """GetdOmega: 2π / (Nt·dt)."""

    def test_formula(self):
        time = ts(t=0.0, tf=1.0, dt=0.01, n=0)
        Nt = time.CalcNt()
        expected = _twopi / (Nt * time.dt)
        assert np.isclose(time.GetdOmega(), expected, rtol=1e-12, atol=1e-12)

    def test_module_level_wrapper(self):
        time = ts(t=0.0, tf=2.0, dt=0.1, n=0)
        assert np.isclose(GetdOmega(time), time.GetdOmega(), rtol=1e-12, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level Get/Set accessors
# ──────────────────────────────────────────────────────────────────────────────


class TestModuleLevelAccessors:
    """Trivial Get*/Set* wrappers delegate to dataclass fields."""

    def test_get_t(self):
        time = ts(t=3.14, tf=10.0, dt=0.1, n=0)
        assert GetT(time) == 3.14

    def test_get_tf(self):
        time = ts(t=0.0, tf=99.0, dt=0.1, n=0)
        assert GetTf(time) == 99.0

    def test_get_dt(self):
        time = ts(t=0.0, tf=1.0, dt=0.05, n=0)
        assert GetDt(time) == 0.05

    def test_get_n(self):
        time = ts(t=0.0, tf=1.0, dt=0.1, n=42)
        assert GetN(time) == 42

    def test_set_t(self):
        time = ts(t=0.0, tf=1.0, dt=0.1, n=0)
        SetT(time, 7.0)
        assert time.t == 7.0

    def test_set_tf(self):
        time = ts(t=0.0, tf=1.0, dt=0.1, n=0)
        SetTf(time, 20.0)
        assert time.tf == 20.0

    def test_set_dt(self):
        time = ts(t=0.0, tf=1.0, dt=0.1, n=0)
        SetDt(time, 0.001)
        assert time.dt == 0.001

    def test_set_n(self):
        time = ts(t=0.0, tf=1.0, dt=0.1, n=0)
        SetN(time, 100)
        assert time.n == 100


# ──────────────────────────────────────────────────────────────────────────────
# File I/O round-trip
# ──────────────────────────────────────────────────────────────────────────────


class TestFileIO:
    """ReadTimeParams / writetimeparams round-trip preserves values."""

    def test_write_read_round_trip(self):
        original = ts(t=1.5e-12, tf=5.0e-12, dt=1.0e-15, n=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "time.params")
            writetimeparams(path, original)

            loaded = ts(t=0.0, tf=0.0, dt=0.0, n=0)
            # ReadTimeParams prints to stdout — suppress by reading manually
            from pulsesuite.PSTD3D.typetime import readtimeparams_sub

            with open(path, "r") as f:
                readtimeparams_sub(f, loaded)

            assert np.isclose(loaded.t, original.t, rtol=1e-12, atol=1e-12)
            assert np.isclose(loaded.tf, original.tf, rtol=1e-12, atol=1e-12)
            assert np.isclose(loaded.dt, original.dt, rtol=1e-12, atol=1e-12)
            assert loaded.n == original.n

    def test_write_read_preserves_femtosecond_precision(self):
        """Typical ultrafast simulation parameters (fs-scale dt)."""
        original = ts(t=0.0, tf=300e-15, dt=0.1e-15, n=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "time.params")
            writetimeparams(path, original)

            loaded = ts(t=0.0, tf=0.0, dt=0.0, n=0)
            from pulsesuite.PSTD3D.typetime import readtimeparams_sub

            with open(path, "r") as f:
                readtimeparams_sub(f, loaded)

            assert np.isclose(loaded.dt, original.dt, rtol=1e-10, atol=1e-30)
