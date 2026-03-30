"""
Test suite for pulsesuite.PSTD3D.absorber
==========================================

Tests are written against **mathematical truth** (mask properties,
unconditional stability, energy decay), not against the Fortran output.

Coverage
--------
- CalcNPML           — PML thickness computation for various grid sizes
- _build_1d_mask     — 1-D mask shape, range, symmetry, monotonicity
- InitAbsorber       — 3-D mask construction, separability
- ApplyAbsorber_E/B  — field attenuation, interior preservation
- GetAbsorberInfo    — npml query interface
"""

import numpy as np
import pytest

from pulsesuite.PSTD3D.absorber import (
    CalcNPML,
    GetAbsorberInfo,
    InitAbsorber,
    _build_1d_mask,
)

# ──────────────────────────────────────────────────────────────────────────────
# CalcNPML
# ──────────────────────────────────────────────────────────────────────────────


class TestCalcNPML:
    """PML thickness: max(6, 5% of N), with safeguards for small grids."""

    def test_small_grid_no_pml(self):
        """Axes with N <= 4 get no absorption (1D-like mode)."""
        assert CalcNPML(1) == 0
        assert CalcNPML(4) == 0

    def test_minimum_six_cells(self):
        """PML is at least 6 cells when N is large enough that 6 < N/10."""
        # N=128: 5% = 6, and 6 <= 128/10 = 12, so npml = 6
        assert CalcNPML(128) >= 6

    def test_five_percent_of_grid(self):
        """For large grids, PML ~ 5% of N."""
        npml = CalcNPML(2048)
        assert npml == max(6, int(0.05 * 2048))

    @pytest.mark.parametrize("N", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    def test_pml_never_exceeds_half_grid(self, N):
        """2 * npml < N: PML can't consume the entire grid."""
        npml = CalcNPML(N)
        assert 2 * npml < N

    def test_cap_at_ten_percent(self):
        """PML capped at N/10 to leave enough interior cells."""
        # For N=64: 5% = 3 (too small, floor at 6),
        # but 6 < 64/10 = 6, so it's fine
        npml = CalcNPML(64)
        assert npml <= 64 // 10 or npml == 6


# ──────────────────────────────────────────────────────────────────────────────
# _build_1d_mask
# ──────────────────────────────────────────────────────────────────────────────


class TestBuild1DMask:
    """1-D mask profile: interior = 1, boundary < 1, monotonic ramp."""

    def test_no_pml_all_ones(self):
        """With npml=0, entire mask is 1.0."""
        mask = _build_1d_mask(32, 0, 1e-18)
        assert np.allclose(mask, 1.0, rtol=1e-12, atol=1e-12)

    def test_shape_and_dtype(self):
        mask = _build_1d_mask(64, 6, 1e-18)
        assert mask.shape == (64,)
        assert mask.dtype == np.float64

    def test_range_zero_to_one(self):
        """All mask values in [0, 1]."""
        mask = _build_1d_mask(128, 10, 1e-18)
        assert np.all(mask >= 0.0)
        assert np.all(mask <= 1.0)

    def test_interior_is_one(self):
        """Interior cells (between PML layers) have mask = 1."""
        npml = 10
        mask = _build_1d_mask(128, npml, 1e-18)
        interior = mask[npml:-npml]
        assert np.allclose(interior, 1.0, rtol=1e-12, atol=1e-12)

    def test_boundary_less_than_one(self):
        """Outermost PML cells have mask < 1."""
        npml = 10
        mask = _build_1d_mask(128, npml, 1e-18)
        assert mask[0] < 1.0
        assert mask[-1] < 1.0

    def test_left_ramp_monotonic(self):
        """Left PML: mask increases monotonically toward interior."""
        npml = 10
        mask = _build_1d_mask(128, npml, 1e-18)
        left = mask[:npml]
        # Each cell should be <= the next (toward interior)
        assert np.all(np.diff(left) >= 0)

    def test_right_ramp_monotonic(self):
        """Right PML: mask decreases monotonically toward boundary."""
        npml = 10
        mask = _build_1d_mask(128, npml, 1e-18)
        right = mask[-npml:]
        assert np.all(np.diff(right) <= 0)

    def test_mask_stable_for_any_dt(self):
        """Mask values in [0,1] regardless of dt — unconditionally stable."""
        for dt in [1e-21, 1e-18, 1e-15, 1e-12, 1.0]:
            mask = _build_1d_mask(64, 6, dt)
            assert np.all(mask >= 0.0)
            assert np.all(mask <= 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# InitAbsorber + GetAbsorberInfo
# ──────────────────────────────────────────────────────────────────────────────


class TestInitAbsorber:
    """3-D mask construction and module-state initialisation."""

    def test_init_sets_info(self):
        InitAbsorber(128, 4, 4, 1e-18)
        npx, npy, npz = GetAbsorberInfo()
        assert npx > 0
        assert npy == 0  # N <= 4 → no PML
        assert npz == 0

    def test_mask_shape(self):
        from pulsesuite.PSTD3D.absorber import _state

        InitAbsorber(64, 16, 8, 1e-18)
        mask = _state["mask"]
        assert mask.shape == (64, 16, 8)

    def test_mask_range(self):
        from pulsesuite.PSTD3D.absorber import _state

        InitAbsorber(64, 16, 8, 1e-18)
        mask = _state["mask"]
        assert np.all(mask >= 0.0)
        assert np.all(mask <= 1.0)

    def test_mask_interior_is_one(self):
        """Interior region of 3D mask should be 1.0."""
        from pulsesuite.PSTD3D.absorber import _state

        Nx, Ny, Nz = 64, 16, 8
        InitAbsorber(Nx, Ny, Nz, 1e-18)
        mask = _state["mask"]

        npx, npy, npz = GetAbsorberInfo()
        if npx > 0 and npy > 0 and npz > 0:
            interior = mask[npx:-npx, npy:-npy, npz:-npz]
            assert np.allclose(interior, 1.0, rtol=1e-12, atol=1e-12)

    def test_separability(self):
        """3-D mask should be separable: mask = mask_x * mask_y * mask_z."""
        from pulsesuite.PSTD3D.absorber import _state

        Nx, Ny, Nz = 32, 16, 8
        dt = 1e-18
        InitAbsorber(Nx, Ny, Nz, dt)
        mask_3d = _state["mask"]

        # Extract 1-D profiles along each axis (at interior positions)
        mid_y, mid_z = Ny // 2, Nz // 2
        mask_x_profile = mask_3d[:, mid_y, mid_z]

        mid_x = Nx // 2
        mask_y_profile = mask_3d[mid_x, :, mid_z]
        mask_z_profile = mask_3d[mid_x, mid_y, :]

        # Reconstruct 3D from outer product
        reconstructed = (
            mask_x_profile[:, np.newaxis, np.newaxis]
            * mask_y_profile[np.newaxis, :, np.newaxis]
            * mask_z_profile[np.newaxis, np.newaxis, :]
        )
        assert np.allclose(mask_3d, reconstructed, rtol=1e-12, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# ApplyAbsorber_E / ApplyAbsorber_B
# ──────────────────────────────────────────────────────────────────────────────


class TestApplyAbsorber:
    """Absorber application: attenuates boundaries, preserves interior."""

    @pytest.fixture(autouse=True)
    def _init_absorber(self):
        """Initialise absorber before each test."""
        self.Nx, self.Ny, self.Nz = 32, 4, 4
        self.dt = 1e-18
        InitAbsorber(self.Nx, self.Ny, self.Nz, self.dt)

    def _make_uniform_kspace(self, val=1.0):
        """Create a uniform k-space field (DC mode only)."""
        shape = (self.Nx, self.Ny, self.Nz)
        F = np.zeros(shape, dtype=np.complex128)
        F[0, 0, 0] = val * self.Nx * self.Ny * self.Nz
        return F

    def test_apply_e_attenuates(self):
        """After applying absorber, field energy should decrease."""
        from pulsesuite.PSTD3D.absorber import ApplyAbsorber_E

        Ex = self._make_uniform_kspace(1.0)
        Ey = self._make_uniform_kspace(1.0)
        Ez = self._make_uniform_kspace(1.0)

        energy_before = (
            np.sum(np.abs(Ex) ** 2) + np.sum(np.abs(Ey) ** 2) + np.sum(np.abs(Ez) ** 2)
        )

        ApplyAbsorber_E(Ex, Ey, Ez)

        energy_after = (
            np.sum(np.abs(Ex) ** 2) + np.sum(np.abs(Ey) ** 2) + np.sum(np.abs(Ez) ** 2)
        )

        # Energy should decrease (absorber removes energy at boundaries)
        assert energy_after < energy_before

    def test_apply_b_attenuates(self):
        """After applying absorber, B field energy should decrease."""
        from pulsesuite.PSTD3D.absorber import ApplyAbsorber_B

        Bx = self._make_uniform_kspace(1.0)
        By = self._make_uniform_kspace(1.0)
        Bz = self._make_uniform_kspace(1.0)

        energy_before = (
            np.sum(np.abs(Bx) ** 2) + np.sum(np.abs(By) ** 2) + np.sum(np.abs(Bz) ** 2)
        )

        ApplyAbsorber_B(Bx, By, Bz)

        energy_after = (
            np.sum(np.abs(Bx) ** 2) + np.sum(np.abs(By) ** 2) + np.sum(np.abs(Bz) ** 2)
        )

        assert energy_after < energy_before

    def test_unconditional_stability(self):
        """Repeated application should only decrease field — never grow."""
        from pulsesuite.PSTD3D.absorber import ApplyAbsorber_E

        Ex = self._make_uniform_kspace(1.0)
        Ey = self._make_uniform_kspace(1.0)
        Ez = self._make_uniform_kspace(1.0)

        prev_energy = float("inf")
        for _ in range(20):
            ApplyAbsorber_E(Ex, Ey, Ez)
            energy = (
                np.sum(np.abs(Ex) ** 2)
                + np.sum(np.abs(Ey) ** 2)
                + np.sum(np.abs(Ez) ** 2)
            )
            assert energy <= prev_energy * (1.0 + 1e-10)
            prev_energy = energy
