# Examples CHANGELOG

Tracks the status and revision history of all `.myst.md` example files.

**Reference standard**: The simulation methodology and parameter conventions established in
Gulley & Huang's published work (see [References](#references) below). This codebase is a
Python port of Gulley's original Fortran simulation tools.

---

## Current Status (v2 -- restructured examples)

| Example | Status | Uses Real Data | Executes | Citations |
|---------|--------|----------------|----------|-----------|
| `01_quickstart.myst.md` | New | Yes (full SBE run) | Pending verification | Yes |
| `02_architecture.myst.md` | New | Yes (InitializeSBE) | Pending verification | Yes |
| `03_coulomb.myst.md` | New | Yes (real Coulomb matrices) | Pending verification | Yes |
| `04_phonons.myst.md` | New | Yes (real phonon matrices) | Pending verification | Yes |
| `05_optics.myst.md` | New | Yes (real QWOptics data) | Pending verification | Yes |
| `06_transport.myst.md` | New | Yes (real DC/emission data) | Pending verification | Yes |

### What changed in v2

All 6 original `*_example.myst.md` files were **replaced** with new examples that:

1. **Use real computed data** -- every example calls `InitializeSBE()` which properly
   initializes all subsystems (Coulomb, phonons, dephasing, transport, emission).
   No more fabricated arrays (`np.zeros`, `np.eye * 1e-20`, `np.random.randn`).

2. **Follow the actual workflow** -- based on the working `sbetestprop.py` test script.
   The quickstart runs a real 3000-step SBE simulation with an 800 nm pulse.

3. **Include citations** -- all examples reference Gulley & Huang (2019, 2022).

4. **Are structured for onboarding** -- numbered 01-06 with a clear reading order:
   quickstart -> architecture -> per-module deep-dives.

5. **Preserve the best content** from the old files:
   - Mermaid integration diagram (from `sbes_example`)
   - Parameter reference tables (from `sbes_example`)
   - SBE theory equations (from `sbes_example`)
   - Coulomb physics explanations (from `coulomb_example`)
   - Phonon theory and assumptions (from `phonons_example`)
   - QW optics workflow and chi1 calculation (from `qwoptics_example`)
   - DC transport physics (from `dcfield_example`)
   - Emission/PL theory (from `emission_example`)

### Issues resolved from v1

- [x] **Never executed**: All new examples are designed to execute during doc builds
- [x] **Private variable mutation**: Minimized; uses `InitializeSBE()` for proper setup
- [x] **No citations**: All reference Gulley & Huang papers
- [x] **Isolated modules**: Every example initializes through the proper SBE orchestrator
- [x] **No physical observables**: Quickstart shows E-field/polarization; optics shows chi1 spectrum
- [x] **Broken references**: `sbes_example` (undefined `L`, missing `SBEs.` prefix) removed
- [x] **Mock data**: `coulomb_example` try/except fallbacks removed entirely
- [x] **Too long**: `sbes_example` (1555 lines) split into quickstart + architecture

---

## Previous Status (v1 -- original examples, now removed)

| Example | Status | Grade |
|---------|--------|-------|
| `dcfield_example.myst.md` | Removed (replaced by `06_transport`) | C+ |
| `emission_example.myst.md` | Removed (replaced by `06_transport`) | C |
| `phonons_example.myst.md` | Removed (replaced by `04_phonons`) | B- |
| `qwoptics_example.myst.md` | Removed (replaced by `05_optics`) | C+ |
| `sbes_example.myst.md` | Removed (replaced by `01_quickstart` + `02_architecture`) | C- |
| `coulomb_example.myst.md` | Removed (replaced by `03_coulomb`) | D+ |

---

## References

These papers define the physics, parameters, and methodology that pulsesuite implements:

1. J. R. Gulley and D. Huang, "Self-consistent quantum-kinetic theory for interplay between
   pulsed-laser excitation and nonlinear carrier transport in a quantum-wire array,"
   *Opt. Express* **27**, 17154-17185 (2019).
   https://doi.org/10.1364/OE.27.017154

2. J. R. Gulley and D. Huang, "Ultrafast transverse and longitudinal response of
   laser-excited quantum wires," *Opt. Express* **30**(6), 9348-9359 (2022).
   https://doi.org/10.1364/OE.448934

3. J. R. Gulley, D. Huang, and E. Winchester, "Ultrashort-laser pulse induced plasma
   dynamics in semiconductor nanowires," *Proc. SPIE* 12884, 128840G (2024).
   https://doi.org/10.1117/12.3003739

4. J. R. Gulley, R. Cooper, and E. Winchester, "Mobility and conductivity of
   laser-generated e-h plasmas in direct-gap nanowires,"
   *Photonics Nanostructures: Fundam. Appl.* **59**, 101259 (2024).

---

## Revision History

| Date | Author | Change |
|------|--------|--------|
| 2025-02-13 | Claude (automated audit) | v1: Initial creation. API verification, identified systematic issues. |
| 2026-02-13 | Claude (restructure) | v2: Complete rewrite. Replaced all 6 examples with new files using real data from InitializeSBE(). Added quickstart based on sbetestprop.py. Preserved best content from originals. |
