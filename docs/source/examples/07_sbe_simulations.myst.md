---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# SBE Simulation Examples

These examples demonstrate end-to-end quantum wire simulations using the
Semiconductor Bloch Equations (SBE) solver in PulseSuite.

```{note}
These simulations are computationally intensive (10 000+ time steps with
Numba JIT-compiled kernels). They are not executed during the documentation
build. Run them directly:

    python examples/sbe_wire_1d.py
    python examples/sbe_wire_3d_longitudinal.py
```

## 1D Quantum Wire — `sbe_wire_1d.py`

Simplest SBE example: drives a 100-pixel quantum wire with Gaussian-enveloped
Ex and Ey pulses at 800 nm, evolves both sub-bands for 10 000 steps (10 as dt),
and records midpoint E-field and polarisation.

**What it demonstrates:**
- `InitializeSBE` setup with momentum and real-space grids
- `QWCalculator` time-stepping for two sub-bands
- Sub-band averaging of polarisation

**Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Nr        | 100   | Grid pixels along wire |
| drr       | 10 nm | Pixel spacing |
| dt        | 10 as | Time step |
| Nt        | 10 000 | Time steps |
| E0x       | 10 MV/m | Peak Ex amplitude |
| E0y       | 20 MV/m | Peak Ey amplitude |
| lambda    | 800 nm | Carrier wavelength |

**Output files:**

| File | Contents |
|------|----------|
| `fields/Ex.dat` | Time vs midpoint Re(Ex) |
| `fields/Ey.dat` | Time vs midpoint Re(Ey) |
| `fields/Px_avg.dat` | Time vs midpoint Re(Px) (sub-band average) |
| `fields/Py_avg.dat` | Time vs midpoint Re(Py) (sub-band average) |

**Source:** [sbe_wire_1d.py](sbe_wire_1d.py)

---

## 3D Quantum Wire with Longitudinal Field Decomposition — `sbe_wire_3d_longitudinal.py`

Full simulation with Helmholtz decomposition of the longitudinal electric
field into contributions from polarisation (P), charge density (Rho), and
their combination (P+Rho). Reads spatial grid from a params file and produces
time-series and 2D slice output.

**What it demonstrates:**
- `ReadSpaceParams` for grid setup from file
- `QuantumWire` call with full longitudinal decomposition outputs
- `MakePlaneWaveX/Y/Z` plane-wave source generation
- `WriteIT2D` for 2D slice output
- Max/min field tracking across all time steps

**Parameters (defaults):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| dt        | 10 as | Time step |
| Nt        | 10 000 | Time steps |
| E0x       | 200 MV/m | Peak Ex amplitude |
| lambda    | 800 nm | Carrier wavelength |
| n0        | 3.1 | Background refractive index |

**Output files (21 time-series + 2D slices):**

Time-series at grid midpoint:
`Ex.dat`, `Ey.dat`, `Ez.dat`, `Jx.dat`, `Jy.dat`, `Jz.dat`, `Rho.dat`,
`Eywireloc.dat`, `Exl.dat`, `Eyl.dat`, `Ezl.dat`,
`ExlfromPRho.dat`, `EylfromPRho.dat`, `EzlfromPRho.dat`,
`ExlfromP.dat`, `EylfromP.dat`, `EzlfromP.dat`,
`ExlfromRho.dat`, `EylfromRho.dat`, `EzlfromRho.dat`,
`RhoBound.dat`, `final_max_min.dat`

2D slices (every `slice_interval` steps) in x, y, z planes for:
ExPRho, EyPRho, EzPRho, ExP, EyP, EzP, ExRho, EyRho, EzRho,
Exl, Eyl, Ezl, Rho, RhoBound

### Sample output

```{figure} ../_static/examples/E_fields.png
:width: 100%
:align: center

Longitudinal electric field components $E_x^\parallel(x,y,0)$ and $E_y^\parallel(x,y,0)$
in the z=0 plane, showing the field structure around the quantum wire.
```

```{figure} ../_static/examples/rho_free.png
:width: 60%
:align: center

Free charge density $\rho_f(x,y,0)$ induced by the laser pulse along the
quantum wire axis.
```

**Source:** [sbe_wire_3d_longitudinal.py](sbe_wire_3d_longitudinal.py)
