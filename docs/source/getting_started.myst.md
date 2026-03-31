---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Getting Started

## Installation

Install PulseSuite using [uv](https://docs.astral.sh/uv/):

```bash
uv add pulsesuite
```

To install from source for development:

```bash
git clone https://github.com/pulsesuite0/pulsesuite.git
cd pulsesuite
uv sync
```

## Basic Usage

PulseSuite provides the **PSTD3D** propagator for simulating ultrafast
laser-matter interactions in semiconductor quantum structures.

```{code-cell} python
:tags: [skip-execution]

from pulsesuite.PSTD3D.typespace import ss
from pulsesuite.PSTD3D.typetime import ts
from pulsesuite.PSTD3D.typepulse import ps
from pulsesuite.PSTD3D.PSTD3D import PSTD3DPropagator

# 1. Define the spatial grid
space = ss(Dims=3, Nx=256, Ny=1, Nz=1, dx=50e-9, dy=50e-9, dz=50e-9, epsr=1.0)

# 2. Configure time-stepping parameters
time = ts(t=0.0, tf=1e-13, dt=1e-16, n=0)

# 3. Specify the input pulse (800 nm, 30 fs, 1e8 V/m)
pulse = ps(lambda_=800e-9, Amp=1e8, Tw=30e-15, Tp=50e-15, chirp=0.0)

# 4. Create and run the propagator
prop = PSTD3DPropagator(space, time, pulse, source_type="ic", boundary_type="mask")
prop.run()
```

## What is PSTD3D?

PSTD3D is a **Pseudo-Spectral Time-Domain** solver that propagates
electromagnetic pulses through semiconductor nanostructures. It couples
Maxwell's equations with the semiconductor Bloch equations to model
light-matter interactions at the quantum level.

## Running Tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=src/pulsesuite --cov-report=html
```
