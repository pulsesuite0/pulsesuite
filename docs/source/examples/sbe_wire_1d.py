"""1D quantum wire SBE time evolution.

Drives a 100-pixel quantum wire with a Gaussian-enveloped laser pulse
in Ex and Ey, evolves the Semiconductor Bloch Equations for 10 000 steps,
and records E-field and polarization at the wire midpoint.

Output: fields/*.dat (time-series of midpoint field/polarization values).
"""

import os
from pathlib import Path

import numpy as np
from scipy.constants import c as c0

from pulsesuite.PSTD3D.SBEs import InitializeSBE, QWCalculator
from pulsesuite.PSTD3D.typespace import GetKArray, GetSpaceArray

# --- Grid and time parameters ---
Nr = 100                # pixels along quantum wire
drr = 10e-9             # pixel size (m)
n0 = 3.1                # background refractive index
Nt = 10_000             # time steps
dt = 10e-18             # time step (s)

# --- Pulse parameters (per polarisation) ---
PULSE = {
    "x": dict(E0=1e7, tw=10e-15, tp=50e-15, lam=800e-9),
    "y": dict(E0=2e7, tw=10e-15, tp=50e-15, lam=800e-9),
    "z": dict(E0=0.0, tw=10e-15, tp=50e-15, lam=800e-9),
}

twopi = 2.0 * np.pi


def _angular_params(lam):
    """Return (omega0, k0, T_cycle) for a given wavelength."""
    w0 = twopi * c0 / lam
    k0 = twopi / lam * n0
    Tc = lam / c0
    return w0, k0, Tc


def _gaussian_pulse(t, E0, w0, tw, tp):
    """Gaussian envelope with super-Gaussian cutoff."""
    u = w0 * (t - tp)
    w0tw = w0 * tw
    return (
        E0
        * np.exp(-(u**2) / w0tw**2)
        * np.cos(u)
        * np.exp(-(u**20) / (2 * w0tw) ** 20)
    )


def run():
    # Derived quantities
    params = {k: {**v, **dict(zip(["w0", "k0", "Tc"], _angular_params(v["lam"])))}
              for k, v in PULSE.items()}
    Emax0 = np.sqrt(sum(p["E0"] ** 2 for p in params.values()))

    # Spatial / momentum grids
    rr = GetSpaceArray(Nr, (Nr - 1) * drr)
    qrr = GetKArray(Nr, Nr * drr)

    # Initialise SBE solver
    InitializeSBE(qrr, rr, 0.0, Emax0, PULSE["x"]["lam"], 2, True)

    # Field and polarisation arrays
    Exx = np.zeros(Nr, dtype=np.complex128)
    Eyy = np.zeros(Nr, dtype=np.complex128)
    Ezz = np.zeros(Nr, dtype=np.complex128)
    Pxx1 = np.zeros(Nr, dtype=np.complex128)
    Pyy1 = np.zeros(Nr, dtype=np.complex128)
    Pzz1 = np.zeros(Nr, dtype=np.complex128)
    Pxx2 = np.zeros(Nr, dtype=np.complex128)
    Pyy2 = np.zeros(Nr, dtype=np.complex128)
    Pzz2 = np.zeros(Nr, dtype=np.complex128)
    Rho = np.zeros(Nr, dtype=np.complex128)
    Vrr = np.zeros(Nr, dtype=np.complex128)
    boolT, boolF = [True], [False]

    mid = Nr // 2
    outdir = Path("fields")
    outdir.mkdir(exist_ok=True)

    # Collect midpoint time-series
    ts = np.empty(Nt)
    Ex_mid = np.empty(Nt)
    Ey_mid = np.empty(Nt)
    Px_mid = np.empty(Nt)
    Py_mid = np.empty(Nt)

    t = 0.0
    for n in range(Nt):
        # Drive fields
        px = params["x"]
        py = params["y"]
        Exx[:] = _gaussian_pulse(t, px["E0"], px["w0"], px["tw"], px["tp"])
        Eyy[:] = _gaussian_pulse(t, py["E0"], py["w0"], py["tw"], py["tp"])

        # Evolve SBEs (two sub-bands)
        QWCalculator(Exx, Eyy, Ezz, Vrr, rr, qrr, dt, 1,
                     Pxx1, Pyy1, Pzz1, Rho, boolT, boolF)
        QWCalculator(Exx, Eyy, Ezz, Vrr, rr, qrr, dt, 2,
                     Pxx2, Pyy2, Pzz2, Rho, boolT, boolF)

        # Record midpoint values
        ts[n] = t
        Ex_mid[n] = np.real(Exx[mid])
        Ey_mid[n] = np.real(Eyy[mid])
        Px_mid[n] = np.real((Pxx1[mid] + Pxx2[mid]) * 0.5)
        Py_mid[n] = np.real((Pyy1[mid] + Pyy2[mid]) * 0.5)

        t += dt

    # Write output
    np.savetxt(outdir / "Ex.dat", np.column_stack([ts, Ex_mid]))
    np.savetxt(outdir / "Ey.dat", np.column_stack([ts, Ey_mid]))
    np.savetxt(outdir / "Px_avg.dat", np.column_stack([ts, Px_mid]))
    np.savetxt(outdir / "Py_avg.dat", np.column_stack([ts, Py_mid]))
    print(f"Done: {Nt} steps, output in {outdir}/")


if __name__ == "__main__":
    run()
