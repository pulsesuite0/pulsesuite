"""typetime — Time grid structure for PSTD3D Maxwell solver.

Port of Fortran ``typetime.f90`` (``updatedPSTD3D/newest/typetime.f90``).

The ``ts`` dataclass holds the four scalars that define a simulation's time
axis.  Non-trivial derived quantities (array generation, conjugate grid) are
methods on the dataclass; trivial accessors are module-level functions that
mirror the Fortran subroutine/function names for API parity.

Architecture
------------
- ``@dataclass ts``: plain storage (t, tf, dt, n) — mirrors Fortran ``type ts``
- Hybrid: non-trivial computations (CalcNt, GetTArray, GetOmegaArray,
  GetdOmega, UpdateT, UpdateN) are methods; trivial getters/setters remain
  as module-level functions for Fortran call-site compatibility.
- File I/O: Python ``open()`` context manager replaces Fortran ``new_unit()``
  / ``open`` / ``close``.
- ``initialize_field`` (private in Fortran, OpenMP-parallel zero-fill): omitted;
  use ``numpy.zeros`` directly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_dp = np.float64
_twopi: float = 2.0 * np.pi

LOGVERBOSE: int = 2  # mirrors Fortran LOGVERBOSE constant


# ──────────────────────────────────────────────────────────────────────────────
# Local stubs (mirror Fortran fileio / logger dependencies)
# ──────────────────────────────────────────────────────────────────────────────


def GetFileParam(f) -> float:
    """Read one numeric parameter from an open text file handle.

    The parameter file format has one value per line; anything after
    the first whitespace-separated token is treated as a comment.
    """
    line = f.readline()
    if not line:
        raise ValueError("Unexpected end of file while reading parameter")
    parts = line.split()
    if not parts:
        raise ValueError(f"Empty line in parameter file: {line!r}")
    try:
        return float(parts[0])
    except ValueError as exc:
        raise ValueError(f"Could not parse parameter from line: {line!r}") from exc


def GetLogLevel() -> int:
    """Return the current logging verbosity level (stub: always 0)."""
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# ts — time grid dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ts:
    """Time grid structure.

    Parameters
    ----------
    t : float
        Current simulation time (s).
    tf : float
        Final simulation time (s).
    dt : float
        Time step / pixel size (s).
    n : int
        Current time index.

    Notes
    -----
    Mirrors Fortran ``type ts`` (``typetime.f90``).  All fields are public;
    the Fortran ``private`` declaration is not enforced in Python.
    """

    t: float  # current time (s)
    tf: float  # final time (s)
    dt: float  # time step (s)
    n: int  # current time index

    # ── non-trivial derived quantities ──────────────────────────────────────

    def CalcNt(self) -> int:
        """Number of time steps remaining: floor((tf − t) / dt).

        Mirrors Fortran ``CalcNt(time)``.
        """
        return int((self.tf - self.t) / self.dt)

    def UpdateT(self, dt: float) -> None:
        """Advance current time by ``dt``: t ← t + dt."""
        self.t += dt

    def UpdateN(self, dn: int) -> None:
        """Advance current time index by ``dn``: n ← n + dn."""
        self.n += dn

    def GetTArray(self) -> NDArray[_dp]:
        """Return a 1-D array of time values starting at ``t``.

        .. code-block:: text

            t[i] = t + i·dt,   i = 0 … CalcNt−1

        Special case: if CalcNt == 1, returns ``[0.0]`` (mirrors Fortran).

        Returns
        -------
        ndarray, shape (Nt,), dtype float64
        """
        Nt = self.CalcNt()
        if Nt == 1:
            return np.array([0.0], dtype=_dp)
        return self.t + np.arange(Nt, dtype=_dp) * self.dt

    def GetOmegaArray(self) -> NDArray[_dp]:
        r"""Return angular-frequency grid conjugate to the time axis.

        Uses the FFT-convention ordering (positive frequencies first, then
        negative), identical to ``numpy.fft.fftfreq(Nt, d=dt) * 2π``:

        .. code-block:: text

            ω[k] = 2π·k / (Nt·dt),   k = 0 … Nt/2
            ω[k] = 2π·(k − Nt) / (Nt·dt),  k = Nt/2+1 … Nt−1

        Returns
        -------
        ndarray, shape (Nt,), dtype float64
        """
        Nt = self.CalcNt()
        return np.fft.fftfreq(Nt, d=self.dt).astype(_dp) * _twopi

    def GetdOmega(self) -> float:
        r"""Return the angular-frequency step: dω = 2π / (Nt·dt)."""
        return _twopi / (self.CalcNt() * self.dt)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level Fortran API — trivial accessors
# ──────────────────────────────────────────────────────────────────────────────


def GetT(time: ts) -> float:
    """Return the current simulation time (s)."""
    return time.t


def GetTf(time: ts) -> float:
    """Return the final simulation time (s)."""
    return time.tf


def GetDt(time: ts) -> float:
    """Return the time step (s)."""
    return time.dt


def GetN(time: ts) -> int:
    """Return the current time index."""
    return time.n


def SetT(time: ts, t: float) -> None:
    """Set the current simulation time."""
    time.t = t


def SetTf(time: ts, tf: float) -> None:
    """Set the final simulation time."""
    time.tf = tf


def SetDt(time: ts, dt: float) -> None:
    """Set the time step."""
    time.dt = dt


def SetN(time: ts, n: int) -> None:
    """Set the current time index."""
    time.n = n


# ── non-trivial module-level wrappers (delegate to ts methods) ───────────────


def CalcNt(time: ts) -> int:
    """Number of time steps remaining.  Delegates to ``time.CalcNt()``."""
    return time.CalcNt()


def UpdateT(time: ts, dt: float) -> None:
    """Advance current time by ``dt``.  Delegates to ``time.UpdateT(dt)``."""
    time.UpdateT(dt)


def UpdateN(time: ts, dn: int) -> None:
    """Advance time index by ``dn``.  Delegates to ``time.UpdateN(dn)``."""
    time.UpdateN(dn)


def GetTArray(time: ts) -> NDArray[_dp]:
    """Return the time array.  Delegates to ``time.GetTArray()``."""
    return time.GetTArray()


def GetOmegaArray(time: ts) -> NDArray[_dp]:
    """Return the angular-frequency array.  Delegates to ``time.GetOmegaArray()``."""
    return time.GetOmegaArray()


def GetdOmega(time: ts) -> float:
    """Return the angular-frequency step.  Delegates to ``time.GetdOmega()``."""
    return time.GetdOmega()


# ── validation functions (mirrors Fortran typetime enhancements) ─────────────

try:
    from scipy.constants import c as _c0_val
except ImportError:  # pragma: no cover
    _c0_val: float = 2.99792458e8


def ValidateTimeStep(time: ts, dx: float, dy: float, dz: float, eps_r: float) -> dict:
    r"""Validate the time step against CFL and physics constraints.

    CFL condition for 3-D PSTD:

    .. math::

        v\,\sqrt{3}\,\frac{\Delta t}{\min(\Delta x,\Delta y,\Delta z)} \le 1

    Parameters
    ----------
    time : ts
        Time grid structure.
    dx, dy, dz : float
        Grid spacings (m).
    eps_r : float
        Relative permittivity.

    Returns
    -------
    dict
        Keys: ``v_phase``, ``min_dx``, ``dt_max``, ``cfl_number``,
        ``stable`` (bool).
    """
    v_phase = _c0_val / np.sqrt(eps_r)
    min_dx = min(dx, dy, dz)
    dt_max = min_dx / (v_phase * np.sqrt(3.0))
    cfl = v_phase * np.sqrt(3.0) * time.dt / min_dx

    return {
        "v_phase": v_phase,
        "min_dx": min_dx,
        "dt_max": dt_max,
        "cfl_number": cfl,
        "stable": cfl <= 1.0,
    }


def CalculateOptimalDt(
    dx: float, dy: float, dz: float, eps_r: float, safety: float = 0.9
) -> float:
    r"""Compute an optimal CFL-safe time step.

    .. math::

        \Delta t_{\text{opt}} = \text{safety} \times
            \frac{\min(\Delta x, \Delta y, \Delta z)}{v\,\sqrt{3}}

    Parameters
    ----------
    dx, dy, dz : float
        Grid spacings (m).
    eps_r : float
        Relative permittivity.
    safety : float, optional
        Safety factor (< 1).  Default 0.9.

    Returns
    -------
    float
        Optimal time step (s).
    """
    v_phase = _c0_val / np.sqrt(eps_r)
    min_dx = min(dx, dy, dz)
    return safety * min_dx / (v_phase * np.sqrt(3.0))


# ──────────────────────────────────────────────────────────────────────────────
# File I/O  (mirrors Fortran ReadTimeParams / WriteTimeParams)
# ──────────────────────────────────────────────────────────────────────────────

_PFRMTA = "{:25.15E}"  # E25.15E3 equivalent


def readtimeparams_sub(u, time: ts) -> None:
    """Populate ``time`` by reading four parameters from open file handle ``u``.

    Parameter order (one per line): t, tf, dt, n.
    """
    time.t = float(GetFileParam(u))
    time.tf = float(GetFileParam(u))
    time.dt = float(GetFileParam(u))
    time.n = int(GetFileParam(u))


def ReadTimeParams(cmd: str, time: ts) -> None:
    """Read time parameters from file *cmd* into ``time``, then dump to stdout.

    Fortran signature parity: ``ReadTimeParams(cmd, time)``.
    """
    with open(cmd, "r", encoding="utf-8") as f:
        readtimeparams_sub(f, time)
    dumptime(time)


def WriteTimeParams_sub(u, time: ts) -> None:
    """Write time parameters to open file handle ``u`` with descriptive comments."""
    u.write(f"{_PFRMTA.format(time.t)}  : Current time of simulation. (s)\n")
    u.write(f"{_PFRMTA.format(time.tf)} : Final time of simulation. (s)\n")
    u.write(f"{_PFRMTA.format(time.dt)} : Time pixel size [dt]. (s)\n")
    u.write(f"{time.n:25d} : Current time index.\n")


def writetimeparams(cmd: str, time: ts) -> None:
    """Write time parameters to file *cmd*.

    Fortran signature parity: ``writetimeparams(cmd, time)``.
    """
    with open(cmd, "w", encoding="utf-8") as f:
        WriteTimeParams_sub(f, time)


def dumptime(params: ts, level: int | None = None) -> None:
    """Print time parameters to stdout if the logging level is sufficient.

    Parameters
    ----------
    params : ts
        Time structure to display.
    level : int, optional
        Minimum log level required.  Defaults to LOGVERBOSE.
    """
    threshold = level if level is not None else LOGVERBOSE
    if GetLogLevel() >= threshold:
        WriteTimeParams_sub(sys.stdout, params)
