"""typepulse — Pulse parameter structure for PSTD3D Maxwell solver.

Port of Fortran ``typepulse.f90``.

The ``ps`` dataclass holds the six scalars that describe a laser pulse.
Derived temporal/spectral/spatial quantities are methods on the class;
trivial accessors and file-I/O functions are module-level wrappers for
Fortran call-site compatibility.

Notes on Fortran divergences
-----------------------------
1. ``lambda`` → ``lambda_``: ``lambda`` is a reserved keyword in Python.
   Module-level ``GetLambda`` / ``SetLambda`` still provide the Fortran API.

2. ``SetLambda`` type fix: the Fortran source declares the input as
   ``integer`` (a bug — the field is ``real(dp)``).  The Python port accepts
   ``float`` as intended.

3. ``CalcRayleigh`` correction: the Fortran source mistakenly substitutes
   ``CalcOmega0(pulse)`` (angular carrier frequency ω₀, units rad/s) for the
   beam waist radius w₀ (units m) in ``z_R = π w₀² / λ``.  This yields a
   dimensionally incorrect result.  The Python port requires ``w₀`` to be
   passed explicitly:

       CalcRayleigh(pulse, w0)   # w0 in metres
       CalcCurvature(pulse, x, w0)
       CalcGouyPhase(pulse, x, w0)

   Callers that relied on the (broken) Fortran behaviour must be updated.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

try:
    from scipy.constants import c as _c0
except ImportError:  # pragma: no cover
    _c0: float = 2.99792458e8

_dp = np.float64
_pi: float = np.pi
_twopi: float = 2.0 * np.pi

LOGVERBOSE: int = 2  # mirrors Fortran LOGVERBOSE constant


# ──────────────────────────────────────────────────────────────────────────────
# Local stubs (mirror Fortran fileio / logger dependencies)
# ──────────────────────────────────────────────────────────────────────────────


def GetFileParam(f) -> float:
    """Read one numeric parameter from an open text file handle."""
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
# ps — pulse parameter dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ps:
    """Laser-pulse parameter structure.

    Parameters
    ----------
    lambda_ : float
        Vacuum carrier wavelength (m).  Named ``lambda_`` because ``lambda``
        is a Python reserved word; Fortran field is ``pulse%lambda``.
    Amp : float
        Peak electric-field amplitude (V/m).
    Tw : float
        Gaussian pulse width parameter (s).  The 1/e intensity half-width
        is τ = Tw / (2√ln2); see ``CalcTau``.
    Tp : float
        Time at which the pulse peak crosses the origin x = 0 (s).
    chirp : float
        Linear frequency chirp coefficient (rad/s²).  The instantaneous
        carrier phase is ω₀·delay + chirp·delay².
    pol : int, optional
        Polarisation index (default 0).

    Notes
    -----
    Mirrors Fortran ``type ps`` (``typepulse.f90``).
    """

    lambda_: float  # vacuum wavelength (m)
    Amp: float      # peak amplitude (V/m)
    Tw: float       # pulse width parameter (s)
    Tp: float       # peak-crossing time (s)
    chirp: float    # chirp coefficient (rad/s²)
    pol: int = 0    # polarisation index
    w0: float = float("inf")  # beam waist radius (m); inf = plane wave

    # ── temporal / spectral properties ──────────────────────────────────────

    def CalcK0(self) -> float:
        r"""Carrier wavenumber :math:`k_0 = 2\pi / \lambda` (rad/m)."""
        return _twopi / self.lambda_

    def CalcFreq0(self) -> float:
        r"""Carrier frequency :math:`\nu_0 = c_0 / \lambda` (Hz)."""
        return _c0 / self.lambda_

    def CalcOmega0(self) -> float:
        r"""Carrier angular frequency :math:`\omega_0 = 2\pi c_0 / \lambda` (rad/s)."""
        return _twopi * _c0 / self.lambda_

    def CalcTau(self) -> float:
        r"""Gaussian 1/e field half-width: :math:`\tau_G = T_w / \sqrt{2\ln 2}` (s).

        For the field convention :math:`E(t) \propto \exp(-t^2/\tau_G^2)`,
        the intensity FWHM is :math:`T_w` (the ``Tw`` attribute).
        """
        return self.Tw / np.sqrt(2.0 * np.log(2.0))

    def CalcDeltaOmega(self) -> float:
        r"""Spectral FWHM (angular frequency, rad/s).

        .. math::

            \Delta\omega = \frac{\sqrt{8 \ln 2\,(1 + a^2)}}{\tau_G}

        where :math:`a = \chi\,\tau_G^2` is the dimensionless chirp parameter.
        For an unchirped pulse (:math:`a = 0`) this reduces to
        :math:`\sqrt{8\ln 2}/\tau_G`.
        """
        tau_G = self.CalcTau()
        a = self.chirp * tau_G**2
        return np.sqrt(8.0 * np.log(2.0) * (1.0 + a**2)) / tau_G

    def CalcTime_BandWidth(self) -> float:
        r"""Time–bandwidth product :math:`\tau_G \cdot \Delta\omega` (dimensionless).

        For an unchirped Gaussian this equals :math:`\sqrt{8\ln 2} \approx 2.355`.
        """
        return self.CalcDeltaOmega() * self.CalcTau()

    # ── spatial / Gaussian-beam properties ──────────────────────────────────

    def CalcRayleigh(self, w0: float) -> float:
        r"""Rayleigh range :math:`z_R = \pi w_0^2 / \lambda` (m).

        Parameters
        ----------
        w0 : float
            Beam waist radius (m) at the focus.

        Notes
        -----
        The Fortran source erroneously used ``CalcOmega0(pulse)`` (angular
        frequency, rad/s) in place of ``w0`` (beam waist, m), yielding a
        dimensionally incorrect result.  This implementation requires ``w0``
        explicitly.
        """
        return _pi * w0**2 / self.lambda_

    def CalcCurvature(self, x: float, w0: float) -> float:
        r"""Wavefront curvature radius :math:`R(x) = x\,[1 + (z_R/x)^2]` (m).

        Returns ``inf`` at ``x = 0`` (planar wavefront at the waist).

        Parameters
        ----------
        x : float
            Propagation distance from the beam waist (m).
        w0 : float
            Beam waist radius (m).
        """
        xR = self.CalcRayleigh(w0)
        if x == 0.0:
            return float("inf")
        return x * (1.0 + (xR / x) ** 2)

    def CalcGouyPhase(self, x: float, w0: float) -> float:
        r"""Gouy phase :math:`\phi_G(x) = \arctan(x / z_R)` (rad).

        Parameters
        ----------
        x : float
            Propagation distance from the beam waist (m).
        w0 : float
            Beam waist radius (m).
        """
        return np.arctan(x / self.CalcRayleigh(w0))

    # ── field evaluation ─────────────────────────────────────────────────────

    def PulseFieldXT(self, x: float, t: float) -> complex:
        r"""Complex pulse electric field at position ``x``, time ``t``.

        .. math::

            E(x,t) = A \exp\!\left(-\frac{\tau^2}{\tau_G^2}\right)
                       \exp\!\left[i\!\left(\omega_0 \tau + \chi \tau^2\right)\right]

        where :math:`\tau = t - T_p - x/c_0` is the retarded time in the
        pulse frame, :math:`\tau_G = T_w / \sqrt{2\ln 2}` is the Gaussian
        1/e field half-width, :math:`\chi` is the chirp coefficient, and the
        carrier is embedded in the exponential (analytic-signal representation).

        Parameters
        ----------
        x : float
            Position along the propagation axis (m).
        t : float
            Observation time (s).

        Returns
        -------
        complex
            Complex electric field amplitude (V/m).
        """
        tau_G = self.CalcTau()
        delay = t - self.Tp - x / _c0
        envelope = self.Amp * np.exp(-(delay**2) / tau_G**2)
        phase = self.CalcOmega0() * delay + self.chirp * delay**2
        return complex(envelope * np.exp(1j * phase))

    def PulseField3D(self, x: float, y: float, z: float, t: float) -> float:
        r"""Real 3-D pulse electric field :math:`E(x,y,z,t)`.

        .. math::

            E = A\,\exp\!\left(-\frac{y^2+z^2}{w_0^2}\right)
                \exp\!\left(-\frac{\tau^2}{\tau_G^2}\right)
                \cos(\omega_0 \tau + \chi \tau^2)

        where :math:`\tau = t - T_p - x/c_0` is the retarded time.

        Parameters
        ----------
        x, y, z : float
            Spatial coordinates (m).
        t : float
            Observation time (s).

        Returns
        -------
        float
            Real electric field amplitude (V/m).
        """
        tau_G = self.CalcTau()
        omega0 = self.CalcOmega0()
        delay = t - self.Tp - x / _c0
        r2 = y**2 + z**2
        gauss_yz = np.exp(-r2 / self.w0**2)
        phase = omega0 * delay + self.chirp * delay**2
        return float(
            self.Amp * gauss_yz * np.exp(-(delay**2) / tau_G**2) * np.cos(phase)
        )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level Fortran API — trivial accessors
# ──────────────────────────────────────────────────────────────────────────────


def GetLambda(pulse: ps) -> float:
    """Return the carrier wavelength (m)."""
    return pulse.lambda_


def GetAmp(pulse: ps) -> float:
    """Return the peak amplitude (V/m)."""
    return pulse.Amp


def GetTw(pulse: ps) -> float:
    """Return the pulse width parameter (s)."""
    return pulse.Tw


def GetTp(pulse: ps) -> float:
    """Return the peak-crossing time (s)."""
    return pulse.Tp


def GetChirp(pulse: ps) -> float:
    """Return the chirp coefficient (rad/s²)."""
    return pulse.chirp


def GetPol(pulse: ps) -> int:
    """Return the polarisation index."""
    return pulse.pol


def SetLambda(pulse: ps, lambda_: float) -> None:
    """Set the carrier wavelength (m).

    Notes
    -----
    The Fortran source declared the argument as ``integer`` (a bug);
    this Python port accepts ``float`` as intended.
    """
    pulse.lambda_ = float(lambda_)


def SetAmp(pulse: ps, Amp: float) -> None:
    """Set the peak amplitude (V/m)."""
    pulse.Amp = Amp


def SetTw(pulse: ps, Tw: float) -> None:
    """Set the pulse width parameter (s)."""
    pulse.Tw = Tw


def SetTp(pulse: ps, Tp: float) -> None:
    """Set the peak-crossing time (s)."""
    pulse.Tp = Tp


def SetChirp(pulse: ps, chirp: float) -> None:
    """Set the chirp coefficient (rad/s²)."""
    pulse.chirp = chirp


def SetPol(pulse: ps, pol: int) -> None:
    """Set the polarisation index."""
    pulse.pol = pol


def GetW0(pulse: ps) -> float:
    """Return the beam waist radius (m).  ``inf`` means plane-wave mode."""
    return pulse.w0


def SetW0(pulse: ps, w0: float) -> None:
    """Set the beam waist radius (m)."""
    pulse.w0 = w0


# ── non-trivial module-level wrappers ────────────────────────────────────────


def CalcK0(pulse: ps) -> float:
    """Carrier wavenumber.  Delegates to ``pulse.CalcK0()``."""
    return pulse.CalcK0()


def CalcFreq0(pulse: ps) -> float:
    """Carrier frequency.  Delegates to ``pulse.CalcFreq0()``."""
    return pulse.CalcFreq0()


def CalcOmega0(pulse: ps) -> float:
    """Carrier angular frequency.  Delegates to ``pulse.CalcOmega0()``."""
    return pulse.CalcOmega0()


def CalcTau(pulse: ps) -> float:
    """1/e intensity half-width.  Delegates to ``pulse.CalcTau()``."""
    return pulse.CalcTau()


def CalcDeltaOmega(pulse: ps) -> float:
    """Spectral half-width.  Delegates to ``pulse.CalcDeltaOmega()``."""
    return pulse.CalcDeltaOmega()


def CalcTime_BandWidth(pulse: ps) -> float:
    """Time–bandwidth product.  Delegates to ``pulse.CalcTime_BandWidth()``."""
    return pulse.CalcTime_BandWidth()


def CalcRayleigh(pulse: ps, w0: float) -> float:
    """Rayleigh range.  Delegates to ``pulse.CalcRayleigh(w0)``."""
    return pulse.CalcRayleigh(w0)


def CalcCurvature(pulse: ps, x: float, w0: float) -> float:
    """Wavefront curvature radius.  Delegates to ``pulse.CalcCurvature(x, w0)``."""
    return pulse.CalcCurvature(x, w0)


def CalcGouyPhase(pulse: ps, x: float, w0: float) -> float:
    """Gouy phase.  Delegates to ``pulse.CalcGouyPhase(x, w0)``."""
    return pulse.CalcGouyPhase(x, w0)


def PulseFieldXT(x: float, t: float, pulse: ps) -> complex:
    """Complex pulse field at (x, t).  Delegates to ``pulse.PulseFieldXT(x, t)``.

    Notes
    -----
    Argument order matches the Fortran signature:
    ``PulseFieldXT(x, t, pulse)`` — note ``pulse`` is last.
    """
    return pulse.PulseFieldXT(x, t)


def PulseField3D(x: float, y: float, z: float, t: float, pulse: ps) -> float:
    """Real 3-D pulse field.  Delegates to ``pulse.PulseField3D(x, y, z, t)``.

    Argument order matches the Fortran signature:
    ``PulseField3D(x, y, z, t, pulse)`` — note ``pulse`` is last.
    """
    return pulse.PulseField3D(x, y, z, t)


# ──────────────────────────────────────────────────────────────────────────────
# File I/O  (mirrors Fortran ReadPulseParams / WritePulseParams)
# ──────────────────────────────────────────────────────────────────────────────

_PFRMTA = "{:25.15E}"


def readpulseparams_sub(u, pulse: ps) -> None:
    """Read five pulse parameters from open file handle ``u``.

    Parameter order: lambda_, Amp, Tw, Tp, chirp.
    ``pol`` is optional and is not present in the Fortran parameter file.
    """
    pulse.lambda_ = float(GetFileParam(u))
    pulse.Amp     = float(GetFileParam(u))
    pulse.Tw      = float(GetFileParam(u))
    pulse.Tp      = float(GetFileParam(u))
    pulse.chirp   = float(GetFileParam(u))


def ReadPulseParams(cmd: str, pulse: ps) -> None:
    """Read pulse parameters from file *cmd* into ``pulse``, then dump to stdout.

    Fortran signature parity: ``ReadPulseParams(cmd, pulse)``.
    """
    with open(cmd, "r", encoding="utf-8") as f:
        readpulseparams_sub(f, pulse)
    dumppulse(pulse)


def writepulseparams_sub(u, pulse: ps) -> None:
    """Write pulse parameters to open file handle ``u`` with descriptive comments."""
    u.write(f"{_PFRMTA.format(pulse.lambda_)} : The pulse wavelength. (m)\n")
    u.write(f"{_PFRMTA.format(pulse.Amp)}     : The pulse amplitude. (V/m)\n")
    u.write(f"{_PFRMTA.format(pulse.Tw)}      : The pulsewidth. (s)\n")
    u.write(f"{_PFRMTA.format(pulse.Tp)}      : The time the pulse crosses the origin. (s)\n")
    u.write(f"{_PFRMTA.format(pulse.chirp)}   : The pulse chirp constant. (rad/s^2)\n")


def WritePulseParams(cmd: str, pulse: ps) -> None:
    """Write pulse parameters to file *cmd*.

    Fortran signature parity: ``WritePulseParams(cmd, pulse)``.
    """
    with open(cmd, "w", encoding="utf-8") as f:
        writepulseparams_sub(f, pulse)


def dumppulse(params: ps, level: int | None = None) -> None:
    """Print pulse parameters to stdout if the logging level is sufficient.

    Parameters
    ----------
    params : ps
        Pulse structure to display.
    level : int, optional
        Minimum log level required.  Defaults to LOGVERBOSE.
    """
    threshold = level if level is not None else LOGVERBOSE
    if GetLogLevel() >= threshold:
        writepulseparams_sub(sys.stdout, params)
