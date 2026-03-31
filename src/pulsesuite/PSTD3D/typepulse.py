"""Laser pulse parameter structure for PSTD3D.

Mirrors Fortran type ps (typepulse.f90). Uses lambda_ instead of lambda (reserved).

Author: Emily S. Hatten
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


@dataclass
class ps:
    """Laser-pulse parameter structure.

    Parameters
    ----------
    lambda_ : float
        Vacuum carrier wavelength (m).
    Amp : float
        Peak electric-field amplitude (V/m).
    Tw : float
        Gaussian pulse width parameter (s). Intensity FWHM; see CalcTau.
    Tp : float
        Time at which the pulse peak crosses the origin x = 0 (s).
    chirp : float
        Linear frequency chirp coefficient (rad/s^2).
    pol : int, optional
        Polarisation index (default 0).
    """

    lambda_: float  # vacuum wavelength (m)
    Amp: float  # peak amplitude (V/m)
    Tw: float  # pulse width parameter (s)
    Tp: float  # peak-crossing time (s)
    chirp: float  # chirp coefficient (rad/s²)
    pol: int = 0  # polarisation index
    w0: float = float("inf")  # beam waist radius (m); inf = plane wave

    def CalcK0(self) -> float:
        """Carrier wavenumber k0 = 2*pi / lambda (rad/m)."""
        return _twopi / self.lambda_

    def CalcFreq0(self) -> float:
        """Carrier frequency nu0 = c0 / lambda (Hz)."""
        return _c0 / self.lambda_

    def CalcOmega0(self) -> float:
        """Carrier angular frequency omega0 = 2*pi*c0 / lambda (rad/s)."""
        return _twopi * _c0 / self.lambda_

    def CalcTau(self) -> float:
        """Gaussian 1/e field half-width: tau_G = Tw / sqrt(2*ln2) (s)."""
        return self.Tw / np.sqrt(2.0 * np.log(2.0))

    def CalcDeltaOmega(self) -> float:
        """Spectral FWHM (angular frequency, rad/s).

        Includes chirp broadening via dimensionless chirp parameter a = chirp * tau_G^2.
        """
        tau_G = self.CalcTau()
        a = self.chirp * tau_G**2
        return np.sqrt(8.0 * np.log(2.0) * (1.0 + a**2)) / tau_G

    def CalcTime_BandWidth(self) -> float:
        """Time-bandwidth product tau_G * Delta_omega (dimensionless)."""
        return self.CalcDeltaOmega() * self.CalcTau()

    def CalcRayleigh(self, w0: float) -> float:
        """Rayleigh range z_R = pi * w0^2 / lambda (m).

        Parameters
        ----------
        w0 : float
            Beam waist radius (m) at the focus.
        """
        return _pi * w0**2 / self.lambda_

    def CalcCurvature(self, x: float, w0: float) -> float:
        """Wavefront curvature radius R(x) = x * [1 + (z_R/x)^2] (m).

        Returns inf at x = 0 (planar wavefront at the waist).

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
        """Gouy phase phi_G(x) = arctan(x / z_R) (rad).

        Parameters
        ----------
        x : float
            Propagation distance from the beam waist (m).
        w0 : float
            Beam waist radius (m).
        """
        return np.arctan(x / self.CalcRayleigh(w0))

    def PulseFieldXT(self, x: float, t: float) -> complex:
        """Complex pulse electric field at position x, time t.

        Uses retarded time tau = t - Tp - x/c0 with chirped Gaussian envelope.

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
        """Real 3-D pulse electric field E(x, y, z, t).

        Chirped Gaussian pulse with transverse Gaussian beam profile.

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


def GetLambda(pulse: ps) -> float:
    return pulse.lambda_


def GetAmp(pulse: ps) -> float:
    return pulse.Amp


def GetTw(pulse: ps) -> float:
    return pulse.Tw


def GetTp(pulse: ps) -> float:
    return pulse.Tp


def GetChirp(pulse: ps) -> float:
    return pulse.chirp


def GetPol(pulse: ps) -> int:
    return pulse.pol


def SetLambda(pulse: ps, lambda_: float) -> None:
    """Notes: Fortran declared the argument as ``integer`` (a bug);
    this port accepts ``float`` as intended.
    """
    pulse.lambda_ = float(lambda_)


def SetAmp(pulse: ps, Amp: float) -> None:
    pulse.Amp = Amp


def SetTw(pulse: ps, Tw: float) -> None:
    pulse.Tw = Tw


def SetTp(pulse: ps, Tp: float) -> None:
    pulse.Tp = Tp


def SetChirp(pulse: ps, chirp: float) -> None:
    pulse.chirp = chirp


def SetPol(pulse: ps, pol: int) -> None:
    pulse.pol = pol


def GetW0(pulse: ps) -> float:
    """inf means plane-wave mode."""
    return pulse.w0


def SetW0(pulse: ps, w0: float) -> None:
    pulse.w0 = w0


def CalcK0(pulse: ps) -> float:
    return pulse.CalcK0()


def CalcFreq0(pulse: ps) -> float:
    return pulse.CalcFreq0()


def CalcOmega0(pulse: ps) -> float:
    return pulse.CalcOmega0()


def CalcTau(pulse: ps) -> float:
    return pulse.CalcTau()


def CalcDeltaOmega(pulse: ps) -> float:
    return pulse.CalcDeltaOmega()


def CalcTime_BandWidth(pulse: ps) -> float:
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


_PFRMTA = "{:25.15E}"


def readpulseparams_sub(u, pulse: ps) -> None:
    """Read five pulse parameters from open file handle ``u``.

    Parameter order: lambda_, Amp, Tw, Tp, chirp.
    ``pol`` is optional and is not present in the Fortran parameter file.
    """
    pulse.lambda_ = float(GetFileParam(u))
    pulse.Amp = float(GetFileParam(u))
    pulse.Tw = float(GetFileParam(u))
    pulse.Tp = float(GetFileParam(u))
    pulse.chirp = float(GetFileParam(u))


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
    u.write(
        f"{_PFRMTA.format(pulse.Tp)}      : The time the pulse crosses the origin. (s)\n"
    )
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
