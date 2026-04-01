"""Spatial grid structure for PSTD3D simulations.

Provides the ss dataclass and accessors for 1D/2D/3D spatial grids.

Author: Rahul R. Sah & Emily S. Hatten
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

try:
    from numba import jit

    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False

    # Fallback: create a no-op decorator
    def jit(*args, **kwargs):  # noqa: ARG001, ARG002
        def decorator(func):
            return func

        if args and callable(args[0]):
            # Called as @jit without parentheses
            return args[0]
        return decorator


# Physical constants (matching Fortran constants module)
pi = np.pi
twopi = 2.0 * np.pi  # 2π constant (matching Fortran twopi)


# Default logging level constants (if not provided by logging module)
LOGVERBOSE = 2


def GetFileParam(file_handle):
    """Read one numeric parameter from an open text file handle."""
    line = file_handle.readline()
    if not line:
        raise ValueError("Unexpected end of file while reading parameter")
    # Extract first numeric value from line (handles comments)
    parts = line.split()
    if not parts:
        raise ValueError(f"Empty line in parameter file: {line}")
    try:
        # Try to convert first part to float
        return float(parts[0])
    except ValueError as exc:
        raise ValueError(f"Could not parse parameter value from line: {line}") from exc


def GetLogLevel():
    """Return the current logging verbosity level (stub: always 0)."""
    return 0  # Default: minimal logging


@dataclass
class ss:
    """Spatial grid structure. Mirrors Fortran type ss (typespace.f90)."""

    Dims: int
    Nx: int
    Ny: int
    Nz: int
    dx: float
    dy: float
    dz: float
    epsr: float


def readspaceparams_sub(file_handle, space):
    """Read eight space parameters from an open file handle into ``space``."""
    space.Dims = int(GetFileParam(file_handle))
    space.Nx = int(GetFileParam(file_handle))
    space.Ny = int(GetFileParam(file_handle))
    space.Nz = int(GetFileParam(file_handle))
    space.dx = GetFileParam(file_handle)
    space.dy = GetFileParam(file_handle)
    space.dz = GetFileParam(file_handle)
    space.epsr = GetFileParam(file_handle)


def ReadSpaceParams(filename, space):
    """Read space parameters from file into ``space``, then dump to stdout."""
    with open(filename, "r", encoding="utf-8") as f:
        readspaceparams_sub(f, space)
    dumpspace(space)


def WriteSpaceParams_sub(file_handle, space):
    """Write space parameters to an open file handle with descriptive comments."""
    # Format: I25 for integers, E25.15E3 for floats (matching Fortran pfrmtA)
    file_handle.write(f"{space.Dims:25d} : Number of dimensions.\n")
    file_handle.write(f"{space.Nx:25d} : Number of space X points.\n")
    file_handle.write(f"{space.Ny:25d} : Number of space Y points.\n")
    file_handle.write(f"{space.Nz:25d} : Number of space Z points.\n")
    # Note: Fixed Fortran typos - original had "With" and all said "X pixel"
    file_handle.write(f"{space.dx:25.15E} : The Width of the X pixel. (m)\n")
    file_handle.write(f"{space.dy:25.15E} : The Width of the Y pixel. (m)\n")
    file_handle.write(f"{space.dz:25.15E} : The Width of the Z pixel. (m)\n")
    file_handle.write(
        f"{space.epsr:25.15E} : The relative background dielectric constant\n"
    )


def writespaceparams(filename, space):
    """Write space parameters to file."""
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        WriteSpaceParams_sub(f, space)


def dumpspace(params, level=None):
    """Print space parameters to stdout if logging level is sufficient."""
    if level is not None:
        if GetLogLevel() >= level:
            WriteSpaceParams_sub(sys.stdout, params)
    else:
        if GetLogLevel() >= LOGVERBOSE:
            WriteSpaceParams_sub(sys.stdout, params)


# Getter functions - these are simple property accessors
# JIT compilation not needed for simple attribute access


def GetNx(space):
    return space.Nx


def GetNy(space):
    return space.Ny


def GetNz(space):
    return space.Nz


# Getter functions with special logic - these can benefit from JIT
# but need to be careful about nopython compatibility


@jit(
    nopython=True, fastmath=True, cache=True
)  # fastmath OK: scalar conditional/multiply
def _GetDx_core(Nx, dx):
    if Nx == 1:
        return 1.0
    else:
        return dx


def GetDx(space):
    """Return dx, or 1.0 if Nx == 1 (collapsed dimension)."""
    if JIT_AVAILABLE:
        try:
            return _GetDx_core(space.Nx, space.dx)
        except (TypeError, ValueError, RuntimeError):
            # Fallback to standard Python
            if space.Nx == 1:
                return 1.0
            else:
                return space.dx
    else:
        if space.Nx == 1:
            return 1.0
        else:
            return space.dx


@jit(
    nopython=True, fastmath=True, cache=True
)  # fastmath OK: scalar conditional/multiply
def _GetDy_core(Ny, dy):
    if Ny == 1:
        return 1.0
    else:
        return dy


def GetDy(space):
    """Return dy, or 1.0 if Ny == 1 (collapsed dimension)."""
    if JIT_AVAILABLE:
        try:
            return _GetDy_core(space.Ny, space.dy)
        except (TypeError, ValueError, RuntimeError):
            # Fallback to standard Python
            if space.Ny == 1:
                return 1.0
            else:
                return space.dy
    else:
        if space.Ny == 1:
            return 1.0
        else:
            return space.dy


@jit(
    nopython=True, fastmath=True, cache=True
)  # fastmath OK: scalar conditional/multiply
def _GetDz_core(Nz, dz):
    if Nz == 1:
        return 1.0
    else:
        return dz


def GetDz(space):
    """Return dz, or 1.0 if Nz == 1 (collapsed dimension)."""
    if JIT_AVAILABLE:
        try:
            return _GetDz_core(space.Nz, space.dz)
        except (TypeError, ValueError, RuntimeError):
            # Fallback to standard Python
            if space.Nz == 1:
                return 1.0
            else:
                return space.dz
    else:
        if space.Nz == 1:
            return 1.0
        else:
            return space.dz


@jit(
    nopython=True, fastmath=True, cache=True
)  # fastmath OK: scalar conditional/multiply
def _GetEpsr_core(Nz, epsr):
    if Nz == 1:
        return 1.0
    else:
        return epsr


def GetEpsr(space):
    """Return epsr, or 1.0 if Nz == 1 (1D/2D case)."""
    if JIT_AVAILABLE:
        try:
            return _GetEpsr_core(space.Nz, space.epsr)
        except (TypeError, ValueError, RuntimeError):
            # Fallback to standard Python
            if space.Nz == 1:
                return 1.0
            else:
                return space.epsr
    else:
        if space.Nz == 1:
            return 1.0
        else:
            return space.epsr


# Setter functions - simple attribute modifiers
# JIT not needed for simple assignments


def SetNx(space, N):
    space.Nx = N


def SetNy(space, N):
    space.Ny = N


def SetNz(space, N):
    space.Nz = N


def SetDx(space, dl):
    space.dx = dl


def SetDy(space, dl):
    space.dy = dl


def SetDz(space, dl):
    space.dz = dl


# Width calculation functions
# These are simple calculations, JIT may help but not critical


@jit(
    nopython=True, fastmath=True, cache=True
)  # fastmath OK: scalar conditional/multiply
def _GetXWidth_core(dx, Nx):
    return dx * (Nx - 1)


def GetXWidth(space):
    """Return x-window width: dx * (Nx - 1)."""
    if JIT_AVAILABLE:
        try:
            return _GetXWidth_core(space.dx, space.Nx)
        except (TypeError, ValueError, RuntimeError):
            return space.dx * (space.Nx - 1)
    else:
        return space.dx * (space.Nx - 1)


@jit(
    nopython=True, fastmath=True, cache=True
)  # fastmath OK: scalar conditional/multiply
def _GetYWidth_core(dy, Ny):
    return dy * (Ny - 1)


def GetYWidth(space):
    """Return y-window width: dy * (Ny - 1)."""
    if JIT_AVAILABLE:
        try:
            return _GetYWidth_core(space.dy, space.Ny)
        except (TypeError, ValueError, RuntimeError):
            return space.dy * (space.Ny - 1)
    else:
        return space.dy * (space.Ny - 1)


@jit(
    nopython=True, fastmath=True, cache=True
)  # fastmath OK: scalar conditional/multiply
def _GetZWidth_core(dz, Nz):
    return dz * (Nz - 1)


def GetZWidth(space):
    """Return z-window width: dz * (Nz - 1)."""
    if JIT_AVAILABLE:
        try:
            return _GetZWidth_core(space.dz, space.Nz)
        except (TypeError, ValueError, RuntimeError):
            return space.dz * (space.Nz - 1)
    else:
        return space.dz * (space.Nz - 1)


# Array generation functions
# These use numpy operations that may not be JIT-compatible


def GetSpaceArray(N, L):
    """Return N positions evenly spaced over [-L/2, L/2]."""
    if N == 1:
        return np.array([0.0])
    else:
        # Create array from -L/2 to L/2
        x = np.linspace(-L / 2.0, L / 2.0, N)
        return x


def GetKArray(Nk, L):
    """Return k-space array in FFTW order (zero-freq at index 0)."""
    if Nk == 1:
        return np.array([0.0])
    else:
        # OLD: centered order (zero at middle) — wrong for raw fftn/ifftn
        # dk = twopi / L if L > 0 else 1.0
        # k = np.arange(Nk, dtype=float) * dk
        # k = k - k[Nk // 2]  # Center at zero
        # return k

        # NEW: FFTW order (zero-freq at index 0), matches Fortran helpers.F90
        dl = twopi / L if L > 0 else 1.0
        k_arr = np.zeros(Nk, dtype=np.float64)
        for i in range(Nk):
            if i <= Nk // 2:
                k_arr[i] = float(i) * dl
            else:
                k_arr[i] = float(i - Nk) * dl
        return k_arr


def GetXArray(space):
    if space.Nx == 1:
        return np.array([0.0])
    else:
        return GetSpaceArray(space.Nx, GetXWidth(space))


def GetYArray(space):
    if space.Ny == 1:
        return np.array([0.0])
    else:
        return GetSpaceArray(space.Ny, GetYWidth(space))


def GetZArray(space):
    if space.Nz == 1:
        return np.array([0.0])
    else:
        return GetSpaceArray(space.Nz, GetZWidth(space))


def GetKxArray(space):
    if space.Nx == 1:
        return np.array([0.0])
    else:
        return GetKArray(GetNx(space), GetXWidth(space))


def GetKyArray(space):
    if space.Ny == 1:
        return np.array([0.0])
    else:
        return GetKArray(GetNy(space), GetYWidth(space))


def GetKzArray(space):
    if space.Nz == 1:
        return np.array([0.0])
    else:
        return GetKArray(GetNz(space), GetZWidth(space))


# Differential functions for conjugate coordinate system


def GetDQx(space):
    """Return k-space step size in x: 2*pi / x_width."""
    width = GetXWidth(space)
    return twopi / width if width > 0 else 0.0


def GetDQy(space):
    """Return k-space step size in y: 2*pi / y_width."""
    width = GetYWidth(space)
    return twopi / width if width > 0 else 0.0


def GetDQz(space):
    """Return k-space step size in z: 2*pi / z_width."""
    width = GetZWidth(space)
    return twopi / width if width > 0 else 0.0


# Volume element functions


def GetDVol(space):
    """Return volume element dx * dy * dz."""
    dVol = 0.0
    dVol = GetDx(space) * GetDy(space) * GetDz(space)
    return dVol


def GetDQVol(space):
    """Return k-space volume element dqx * dqy * dqz."""
    dQVol = 0.0
    dQVol = GetDQx(space) * GetDQy(space) * GetDQz(space)
    return dQVol


# Field I/O functions
# These handle binary and text file I/O


def writefield(fnout, e, space, binmode, single, fnspace=None):
    """Write space structure and field array to file (binary or text)."""
    if fnout == "stdout" or fnout == "-":
        # Use stdout
        file_handle = sys.stdout
        close_file = False
    else:
        dirpath = os.path.dirname(fnout)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        if binmode:
            file_handle = open(fnout, "wb")
        else:
            file_handle = open(fnout, "w", encoding="utf-8")
        close_file = True

    try:
        if single:
            if binmode:
                unformatted_write_space(file_handle, space)
            else:
                WriteSpaceParams_sub(file_handle, space)
        else:
            if fnspace is not None:
                writespaceparams(fnspace, space)

        writefield_to_unit(file_handle, e, binmode)
    finally:
        if close_file:
            file_handle.close()


def readspace_only(fnin, space, binmode, single, fnspace=None):
    """Read only the space structure from a field file."""
    if fnin == "stdin" or fnin == "-":
        # Use stdin
        file_handle = sys.stdin
        close_file = False
    else:
        if binmode:
            file_handle = open(fnin, "rb")
        else:
            file_handle = open(fnin, "r", encoding="utf-8")
        close_file = True

    try:
        if single:
            if binmode:
                unformatted_read_space(file_handle, space)
            else:
                readspaceparams_sub(file_handle, space)
        else:
            if fnspace is not None:
                ReadSpaceParams(fnspace, space)
    finally:
        if close_file:
            file_handle.close()


# Helper functions for binary I/O


def writefield_to_unit(file_handle, e, binmode):
    """Write field array to an open file handle."""
    if binmode:
        # Binary mode: use numpy save
        np.save(file_handle, e)
    else:
        # Text mode: write real and imaginary parts
        for k in range(e.shape[2]):
            for j in range(e.shape[1]):
                for i in range(e.shape[0]):
                    file_handle.write(
                        f"{np.real(e[i, j, k]):25.15E} {np.imag(e[i, j, k]):25.15E}\n"
                    )


def unformatted_write_space(file_handle, space):
    """Write space structure in binary format."""
    # Write space structure fields as a dictionary
    space_dict = {
        "Dims": space.Dims,
        "Nx": space.Nx,
        "Ny": space.Ny,
        "Nz": space.Nz,
        "dx": space.dx,
        "dy": space.dy,
        "dz": space.dz,
        "epsr": space.epsr,
    }
    np.save(file_handle, space_dict)


def unformatted_read_space(file_handle, space):
    """Read space structure from binary format."""
    space_dict = np.load(file_handle, allow_pickle=True).item()
    space.Dims = int(space_dict["Dims"])
    space.Nx = int(space_dict["Nx"])
    space.Ny = int(space_dict["Ny"])
    space.Nz = int(space_dict["Nz"])
    space.dx = float(space_dict["dx"])
    space.dy = float(space_dict["dy"])
    space.dz = float(space_dict["dz"])
    space.epsr = float(space_dict["epsr"])


def initialize_field(e):
    """Zero out a complex field array in-place."""
    e[:] = 0.0 + 0.0j


def unformatted_write_e(file_handle, e):
    """Write field array in binary format."""
    np.save(file_handle, e)


def unformatted_read_e(file_handle, e):
    """Read field array from binary format in-place."""
    e[:] = np.load(file_handle)


def readfield_from_unit(file_handle, e, binmode):
    """Read field array from an open file handle."""
    if binmode:
        unformatted_read_e(file_handle, e)
    else:
        # Text mode: read real and imaginary parts
        for k in range(e.shape[2]):
            for j in range(e.shape[1]):
                for i in range(e.shape[0]):
                    line = file_handle.readline()
                    if not line:
                        raise ValueError("Unexpected end of file while reading field")
                    parts = line.split()
                    if len(parts) < 2:
                        raise ValueError(f"Invalid field data format: {line}")
                    re = float(parts[0])
                    im = float(parts[1])
                    e[i, j, k] = re + 1j * im


def readfield(fnin, e, space, binmode, single, fnspace=None):
    """Read space structure and field array from file. Returns the field array."""
    if fnin == "stdin" or fnin == "-":
        # Use stdin
        file_handle = sys.stdin
        close_file = False
    else:
        if binmode:
            file_handle = open(fnin, "rb")
        else:
            file_handle = open(fnin, "r", encoding="utf-8")
        close_file = True

    try:
        # Read space parameters
        if single:
            if binmode:
                unformatted_read_space(file_handle, space)
            else:
                readspaceparams_sub(file_handle, space)
        else:
            if fnspace is not None:
                ReadSpaceParams(fnspace, space)

        # Allocate or resize field array
        nx = GetNx(space)
        ny = GetNy(space)
        nz = GetNz(space)

        # Determine if we need a new array
        need_new_array = (e is None) or (e.shape != (nx, ny, nz))

        if need_new_array:
            # Allocate new array
            e = np.zeros((nx, ny, nz), dtype=complex)
        else:
            # Initialize existing array to zero
            initialize_field(e)

        # Read field data
        readfield_from_unit(file_handle, e, binmode)

    finally:
        if close_file:
            file_handle.close()

    return e
