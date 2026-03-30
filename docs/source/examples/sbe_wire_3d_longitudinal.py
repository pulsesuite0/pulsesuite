"""
SBE 3D longitudinal field decomposition example.

Simulates quantum wire response to plane wave electromagnetic fields,
tracking longitudinal E-field components from polarization and charge density.
"""

import os

import numpy as np
from numba import jit

from pulsesuite.PSTD3D.rhoPJ import QuantumWire
from pulsesuite.PSTD3D.SBEs import InitializeSBE
from pulsesuite.PSTD3D.typespace import (
    GetKyArray,
    GetNx,
    GetNy,
    GetNz,
    GetXArray,
    GetYArray,
    ReadSpaceParams,
    ss,
)
from pulsesuite.PSTD3D.usefulsubs import WriteIT2D

# Physical constants
c0 = 299792458.0  # Speed of light (m/s)
pi = np.pi
twopi = 2.0 * pi

# Global parameters (matching Fortran defaults)
DEFAULT_DT = 10e-18  # Time step (s)
DEFAULT_NT = 10000  # Number of time steps
DEFAULT_TP = 50e-15  # Pulse peak time (s)
DEFAULT_LAM = 800e-9  # Wavelength (m)
DEFAULT_TW = 10e-15  # Pulse width (s)
DEFAULT_N0 = 3.1  # Background refractive index
DEFAULT_E0X = 2e8  # Peak Ex field (V/m)
DEFAULT_E0Y = 0.0  # Peak Ey field (V/m)
DEFAULT_E0Z = 0.0  # Peak Ez field (V/m)


def initializefields(
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
    F11, F12, F13, F14, F15, F16, F17, F18, F19, F20,
):
    """Initialize all 20 field arrays to zero."""
    F1[:] = 0.0 + 0.0j
    F2[:] = 0.0 + 0.0j
    F3[:] = 0.0 + 0.0j
    F4[:] = 0.0 + 0.0j
    F5[:] = 0.0 + 0.0j
    F6[:] = 0.0 + 0.0j
    F7[:] = 0.0 + 0.0j
    F8[:] = 0.0 + 0.0j
    F9[:] = 0.0 + 0.0j
    F10[:] = 0.0 + 0.0j
    F11[:] = 0.0 + 0.0j
    F12[:] = 0.0 + 0.0j
    F13[:] = 0.0 + 0.0j
    F14[:] = 0.0 + 0.0j
    F15[:] = 0.0 + 0.0j
    F16[:] = 0.0 + 0.0j
    F17[:] = 0.0 + 0.0j
    F18[:] = 0.0 + 0.0j
    F19[:] = 0.0 + 0.0j
    F20[:] = 0.0 + 0.0j


@jit(nopython=True, cache=True, fastmath=True)
def _compute_plane_wave_jit(u, Emax0, w0, tw):
    """JIT-compiled Gaussian envelope with super-Gaussian cutoff."""
    N = len(u)
    result = np.zeros(N, dtype=np.complex128)
    w0tw = w0 * tw
    w0tw20 = (2.0 * w0tw) ** 20

    for i in range(N):
        u_val = u[i]
        gaussian = np.exp(-(u_val**2) / w0tw**2)
        supergaussian = np.exp(-(u_val**20) / w0tw20)
        carrier = np.cos(u_val)
        result[i] = Emax0 * gaussian * carrier * supergaussian

    return result


def MakePlaneWaveX(Ey, space, t, Emax0, lam, tw, tp):
    """Create a y-polarized plane wave propagating in the x-direction."""
    n0 = 3.1
    w0 = twopi * c0 / lam
    k0 = twopi / lam * n0
    x = GetXArray(space)
    u = k0 * x - w0 * (t - tp)
    field = _compute_plane_wave_jit(u, Emax0, w0, tw)

    Nx = GetNx(space)
    Ny = GetNy(space)
    Nz = GetNz(space)

    for k in range(Nz):
        for j in range(Ny):
            Ey[:, j, k] = field


def MakePlaneWaveY(Ex, space, t, Emax0, lam, tw, tp):
    """Create an x-polarized plane wave propagating in the y-direction."""
    n0 = 3.1
    w0 = twopi * c0 / lam
    k0 = twopi / lam * n0
    y = GetYArray(space)
    u = k0 * y - w0 * (t - tp)
    field = _compute_plane_wave_jit(u, Emax0, w0, tw)

    Nx = GetNx(space)
    Ny = GetNy(space)
    Nz = GetNz(space)

    for k in range(Nz):
        for i in range(Nx):
            Ex[i, :, k] = field


def MakePlaneWaveZ(Ez, space, t, Emax0, lam, tw, tp):
    """Create a z-polarized plane wave propagating in the x-direction."""
    n0 = 3.1
    w0 = twopi * c0 / lam
    k0 = twopi / lam * n0
    x = GetXArray(space)
    u = k0 * x - w0 * (t - tp)
    field = _compute_plane_wave_jit(u, Emax0, w0, tw)

    Nx = GetNx(space)
    Ny = GetNy(space)
    Nz = GetNz(space)

    for k in range(Nz):
        for j in range(Ny):
            Ez[:, j, k] = field


def MakePlaneWaveTemporal(Ex, t, Emax0, lam, tw, tp):
    """Create a temporally-varying spatially-uniform field."""
    w0 = 2.0 * pi * c0 / lam
    u = w0 * (t - tp)
    env = Emax0 * np.cos(u) * np.exp(-((t - tp) ** 2) / tw**2)
    Ex[:, :, :] = env


def ElongSeparate(space, Ex, Ey, Ez, Exl, Eyl, Ezl):
    """Separate longitudinal and transverse electric field components (stub)."""
    Exl[:] = 0.0 + 0.0j
    Eyl[:] = 0.0 + 0.0j
    Ezl[:] = 0.0 + 0.0j


def SBETest(
    space_params_file="params/space.params",
    dt=DEFAULT_DT,
    Nt=DEFAULT_NT,
    tp=DEFAULT_TP,
    lam=DEFAULT_LAM,
    tw=DEFAULT_TW,
    n0=DEFAULT_N0,
    E0x=DEFAULT_E0X,
    E0y=DEFAULT_E0Y,
    E0z=DEFAULT_E0Z,
    output_dir="fields",
    write_2d_slices=True,
    slice_interval=10,
):
    """
    Main SBE test program with full longitudinal field analysis.

    Simulates quantum wire response to plane wave electromagnetic fields with
    complete tracking of longitudinal field components from polarization (P),
    charge density (Rho), and their combination.

    Returns
    -------
    dict
        Dictionary containing final field arrays and statistics.
    """
    print("=" * 70)
    print("SBE Test Program - Full Longitudinal Field Analysis")
    print("=" * 70)

    # Read spatial grid structure
    space = ss(Dims=0, Nx=0, Ny=0, Nz=0, dx=0.0, dy=0.0, dz=0.0, epsr=1.0)
    ReadSpaceParams(space_params_file, space)

    Nx = GetNx(space)
    Ny = GetNy(space)
    Nz = GetNz(space)

    print(f"Grid dimensions: Nx={Nx}, Ny={Ny}, Nz={Nz}")
    print(f"Time steps: {Nt}, dt={dt*1e15:.2f} fs")
    print(f"Wavelength: {lam*1e9:.1f} nm, Pulse width: {tw*1e15:.2f} fs")
    print(f"Peak fields: E0x={E0x:.2e} V/m, E0y={E0y:.2e} V/m, E0z={E0z:.2e} V/m")

    # Allocate Maxwell 3D arrays
    Ex = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Ey = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Ez = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Jx = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Jy = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Jz = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Rho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Exl = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Eyl = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Ezl = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    # Longitudinal field decomposition arrays
    ExlfromPRho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    EylfromPRho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    EzlfromPRho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    ExlfromP = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    EylfromP = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    EzlfromP = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    ExlfromRho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    EylfromRho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    EzlfromRho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    RhoBound = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    # Allocate 1D arrays for quantum wire
    rr = np.zeros(Nx, dtype=np.float64)
    qrr = np.zeros(Nx, dtype=np.float64)

    # Initialize max/min tracking variables
    Ex_max, Ex_min = -np.inf, np.inf
    Ey_max, Ey_min = -np.inf, np.inf
    Ez_max, Ez_min = -np.inf, np.inf
    Rho_max, Rho_min = -np.inf, np.inf
    Jx_max, Jx_min = -np.inf, np.inf
    Jy_max, Jy_min = -np.inf, np.inf
    Jz_max, Jz_min = -np.inf, np.inf
    Exl_max, Exl_min = -np.inf, np.inf
    Eyl_max, Eyl_min = -np.inf, np.inf
    Ezl_max, Ezl_min = -np.inf, np.inf
    ExlPRho_max, ExlPRho_min = -np.inf, np.inf
    EylPRho_max, EylPRho_min = -np.inf, np.inf
    EzlPRho_max, EzlPRho_min = -np.inf, np.inf
    ExlP_max, ExlP_min = -np.inf, np.inf
    EylP_max, EylP_min = -np.inf, np.inf
    EzlP_max, EzlP_min = -np.inf, np.inf
    ExlRho_max, ExlRho_min = -np.inf, np.inf
    EylRho_max, EylRho_min = -np.inf, np.inf
    EzlRho_max, EzlRho_min = -np.inf, np.inf

    # Initialize the Maxwell arrays
    initializefields(
        Ex, Ey, Ez, Jx, Jy, Jz, Rho,
        Exl, Eyl, Ezl,
        ExlfromPRho, EylfromPRho, EzlfromPRho,
        ExlfromP, EylfromP, EzlfromP,
        ExlfromRho, EylfromRho, EzlfromRho,
        RhoBound,
    )

    # Calculate angular frequencies and optical cycle (for Y-direction)
    w0y = twopi * c0 / lam
    k0y = twopi / lam * n0
    Tcy = lam / c0

    # Calculate maximum field possible during simulation
    Emax = np.sqrt(E0x**2 + E0y**2 + E0z**2)
    print(f"Maximum field: {Emax:.2e} V/m")

    # Calculate real-space and q-space arrays
    rr = GetYArray(space)
    qrr = GetKyArray(space)

    # Initialize the SBEs
    print("Initializing SBE solver...")
    InitializeSBE(qrr, rr, 0.0, Emax, lam, 4, True)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output files
    output_files = {
        "Ex": f"{output_dir}/Ex.dat",
        "Ey": f"{output_dir}/Ey.dat",
        "Ez": f"{output_dir}/Ez.dat",
        "Jx": f"{output_dir}/Jx.dat",
        "Jy": f"{output_dir}/Jy.dat",
        "Jz": f"{output_dir}/Jz.dat",
        "Rho": f"{output_dir}/Rho.dat",
        "Eywireloc": f"{output_dir}/Eywireloc.dat",
        "Exl": f"{output_dir}/Exl.dat",
        "Eyl": f"{output_dir}/Eyl.dat",
        "Ezl": f"{output_dir}/Ezl.dat",
        "ExlfromPRho": f"{output_dir}/ExlfromPRho.dat",
        "EylfromPRho": f"{output_dir}/EylfromPRho.dat",
        "EzlfromPRho": f"{output_dir}/EzlfromPRho.dat",
        "ExlfromP": f"{output_dir}/ExlfromP.dat",
        "EylfromP": f"{output_dir}/EylfromP.dat",
        "EzlfromP": f"{output_dir}/EzlfromP.dat",
        "ExlfromRho": f"{output_dir}/ExlfromRho.dat",
        "EylfromRho": f"{output_dir}/EylfromRho.dat",
        "EzlfromRho": f"{output_dir}/EzlfromRho.dat",
        "final_max_min": f"{output_dir}/final_max_min.dat",
        "RhoBound": f"{output_dir}/RhoBound.dat",
    }

    print("Starting time loop...")
    print("=" * 70)

    # Open all output files with context managers
    with open(output_files["Ex"], "w", encoding="utf-8") as f_Ex, \
         open(output_files["Ey"], "w", encoding="utf-8") as f_Ey, \
         open(output_files["Ez"], "w", encoding="utf-8") as f_Ez, \
         open(output_files["Jx"], "w", encoding="utf-8") as f_Jx, \
         open(output_files["Jy"], "w", encoding="utf-8") as f_Jy, \
         open(output_files["Jz"], "w", encoding="utf-8") as f_Jz, \
         open(output_files["Rho"], "w", encoding="utf-8") as f_Rho, \
         open(output_files["Eywireloc"], "w", encoding="utf-8") as f_Eywire, \
         open(output_files["Exl"], "w", encoding="utf-8") as f_Exl, \
         open(output_files["Eyl"], "w", encoding="utf-8") as f_Eyl, \
         open(output_files["Ezl"], "w", encoding="utf-8") as f_Ezl, \
         open(output_files["ExlfromPRho"], "w", encoding="utf-8") as f_ExlPR, \
         open(output_files["EylfromPRho"], "w", encoding="utf-8") as f_EylPR, \
         open(output_files["EzlfromPRho"], "w", encoding="utf-8") as f_EzlPR, \
         open(output_files["ExlfromP"], "w", encoding="utf-8") as f_ExlP, \
         open(output_files["EylfromP"], "w", encoding="utf-8") as f_EylP, \
         open(output_files["EzlfromP"], "w", encoding="utf-8") as f_EzlP, \
         open(output_files["ExlfromRho"], "w", encoding="utf-8") as f_ExlR, \
         open(output_files["EylfromRho"], "w", encoding="utf-8") as f_EylR, \
         open(output_files["EzlfromRho"], "w", encoding="utf-8") as f_EzlR, \
         open(output_files["final_max_min"], "w", encoding="utf-8") as f_mm, \
         open(output_files["RhoBound"], "w", encoding="utf-8") as f_RhoB:

        file_handles = {
            "Ex": f_Ex, "Ey": f_Ey, "Ez": f_Ez,
            "Jx": f_Jx, "Jy": f_Jy, "Jz": f_Jz,
            "Rho": f_Rho, "Eywireloc": f_Eywire,
            "Exl": f_Exl, "Eyl": f_Eyl, "Ezl": f_Ezl,
            "ExlfromPRho": f_ExlPR, "EylfromPRho": f_EylPR,
            "EzlfromPRho": f_EzlPR,
            "ExlfromP": f_ExlP, "EylfromP": f_EylP, "EzlfromP": f_EzlP,
            "ExlfromRho": f_ExlR, "EylfromRho": f_EylR, "EzlfromRho": f_EzlR,
            "final_max_min": f_mm, "RhoBound": f_RhoB,
        }

        # Time loop
        t = 0.0
        for n in range(1, Nt + 1):
            if n % 1000 == 0:
                print(f"Step {n}/{Nt}")

            # Create plane wave excitation (Y-direction propagation)
            MakePlaneWaveY(Ex, space, t, E0x, lam, tw, tp)

            # Update quantum wire response with full longitudinal field decomposition
            QuantumWire(
                space, dt, n,
                Ex, Ey, Ez, Jx, Jy, Jz, Rho,
                ExlfromPRho, EylfromPRho, EzlfromPRho,
                ExlfromP, EylfromP, EzlfromP,
                ExlfromRho, EylfromRho, EzlfromRho,
                RhoBound,
            )

            # Separate longitudinal field components (Helmholtz decomposition stub)
            ElongSeparate(space, Ex, Ey, Ez, Exl, Eyl, Ezl)

            # Print condensed diagnostics every 1000 steps
            if n % 1000 == 0:
                print(f"  Ex-max          = {np.max(np.abs(Ex)):.4e}")
                print(f"  Ey-max          = {np.max(np.abs(Ey)):.4e}")
                print(f"  Ez-max          = {np.max(np.abs(Ez)):.4e}")
                print(f"  Rho-max         = {np.max(np.abs(Rho)):.4e}")
                print(f"  Jx-max          = {np.max(np.abs(Jx)):.4e}")
                print(f"  Jy-max          = {np.max(np.abs(Jy)):.4e}")
                print(f"  Jz-max          = {np.max(np.abs(Jz)):.4e}")
                print(f"  ExlfromPRho-max = {np.max(np.real(ExlfromPRho)):.4e}")
                print(f"  EylfromPRho-max = {np.max(np.real(EylfromPRho)):.4e}")
                print(f"  EzlfromPRho-max = {np.max(np.real(EzlfromPRho)):.4e}")
                print(f"  ExlfromP-max    = {np.max(np.real(ExlfromP)):.4e}")
                print(f"  EylfromP-max    = {np.max(np.real(EylfromP)):.4e}")
                print(f"  EzlfromP-max    = {np.max(np.real(EzlfromP)):.4e}")
                print(f"  ExlfromRho-max  = {np.max(np.real(ExlfromRho)):.4e}")
                print(f"  EylfromRho-max  = {np.max(np.real(EylfromRho)):.4e}")
                print(f"  EzlfromRho-max  = {np.max(np.real(EzlfromRho)):.4e}")
                print(f"  RhoBound-max    = {np.max(np.real(RhoBound)):.4e}")
                print(f"  Ex:     max={Ex_max:.4e}  min={Ex_min:.4e}")
                print(f"  Ey:     max={Ey_max:.4e}  min={Ey_min:.4e}")
                print(f"  Rho:    max={Rho_max:.4e}  min={Rho_min:.4e}")

            # Update max/min values
            Ex_max = max(Ex_max, np.max(np.abs(Ex)))
            Ex_min = min(Ex_min, np.min(np.real(Ex)))
            Ey_max = max(Ey_max, np.max(np.abs(Ey)))
            Ey_min = min(Ey_min, np.min(np.real(Ey)))
            Ez_max = max(Ez_max, np.max(np.abs(Ez)))
            Ez_min = min(Ez_min, np.min(np.real(Ez)))

            Rho_max = max(Rho_max, np.max(np.abs(Rho)))
            Rho_min = min(Rho_min, np.min(np.real(Rho)))
            Jx_max = max(Jx_max, np.max(np.abs(Jx)))
            Jx_min = min(Jx_min, np.min(np.real(Jx)))
            Jy_max = max(Jy_max, np.max(np.abs(Jy)))
            Jy_min = min(Jy_min, np.min(np.real(Jy)))
            Jz_max = max(Jz_max, np.max(np.abs(Jz)))
            Jz_min = min(Jz_min, np.min(np.real(Jz)))

            Exl_max = max(Exl_max, np.max(np.abs(Exl)))
            Exl_min = min(Exl_min, np.min(np.real(Exl)))
            Eyl_max = max(Eyl_max, np.max(np.abs(Eyl)))
            Eyl_min = min(Eyl_min, np.min(np.real(Eyl)))
            Ezl_max = max(Ezl_max, np.max(np.abs(Ezl)))
            Ezl_min = min(Ezl_min, np.min(np.real(Ezl)))

            ExlPRho_max = max(ExlPRho_max, np.max(np.real(ExlfromPRho)))
            ExlPRho_min = min(ExlPRho_min, np.min(np.real(ExlfromPRho)))
            EylPRho_max = max(EylPRho_max, np.max(np.real(EylfromPRho)))
            EylPRho_min = min(EylPRho_min, np.min(np.real(EylfromPRho)))
            EzlPRho_max = max(EzlPRho_max, np.max(np.real(EzlfromPRho)))
            EzlPRho_min = min(EzlPRho_min, np.min(np.real(EzlfromPRho)))

            ExlP_max = max(ExlP_max, np.max(np.real(ExlfromP)))
            ExlP_min = min(ExlP_min, np.min(np.real(ExlfromP)))
            EylP_max = max(EylP_max, np.max(np.real(EylfromP)))
            EylP_min = min(EylP_min, np.min(np.real(EylfromP)))
            EzlP_max = max(EzlP_max, np.max(np.real(EzlfromP)))
            EzlP_min = min(EzlP_min, np.min(np.real(EzlfromP)))

            ExlRho_max = max(ExlRho_max, np.max(np.real(ExlfromRho)))
            ExlRho_min = min(ExlRho_min, np.min(np.real(ExlfromRho)))
            EylRho_max = max(EylRho_max, np.max(np.real(EylfromRho)))
            EylRho_min = min(EylRho_min, np.min(np.real(EylfromRho)))
            EzlRho_max = max(EzlRho_max, np.max(np.real(EzlfromRho)))
            EzlRho_min = min(EzlRho_min, np.min(np.real(EzlfromRho)))

            # Write time series data to files
            file_handles["Ex"].write(f"{t:.15e} {np.real(Ex[0, 0, 0]):.15e}\n")
            file_handles["Ey"].write(f"{t:.15e} {np.real(Ey[0, Ny//2, 0]):.15e}\n")
            file_handles["Ez"].write(f"{t:.15e} {np.real(Ez[0, 0, 0]):.15e}\n")
            file_handles["Jx"].write(f"{t:.15e} {np.real(Jx[0, Ny//2, 0]):.15e}\n")
            file_handles["Jy"].write(f"{t:.15e} {np.real(Jy[0, Ny//2, 0]):.15e}\n")
            file_handles["Jz"].write(f"{t:.15e} {np.real(Jz[0, Ny//2, 0]):.15e}\n")
            file_handles["Rho"].write(f"{t:.15e} {np.real(Rho[0, Ny//2, 0]):.15e}\n")
            file_handles["Eywireloc"].write(
                f"{t:.15e} {np.real(Ey[Nx//4, Ny//2, Nz//2]):.15e}\n"
            )

            file_handles["ExlfromPRho"].write(
                f"{t:.15e} {np.real(ExlfromPRho[0, 0, 0]):.15e}\n"
            )
            file_handles["EylfromPRho"].write(
                f"{t:.15e} {np.real(EylfromPRho[0, 0, 0]):.15e}\n"
            )
            file_handles["EzlfromPRho"].write(
                f"{t:.15e} {np.real(EzlfromPRho[0, 0, 0]):.15e}\n"
            )

            file_handles["ExlfromP"].write(f"{t:.15e} {np.real(ExlfromP[0, 0, 0]):.15e}\n")
            file_handles["EylfromP"].write(f"{t:.15e} {np.real(EylfromP[0, 0, 0]):.15e}\n")
            file_handles["EzlfromP"].write(f"{t:.15e} {np.real(EzlfromP[0, 0, 0]):.15e}\n")

            file_handles["ExlfromRho"].write(
                f"{t:.15e} {np.real(ExlfromRho[0, 0, 0]):.15e}\n"
            )
            file_handles["EylfromRho"].write(
                f"{t:.15e} {np.real(EylfromRho[0, 0, 0]):.15e}\n"
            )
            file_handles["EzlfromRho"].write(
                f"{t:.15e} {np.real(EzlfromRho[0, 0, 0]):.15e}\n"
            )

            file_handles["RhoBound"].write(
                f"{t:.15e} {np.real(RhoBound[0, Ny//2, 0]):.15e}\n"
            )

            # Write 2D slices at intervals
            if write_2d_slices and (n % slice_interval == 0):
                # ExlfromPRho slices
                WriteIT2D(np.real(ExlfromPRho[:, :, Nz // 2]), f"ExPRho.{n}.z")
                WriteIT2D(np.real(EylfromPRho[:, :, Nz // 2]), f"EyPRho.{n}.z")
                WriteIT2D(np.real(EzlfromPRho[:, :, Nz // 2]), f"EzPRho.{n}.z")

                WriteIT2D(np.real(ExlfromPRho[:, Ny // 2, :]), f"ExPRho.{n}.y")
                WriteIT2D(np.real(EylfromPRho[:, Ny // 2, :]), f"EyPRho.{n}.y")
                WriteIT2D(np.real(EzlfromPRho[:, Ny // 2, :]), f"EzPRho.{n}.y")

                WriteIT2D(np.real(ExlfromPRho[Nx // 2, :, :]), f"ExPRho.{n}.x")
                WriteIT2D(np.real(EylfromPRho[Nx // 2, :, :]), f"EyPRho.{n}.x")
                WriteIT2D(np.real(EzlfromPRho[Nx // 2, :, :]), f"EzPRho.{n}.x")

                # Rho slices
                WriteIT2D(np.real(Rho[:, :, Nz // 2]), f"Rho.{n}.z")
                WriteIT2D(np.real(Rho[:, Ny // 2, :]), f"Rho.{n}.y")
                WriteIT2D(np.real(Rho[Nx // 2, :, :]), f"Rho.{n}.x")

                # ExlfromP slices
                WriteIT2D(np.real(ExlfromP[:, :, Nz // 2]), f"ExP.{n}.z")
                WriteIT2D(np.real(EylfromP[:, :, Nz // 2]), f"EyP.{n}.z")
                WriteIT2D(np.real(EzlfromP[:, :, Nz // 2]), f"EzP.{n}.z")

                WriteIT2D(np.real(ExlfromP[:, Ny // 2, :]), f"ExP.{n}.y")
                WriteIT2D(np.real(EylfromP[:, Ny // 2, :]), f"EyP.{n}.y")
                WriteIT2D(np.real(EzlfromP[:, Ny // 2, :]), f"EzP.{n}.y")

                WriteIT2D(np.real(ExlfromP[Nx // 2, :, :]), f"ExP.{n}.x")
                WriteIT2D(np.real(EylfromP[Nx // 2, :, :]), f"EyP.{n}.x")
                WriteIT2D(np.real(EzlfromP[Nx // 2, :, :]), f"EzP.{n}.x")

                # ExlfromRho slices
                WriteIT2D(np.real(ExlfromRho[:, :, Nz // 2]), f"ExRho.{n}.z")
                WriteIT2D(np.real(EylfromRho[:, :, Nz // 2]), f"EyRho.{n}.z")
                WriteIT2D(np.real(EzlfromRho[:, :, Nz // 2]), f"EzRho.{n}.z")

                WriteIT2D(np.real(ExlfromRho[:, Ny // 2, :]), f"ExRho.{n}.y")
                WriteIT2D(np.real(EylfromRho[:, Ny // 2, :]), f"EyRho.{n}.y")
                WriteIT2D(np.real(EzlfromRho[:, Ny // 2, :]), f"EzRho.{n}.y")

                WriteIT2D(np.real(ExlfromRho[Nx // 2, :, :]), f"ExRho.{n}.x")
                WriteIT2D(np.real(EylfromRho[Nx // 2, :, :]), f"EyRho.{n}.x")
                WriteIT2D(np.real(EzlfromRho[Nx // 2, :, :]), f"EzRho.{n}.x")

                # Exl slices
                WriteIT2D(np.real(Exl[:, :, Nz // 2]), f"Exl.{n}.z")
                WriteIT2D(np.real(Eyl[:, :, Nz // 2]), f"Eyl.{n}.z")
                WriteIT2D(np.real(Ezl[:, :, Nz // 2]), f"Ezl.{n}.z")

                WriteIT2D(np.real(Exl[:, Ny // 2, :]), f"Exl.{n}.y")
                WriteIT2D(np.real(Eyl[:, Ny // 2, :]), f"Eyl.{n}.y")
                WriteIT2D(np.real(Ezl[:, Ny // 2, :]), f"Ezl.{n}.y")

                WriteIT2D(np.real(Exl[Nx // 2, :, :]), f"Exl.{n}.x")
                WriteIT2D(np.real(Eyl[Nx // 2, :, :]), f"Eyl.{n}.x")
                WriteIT2D(np.real(Ezl[Nx // 2, :, :]), f"Ezl.{n}.x")

                # RhoBound slices
                WriteIT2D(np.real(RhoBound[:, :, Nz // 2]), f"RhoB.{n}.z")
                WriteIT2D(np.real(RhoBound[:, Ny // 2, :]), f"RhoB.{n}.y")
                WriteIT2D(np.real(RhoBound[Nx // 2, :, :]), f"RhoB.{n}.x")

            # Write max/min summary every 1000 steps
            if n % 1000 == 0:
                fh = file_handles["final_max_min"]
                fh.write(f"Max/Min Field Values Over {n} Time Steps\n")
                fh.write(f"Ex:           Max = {Ex_max:.6e}  Min = {Ex_min:.6e}\n")
                fh.write(f"Ey:           Max = {Ey_max:.6e}  Min = {Ey_min:.6e}\n")
                fh.write(f"Ez:           Max = {Ez_max:.6e}  Min = {Ez_min:.6e}\n")
                fh.write(f"Rho:          Max = {Rho_max:.6e}  Min = {Rho_min:.6e}\n")
                fh.write(f"Jx:           Max = {Jx_max:.6e}  Min = {Jx_min:.6e}\n")
                fh.write(f"Jy:           Max = {Jy_max:.6e}  Min = {Jy_min:.6e}\n")
                fh.write(f"Jz:           Max = {Jz_max:.6e}  Min = {Jz_min:.6e}\n")
                fh.write(f"Exl:          Max = {Exl_max:.6e}  Min = {Exl_min:.6e}\n")
                fh.write(f"Eyl:          Max = {Eyl_max:.6e}  Min = {Eyl_min:.6e}\n")
                fh.write(f"Ezl:          Max = {Ezl_max:.6e}  Min = {Ezl_min:.6e}\n")
                fh.write(
                    f"ExlfromPRho:  Max = {ExlPRho_max:.6e}  Min = {ExlPRho_min:.6e}\n"
                )
                fh.write(
                    f"EylfromPRho:  Max = {EylPRho_max:.6e}  Min = {EylPRho_min:.6e}\n"
                )
                fh.write(
                    f"EzlfromPRho:  Max = {EzlPRho_max:.6e}  Min = {EzlPRho_min:.6e}\n"
                )
                fh.write(f"ExlfromP:     Max = {ExlP_max:.6e}  Min = {ExlP_min:.6e}\n")
                fh.write(f"EylfromP:     Max = {EylP_max:.6e}  Min = {EylP_min:.6e}\n")
                fh.write(f"EzlfromP:     Max = {EzlP_max:.6e}  Min = {EzlP_min:.6e}\n")
                fh.write(f"ExlfromRho:   Max = {ExlRho_max:.6e}  Min = {ExlRho_min:.6e}\n")
                fh.write(f"EylfromRho:   Max = {EylRho_max:.6e}  Min = {EylRho_min:.6e}\n")
                fh.write(f"EzlfromRho:   Max = {EzlRho_max:.6e}  Min = {EzlRho_min:.6e}\n")
                fh.write("\n")

            # Increment time
            t = t + dt

        # Write final max/min to file
        fh = file_handles["final_max_min"]
        fh.write("Max/Min Field Values Over All Time Steps\n")
        fh.write(f"Ex:           Max = {Ex_max:.6e}  Min = {Ex_min:.6e}\n")
        fh.write(f"Ey:           Max = {Ey_max:.6e}  Min = {Ey_min:.6e}\n")
        fh.write(f"Ez:           Max = {Ez_max:.6e}  Min = {Ez_min:.6e}\n")
        fh.write(f"Rho:          Max = {Rho_max:.6e}  Min = {Rho_min:.6e}\n")
        fh.write(f"Jx:           Max = {Jx_max:.6e}  Min = {Jx_min:.6e}\n")
        fh.write(f"Jy:           Max = {Jy_max:.6e}  Min = {Jy_min:.6e}\n")
        fh.write(f"Jz:           Max = {Jz_max:.6e}  Min = {Jz_min:.6e}\n")
        fh.write(f"Exl:          Max = {Exl_max:.6e}  Min = {Exl_min:.6e}\n")
        fh.write(f"Eyl:          Max = {Eyl_max:.6e}  Min = {Eyl_min:.6e}\n")
        fh.write(f"Ezl:          Max = {Ezl_max:.6e}  Min = {Ezl_min:.6e}\n")
        fh.write(f"ExlfromPRho:  Max = {ExlPRho_max:.6e}  Min = {ExlPRho_min:.6e}\n")
        fh.write(f"EylfromPRho:  Max = {EylPRho_max:.6e}  Min = {EylPRho_min:.6e}\n")
        fh.write(f"EzlfromPRho:  Max = {EzlPRho_max:.6e}  Min = {EzlPRho_min:.6e}\n")
        fh.write(f"ExlfromP:     Max = {ExlP_max:.6e}  Min = {ExlP_min:.6e}\n")
        fh.write(f"EylfromP:     Max = {EylP_max:.6e}  Min = {EylP_min:.6e}\n")
        fh.write(f"EzlfromP:     Max = {EzlP_max:.6e}  Min = {EzlP_min:.6e}\n")
        fh.write(f"ExlfromRho:   Max = {ExlRho_max:.6e}  Min = {ExlRho_min:.6e}\n")
        fh.write(f"EylfromRho:   Max = {EylRho_max:.6e}  Min = {EylRho_min:.6e}\n")
        fh.write(f"EzlfromRho:   Max = {EzlRho_max:.6e}  Min = {EzlRho_min:.6e}\n")

    print("=" * 70)
    print("Simulation complete!")
    print(f"Output files written to: {output_dir}/")
    print("=" * 70)

    # Return field arrays and statistics
    return {
        "Ex": Ex, "Ey": Ey, "Ez": Ez,
        "Jx": Jx, "Jy": Jy, "Jz": Jz,
        "Rho": Rho,
        "Exl": Exl, "Eyl": Eyl, "Ezl": Ezl,
        "ExlfromPRho": ExlfromPRho, "EylfromPRho": EylfromPRho,
        "EzlfromPRho": EzlfromPRho,
        "ExlfromP": ExlfromP, "EylfromP": EylfromP, "EzlfromP": EzlfromP,
        "ExlfromRho": ExlfromRho, "EylfromRho": EylfromRho,
        "EzlfromRho": EzlfromRho,
        "RhoBound": RhoBound,
        "stats": {
            "Ex_max": Ex_max, "Ex_min": Ex_min,
            "Ey_max": Ey_max, "Ey_min": Ey_min,
            "Ez_max": Ez_max, "Ez_min": Ez_min,
            "Rho_max": Rho_max, "Rho_min": Rho_min,
        },
    }


if __name__ == "__main__":
    SBETest()
