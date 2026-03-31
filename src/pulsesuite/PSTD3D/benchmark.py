"""
Benchmarking and run-summary for PulseSuite simulations.

Writes a ``run_summary.txt`` file inside the current working directory
(expected to be the timestamped run directory) containing:

* Wall-clock timing (total, init, time-loop)
* Hardware info (CPU, cores, RAM, GPU if present)
* Compute backend availability and dispatch chain
* Python / NumPy / Numba / CuPy version info
* Performance metrics (time per step, steps/s)
"""

import os
import platform
import sys
import time
from datetime import datetime

# ── Timer ────────────────────────────────────────────────────────────

_timers: dict = {}


def timer_start(label="total"):
    """Record the start time for *label*."""
    _timers[label] = {"start": time.perf_counter(), "end": None}


def timer_stop(label="total"):
    """Record the end time for *label*."""
    if label in _timers:
        _timers[label]["end"] = time.perf_counter()


def _elapsed(label):
    """Return elapsed seconds for *label*, or None."""
    t = _timers.get(label)
    if t and t["start"] is not None and t["end"] is not None:
        return t["end"] - t["start"]
    return None


# ── Per-function profiler ────────────────────────────────────────────

_profile_accum: dict = {}  # label -> {"total": float, "calls": int}


def profile_start():
    """Return a timestamp for use with :func:`profile_record`."""
    return time.perf_counter()


def profile_record(label, start_time):
    """Accumulate elapsed time since *start_time* under *label*."""
    elapsed = time.perf_counter() - start_time
    if label not in _profile_accum:
        _profile_accum[label] = {"total": 0.0, "calls": 0}
    _profile_accum[label]["total"] += elapsed
    _profile_accum[label]["calls"] += 1


def get_profile_data():
    """Return a copy of the accumulated profile data."""
    return dict(_profile_accum)


# ── Hardware detection ───────────────────────────────────────────────


def _cpu_info():
    """Return a short CPU description."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or "unknown"


def _gpu_info():
    """Try to detect GPU via nvidia-smi or CUDA."""
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, OSError):
        pass
    try:
        from numba import cuda

        if cuda.is_available():
            dev = cuda.get_current_device()
            return f"{dev.name} (compute {dev.compute_capability})"
    except Exception:
        pass
    return "none detected"


def _mem_info():
    """Return total RAM in GB."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return f"{kb / 1024 / 1024:.1f} GB"
    except (OSError, ValueError):
        pass
    return "unknown"


# ── Compute backend detection ────────────────────────────────────────


def _detect_backends():
    """
    Check which compute backends are installed and which the dispatch
    logic in SBEs.py / qwoptics.py will actually use.

    Returns a dict with availability flags, versions, and a plain-English
    description of the dispatch chain.
    """
    info = {
        "numba_installed": False,
        "numba_version": "not installed",
        "numba_jit_working": False,
        "cuda_available": False,
        "cupy_installed": False,
        "cupy_version": "not installed",
        "numpy_version": "unknown",
        "scipy_version": "unknown",
        "sbes_cuda": False,
        "qwoptics_cuda": False,
        "dispatch_chain": [],
        "active_backend": "unknown",
    }

    # NumPy
    try:
        import numpy

        info["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    # SciPy
    try:
        import scipy

        info["scipy_version"] = scipy.__version__
    except ImportError:
        pass

    # Numba
    try:
        import numba

        info["numba_installed"] = True
        info["numba_version"] = numba.__version__
        # Verify JIT actually works (not just imported)
        try:
            from numba import jit as _jit

            @_jit(nopython=True)
            def _test_jit(x):
                return x + 1

            _test_jit(1)
            info["numba_jit_working"] = True
        except Exception:
            pass
    except ImportError:
        pass

    # CUDA via numba
    try:
        from numba import cuda

        if cuda.is_available():
            info["cuda_available"] = True
    except (ImportError, RuntimeError):
        pass

    # CuPy
    try:
        import cupy

        info["cupy_installed"] = True
        info["cupy_version"] = cupy.__version__
    except ImportError:
        pass

    # Check what SBEs.py and qwoptics.py detected at import time
    try:
        from pulsesuite.PSTD3D.SBEs import _HAS_CUDA as sbe_flag

        info["sbes_cuda"] = bool(sbe_flag)
    except (ImportError, AttributeError):
        pass

    try:
        from pulsesuite.PSTD3D.qwoptics import _HAS_CUDA as qw_flag

        info["qwoptics_cuda"] = bool(qw_flag)
    except (ImportError, AttributeError):
        pass

    # Determine the dispatch chain (matches the try/except logic in SBEs.py)
    # Pattern: if _HAS_CUDA -> try CUDA -> except -> try JIT -> except -> fallback
    #          else          -> try JIT -> except -> fallback
    chain = []
    if info["sbes_cuda"] or info["qwoptics_cuda"]:
        chain.append("CUDA (GPU)")
    if info["numba_jit_working"]:
        chain.append("Numba JIT (CPU parallel)")
    chain.append("NumPy fallback (CPU serial)")

    info["dispatch_chain"] = chain
    info["active_backend"] = chain[0] if chain else "unknown"

    return info


# ── Summary writer ───────────────────────────────────────────────────


def write_summary(filename="run_summary.txt", sim_params=None):
    """
    Write a human-readable run summary to *filename*.

    Parameters
    ----------
    filename : str
        Output file path (default: ``run_summary.txt`` in cwd).
    sim_params : dict, optional
        Simulation parameters to include (Nt, dt, Nr, etc.).
    """
    info = _detect_backends()

    lines = []
    lines.append("=" * 72)
    lines.append("  PULSESUITE RUN SUMMARY")
    lines.append("=" * 72)
    lines.append("")

    # Timestamp
    lines.append(f"Date/Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Working dir:     {os.getcwd()}")
    lines.append("")

    # ── Timing ──
    lines.append("--- Timing ---")
    for label in ["total", "init", "timeloop"]:
        elapsed = _elapsed(label)
        if elapsed is not None:
            mins, secs = divmod(elapsed, 60)
            hrs, mins = divmod(mins, 60)
            lines.append(
                f"  {label:12s}  {int(hrs):02d}:{int(mins):02d}:{secs:06.3f}"
                f"  ({elapsed:.3f} s)"
            )
    lines.append("")

    # ── Simulation params ──
    if sim_params:
        lines.append("--- Simulation Parameters ---")
        for k, v in sim_params.items():
            lines.append(f"  {k:16s}  {v}")
        lines.append("")

    # ── Hardware ──
    lines.append("--- Hardware ---")
    lines.append(f"  CPU:           {_cpu_info()}")
    lines.append(f"  Cores (os):    {os.cpu_count()}")
    lines.append(f"  RAM:           {_mem_info()}")
    lines.append(f"  GPU:           {_gpu_info()}")
    lines.append(f"  Platform:      {platform.platform()}")
    lines.append("")

    # ── Threading ──
    lines.append("--- Threading ---")
    numba_threads = os.environ.get(
        "NUMBA_NUM_THREADS", "not set (defaults to all cores)"
    )
    omp_threads = os.environ.get("OMP_NUM_THREADS", "not set")
    mkl_threads = os.environ.get("MKL_NUM_THREADS", "not set")
    lines.append(f"  NUMBA_NUM_THREADS:  {numba_threads}")
    lines.append(f"  OMP_NUM_THREADS:    {omp_threads}")
    lines.append(f"  MKL_NUM_THREADS:    {mkl_threads}")
    try:
        import numba

        lines.append(f"  Numba active threads:  {numba.config.NUMBA_NUM_THREADS}")
    except (ImportError, AttributeError):
        pass
    lines.append("")

    # ── Software versions ──
    lines.append("--- Software ---")
    lines.append(f"  Python:        {sys.version.split()[0]}")
    lines.append(f"  NumPy:         {info['numpy_version']}")
    lines.append(f"  SciPy:         {info['scipy_version']}")
    lines.append(f"  Numba:         {info['numba_version']}")
    lines.append(f"  CuPy:          {info['cupy_version']}")
    lines.append("")

    # ── Backend availability ──
    lines.append("--- Compute Backend Availability ---")
    lines.append(f"  Numba installed:     {'YES' if info['numba_installed'] else 'NO'}")
    lines.append(
        f"  Numba JIT working:   {'YES' if info['numba_jit_working'] else 'NO'}"
    )
    lines.append(f"  CUDA available:      {'YES' if info['cuda_available'] else 'NO'}")
    lines.append(f"  CuPy installed:      {'YES' if info['cupy_installed'] else 'NO'}")
    lines.append(f"  SBEs CUDA flag:      {'YES' if info['sbes_cuda'] else 'NO'}")
    lines.append(f"  qwoptics CUDA flag:  {'YES' if info['qwoptics_cuda'] else 'NO'}")
    lines.append("")

    # ── Dispatch chain ──
    lines.append("--- Dispatch Chain (what actually runs) ---")
    lines.append(f"  Active backend:  {info['active_backend']}")
    lines.append("")
    lines.append("  Dispatch order (first success wins):")
    for i, backend in enumerate(info["dispatch_chain"], 1):
        marker = " <-- ACTIVE" if i == 1 else ""
        lines.append(f"    {i}. {backend}{marker}")
    lines.append("")

    if info["active_backend"].startswith("CUDA"):
        lines.append(
            "  >> GPU is being used for heavy compute (dpdt, dCdt, dDdt, etc.)"
        )
    elif info["active_backend"].startswith("Numba"):
        lines.append("  >> CPU parallel via Numba @jit(parallel=True, nopython=True)")
        lines.append("  >> Functions: dpdt, dCdt, dDdt, QWPolarization3, QWRho5, etc.")
    else:
        lines.append("  >> Plain NumPy loops - no JIT or GPU acceleration")
        lines.append("  >> Install numba for significant speedup: pip install numba")
    lines.append("")

    # ── Performance ──
    loop_time = _elapsed("timeloop")
    if sim_params and loop_time and "Nt" in sim_params:
        nt = int(sim_params["Nt"])
        if nt > 0:
            lines.append("--- Performance ---")
            lines.append(f"  Time per step:     {loop_time / nt * 1000:.3f} ms")
            lines.append(f"  Steps per second:  {nt / loop_time:.1f}")
            lines.append("")

    # ── Per-function profile breakdown ──
    pdata = get_profile_data()
    if pdata:
        total_time = _elapsed("total") or 1.0
        lines.append("--- Profile Breakdown ---")
        # Sort by total time, descending
        sorted_items = sorted(pdata.items(), key=lambda x: x[1]["total"], reverse=True)
        profiled_total = sum(v["total"] for v in pdata.values())
        lines.append(
            f"  {'Function':<24s} {'Time (s)':>10s} {'%Total':>8s}"
            f"  {'Calls':>8s}  {'Avg (ms)':>10s}"
        )
        lines.append(f"  {'-' * 24} {'-' * 10} {'-' * 8}  {'-' * 8}  {'-' * 10}")
        for label, vals in sorted_items:
            pct = 100.0 * vals["total"] / total_time
            avg_ms = vals["total"] / vals["calls"] * 1000 if vals["calls"] else 0
            lines.append(
                f"  {label:<24s} {vals['total']:>10.3f} {pct:>7.1f}%"
                f"  {vals['calls']:>8d}  {avg_ms:>10.3f}"
            )
        unaccounted = total_time - profiled_total
        if unaccounted > 0:
            pct = 100.0 * unaccounted / total_time
            lines.append(
                f"  {'(other/overhead)':<24s} {unaccounted:>10.3f} {pct:>7.1f}%"
            )
        lines.append("")

    lines.append("=" * 72)

    text = "\n".join(lines) + "\n"

    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    # Also print to stdout
    print(text)

    return filename


# ── Standalone benchmark (matches Fortran benchmark_report.txt format) ───


def run_benchmark(params_dir="params", output_file="benchmark_report.txt", nsteps=50):
    """
    Run a standalone PSTD3D benchmark using the existing propagator.

    Uses PSTD3DPropagator with a short time window to measure per-step
    performance. Writes a machine-readable report compatible with the
    Fortran benchmark for cross-language comparison.
    """
    import numpy as np

    from pulsesuite.PSTD3D.PSTD3D import PSTD3DPropagator, _c0
    from pulsesuite.PSTD3D.typepulse import ReadPulseParams, ps
    from pulsesuite.PSTD3D.typespace import (
        GetDx,
        GetDy,
        GetDz,
        GetEpsr,
        GetNx,
        GetNy,
        GetNz,
        ReadSpaceParams,
        ss,
    )
    from pulsesuite.PSTD3D.typetime import GetDt, ReadTimeParams, SetT, SetTf, ts

    # Load params — dummy values required by dataclass, overwritten by Read*
    space = ss(Dims=3, Nx=1, Ny=1, Nz=1, dx=1e-9, dy=1e-9, dz=1e-9, epsr=1.0)
    tm = ts(t=0.0, tf=1e-15, dt=1e-18, n=0)
    pulse = ps(lambda_=800e-9, Amp=1e8, Tw=5e-15, Tp=0.0, chirp=0.0, pol=0, w0=1.0)
    ReadSpaceParams(f"{params_dir}/space.params", space)
    ReadTimeParams(f"{params_dir}/time.params", tm)
    ReadPulseParams(f"{params_dir}/pulse.params", pulse)

    Nx, Ny, Nz = GetNx(space), GetNy(space), GetNz(space)
    dx, dy, dz = GetDx(space), GetDy(space), GetDz(space)
    epsr, dt = GetEpsr(space), GetDt(tm)
    v = _c0 / np.sqrt(epsr)
    v2 = _c0**2 / epsr
    total_cells = Nx * Ny * Nz
    cfl = v * np.sqrt(3.0) * dt / min(dx, dy, dz)
    nthreads = os.cpu_count() or 1
    mem_mb = 7 * total_cells * 16 / 1e6

    # Override time window to run exactly nsteps
    SetT(tm, 0.0)
    SetTf(tm, nsteps * dt)

    print("=" * 60)
    print("PSTD3D Python Benchmark")
    print("=" * 60)
    print(f"  Grid       : {Nx} x {Ny} x {Nz}")
    print(f"  Cells      : {total_cells}")
    print(f"  CFL        : {cfl:.4f}")
    print(f"  Memory est : {mem_mb:.1f} MB")
    print(f"  Threads    : {nthreads}")
    print(f"  Timed steps: {nsteps}")
    print()

    def _em_energy(prop, _v2=v2):
        """Compute EM energy from propagator fields (IFFT to real space)."""
        E2 = (
            np.sum(np.abs(np.fft.ifftn(prop.Ex)) ** 2)
            + np.sum(np.abs(np.fft.ifftn(prop.Ey)) ** 2)
            + np.sum(np.abs(np.fft.ifftn(prop.Ez)) ** 2)
        )
        B2 = (
            np.sum(np.abs(np.fft.ifftn(prop.Bx)) ** 2)
            + np.sum(np.abs(np.fft.ifftn(prop.By)) ** 2)
            + np.sum(np.abs(np.fft.ifftn(prop.Bz)) ** 2)
        )
        return E2 + _v2 * B2

    # Build propagator (IC source, mask absorber, no snapshots)
    prop = PSTD3DPropagator(
        space,
        tm,
        pulse,
        source_type="ic",
        boundary_type="mask",
        output_dir="/tmp/pstd3d_bench",
        snapshot_interval=nsteps + 1,
    )

    # Measure initial energy (after IC seeding, before time loop)
    U_initial = _em_energy(prop)
    print(f"  Initial EM energy U0 = {U_initial:.5e}")

    # Run the full propagation with timing
    timer_start("run")
    prop.run()
    timer_stop("run")

    # Measure final energy
    U_final = _em_energy(prop)
    absorption = (1.0 - U_final / U_initial) if U_initial > 0 else 0.0
    energy_stable = U_final <= U_initial * 1.05

    t_wall = _elapsed("run")
    ms_per_step = t_wall / nsteps * 1000
    mcells_s = total_cells * nsteps / t_wall / 1e6
    ffts_per_step = 26
    mfft_s = ffts_per_step * nsteps / t_wall / 1e6

    # Detect FFT backend
    fft_lib = "numpy.fft"
    try:
        import pyfftw  # noqa: F401

        fft_lib = "pyFFTW"
    except ImportError:
        try:
            import scipy.fft  # noqa: F401

            fft_lib = "scipy.fft"
        except ImportError:
            pass

    # Detect compute backend
    backend = "NumPy"
    try:
        import numba  # noqa: F401

        backend = f"Numba {numba.__version__}"
    except ImportError:
        pass

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Total wall time : {t_wall:10.3f} s")
    print(f"  Time per step   : {ms_per_step:10.3f} ms")
    print(f"  Throughput      : {mcells_s:10.2f} Mcells/s")
    print(f"  FFT ops/s       : {mfft_s:10.2f} MFFT/s")
    print(f"  FFTs per step   : {ffts_per_step}")
    print()
    print("--- EM Energy ---")
    print(f"  U initial       : {U_initial:.5e}")
    print(f"  U final         : {U_final:.5e}")
    print(f"  Absorption      : {absorption * 100:8.1f}% (over {nsteps} steps)")
    if not energy_stable:
        print("  WARNING: Energy grew during run -- possible instability!")
    print("=" * 60)

    # Machine-readable report (same keys as Fortran for direct comparison)
    with open(output_file, "w") as f:
        f.write("# PSTD3D Benchmark Report\n")
        f.write("# Machine-readable key=value format\n")
        f.write("# Compare with Fortran implementation\n\n")
        f.write("language = Python\n")
        f.write(f"compute_backend = {backend}\n")
        f.write(f"fft_library = {fft_lib}\n")
        f.write(f"python_version = {sys.version.split()[0]}\n")
        f.write("method = PSTD\n")
        f.write("time_integration = Leapfrog\n")
        f.write("boundary = masking_absorber\n")
        f.write(f"threads = {nthreads}\n\n")
        f.write(f"Nx = {Nx}\nNy = {Ny}\nNz = {Nz}\n")
        f.write(f"total_cells = {total_cells}\n")
        f.write(f"dx = {dx:15.8e}\nepsr = {epsr:15.8e}\ndt = {dt:15.8e}\n")
        f.write(f"CFL = {cfl:10.6f}\n\n")
        f.write("warmup_steps = 0\n")
        f.write(f"timed_steps = {nsteps}\n")
        f.write(f"ffts_per_step = {ffts_per_step}\n\n")
        f.write(f"wall_time_s = {t_wall:15.8e}\n")
        f.write(f"ms_per_step = {ms_per_step:15.8e}\n")
        f.write(f"Mcells_per_s = {mcells_s:15.8e}\n")
        f.write(f"MFFT_per_s = {mfft_s:15.8e}\n")
        f.write(f"memory_MB = {mem_mb:15.8e}\n\n")
        f.write(f"U_initial = {U_initial:15.8e}\n")
        f.write(f"U_final = {U_final:15.8e}\n")
        f.write(f"absorption_fraction = {absorption:15.8e}\n")
        f.write(f"energy_stable = {'T' if energy_stable else 'F'}\n")

    print(f"  Report: {output_file}")

    # Cleanup temp dir
    import shutil

    shutil.rmtree("/tmp/pstd3d_bench", ignore_errors=True)


def generate_summary(report_file="benchmark_report.txt"):
    """Generate a human-readable summary from a benchmark report.

    Reads the key=value benchmark report and produces a concise summary
    highlighting runtime, CPU usage, major bottlenecks, and performance.

    Parameters
    ----------
    report_file : str
        Path to the benchmark_report.txt file.

    Returns
    -------
    str
        Human-readable summary text.
    """
    data = {}
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    data[key.strip()] = val.strip()
    except FileNotFoundError:
        return f"Report file not found: {report_file}"

    lines = []
    lines.append("BENCHMARK SUMMARY")
    lines.append("=" * 50)

    # Runtime
    wall = float(data.get("wall_time_s", 0))
    steps = int(data.get("timed_steps", 0))
    ms_step = float(data.get("ms_per_step", 0))
    lines.append(f"Total runtime: {wall:.2f}s ({steps} steps)")
    lines.append(f"Per step: {ms_step:.2f} ms")

    # CPU
    threads = data.get("threads", "unknown")
    backend = data.get("compute_backend", "unknown")
    lines.append(f"CPU threads: {threads}")
    lines.append(f"Compute backend: {backend}")
    lines.append(f"FFT library: {data.get('fft_library', 'unknown')}")

    # Grid
    nx = data.get("Nx", "?")
    ny = data.get("Ny", "?")
    nz = data.get("Nz", "?")
    cells = data.get("total_cells", "?")
    lines.append(f"Grid: {nx} x {ny} x {nz} ({cells} cells)")

    # Performance
    mcells = float(data.get("Mcells_per_s", 0))
    lines.append(f"Throughput: {mcells:.2f} Mcells/s")

    # Energy stability
    stable = data.get("energy_stable", "?")
    absorption = float(data.get("absorption_fraction", 0)) * 100
    lines.append(f"Energy stable: {stable}")
    lines.append(f"Boundary absorption: {absorption:.1f}%")

    # Major bottlenecks from profile data
    pdata = get_profile_data()
    if pdata:
        total = sum(v["total"] for v in pdata.values())
        sorted_items = sorted(pdata.items(), key=lambda x: x[1]["total"], reverse=True)
        lines.append("")
        lines.append("MAJOR TIME-CONSUMING OPERATIONS:")
        for label, vals in sorted_items[:5]:
            pct = 100.0 * vals["total"] / total if total > 0 else 0
            lines.append(
                f"  {label}: {vals['total']:.3f}s ({pct:.1f}%, {vals['calls']} calls)"
            )

    lines.append("=" * 50)
    summary = "\n".join(lines) + "\n"

    summary_file = report_file.replace(".txt", "_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PSTD3D Python Benchmark")
    parser.add_argument("--params-dir", default="params")
    parser.add_argument("--output", default="benchmark_report.txt")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    run_benchmark(
        params_dir=args.params_dir, output_file=args.output, nsteps=args.steps
    )
