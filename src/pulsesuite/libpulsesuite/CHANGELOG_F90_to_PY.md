# libpulsesuite: Fortran → Python Port Changelog

Tracks every `.F90` module in the original Fortran `libpulsesuite/src/` and
its status in the Python `pulsesuite.libpulsesuite` package.

## Ported

| Fortran file | Python file | Notes |
|---|---|---|
| `nrutils.F90` | `nrutils.py` | 1:1 port. camelCase API preserved. Vectorised with NumPy. |
| `integrator.F90` | `integrator.py` | ODE integrators (Bulirsch-Stoer, etc.) ported to NumPy/SciPy. |
| `logger.F90` | `logger.py` | Thin wrapper around Python `logging` stdlib. Custom `DEBUG2`/`DEBUG3` levels added to match Fortran hierarchy. Does **not** reimplement the Fortran logger class — uses `logging.getLogger()` instead. |
| `helpers.F90` | `helpers.py` | Utility routines ported. |
| `spliner.F90` | `spliner.py` | Spline interpolation ported. |
| `units.F90` | `units.py` | SI/binary prefix handling, unit parsing. |
| `materialproperties.F90` | `materialproperties.py` | Material/optical property lookups. |
| `constants.F90` | *(lives in `core/constants.py`)* | Physical constants moved to `pulsesuite.core`. |
| `pulsegenerator.F90` | `pulsegenerator.py` | Gaussian, Sech², Square, file-based envelopes + multi-pulse trains. `FilePulse` uses `np.loadtxt` instead of Fortran `readmatrix`. Fortran names preserved (`GaussPulse`, `Sech2Pulse`, `pulsegen`, `multipulsegen`). |

## Skipped (not needed in Python)

| Fortran file | Reason |
|---|---|
| `fileio.F90` | Entire module works around Fortran's unit-number I/O model. Python has `open()`, `configparser`, `np.loadtxt`/`np.savetxt`, `sys.stdin`/`sys.stdout` — no equivalent needed. |
| `strings.F90` | Fortran string manipulation (`int2str`, `str2int`, trim, etc.). Python strings are first-class with `str()`, `int()`, f-strings, `.strip()` — completely redundant. |
| `f2kcli.F90` | Fortran 2003 command-line argument wrapper (`get_command_argument`). Python has `sys.argv` and `argparse`. |
| `nagf2kcli.F90` | NAG-compiler-specific variant of `f2kcli.F90`. Same reason. |
| `parsecommandline.F90` | CLI parser built on `f2kcli`. Python has `argparse`/`click`. |
| `types.F90` | Fortran derived-type definitions (kind parameters, dp/sp). Python uses native `float`/`complex` and NumPy dtypes. Type modules live in `core/` (`typelens.py`, `typemedium.py`, etc.). |
| `calcintlength.F90` | Workaround for Absoft compiler bug. `CalcIntLength(i)` → `len(str(i))`. `CalcDblLength` computes Fortran format widths — Python formatting handles this natively. |
| `fitter.F90` | Manual Levenberg-Marquardt (NR `mrqmin` + `gaussj`). Use `scipy.optimize.curve_fit` / `scipy.optimize.least_squares` instead — better algorithms, bounds support, sparse Jacobians. |
| `helpers.backup.F90` | Backup copy of `helpers.F90`. Dead file. |
