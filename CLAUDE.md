# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IMPORTANT: Always use the python/.venv environment when running Python commands!
Example: cd python && .venv/bin/python -m pytest tests/

IMPORTANT: Never create cache files, virtual environments, or other generated
artifacts in the project's root folder. All Python work — including the
virtual environment (`python/.venv`), caches, and build outputs — belongs in
the `python/` subfolder.

Python port of the R `greybox` package — a toolbox for regression model building and forecasting. The R package (root directory) is the mature original; the Python port (`python/`) is the active development target. Both are developed on `master`; the `Python` branch is historical.

## Build, Lint, and Test Commands

All commands run from `python/`:

```bash
make build          # pip install -e .
make lint           # flake8 .
make typecheck      # mypy src/greybox/ --ignore-missing-imports
make test           # pytest
pytest tests/test_alm.py                              # single file
pytest tests/test_alm.py::TestFormula::test_formula_with_response  # single test
ptw                 # watch mode
```

IMPORTANT: Always run `make typecheck` (mypy) together with the linting
checks — treat type checking as part of "linting", never skip it. New or
changed code must pass `make lint` AND `make typecheck` before it is
considered done.

CI (`.github/workflows/python-test.yml`) runs three separate lint steps over
`src/greybox/` — `flake8`, `ruff check` and `ruff format --check` — plus the
full pytest suite on ubuntu, windows and macos. `make lint` only covers the
first, so run `ruff check src/greybox/` and `ruff format --check src/greybox/`
as well before calling a change done.

R-vs-Python comparison tests require `rpy2` and R `greybox` installed:
```bash
pytest tests/test_r_python_compare.py
pytest tests/test_alm_distributions_compare.py
```

## Architecture

### Python package (`python/src/greybox/`)

- **`alm.py`** — Core `ALM` class (scikit-learn-style estimator). Supports 27 distributions and 7 loss functions. Uses `nlopt` for optimization (default: Nelder-Mead).
- **`formula.py`** — R-style formula parser (`formula()`, `expand_formula()`). Supports `y ~ x1 + x2`, `y ~ 0 + x1` (no intercept), `log(y) ~ x`, `I(x+z)` (protected expressions), `trend` auto-generation.
- **`fitters.py`** — Internal fitting machinery: `scaler_internal`, `fitter`, `fitter_recursive`. Wraps `cost_function.py`.
- **`cost_function.py`** — Loss functions: `likelihood`, `MSE`, `MAE`, `HAM`, `LASSO`, `RIDGE`, `ROLE`.
- **`predict.py`** — `predict_basic()` returning `PredictionResult` dataclass with `.mean`, `.lower`, `.upper`.
- **`distributions/`** — 27 distributions with d/p/q/r functions (density, CDF, quantile, random).
- **`methods/summary.py`** — `SummaryResult` dataclass with formatted `__str__`.
- **`selection.py`** — `stepwise()`, `CALM()`.
- **`point_measures.py`** — 20+ accuracy metrics (MAE, MSE, RMSE, MAPE, MASE, etc.).
- **`xreg.py`** — Variable processing: lag/lead expansion, transformations, temporal dummies.
- **`rolling.py`** — `rolling_origin()` cross-validation with `RollingOriginResult`.
- **`association.py`** — `pcor()`, `mcor()`, `association()`, `determination()` (partial/multiple correlations).
- **`hm.py`** — Half-moment measures: `hm()`, `ham()`, `asymmetry()`, `extremity()`, `mre()`.
- **`quantile_measures.py`** — `pinball()` quantile/expectile scoring, `MIS`.
- **`diagnostics.py`** — `outlier_dummy()` with `OutlierResult`.
- **`stick.py`** — `stick()` (Seasonality, Trend, Irregular contribution via ANOVA) with `StickResult` (`.strength`, `.anova`, `.plot()`).
- **`transforms.py`** — `bc_transform()`, `bc_transform_inv()`, `mean_fast()`.
- **`smoothers.py`** — `lowess()` and `supsmu()`, thin wrappers over the native extensions below.
- **`aid.py`** — `aid()` automatic intermittent-demand classification.
- **`rmcb.py`** — `rmcb()` regression for multiple comparison with the best.
- **`pointlik.py`** — `point_lik()` point-wise likelihoods per observation.
- **`data.py`** — `mtcars` dataset.
- **`_native/`** — C++ sources built as pybind11 extensions: `lowess.cpp`, `supsmu.cpp`, `dgamma.cpp` (`greybox._native_lowess` / `_native_supsmu` / `_native_dgamma`).

### Key design decisions

- Variance = `X @ vcov @ X'`, scale uses `df_residual = n - k`.
- `nlopt>=2.7.0` is a required runtime dependency (declared in pyproject.toml).

### Never let the compiler contract floating-point operations

`setup.py` builds every native extension with `-ffp-contract=off`
(`/fp:precise` on MSVC) through the `BuildExtNoFPContract` command class. The
extensions exist to reproduce R bit for bit, and a fused multiply-add rounds
once where the source asks for twice. Do not remove that flag, and do not add
`-ffast-math`, `-Ofast` or `-march=native`: any of them re-enables contraction.

`lowess.cpp` accumulates `c += w[k] * (diff * diff)`, which Clang (default
`-ffp-contract=on`) will fuse wherever an FMA instruction exists — every arm64
chip, and not baseline x86-64. That asymmetry made the macOS CI job fail the
four `TestLowessRParity` cases while Linux and Windows passed. Reproduce on
x86-64 by rebuilding with `-mfma -ffp-contract=fast`.

## Code Style

- 4-space indentation, 88-char line limit (`setup.cfg`: `max-line-length = 88`)
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Type hints on all function signatures
- Import order: stdlib, third-party, local (blank line between groups)
- Linter/formatter: `ruff`; type checker: `mypy`
