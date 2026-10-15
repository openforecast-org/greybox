# Greybox Python Port - Project Status

## Overview

Python port of the R `greybox` package: regression model building and forecasting
with the Augmented Linear Model (ALM), model selection and combination, the
distributions of greybox, accuracy measures, association measures, variable
processing and diagnostics.

The port is numerical, not only functional: on the same data, Python returns what
R returns. The coefficients, likelihoods, information criteria, covariance matrices
and predictions of `ALM` match R's to the last bit on the parity tests, because
Python computes them as R does:

- the starting values with R's `.lm.fit()` (LINPACK's dqrdc2 and dqrsl, the same
  BLAS calls, `fitters._lm_fit`), also used by the scale model, `stepwise()`,
  `pcor()`, `mcor()`, `association()`, `rmcb()` and `stick()`;
- the product of the regressors in R's column-major order, `exp()` of the log-link
  distributions through libm, and the sums of the losses in R's order and long
  double accumulator;
- the densities through R's nmath code, in C++ (`src/greybox/_native/`, built with
  `-ffp-contract=off`): dgamma, dlogis, dbeta, the noncentral t and chi-squared
  (pnt, qnt, dnt, pnchisq, qnchisq, dnchisq with toms708), and greybox's own
  Laplace, S and Generalised Normal;
- the latent sizes of the ARI models with an occurrence model through a C++ header
  shared with R (`src/headers/almLatent.h`).

The R package and the Python port live in the same repository. A change to the
behaviour of one is made in the other in the same commit.

## Test Status

- **Total tests**: 2,900+ in 39 files, 15 of them comparing Python with R through
  rpy2 (`tests/*_compare.py`).
- **Failing**: 0. **xfail**: 0.
- **Runtime**: about 6 minutes for the whole suite with R available; the
  comparison tests are skipped without rpy2 and R.
- The R-comparison tests need greybox from this branch in R (install it into a
  library and point `R_LIBS` to it).

## Main Features

1. **Regression models**
   - `ALM`: 27 distributions and the losses likelihood, MSE, MAE, HAM, LASSO,
     RIDGE, ROLE and custom functions; ARI orders (`orders`), occurrence models
     (`occurrence="plogis"` / `"pnorm"` or a fitted `ALM`), the scale model
     (`scale_formula`).
   - Methods: `predict()` (confidence and prediction intervals, `side`, several
     levels, the occurrence as R's `$occurrence`), `summary()`, `confint()`,
     `vcov()` (analytical, Hessian or bootstrap), `plot()` (R's 16 diagnostic
     plots), `hatvalues()`, `rstandard()`, `rstudent()`, `extract_scale()`,
     `extract_sigma()`, `multipliers()`, and the attributes of R's `alm` object.
   - `sm()`, `implant()`, `ScaleModel`: the scale model.
   - `coefbootstrap()`: R's "cr" and "dsr" bootstraps.
2. **Selection and combination**: `stepwise()`, `CALM()` (R's `calm()` /
   `lmCombine()`).
3. **Distributions**: d/p/q/r functions of greybox (alaplace, bcnorm, fnorm,
   gnorm, laplace, lgnorm, llaplace, logitnorm, ls, rectnorm, s) and wrappers of
   R's base distributions with R's argument names (norm, lnorm, logis, t, chisq,
   gamma, exp, invgauss, pois, nbinom, binom, geom, beta), with `lower_tail`,
   `log_p` and `ncp` where R has them.
4. **Measures**: the point measures of R (`mae`, `mse`, `rmse`, `mpe`, `mape`,
   `mase`, `rmsse`, `me`, `mre`, `gmrae`, `same`, the relative and scaled ones,
   `measures()`), the interval and quantile measures (`mis`, `smis`, `rmis`,
   `pinball`, `spis`, `sce`), and the half-moments (`hm`, `ham`, `asymmetry`,
   `extremity`, `cextremity`).
5. **Association**: `association()` / `assoc()`, `pcor()`, `mcor()`, `cramer()`,
   `determination()` / `determ()`.
6. **Variable processing**: `formula()` (R's terms and model.matrix: factors with
   the training levels at prediction, interactions, `^` crossing, `%in%`,
   `offset()`, transformations and user functions, `I()`, `trend`),
   `xreg_expander()`, `xreg_multiplier()`, `xreg_transformer()`,
   `temporal_dummy()`, `B()`.
7. **Diagnostics and other tools**: `outlier_dummy()`, `point_lik()`, `aid()`,
   `aid_cat()`, `rmcb()`, `stick()`, `rolling_origin()`, `lowess()`, `supsmu()`.
8. **Plots**: `ALM.plot()`, `PredictionResult.plot()`, `graphmaker()`,
   `tableplot()`, `spread()`.

## R vs Python

| R | Python | Status |
|---|---|---|
| `alm()` | `ALM` | Implemented |
| `sm()`, `implant()` | `sm()`, `implant()`, `ScaleModel` | Implemented |
| `stepwise()` | `stepwise()` | Implemented |
| `calm()`, `lmCombine()` | `CALM()` | Implemented |
| `predict()`, `plot()` of alm and of its predictions | `ALM.predict()`, `ALM.plot()`, `PredictionResult.plot()` | Implemented |
| `coefbootstrap()` | `coefbootstrap()` | Implemented |
| `hatvalues()`, `rstandard()`, `rstudent()` | `ALM` methods | Implemented |
| `outlierdummy()`, `pointLik()` | `outlier_dummy()`, `point_lik()` | Implemented |
| `extractScale()`, `extractSigma()`, `nparam()`, `actuals()`, `AICc()`, `BICc()` | `ALM` methods and attributes | Implemented |
| `association()`, `assoc()`, `pcor()`, `mcor()`, `cramer()`, `determination()`, `determ()` | same names | Implemented |
| `aid()`, `aidCat()`, `rmcb()`, `stick()`, `ro()` | `aid()`, `aid_cat()`, `rmcb()`, `stick()`, `rolling_origin()` | Implemented |
| `xregExpander()`, `xregMultiplier()`, `xregTransformer()`, `temporaldummy()`, `B()`, `multipliers()` | `xreg_expander()`, `xreg_multiplier()`, `xreg_transformer()`, `temporal_dummy()`, `B()`, `multipliers()` | Implemented |
| `graphmaker()`, `tableplot()`, `spread()` | same names | Implemented |
| The measures (`MAE()` ... `sMIS()`, `pinball()`, `hm()`, `ham()`, `measures()`) | lower-case names | Implemented |
| The distributions of greybox | same names | Implemented |
| `lmDynamic()` | - | Not ported |
| `forecast()` of alm, `accuracy()` | - | Not ported (`predict()` and `measures()` cover them) |
| `pAIC()`, `pAICc()`, `pBIC()`, `pBICc()` | - | Not ported |
| `dtplnorm()` family | - | Not ported |
| `detectdst()`, `detectleap()`, `dsrboot()` | - | Not ported (`dsrboot()` is private, inside `coefbootstrap(method="dsr")`) |
| `nvariate()`, `errorType()`, `extractAIC()`, `cooks.distance()`, `is.*()`, `xtable()` | - | Not ported (Python uses `isinstance`; Cook's distance is plot 12 of `ALM.plot()`) |

## Known differences from R

- `hatvalues()` of dbeta fails in both languages (two linear predictors),
  openforecast-org/greybox#79. Plot 12 is not available for dbeta.
- The noncentral Beta: R's own `pbeta(..., ncp)` is accurate to about 1e-9
  absolute; Python takes the Poisson mixture of the central `pbeta`, which is
  exact.
- `qt()` with a large `ncp` hangs in R for some lower-tail probabilities below
  1e-100 (its bisection stops at a root of exactly 0); the port of R's `qnt`
  hangs the same way.
- The levels of character columns follow R's `factor()` in the locale named by
  `LC_ALL` / `LC_COLLATE` / `LANG`: code points in the C locale, the Unicode
  collation otherwise, implemented for Latin text, digits and ASCII punctuation;
  other scripts follow the Latin letters in code-point order, as ICU's root order
  does for the common ones.
- On Windows built with MSVC, `long double` is double precision, so the last bits
  of the long double sums and of R's nmath code can differ from R's there.

## Project Structure

```
python/src/greybox/
├── alm.py                # ALM, PredictionResult
├── sm.py                 # sm(), implant(), ScaleModel
├── fitters.py            # fitter, scaler, _lm_fit (R's .lm.fit), latent sizes
├── cost_function.py      # the losses
├── predict.py            # predict() from a data frame
├── formula.py            # formula(), expand_formula()
├── selection.py          # stepwise(), CALM()
├── bootstrap.py          # coefbootstrap(), BootstrapResult
├── association.py        # association(), pcor(), mcor(), cramer(), determination()
├── point_measures.py     # point measures, measures()
├── quantile_measures.py  # pinball(), interval measures
├── hm.py                 # half-moment measures
├── diagnostics.py        # outlier_dummy()
├── pointlik.py           # point_lik()
├── plots.py              # graphmaker(), tableplot(), spread(), the ALM plots
├── xreg.py               # xreg_expander() and the variable processing
├── aid.py, rmcb.py, stick.py, rolling.py, smoothers.py, transforms.py, data.py
├── methods/summary.py    # SummaryResult
├── distributions/        # the d/p/q/r functions
└── _native/              # C++ (pybind11): densities and nmath, dgamma, lowess,
                          # supsmu, the latent sizes (with ../../src/headers)
```

## Usage

```python
from greybox import ALM, formula, mtcars

y, X = formula("mpg ~ wt + hp", mtcars)
model = ALM(distribution="dlaplace").fit(X, y)
print(model.summary())

prediction = model.predict(X.iloc[:5], interval="prediction", level=[0.8, 0.95])
prediction.plot()
model.plot(which=(1, 2, 4, 6))
```

## References

- R package: https://github.com/openforecast-org/greybox
- CRAN: https://cran.r-project.org/package=greybox

---

*Last Updated: October 2026*
