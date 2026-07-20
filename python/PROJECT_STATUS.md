# Greybox Python Port - Project Status

## Overview

Python port of the R `greybox` package — a toolbox for regression model building and forecasting. The Python implementation provides scikit-learn-style API parity with the R package for core regression and forecasting tasks.

## Test Status

- **Total Tests**: 348
- **Failing**: 0
- **xfail**: 0
- **Runtime**: ~13 seconds
- **Coverage**: Formula parsing (with custom functions), ALM fitting (all 26 distributions), prediction intervals, model selection, accuracy measures, variable processing, R-vs-Python comparison tests

## Main Features Implemented

1. **Regression Models**
   - `ALM` class with 26 distributions and 7 loss functions (likelihood, MSE, MAE, HAM, LASSO, RIDGE, ROLE)
   - `dbeta` two-part model (2 × n_features parameters: shape1 + shape2)

2. **Distributions (26)**
   - Normal family: `dnorm`, `dfnorm`, `drectnorm`, `dbcnorm`, `dgnorm`, `dlgnorm`, `dlnorm`, `dlogitnorm`
   - Laplace family: `dlaplace`, `dalaplace`, `dllaplace`
   - Logistic: `dlogis`
   - Student-t: `dt`
   - S-distribution: `ds`
   - Gamma family: `dgamma`, `dexp`, `dchisq`, `dinvgauss`
   - Count: `dpois`, `dnbinom`, `dbinom`, `dgeom`
   - Other: `dbeta`, `dls`
   - Cumulative: `pnorm`, `plogis`

3. **Model Selection**
   - `stepwise()` — stepwise regression using partial correlations (matching R implementation)
   - `CALM()` — model combination via IC weights

4. **Variable Processing**
   - `xreg_expander` — lag/lead expansion
   - `xreg_multiplier` — cross-product generation
   - `xreg_transformer` — mathematical transformations (log, exp, inv, sqrt, square)
   - `temporal_dummy` — temporal dummy variables

5. **Forecasting & Prediction**
   - Point forecasts with confidence/prediction intervals (t-distribution based)
   - Rolling origin evaluation for time series cross-validation

6. **Formula System**
   - R-style formula parsing (`y ~ x1 + x2`)
   - Built-in transformations: log, log10, log2, sqrt, exp, abs, sin, cos, tan
   - Custom functions: Use any user-defined or imported function in formulas
   - I() wrapper for protected expressions
   - Polynomial terms (x^2, x^3)

7. **Accuracy Measures**
   - MAE, MSE, RMSE, MPE, MAPE, MASE, accuracy

8. **Association & Correlation**
   - `association` — measures of association for different variable types
   - `pcor` — partial correlations
   - `mcor` — multiple correlation

9. **Model Methods**
   - `predict()`, `summary()`, `confint()`
   - Properties: `coef`, `vcov`, `nobs`, `nparam`, `residuals`, `fitted`, `actuals`, `sigma`, `log_lik`, `formula`
   - `determination` — R-squared and adjusted R-squared
   - `outlier_dummy` — outlier detection and dummy variable creation
   - `pointLik` — point likelihood values

## Project Structure

```
python/src/greybox/
├── __init__.py
├── alm.py                   # Main ALM model class
├── formula.py               # R-style formula parser
├── fitters.py               # Internal fitting machinery
├── cost_function.py         # Loss functions
├── predict.py               # Prediction utilities
├── transforms.py            # Data transformations
├── diagnostics.py           # outlier_dummy
├── measures.py              # Accuracy metrics, pcor, mcor, association
├── selection.py             # stepwise, CALM
├── xreg.py                  # Variable processing
├── pointlik.py              # Point likelihood functions
├── rolling.py               # Rolling origin evaluation
│
├── distributions/
│   ├── __init__.py
│   ├── helper.py
│   ├── alaplace.py          # Asymmetric Laplace
│   ├── bcnorm.py            # Box-Cox Normal
│   ├── beta.py
│   ├── binom.py
│   ├── chi2.py
│   ├── exp.py
│   ├── fnorm.py             # Folded Normal
│   ├── gamma.py
│   ├── geom.py
│   ├── gnorm.py             # Generalised Normal
│   ├── invgauss.py          # Inverse Gaussian
│   ├── laplace.py
│   ├── lgnorm.py            # Log-Generalised Normal
│   ├── llaplace.py          # Log-Laplace
│   ├── lnorm.py             # Log-Normal
│   ├── logis.py
│   ├── logitnorm.py         # Logit-Normal
│   ├── ls.py                # Location-Scale
│   ├── nbinom.py            # Negative Binomial
│   ├── pois.py
│   ├── rectnorm.py          # Rectified Normal
│   ├── s.py                 # S-distribution
│   └── t.py
│
└── methods/
    ├── __init__.py
    └── summary.py           # SummaryResult class
```

## R vs Python Comparison

### Core Model Fitting

| R function | Python equivalent | Status |
|---|---|---|
| `alm()` | `ALM` class | Implemented |
| `sm()` (scale model) | — | Not implemented |
| `stepwise()` | `stepwise()` | Implemented (matches R algorithm) |
| `lmCombine()` | `CALM()` | Implemented |
| `lmDynamic()` | — | Not implemented |

### Distribution Families (d/p/q/r functions)

| R distribution | Python module | Status |
|---|---|---|
| `dnorm` / `pnorm` / `qnorm` / `rnorm` | `distributions/` (scipy) | Implemented |
| `dlogis` / `plogis` / `qlogis` / `rlogis` | `distributions/logis.py` | Implemented |
| `dlaplace` / `plaplace` / `qlaplace` / `rlaplace` | `distributions/laplace.py` | Implemented |
| `dalaplace` / `palaplace` / `qalaplace` / `ralaplace` | `distributions/alaplace.py` | Implemented |
| `dllaplace` / `pllaplace` / `qllaplace` / `rllaplace` | `distributions/llaplace.py` | Implemented |
| `ds` / `ps` / `qs` / `rs` | `distributions/s.py` | Implemented |
| `dfnorm` / `pfnorm` / `qfnorm` / `rfnorm` | `distributions/fnorm.py` | Implemented |
| `drectnorm` / `prectnorm` / `qrectnorm` / `rrectnorm` | `distributions/rectnorm.py` | Implemented |
| `dbcnorm` / `pbcnorm` / `qbcnorm` / `rbcnorm` | `distributions/bcnorm.py` | Implemented |
| `dgnorm` / `pgnorm` / `qgnorm` / `rgnorm` | `distributions/gnorm.py` | Implemented |
| `dlgnorm` / `plgnorm` / `qlgnorm` / `rlgnorm` | `distributions/lgnorm.py` | Implemented |
| `dlnorm` / `plnorm` / `qlnorm` / `rlnorm` | `distributions/lnorm.py` | Implemented |
| `dlogitnorm` / `plogitnorm` / `qlogitnorm` / `rlogitnorm` | `distributions/logitnorm.py` | Implemented |
| `dt` / `pt` / `qt` / `rt` | `distributions/t.py` | Implemented |
| `dgamma` / `pgamma` / `qgamma` / `rgamma` | `distributions/gamma.py` | Implemented |
| `dexp` / `pexp` / `qexp` / `rexp` | `distributions/exp.py` | Implemented |
| `dchisq` / `pchisq` / `qchisq` / `rchisq` | `distributions/chi2.py` | Implemented |
| `dinvgauss` / `pinvgauss` / `qinvgauss` / `rinvgauss` | `distributions/invgauss.py` | Implemented |
| `dpois` / `ppois` / `qpois` / `rpois` | `distributions/pois.py` | Implemented |
| `dnbinom` / `pnbinom` / `qnbinom` / `rnbinom` | `distributions/nbinom.py` | Implemented |
| `dbinom` / `pbinom` / `qbinom` / `rbinom` | `distributions/binom.py` | Implemented |
| `dgeom` / `pgeom` / `qgeom` / `rgeom` | `distributions/geom.py` | Implemented |
| `dbeta` / `pbeta` / `qbeta` / `rbeta` | `distributions/beta.py` | Implemented |
| `dls` / `pls` / `qls` / `rls` | `distributions/ls.py` | Implemented |
| `dtplnorm` / `ptplnorm` / `qtplnorm` / `rtplnorm` | — | Not implemented |

### Prediction & Forecasting

| R function | Python equivalent | Status |
|---|---|---|
| `predict.alm()` | `ALM.predict()` | Implemented |
| `forecast.alm()` | — | Not implemented |
| `ro()` (rolling origin) | `rolling.ro()` | Implemented |

### Accuracy Measures

| R function | Python equivalent | Status |
|---|---|---|
| `MAE()` | `mae()` | Implemented |
| `MSE()` | `mse()` | Implemented |
| `RMSE()` | `rmse()` | Implemented |
| `MPE()` | `mpe()` | Implemented |
| `MAPE()` | `mape()` | Implemented |
| `MASE()` | `mase()` | Implemented |
| `accuracy()` | `accuracy()` | Implemented |
| `ME()` | — | Not implemented |
| `MRE()` | — | Not implemented |
| `RMSSE()` | — | Not implemented |
| `MIS()` | — | Not implemented |
| `rMAE()` / `rRMSE()` / `rAME()` / `rMIS()` | — | Not implemented |
| `sMSE()` / `sPIS()` / `sCE()` / `sMIS()` | — | Not implemented |
| `GMRAE()` / `SAME()` | — | Not implemented |
| `pinball()` | — | Not implemented |
| `hm()` | — | Not implemented |
| `ham()` | — | Not implemented |
| `asymmetry()` / `extremity()` / `cextremity()` | — | Not implemented |

### Information Criteria

| R function | Python equivalent | Status |
|---|---|---|
| `AICc()` | Via `ALM` (built-in) | Implemented |
| `BICc()` | Via `ALM` (built-in) | Implemented |
| `pointLik()` | `pointlik.py` | Implemented |
| `pAIC()` / `pAICc()` | — | Not implemented |
| `pBIC()` / `pBICc()` | — | Not implemented |

### Model Methods

| R function | Python equivalent | Status |
|---|---|---|
| `coef()` | `ALM.coef` | Implemented |
| `confint()` | `ALM.confint()` | Implemented |
| `vcov()` | `ALM.vcov` | Implemented |
| `sigma()` | `ALM.sigma` | Implemented |
| `logLik()` | `ALM.log_lik` | Implemented |
| `nobs()` | `ALM.nobs` | Implemented |
| `nparam()` | `ALM.nparam` | Implemented |
| `summary()` | `ALM.summary()` | Implemented |
| `determination()` | `ALM.determination()` | Implemented |
| `outlierdummy()` | `outlier_dummy()` | Implemented |
| `coefbootstrap()` | — | Not implemented |
| `hatvalues()` | — | Not implemented |
| `rstandard()` | — | Not implemented |
| `rstudent()` | — | Not implemented |
| `cooks.distance()` | — | Not implemented |
| `extractAIC()` | — | Not implemented |

### Association & Correlation

| R function | Python equivalent | Status |
|---|---|---|
| `association()` | `association()` | Implemented |
| `mcor()` | `mcor()` | Implemented |
| `pcor()` | `pcor()` | Implemented |
| `cramer()` | — | Not implemented |

### Variable Engineering

| R function | Python equivalent | Status |
|---|---|---|
| `xregExpander()` | `xreg_expander()` | Implemented |
| `xregMultiplier()` | `xreg_multiplier()` | Implemented |
| `xregTransformer()` | `xreg_transformer()` | Implemented |
| `temporaldummy()` | `temporal_dummy()` | Implemented |

### Plotting

| R function | Python equivalent | Status |
|---|---|---|
| `plot.greybox()` | — | Not implemented |
| `graphmaker()` | — | Not implemented |
| `tableplot()` | — | Not implemented |
| `spread()` | — | Not implemented |

### Other R Functions Not Implemented

| R function | Description |
|---|---|
| `sm()` / `implant()` | Scale model and its implant method |
| `lmDynamic()` | Dynamic linear model with AR components |
| `aid()` / `aidCat()` | Automatic Identification of Distributions |
| `rmcb()` / `dsrboot()` | Regression model comparison bootstrap |
| `detectdst()` / `detectleap()` | DST / leap year detection |
| `dtplnorm()` family | Three-Parameter Log-Normal distribution |

### Summary Counts

| Category | Implemented | Not implemented |
|---|---|---|
| Core model fitting | 3 | 2 |
| Distribution families (d/p/q/r) | 24 | 1 |
| Prediction & forecasting | 3 | 0 |
| Accuracy measures | 7 | ~19 |
| Information criteria | 3 | 4 |
| Model methods | 10 | 6 |
| Association & correlation | 3 | 1 |
| Variable engineering | 4 | 0 |
| Plotting | 0 | 4 |
| Other utilities | 0 | ~8 |
| **Total** | **~57** | **~45** |

## Usage Examples

### Basic ALM Model
```python
from greybox.formula import formula
from greybox.alm import ALM

data = {'y': [1, 2, 3, 4, 5], 'x1': [1, 2, 3, 4, 5], 'x2': [2, 4, 6, 8, 10]}
y, X = formula("y ~ x1 + x2", data)

model = ALM(distribution="dnorm", loss="likelihood")
model.fit(X, y)

# Predictions with confidence intervals
result = model.predict(X, interval="confidence", level=0.95)
```

### Stepwise Regression
```python
from greybox.selection import stepwise

data = {'y': [1, 2, 3, 4, 5], 'x1': [1, 2, 3, 4, 5],
        'x2': [2, 4, 6, 8, 10], 'x3': [3, 6, 9, 12, 15]}
model = stepwise(data, ic="AICc")

# Access IC values and timing
print(model.ic_values)      # List of IC values at each step
print(model.time_elapsed)  # Time taken for selection
```

### Formula with Custom Functions
```python
from greybox.formula import formula
import numpy as np
from scipy.special import erfc

# User-defined function
def my_transform(x):
    return x * 2

# Or use imported functions directly
data = {'y': [1, 2, 3], 'x': [1, 2, 3]}
y, X = formula("y ~ my_transform(x)", data)  # Custom function
y, X = formula("y ~ erfc(x)", data)           # Imported function
y, X = formula("my_transform(y) ~ x", data)  # Custom function on LHS
```

### Accuracy Measures
```python
from greybox.measures import accuracy, mae, mape

actual = [1, 2, 3, 4, 5]
forecast = [1.1, 2.0, 3.2, 3.9, 5.1]

measures = accuracy(actual, forecast)
```

## References

- Original R package: https://github.com/openforecast-org/greybox
- R greybox on CRAN: https://cran.r-project.org/web/packages/greybox/

---

*Last Updated: February 2026*
