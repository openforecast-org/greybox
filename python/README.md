# greybox

[![PyPI version](https://img.shields.io/pypi/v/greybox.svg)](https://pypi.org/project/greybox/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/greybox.svg)](https://pypi.org/project/greybox/)
[![Python CI](https://github.com/openforecast-org/greybox/actions/workflows/python-test.yml/badge.svg)](https://github.com/openforecast-org/greybox/actions/workflows/python-test.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/greybox.svg)](https://pypi.org/project/greybox/)
[![License: LGPL-2.1](https://img.shields.io/badge/License-LGPL--2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)

Python port of the [R greybox package](https://cran.r-project.org/package=greybox) — a toolbox for regression model building and forecasting.

![hex-sticker of the greybox package for Python](https://github.com/openforecast-org/greybox/blob/master/python/img/greybox-python-web.png?raw=true)

## Installation

```bash
pip install greybox
```

For more installation options, see the [Installation](https://github.com/openforecast-org/greybox/wiki/Installation) wiki page.

## Quick Example

```python
import numpy as np
import pandas as pd
from greybox import ALM, formula

# Generate sample data
np.random.seed(42)
n = 200
data = pd.DataFrame({
    "y": np.random.normal(10, 2, n),
    "x1": np.random.normal(5, 1, n),
    "x2": np.random.normal(3, 1, n),
})
data["y"] = 2 + 0.5 * data["x1"] - 0.3 * data["x2"] + np.random.normal(0, 1, n)

# Parse formula and fit model
y, X = formula("y ~ x1 + x2", data=data)
model = ALM(distribution="dnorm")
model.fit(X, y)

# Summary
print(model.summary())

# Predict with intervals
pred = model.predict(result.data, interval="prediction", level=0.95)
print(pred.mean[:5])

# Include AR terms (ARIMA-like models)
# For example, ARIMA(1,1,0) model with Log-Normal distribution:
model = ALM(distribution="dlnorm", orders=(1, 1, 0))
model.fit(X, y)
```

## Supported Distributions

| Category | Distributions |
|----------|--------------|
| **Continuous** | `dnorm`, `dlaplace`, `ds`, `dgnorm`, `dlgnorm`, `dfnorm`, `drectnorm`, `dt` |
| **Positive** | `dlnorm`, `dinvgauss`, `dgamma`, `dexp`, `dchisq` |
| **Count** | `dpois`, `dnbinom`, `dgeom` |
| **Bounded** | `dbeta`, `dlogitnorm`, `dbcnorm` |
| **CDF-based** | `pnorm`, `plogis` |
| **Other** | `dalaplace`, `dbinom` |

## Smoothers (`lowess`, `supsmu`)

See the [Smoothers wiki page](https://github.com/openforecast-org/greybox/wiki/Smoothers) for the full reference.

Non-parametric smoothers that reproduce R's `stats::lowess` and `stats::supsmu` to machine precision (native pybind11 implementations).

```python
import numpy as np
from greybox import lowess, supsmu

rng = np.random.default_rng(0)
x = np.linspace(0, 6, 80)
y = np.sin(x) + rng.normal(0, 0.2, 80)

# Cleveland's LOWESS — robust local regression
lo = lowess(x, y, f=0.4)
print(lo["x"], lo["y"])   # sorted x, smoothed y

# Friedman's SuperSmoother — variable-span cross-validation
sm_cv    = supsmu(x, y)              # automatic span selection
sm_fixed = supsmu(x, y, span=0.3)    # fixed span
```

References:
[Cleveland (1979)](https://doi.org/10.1080/01621459.1979.10481038) for LOWESS,
[Friedman (1984)](https://www.osti.gov/biblio/1447470) for SuperSmoother.

## Automatic Identification of Demand (`aid`, `aid_cat`)

See the [AID wiki page](https://github.com/openforecast-org/greybox/wiki/AID) for the full reference.

Classifies a time series into one of six demand types and flags stockouts, new products, and obsolete products. Port of R's `greybox::aid()` / `aidCat()`.

```python
import numpy as np
import matplotlib.pyplot as plt
from greybox import aid, aid_cat

rng = np.random.default_rng(42)

# Intermittent count demand: Poisson(0.7)
y = rng.poisson(0.7, 120).astype(float)
result = aid(y)
print(result)                       # human-readable summary
print(result.name)                  # e.g. "smooth intermittent count"
print(result.type.type1)            # "count" or "fractional" — R-style attribute access
print(result.type["type1"])         # dict-style fallback also works

# Detect injected stockouts and plot them
y2 = rng.poisson(3, 100).astype(float)
y2[40:50] = 0
result2 = aid(y2)
print(result2.stockouts.start, result2.stockouts.end)  # 1-based, [41] [50]
ax = result2.plot()                 # series + grey-shaded stockout span
plt.show()

# Apply to multiple series at once
series = {
    "a": rng.poisson(1, 80).astype(float),
    "b": rng.poisson(5, 80).astype(float),
    "c": rng.normal(10, 2, 80),
}
cat = aid_cat(series)
print(cat.types)        # 2x3 demand-category frequency table
print(cat.anomalies)    # counts of new / stockouts / old products
ax = cat.plot()         # 2x3 demand-category panel
plt.show()
```

The nested `type` and `stockouts` fields are typed dataclasses that
support both `result.stockouts.start` (R-style attribute access) and
`result.stockouts["start"]` (dict-style fallback).

## Features

- **[ALM (Augmented Linear Model)](https://github.com/openforecast-org/greybox/wiki/ALM)**: Likelihood-based regression with 26 distributions
- **Formula parser**: R-style formulas (`y ~ x1 + x2`, `log(y) ~ .`, `y ~ 0 + x1`) with support for backshift operator
- **[stepwise()](https://github.com/openforecast-org/greybox/wiki/stepwise)**: IC-based variable selection with partial correlations
- **[CALM()](https://github.com/openforecast-org/greybox/wiki/CALM)**: Combine ALM models based on IC weights
- **[Forecast error measures](https://github.com/openforecast-org/greybox/wiki/measures)**: MAE, MSE, RMSE, MAPE, MASE, MPE, sMAPE, and more
- **[Variable processing](https://github.com/openforecast-org/greybox/wiki/manipulations)**: `xreg_expander` (lags/leads), `xreg_multiplier` (interactions), `temporal_dummy`
- **[Distributions](https://github.com/openforecast-org/greybox/wiki/distributions)**: 27 distribution families with density, CDF, quantile, and random generation
- **[Association](https://github.com/openforecast-org/greybox/wiki/association)**: Partial correlations and measures of association
- **[Diagnostics](https://github.com/openforecast-org/greybox/wiki/diagnostics)**: Model diagnostics and validation
- **[Smoothers](https://github.com/openforecast-org/greybox/wiki/Smoothers)**: `lowess` and `supsmu` matching R's `stats::lowess` and `stats::supsmu` to machine precision
- **[Demand identification](https://github.com/openforecast-org/greybox/wiki/AID)**: `aid()` and `aid_cat()` for automatic classification of demand series and stockout detection
- **[Method comparison (RMCB)](https://github.com/openforecast-org/greybox/wiki/RMCB)**: `rmcb()` for the Nemenyi/MCB test comparing forecasting methods, with `"mcb"` and `"lines"` plots

## Links

- [Wiki documentation](https://github.com/openforecast-org/greybox/wiki)
- [GitHub repository](https://github.com/openforecast-org/greybox)
- [R package on CRAN](https://cran.r-project.org/package=greybox)

## License

[LGPL-2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)
