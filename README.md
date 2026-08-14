# greybox

**R**

[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/greybox)](https://cran.r-project.org/package=greybox)
[![Downloads](http://cranlogs.r-pkg.org/badges/greybox)](https://cran.r-project.org/package=greybox)
[![R-CMD-check](https://github.com/openforecast-org/greybox/actions/workflows/test.yml/badge.svg)](https://github.com/openforecast-org/greybox/actions/workflows/test.yml)
[![License: LGPL-2.1](https://img.shields.io/badge/License-LGPL--2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/r-greybox)](https://anaconda.org/conda-forge/r-greybox)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/r-greybox)](https://anaconda.org/conda-forge/r-greybox)

**Python**

[![PyPI version](https://img.shields.io/pypi/v/greybox.svg)](https://pypi.org/project/greybox/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/greybox.svg)](https://pypi.org/project/greybox/)
[![Python CI](https://github.com/openforecast-org/greybox/actions/workflows/python-test.yml/badge.svg)](https://github.com/openforecast-org/greybox/actions/workflows/python-test.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/greybox.svg)](https://pypi.org/project/greybox/)

The package _greybox_ contains functions for model building, which is currently done via the model selection and combinations based on information criteria. The resulting model can then be used in analysis and forecasting.

![hex-sticker of the greybox package for R](https://github.com/openforecast-org/greybox/blob/master/man/figures/greybox-web.png?raw=true) ![hex-sticker of the greybox package for Python](https://github.com/openforecast-org/greybox/blob/master/python/img/greybox-python-web.png?raw=true)

There are several groups of functions in the package.

## Installation
[Installation](https://github.com/openforecast-org/greybox/wiki/Installation) wiki explains how the package can be installed in R/Python.

### R Package

The stable version of the package is available on CRAN, so you can install it by running:
```r
install.packages("greybox")
```

A recent, development version, is available via github and can be installed using "remotes" in R. First make sure that you have remotes:
```r
if (!require("remotes")){install.packages("remotes")}
```

and after that run:
```r
remotes::install_github("openforecast-org/greybox")
```

## Python Package

A Python port of greybox is available in the [python](https://github.com/openforecast-org/greybox/tree/master/python) subdirectory. It provides a scikit-learn-style API with support for 26+ distributions and R-style formula syntax.

```bash
pip install greybox
```

## Supported functions

See [wiki](https://github.com/openforecast-org/greybox/wiki) for full documentation, examples, and supported distributions.

### Regression model functions
1. alm - Augmented Linear (regression) Model that implements likelihood estimation of parameters for Normal, Laplace, Asymmetric Laplace, Logistic, Student's t, S, Generalised Normal, Folded Normal, Log Normal, Box-Cox Normal, Logit Normal, Inverse Gaussian, Gamma, Poisson, Negative Binomial, Cumulative Logistic and Cumulative Normal distributions. In a sense this is similar to `glm()` function, but with a different set of distributions and with a focus on forecasting.
2. sm - Scale Model which constructs a regression for scale parameter of a distribution (e.g. for variance in normal distribution). Works like a method applied to already existing model (lm / alm).
3. stepwise - function implements stepwise IC based on partial correlations for the location model.
4. calm - function combines the regression models (alm) from the provided data, based on IC weights and returns the combined alm object.

### Exogenous variables transformation functions
1. xregExpander - function produces lags and leads of the provided data.
2. xregTransformer - function produces non-linear transformations of the provided data (logs, inverse etc).
3. xregMultiplier - function produces cross-products of the variables in the matrix. Could be useful when exploring interaction effects of dummy variables.

### The data analysis functions
1. aid - Automated Identification of Demand, function that detects stockouts and classifies demand into regulular/intermittent, count/fractional, smooth/lumpy.
2. cramer - calculates Cramer's V for two categorical variables. Plus tests the significance of such association.
3. mcor - function returns the coefficients of multiple correlation between the variables. This is useful when measuring association between categorical and numerical variables.
4. association (aka 'assoc()') - function returns matrix of measures of association, choosing between cramer(), mcor() and cor() depending on the types of variables.
5. determination (and the method 'determ()') - function returns the vector of coefficients of determination (R^2) for the provided data. This is useful for the diagnostics of multicollinearity.
6. tableplot - plots the graph for two categorical variables.
7. spread - plots the matrix of scatter / boxplot / tableplot diagrams - depending on the type of the provided variables.
8. graphmaker - plots the original series, the fitted values and the forecasts.
9. stick - Seasonal, Trend, Irregular Contribution Kit based on Hans' Levenbach STI classification. Shows the proportion of the variability attributed to the specific time series components.

### Models evaluation functions
1. ro - rolling origin evaluation (see the vignette).
2. rmcb - Regression for Multiple Comparison with the Best. This is a simplified version of the nemenyi / MCB test, relying on regression on ranks of methods.
3. measures - the error measures for the provided forecasts. Includes MPE, MAPE, MASE, sMAE, sMSE, RelMAE, RelRMSE, MIS, sMIS, RelMIS, pinball and others.
<!--5. nemenyi - non-parametric test for comparison of multiple classifiers / methods. This function not only conducts the test, but also provide the plots, showing the ranks of the different methods together with their confidence intervals.-->

### Distribution functions:
1. qlaplace, dlaplace, rlaplace, plaplace - functions for Laplace distribution.
2. qalaplace, dalaplace, ralaplace, palaplace - functions for Asymmetric Laplace distribution.
3. qs, ds, rs, ps - functions for S distribution.
4. qgnorm, dgnorm, rgnorm, pgnorm - functions for Generalised normal distribution.
5. qfnorm, dfnorm, rfnorm, pfnorm - functions for folded normal distribution.
6. qtplnorm, dtplnorm, rtplnorm, ptplnorm - functions for three parameter log normal distribution.
7. qbcnorm, dbcnorm, rbcnorm, pbcnorm - functions for Box-Cox normal distribution (discussed in Box & Cox, 1964).
8. qlogitnorm, dlogitnorm, rlogitnorm, plogitnorm - functions for Logit-normal distribution.

### Methods for the introduced and some existing classes:
1. temporaldummy - the method that creates a matrix of dummy variables for an object based on the selected frequency. e.g. this can create week of year based on the provided zoo object.
2. outlierdummy - the method that creates a matrix of dummy variables based on the residuals of an object, selected confidence level and type of residuals.
3. pointLik - point likelihood method for the time series models.
4. pAIC, pAICc, pBIC, pBICc - respective point values for the information criteria, based on pointLik.
5. coefbootstrap - the method that returns bootstrapped coefficients of the model. Useful for the calculation of covariance matrix and confidence intervals for parameters.
6. summary - returns summary of the regression (either selected or combined).
7. vcov - covariance matrix for combined models. This is an approximate thing. The real one is quite messy and not yet available.
8. confint - confidence intervals for combined models.
9. predict, forecast - point and interval forecasts for the response variable. forecast method relies on the parameter h (the forecast horizon), while predict is focused on the newdata. See vignettes for the details.
10. nparam - returns the number of estimated parameters in the model (including location, scale, shift).
11. nvariate - returns the number of dimensions of the response variable.
12. actuals - returns the response variable from the model.
13. plot - plots several graphs for the analysis of the residuals (see documentation for more details).
14. AICc - AICc for regression with normally distributed residuals.
15. BICc - BICc for regression with normally distributed residuals.
16. is.greybox, is.alm etc. - functions to check if the object was generated by respective functions.

### Experimental functions:
1. lmDynamic - linear regression with time varying parameters based on pAIC.

## About

`greybox` is developed and maintained by [OpenForecast](https://openforecast.org), a demand forecasting and inventory management consultancy. The package implements the methods we use in our [consulting](https://openforecast.org/consulting/) and teach in our [training courses](https://openforecast.org/training/).
