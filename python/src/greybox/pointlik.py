"""Point likelihood functions.

This module provides functions for calculating point-wise likelihood values
and point-wise cumulative likelihood values.
"""

import numpy as np
from scipy import stats

from . import distributions as dist
from .alm import ALM
from .fitters import scale_sd


def point_lik_cumulative(model: ALM) -> np.ndarray:
    """Point cumulative likelihood values.

    Returns the value of the cumulative distribution function (CDF) of the
    fitted model evaluated at the actual response. This is the Python
    equivalent of R's ``greybox::pointLikCumulative``.

    Supported distributions: ``dgeom``, ``dpois``, ``dnbinom``, ``dbinom``.

    Parameters
    ----------
    model : ALM
        Fitted ALM model with a supported discrete distribution.

    Returns
    -------
    np.ndarray
        CDF values evaluated at the model's actuals.
    """
    if not isinstance(model, ALM):
        raise TypeError("model must be a fitted ALM")
    distribution = model.distribution
    y = np.asarray(model.actuals, dtype=float)
    mu = np.asarray(model.fitted, dtype=float)

    if distribution == "dgeom":
        return dist.pgeom(y, prob=1.0 / (mu + 1.0))
    if distribution == "dpois":
        return dist.ppois(y, loc=mu)
    if distribution == "dnbinom":
        size = model.other_ if model.other_ is not None else 1.0
        return dist.pnbinom(y, loc=mu, size=size)
    if distribution == "dbinom":
        return dist.pbinom(y, size=int(model.size_ or 1), prob=1.0 / (mu + 1.0))
    raise ValueError(
        f"point_lik_cumulative is not defined for distribution={distribution!r}"
    )


def point_lik(
    model: ALM,
    log: bool = True,
) -> np.ndarray:
    """Point likelihood values.

    This function returns a vector of logarithms of likelihoods for
    each observation.

    Instead of taking the expected log-likelihood for the whole series,
    this function calculates the individual value for each separate
    observation.

    Parameters
    ----------
    model : ALM
        Fitted ALM model.
    log : bool, default=True
        Whether to take logarithm of likelihoods.

    Returns
    -------
    np.ndarray
        Vector of point-wise likelihood values.

    Examples
    --------
    >>> from greybox import formula, ALM
    >>> data = {'y': [1.2, 2.1, 2.9, 4.2, 5.1], 'x': [1, 2, 3, 4, 5]}
    >>> y, X = formula("y ~ x", data)
    >>> model = ALM(distribution="dnorm", loss="likelihood")
    >>> _ = model.fit(X, y)
    >>> np.round(point_lik(model), 4)
    array([ 1.0297,  1.2967, -0.394 ,  0.7882,  1.2839])
    """
    if not isinstance(model, ALM):
        raise TypeError("object must be a fitted ALM model")

    distribution = model.distribution
    y_all = np.asarray(model.actuals, dtype=float)
    # With an occurrence model, the density is that of the non-zeroes, and the
    # occurrence model's point likelihoods are added to all the observations
    occurrence = model.occurrence if isinstance(model.occurrence, ALM) else None
    otU = y_all != 0 if occurrence is not None else np.ones(len(y_all), dtype=bool)
    y = y_all[otU]
    # The location of the distribution, as R's pointLik.alm uses object$mu
    mu = np.asarray(model.mu_, dtype=float)[otU]
    scale = scale_sd(distribution, model.scale)
    if np.size(scale) > 1:
        scale = np.asarray(scale)[otU]
    other = model.other_

    def log_y_density(density):
        # The log-domain distributions: the density of log(y) and its Jacobian
        return density(np.log(y)) - np.log(y)

    densities = {
        "dnorm": lambda: dist.dnorm(y, loc=mu, scale=scale, log=True),
        "dlnorm": lambda: dist.dlnorm(y, loc=mu, scale=scale, log=True),
        "dgnorm": lambda: dist.dgnorm(y, loc=mu, scale=scale, shape=other, log=True),
        "dlgnorm": lambda: log_y_density(
            lambda q: dist.dgnorm(q, loc=mu, scale=scale, shape=other, log=True)
        ),
        "dfnorm": lambda: dist.dfnorm(y, loc=mu, scale=scale, log=True),
        "drectnorm": lambda: dist.drectnorm(y, loc=mu, scale=scale, log=True),
        "dbcnorm": lambda: dist.dbcnorm(
            y, loc=mu, scale=scale, lambda_bc=other, log=True
        ),
        "dlogitnorm": lambda: dist.dlogitnorm(y, loc=mu, scale=scale, log=True),
        "dexp": lambda: dist.dexp(y, scale=mu, log=True),
        "dinvgauss": lambda: dist.dinvgauss(y, loc=mu, scale=scale / mu, log=True),
        "dgamma": lambda: dist.dgamma(y, shape=1 / scale, scale=scale * mu, log=True),
        "dlaplace": lambda: dist.dlaplace(y, loc=mu, scale=scale, log=True),
        "dllaplace": lambda: log_y_density(
            lambda q: dist.dlaplace(q, loc=mu, scale=scale, log=True)
        ),
        "dalaplace": lambda: dist.dalaplace(
            y, loc=mu, scale=scale, alpha=other, log=True
        ),
        "dlogis": lambda: dist.dlogis(y, loc=mu, scale=scale, log=True),
        "dt": lambda: dist.dt(y - mu, df=scale, log=True),
        "ds": lambda: dist.ds(y, loc=mu, scale=scale, log=True),
        "dls": lambda: log_y_density(
            lambda q: dist.ds(q, loc=mu, scale=scale, log=True)
        ),
        "dgeom": lambda: dist.dgeom(y, prob=1 / (mu + 1), log=True),
        "dpois": lambda: dist.dpois(y, loc=mu, log=True),
        "dnbinom": lambda: dist.dnbinom(y, loc=mu, size=other, log=True),
        "dbinom": lambda: dist.dbinom(
            y - int(occurrence is not None),
            size=model.size_,
            prob=1 / (mu + 1),
            log=True,
        ),
        "dchisq": lambda: stats.ncx2.logpdf(y, df=scale, nc=mu),
        "dbeta": lambda: dist.dbeta(y, a=mu, b=scale, log=True),
        # The occurrence models: the probability of the observed outcome
        "plogis": lambda: np.where(
            y != 0, stats.logistic.logcdf(mu), stats.logistic.logsf(mu)
        ),
        "pnorm": lambda: np.where(y != 0, stats.norm.logcdf(mu), stats.norm.logsf(mu)),
    }
    if distribution not in densities:
        raise ValueError(f"point_lik is not defined for distribution={distribution!r}")
    lik = np.zeros(len(y_all))
    lik[otU] = densities[distribution]()
    if occurrence is not None:
        lik = lik + point_lik(occurrence)
    return lik if log else np.exp(lik)
