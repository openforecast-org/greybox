"""Point likelihood functions.

This module provides functions for calculating point-wise likelihood values
and point-wise cumulative likelihood values.
"""

import numpy as np
from scipy import stats

from . import distributions as dist
from .alm import ALM


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
        size = model.size if model.size is not None else 1
        return dist.pbinom(y, size=int(size), prob=1.0 / (mu + 1.0))
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
    y = model.actuals
    fitted = model.fitted
    mu = fitted
    scale = model.scale

    if distribution == "dnorm":
        if log:
            lik = stats.norm.logpdf(y, loc=mu, scale=scale)
        else:
            lik = stats.norm.pdf(y, loc=mu, scale=scale)

    elif distribution == "dlaplace":
        if log:
            lik = stats.laplace.logpdf(y, loc=mu, scale=scale)
        else:
            lik = stats.laplace.pdf(y, loc=mu, scale=scale)

    elif distribution == "dlogis":
        if log:
            lik = stats.logistic.logpdf(y, loc=mu, scale=scale)
        else:
            lik = stats.logistic.pdf(y, loc=mu, scale=scale)

    elif distribution == "dt":
        df_res = model.df_residual_
        if log:
            lik = stats.t.logpdf(y, df=df_res, loc=mu, scale=scale)
        else:
            lik = stats.t.pdf(y, df=df_res, loc=mu, scale=scale)

    elif distribution == "dlnorm":
        sdlog = scale
        meanlog = np.log(mu)
        if log:
            lik = stats.lognorm.logpdf(y, s=sdlog, scale=np.exp(meanlog))
        else:
            lik = stats.lognorm.pdf(y, s=sdlog, scale=np.exp(meanlog))

    elif distribution == "dgnorm":
        shape = model.other_ if model.other_ is not None else 2.0
        if log:
            lik = stats.gennorm.logpdf(y, beta=shape, loc=mu, scale=scale)
        else:
            lik = stats.gennorm.pdf(y, beta=shape, loc=mu, scale=scale)

    elif distribution == "dgamma":
        if scale is None or scale == 0:
            shape = 1.0
        else:
            shape = 1 / scale
        if log:
            lik = stats.gamma.logpdf(y, a=shape, scale=scale * mu)
        else:
            lik = stats.gamma.pdf(y, a=shape, scale=scale * mu)

    elif distribution == "dexp":
        if log:
            lik = stats.expon.logpdf(y, scale=mu)
        else:
            lik = stats.expon.pdf(y, scale=mu)

    elif distribution == "dpois":
        if log:
            lik = stats.poisson.logpmf(np.round(y), mu=mu)
        else:
            lik = stats.poisson.pmf(np.round(y), mu=mu)

    elif distribution == "dnbinom":
        size = model.other_ if model.other_ is not None else 1.0
        p_val = size / (size + mu)
        if log:
            lik = stats.nbinom.logpmf(np.round(y), n=size, p=p_val)
        else:
            lik = stats.nbinom.pmf(np.round(y), n=size, p=p_val)

    elif distribution == "dlogitnorm":
        lik = dist.dlogitnorm(y, loc=mu, scale=scale, log=log)

    else:
        if log:
            lik = stats.norm.logpdf(y, loc=mu, scale=scale)
        else:
            lik = stats.norm.pdf(y, loc=mu, scale=scale)

    return lik
