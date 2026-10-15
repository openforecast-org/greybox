"""Folded Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the folded normal distribution.
"""

import warnings

import numpy as np
from scipy import optimize, stats

from .helper import dnorm, pnorm


def dfnorm(q, mu=0, sigma=1, log=False, *, loc=None, scale=None):
    """Folded Normal distribution density.

    f(x) = 1/sqrt(2*pi*sigma^2) * (
        exp(-(x-mu)^2 / (2*sigma^2)) + exp(-(x+mu)^2 / (2*sigma^2)))

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    log : bool
        If True, return log-density.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sigma``, kept for one release.

    Returns
    -------
    array
        Density values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sigma' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sigma = scale
    q, mu, sigma = (np.asarray(v, dtype=float) for v in (q, mu, sigma))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if log:
            # As R: f(x) = dnorm(x, |mu|, sigma) * (1 + exp(-2*x*|mu|/sigma^2))
            density = dnorm(q, np.abs(mu), sigma, log=True) + np.log1p(
                np.exp(-2 * q * np.abs(mu) / sigma**2)
            )
        else:
            density = (
                1
                / (np.sqrt(2 * np.pi) * sigma)
                * (
                    np.exp(-((q - mu) ** 2) / (2 * sigma**2))
                    + np.exp(-((q + mu) ** 2) / (2 * sigma**2))
                )
            )
    return np.where(q < 0, -np.inf if log else 0.0, density)[()]


def pfnorm(q, mu=0, sigma=1, *, loc=None, scale=None):
    """Folded Normal distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sigma``, kept for one release.

    Returns
    -------
    array
        CDF values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sigma' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sigma = scale
    q, mu = np.asarray(q, dtype=float), np.asarray(mu, dtype=float)
    result = pnorm(q, mu, sigma) + pnorm(q, -mu, sigma) - 1
    return np.where(q < 0, 0.0, result)[()]


def qfnorm(p, mu=0, sigma=1, *, loc=None, scale=None):
    """Folded Normal distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sigma``, kept for one release.

    Returns
    -------
    array
        Quantile values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sigma' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sigma = scale
    p, mu, sigma = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (p, mu, sigma))
    )
    result = np.zeros(p.shape)

    # The root of pfnorm(x) - p, as R, on [0, |mu| + 10*sigma]
    for i in np.ndindex(p.shape):
        if np.isnan(p[i]) or p[i] < 0 or p[i] > 1:
            result[i] = np.nan
        elif p[i] == 1:
            result[i] = np.inf
        elif p[i] > 0:
            try:
                result[i] = optimize.brentq(
                    lambda x, i=i: pfnorm(x, mu[i], sigma[i]) - p[i],
                    0,
                    abs(mu[i]) + 10 * sigma[i],
                    xtol=1e-300,
                    rtol=4 * np.finfo(float).eps,
                    maxiter=1000,
                )
            except (ValueError, RuntimeError):
                result[i] = np.nan

    return result[()]


def rfnorm(n=1, mu=0, sigma=1, *, loc=None, scale=None):
    """Folded Normal distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sigma``, kept for one release.

    Returns
    -------
    array
        Random values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sigma' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sigma = scale
    return np.abs(stats.norm.rvs(loc=mu, scale=sigma, size=n))
