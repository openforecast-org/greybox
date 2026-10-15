"""Logit-Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the distribution that becomes normal after the Logit
transformation.
"""

import warnings

import numpy as np

from .helper import pnorm, qnorm


def dlogitnorm(q, mu=0, sigma=1, log=False, *, loc=None, scale=None):
    """Logit-Normal distribution density.

    f(y) = 1/(sqrt(2*pi)*sigma*y*(1-y)) * exp(-(logit(y) - mu)^2 / (2*sigma^2))

    Parameters
    ----------
    q : array_like
        Quantiles (must be in (0, 1)).
    mu : float
        Location parameter (on logit scale).
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
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logit_q = np.log(q / (1 - q))
        # The log analytically, as the log of the density underflows in the tails
        if log:
            density = -np.log(sigma * np.sqrt(2 * np.pi) * q * (1 - q)) - (
                logit_q - mu
            ) ** 2 / (2 * sigma**2)
        else:
            density = (
                1
                / (sigma * np.sqrt(2 * np.pi) * q * (1 - q))
                * np.exp(-((logit_q - mu) ** 2) / (2 * sigma**2))
            )
    return density[()]


def plogitnorm(q, mu=0, sigma=1, *, loc=None, scale=None):
    """Logit-Normal distribution CDF.

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
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = pnorm(np.log(q / (1 - q)), mu, sigma)
    result = np.where(q < 0, 0.0, result)
    return np.where(q >= 1, 1.0, result)[()]


def qlogitnorm(p, mu=0, sigma=1, *, loc=None, scale=None):
    """Logit-Normal distribution quantile function.

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
    result = 1 / (1 + np.exp(-qnorm(p, mu, sigma)))
    return np.where(np.isnan(result), 0, result)[()]


def rlogitnorm(n=1, mu=0, sigma=1, *, loc=None, scale=None):
    """Logit-Normal distribution random number generation.

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
    return qlogitnorm(np.random.uniform(0, 1, n), mu, sigma)
