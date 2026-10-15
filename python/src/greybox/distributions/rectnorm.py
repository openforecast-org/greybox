"""Rectified Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Rectified Normal distribution.
If x ~ N(mu, sigma^2) then y = max(0, x) follows Rectified Normal distribution.
"""

import warnings

import numpy as np
from scipy import stats

from .helper import dnorm, pnorm, qnorm


def drectnorm(q, mu=0, sigma=1, log=False, *, loc=None, scale=None):
    """Rectified Normal distribution density.

    f_y = I(x<=0) * F_x(mu, sigma) + I(x>0) * f_x(x, mu, sigma)

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
    q = np.asarray(q, dtype=float)
    if log:
        log_density = np.where(
            q > 0, dnorm(q, mu, sigma, log=True), pnorm(0, mu, sigma, log_p=True)
        )
        return np.where(np.isnan(q), np.nan, log_density)[()]
    # As R, (q>0)*dnorm + (q<=0)*pnorm, with NaN where q is NaN
    indicator = np.where(np.isnan(q), np.nan, q > 0)
    density = indicator * dnorm(q, mu, sigma) + (1 - indicator) * pnorm(0, mu, sigma)
    return density[()]


def prectnorm(q, mu=0, sigma=1, *, loc=None, scale=None):
    """Rectified Normal distribution CDF.

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
    indicator = np.where(np.isnan(q), np.nan, q > 0)
    return (indicator * pnorm(q, mu, sigma) + (1 - indicator) * pnorm(0, mu, sigma))[()]


def qrectnorm(p, mu=0, sigma=1, *, loc=None, scale=None):
    """Rectified Normal distribution quantile function.

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
    return np.maximum(qnorm(p, mu, sigma), 0)[()]


def rrectnorm(n=1, mu=0, sigma=1, *, loc=None, scale=None):
    """Rectified Normal distribution random number generation.

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
    return np.maximum(stats.norm.rvs(loc=mu, scale=sigma, size=n), 0)
