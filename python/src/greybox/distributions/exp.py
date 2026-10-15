"""Exponential distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Exponential distribution. ``loc`` shifts it; R's ``dexp``
has no such argument.
"""

import warnings

import numpy as np
from scipy import stats


def dexp(x=None, rate=1, log=False, *, loc=0, q=None, scale=None):
    """Exponential distribution density.

    Parameters
    ----------
    x : array_like
        Quantiles (must be >= loc).
    rate : float
        Rate parameter (lambda).
    log : bool
        If True, return log-density.
    loc : float
        Location parameter.
    q : array_like, optional
        Deprecated alias of ``x``, kept for one release.
    scale : float, optional
        Deprecated, kept for one release: the reciprocal of ``rate``.

    Returns
    -------
    array
        Density values.
    """
    if q is not None:
        warnings.warn(
            "'q' is deprecated, use 'x' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        x = q
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'rate' (1 / scale) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        rate = 1 / np.asarray(scale)
    if x is None:
        raise TypeError("dexp() missing required argument: 'x'")
    # As R, through the scale 1 / rate
    if log:
        return stats.expon.logpdf(x, loc=loc, scale=1 / np.asarray(rate))
    return stats.expon.pdf(x, loc=loc, scale=1 / np.asarray(rate))


def pexp(q, rate=1, lower_tail=True, log_p=False, *, loc=0, scale=None):
    """Exponential distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    rate : float
        Rate parameter (lambda).
    lower_tail : bool
        If True, return lower tail probability.
    log_p : bool
        If True, return log-CDF.
    loc : float
        Location parameter.
    scale : float, optional
        Deprecated, kept for one release: the reciprocal of ``rate``.

    Returns
    -------
    array
        CDF values.
    """
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'rate' (1 / scale) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        rate = 1 / np.asarray(scale)
    method = ("log" if log_p else "") + ("cdf" if lower_tail else "sf")
    return getattr(stats.expon, method)(q, loc=loc, scale=1 / np.asarray(rate))


def qexp(p, rate=1, lower_tail=True, log_p=False, *, loc=0, scale=None):
    """Exponential distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    rate : float
        Rate parameter (lambda).
    lower_tail : bool
        If True, probabilities are P[X <= x], otherwise P[X > x].
    log_p : bool
        If True, probabilities are given as log(p).
    loc : float
        Location parameter.
    scale : float, optional
        Deprecated, kept for one release: the reciprocal of ``rate``.

    Returns
    -------
    array
        Quantile values.
    """
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'rate' (1 / scale) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        rate = 1 / np.asarray(scale)
    p = np.exp(p) if log_p else p
    method = stats.expon.ppf if lower_tail else stats.expon.isf
    return method(p, loc=loc, scale=1 / np.asarray(rate))


def rexp(n, rate=1, *, loc=0, scale=None):
    """Exponential distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    rate : float
        Rate parameter (lambda).
    loc : float
        Location parameter.
    scale : float, optional
        Deprecated, kept for one release: the reciprocal of ``rate``.

    Returns
    -------
    array
        Random values.
    """
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'rate' (1 / scale) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        rate = 1 / np.asarray(scale)
    return stats.expon.rvs(loc=loc, scale=1 / np.asarray(rate), size=n)
