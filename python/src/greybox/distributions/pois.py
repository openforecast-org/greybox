"""Poisson distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Poisson distribution. R's ``lambda`` is a keyword of
Python, so the mean is ``lambda_``.
"""

import warnings

import numpy as np
from scipy import stats


def dpois(x=None, lambda_=None, log=False, *, q=None, loc=None):
    """Poisson distribution probability mass function.

    Parameters
    ----------
    x : array_like
        Quantiles (non-negative integers).
    lambda_ : float
        Mean parameter (R's ``lambda``).
    log : bool
        If True, return log-probability.
    q : array_like, optional
        Deprecated alias of ``x``, kept for one release.
    loc : float, optional
        Deprecated alias of ``lambda_``, kept for one release.

    Returns
    -------
    array
        Probability mass values.
    """
    if q is not None:
        warnings.warn(
            "'q' is deprecated, use 'x' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        x = q
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'lambda_' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        lambda_ = loc
    if x is None:
        raise TypeError("dpois() missing required argument: 'x'")
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore"):
        result = (stats.poisson.logpmf if log else stats.poisson.pmf)(x, mu=lambda_)
    # As R, an infinite x has no mass
    return np.where(
        np.isinf(x) & (np.asarray(lambda_) >= 0), -np.inf if log else 0.0, result
    )[()]


def ppois(q, lambda_=None, lower_tail=True, log_p=False, *, loc=None):
    """Poisson distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    lambda_ : float
        Mean parameter (R's ``lambda``).
    lower_tail : bool
        If True, return lower tail probability.
    log_p : bool
        If True, return log-CDF.
    loc : float, optional
        Deprecated alias of ``lambda_``, kept for one release.

    Returns
    -------
    array
        CDF values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'lambda_' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        lambda_ = loc
    family, args = stats.poisson, (q, lambda_)
    tails = (family.cdf(*args), family.sf(*args))
    tail, other = tails if lower_tail else tails[::-1]
    if not log_p:
        return tail
    # As R, the log of a tail close to one from the other tail
    with np.errstate(divide="ignore"):
        return np.where(tail < 0.5, np.log(tail), np.log1p(-other))[()]


def qpois(p, lambda_=None, lower_tail=True, log_p=False, *, loc=None):
    """Poisson distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    lambda_ : float
        Mean parameter (R's ``lambda``).
    lower_tail : bool
        If True, probabilities are P[X <= x], otherwise P[X > x].
    log_p : bool
        If True, probabilities are given as log(p).
    loc : float, optional
        Deprecated alias of ``lambda_``, kept for one release.

    Returns
    -------
    array
        Quantile values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'lambda_' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        lambda_ = loc
    p = np.exp(p) if log_p else np.asarray(p)
    method = stats.poisson.ppf if lower_tail else stats.poisson.isf
    # As R, the bound of p at the lower end of the support gives that end,
    # where scipy gives one below it
    return np.where(p == (0 if lower_tail else 1), 1, 0) + method(p, mu=lambda_)


def rpois(n, lambda_=None, *, loc=None):
    """Poisson distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    lambda_ : float
        Mean parameter (R's ``lambda``).
    loc : float, optional
        Deprecated alias of ``lambda_``, kept for one release.

    Returns
    -------
    array
        Random values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'lambda_' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        lambda_ = loc
    return stats.poisson.rvs(mu=lambda_, size=n)
