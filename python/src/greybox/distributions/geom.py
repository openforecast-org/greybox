"""Geometric distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Geometric distribution.
"""

import warnings

import numpy as np
from scipy import stats


def dgeom(x=None, prob=0.5, log=False, *, q=None):
    """Geometric distribution probability mass function.

    Parameters
    ----------
    x : array_like
        Quantiles (non-negative integers, number of failures before first success).
    prob : float
        Probability of success.
    log : bool
        If True, return log-probability.
    q : array_like, optional
        Deprecated alias of ``x``, kept for one release.

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
    if x is None:
        raise TypeError("dgeom() missing required argument: 'x'")
    x = np.asarray(x, dtype=float)
    if log:
        return stats.geom.logpmf(x + 1, p=prob)
    return stats.geom.pmf(x + 1, p=prob)


def pgeom(q, prob=0.5, lower_tail=True, log_p=False):
    """Geometric distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    prob : float
        Probability of success.
    lower_tail : bool
        If True, return lower tail probability.
    log_p : bool
        If True, return log-CDF.

    Returns
    -------
    array
        CDF values.
    """
    family, args = stats.geom, (np.asarray(q, dtype=float) + 1, prob)
    tails = (family.cdf(*args), family.sf(*args))
    tail, other = tails if lower_tail else tails[::-1]
    if not log_p:
        return tail
    # As R, the log of a tail close to one from the other tail
    with np.errstate(divide="ignore"):
        return np.where(tail < 0.5, np.log(tail), np.log1p(-other))[()]


def qgeom(p, prob=0.5, lower_tail=True, log_p=False):
    """Geometric distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    prob : float
        Probability of success.
    lower_tail : bool
        If True, probabilities are P[X <= x], otherwise P[X > x].
    log_p : bool
        If True, probabilities are given as log(p).

    Returns
    -------
    array
        Quantile values.
    """
    p = np.exp(p) if log_p else np.asarray(p)
    method = stats.geom.ppf if lower_tail else stats.geom.isf
    # As R, the bound of p at the lower end of the support gives that end,
    # where scipy gives one below it
    return np.where(p == (0 if lower_tail else 1), 1, 0) + method(p, p=prob) - 1


def rgeom(n, prob=0.5):
    """Geometric distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    prob : float
        Probability of success.

    Returns
    -------
    array
        Random values.
    """
    return stats.geom.rvs(p=prob, size=n) - 1
