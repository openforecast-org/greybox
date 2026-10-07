"""Laplace distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Laplace distribution.
"""

import numpy as np
from scipy import stats

try:
    from greybox import _native_densities  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - source checkout without the extension
    _native_densities = None


def dlaplace(q, loc=0, scale=1, log=False):
    """Laplace distribution density.

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Location parameter (mu).
    scale : float
        Scale parameter.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    if log:
        # Analytically and through libm, as R computes it
        if _native_densities is None:  # pragma: no cover - fallback path
            return stats.laplace.logpdf(q, loc=loc, scale=scale)
        return _native_densities.dlaplace_log(
            np.asarray(q, dtype=float),
            np.atleast_1d(np.asarray(loc, dtype=float)),
            np.atleast_1d(np.asarray(scale, dtype=float)),
        )
    return stats.laplace.pdf(q, loc=loc, scale=scale)


def plaplace(q, loc=0, scale=1):
    """Laplace distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        CDF values.
    """
    return stats.laplace.cdf(q, loc=loc, scale=scale)


def qlaplace(p, loc=0, scale=1):
    """Laplace distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.laplace.ppf(p, loc=loc, scale=scale)


def rlaplace(n, loc=0, scale=1):
    """Laplace distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Random values.
    """
    return stats.laplace.rvs(loc=loc, scale=scale, size=n)
