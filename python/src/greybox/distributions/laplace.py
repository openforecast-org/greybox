"""Laplace distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Laplace distribution.
"""

import warnings

import numpy as np
from scipy import stats

from .gnorm import qgnorm

try:
    from greybox import _native_densities  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - source checkout without the extension
    _native_densities = None


def dlaplace(q, mu=0, scale=1, log=False, *, loc=None):
    """Laplace distribution density.

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
        Location parameter (mu).
    scale : float
        Scale parameter.
    log : bool
        If True, return log-density.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.

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
    if _native_densities is None:  # pragma: no cover - fallback path
        if log:
            return stats.laplace.logpdf(q, loc=mu, scale=scale)
        return stats.laplace.pdf(q, loc=mu, scale=scale)

    # Through libm, as R computes it, and the log analytically
    args = (
        np.asarray(q, dtype=float),
        np.atleast_1d(np.asarray(mu, dtype=float)),
        np.atleast_1d(np.asarray(scale, dtype=float)),
    )
    density = _native_densities.dlaplace_log if log else _native_densities.dlaplace
    return density(*args).reshape(np.broadcast(q, mu, scale).shape)[()]


def plaplace(q, mu=0, scale=1, *, loc=None):
    """Laplace distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.

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
    q, mu, scale = (np.asarray(v, dtype=float) for v in (q, mu, scale))
    # The tail as exp() directly, without the cancellation of 1-exp()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sign = np.sign(q - mu)
        result = 0.5 * (1 + sign) - sign * 0.5 * np.exp(-np.abs(q - mu) / scale)
    return np.where(scale < 0, np.nan, result)[()]


def qlaplace(p, mu=0, scale=1, *, loc=None):
    """Laplace distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.

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
    return qgnorm(p, mu=mu, scale=scale, shape=1)


def rlaplace(n=1, mu=0, scale=1, *, loc=None):
    """Laplace distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.

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
    return stats.laplace.rvs(loc=mu, scale=scale, size=n)
