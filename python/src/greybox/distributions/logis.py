"""Logistic distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Logistic distribution.
"""

import warnings

import numpy as np
from scipy import stats

try:
    from greybox import _native_densities  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - the extension is built with the package
    _native_densities = None


def dlogis(x=None, location=0, scale=1, log=False, *, q=None, loc=None):
    """Logistic distribution density.

    Parameters
    ----------
    x : array_like
        Quantiles.
    location : float
        Location parameter.
    scale : float
        Scale parameter.
    log : bool
        If True, return log-density.
    q : array_like, optional
        Deprecated alias of ``x``, kept for one release.
    loc : float, optional
        Deprecated alias of ``location``, kept for one release.

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
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'location' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        location = loc
    if x is None:
        raise TypeError("dlogis() missing required argument: 'x'")
    if _native_densities is None:  # pragma: no cover - fallback path
        if log:
            return stats.logistic.logpdf(x, loc=location, scale=scale)
        return stats.logistic.pdf(x, loc=location, scale=scale)

    # R's nmath dlogis(), through libm
    args = (
        np.asarray(x, dtype=float),
        np.atleast_1d(np.asarray(location, dtype=float)),
        np.atleast_1d(np.asarray(scale, dtype=float)),
    )
    density = _native_densities.dlogis_log if log else _native_densities.dlogis
    return density(*args).reshape(np.broadcast(x, location, scale).shape)[()]


def plogis(
    q=None,
    location=0.0,
    scale=1.0,
    lower_tail=True,
    log_p=False,
    *,
    y=None,
    loc=None,
    log=None,
):
    """Logistic distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    location : float
        Location parameter.
    scale : float
        Scale parameter.
    lower_tail : bool
        If True, return lower tail probability.
    log_p : bool
        If True, return log-CDF.
    y : array_like, optional
        Deprecated alias of ``q``, kept for one release.
    loc : float, optional
        Deprecated alias of ``location``, kept for one release.
    log : bool, optional
        Deprecated alias of ``log_p``, kept for one release.

    Returns
    -------
    array
        CDF values.
    """
    if y is not None:
        warnings.warn(
            "'y' is deprecated, use 'q' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        q = y
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'location' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        location = loc
    if log is not None:
        warnings.warn(
            "'log' is deprecated, use 'log_p' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        log_p = log
    if q is None:
        raise TypeError("plogis() missing required argument: 'q'")
    method = ("log" if log_p else "") + ("cdf" if lower_tail else "sf")
    return getattr(stats.logistic, method)(q, loc=location, scale=scale)


def qlogis(p, location=0, scale=1, lower_tail=True, log_p=False, *, loc=None):
    """Logistic distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    location : float
        Location parameter.
    scale : float
        Scale parameter.
    lower_tail : bool
        If True, probabilities are P[X <= x], otherwise P[X > x].
    log_p : bool
        If True, probabilities are given as log(p).
    loc : float, optional
        Deprecated alias of ``location``, kept for one release.

    Returns
    -------
    array
        Quantile values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'location' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        location = loc
    p = np.exp(p) if log_p else p
    method = stats.logistic.ppf if lower_tail else stats.logistic.isf
    return method(p, loc=location, scale=scale)


def rlogis(n, location=0, scale=1, *, loc=None):
    """Logistic distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    location : float
        Location parameter.
    scale : float
        Scale parameter.
    loc : float, optional
        Deprecated alias of ``location``, kept for one release.

    Returns
    -------
    array
        Random values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'location' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        location = loc
    return stats.logistic.rvs(loc=location, scale=scale, size=n)
