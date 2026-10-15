"""Inverse Gaussian distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Inverse Gaussian distribution.

Parameterised by ``mean`` and ``dispersion``, matching R's
``statmod::dinvgauss(mean=, dispersion=)``, which is what ADAM's
``dinvgauss`` error distribution is written against. The order of the
arguments is (mean, dispersion): statmod's ``shape`` (1 / dispersion), its
third argument, is not taken.

SciPy parameterises the same distribution differently: its standard form is
``IG(mean=mu, lambda=1)``, and a ``scale`` of ``s`` gives ``IG(mean=s*mu,
lambda=s)``. With ``lambda = 1 / dispersion`` that makes the mapping
``mu = mean * dispersion`` and SciPy's ``scale = 1 / dispersion``.
"""

import warnings

import numpy as np
from scipy import stats


def _scipy_args(mean, dispersion):
    """Map ``(mean, dispersion)`` onto SciPy's ``(mu, scale)``."""
    return {"mu": mean * dispersion, "scale": 1.0 / dispersion}


def dinvgauss(x=None, mean=1, dispersion=1, log=False, *, q=None, loc=None, scale=None):
    """Inverse Gaussian distribution density.

    Parameters
    ----------
    x : array_like
        Quantiles (must be positive).
    mean : float
        Mean of the distribution.
    dispersion : float
        Dispersion (the reciprocal of the shape ``lambda``).
    log : bool
        If True, return log-density.
    q : array_like, optional
        Deprecated alias of ``x``, kept for one release.
    loc : float, optional
        Deprecated alias of ``mean``, kept for one release.
    scale : float, optional
        Deprecated alias of ``dispersion``, kept for one release.

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
            "'loc' is deprecated, use 'mean' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mean = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'dispersion' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        dispersion = scale
    if x is None:
        raise TypeError("dinvgauss() missing required argument: 'x'")
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore"):
        result = stats.invgauss.logpdf(x, **_scipy_args(mean, dispersion))
    # As R, an infinite x has no density, and the density is exp() of the log
    result = np.where(np.isposinf(x), -np.inf, result)
    return (result if log else np.exp(result))[()]


def pinvgauss(
    q,
    mean=1,
    dispersion=1,
    lower_tail=True,
    log_p=False,
    *,
    loc=None,
    scale=None,
):
    """Inverse Gaussian distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    mean : float
        Mean of the distribution.
    dispersion : float
        Dispersion (the reciprocal of the shape ``lambda``).
    lower_tail : bool
        If True, return lower tail probability.
    log_p : bool
        If True, return log-CDF.
    loc : float, optional
        Deprecated alias of ``mean``, kept for one release.
    scale : float, optional
        Deprecated alias of ``dispersion``, kept for one release.

    Returns
    -------
    array
        CDF values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mean' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mean = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'dispersion' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        dispersion = scale
    method = ("log" if log_p else "") + ("cdf" if lower_tail else "sf")
    return getattr(stats.invgauss, method)(q, **_scipy_args(mean, dispersion))


def qinvgauss(
    p,
    mean=1,
    dispersion=1,
    lower_tail=True,
    log_p=False,
    *,
    loc=None,
    scale=None,
):
    """Inverse Gaussian distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    mean : float
        Mean of the distribution.
    dispersion : float
        Dispersion (the reciprocal of the shape ``lambda``).
    lower_tail : bool
        If True, probabilities are P[X <= x], otherwise P[X > x].
    log_p : bool
        If True, probabilities are given as log(p).
    loc : float, optional
        Deprecated alias of ``mean``, kept for one release.
    scale : float, optional
        Deprecated alias of ``dispersion``, kept for one release.

    Returns
    -------
    array
        Quantile values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mean' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mean = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'dispersion' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        dispersion = scale
    p = np.exp(p) if log_p else p
    method = stats.invgauss.ppf if lower_tail else stats.invgauss.isf
    return method(p, **_scipy_args(mean, dispersion))


def rinvgauss(n, mean=1, dispersion=1, *, loc=None, scale=None):
    """Inverse Gaussian distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    mean : float
        Mean of the distribution.
    dispersion : float
        Dispersion (the reciprocal of the shape ``lambda``).
    loc : float, optional
        Deprecated alias of ``mean``, kept for one release.
    scale : float, optional
        Deprecated alias of ``dispersion``, kept for one release.

    Returns
    -------
    array
        Random values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mean' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mean = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'dispersion' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        dispersion = scale
    return stats.invgauss.rvs(size=n, **_scipy_args(mean, dispersion))
