"""Beta distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Beta distribution, central or, with a nonzero ``ncp``,
noncentral: the distribution of X / (X + Y) for X ~ chi2(2*shape1, ncp) and
Y ~ chi2(2*shape2), as R defines it. SciPy has no noncentral Beta: its
probabilities and quantiles come from the noncentral F of
(X / (2*shape1)) / (Y / (2*shape2)).
"""

import warnings

import numpy as np
from scipy import special, stats

try:
    from greybox import _native_densities  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - the extension is built with the package
    _native_densities = None


def dbeta(x=None, shape1=1, shape2=1, ncp=0, log=False, *, q=None, a=None, b=None):
    """Beta distribution density.

    Parameters
    ----------
    x : array_like
        Quantiles (must be in [0, 1]).
    shape1 : float
        First shape parameter (alpha).
    shape2 : float
        Second shape parameter (beta).
    ncp : float
        Non-centrality parameter.
    log : bool
        If True, return log-density.
    q : array_like, optional
        Deprecated alias of ``x``, kept for one release.
    a : float, optional
        Deprecated alias of ``shape1``, kept for one release.
    b : float, optional
        Deprecated alias of ``shape2``, kept for one release.

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
    if a is not None:
        warnings.warn(
            "'a' is deprecated, use 'shape1' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        shape1 = a
    if b is not None:
        warnings.warn(
            "'b' is deprecated, use 'shape2' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        shape2 = b
    if x is None:
        raise TypeError("dbeta() missing required argument: 'x'")
    if np.all(np.asarray(ncp) == 0):
        if _native_densities is None:  # pragma: no cover - fallback path
            if log:
                return stats.beta.logpdf(x, a=shape1, b=shape2)
            return stats.beta.pdf(x, a=shape1, b=shape2)
        # R's nmath dbeta(), through libm
        args = (
            np.asarray(x, dtype=float),
            np.atleast_1d(np.asarray(shape1, dtype=float)),
            np.atleast_1d(np.asarray(shape2, dtype=float)),
        )
        density = _native_densities.dbeta_log(*args).reshape(
            np.broadcast(x, shape1, shape2).shape
        )
        return (density if log else np.exp(density))[()]

    # As R's dnbeta, the Poisson(ncp/2) mixture of Beta(shape1 + j, shape2),
    # summed far enough into the tail of the Poisson for its weights to vanish
    x, shape1, shape2, half = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, shape1, shape2, np.asarray(ncp) / 2))
    )
    terms = np.arange(np.ceil(np.max(half) + 10 * np.sqrt(np.max(half)) + 40))
    terms = terms.reshape((-1,) + (1,) * x.ndim)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = special.logsumexp(
            stats.poisson.logpmf(terms, half)
            + stats.beta.logpdf(x, shape1 + terms, shape2),
            axis=0,
        )
    return (density if log else np.exp(density))[()]


def pbeta(
    q, shape1=1, shape2=1, ncp=0, lower_tail=True, log_p=False, *, a=None, b=None
):
    """Beta distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    shape1 : float
        First shape parameter.
    shape2 : float
        Second shape parameter.
    ncp : float
        Non-centrality parameter.
    lower_tail : bool
        If True, return lower tail probability.
    log_p : bool
        If True, return log-CDF.
    a : float, optional
        Deprecated alias of ``shape1``, kept for one release.
    b : float, optional
        Deprecated alias of ``shape2``, kept for one release.

    Returns
    -------
    array
        CDF values.
    """
    if a is not None:
        warnings.warn(
            "'a' is deprecated, use 'shape1' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        shape1 = a
    if b is not None:
        warnings.warn(
            "'b' is deprecated, use 'shape2' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        shape2 = b
    method = ("log" if log_p else "") + ("cdf" if lower_tail else "sf")
    if np.all(np.asarray(ncp) == 0):
        return getattr(stats.beta, method)(q, a=shape1, b=shape2)

    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore"):
        ratio = np.where(q < 1, q / (1 - q), np.inf) * shape2 / shape1
    return getattr(stats.ncf, method)(ratio, 2 * shape1, 2 * shape2, ncp)


def qbeta(
    p, shape1=1, shape2=1, ncp=0, lower_tail=True, log_p=False, *, a=None, b=None
):
    """Beta distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    shape1 : float
        First shape parameter.
    shape2 : float
        Second shape parameter.
    ncp : float
        Non-centrality parameter.
    lower_tail : bool
        If True, probabilities are P[X <= x], otherwise P[X > x].
    log_p : bool
        If True, probabilities are given as log(p).
    a : float, optional
        Deprecated alias of ``shape1``, kept for one release.
    b : float, optional
        Deprecated alias of ``shape2``, kept for one release.

    Returns
    -------
    array
        Quantile values.
    """
    if a is not None:
        warnings.warn(
            "'a' is deprecated, use 'shape1' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        shape1 = a
    if b is not None:
        warnings.warn(
            "'b' is deprecated, use 'shape2' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        shape2 = b
    p = np.exp(p) if log_p else p
    if np.all(np.asarray(ncp) == 0):
        method = stats.beta.ppf if lower_tail else stats.beta.isf
        return method(p, a=shape1, b=shape2)

    method = stats.ncf.ppf if lower_tail else stats.ncf.isf
    with np.errstate(divide="ignore"):
        return 1 / (1 + shape2 / shape1 / method(p, 2 * shape1, 2 * shape2, ncp))


def rbeta(n, shape1=1, shape2=1, ncp=0, *, a=None, b=None):
    """Beta distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    shape1 : float
        First shape parameter.
    shape2 : float
        Second shape parameter.
    ncp : float
        Non-centrality parameter.
    a : float, optional
        Deprecated alias of ``shape1``, kept for one release.
    b : float, optional
        Deprecated alias of ``shape2``, kept for one release.

    Returns
    -------
    array
        Random values.
    """
    if a is not None:
        warnings.warn(
            "'a' is deprecated, use 'shape1' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        shape1 = a
    if b is not None:
        warnings.warn(
            "'b' is deprecated, use 'shape2' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        shape2 = b
    if np.all(np.asarray(ncp) == 0):
        return stats.beta.rvs(a=shape1, b=shape2, size=n)
    # As R's rbeta with ncp
    numerator = stats.ncx2.rvs(2 * shape1, ncp, size=n)
    return numerator / (numerator + stats.chi2.rvs(2 * shape2, size=n))
