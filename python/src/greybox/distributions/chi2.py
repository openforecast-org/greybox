"""Chi-squared distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Chi-squared distribution, central or, with a nonzero
``ncp``, noncentral. ``dchi2``, ``pchi2``, ``qchi2`` and ``rchi2`` are the
deprecated names of ``dchisq``, ``pchisq``, ``qchisq`` and ``rchisq``.
"""

import warnings

import numpy as np
from scipy import stats


def dchisq(x, df, ncp=0, log=False):
    """Chi-squared distribution density.

    Parameters
    ----------
    x : array_like
        Quantiles (must be non-negative).
    df : float
        Degrees of freedom.
    ncp : float
        Non-centrality parameter.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    x = np.asarray(x, dtype=float)
    central = np.all(np.asarray(ncp) == 0)
    family, args = (stats.chi2, (df,)) if central else (stats.ncx2, (df, ncp))
    with np.errstate(invalid="ignore"):
        result = (family.logpdf if log else family.pdf)(x, *args)
    # As R, an infinite x has no density
    return np.where(np.isposinf(x), -np.inf if log else 0.0, result)[()]


def pchisq(q, df, ncp=0, lower_tail=True, log_p=False):
    """Chi-squared distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    df : float
        Degrees of freedom.
    ncp : float
        Non-centrality parameter.
    lower_tail : bool
        If True, return lower tail probability.
    log_p : bool
        If True, return log-CDF.

    Returns
    -------
    array
        CDF values.
    """
    central = np.all(np.asarray(ncp) == 0)
    family, args = (stats.chi2, (df,)) if central else (stats.ncx2, (df, ncp))
    method = ("log" if log_p else "") + ("cdf" if lower_tail else "sf")
    return getattr(family, method)(q, *args)


def qchisq(p, df, ncp=0, lower_tail=True, log_p=False):
    """Chi-squared distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    df : float
        Degrees of freedom.
    ncp : float
        Non-centrality parameter.
    lower_tail : bool
        If True, probabilities are P[X <= x], otherwise P[X > x].
    log_p : bool
        If True, probabilities are given as log(p).

    Returns
    -------
    array
        Quantile values.
    """
    central = np.all(np.asarray(ncp) == 0)
    family, args = (stats.chi2, (df,)) if central else (stats.ncx2, (df, ncp))
    p = np.exp(p) if log_p else p
    return (family.ppf if lower_tail else family.isf)(p, *args)


def rchisq(n, df, ncp=0):
    """Chi-squared distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    df : float
        Degrees of freedom.
    ncp : float
        Non-centrality parameter.

    Returns
    -------
    array
        Random values.
    """
    central = np.all(np.asarray(ncp) == 0)
    family, args = (stats.chi2, (df,)) if central else (stats.ncx2, (df, ncp))
    return family.rvs(*args, size=n)


def dchi2(q, df, log=False):
    """Deprecated name of :func:`dchisq`, kept for one release."""
    warnings.warn(
        "'dchi2' is deprecated, use 'dchisq' instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return dchisq(q, df, log=log)


def pchi2(q, df):
    """Deprecated name of :func:`pchisq`, kept for one release."""
    warnings.warn(
        "'pchi2' is deprecated, use 'pchisq' instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return pchisq(q, df)


def qchi2(p, df):
    """Deprecated name of :func:`qchisq`, kept for one release."""
    warnings.warn(
        "'qchi2' is deprecated, use 'qchisq' instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return qchisq(p, df)


def rchi2(n, df):
    """Deprecated name of :func:`rchisq`, kept for one release."""
    warnings.warn(
        "'rchi2' is deprecated, use 'rchisq' instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return rchisq(n, df)
