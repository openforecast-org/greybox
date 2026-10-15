"""T-distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Student's t-distribution, central or, with ``ncp``,
noncentral. ``loc`` and ``scale`` shift and stretch it; R's ``dt`` has neither.
"""

import warnings

import numpy as np
from scipy import stats


def dt(x=None, df=None, ncp=None, log=False, *, loc=0, scale=1, q=None):
    """T-distribution density.

    Parameters
    ----------
    x : array_like
        Quantiles.
    df : float
        Degrees of freedom.
    ncp : float, optional
        Non-centrality parameter. If None, the central t-distribution.
    log : bool
        If True, return log-density.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    q : array_like, optional
        Deprecated alias of ``x``, kept for one release.

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
    if x is None:
        raise TypeError("dt() missing required argument: 'x'")
    family, args = (stats.t, (df,)) if ncp is None else (stats.nct, (df, ncp))
    return (family.logpdf if log else family.pdf)(x, *args, loc=loc, scale=scale)


def pt(q, df, ncp=None, lower_tail=True, log_p=False, *, loc=0, scale=1):
    """T-distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    df : float
        Degrees of freedom.
    ncp : float, optional
        Non-centrality parameter. If None, the central t-distribution.
    lower_tail : bool
        If True, return lower tail probability.
    log_p : bool
        If True, return log-CDF.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        CDF values.
    """
    family, args = (stats.t, (df,)) if ncp is None else (stats.nct, (df, ncp))
    method = ("log" if log_p else "") + ("cdf" if lower_tail else "sf")
    return getattr(family, method)(q, *args, loc=loc, scale=scale)


def qt(p, df, ncp=None, lower_tail=True, log_p=False, *, loc=0, scale=1):
    """T-distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    df : float
        Degrees of freedom.
    ncp : float, optional
        Non-centrality parameter. If None, the central t-distribution.
    lower_tail : bool
        If True, probabilities are P[X <= x], otherwise P[X > x].
    log_p : bool
        If True, probabilities are given as log(p).
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Quantile values.
    """
    family, args = (stats.t, (df,)) if ncp is None else (stats.nct, (df, ncp))
    p = np.exp(p) if log_p else p
    method = family.ppf if lower_tail else family.isf
    return method(p, *args, loc=loc, scale=scale)


def rt(n, df, ncp=None, *, loc=0, scale=1):
    """T-distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    df : float
        Degrees of freedom.
    ncp : float, optional
        Non-centrality parameter. If None, the central t-distribution.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Random values.
    """
    family, args = (stats.t, (df,)) if ncp is None else (stats.nct, (df, ncp))
    return family.rvs(*args, loc=loc, scale=scale, size=n)
