"""Log-Normal distribution functions.

Density, cumulative distribution and quantile functions for the Log-Normal distribution.
Note: Random generation is not implemented as per requirements.
"""

import warnings

import numpy as np

from .helper import LN_SQRT_2PI, ONE_OVER_SQRT_2PI, pnorm, qnorm


def dlnorm(x=None, meanlog=0, sdlog=1, log=False, *, q=None, loc=None, scale=None):
    """Log-Normal distribution density.

    Parameters
    ----------
    x : array_like
        Quantiles (must be positive).
    meanlog : float
        Mean of the underlying normal distribution (on log scale).
    sdlog : float
        Standard deviation of the underlying normal distribution.
    log : bool
        If True, return log-density.
    q : array_like, optional
        Deprecated alias of ``x``, kept for one release.
    loc : float, optional
        Deprecated alias of ``meanlog``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sdlog``, kept for one release.

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
            "'loc' is deprecated, use 'meanlog' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        meanlog = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sdlog' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sdlog = scale
    if x is None:
        raise TypeError("dlnorm() missing required argument: 'x'")
    x, meanlog, sdlog = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (x, meanlog, sdlog))
    )

    # R works from log(x) directly rather than routing meanlog through exp()
    # and back, takes a single log of the product x*sdlog, and -- unlike
    # dnorm -- never splits the exponent.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = (np.log(x) - meanlog) / sdlog
        if log:
            out = -(LN_SQRT_2PI + 0.5 * z * z + np.log(x * sdlog))
        else:
            out = ONE_OVER_SQRT_2PI * np.exp(-0.5 * z * z) / (x * sdlog)
        out = np.where(
            (x <= 0) & ~np.isnan(meanlog) & ~np.isnan(sdlog),
            -np.inf if log else 0.0,
            out,
        )
        # As R: a zero sdlog is a point mass at exp(meanlog), a negative one NaN
        out = np.where(
            (sdlog == 0) & ~np.isnan(x) & ~np.isnan(meanlog),
            np.where(np.log(x) == meanlog, np.inf, -np.inf if log else 0.0),
            out,
        )
        out = np.where(sdlog < 0, np.nan, out)

    return out if out.ndim else out[()]


def plnorm(
    q, meanlog=0, sdlog=1, lower_tail=True, log_p=False, *, loc=None, scale=None
):
    """Log-Normal distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    meanlog : float
        Mean of the underlying normal distribution.
    sdlog : float
        Standard deviation of the underlying normal distribution.
    lower_tail : bool
        If True, return lower tail probability.
    log_p : bool
        If True, return log-CDF.
    loc : float, optional
        Deprecated alias of ``meanlog``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sdlog``, kept for one release.

    Returns
    -------
    array
        CDF values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'meanlog' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        meanlog = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sdlog' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sdlog = scale
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pnorm(np.log(q), meanlog, sdlog, lower_tail, log_p)
        bound = np.log(float(not lower_tail)) if log_p else float(not lower_tail)
    out = np.where((q <= 0) & (np.asarray(sdlog) >= 0) & ~np.isnan(meanlog), bound, out)
    return out[()]


def qlnorm(
    p, meanlog=0, sdlog=1, lower_tail=True, log_p=False, *, loc=None, scale=None
):
    """Log-Normal distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    meanlog : float
        Mean of the underlying normal distribution.
    sdlog : float
        Standard deviation of the underlying normal distribution.
    lower_tail : bool
        If True, probabilities are P[X <= x], otherwise P[X > x].
    log_p : bool
        If True, probabilities are given as log(p).
    loc : float, optional
        Deprecated alias of ``meanlog``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sdlog``, kept for one release.

    Returns
    -------
    array
        Quantile values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'meanlog' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        meanlog = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sdlog' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sdlog = scale
    return np.exp(qnorm(p, meanlog, sdlog, lower_tail, log_p))
