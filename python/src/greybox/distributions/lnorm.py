"""Log-Normal distribution functions.

Density, cumulative distribution and quantile functions for the Log-Normal distribution.
Note: Random generation is not implemented as per requirements.
"""

import numpy as np
from scipy import stats

from .helper import LN_SQRT_2PI, ONE_OVER_SQRT_2PI


def dlnorm(q, loc=0, scale=1, log=False):
    """Log-Normal distribution density.

    ``loc`` is the mean of the underlying normal on the log scale (meanlog),
    ``scale`` is the corresponding standard deviation (sdlog).

    Parameters
    ----------
    q : array_like
        Quantiles (must be positive).
    loc : float
        Mean of the underlying normal distribution (on log scale).
    scale : float
        Standard deviation of the underlying normal distribution.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    q, loc, scale = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (q, loc, scale))
    )

    # R works from log(q) directly rather than routing meanlog through exp()
    # and back, takes a single log of the product q*sdlog, and -- unlike
    # dnorm -- never splits the exponent.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = (np.log(q) - loc) / scale
        if log:
            out = -(LN_SQRT_2PI + 0.5 * z * z + np.log(q * scale))
        else:
            out = ONE_OVER_SQRT_2PI * np.exp(-0.5 * z * z) / (q * scale)
        out = np.where(q <= 0, -np.inf if log else 0.0, out)

    return out if out.ndim else out[()]


def plnorm(q, loc=0, scale=1):
    """Log-Normal distribution CDF.

    ``loc`` is the mean of the underlying normal on the log scale (meanlog),
    ``scale`` is the corresponding standard deviation (sdlog).

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Mean of the underlying normal distribution.
    scale : float
        Standard deviation of the underlying normal distribution.

    Returns
    -------
    array
        CDF values.
    """
    return stats.lognorm.cdf(q, s=scale, scale=np.exp(loc))


def qlnorm(p, loc=0, scale=1):
    """Log-Normal distribution quantile function.

    ``loc`` is the mean of the underlying normal on the log scale (meanlog),
    ``scale`` is the corresponding standard deviation (sdlog).

    Parameters
    ----------
    p : array_like
        Probabilities.
    loc : float
        Mean of the underlying normal distribution.
    scale : float
        Standard deviation of the underlying normal distribution.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.lognorm.ppf(p, s=scale, scale=np.exp(loc))
