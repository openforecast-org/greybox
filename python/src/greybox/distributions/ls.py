"""Log-S distribution functions.

Density and quantile functions for the Log-S distribution.
Log-S is the distribution of exp(X) where X ~ S-distribution.
Note: Random generation and CDF are not implemented as per requirements.
"""

import numpy as np

from .s import ds


def dls(q, loc=0, scale=1, log=False):
    """Log-S distribution density.

    The density is obtained by transforming an S-distribution
    through the exponential function with Jacobian adjustment.

    Parameters
    ----------
    q : array_like
        Quantiles (must be positive).
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    q = np.asarray(q)
    log_q = np.log(q)
    density = ds(log_q, loc=loc, scale=scale) / q
    density = np.maximum(density, 1e-300)
    if log:
        return np.log(density)
    return density


def qls(p, loc=0, scale=1):
    """Log-S distribution quantile function.

    Quantiles are obtained by exponentiating S-distribution quantiles.

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
    from .s import qs

    return np.exp(qs(p, loc=loc, scale=scale))
