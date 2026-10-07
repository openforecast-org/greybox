"""Log-S distribution functions.

Density and quantile functions for the Log-S distribution.
Log-S is the distribution of exp(X) where X ~ S-distribution.
Note: Random generation and CDF are not implemented as per requirements.
"""

import numpy as np

try:
    from greybox import _native_densities  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - source checkout without the extension
    _native_densities = None

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
    if log:
        # As alm() takes it, ds(log(q), log=TRUE) - log(q): analytically and
        # through libm, as R computes it
        if _native_densities is None:  # pragma: no cover - fallback path
            log_q = np.log(q)
            return ds(log_q, loc=loc, scale=scale, log=True) - log_q
        return _native_densities.dls_log(
            np.asarray(q, dtype=float),
            np.atleast_1d(np.asarray(loc, dtype=float)),
            np.atleast_1d(np.asarray(scale, dtype=float)),
        )
    log_q = np.log(q)
    density = ds(log_q, loc=loc, scale=scale) / q
    density = np.maximum(density, 1e-300)
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
