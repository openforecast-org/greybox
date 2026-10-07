"""Log-Laplace distribution functions.

Density and quantile functions for the Log-Laplace distribution.
Log-Laplace is the distribution of exp(X) where X ~ Laplace.
Note: Random generation and CDF are not implemented as per requirements.
"""

import numpy as np
from scipy import stats

try:
    from greybox import _native_densities  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - source checkout without the extension
    _native_densities = None

from .laplace import dlaplace


def dllaplace(q, loc=0, scale=1, log=False):
    """Log-Laplace distribution density.

    The density is obtained by transforming a Laplace distribution
    through the exponential function with Jacobian adjustment.

    f(y) = (1/scale) * exp(-(abs(log(y) - loc) / scale)) / y

    Parameters
    ----------
    q : array_like
        Quantiles (must be positive).
    loc : float
        Location parameter (of underlying Laplace).
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
        # As alm() takes it, dlaplace(log(q), log=TRUE) - log(q): analytically
        # and through libm, as R computes it
        if _native_densities is None:  # pragma: no cover - fallback path
            log_q = np.log(q)
            return dlaplace(log_q, loc=loc, scale=scale, log=True) - log_q
        return _native_densities.dllaplace_log(
            np.asarray(q, dtype=float),
            np.atleast_1d(np.asarray(loc, dtype=float)),
            np.atleast_1d(np.asarray(scale, dtype=float)),
        )
    log_q = np.log(q)
    density = stats.laplace.pdf(log_q, loc=loc, scale=scale) / q
    density = np.maximum(density, 1e-300)
    return density


def qllaplace(p, loc=0, scale=1):
    """Log-Laplace distribution quantile function.

    Quantiles are obtained by exponentiating Laplace quantiles.

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
    return np.exp(stats.laplace.ppf(p, loc=loc, scale=scale))
