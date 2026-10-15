"""Log-S distribution functions.

Density and quantile functions for the Log-S distribution.
Log-S is the distribution of exp(X) where X ~ S-distribution.
Note: Random generation and CDF are not implemented as per requirements.
"""

import warnings

import numpy as np

try:
    from greybox import _native_densities  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - source checkout without the extension
    _native_densities = None

from .s import ds


def dls(q, mu=0, scale=1, log=False, *, loc=None):
    """Log-S distribution density.

    The density is obtained by transforming an S-distribution
    through the exponential function with Jacobian adjustment.

    Parameters
    ----------
    q : array_like
        Quantiles (must be positive).
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    log : bool
        If True, return log-density.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.

    Returns
    -------
    array
        Density values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    q = np.asarray(q)
    if log:
        # As alm() takes it, ds(log(q), log=TRUE) - log(q): analytically and
        # through libm, as R computes it
        if _native_densities is None:  # pragma: no cover - fallback path
            log_q = np.log(q)
            return ds(log_q, mu=mu, scale=scale, log=True) - log_q
        return _native_densities.dls_log(
            np.asarray(q, dtype=float),
            np.atleast_1d(np.asarray(mu, dtype=float)),
            np.atleast_1d(np.asarray(scale, dtype=float)),
        ).reshape(np.broadcast(q, mu, scale).shape)[()]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_q = np.log(q)
        density = ds(log_q, mu=mu, scale=scale) / q
    return density


def qls(p, mu=0, scale=1, *, loc=None):
    """Log-S distribution quantile function.

    Quantiles are obtained by exponentiating S-distribution quantiles.

    Parameters
    ----------
    p : array_like
        Probabilities.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.

    Returns
    -------
    array
        Quantile values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    from .s import qs

    return np.exp(qs(p, mu=mu, scale=scale))
