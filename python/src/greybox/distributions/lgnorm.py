"""Log-Generalised Normal distribution functions.

Density and quantile functions for the Log-Generalised Normal distribution.
Log-GN is the distribution of exp(X) where X ~ Generalised Normal.
Note: Random generation and CDF are not implemented as per requirements.
"""

import numpy as np

try:
    from greybox import _native_densities  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - source checkout without the extension
    _native_densities = None

from .gnorm import _density_parameters, dgnorm, qgnorm


def dlgnorm(q, loc=0, scale=1, shape=1, log=False):
    """Log-Generalised Normal distribution density.

    The density is obtained by transforming a Generalised Normal distribution
    through the exponential function with Jacobian adjustment.

    Parameters
    ----------
    q : array_like
        Quantiles (must be positive).
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    shape : float
        Shape parameter.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    q = np.asarray(q)
    if log:
        # As alm() takes it, dgnorm(log(q), log=TRUE) - log(q): analytically and
        # through libm, with R's lgamma(), as R computes it
        if _native_densities is None:  # pragma: no cover - fallback path
            log_q = np.log(q)
            return dgnorm(log_q, loc=loc, scale=scale, shape=shape, log=True) - log_q
        scale, shape = _density_parameters(scale, shape)
        return _native_densities.dlgnorm_log(
            np.asarray(q, dtype=float),
            np.atleast_1d(np.asarray(loc, dtype=float)),
            scale.astype(float),
            shape.astype(float),
        )
    log_q = np.log(q)
    density = dgnorm(log_q, loc=loc, scale=scale, shape=shape) / q
    density = np.maximum(density, 1e-300)
    return density


def qlgnorm(p, loc=0, scale=1, shape=1):
    """Log-Generalised Normal distribution quantile function.

    Quantiles are obtained by exponentiating Generalised Normal quantiles.

    Parameters
    ----------
    p : array_like
        Probabilities.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    shape : float
        Shape parameter.

    Returns
    -------
    array
        Quantile values.
    """
    return np.exp(qgnorm(p, loc=loc, scale=scale, shape=shape))
