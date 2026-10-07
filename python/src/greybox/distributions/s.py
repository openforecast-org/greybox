"""S-Distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the S distribution.
"""

import numpy as np

try:
    from greybox import _native_densities  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - source checkout without the extension
    _native_densities = None

from .gnorm import qgnorm


def ds(q, loc=0, scale=1, log=False):
    """S-distribution density.

    Density function: f(x) = 1/(4*scale^2) * exp(-sqrt(abs(loc - x)) / scale)

    Parameters
    ----------
    q : array_like
        Quantiles.
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
    if _native_densities is None:  # pragma: no cover - fallback path
        if log:
            return -2 * np.log(2 * scale) - np.sqrt(np.abs(loc - q)) / scale
        return 1 / (4 * scale**2) * np.exp(-np.sqrt(np.abs(loc - q)) / scale)

    # Through libm, as R computes it, and the log analytically
    args = (
        np.asarray(q, dtype=float),
        np.atleast_1d(np.asarray(loc, dtype=float)),
        np.atleast_1d(np.asarray(scale, dtype=float)),
    )
    if log:
        return _native_densities.ds_log(*args)
    return _native_densities.ds(*args)


def ps(q, loc=0, scale=1):
    """S-distribution CDF."""
    sign_val = np.sign(q - loc)
    sqrt_term = np.sqrt(np.abs(loc - q))
    return 0.5 + 0.5 * sign_val * (
        1 - 1 / scale * (sqrt_term + scale) * np.exp(-sqrt_term / scale)
    )


def qs(p, loc=0, scale=1):
    """S-distribution quantile function."""
    p = np.asarray(p)
    return qgnorm(p, loc=loc, scale=scale**2, shape=0.5)


def rs(n, loc=0, scale=1):
    """S-distribution random number generation."""
    return qs(np.random.uniform(0, 1, n), loc, scale)
