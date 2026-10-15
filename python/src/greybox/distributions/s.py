"""S-Distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the S distribution.
"""

import warnings

import numpy as np

try:
    from greybox import _native_densities  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - source checkout without the extension
    _native_densities = None

from .gnorm import qgnorm


def ds(q, mu=0, scale=1, log=False, *, loc=None):
    """S-distribution density.

    Density function: f(x) = 1/(4*scale^2) * exp(-sqrt(abs(mu - x)) / scale)

    Parameters
    ----------
    q : array_like
        Quantiles.
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
    if _native_densities is None:  # pragma: no cover - fallback path
        if log:
            return -2 * np.log(2 * scale) - np.sqrt(np.abs(mu - q)) / scale
        return 1 / (4 * scale**2) * np.exp(-np.sqrt(np.abs(mu - q)) / scale)

    # Through libm, as R computes it, and the log analytically
    args = (
        np.asarray(q, dtype=float),
        np.atleast_1d(np.asarray(mu, dtype=float)),
        np.atleast_1d(np.asarray(scale, dtype=float)),
    )
    density = _native_densities.ds_log if log else _native_densities.ds
    return density(*args).reshape(np.broadcast(q, mu, scale).shape)[()]


def ps(q, mu=0, scale=1, *, loc=None):
    """S-distribution CDF."""
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    q, mu, scale = (np.asarray(v, dtype=float) for v in (q, mu, scale))
    # The tail as exp() directly, without the cancellation of 1-exp()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sign_val = np.sign(q - mu)
        sqrt_term = np.sqrt(np.abs(mu - q))
        result = 0.5 * (1 + sign_val) - sign_val * 0.5 * (
            1 / scale * (sqrt_term + scale) * np.exp(-sqrt_term / scale)
        )
    return np.where(scale < 0, np.nan, result)[()]


def qs(p, mu=0, scale=1, *, loc=None):
    """S-distribution quantile function."""
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    return qgnorm(p, mu=mu, scale=np.asarray(scale, dtype=float) ** 2, shape=0.5)


def rs(n=1, mu=0, scale=1, *, loc=None):
    """S-distribution random number generation."""
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    return qs(np.random.uniform(0, 1, n), mu, scale)
