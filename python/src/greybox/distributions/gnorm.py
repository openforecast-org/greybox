"""Generalized Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Generalized Normal distribution.
"""

import warnings

import numpy as np
from scipy import stats
from scipy.special import gamma

try:
    from greybox import _native_densities  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - source checkout without the extension
    _native_densities = None


def _density_parameters(scale, shape):
    """The scale and shape of the densities, with their failsafes applied."""
    scale = np.atleast_1d(scale)
    shape = np.atleast_1d(shape)
    scale = np.where(np.isnan(scale), 0, scale)
    scale = np.where(scale < 0, 0, scale)
    shape = np.where(np.isnan(shape), 0, shape)
    return scale, shape


def dgnorm(q, mu=0, scale=1, shape=1, log=False, *, loc=None):
    """Generalized Normal distribution density."""
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    dims = np.broadcast(q, mu, scale, shape).shape
    scale, shape = _density_parameters(scale, shape)
    q = np.atleast_1d(q)

    if _native_densities is None:  # pragma: no cover - fallback path
        if log:
            return (
                np.log(shape)
                - np.log(2 * scale)
                - np.log(gamma(1 / shape))
                - (np.abs(q - mu) / scale) ** shape
            )
        return (
            np.exp(-((np.abs(q - mu) / scale) ** shape))
            * shape
            / (2 * scale * gamma(1 / shape))
        )

    # Through libm, with R's gamma() and lgamma(), as R computes it, and the log
    # analytically
    args = (
        np.asarray(q, dtype=float),
        np.atleast_1d(np.asarray(mu, dtype=float)),
        scale.astype(float),
        shape.astype(float),
    )
    density = _native_densities.dgnorm_log if log else _native_densities.dgnorm
    return density(*args).reshape(dims)[()]


def pgnorm(
    q, mu=0, scale=1, shape=1, lower_tail=True, log_p=False, *, loc=None, log=None
):
    """Generalized Normal distribution CDF."""
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    if log is not None:
        warnings.warn(
            "'log' is deprecated, use 'log_p' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        log_p = log
    q = np.asarray(q, dtype=float)
    scale = np.asarray(scale, dtype=float)
    shape = np.asarray(shape, dtype=float)

    scale = np.where(np.isnan(scale), 0, scale)
    scale = np.where(scale < 0, 0, scale)
    shape = np.where(np.isnan(shape), 0, shape)

    # As R, with the rate (1/scale)^shape; uniform where shape is too high
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        p = (
            1 / 2
            + np.sign(q - mu)
            * stats.gamma.cdf(
                np.abs(q - mu) ** shape, a=1 / shape, scale=1 / (1 / scale) ** shape
            )
            / 2
        )
        lower, upper = mu - scale, mu + scale
        p = np.where(
            shape > 100,
            np.where(
                q >= upper,
                1.0,
                np.where(q <= lower, 0.0, (q - lower) / (upper - lower)),
            ),
            p,
        )

    if not lower_tail:
        p = 1 - p

    if log_p:
        with np.errstate(divide="ignore"):
            p = np.log(p)

    return p[()]


def qgnorm(
    p, mu=0, scale=1, shape=1, lower_tail=True, log_p=False, *, loc=None, log=None
):
    """Generalized Normal distribution quantile function."""
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    if log is not None:
        warnings.warn(
            "'log' is deprecated, use 'log_p' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        log_p = log
    p = np.asarray(p, dtype=float)
    scale = np.asarray(scale, dtype=float)
    shape = np.asarray(shape, dtype=float)

    scale = np.where(np.isnan(scale), 0, scale)
    scale = np.where(scale < 0, 0, scale)
    shape = np.where(np.isnan(shape), 0, shape)

    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p

    if np.all(shape > 100):
        lower, upper = mu - scale, mu + scale
        result = np.where((p < 0) | (p > 1), np.nan, lower + p * (upper - lower))
    else:
        # The upper tail of the gamma from the smaller of p and 1-p, as |p-0.5|*2
        # cancels near p=0. The bounds of the tail first, as R's qgamma
        tail = 2 * np.minimum(p, 1 - p)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lambdaScale = (
                np.ceil(scale) / 10 if np.any((1 / scale) ** shape < 1e-300) else 1
            )
            gamma_scale = (scale / lambdaScale) ** shape
            bounded = ~np.isnan(gamma_scale)
            quantile = np.where(
                bounded & (tail == 0),
                np.inf,
                np.where(
                    bounded & (tail == 1),
                    0.0,
                    stats.gamma.isf(tail, a=1 / shape, scale=gamma_scale),
                ),
            )
            result = np.sign(p - 0.5) * quantile ** (1 / shape) * lambdaScale + mu

    return result[()]


def rgnorm(n, mu=0, scale=1, shape=1, *, loc=None):
    """Generalized Normal distribution random number generation."""
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    return qgnorm(np.random.uniform(0, 1, n), mu=mu, scale=scale, shape=shape)
