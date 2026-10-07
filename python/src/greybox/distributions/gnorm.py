"""Generalized Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Generalized Normal distribution.
"""

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
    shape = np.where(shape == 0, 1e-10, shape)
    return scale, shape


def dgnorm(q, loc=0, scale=1, shape=1, log=False):
    """Generalized Normal distribution density."""
    scale, shape = _density_parameters(scale, shape)
    q = np.atleast_1d(q)

    if _native_densities is None:  # pragma: no cover - fallback path
        if log:
            return (
                np.log(shape)
                - np.log(2 * scale)
                - np.log(gamma(1 / shape))
                - (np.abs(q - loc) / scale) ** shape
            )
        return (
            np.exp(-((np.abs(q - loc) / scale) ** shape))
            * shape
            / (2 * scale * gamma(1 / shape))
        )

    # Analytically and through libm, with R's lgamma(), as R computes it
    if log:
        return _native_densities.dgnorm_log(
            np.asarray(q, dtype=float),
            np.atleast_1d(np.asarray(loc, dtype=float)),
            scale.astype(float),
            shape.astype(float),
        )
    return (
        np.exp(-((np.abs(q - loc) / scale) ** shape))
        * shape
        / (2 * scale * _native_densities.gammafn(1 / shape))
    )


def pgnorm(q, loc=0, scale=1, shape=1, lower_tail=True, log=False):
    """Generalized Normal distribution CDF."""
    scale = np.atleast_1d(scale)
    shape = np.atleast_1d(shape)

    scale = np.where(np.isnan(scale), 0, scale)
    scale = np.where(scale < 0, 0, scale)
    shape = np.where(np.isnan(shape), 0, shape)
    shape = np.where(shape == 0, 1e-10, shape)

    if np.any(shape > 100):
        p = stats.uniform.cdf(q, loc=loc - scale, scale=2 * scale)
    else:
        p = (
            1 / 2
            + np.sign(q - loc)
            * stats.gamma.cdf(
                np.abs(q - loc) ** shape, a=1 / shape, scale=(1 / scale) ** shape
            )
            / 2
        )

    if not lower_tail:
        p = 1 - p

    if log:
        p = np.log(np.clip(p, 1e-300, None))

    return p


def qgnorm(p, loc=0, scale=1, shape=1, lower_tail=True, log=False):
    """Generalized Normal distribution quantile function."""
    p = np.asarray(p)
    scale = np.atleast_1d(scale)
    shape = np.atleast_1d(shape)

    scale = np.where(np.isnan(scale), 0, scale)
    scale = np.where(scale < 0, 0, scale)
    shape = np.where(np.isnan(shape), 0, shape)
    shape = np.where(shape == 0, 1e-10, shape)

    if log:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p

    if np.all(shape > 100):
        result = stats.uniform.ppf(p, loc=loc - scale, scale=2 * scale)
    elif np.any((1 / scale) ** shape < 1e-300):
        lambdaScale = np.ceil(scale) / 10
        lambda_val = (scale / lambdaScale) ** shape
        result = (
            np.sign(p - 0.5)
            * (
                stats.gamma.ppf(np.abs(p - 0.5) * 2, a=1 / shape, scale=lambda_val)
                ** (1 / shape)
            )
            * lambdaScale
            + loc
        )
    else:
        lambda_val = scale**shape
        result = (
            np.sign(p - 0.5)
            * stats.gamma.ppf(np.abs(p - 0.5) * 2, a=1 / shape, scale=lambda_val)
            ** (1 / shape)
            + loc
        )

    return result


def rgnorm(n, loc=0, scale=1, shape=1):
    """Generalized Normal distribution random number generation."""
    return qgnorm(np.random.uniform(0, 1, n), loc=loc, scale=scale, shape=shape)
