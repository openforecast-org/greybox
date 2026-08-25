"""Helper distribution functions."""

import numpy as np
from scipy import stats

# R's nmath spells these as correctly-rounded literals. SciPy instead computes
# log(sqrt(2*pi)) at import time, landing 1 ULP low, which is enough to shift a
# summed log-likelihood in the last few digits away from R's.
LN_SQRT_2PI = 0.918938533204672741780329736406
ONE_OVER_SQRT_2PI = 0.398942280401432677939946059934

# Beyond this |z| the density underflows to zero anyway:
# sqrt(-2*ln2*(DBL_MIN_EXP + 1 - DBL_MANT_DIG)).  R/nmath/dnorm.c, following
# Welinder's PR#15620.
_Z_UNDERFLOW = 38.56804181549334


def dnorm(q, loc=0.0, scale=1.0, log=False):
    """Normal distribution density.

    For the normal distribution, ``loc`` is the mean (μ) and ``scale`` is
    the standard deviation (σ).

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Location parameter (mean, μ).
    scale : float
        Scale parameter (standard deviation, σ).
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

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = np.abs((q - loc) / scale)

        if log:
            out = -(LN_SQRT_2PI + 0.5 * z * z + np.log(scale))
        else:
            out = np.where(scale < 0, np.nan, _dnorm_density(z, scale))

        # A zero scale is a point mass at loc, infinite on either measure.
        # Everywhere else z already carries the degenerate inputs through to
        # the right answer.
        out = np.where(
            scale == 0,
            np.where(q == loc, np.inf, -np.inf if log else 0.0),
            out,
        )

    return out if out.ndim else out[()]


def _dnorm_density(z, scale):
    """Normal density from |z|, reproducing R's two branches term for term.

    Below |z| = 5 R divides the whole product by sigma; above it R divides the
    constant by sigma first and splits z = z1 + z2 with |z2| <= 2^-16 so that
    z1*z1 is exact. The two groupings round differently, so both are kept.
    """
    simple = ONE_OVER_SQRT_2PI * np.exp(-0.5 * z * z) / scale

    z1 = np.ldexp(np.round(np.ldexp(z, 16)), -16)
    z2 = z - z1
    split = (ONE_OVER_SQRT_2PI / scale) * (
        np.exp(-0.5 * z1 * z1) * np.exp((-0.5 * z2 - z1) * z2)
    )

    return np.where(z < 5.0, simple, np.where(z > _Z_UNDERFLOW, 0.0, split))


def plogis(y, loc=0.0, scale=1.0, log=False, lower_tail=True):
    """Logistic distribution CDF.

    Parameters
    ----------
    y : array_like
        Quantiles.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    log : bool
        If True, return log-CDF.
    lower_tail : bool
        If True, return lower tail probability.

    Returns
    -------
    array
        CDF values.
    """
    if log:
        if lower_tail:
            return stats.logistic.logcdf(y, loc=loc, scale=scale)
        else:
            return stats.logistic.logsf(y, loc=loc, scale=scale)
    result = stats.logistic.cdf(y, loc=loc, scale=scale)
    if not lower_tail:
        result = 1 - result
    return result


def pnorm(y, loc=0.0, scale=1.0, log=False, lower_tail=True):
    """Normal distribution CDF.

    Parameters
    ----------
    y : array_like
        Quantiles.
    loc : float
        Location parameter (mean).
    scale : float
        Scale parameter (standard deviation).
    log : bool
        If True, return log-CDF.
    lower_tail : bool
        If True, return lower tail probability.

    Returns
    -------
    array
        CDF values.
    """
    if log:
        if lower_tail:
            return stats.norm.logcdf(y, loc=loc, scale=scale)
        else:
            return stats.norm.logsf(y, loc=loc, scale=scale)
    result = stats.norm.cdf(y, loc=loc, scale=scale)
    if not lower_tail:
        result = 1 - result
    return result


def qnorm(p, loc=0.0, scale=1.0):
    """Normal distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    loc : float
        Location parameter (mean).
    scale : float
        Scale parameter (standard deviation).

    Returns
    -------
    array
        Quantile values.
    """
    return stats.norm.ppf(p, loc=loc, scale=scale)
