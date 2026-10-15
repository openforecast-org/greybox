"""Helper distribution functions."""

import warnings

import numpy as np
from scipy import special

# R's nmath spells these as correctly-rounded literals. SciPy instead computes
# log(sqrt(2*pi)) at import time, landing 1 ULP low, which is enough to shift a
# summed log-likelihood in the last few digits away from R's.
LN_SQRT_2PI = 0.918938533204672741780329736406
ONE_OVER_SQRT_2PI = 0.398942280401432677939946059934

# Beyond this |z| the density underflows to zero anyway:
# sqrt(-2*ln2*(DBL_MIN_EXP + 1 - DBL_MANT_DIG)).  R/nmath/dnorm.c, following
# Welinder's PR#15620.
_Z_UNDERFLOW = 38.56804181549334


def dnorm(x=None, mean=0.0, sd=1.0, log=False, *, q=None, loc=None, scale=None):
    """Normal distribution density.

    Parameters
    ----------
    x : array_like
        Quantiles.
    mean : float
        Mean (μ).
    sd : float
        Standard deviation (σ).
    log : bool
        If True, return log-density.
    q : array_like, optional
        Deprecated alias of ``x``, kept for one release.
    loc : float, optional
        Deprecated alias of ``mean``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sd``, kept for one release.

    Returns
    -------
    array
        Density values.
    """
    if q is not None:
        warnings.warn(
            "'q' is deprecated, use 'x' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        x = q
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mean' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mean = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sd' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sd = scale
    if x is None:
        raise TypeError("dnorm() missing required argument: 'x'")
    x, mean, sd = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (x, mean, sd))
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = np.abs((x - mean) / sd)

        if log:
            out = -(LN_SQRT_2PI + 0.5 * z * z + np.log(sd))
        else:
            out = np.where(sd < 0, np.nan, _dnorm_density(z, sd))

        # A zero sd is a point mass at the mean, infinite on either measure, and
        # an infinite one has no density, as in R. Everywhere else z already
        # carries the degenerate inputs through to the right answer.
        valid = ~np.isnan(x) & ~np.isnan(mean)
        out = np.where(
            (sd == 0) & valid,
            np.where(x == mean, np.inf, -np.inf if log else 0.0),
            out,
        )
        out = np.where(np.isposinf(sd) & valid, -np.inf if log else 0.0, out)

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


def pnorm(
    q=None,
    mean=0.0,
    sd=1.0,
    lower_tail=True,
    log_p=False,
    *,
    y=None,
    loc=None,
    scale=None,
    log=None,
):
    """Normal distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    mean : float
        Mean.
    sd : float
        Standard deviation.
    lower_tail : bool
        If True, return lower tail probability.
    log_p : bool
        If True, return log-CDF.
    y : array_like, optional
        Deprecated alias of ``q``, kept for one release.
    loc : float, optional
        Deprecated alias of ``mean``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sd``, kept for one release.
    log : bool, optional
        Deprecated alias of ``log_p``, kept for one release.

    Returns
    -------
    array
        CDF values.
    """
    if y is not None:
        warnings.warn(
            "'y' is deprecated, use 'q' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        q = y
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mean' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mean = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sd' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sd = scale
    if log is not None:
        warnings.warn(
            "'log' is deprecated, use 'log_p' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        log_p = log
    if q is None:
        raise TypeError("pnorm() missing required argument: 'q'")
    q, mean, sd = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (q, mean, sd))
    )

    # As R's pnorm: a zero sd, or an infinite z, is a step at the mean
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (q - mean) / sd
        out = (special.log_ndtr if log_p else special.ndtr)(z if lower_tail else -z)
        step = (q < mean) == lower_tail
        out = np.where(
            np.isfinite(z) | np.isnan(q) | np.isnan(mean) | np.isnan(sd),
            out,
            np.where(step, -np.inf if log_p else 0.0, 0.0 if log_p else 1.0),
        )
        out = np.where((sd < 0) | (np.isinf(q) & (q == mean)), np.nan, out)

    return out if out.ndim else out[()]


def qnorm(p, mean=0.0, sd=1.0, lower_tail=True, log_p=False, *, loc=None, scale=None):
    """Normal distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    mean : float
        Mean.
    sd : float
        Standard deviation.
    lower_tail : bool
        If True, probabilities are P[X <= x], otherwise P[X > x].
    log_p : bool
        If True, probabilities are given as log(p).
    loc : float, optional
        Deprecated alias of ``mean``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sd``, kept for one release.

    Returns
    -------
    array
        Quantile values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mean' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mean = loc
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sd' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sd = scale
    p, mean, sd = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (p, mean, sd))
    )

    # As R's qnorm: the bounds of p first, then a zero sd is a point mass. The
    # upper tail by the symmetry of z, as R takes it, without the 1-p
    with np.errstate(invalid="ignore"):
        z = (special.ndtri_exp if log_p else special.ndtri)(p)
        z = z if lower_tail else -z
        out = np.where(sd < 0, np.nan, np.where(sd == 0, mean, mean + sd * z))
        out = np.where(np.isfinite(z) | np.isnan(mean) | np.isnan(sd), out, z)

    return out if out.ndim else out[()]
