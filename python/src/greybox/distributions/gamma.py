"""Gamma distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Gamma distribution.
"""

import numpy as np
from scipy import stats

try:
    from greybox import _native_dgamma as _dgamma_native  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - source checkout without the extension
    _dgamma_native = None


def dgamma(q, shape=1, scale=1, log=False):
    """Gamma distribution density.

    Matches R's :func:`stats::dgamma`. The log-density is computed by Loader's
    saddle-point form, as R does, rather than by evaluating
    ``(a-1)*log(x) - x/s - lgamma(a) - a*log(s)`` directly: that expression
    cancels once the shape grows, losing about 1.4e-13 at a shape of 143, which
    is enough to move an optimiser off a boundary. The saddle-point form is
    also faster here, because it avoids SciPy's per-call overhead.

    Parameters
    ----------
    q : array_like
        Quantiles (must be positive).
    shape : float or array_like
        Shape parameter (alpha). May vary per observation.
    scale : float or array_like
        Scale parameter (theta). May vary per observation.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    if _dgamma_native is None:  # pragma: no cover - fallback path
        if log:
            return stats.gamma.logpdf(q, a=shape, scale=scale)
        return stats.gamma.pdf(q, a=shape, scale=scale)

    out = _dgamma_native.dgamma_log(
        np.asarray(q, dtype=float),
        np.atleast_1d(np.asarray(shape, dtype=float)),
        np.atleast_1d(np.asarray(scale, dtype=float)),
    )
    return out if log else np.exp(out)


def pgamma(q, shape=1, scale=1):
    """Gamma distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    shape : float
        Shape parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        CDF values.
    """
    return stats.gamma.cdf(q, a=shape, scale=scale)


def qgamma(p, shape=1, scale=1):
    """Gamma distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    shape : float
        Shape parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.gamma.ppf(p, a=shape, scale=scale)


def rgamma(n, shape=1, scale=1):
    """Gamma distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    shape : float
        Shape parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Random values.
    """
    return stats.gamma.rvs(a=shape, scale=scale, size=n)
