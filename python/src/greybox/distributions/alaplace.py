"""Asymmetric Laplace distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Asymmetric Laplace distribution.
"""

import warnings

import numpy as np


def dalaplace(q, mu=0, scale=1, alpha=0.5, log=False, *, loc=None):
    """Asymmetric Laplace distribution density.

    f(x) = alpha * (1-alpha) / scale * exp(-(x-mu)/scale * (alpha - I(x<=mu)))

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    alpha : float
        Asymmetry parameter (0 < alpha < 1).
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
    q, mu, scale, alpha = (np.asarray(v, dtype=float) for v in (q, mu, scale, alpha))
    indicator = (q <= mu).astype(float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if log:
            density = np.log(alpha * (1 - alpha) / scale) - (q - mu) / scale * (
                alpha - indicator
            )
        else:
            density = (
                alpha
                * (1 - alpha)
                / scale
                * np.exp(-(q - mu) / scale * (alpha - indicator))
            )
    return density[()]


def palaplace(q, mu=0, scale=1, alpha=0.5, *, loc=None):
    """Asymmetric Laplace distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    alpha : float
        Asymmetry parameter (0 < alpha < 1).
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.

    Returns
    -------
    array
        CDF values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    q, mu, scale, alpha = (np.asarray(v, dtype=float) for v in (q, mu, scale, alpha))
    indicator = (q <= mu).astype(float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = (
            1
            - indicator
            - (1 - indicator - alpha) * np.exp((indicator - alpha) / scale * (q - mu))
        )
    return result[()]


def qalaplace(p, mu=0, scale=1, alpha=0.5, *, loc=None):
    """Asymmetric Laplace distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    alpha : float
        Asymmetry parameter (0 < alpha < 1).
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
    # Vectorised over p, mu, scale and alpha, as R's qalaplace
    p, mu, scale, alpha = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (p, mu, scale, alpha))
    )
    result = np.empty_like(p, dtype=float)

    mask_0 = p == 0
    mask_1 = p == 1
    mask_mid = ~(mask_0 | mask_1)

    result[mask_0] = -np.inf
    result[mask_1] = np.inf

    if np.any(mask_mid):
        p_mid, alpha_mid = p[mask_mid], alpha[mask_mid]
        indicator = (p_mid <= alpha_mid).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            result[mask_mid] = mu[mask_mid] + scale[mask_mid] / (
                indicator - alpha_mid
            ) * np.log((1 - indicator - p_mid) / (1 - indicator - alpha_mid))

    return result[()]


def ralaplace(n=1, mu=0, scale=1, alpha=0.5, *, loc=None):
    """Asymmetric Laplace distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    alpha : float
        Asymmetry parameter (0 < alpha < 1).
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.

    Returns
    -------
    array
        Random values.
    """
    if loc is not None:
        warnings.warn(
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    return qalaplace(np.random.uniform(0, 1, n), mu, scale, alpha)
