"""Negative Binomial distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Negative Binomial distribution. The arguments keep the order
(mu, size), with R's ``prob`` by keyword, where R's is (size, prob, mu).
"""

import warnings

import numpy as np
from scipy import special, stats


def dnbinom(x=None, mu=None, size=1, log=False, *, prob=None, q=None, loc=None):
    """Negative Binomial distribution probability mass function.

    Parameters
    ----------
    x : array_like
        Quantiles (non-negative integers).
    mu : float, optional
        Mean parameter. The default, without ``prob``, is 1.
    size : float
        Dispersion parameter (number of successes).
    log : bool
        If True, return log-probability.
    prob : float, optional
        Probability of success in each trial, instead of ``mu``.
    q : array_like, optional
        Deprecated alias of ``x``, kept for one release.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.

    Returns
    -------
    array
        Probability mass values.
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
            "'loc' is deprecated, use 'mu' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = loc
    if x is None:
        raise TypeError("dnbinom() missing required argument: 'x'")
    if prob is not None and mu is not None:
        raise ValueError("'prob' and 'mu' both specified")
    if prob is not None:
        mu = size * (1 - np.asarray(prob)) / prob
    mu = 1 if mu is None else mu
    x, mu, size = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, mu, size))
    )

    # As R's dnbinom(mu=), in the mu form. Above size=15, lgamma(x+size)-lgamma(size)
    # loses its digits: it is taken with Stirling's series of R's stirlerr()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        stirling = [
            (
                1 / 12
                - (1 / 360 - (1 / 1260 - (1 / 1680 - 1 / 1188 / z**2) / z**2) / z**2)
                / z**2
            )
            / z
            for z in (x + size, size)
        ]
        log_density = np.where(
            size > 15,
            x * np.log(mu)
            - special.gammaln(x + 1)
            + (x + size - 0.5) * np.log1p(x / size)
            - x
            - (x + size) * np.log1p(mu / size)
            + stirling[0]
            - stirling[1],
            special.gammaln(x + size)
            - special.gammaln(size)
            - special.gammaln(x + 1)
            + size * np.log(size / (size + mu))
            + x * np.log(mu / (size + mu)),
        )
        log_density = np.where(
            x == 0,
            size
            * np.where(
                size < mu, np.log(size / (size + mu)), np.log1p(-mu / (size + mu))
            ),
            log_density,
        )
        log_density = np.where(
            np.isinf(size), stats.poisson.logpmf(x, mu=mu), log_density
        )
        log_density = np.where(size == 0, np.where(x == 0, 0.0, -np.inf), log_density)
        log_density = np.where(
            (x < 0) | np.isinf(x) | (x != np.floor(x)), -np.inf, log_density
        )
    log_density = np.where(
        np.isnan(x) | np.isnan(mu) | np.isnan(size) | (mu < 0) | (size < 0),
        np.nan,
        log_density,
    )
    return (log_density if log else np.exp(log_density))[()]


def pnbinom(q, mu=None, size=1, lower_tail=True, log_p=False, *, prob=None, loc=None):
    """Negative Binomial distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float, optional
        Mean parameter. The default, without ``prob``, is 1.
    size : float
        Dispersion parameter (number of successes).
    lower_tail : bool
        If True, return lower tail probability.
    log_p : bool
        If True, return log-CDF.
    prob : float, optional
        Probability of success in each trial, instead of ``mu``.
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
    if prob is not None and mu is not None:
        raise ValueError("'prob' and 'mu' both specified")
    mu = 1 if mu is None and prob is None else mu
    prob = size / (size + mu) if prob is None else prob
    family, args = stats.nbinom, (q, size, prob)
    tails = (family.cdf(*args), family.sf(*args))
    tail, other = tails if lower_tail else tails[::-1]
    if not log_p:
        return tail
    # As R, the log of a tail close to one from the other tail
    with np.errstate(divide="ignore"):
        return np.where(tail < 0.5, np.log(tail), np.log1p(-other))[()]


def qnbinom(p, mu=None, size=1, lower_tail=True, log_p=False, *, prob=None, loc=None):
    """Negative Binomial distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    mu : float, optional
        Mean parameter. The default, without ``prob``, is 1.
    size : float
        Dispersion parameter (number of successes).
    lower_tail : bool
        If True, probabilities are P[X <= x], otherwise P[X > x].
    log_p : bool
        If True, probabilities are given as log(p).
    prob : float, optional
        Probability of success in each trial, instead of ``mu``.
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
    if prob is not None and mu is not None:
        raise ValueError("'prob' and 'mu' both specified")
    mu = 1 if mu is None and prob is None else mu
    prob = size / (size + mu) if prob is None else prob
    p = np.exp(p) if log_p else np.asarray(p)
    method = stats.nbinom.ppf if lower_tail else stats.nbinom.isf
    # As R, the bound of p at the lower end of the support gives that end,
    # where scipy gives one below it
    return np.where(p == (0 if lower_tail else 1), 1, 0) + method(p, n=size, p=prob)


def rnbinom(n, mu=None, size=1, *, prob=None, loc=None):
    """Negative Binomial distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    mu : float, optional
        Mean parameter. The default, without ``prob``, is 1.
    size : float
        Dispersion parameter (number of successes).
    prob : float, optional
        Probability of success in each trial, instead of ``mu``.
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
    if prob is not None and mu is not None:
        raise ValueError("'prob' and 'mu' both specified")
    mu = 1 if mu is None and prob is None else mu
    prob = size / (size + mu) if prob is None else prob
    return stats.nbinom.rvs(n=size, p=prob, size=n)
