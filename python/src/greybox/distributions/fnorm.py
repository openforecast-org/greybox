"""Folded Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the folded normal distribution.
"""

import numpy as np
from scipy import optimize, stats


def dfnorm(q, loc=0, scale=1, log=False):
    """Folded Normal distribution density.

    f(x) = 1/sqrt(2*pi*scale^2) * (
        exp(-(x-loc)^2 / (2*scale^2)) + exp(-(x+loc)^2 / (2*scale^2)))

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
    q = np.asarray(q)
    abs_q = np.abs(q)
    if log:
        # log(f(x)) = log(exp(a) + exp(b)) - log(sqrt(2*pi)*scale)
        # where a = -(|x|-loc)^2/(2*scale^2), b = -(|x|+loc)^2/(2*scale^2)
        a = -((abs_q - loc) ** 2) / (2 * scale**2)
        b = -((abs_q + loc) ** 2) / (2 * scale**2)
        log_sum = np.logaddexp(a, b)
        log_density = log_sum - np.log(np.sqrt(2 * np.pi) * scale)
        return np.where(q < 0, -np.inf, log_density)
    density = (
        1
        / (np.sqrt(2 * np.pi) * scale)
        * (
            np.exp(-((abs_q - loc) ** 2) / (2 * scale**2))
            + np.exp(-((abs_q + loc) ** 2) / (2 * scale**2))
        )
    )
    return np.where(q < 0, 0.0, density)


def pfnorm(q, loc=0, scale=1):
    """Folded Normal distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        CDF values.
    """
    q = np.asarray(q)
    result = (
        stats.norm.cdf(q, loc=loc, scale=scale)
        + stats.norm.cdf(q, loc=-loc, scale=scale)
        - 1
    )
    result = np.where(q < 0, 0, result)
    return result


def qfnorm(p, loc=0, scale=1):
    """Folded Normal distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Quantile values.
    """
    p = np.atleast_1d(p)
    result = np.zeros_like(p, dtype=float)

    for i, pi in enumerate(p):
        if pi == 0:
            result[i] = 0
        elif pi == 1:
            result[i] = np.inf
        else:

            def objective(x, pi=pi):
                return pfnorm(x, loc, scale) - pi

            try:
                result[i] = optimize.brentq(objective, 0, loc + 10 * scale)
            except ValueError:
                result[i] = abs(stats.norm.ppf(pi, loc=loc, scale=scale))

    return result.squeeze()


def rfnorm(n, loc=0, scale=1):
    """Folded Normal distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Random values.
    """
    return np.abs(stats.norm.rvs(loc=loc, scale=scale, size=n))
