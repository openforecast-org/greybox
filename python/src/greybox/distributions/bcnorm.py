"""Box-Cox Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the distribution that becomes normal after the Box-Cox
transformation.

References
----------
.. [1] Box, G. E. P., & Cox, D. R. (1964).  An analysis of transformations.
       *Journal of the Royal Statistical Society. Series B (Methodological)*,
       26(2), 211-252.  https://www.jstor.org/stable/2984418
.. [2] Granger, C. W. J., & Newbold, P. (1976).  Forecasting transformed
       series.  *Journal of the Royal Statistical Society. Series B
       (Methodological)*, 38(2), 189-203.
       doi:10.1111/j.2517-6161.1976.tb01585.x
.. [3] Pankratz, A., & Dudley, U. (1987).  Forecasts of power-transformed
       series.  *Journal of Forecasting*, 6(4), 239-248.
       doi:10.1002/for.3980060403
.. [4] Guerrero, V. M. (1993).  Time-series analysis supported by power
       transformations.  *Journal of Forecasting*, 12(1), 37-48.
       doi:10.1002/for.3980120104
"""

import warnings

import numpy as np

from .helper import dnorm, pnorm, qnorm
from .lnorm import dlnorm, plnorm, qlnorm


def dbcnorm(q, mu=0, sigma=1, lambda_bc=0, log=False, *, loc=None, scale=None):
    """Box-Cox Normal distribution density.

    f(y) = y^(lambda-1) * 1/sqrt(2*pi) * exp(
        -((y^lambda-1)/lambda - mu)^2 / (2*sigma^2))

    Parameters
    ----------
    q : array_like
        Quantiles (must be non-negative).
    mu : float
        Location parameter (on transformed scale).
    sigma : float
        Scale parameter.
    lambda_bc : float
        Box-Cox transformation parameter.
    log : bool
        If True, return log-density.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sigma``, kept for one release.

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
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sigma' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sigma = scale
    q = np.asarray(q, dtype=float)
    if lambda_bc == 0:
        return dlnorm(q, mu, sigma, log=log)
    if lambda_bc == 1:
        return dnorm(q, np.asarray(mu) + 1, sigma, log=log)
    # The log analytically, as the log of the density underflows in the tails
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y_transformed = (np.power(q, lambda_bc) - 1) / lambda_bc
        if log:
            density = (lambda_bc - 1) * np.log(q) + dnorm(
                y_transformed, mu, sigma, log=True
            )
        else:
            density = np.power(q, lambda_bc - 1) * dnorm(y_transformed, mu, sigma)
    return np.where(q <= 0, -np.inf if log else 0.0, density)[()]


def pbcnorm(q, mu=0, sigma=1, lambda_bc=0, *, loc=None, scale=None):
    """Box-Cox Normal distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    lambda_bc : float
        Box-Cox transformation parameter.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sigma``, kept for one release.

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
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sigma' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sigma = scale
    q = np.asarray(q, dtype=float)
    if lambda_bc == 0:
        return plnorm(q, mu, sigma)
    # As R, q=0 maps onto -1/lambda
    with np.errstate(divide="ignore", invalid="ignore"):
        result = pnorm((np.power(q, lambda_bc) - 1) / lambda_bc, mu, sigma)
    return np.where(q < 0, 0.0, result)[()]


def qbcnorm(p, mu=0, sigma=1, lambda_bc=0, *, loc=None, scale=None):
    """Box-Cox Normal distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    lambda_bc : float
        Box-Cox transformation parameter.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sigma``, kept for one release.

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
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sigma' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sigma = scale
    p = np.asarray(p, dtype=float)
    if lambda_bc == 0:
        return qlnorm(p, mu, sigma)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if lambda_bc > 0:
            # BC inverse support is x > -1/lambda (lower-bounded); the
            # underlying Normal can place mass below that boundary which
            # corresponds to (formally) negative y. That mass is mapped to
            # y=0 (the natural lower BC support edge), as is a NaN.
            base = qnorm(p, mu, sigma) * lambda_bc + 1
            result = np.power(base, 1 / lambda_bc)
            result = np.where(np.isnan(result) | (base < 0), 0, result)
        else:
            # lambda_bc < 0: BC inverse support is x < -1/lambda
            # (upper-bounded).  The underlying Normal puts non-zero mass
            # *above* -1/lambda which would correspond to y = +infinity
            # (the mass at the upper support boundary).  The unrenormalized
            # BC CDF therefore saturates at
            #   P_valid = pnorm(-1/lambda, mu, sigma) < 1
            # and the naive quantile is NaN for any p > P_valid.
            #
            # Renormalize to the truncated distribution on (0, +infinity)
            # by conditioning on the underlying Normal falling within the
            # valid BC inverse support, i.e. invert
            #   F_trunc(y) = pbcnorm(y, ...) / P_valid
            # which IS a proper CDF.  The renormalized quantile is finite
            # for every p in (0, 1) and is +infinity at p = 1.  See Granger &
            # Newbold (1976), Pankratz & Dudley (1987), Guerrero (1993) for
            # the standard truncated-distribution treatment of BC at
            # lambda < 0.
            p_valid = pnorm(-1 / lambda_bc, mu, sigma)
            result = np.power(
                qnorm(p * p_valid, mu, sigma) * lambda_bc + 1, 1 / lambda_bc
            )
            result = np.where(p == 1, np.inf, result)
    return result[()]


def rbcnorm(n=1, mu=0, sigma=1, lambda_bc=0, *, loc=None, scale=None):
    """Box-Cox Normal distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    lambda_bc : float
        Box-Cox transformation parameter.
    loc : float, optional
        Deprecated alias of ``mu``, kept for one release.
    scale : float, optional
        Deprecated alias of ``sigma``, kept for one release.

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
    if scale is not None:
        warnings.warn(
            "'scale' is deprecated, use 'sigma' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sigma = scale
    return qbcnorm(np.random.uniform(0, 1, n), mu, sigma, lambda_bc)
