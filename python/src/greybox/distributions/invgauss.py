"""Inverse Gaussian distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Inverse Gaussian distribution.

Parameterised by ``loc`` (the mean) and ``scale`` (the dispersion), matching
R's ``statmod::dinvgauss(mean=, dispersion=)``, which is what ADAM's
``dinvgauss`` error distribution is written against.

SciPy parameterises the same distribution differently: its standard form is
``IG(mean=mu, lambda=1)``, and a ``scale`` of ``s`` gives ``IG(mean=s*mu,
lambda=s)``. With ``lambda = 1 / dispersion`` that makes the mapping
``mu = mean * dispersion`` and SciPy's ``scale = 1 / dispersion``.
"""

from scipy import stats


def _scipy_args(loc, scale):
    """Map ``(mean, dispersion)`` onto SciPy's ``(mu, scale)``."""
    return {"mu": loc * scale, "scale": 1.0 / scale}


def dinvgauss(q, loc=1, scale=1, log=False):
    """Inverse Gaussian distribution density.

    Parameters
    ----------
    q : array_like
        Quantiles (must be positive).
    loc : float
        Mean of the distribution.
    scale : float
        Dispersion (the reciprocal of the shape ``lambda``).
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    if log:
        return stats.invgauss.logpdf(q, **_scipy_args(loc, scale))
    return stats.invgauss.pdf(q, **_scipy_args(loc, scale))


def pinvgauss(q, loc=1, scale=1):
    """Inverse Gaussian distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Mean of the distribution.
    scale : float
        Dispersion (the reciprocal of the shape ``lambda``).

    Returns
    -------
    array
        CDF values.
    """
    return stats.invgauss.cdf(q, **_scipy_args(loc, scale))


def qinvgauss(p, loc=1, scale=1):
    """Inverse Gaussian distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    loc : float
        Mean of the distribution.
    scale : float
        Dispersion (the reciprocal of the shape ``lambda``).

    Returns
    -------
    array
        Quantile values.
    """
    return stats.invgauss.ppf(p, **_scipy_args(loc, scale))


def rinvgauss(n, loc=1, scale=1):
    """Inverse Gaussian distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    loc : float
        Mean of the distribution.
    scale : float
        Dispersion (the reciprocal of the shape ``lambda``).

    Returns
    -------
    array
        Random values.
    """
    return stats.invgauss.rvs(size=n, **_scipy_args(loc, scale))
