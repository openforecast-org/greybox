"""Fitter functions for ALM models.

This module contains the core fitting functions that correspond to the R functions
in alm.R: scalerInternal, extractorFitted, extractorResiduals, fitter,
fitterRecursive, and the cost function (cf).
"""

import numpy as np

from . import distributions as dist
from .cost_function import cf  # noqa: F401 - re-exported for backwards compatibility
from .transforms import bc_transform, bc_transform_inv, mean_fast


def _plogis_log_residual(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Numerically stable log-residual for plogis distribution.

    For binary y in {0, 1}:
      y=1: log((2 + exp(mu)) / exp(mu)) = log1p(2*exp(-mu))
      y=0: log(1 / (1 + 2*exp(mu)))     = -log1p(2*exp(mu))

    With overflow protection for |mu| > 500.
    """
    result = np.empty_like(mu, dtype=float)
    mask1 = y == 1
    mask0 = ~mask1

    mu1 = mu[mask1]
    large_pos = mu1 > 500
    result_1 = np.empty_like(mu1)
    result_1[~large_pos] = np.log1p(2.0 * np.exp(-mu1[~large_pos]))
    result_1[large_pos] = 2.0 * np.exp(-mu1[large_pos])
    result[mask1] = result_1

    mu0 = mu[mask0]
    large_pos0 = mu0 > 500
    result_0 = np.empty_like(mu0)
    result_0[~large_pos0] = -np.log1p(2.0 * np.exp(mu0[~large_pos0]))
    result_0[large_pos0] = -(np.log(2.0) + mu0[large_pos0])
    result[mask0] = result_0

    return result


# The distributions built on the Normal one, whose scale is the variance sigma^2
VARIANCE_SCALE_DISTRIBUTIONS = (
    "dnorm",
    "dlnorm",
    "dbcnorm",
    "dlogitnorm",
    "dfnorm",
    "drectnorm",
)


def scale_sd(distribution: str, scale):
    """Return the scale in the form that the densities of the model take.

    The stored scale of the Normal-based distributions (``dnorm``, ``dlnorm``,
    ``dbcnorm``, ``dlogitnorm``, ``dfnorm``, ``drectnorm``) is the variance
    sigma^2 (ADAM monograph, Tables 11.1-11.2), so its square root is returned
    for them. The scale of every other distribution is returned as is.
    """
    if distribution in VARIANCE_SCALE_DISTRIBUTIONS and scale is not None:
        return np.sqrt(scale)
    return scale


def scaler_internal(
    B: np.ndarray,
    distribution: str,
    y: np.ndarray,
    matrix_xreg: np.ndarray,
    mu: np.ndarray,
    other,
    otU: np.ndarray,
    df: int,
    trim: float = 0.0,
    side: str = "both",
) -> float:
    """Calculate scale parameter based on distribution.

    Parameters
    ----------
    B : np.ndarray
        Parameter vector.
    distribution : str
        Distribution name.
    y : np.ndarray
        Observed values.
    matrix_xreg : np.ndarray
        Design matrix.
    mu : np.ndarray
        Location parameter values.
    other : various
        Additional distribution parameter (shape, alpha, etc.).
    otU : np.ndarray
        Boolean mask for non-zero observations.
    df : int
        Degrees of freedom.
    trim : float, default=0.0
        Trim proportion for mean_fast.
    side : str, default="both"
        Trim side.

    Returns
    -------
    float
        Scale parameter value.
    """
    y_otU = y[otU]
    mu_otU = mu[otU]

    if distribution == "dbeta":
        return np.exp(matrix_xreg @ B[-len(B) // 2 :])

    elif distribution == "dnorm":
        return np.sqrt(mean_fast((y_otU - mu_otU) ** 2, df, trim, side))

    elif distribution == "dlaplace":
        return mean_fast(np.abs(y_otU - mu_otU), df, trim, side)

    elif distribution == "ds":
        return mean_fast(np.sqrt(np.abs(y_otU - mu_otU)), df * 2, trim, side)

    elif distribution == "dgnorm":
        return (other * mean_fast(np.abs(y_otU - mu_otU) ** other, df, trim, side)) ** (
            1 / other
        )

    elif distribution == "dlogis":
        return (
            np.sqrt(mean_fast((y_otU - mu_otU) ** 2, df, trim, side))
            * np.sqrt(3)
            / np.pi
        )

    elif distribution == "dalaplace":
        indicator = (y_otU <= mu_otU).astype(float)
        return mean_fast((y_otU - mu_otU) * (other - indicator), df, trim)

    elif distribution == "dlnorm":
        return np.sqrt(mean_fast((np.log(y_otU) - mu_otU) ** 2, df, trim, side))

    elif distribution == "dllaplace":
        return mean_fast(np.abs(np.log(y_otU) - mu_otU), df, trim, side)

    elif distribution == "dls":
        return mean_fast(np.sqrt(np.abs(np.log(y_otU) - mu_otU)), df * 2, trim, side)

    elif distribution == "dlgnorm":
        return (
            other * mean_fast(np.abs(np.log(y_otU) - mu_otU) ** other, df, trim, side)
        ) ** (1 / other)

    elif distribution == "dbcnorm":
        y_transformed = bc_transform(y_otU, other)
        return np.sqrt(mean_fast((y_transformed - mu_otU) ** 2, df, trim, side))

    elif distribution == "dinvgauss":
        return mean_fast(((y_otU / mu_otU - 1) ** 2) / (y_otU / mu_otU), df, trim, side)

    elif distribution == "dgamma":
        return mean_fast((y_otU / mu_otU - 1) ** 2, df, trim, side)

    elif distribution == "dlogitnorm":
        return np.sqrt(
            mean_fast((np.log(y_otU / (1 - y_otU)) - mu_otU) ** 2, df, trim, side)
        )

    elif (
        distribution in ("dfnorm", "drectnorm", "dt", "dchisq")
        or distribution == "dnbinom"
    ):
        return np.abs(other) if other is not None else 1.0

    elif distribution == "dbinom":
        return other if other is not None else 1.0

    elif distribution == "dpois":
        return mu_otU

    elif distribution == "pnorm":
        p = (y - dist.pnorm(mu, 0, 1) + 1) / 2
        q = dist.qnorm(p, 0, 1)
        return np.sqrt(mean_fast(q**2))

    elif distribution == "plogis":
        log_term = _plogis_log_residual(y, mu)
        return np.sqrt(mean_fast(log_term**2))

    else:
        return 1.0


def extractor_fitted(
    distribution: str, mu: np.ndarray, scale: float | np.ndarray, lambda_bc: float = 0.0
) -> np.ndarray:
    """Extract fitted values in original scale.

    Parameters
    ----------
    distribution : str
        Distribution name.
    mu : np.ndarray
        Location parameter (on transformed scale).
    scale : float or np.ndarray
        Scale parameter.
    lambda_bc : float, default=0.0
        Box-Cox parameter.

    Returns
    -------
    np.ndarray
        Fitted values in original scale.
    """
    if distribution == "dfnorm":
        return np.sqrt(2 / np.pi) * scale * np.exp(-(mu**2) / (2 * scale**2)) + mu * (
            1 - 2 * dist.pnorm(-mu / scale, loc=0, scale=1)
        )

    elif distribution == "drectnorm":
        return mu * (1 - dist.pnorm(0, loc=mu, scale=scale)) + scale * dist.dnorm(
            0, loc=mu, scale=scale
        )

    elif distribution in (
        "dnorm",
        "dgnorm",
        "dinvgauss",
        "dgamma",
        "dexp",
        "dlaplace",
        "dalaplace",
        "dlogis",
        "dt",
        "ds",
        "dgeom",
        "dpois",
        "dnbinom",
    ):
        return mu

    elif distribution == "dbinom":
        return 1 / (1 + mu) * scale

    elif distribution == "dchisq":
        return mu + scale

    elif distribution in ("dlnorm", "dllaplace", "dls", "dlgnorm"):
        return np.exp(mu)

    elif distribution == "dlogitnorm":
        return np.exp(mu) / (1 + np.exp(mu))

    elif distribution == "dbcnorm":
        return bc_transform_inv(mu, lambda_bc)

    elif distribution == "dbeta":
        return mu / (mu + scale)

    elif distribution == "pnorm":
        return dist.pnorm(mu, loc=0, scale=1)

    elif distribution == "plogis":
        return dist.plogis(mu, loc=0, scale=1)

    else:
        return mu


def extractor_residuals(
    distribution: str, mu: np.ndarray, y: np.ndarray, lambda_bc: float = 0.0
) -> np.ndarray:
    """Extract residuals in transformed scale.

    Parameters
    ----------
    distribution : str
        Distribution name.
    mu : np.ndarray
        Location parameter.
    y : np.ndarray
        Observed values.
    lambda_bc : float, default=0.0
        Box-Cox parameter.

    Returns
    -------
    np.ndarray
        Residuals in transformed scale.
    """
    if distribution in ("dbinom", "dbeta"):
        return y - extractor_fitted(distribution, mu, 1.0)

    elif distribution in (
        "dfnorm",
        "drectnorm",
        "dnorm",
        "dlaplace",
        "ds",
        "dgnorm",
        "dalaplace",
        "dlogis",
        "dt",
        "dgeom",
        "dnbinom",
        "dpois",
        "dinvgauss",
        "dgamma",
    ):
        return y - mu

    elif distribution == "dexp":
        return y / mu

    elif distribution == "dchisq":
        return np.sqrt(y) - np.sqrt(mu)

    elif distribution in ("dlnorm", "dllaplace", "dls", "dlgnorm"):
        return np.log(y) - mu

    elif distribution == "dbcnorm":
        return bc_transform(y, lambda_bc) - mu

    elif distribution == "dlogitnorm":
        return np.log(y / (1 - y)) - mu

    elif distribution == "pnorm":
        p = (y - dist.pnorm(mu, 0, 1) + 1) / 2
        return dist.qnorm(p, 0, 1)

    elif distribution == "plogis":
        return _plogis_log_residual(y, mu)

    else:
        return y - mu


def fitter(
    B: np.ndarray,
    distribution: str,
    y: np.ndarray,
    matrix_xreg: np.ndarray,
    other=None,
    ar_order: int = 0,
    i_order: int = 0,
    poly1=None,
    poly2=None,
    n_variables: int = 0,
    loss: str = "likelihood",
    lambda_val: float = 0.0,
    a_parameter_provided: bool = False,
) -> dict:
    """Basic fitter for non-dynamic models.

    Parameters
    ----------
    B : np.ndarray
        Parameter vector.
    distribution : str
        Distribution name.
    y : np.ndarray
        Observed values.
    matrix_xreg : np.ndarray
        Design matrix.
    other : various, optional
        Additional distribution parameter.
    ar_order : int, default=0
        AR order.
    i_order : int, default=0
        Integration order.
    poly1 : np.ndarray, optional
        AR polynomial.
    poly2 : np.ndarray, optional
        MA polynomial.
    n_variables : int, default=0
        Number of variables.
    loss : str, default="likelihood"
        Loss function type.
    lambda_val : float, default=0.0
        LASSO/Ridge parameter.
    a_parameter_provided : bool, default=False
        Whether additional parameter was provided.

    Returns
    -------
    dict
        Dictionary with mu, scale, other, poly1 keys.
    """
    B = B.copy()

    if distribution == "dalaplace":
        if not a_parameter_provided:
            other = B[0]
            B = B[1:]
        else:
            other = other if other is not None else 0.5

    elif (
        distribution == "dnbinom"
        or distribution == "dchisq"
        or distribution in ("dfnorm", "drectnorm")
        or distribution in ("dgnorm", "dlgnorm")
        or distribution == "dbcnorm"
        or distribution == "dt"
    ):
        if not a_parameter_provided:
            other = B[0]
            B = B[1:]

    else:
        other = None

    if ar_order > 0 or i_order > 0:
        # Initialize poly2 (I polynomial: (1-B)^d) when not provided externally
        if poly2 is None:
            if i_order > 0:
                poly2 = np.array([1.0, -1.0])
                for _ in range(i_order - 1):
                    poly2 = np.convolve(poly2, np.array([1.0, -1.0]))
            else:
                poly2 = np.array([1.0])  # identity (no differencing)
        # Initialize poly1 (AR polynomial) for ARI combined case (p>0, d>0)
        if poly1 is None and ar_order > 0 and i_order > 0:
            poly1 = np.ones(ar_order + 1)
        # Compute n_variables from B length when not provided externally
        if n_variables == 0:
            n_variables = len(B) - ar_order

        if poly1 is not None and ar_order > 0:
            poly1[1:] = -B[-ar_order:]
            if n_variables > ar_order:
                B = np.concatenate(
                    [B[: len(B) - ar_order], -np.convolve(poly2, poly1)[1:]]
                )
            else:
                B = -np.convolve(poly2, poly1)[1:]
        elif i_order > 0:
            B = np.concatenate([B, -poly2[1:]])

    if loss in ("LASSO", "RIDGE") and lambda_val == 1.0:
        B[1:] = 0

    mu = np.zeros_like(y, dtype=float)

    if distribution in (
        "dinvgauss",
        "dgamma",
        "dexp",
        "dpois",
        "dnbinom",
        "dbinom",
        "dgeom",
    ):
        mu = np.exp(matrix_xreg @ B)
    elif distribution == "dchisq":
        linear_pred = matrix_xreg @ B
        mu = np.where(linear_pred < 0, 1e100, linear_pred**2)
    elif distribution == "dbeta":
        half_len = len(B) // 2
        mu = np.exp(matrix_xreg @ B[:half_len])
    else:
        mu = matrix_xreg @ B

    otU = np.ones(len(y), dtype=bool)
    df = np.sum(otU)

    scale = scaler_internal(B, distribution, y, matrix_xreg, mu, other, otU, df)

    return {
        "mu": mu,
        "scale": scale,
        "other": other,
        "poly1": poly1 if poly1 is not None else np.array([1.0]),
    }


def fitter_recursive(
    B: np.ndarray,
    distribution: str,
    y: np.ndarray,
    matrix_xreg: np.ndarray,
    ari_order: int = 0,
    other=None,
    a_parameter_provided: bool = False,
    lambda_bc: float = 0.0,
    **kwargs,
) -> dict:
    """Fitter for dynamic (ARIMA) models.

    Parameters
    ----------
    B : np.ndarray
        Parameter vector.
    distribution : str
        Distribution name.
    y : np.ndarray
        Observed values.
    matrix_xreg : np.ndarray
        Design matrix (will be modified in place).
    ari_order : int, default=0
        ARIMA order (sum of p+d+q).
    other : various, optional
        Additional distribution parameter.
    a_parameter_provided : bool, default=False
        Whether additional parameter was provided.
    lambda_bc : float, default=0.0
        Box-Cox parameter.
    **kwargs
        Additional arguments passed to fitter.

    Returns
    -------
    dict
        Dictionary with mu, scale, other, matrix_xreg keys.
    """
    fitter_return = fitter(
        B,
        distribution,
        y,
        matrix_xreg,
        other=other,
        a_parameter_provided=a_parameter_provided,
        **kwargs,
    )

    return {
        "mu": fitter_return["mu"],
        "scale": fitter_return["scale"],
        "other": fitter_return["other"],
        "poly1": fitter_return["poly1"],
        "matrix_xreg": matrix_xreg,
    }
