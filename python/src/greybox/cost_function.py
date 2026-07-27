"""Cost function for optimization.

This module contains the cost function (cf) that combines the fitter with
the loss function (likelihood, MSE, MAE, etc.).
"""

import numpy as np

from . import distributions as dist
from .transforms import mean_fast


def _compute_log_lik_array(
    distribution: str,
    y_otU: np.ndarray,
    mu_otU: np.ndarray,
    scale: float,
    other_val: float,
    occurrence_model: bool,
    size: float,
    lambda_bc: float,
) -> np.ndarray:
    """Compute log-likelihood array for a distribution.

    Parameters
    ----------
    distribution : str
        Distribution name.
    y_otU : np.ndarray
        Observed values (non-zero subset).
    mu_otU : np.ndarray
        Location parameter values.
    scale : float
        Scale parameter.
    other_val : float
        Other distribution parameter.
    occurrence_model : bool
        Whether this is an occurrence model.
    size : float
        Size parameter for binomial.
    lambda_bc : float
        Box-Cox parameter.

    Returns
    -------
    np.ndarray
        Log-likelihood values (array form, not summed).
    """
    if distribution == "dnorm":
        return dist.dnorm(y_otU, loc=mu_otU, scale=scale, log=True)
    elif distribution == "dlaplace":
        return dist.dlaplace(y_otU, loc=mu_otU, scale=scale, log=True)
    elif distribution == "ds":
        return dist.ds(y_otU, loc=mu_otU, scale=scale, log=True)
    elif distribution == "dgnorm":
        return dist.dgnorm(y_otU, loc=mu_otU, scale=scale, shape=other_val, log=True)
    elif distribution == "dlogis":
        return dist.dlogis(y_otU, loc=mu_otU, scale=scale, log=True)
    elif distribution == "dt":
        return dist.dt(y_otU - mu_otU, df=scale, loc=0, scale=1, log=True)
    elif distribution == "dalaplace":
        return dist.dalaplace(y_otU, loc=mu_otU, scale=scale, alpha=other_val, log=True)
    elif distribution == "dlnorm":
        return dist.dlnorm(y_otU, loc=mu_otU, scale=scale, log=True)
    elif distribution == "dllaplace":
        return dist.dllaplace(y_otU, loc=mu_otU, scale=scale, log=True)
    elif distribution == "dls":
        return dist.dls(y_otU, loc=mu_otU, scale=scale, log=True)
    elif distribution == "dlgnorm":
        return dist.dlgnorm(y_otU, loc=mu_otU, scale=scale, shape=other_val, log=True)
    elif distribution == "dbcnorm":
        mask_nonzero = y_otU != 0
        result = np.zeros_like(y_otU, dtype=float)
        result[mask_nonzero] = dist.dbcnorm(
            y_otU[mask_nonzero],
            loc=mu_otU[mask_nonzero],
            scale=scale,
            lambda_bc=lambda_bc,
            log=True,
        )
        return result
    elif distribution == "dfnorm":
        return dist.dfnorm(y_otU, loc=mu_otU, scale=scale, log=True)
    elif distribution == "drectnorm":
        return dist.drectnorm(y_otU, loc=mu_otU, scale=scale, log=True)
    elif distribution == "dinvgauss":
        disp = scale / (mu_otU + 1e-300)
        lam = 1.0 / (disp + 1e-300)
        return dist.dinvgauss(
            y_otU, loc=scale * np.ones_like(y_otU), scale=lam, log=True
        )
    elif distribution == "dgamma":
        return dist.dgamma(y_otU, shape=1 / scale, scale=scale * mu_otU, log=True)
    elif distribution == "dexp":
        return dist.dexp(y_otU, loc=0, scale=mu_otU, log=True)
    elif distribution == "dchisq":
        return dist.dchi2(y_otU, df=scale, log=True)
    elif distribution == "dgeom":
        return dist.dgeom(y_otU, prob=1 / (mu_otU + 1), log=True)
    elif distribution == "dpois":
        return dist.dpois(y_otU, loc=mu_otU, log=True)
    elif distribution == "dnbinom":
        return dist.dnbinom(y_otU, loc=mu_otU, size=scale, log=True)
    elif distribution == "dbinom":
        return dist.dbinom(
            y_otU.astype(int) - occurrence_model * 1,
            size=int(size),
            prob=1 / (mu_otU + 1),
            log=True,
        )
    elif distribution == "dlogitnorm":
        return dist.dlogitnorm(y_otU, loc=mu_otU, scale=scale, log=True)
    elif distribution == "dbeta":
        return dist.dbeta(y_otU, a=mu_otU, b=scale, log=True)
    elif distribution == "pnorm":
        ot = y_otU != 0
        result = np.zeros_like(y_otU, dtype=float)
        result[ot] = dist.pnorm(mu_otU[ot], loc=0, scale=1, log=True)
        result[~ot] = dist.pnorm(
            mu_otU[~ot], loc=0, scale=1, log=True, lower_tail=False
        )
        return result
    elif distribution == "plogis":
        ot = y_otU != 0
        result = np.zeros_like(y_otU, dtype=float)
        result[ot] = dist.plogis(mu_otU[ot], loc=0, scale=1, log=True)
        result[~ot] = dist.plogis(
            mu_otU[~ot], loc=0, scale=1, log=True, lower_tail=False
        )
        return result
    else:
        return np.zeros_like(y_otU, dtype=float)


def _entropy_adjustment(
    distribution: str,
    scale: float,
    other_val: float,
    mu: np.ndarray,
    otU: np.ndarray,
    obs_zero: int,
) -> float:
    """Calculate differential entropy for occurrence model.

    Returns the entropy adjustment term for distributions when
    recursive_model=True and occurrence_model=True.

    Parameters
    ----------
    distribution : str
        Distribution name.
    scale : float
        Scale parameter.
    other_val : float
        Other distribution parameter (shape, alpha, etc.).
    mu : np.ndarray
        Location parameter values.
    otU : np.ndarray
        Boolean mask for non-zero observations.
    obs_zero : int
        Number of zero observations.

    Returns
    -------
    float
        Entropy adjustment value.
    """
    from scipy.special import betaln, digamma, gammaln

    mu_otU = mu[otU]

    if distribution in ("dnorm", "dfnorm", "dbcnorm", "dlogitnorm"):
        return obs_zero * (np.log(np.sqrt(2 * np.pi) * scale) + 0.5)
    elif distribution == "dlnorm":
        return obs_zero * (np.log(np.sqrt(2 * np.pi) * scale) + 0.5) + np.sum(mu[~otU])
    elif distribution in ("dgnorm", "dlgnorm"):
        return obs_zero * (
            1 / other_val - np.log(other_val / (2 * scale * gammaln(1 / other_val)))
        )
    elif distribution == "dinvgauss":
        return 0.5 * (
            obs_zero * (np.log(np.pi / 2) + 1 + np.log(scale))
            - np.sum(np.log(mu[~otU]))
        )
    elif distribution == "dgamma":
        return np.sum(
            gammaln(1 / scale)
            + (scale * mu_otU)
            + (1 - scale * mu_otU) * digamma(scale * mu_otU)
            + scale * mu_otU
        )
    elif distribution == "dexp":
        return obs_zero
    elif distribution == "dls":
        return obs_zero * (2 + 2 * np.log(2 * scale))
    elif distribution == "dalaplace":
        return obs_zero * (1 + np.log(2 * scale))
    elif distribution == "dlogis":
        return obs_zero * 2
    elif distribution == "dt":
        return obs_zero * (
            (scale + 1) / 2 * (digamma((scale + 1) / 2) - digamma(scale / 2))
            + np.log(np.sqrt(scale) * np.exp(betaln(scale / 2, 0.5)))
        )
    elif distribution == "dchisq":
        return obs_zero * (
            np.log(2) * gammaln(scale / 2)
            - (1 - scale / 2) * digamma(scale / 2)
            + scale / 2
        )
    elif distribution == "dbeta":
        return (
            np.sum(
                np.log(scale)
                + gammaln(mu_otU)
                + gammaln(scale)
                - gammaln(mu_otU + scale)
            )
            - (mu_otU - 1) * digamma(mu_otU)
            - (scale - 1) * digamma(mu_otU + scale)
            + (mu_otU + scale - 2) * digamma(mu_otU + scale)
        )
    else:
        return 0.0


def cf(
    B: np.ndarray,
    distribution: str,
    loss: str,
    y: np.ndarray,
    matrix_xreg: np.ndarray,
    recursive_model: bool = False,
    denominator: float = 1.0,
    otU: np.ndarray | None = None,
    obs_insample: int | None = None,
    obs_zero: int = 0,
    obs_nonzero: int | None = None,
    occurrence_model: bool = False,
    trim: float = 0.0,
    lambda_val: float = 0.0,
    other=None,
    a_parameter_provided: bool = False,
    ar_order: int = 0,
    i_order: int = 0,
    loss_function=None,
    lambda_bc: float = 0.0,
    size: float = 1.0,
    fitter_func=None,
) -> float:
    """Cost function for optimization.

    This is the main objective function that combines the fitter with
    the loss function (likelihood, MSE, MAE, etc.).

    Parameters
    ----------
    B : np.ndarray
        Parameter vector.
    distribution : str
        Distribution name.
    loss : str
        Loss function type.
    y : np.ndarray
        Observed values.
    matrix_xreg : np.ndarray
        Design matrix.
    recursive_model : bool, default=False
        Whether this is a recursive (ARIMA) model.
    denominator : float, default=1.0
        Denominator for RIDGE scaling.
    otU : np.ndarray, optional
        Boolean mask for non-zero observations.
    obs_insample : int, optional
        Number of in-sample observations.
    obs_zero : int, default=0
        Number of zero observations.
    obs_nonzero : int, optional
        Number of non-zero observations.
    occurrence_model : bool, default=False
        Whether there's an occurrence model.
    trim : float, default=0.0
        Trim proportion for ROLE loss.
    lambda_val : float, default=0.0
        LASSO/Ridge parameter.
    other : various, optional
        Additional distribution parameter.
    a_parameter_provided : bool, default=False
        Whether additional parameter was provided.
    ar_order : int, default=0
        AR order.
    i_order : int, default=0
        Integration order.
    loss_function : callable, optional
        Custom loss function.
    lambda_bc : float, default=0.0
        Box-Cox parameter.
    size : float, default=1.0
        Size parameter for binomial.
    fitter_func : callable, optional
        Fitter function to call.

    Returns
    -------
    float
        Cost function value (negative log-likelihood or loss).
    """
    from .fitters import extractor_fitted, fitter

    if otU is None:
        otU = np.ones(len(y), dtype=bool)
    if obs_nonzero is None:
        obs_nonzero = np.sum(otU)
    if obs_insample is None:
        obs_insample = len(y)

    y_otU = y[otU]

    fitter_return = fitter(
        B,
        distribution,
        y,
        matrix_xreg,
        other=other,
        a_parameter_provided=a_parameter_provided,
        ar_order=ar_order,
        i_order=i_order,
        loss=loss,
        lambda_val=lambda_val,
    )

    mu = fitter_return["mu"]
    mu_otU = mu[otU]
    scale = fitter_return["scale"]
    other_val = fitter_return["other"]

    if loss in ("likelihood", "ROLE"):
        # For dbcnorm, lambda_bc is the "other" param when estimated
        effective_lambda_bc = (
            other_val
            if distribution == "dbcnorm" and not a_parameter_provided
            else lambda_bc
        )
        log_lik_array = _compute_log_lik_array(
            distribution,
            y_otU,
            mu_otU,
            scale,
            other_val,
            occurrence_model,
            size,
            effective_lambda_bc,
        )

        if loss == "likelihood":
            cf_value = -np.sum(log_lik_array)
        else:
            cf_value = (
                -mean_fast(-log_lik_array, obs_insample, trim=trim, side="both")
                * obs_insample
            )

        if recursive_model and occurrence_model:
            cf_value += _entropy_adjustment(
                distribution, scale, other_val, mu, otU, obs_zero
            )

    else:
        effective_lambda_bc = (
            other_val
            if distribution == "dbcnorm" and not a_parameter_provided
            else lambda_bc
        )
        y_fitted = extractor_fitted(distribution, mu, scale, effective_lambda_bc)

        if loss == "MSE":
            cf_value = np.mean((y - y_fitted) ** 2)
        elif loss == "MAE":
            cf_value = np.mean(np.abs(y - y_fitted))
        elif loss == "HAM":
            cf_value = np.mean(np.sqrt(np.abs(y - y_fitted)))
        elif loss == "LASSO":
            cf_value = (1 - lambda_val) * np.mean(
                (y - y_fitted) ** 2
            ) + lambda_val * np.sum(np.abs(B))
        elif loss == "RIDGE":
            B_scaled = B * denominator
            if len(B_scaled) > 1:
                cf_value = (1 - lambda_val) * np.mean(
                    (y - y_fitted) ** 2
                ) + lambda_val * np.sqrt(np.sum(B_scaled[1:] ** 2))
            else:
                cf_value = (1 - lambda_val) * np.mean(
                    (y - y_fitted) ** 2
                ) + lambda_val * np.sqrt(np.sum(B_scaled**2))
            if lambda_val == 1.0:
                cf_value = np.mean((y - y_fitted) ** 2)
        elif loss == "custom" and loss_function is not None:
            cf_value = loss_function(y, y_fitted, B, matrix_xreg)
        else:
            cf_value = np.mean((y - y_fitted) ** 2)

    if np.isnan(cf_value) or np.isnan(cf_value) or np.isinf(cf_value):
        cf_value = 1e300

    return cf_value
