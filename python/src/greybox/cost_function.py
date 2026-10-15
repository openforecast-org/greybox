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
        return dist.dnorm(y_otU, mean=mu_otU, sd=scale, log=True)
    elif distribution == "dlaplace":
        return dist.dlaplace(y_otU, mu=mu_otU, scale=scale, log=True)
    elif distribution == "ds":
        return dist.ds(y_otU, mu=mu_otU, scale=scale, log=True)
    elif distribution == "dgnorm":
        return dist.dgnorm(y_otU, mu=mu_otU, scale=scale, shape=other_val, log=True)
    elif distribution == "dlogis":
        return dist.dlogis(y_otU, location=mu_otU, scale=scale, log=True)
    elif distribution == "dt":
        return dist.dt(y_otU - mu_otU, df=scale, loc=0, scale=1, log=True)
    elif distribution == "dalaplace":
        return dist.dalaplace(y_otU, mu=mu_otU, scale=scale, alpha=other_val, log=True)
    elif distribution == "dlnorm":
        return dist.dlnorm(y_otU, meanlog=mu_otU, sdlog=scale, log=True)
    elif distribution == "dllaplace":
        return dist.dllaplace(y_otU, mu=mu_otU, scale=scale, log=True)
    elif distribution == "dls":
        return dist.dls(y_otU, mu=mu_otU, scale=scale, log=True)
    elif distribution == "dlgnorm":
        return dist.dlgnorm(y_otU, mu=mu_otU, scale=scale, shape=other_val, log=True)
    elif distribution == "dbcnorm":
        mask_nonzero = y_otU != 0
        result = np.zeros_like(y_otU, dtype=float)
        result[mask_nonzero] = dist.dbcnorm(
            y_otU[mask_nonzero],
            mu=mu_otU[mask_nonzero],
            sigma=scale,
            lambda_bc=lambda_bc,
            log=True,
        )
        return result
    elif distribution == "dfnorm":
        return dist.dfnorm(y_otU, mu=mu_otU, sigma=scale, log=True)
    elif distribution == "drectnorm":
        return dist.drectnorm(y_otU, mu=mu_otU, sigma=scale, log=True)
    elif distribution == "dinvgauss":
        # R: dinvgauss(y, mean=mu, dispersion=scale/mu); greybox's dinvgauss takes
        # the mean and the dispersion, as R's does
        return dist.dinvgauss(y_otU, mean=mu_otU, dispersion=scale / mu_otU, log=True)
    elif distribution == "dgamma":
        return dist.dgamma(y_otU, shape=1 / scale, scale=scale * mu_otU, log=True)
    elif distribution == "dexp":
        return dist.dexp(y_otU, rate=1 / mu_otU, log=True)
    elif distribution == "dchisq":
        # R: dchisq(y, df=scale, ncp=mu), the non-central chi-squared
        return dist.dchisq(y_otU, df=scale, ncp=mu_otU, log=True)
    elif distribution == "dgeom":
        return dist.dgeom(y_otU, prob=1 / (mu_otU + 1), log=True)
    elif distribution == "dpois":
        return dist.dpois(y_otU, lambda_=mu_otU, log=True)
    elif distribution == "dnbinom":
        return dist.dnbinom(y_otU, mu=mu_otU, size=scale, log=True)
    elif distribution == "dbinom":
        return dist.dbinom(
            y_otU.astype(int) - occurrence_model * 1,
            size=int(size),
            prob=1 / (mu_otU + 1),
            log=True,
        )
    elif distribution == "dlogitnorm":
        return dist.dlogitnorm(y_otU, mu=mu_otU, sigma=scale, log=True)
    elif distribution == "dbeta":
        return dist.dbeta(y_otU, shape1=mu_otU, shape2=scale, log=True)
    elif distribution == "pnorm":
        ot = y_otU != 0
        result = np.zeros_like(y_otU, dtype=float)
        result[ot] = dist.pnorm(mu_otU[ot], mean=0, sd=1, log_p=True)
        result[~ot] = dist.pnorm(
            mu_otU[~ot], mean=0, sd=1, log_p=True, lower_tail=False
        )
        return result
    elif distribution == "plogis":
        ot = y_otU != 0
        result = np.zeros_like(y_otU, dtype=float)
        result[ot] = dist.plogis(mu_otU[ot], location=0, scale=1, log_p=True)
        result[~ot] = dist.plogis(
            mu_otU[~ot], location=0, scale=1, log_p=True, lower_tail=False
        )
        return result
    else:
        return np.zeros_like(y_otU, dtype=float)


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
    latent: dict | None = None,
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
    latent : dict, optional
        The settings of the latent sizes of a dynamic model with an occurrence model
        (see ``fitter_recursive``): the sizes at the zeros go on with the recursion
        of the model, and the zeros have only the likelihood of the occurrence.

    Returns
    -------
    float
        Cost function value (negative log-likelihood or loss).
    """
    from .fitters import extractor_fitted, fitter, fitter_recursive

    if otU is None:
        otU = np.ones(len(y), dtype=bool)
    if obs_nonzero is None:
        obs_nonzero = np.sum(otU)
    if obs_insample is None:
        obs_insample = len(y)

    y_otU = y[otU]

    fitter_kwargs = {
        "other": other,
        "a_parameter_provided": a_parameter_provided,
        "ar_order": ar_order,
        "i_order": i_order,
        "loss": loss,
        "lambda_val": lambda_val,
        "otU": otU,
        "trim": trim,
    }
    if latent is not None:
        fitter_return = fitter_recursive(
            B, distribution, y, matrix_xreg, latent, **fitter_kwargs
        )
    else:
        fitter_return = fitter(B, distribution, y, matrix_xreg, **fitter_kwargs)

    mu = fitter_return["mu"]
    mu_otU = mu[otU]
    scale = fitter_return["scale"]
    other_val = fitter_return["other"]

    if loss in ("likelihood", "ROLE"):
        # The errors after the latent sizes have the variances of the filter
        if latent is not None and latent["filter"]:
            scale = scale * fitter_return["variance"][otU] ** (
                1 / latent["scale_power"]
            )
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
            # R's sum(): sequential, in a long double accumulator
            cf_value = -float(np.cumsum(log_lik_array, dtype=np.longdouble)[-1])
        else:
            cf_value = (
                # R: -meanFast(pointLiks, trim=trim) * obsInsample, trimming the lowest
                -mean_fast(log_lik_array, trim=trim) * obs_insample
            )

    else:
        effective_lambda_bc = (
            other_val
            if distribution == "dbcnorm" and not a_parameter_provided
            else lambda_bc
        )
        y_fitted = extractor_fitted(distribution, mu, scale, effective_lambda_bc)
        # LASSO / RIDGE work with the response in the space of the model, as R
        if loss in ("LASSO", "RIDGE") and distribution in (
            "dlnorm",
            "dllaplace",
            "dls",
            "dlgnorm",
            "dlogitnorm",
            "dbcnorm",
        ):
            y_fitted = mu

        # meanFast() and sum() of R, in its order and accumulator
        if loss == "MSE":
            cf_value = mean_fast((y - y_fitted) ** 2)
        elif loss == "MAE":
            cf_value = mean_fast(np.abs(y - y_fitted))
        elif loss == "HAM":
            cf_value = mean_fast(np.sqrt(np.abs(y - y_fitted)))
        elif loss == "LASSO":
            cf_value = (1 - lambda_val) * mean_fast(
                (y - y_fitted) ** 2
            ) + lambda_val * float(np.cumsum(np.abs(B), dtype=np.longdouble)[-1])
        elif loss == "RIDGE":
            B_scaled = B * denominator
            if len(B_scaled) > 1:
                cf_value = (1 - lambda_val) * mean_fast(
                    (y - y_fitted) ** 2
                ) + lambda_val * np.sqrt(
                    float(np.cumsum(B_scaled[1:] ** 2, dtype=np.longdouble)[-1])
                )
            else:
                cf_value = (1 - lambda_val) * mean_fast(
                    (y - y_fitted) ** 2
                ) + lambda_val * np.sqrt(
                    float(np.cumsum(B_scaled**2, dtype=np.longdouble)[-1])
                )
            if lambda_val == 1.0:
                cf_value = mean_fast((y - y_fitted) ** 2)
        elif loss == "custom" and loss_function is not None:
            cf_value = loss_function(y, y_fitted, B, matrix_xreg)
        else:
            cf_value = mean_fast((y - y_fitted) ** 2)

    if np.isnan(cf_value) or np.isinf(cf_value):
        cf_value = 1e300

    # A non-stationary AR polynomial divides the cost by its smallest root, as R
    if ar_order > 0:
        roots = np.abs(np.polynomial.polynomial.polyroots(fitter_return["poly1"]))
        if np.any(roots < 1):
            cf_value = cf_value / np.min(roots)

    return cf_value
