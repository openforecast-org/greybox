"""Cost function for optimization.

This module contains the cost function (cf) that combines the fitter with
the loss function (likelihood, MSE, MAE, etc.).
"""

import numpy as np
from scipy import stats

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
        # R: dinvgauss(y, mean=mu, dispersion=scale/mu); greybox's dinvgauss takes
        # the mean and the dispersion, as R's does
        return dist.dinvgauss(y_otU, loc=mu_otU, scale=scale / mu_otU, log=True)
    elif distribution == "dgamma":
        return dist.dgamma(y_otU, shape=1 / scale, scale=scale * mu_otU, log=True)
    elif distribution == "dexp":
        return dist.dexp(y_otU, loc=0, scale=mu_otU, log=True)
    elif distribution == "dchisq":
        # R: dchisq(y, df=scale, ncp=mu), the non-central chi-squared
        return stats.ncx2.logpdf(y_otU, df=scale, nc=mu_otU)
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


def entropy_zero(distribution: str, scale, mu, other=None) -> np.ndarray:
    """Differential entropy of each zero of an occurrence model (R: ``entropyZero``).

    Minus the expected log-density at the location ``mu``, with the scale in the
    form that the density takes (the standard deviation for the Normal-based
    distributions). The log-domain distributions add the Jacobian
    ``E[log(y)] = mu``. Where there is no closed form, it is the entropy of a
    Normal distribution with the same variance.
    """
    from scipy.special import betaln, digamma, gammaln

    mu = np.asarray(mu, dtype=float)
    log_sqrt_2pi = 0.5 * np.log(2 * np.pi)
    if distribution in ("dnorm", "dfnorm", "drectnorm", "dbcnorm", "dlogitnorm"):
        # Normal approximation for the folded, rectified, Box-Cox and logit Normal
        entropy = log_sqrt_2pi + np.log(scale) + 0.5
    elif distribution == "dlnorm":
        entropy = log_sqrt_2pi + np.log(scale) + 0.5 + mu
    elif distribution in ("dgnorm", "dlgnorm"):
        entropy = 1 / other - np.log(other / (2 * scale)) + gammaln(1 / other)
        if distribution == "dlgnorm":
            entropy = entropy + mu
    elif distribution in ("dlaplace", "dllaplace"):
        entropy = 1 + np.log(2 * scale)
        if distribution == "dllaplace":
            entropy = entropy + mu
    elif distribution == "dalaplace":
        entropy = 1 + np.log(scale / (other * (1 - other)))
    elif distribution in ("ds", "dls"):
        entropy = 2 + 2 * np.log(2 * scale)
        if distribution == "dls":
            entropy = entropy + mu
    elif distribution == "dlogis":
        entropy = 2 + np.log(scale)
    elif distribution == "dt":
        entropy = (
            (scale + 1) / 2 * (digamma((scale + 1) / 2) - digamma(scale / 2))
            + np.log(np.sqrt(scale))
            + betaln(scale / 2, 0.5)
        )
    elif distribution == "dexp":
        entropy = 1 + np.log(mu)
    elif distribution == "dgamma":
        entropy = (
            1 / scale
            + np.log(scale * mu)
            + gammaln(1 / scale)
            + (1 - 1 / scale) * digamma(1 / scale)
        )
    elif distribution == "dinvgauss":
        # Normal approximation with the variance mu^2 * scale
        entropy = np.log(np.sqrt(2 * np.pi * scale) * mu) + 0.5
    elif distribution == "dchisq":
        # The central chi-squared with df=scale, without the non-centrality
        entropy = (
            scale / 2
            + np.log(2)
            + gammaln(scale / 2)
            + (1 - scale / 2) * digamma(scale / 2)
        )
    elif distribution == "dbeta":
        entropy = (
            betaln(mu, scale)
            - (mu - 1) * digamma(mu)
            - (scale - 1) * digamma(scale)
            + (mu + scale - 2) * digamma(mu + scale)
        )
    # Normal approximations for the counts
    elif distribution == "dpois":
        entropy = 0.5 * np.log(2 * np.pi * mu) + 0.5
    elif distribution == "dnbinom":
        entropy = 0.5 * np.log(2 * np.pi * (mu + mu**2 / other)) + 0.5
    elif distribution == "dgeom":
        entropy = 0.5 * np.log(2 * np.pi * mu * (1 + mu)) + 0.5
    elif distribution == "dbinom":
        entropy = 0.5 * np.log(2 * np.pi * other * mu / (1 + mu) ** 2) + 0.5
    else:
        entropy = 0.0
    # One value per zero, also where the entropy does not depend on mu
    return np.broadcast_to(np.asarray(entropy, dtype=float), mu.shape).copy()


def _entropy_adjustment(
    distribution: str,
    scale,
    other_val,
    mu: np.ndarray,
    otU: np.ndarray,
) -> float:
    """The differential entropy of the zeros (recursive occurrence models)."""
    scale_zero = scale[~otU] if np.ndim(scale) > 0 and np.size(scale) > 1 else scale
    return float(np.sum(entropy_zero(distribution, scale_zero, mu[~otU], other_val)))


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
        otU=otU,
        trim=trim,
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
                # R: -meanFast(pointLiks, trim=trim) * obsInsample, trimming the lowest
                -mean_fast(log_lik_array, trim=trim) * obs_insample
            )

        if recursive_model and occurrence_model:
            cf_value += _entropy_adjustment(
                distribution,
                scale,
                size if distribution == "dbinom" else other_val,
                mu,
                otU,
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
