"""Prediction functions for greybox models.

This module provides prediction functionality similar to R's predict.greybox
and predict.alm methods.
"""

import numpy as np
from scipy import stats

from .alm import PredictionResult
from .fitters import scale_sd


def predict_basic(model, X, interval="none", level=0.95, side="both"):
    """Basic prediction function (similar to R's predict.greybox).

    Parameters
    ----------
    model : ALM
        Fitted ALM model.
    X : np.ndarray
        Design matrix for predictions.
    interval : str, optional
        Type of interval: "none", "confidence", or "prediction".
    level : float, optional
        Confidence level (default 0.95 for 95%).
    side : str, optional
        Side of interval: "both", "upper", or "lower".

    Returns
    -------
    PredictionResult
        Prediction results with mean, lower, upper bounds, etc.
    """
    if not hasattr(model, "_coef") or model._coef is None:
        raise ValueError("Model not fitted. Call fit() first.")

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[1] != len(model.coef) + 1:
        raise ValueError(
            f"X has {X.shape[1]} columns but model expects {len(model.coef) + 1}"
        )

    if interval not in ("none", "confidence", "prediction"):
        raise ValueError("interval must be 'none', 'confidence', or 'prediction'")

    if side not in ("both", "upper", "lower"):
        raise ValueError("side must be 'both', 'upper', or 'lower'")

    if isinstance(level, (int, float)):
        level = [level]

    n_levels = len(level)
    n_obs = X.shape[0]

    if side == "upper":
        level_low = [0.0] * n_levels
        level_up = level
    elif side == "lower":
        level_low = [1 - lev for lev in level]
        level_up = [1.0] * n_levels
    else:
        level_low = [(1 - lev) / 2 for lev in level]
        level_up = [(1 + lev) / 2 for lev in level]

    param_quantiles = stats.t.ppf(level_low + level_up, df=model.df_residual_)

    coef_array = np.concatenate([[model.intercept_], model.coef])
    mean = X @ coef_array

    variances = None
    lower = None
    upper = None

    if interval != "none":
        scale = scale_sd(model.distribution, model.scale)
        variances = np.abs(np.diag(X @ X.T)) * (scale**2)

        if interval == "confidence":
            se = np.sqrt(variances)
            for i in range(n_levels):
                lower_i = mean + param_quantiles[i] * se
                upper_i = mean + param_quantiles[i + n_levels] * se
                if lower is None:
                    lower = np.zeros((n_obs, n_levels))
                    upper = np.zeros((n_obs, n_levels))
                lower[:, i] = lower_i
                upper[:, i] = upper_i

        elif interval == "prediction":
            sigma_sq = scale**2
            total_var = variances + sigma_sq
            se = np.sqrt(total_var)
            for i in range(n_levels):
                lower_i = mean + param_quantiles[i] * se
                upper_i = mean + param_quantiles[i + n_levels] * se
                if lower is None:
                    lower = np.zeros((n_obs, n_levels))
                    upper = np.zeros((n_obs, n_levels))
                lower[:, i] = lower_i
                upper[:, i] = upper_i

    return PredictionResult(
        mean=mean,
        lower=lower,
        upper=upper,
        level=level,
        variances=variances,
        side=side,
        interval=interval,
    )


def predict(
    model,
    newdata=None,
    interval="none",
    level=0.95,
    side="both",
    occurrence=None,
):
    """Prediction function for ALM models (similar to R's predict.alm).

    Parameters
    ----------
    model : ALM
        Fitted ALM model.
    newdata : dict, DataFrame, or np.ndarray, optional
        New data for predictions. If None, uses training data.
    interval : str, optional
        Type of interval: "none", "confidence", or "prediction".
    level : float or list, optional
        Confidence level(s) (default 0.95 for 95%).
    side : str, optional
        Side of interval: "both", "upper", or "lower".
    occurrence : np.ndarray, optional
        Occurrence values for zero-inflated models.

    Returns
    -------
    PredictionResult
        Prediction results with mean, lower, upper bounds, etc.
    """
    if not hasattr(model, "_coef") or model._coef is None:
        raise ValueError("Model not fitted. Call fit() first.")

    if interval not in ("none", "confidence", "prediction"):
        raise ValueError("interval must be 'none', 'confidence', or 'prediction'")

    if side not in ("both", "upper", "lower"):
        raise ValueError("side must be 'both', 'upper', or 'lower'")

    if isinstance(level, (int, float)):
        level = [level]

    n_levels = len(level)

    if newdata is None:
        X = model._X_train_
    else:
        from .formula import formula

        if isinstance(newdata, dict):
            if "y" in newdata:
                _, X = formula("y ~ .", newdata)
            else:
                X = formula("~ .", newdata, return_type="X")
        elif hasattr(newdata, "to_dict"):
            data_dict = newdata.to_dict(orient="list")
            if "y" in data_dict:
                _, X = formula("y ~ .", data_dict)
            else:
                X = formula("~ .", data_dict, return_type="X")
        elif isinstance(newdata, np.ndarray):
            X = newdata
        else:
            raise ValueError("newdata must be dict, DataFrame, or numpy array")

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    result = predict_basic(model, X, interval, level, side)

    h = X.shape[0]

    if occurrence is not None:
        occurrence = np.asarray(occurrence, dtype=float)
        if len(occurrence) != h:
            raise ValueError(
                f"occurrence has {len(occurrence)} elements but newdata has {h} rows"
            )

    if interval != "none":
        level_matrix = np.tile(level, (h, 1))

        if side == "upper":
            level_low = np.zeros((h, n_levels))
            level_up = level_matrix
        elif side == "lower":
            level_low = 1 - level_matrix
            level_up = np.ones((h, n_levels))
        else:
            level_low = (1 - level_matrix) / 2
            level_up = (1 + level_matrix) / 2

        level_low = np.clip(level_low, 0, 1)
        level_up = np.clip(level_up, 0, 1)

        if model.distribution == "dnorm" and result.lower is not None:
            sd = scale_sd(model.distribution, model.scale)
            result.lower = stats.norm.ppf(level_low, loc=result.mean, scale=sd)
            result.upper = stats.norm.ppf(level_up, loc=result.mean, scale=sd)

    return result
