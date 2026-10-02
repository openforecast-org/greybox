"""Prediction functions for greybox models.

This module provides prediction functionality similar to R's predict.greybox
and predict.alm methods.
"""

import numpy as np
from scipy import stats

from .fitters import scale_sd


def predict_basic(model, X, interval="none", level=0.95, side="both"):
    """Basic prediction function (similar to R's predict.greybox).

    It is :meth:`ALM.predict`, which handles every distribution, the link
    functions and the ARIMA terms; this function kept its own Normal-only
    calculation before, which failed for AR models and counts.

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
    return model.predict(X, interval=interval, level=level, side=side)


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
