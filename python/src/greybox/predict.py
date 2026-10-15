"""Prediction functions for greybox models.

This module provides prediction functionality similar to R's predict.greybox
and predict.alm methods.
"""

import re

import numpy as np
import pandas as pd


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
        New data for predictions: the variables of the formula of the model (its
        factors with their levels in the training data), or its design matrix.
        If None, the in-sample values, as R's ``newdata=NULL``.
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

    X = newdata
    if isinstance(newdata, dict) or hasattr(newdata, "to_dict"):
        from .formula import formula

        # The design matrix of the formula of the model (the data without its
        # response), with the levels of the factors of the training data and the
        # trend after the in-sample one, as R's predict.greybox()
        frame = pd.DataFrame(newdata)
        frame.attrs["_xlevels"] = getattr(model, "_xlevels", None) or {}
        formula_str = model._formula_ or ("y ~ ." if "y" in frame else "~ .")
        if "trend" not in frame and re.search(
            r"(?<![\w.])trend(?![\w.])", formula_str.split("~")[-1]
        ):
            frame["trend"] = len(model._y_train_) + np.arange(1.0, len(frame) + 1)
        X = formula(formula_str, frame, return_type="X")
    elif newdata is not None and not isinstance(newdata, np.ndarray):
        raise ValueError("newdata must be dict, DataFrame, or numpy array")

    if occurrence is not None and X is not None and len(occurrence) != len(X):
        raise ValueError(
            f"occurrence has {len(occurrence)} elements but newdata has {len(X)} rows"
        )

    # ALM.predict() gives the intervals of all the distributions and the occurrence,
    # as R's predict.alm()
    return model.predict(
        None if X is None else np.asarray(X, dtype=float),
        interval=interval,
        level=level,
        side=side,
        occurrence=occurrence,
    )
