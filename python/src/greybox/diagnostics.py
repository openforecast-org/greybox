"""Diagnostic functions for regression models.

This module provides functions for outlier detection and model diagnostics.
"""

from typing import Literal

import numpy as np
from scipy import stats

from .fitters import scale_sd


class OutlierResult:
    """Result of outlier detection.

    Attributes
    ----------
    outliers : np.ndarray or None
        Matrix with dummy variables, flagging outliers.
    statistic : np.ndarray
        The value of the statistic for the normalized variable.
    id : np.ndarray
        The ids of the outliers (which observations have them).
    level : float
        The confidence level used in the process.
    type : str
        The type of the residuals used.
    errors : np.ndarray
        The errors used in the detection.
    """

    def __init__(
        self,
        outliers: np.ndarray | None,
        statistic: np.ndarray,
        id: np.ndarray,
        level: float,
        type: str,
        errors: np.ndarray,
    ):
        self.outliers = outliers
        self.statistic = statistic
        self.id = id
        self.level = level
        self.type = type
        self.errors = errors


def outlier_dummy(
    model,
    level: float = 0.999,
    type: Literal["rstandard", "rstudent"] = "rstandard",
) -> OutlierResult:
    """Detect outliers and create dummy variables.

    Function detects outliers and creates a matrix with dummy variables.
    Only point outliers are considered (no level shifts).

    The detection is done based on the type of distribution used and
    confidence level specified by user.

    Parameters
    ----------
    model : ALM
        Fitted ALM model.
    level : float, default=0.999
        Confidence level to use. Everything outside the constructed
        bounds based on that is flagged as outliers.
    type : {"rstandard", "rstudent"}, default="rstandard"
        Type of residuals to use: either standardised or studentised.

    Returns
    -------
    OutlierResult
        Object containing:
        - outliers: matrix with dummy variables flagging outliers
        - statistic: value of the statistic for the normalised variable
        - id: ids of the outliers
        - level: confidence level used
        - type: type of residuals used
        - errors: the errors used in detection

    Examples
    --------
    >>> from greybox import formula, ALM, outlier_dummy
    >>> data = {'y': [1, 2, 3, 4, 5, 100], 'x': [1, 2, 3, 4, 5, 6]}
    >>> y, X = formula("y ~ x", data)
    >>> model = ALM(distribution="dnorm", loss="likelihood")
    >>> _ = model.fit(X, y)
    >>> result = outlier_dummy(model)
    """
    if not hasattr(model, "residuals") or model.residuals is None:
        raise ValueError("Model must be a fitted ALM model")

    residuals = model.residuals
    nobs = len(residuals)
    distribution = model.distribution

    scale = scale_sd(distribution, model.scale)
    df_residual = model.df_residual_

    if type == "rstandard":
        leverage = _calculate_leverage(model)
        errors = residuals / np.sqrt(scale**2 * (1 - leverage))
    else:
        h = _calculate_leverage(model)
        n = nobs
        k = model.nparam
        se = scale**2 * (1 - h) * (n - k - 1) / (n - k - 2)
        errors = residuals / np.sqrt(se)

    if distribution in ("dlaplace", "dllaplace"):
        statistic = stats.laplace.ppf(
            np.array([(1 - level) / 2, (1 + level) / 2]), loc=0, scale=1
        )
    elif distribution == "dlogis":
        statistic = stats.logistic.ppf(
            np.array([(1 - level) / 2, (1 + level) / 2]), loc=0, scale=1
        )
    elif distribution == "dt":
        statistic = stats.t.ppf(
            np.array([(1 - level) / 2, (1 + level) / 2]), df=df_residual
        )
    elif distribution in ("dgnorm", "dlgnorm"):
        shape = model.other_ if model.other_ is not None else 2.0
        quantile = [(1 - level) / 2, (1 + level) / 2]
        statistic = stats.gennorm.ppf(quantile, loc=0, scale=1, beta=shape)
    elif distribution in ("ds", "dls"):
        statistic = stats.norm.ppf(
            np.array([(1 - level) / 2, (1 + level) / 2]), loc=0, scale=1
        )
    elif distribution == "dgamma":
        statistic = stats.gamma.ppf(
            np.array([(1 - level) / 2, (1 + level) / 2]),
            a=1 / scale,
            scale=scale,
        )
    elif distribution == "dexp":
        statistic = stats.expon.ppf(
            np.array([(1 - level) / 2, (1 + level) / 2]), scale=1
        )
    else:
        statistic = stats.norm.ppf(
            np.array([(1 - level) / 2, (1 + level) / 2]), loc=0, scale=1
        )

    is_outlier = (errors > statistic[1]) | (errors < statistic[0])
    outlier_ids = np.where(is_outlier)[0]
    n_outliers = len(outlier_ids)

    if n_outliers > 0:
        outliers = np.zeros((nobs, n_outliers))
        for i, oid in enumerate(outlier_ids):
            outliers[oid, i] = 1
    else:
        outliers = None

    return OutlierResult(
        outliers=outliers,
        statistic=statistic,
        id=outlier_ids,
        level=level,
        type=type,
        errors=errors,
    )


def _calculate_leverage(model) -> np.ndarray:
    """Calculate leverage (hat values) for each observation."""
    if model._X_train_ is None:
        raise ValueError("Model not fitted")

    X = model._X_train_
    XtX = X.T @ X
    XtX += np.eye(XtX.shape[0]) * 1e-10
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    leverage = np.diag(X @ XtX_inv @ X.T)
    return leverage
