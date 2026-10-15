"""Diagnostic functions for regression models.

This module provides functions for outlier detection and model diagnostics.
"""

from typing import Literal

import numpy as np
import pandas as pd

from . import distributions as dist


class OutlierResult:
    """Result of outlier detection.

    Attributes
    ----------
    outliers : pd.DataFrame or None
        Matrix with dummy variables, flagging outliers, with the columns
        ``outlier1``, ``outlier2``, ... as in R.
    statistic : np.ndarray
        The value of the statistic for the normalized variable.
    id : np.ndarray
        The ids of the outliers (which observations have them), from zero.
    level : float
        The confidence level used in the process.
    type : str
        The type of the residuals used.
    errors : np.ndarray
        The errors used in the detection.
    """

    def __init__(
        self,
        outliers: pd.DataFrame | None,
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
    confidence level specified by user, as R's ``outlierdummy()``: the
    standardised or studentised residuals of the model are compared with the
    quantiles of the standardised distribution (the standard Normal ones for
    the count distributions, whose residuals are transformed into it).

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
    if type not in ("rstandard", "rstudent"):
        raise ValueError("type should be either 'rstandard' or 'rstudent'")
    errors = model.rstandard() if type == "rstandard" else model.rstudent()

    p = np.array([(1 - level) / 2, (1 + level) / 2])
    # The scale of R's extractScale(): 1 for a scale model
    nobs, nparam, other = model.nobs, model.nparam, model.other_
    scale = model.extract_scale()
    distribution = model.distribution
    if distribution in ("dlaplace", "dllaplace"):
        statistic = dist.qlaplace(p, 0, 1)
    elif distribution == "dalaplace":
        statistic = dist.qalaplace(p, 0, 1, other)
    elif distribution == "dlogis":
        statistic = dist.qlogis(p, 0, 1)
    elif distribution == "dt":
        statistic = dist.qt(p, nobs - nparam)
    elif distribution in ("dgnorm", "dlgnorm"):
        statistic = dist.qgnorm(p, 0, 1, other)
    elif distribution in ("ds", "dls"):
        statistic = dist.qs(p, 0, 1)
    elif distribution == "dinvgauss":
        # The scale is debiased, taking n-k into account
        statistic = dist.qinvgauss(p, mean=1, dispersion=scale * nobs / (nobs - nparam))
    elif distribution == "dgamma":
        statistic = dist.qgamma(p, shape=1 / scale, scale=scale)
    elif distribution == "dexp":
        statistic = dist.qexp(p, rate=1)
    elif distribution == "dfnorm":
        statistic = dist.qfnorm(p, 0, 1)
    elif distribution == "drectnorm":
        statistic = dist.qrectnorm(p, 0, 1)
    else:
        # Including the count distributions, transformed into the standard Normal
        statistic = dist.qnorm(p, 0, 1)
    statistic = np.asarray(statistic, dtype=float)

    outlier_ids = np.flatnonzero((errors > statistic[1]) | (errors < statistic[0]))
    outliers = None
    if len(outlier_ids) > 0:
        dummies = np.zeros((nobs, len(outlier_ids)))
        dummies[outlier_ids, np.arange(len(outlier_ids))] = 1
        outliers = pd.DataFrame(
            dummies, columns=[f"outlier{i}" for i in range(1, len(outlier_ids) + 1)]
        )

    return OutlierResult(
        outliers=outliers,
        statistic=statistic,
        id=outlier_ids,
        level=level,
        type=type,
        errors=errors,
    )
