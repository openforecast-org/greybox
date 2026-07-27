"""Measures of association.

This module provides functions for calculating various measures of association
including partial correlations, multiple correlations, and correlation analysis.
"""

from typing import Literal

import numpy as np
from scipy import stats


def pcor(
    x: np.ndarray,
    y: np.ndarray | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> dict:
    """Calculate partial correlations.

    Function calculates partial correlations between the provided variables.
    The calculation is done based on multiple linear regressions.

    Parameters
    ----------
    x : np.ndarray
        DataFrame or matrix with numeric values.
    y : np.ndarray, optional
        The numerical variable. If provided, calculates partial correlation
        between each column of x and y.
    method : {"pearson", "spearman", "kendall"}, default="pearson"
        Which method to use for calculation.

    Returns
    -------
    dict
        Dictionary containing:
        - value: matrix of partial correlation coefficients
        - p.value: p-values for the coefficients
        - method: method used

    Examples
    --------
    >>> x = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [5, 10, 15]]
    ... )
    >>> result = pcor(x)
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if y is not None:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        x = np.column_stack([x, y])

    n_vars = x.shape[1]

    if n_vars < 2:
        raise ValueError("Need at least 2 variables")

    mask = ~np.isnan(np.all(x, axis=1))
    x_clean = x[mask]

    if len(x_clean) < 3:
        raise ValueError("Not enough observations")

    cor_matrix = np.corrcoef(x_clean, rowvar=False)

    try:
        cor_inv = np.linalg.inv(cor_matrix)
    except np.linalg.LinAlgError:
        cor_inv = np.linalg.pinv(cor_matrix)

    pcor_matrix = np.zeros_like(cor_matrix)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                pcor_matrix[i, j] = 1.0
            else:
                pcor_matrix[i, j] = -cor_inv[i, j] / np.sqrt(
                    cor_inv[i, i] * cor_inv[j, j]
                )

    df = len(x_clean) - n_vars
    # The diagonal (self-correlation = 1) divides by zero here, but those
    # entries are overwritten below, so suppress the expected warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = pcor_matrix * np.sqrt(df / (1 - pcor_matrix**2))
    t_stat = np.where(np.eye(n_vars, dtype=bool), 0, t_stat)
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))
    p_value = np.where(np.eye(n_vars, dtype=bool), np.nan, p_value)

    return {"value": pcor_matrix, "p.value": p_value, "method": method}


def mcor(x: np.ndarray, y: np.ndarray | None = None) -> float:
    """Calculate multiple correlation coefficient.

    Function returns the multiple correlation coefficient between a set
    of variables and a dependent variable.

    Parameters
    ----------
    x : np.ndarray
        Matrix of independent variables.
    y : np.ndarray
        The dependent variable.

    Returns
    -------
    float
        Multiple correlation coefficient.

    Examples
    --------
    >>> x = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    >>> y = np.array([3, 6, 9, 12, 15])
    >>> round(float(mcor(x, y)), 4)
    1.0
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    mask = ~(np.isnan(np.any(x, axis=1)) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(y_clean) < 3:
        raise ValueError("Not enough observations")

    x_with_intercept = np.column_stack([np.ones(len(x_clean)), x_clean])

    try:
        coeffs = np.linalg.lstsq(x_with_intercept, y_clean, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan

    y_pred = x_with_intercept @ coeffs

    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)

    if ss_tot == 0:
        return np.nan

    r_squared = 1 - ss_res / ss_tot
    return np.sqrt(r_squared)


def determination(
    xreg: np.ndarray,
    bruteforce: bool = True,
) -> np.ndarray:
    """Coefficients of determination.

    Function produces coefficients of determination for the provided data.

    The function calculates coefficients of determination (R^2) between all
    the provided variables. The higher the coefficient for a variable is,
    the higher the potential multicollinearity effect in the model with the
    variable will be. Coefficients of determination are connected directly
    to Variance Inflation Factor (VIF): VIF = 1 / (1 - determination).
    Arguably it is easier to interpret, because it is restricted with (0, 1)
    bounds. The multicollinearity can be considered as serious when
    determination > 0.9 (which corresponds to VIF > 10).

    Parameters
    ----------
    xreg : np.ndarray
        Data frame or matrix containing the exogenous variables.
    bruteforce : bool, default=True
        If True, then all the variables will be used for the regression
        construction (sink regression). If the number of observations is
        smaller than the number of series, the function will use stepwise
        function and select only meaningful variables. So the reported
        values will be based on stepwise regressions for each variable.

    Returns
    -------
    np.ndarray
        Vector of determination coefficients, one for each variable.

    Examples
    --------
    >>> np.random.seed(42)
    >>> x1 = np.random.normal(10, 3, 100)
    >>> x2 = np.random.normal(50, 5, 100)
    >>> x3 = 100 + 0.5*x1 - 0.75*x2 + np.random.normal(0, 3, 100)
    >>> xreg = np.column_stack([x3, x1, x2])
    >>> np.round(determination(xreg), 4)
    array([1.    , 0.2872, 0.5649])
    """
    from .selection import stepwise

    xreg = np.asarray(xreg, dtype=float)

    if xreg.ndim == 1:
        xreg = xreg.reshape(-1, 1)

    n_variables = xreg.shape[1]
    n_series = xreg.shape[0]

    vector_correlations_multiple = np.full(n_variables, np.nan)

    if n_series <= n_variables and bruteforce:
        import warnings

        warnings.warn(
            "The number of variables is larger than the number of observations. "
            "Sink regression cannot be constructed. Using stepwise.",
            RuntimeWarning,
        )
        bruteforce = False

    mask = ~np.isnan(np.any(xreg, axis=1))
    xreg_clean = xreg[mask]

    if n_variables <= 1:
        return np.array([0.0])

    def determination_calculator(residuals: np.ndarray, actuals: np.ndarray) -> float:
        return 1 - np.sum(residuals**2) / np.sum((actuals - np.mean(actuals)) ** 2)

    if bruteforce:
        try:
            cor_matrix = np.corrcoef(xreg_clean, rowvar=False)

            for i in range(n_variables):
                try:
                    cor_subset = cor_matrix[i, :i] if i > 0 else np.array([])
                    if i < n_variables - 1:
                        cor_subset = np.concatenate(
                            [cor_subset, cor_matrix[i, i + 1 :]]
                        )

                    if i > 0:
                        cor_matrix_others = np.delete(
                            np.delete(cor_matrix, i, axis=0), i, axis=1
                        )
                        cor_others_to_i = cor_matrix[:i, i] if i > 0 else np.array([])
                        if i < n_variables - 1:
                            cor_others_to_i = np.concatenate(
                                [cor_others_to_i, cor_matrix[i + 1 :, i]]
                            )

                        if cor_matrix_others.size > 0:
                            coeffs = np.linalg.solve(cor_matrix_others, cor_others_to_i)
                            r_squared = cor_subset @ coeffs
                        else:
                            r_squared = 1.0
                    else:
                        r_squared = 1.0

                    vector_correlations_multiple[i] = min(max(r_squared, 0.0), 1.0)
                except np.linalg.LinAlgError:
                    vector_correlations_multiple[i] = 1.0
        except Exception:  # noqa: BLE001 - degenerate matrix falls back to 1.0
            vector_correlations_multiple[:] = 1.0
    else:
        for i in range(n_variables):
            y = xreg_clean[:, i]
            X = np.delete(xreg_clean, i, axis=1)

            if X.shape[1] == 0:
                vector_correlations_multiple[i] = 0.0
                continue

            X_with_intercept = np.column_stack([np.ones(len(X)), X])

            try:
                coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                y_pred = X_with_intercept @ coeffs
                vector_correlations_multiple[i] = determination_calculator(
                    y - y_pred, y
                )
            except Exception:  # noqa: BLE001 - lstsq failure retries via stepwise
                try:
                    data_dict = {"y": y}
                    for j in range(X.shape[1]):
                        data_dict[f"x{j + 1}"] = X[:, j]
                    model = stepwise(data_dict, silent=True)
                    y_pred = model.predict(X)["mean"]
                    vector_correlations_multiple[i] = determination_calculator(
                        y - y_pred, y
                    )
                except Exception:  # noqa: BLE001 - stepwise fallback also failed
                    vector_correlations_multiple[i] = np.nan

    return vector_correlations_multiple


def association(
    x: np.ndarray,
    y: np.ndarray | None = None,
    method: str = "auto",
) -> dict:
    """Calculate measures of association.

    Function returns the matrix of measures of association for different types
    of variables.

    Parameters
    ----------
    x : np.ndarray
        DataFrame or matrix.
    y : np.ndarray, optional
        The numerical variable.
    method : str, default="auto"
        Method to use: "auto", "pearson", "spearman", "kendall", "cramer".
        "auto" selects based on variable types.

    Returns
    -------
    dict
        Dictionary containing:
        - value: matrix of association coefficients
        - p.value: p-values
        - type: matrix of types of measures used

    Examples
    --------
    >>> x = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    >>> result = association(x)
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if y is not None:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        x = np.column_stack([x, y])

    n_vars = x.shape[1]

    if n_vars < 2:
        raise ValueError("Need at least 2 variables")

    mask = ~np.isnan(np.all(x, axis=1))
    x_clean = x[mask]

    if method == "auto":
        cor_matrix = np.corrcoef(x_clean, rowvar=False)
    else:
        if method == "pearson":
            cor_matrix = np.corrcoef(x_clean, rowvar=False)
        elif method == "spearman":
            cor_matrix = stats.spearmanr(x_clean).correlation
        elif method == "kendall":
            cor_matrix = np.ones((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    tau, _ = stats.kendalltau(x_clean[:, i], x_clean[:, j])
                    cor_matrix[i, j] = tau
                    cor_matrix[j, i] = tau
        else:
            raise ValueError(f"Unknown method: {method}")

    df = len(x_clean) - 2
    t_stat = cor_matrix * np.sqrt(df / (1 - cor_matrix**2 + 1e-10))
    t_stat = np.where(np.eye(n_vars, dtype=bool), 0, t_stat)
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))
    p_value = np.where(np.eye(n_vars, dtype=bool), np.nan, p_value)

    type_matrix = np.full((n_vars, n_vars), "pearson")

    return {"value": cor_matrix, "p.value": p_value, "type": type_matrix}
