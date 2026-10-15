"""Measures of association.

This module provides functions for calculating various measures of association
including partial correlations, multiple correlations, and correlation analysis.
"""

import warnings
from dataclasses import dataclass
from itertools import permutations
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import linalg, stats

from .fitters import _lm_fit


@dataclass
class McorResult:
    """Result of :func:`mcor`. Mirrors R's ``mcor`` S3 class.

    Parameters
    ----------
    value : float
        The multiple correlation coefficient.
    statistic : float
        The F-statistic of the regression.
    df : float
        The number of degrees of freedom of the regression.
    df_residual : float
        The number of degrees of freedom of the residuals.
    p_value : float
        The p-value of the F-test.
    """

    value: float
    statistic: float
    df: float
    df_residual: float
    p_value: float

    def __str__(self) -> str:
        return (
            f"Multiple correlations value: {round(self.value, 4)}\n"
            f"F-statistics = {round(self.statistic, 4)}, df: {self.df}, "
            f"df resid: {self.df_residual}, p-value: {round(self.p_value, 4)}"
        )


@dataclass
class CramerResult:
    """Result of :func:`cramer`. Mirrors R's ``cramer`` S3 class.

    Parameters
    ----------
    value : float
        Cramer's V.
    statistic : float
        The Chi-squared statistic of the test of independence.
    p_value : float
        The p-value of the Chi-squared test.
    df : float
        The number of degrees of freedom of the test.
    """

    value: float
    statistic: float
    p_value: float
    df: float

    def __str__(self) -> str:
        return (
            f"Cramer's V: {round(self.value, 4)}\n"
            f"Chi^2 statistics = {round(self.statistic, 4)}, df: {self.df}, "
            f"p-value: {round(self.p_value, 4)}"
        )


def pcor(
    x: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.DataFrame | None = None,
    use: Literal[
        "na.or.complete", "complete.obs", "everything", "all.obs"
    ] = "na.or.complete",
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> dict:
    """Calculate partial correlations.

    Function calculates partial correlations between the provided variables.
    The calculation is done based on multiple linear regressions: for each pair
    of variables, the correlation (via R's ``cor.test``) between the residuals
    of the regressions of the two variables on all the other ones. As in R, the
    regressions have no intercept, and the p-values use the degrees of freedom
    of ``cor.test`` on the residuals. Categorical variables are replaced by the
    numbers of their levels, with a warning.

    Parameters
    ----------
    x : np.ndarray or pd.DataFrame
        DataFrame or matrix with numeric values.
    y : np.ndarray or pd.DataFrame, optional
        The numerical variable. If provided, calculates partial correlation
        between each column of y and each column of x.
    use : {"na.or.complete", "complete.obs", "everything", "all.obs"}
        What observations to use, as in R's ``cor``: incomplete observations
        are removed, except for ``"all.obs"`` and ``"everything"``, which
        raise an error if there are any (the latter as R's ``.lm.fit``).
    method : {"pearson", "spearman", "kendall"}, default="pearson"
        Which method to use for calculation.

    Returns
    -------
    dict
        Dictionary containing:

        - value: matrix of partial correlation coefficients (n_y x n_x if y is
          provided)
        - p.value: p-values for the coefficients
        - method: method used

    Examples
    --------
    >>> x = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [5, 10, 15]]
    ... )
    >>> result = pcor(x)
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(f"Unknown method: {method}")
    if use not in ("na.or.complete", "complete.obs", "everything", "all.obs"):
        raise ValueError(f"Unknown use: {use}")

    frames = [pd.DataFrame(v) for v in (x, y) if v is not None]
    if not all(is_numeric_dtype(col) for frame in frames for _, col in frame.items()):
        warnings.warn(
            "Some of the variables are in categorical scales. "
            "Using partial correlations might be meaningless!",
            stacklevel=2,
        )
    # The categorical variables as the numbers of their levels, as R
    for frame in frames:
        for name, col in frame.items():
            if not is_numeric_dtype(col):
                codes = col.astype("category").cat.codes
                frame[name] = codes.where(codes >= 0) + 1
    x = frames[0].to_numpy(dtype=float)
    n_x = x.shape[1]

    if y is None:
        if n_x == 1:
            return {"value": 1.0, "p.value": 1.0, "method": method}
        data = x
    else:
        y = frames[1].to_numpy(dtype=float)
        data = np.column_stack([x, y])
    obs_na = np.isnan(data).any(axis=1)
    if obs_na.any():
        if use == "all.obs":
            raise ValueError("Missing observations in pcor")
        if use == "everything":
            raise ValueError("NA/NaN/Inf in 'x'")
        data = data[~obs_na]
    n_obs, n_vars = data.shape

    # (row, column) in the output and the two columns of data for each pair
    if y is None:
        value = np.ones((n_vars, n_vars))
        pairs = [(i, j, i, j) for i in range(n_vars) for j in range(i + 1, n_vars)]
    else:
        value = np.ones((y.shape[1], n_x))
        pairs = [(j, i, n_x + j, i) for j in range(y.shape[1]) for i in range(n_x)]
    p_value = np.zeros_like(value)

    for row, col, first, second in pairs:
        others = np.delete(data, [first, second], axis=1)
        pair = data[:, [first, second]]
        u, v = _lm_fit(others, pair)[1].T
        ties = min(np.unique(u).size, np.unique(v).size) < n_obs

        # The estimate and the p-value of R's cor.test()
        if method == "kendall":
            exact = n_obs < 50 and not ties
            kendall = stats.kendalltau(u, v, method="exact" if exact else "asymptotic")
            r, p = kendall.statistic, kendall.pvalue
        elif method == "pearson" or ties or n_obs > 1290:
            if method == "pearson":
                r = np.corrcoef(u, v)[0, 1]
            else:
                r = np.corrcoef(stats.rankdata(u), stats.rankdata(v))[0, 1]
            with np.errstate(divide="ignore"):
                t_stat = r * np.sqrt((n_obs - 2) / (1 - r**2))
            p = 2 * stats.t.sf(np.abs(t_stat), n_obs - 2)
        else:
            # Exact Spearman's test: algorithm AS 89, as R's prho()
            r = np.corrcoef(stats.rankdata(u), stats.rankdata(v))[0, 1]
            q = (n_obs**3 - n_obs) * (1 - r) / 6
            lower = q <= (n_obs**3 - n_obs) / 6
            s_value = np.round(q) + 2 * lower
            b = 1 / n_obs
            z = (6 * (s_value - 1) * b / (n_obs**2 - 1) - 1) * np.sqrt(n_obs - 1)
            if s_value <= 0:
                p = float(not lower)
            elif s_value > n_obs * (n_obs**2 - 1) / 3:
                p = float(lower)
            elif n_obs <= 9:
                perms = np.array(list(permutations(range(n_obs))))
                s_all = np.sum((perms - np.arange(n_obs)) ** 2, axis=1)
                p = np.mean(s_all < s_value if lower else s_all >= s_value)
            else:
                z2 = z**2
                c_1 = 0.0879 + 0.0151 * b
                c_2 = 0.0072 - 0.0831 * b + z2 * b * (0.0131 - 4.6e-4 * z2)
                c_3 = -0.0758 + b * (0.1033 + 0.3932 * b) - z2 * b * (c_1 - z2 * c_2)
                c_4 = 0.2274 + b * (0.2531 + 0.1745 * b) + z2 * c_3
                edgeworth = z * b * c_4 / np.exp(z2 / 2)
                if lower:
                    p = stats.norm.cdf(z) - edgeworth
                else:
                    p = stats.norm.sf(z) + edgeworth
                p = min(max(p, 0.0), 1.0)
            p = min(2 * p, 1.0)
        value[row, col] = r
        p_value[row, col] = p

    if y is None:
        value = np.triu(value) + np.triu(value, 1).T
        p_value = p_value + p_value.T

    return {"value": value, "p.value": p_value, "method": method}


def mcor(
    x: np.ndarray | pd.Series | pd.DataFrame,
    y: np.ndarray | pd.Series | None = None,
    use: Literal[
        "na.or.complete", "complete.obs", "everything", "all.obs"
    ] = "na.or.complete",
) -> McorResult:
    """Calculate multiple correlation coefficient.

    Function returns the multiple correlation coefficient between a set
    of variables and a dependent variable, based on the linear regression of
    ``y`` on ``x``, together with the F-statistic of the regression (testing
    that all its parameters are zero) and the associated p-value.

    As in R, a vector ``x`` that is categorical (or numeric with at most 10
    unique values and more than 10 observations) is treated as a factor (a set
    of dummy variables), a matrix gets an intercept if its first column is not
    one, and a DataFrame is expanded as R's ``model.matrix(~., data=x)``.

    Parameters
    ----------
    x : np.ndarray, pd.Series or pd.DataFrame
        Vector, matrix or DataFrame of independent variables.
    y : np.ndarray or pd.Series
        The dependent variable. A categorical one is replaced by its dummy
        variables, with a warning.
    use : {"na.or.complete", "complete.obs", "everything", "all.obs"}
        What observations to use, as in R's ``cor``. With missing values,
        ``"everything"`` returns NaN, ``"all.obs"`` raises an error, and the
        other two remove the incomplete observations. If fewer than two are
        left, ``"na.or.complete"`` returns NaN and ``"complete.obs"`` raises
        an error.

    Returns
    -------
    McorResult
        The coefficient (``value``), the F-statistic (``statistic``), the
        degrees of freedom of the regression (``df``) and of the residuals
        (``df_residual``), and the p-value (``p_value``).

    Examples
    --------
    >>> x = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    >>> y = np.array([3, 6, 9, 12, 15])
    >>> round(mcor(x, y).value, 4)
    1.0
    """
    if use not in ("na.or.complete", "complete.obs", "everything", "all.obs"):
        raise ValueError(f"Unknown use: {use}")

    x_frame = pd.DataFrame(x)
    y_vector = y if isinstance(y, pd.Series) else pd.Series(np.ravel(np.asarray(y)))
    obs_na = x_frame.isna().any(axis=1).to_numpy() | y_vector.isna().to_numpy()
    if obs_na.any():
        if use == "everything":
            return McorResult(np.nan, np.nan, np.nan, np.nan, np.nan)
        if use == "all.obs" or (use == "complete.obs" and np.sum(~obs_na) < 2):
            raise ValueError("Missing observations in mcor")
        if np.sum(~obs_na) < 2:
            return McorResult(np.nan, np.nan, np.nan, np.nan, np.nan)
        x_frame, y_vector = x_frame[~obs_na], y_vector[~obs_na]

    ones = np.ones((len(y_vector), 1))
    if isinstance(x, pd.DataFrame):
        x_design = np.hstack(
            [ones, pd.get_dummies(x_frame, drop_first=True, dtype=float)]
        )
    elif np.ndim(x) == 2:
        x_design = x_frame.to_numpy(dtype=float)
        if not np.all(x_design[:, 0] == 1):
            x_design = np.hstack([ones, x_design])
    else:
        x_vector = x_frame.iloc[:, 0]
        if not is_numeric_dtype(x_vector) or (
            x_vector.nunique() <= 10 and len(x_vector) > 10
        ):
            x_vector = pd.get_dummies(x_vector, drop_first=True, dtype=float)
        x_design = np.hstack(
            [ones, np.asarray(x_vector, dtype=float).reshape(len(y_vector), -1)]
        )

    if is_numeric_dtype(y_vector):
        y_matrix = y_vector.to_numpy(dtype=float).reshape(-1, 1)
    else:
        warnings.warn(
            "The y variable should be numeric in order for this to work! "
            "The reported value might be meaningless",
            stacklevel=2,
        )
        y_matrix = pd.get_dummies(y_vector, dtype=float).to_numpy()

    _, residuals, rank = _lm_fit(x_design, y_matrix)
    df_residual = residuals.size - rank
    df = rank - 1
    # abs() as in R, against the rounding issues
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.sqrt(
            np.abs(1 - np.sum(residuals**2) / np.sum((y_matrix - y_matrix.mean()) ** 2))
        )
        statistic = (value**2 / df) / ((1 - value**2) / df_residual)
    return McorResult(
        float(value),
        float(statistic),
        df,
        df_residual,
        float(stats.f.sf(statistic, df, df_residual)),
    )


def cramer(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    use: Literal[
        "na.or.complete", "complete.obs", "everything", "all.obs"
    ] = "na.or.complete",
    unbiased: bool = True,
) -> CramerResult:
    """Calculate Cramer's V for categorical variables.

    Function calculates Cramer's V for two categorical variables based on
    their contingency table, and returns the statistics of the Chi-squared
    test with the null hypothesis of independence of the two variables (with
    Yates' continuity correction for a 2x2 table, as R's ``chisq.test``).

    Parameters
    ----------
    x : np.ndarray or pd.Series
        First categorical variable.
    y : np.ndarray or pd.Series
        Second categorical variable.
    use : {"na.or.complete", "complete.obs", "everything", "all.obs"}
        What observations to use, as in R's ``cor``. With missing values,
        ``"everything"`` returns NaN, ``"all.obs"`` raises an error, and the
        other two remove the incomplete observations. If fewer than two are
        left, ``"na.or.complete"`` returns NaN and ``"complete.obs"`` raises
        an error.
    unbiased : bool, default=True
        Whether to calculate the version of Cramer's V with the small sample
        correction (Bergsma, 2013) or the biased one.

    Returns
    -------
    CramerResult
        Cramer's V (``value``), the Chi-squared statistic (``statistic``), its
        p-value (``p_value``) and degrees of freedom (``df``).

    References
    ----------
    Wicher Bergsma (2013), A bias-correction for Cramer's V and Tschuprow's T.
    Journal of the Korean Statistical Society, 42, pp. 323-328.

    Examples
    --------
    >>> am = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    >>> gear = np.array([4, 4, 4, 3, 3, 3, 3, 4, 4, 5, 5, 3])
    >>> result = cramer(am, gear)
    """
    if use not in ("na.or.complete", "complete.obs", "everything", "all.obs"):
        raise ValueError(f"Unknown use: {use}")

    x, y = pd.Series(x), pd.Series(y)
    for variable, name in ((x, "x"), (y, "y")):
        if is_numeric_dtype(variable) and variable.nunique(dropna=False) > 10:
            warnings.warn(
                f"It seems that {name} is numeric, not categorical. Other measures "
                "of association might be more informative.",
                stacklevel=2,
            )

    obs_na = x.isna().to_numpy() | y.isna().to_numpy()
    if obs_na.any():
        if use == "everything":
            return CramerResult(np.nan, np.nan, np.nan, np.nan)
        if use == "all.obs" or (use == "complete.obs" and np.sum(~obs_na) < 2):
            raise ValueError("Missing observations in cramer")
        if np.sum(~obs_na) < 2:
            return CramerResult(np.nan, np.nan, np.nan, np.nan)
        x, y = x[~obs_na], y[~obs_na]

    table = pd.crosstab(x.to_numpy(), y.to_numpy()).to_numpy()
    chi_test = stats.chi2_contingency(table)
    n_obs = table.sum()
    n_rows, n_cols = table.shape
    phi2 = chi_test.statistic / n_obs
    # The division by zero for very small samples gives NaN or Inf, as in R
    with np.errstate(divide="ignore", invalid="ignore"):
        if unbiased:
            phi2 = max(phi2 - (n_cols - 1) * (n_rows - 1) / (n_obs - 1), 0)
            n_cols = n_cols - (n_cols - 1) ** 2 / (n_obs - 1)
            n_rows = n_rows - (n_rows - 1) ** 2 / (n_obs - 1)
        value = np.sqrt(np.float64(phi2) / min(n_cols - 1, n_rows - 1))
    return CramerResult(
        float(value), float(chi_test.statistic), float(chi_test.pvalue), chi_test.dof
    )


def determination(
    xreg: np.ndarray | pd.DataFrame,
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
    determination > 0.9 (which corresponds to VIF > 10). Incomplete
    observations are removed.

    Parameters
    ----------
    xreg : np.ndarray or pd.DataFrame
        Data frame or matrix containing the exogenous variables. With
        categorical variables, the coefficients are the squares of the multiple
        correlations of :func:`mcor`, as in R.
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
    array([0.6497, 0.2872, 0.5649])
    """
    from .selection import stepwise

    frame = pd.DataFrame(xreg)
    n_series, n_variables = frame.shape

    if n_series <= n_variables and bruteforce:
        warnings.warn(
            "The number of variables is larger than the number of observations. "
            "Sink regression cannot be constructed. Using stepwise.",
            RuntimeWarning,
        )
        bruteforce = False

    if n_variables <= 1:
        return np.array([0.0])

    # With categorical variables, the squares of the multiple correlations
    if bruteforce and not all(is_numeric_dtype(col) for _, col in frame.items()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.array(
                [
                    mcor(
                        frame.iloc[:, np.arange(n_variables) != i], frame.iloc[:, i]
                    ).value
                    ** 2
                    for i in range(n_variables)
                ]
            )

    xreg = frame.to_numpy(dtype=float)
    xreg = xreg[~np.isnan(xreg).any(axis=1)]
    vector_correlations_multiple = np.full(n_variables, np.nan)

    if bruteforce:
        cor_matrix = np.corrcoef(xreg, rowvar=False)
        for i in range(n_variables):
            others = np.delete(np.arange(n_variables), i)
            try:
                factor = linalg.cho_factor(
                    cor_matrix[np.ix_(others, others)], check_finite=False
                )
                vector_correlations_multiple[i] = cor_matrix[i, others] @ (
                    linalg.cho_solve(factor, cor_matrix[others, i], check_finite=False)
                )
            except linalg.LinAlgError:
                vector_correlations_multiple[i] = 1.0
    else:
        names = [f"x{j + 1}" for j in range(n_variables)]
        for i in range(n_variables):
            # As in R: the variable i swaps places with the first one
            swapped = xreg.copy()
            swapped[:, [0, i]] = swapped[:, [i, 0]]
            model = stepwise(pd.DataFrame(swapped, columns=names), silent=True)
            actuals = model.actuals
            vector_correlations_multiple[i] = 1 - np.sum(model.residuals**2) / np.sum(
                (actuals - np.mean(actuals)) ** 2
            )

    return vector_correlations_multiple


def association(
    x: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.DataFrame | None = None,
    use: Literal[
        "na.or.complete", "complete.obs", "everything", "all.obs"
    ] = "na.or.complete",
    method: str = "auto",
) -> dict:
    """Calculate measures of association.

    Function returns the matrix of measures of association for different types
    of variables. With ``method="auto"``, a variable that is not numeric or has
    at most 10 unique values is treated as categorical, and the measure for
    each pair is Pearson's correlation (both numeric, on the pairwise complete
    observations), Cramer's V (both categorical, :func:`cramer`) or the
    multiple correlation (one of each, :func:`mcor`). The other methods use
    the same measure for all pairs, replacing the categorical variables by the
    numbers of their levels (with a warning) for the correlations.

    ``assoc`` is a short name for ``association``.

    Parameters
    ----------
    x : np.ndarray or pd.DataFrame
        DataFrame or matrix.
    y : np.ndarray or pd.DataFrame, optional
        The numerical variable.
    use : {"na.or.complete", "complete.obs", "everything", "all.obs"}
        What observations to use, as in R. With missing values, ``"all.obs"``
        raises an error, as does ``"complete.obs"`` if fewer than two complete
        observations are left; otherwise it is passed to :func:`cramer` and
        :func:`mcor`.
    method : str, default="auto"
        Method to use: "auto", "pearson", "spearman", "kendall", "cramer".
        "auto" selects based on variable types.

    Returns
    -------
    dict
        Dictionary containing:

        - value: matrix of association coefficients (n_y x n_x if y is
          provided)
        - p.value: p-values
        - type: matrix of types of measures used

    Examples
    --------
    >>> x = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    >>> result = association(x)
    """
    if method not in ("auto", "pearson", "spearman", "kendall", "cramer"):
        raise ValueError(f"Unknown method: {method}")
    if use not in ("na.or.complete", "complete.obs", "everything", "all.obs"):
        raise ValueError(f"Unknown use: {use}")

    frames = [pd.DataFrame(v).reset_index(drop=True) for v in (x, y) if v is not None]
    n_x = frames[0].shape[1]
    if y is None and n_x == 1:
        return {"value": 1.0, "p.value": 1.0, "type": "none"}
    names = [name for frame in frames for name in frame.columns]
    data = pd.concat(frames, axis=1, ignore_index=True)
    n_vars = data.shape[1]

    if method == "auto":
        numeric = np.array(
            [
                is_numeric_dtype(col) and col.nunique(dropna=False) > 10
                for _, col in data.items()
            ]
        )
    else:
        numeric = np.full(n_vars, method != "cramer")
    if method not in ("auto", "cramer") and not all(
        is_numeric_dtype(col) for _, col in data.items()
    ):
        warnings.warn(
            "Some of the variables are in categorical scales. Using "
            f'"{method}" correlation for the calculations might be meaningless!',
            stacklevel=2,
        )
        # The categorical variables as the numbers of their levels, as R
        for name, col in data.items():
            if not is_numeric_dtype(col):
                codes = col.astype("category").cat.codes
                data[name] = codes.where(codes >= 0) + 1
    cor_method = "pearson" if method in ("auto", "cramer") else method

    variables_na = data.isna().any(axis=0).to_numpy()
    complete = ~data.isna().any(axis=1).to_numpy()
    if variables_na.any() and (
        use == "all.obs" or (use == "complete.obs" and complete.sum() < 2)
    ):
        missing = ", ".join(str(names[i]) for i in np.flatnonzero(variables_na))
        raise ValueError(f"Missing observations in the variables: {missing}")

    # (row, column) in the output and the two columns of data for each pair
    if y is None:
        shape = (n_vars, n_vars)
        pairs = [(i, j, i, j) for i in range(n_vars) for j in range(i + 1, n_vars)]
    else:
        shape = (n_vars - n_x, n_x)
        pairs = [(i, j, n_x + i, j) for i in range(n_vars - n_x) for j in range(n_x)]
    value = np.ones(shape)
    p_value = np.zeros(shape)
    type_matrix = np.full(shape, "none", dtype=object)

    for row, col, first, second in pairs:
        if numeric[first] and numeric[second]:
            # cor.test() on the pairwise complete observations
            pair = data.iloc[:, [first, second]].to_numpy(dtype=float)
            pair = pair[~np.isnan(pair).any(axis=1)]
            cor_output = pcor(pair, method=cor_method)  # type: ignore[arg-type]
            value[row, col] = cor_output["value"][0, 1]
            p_value[row, col] = cor_output["p.value"][0, 1]
            type_matrix[row, col] = cor_method
            continue
        if not numeric[first] and not numeric[second]:
            output: CramerResult | McorResult = cramer(
                data.iloc[:, first], data.iloc[:, second], use=use
            )
            type_matrix[row, col] = "cramer"
        else:
            # Multiple correlation of the numeric on the categorical variable
            categorical, numerical = (
                (second, first) if numeric[first] else (first, second)
            )
            output = mcor(data.iloc[:, categorical], data.iloc[:, numerical], use=use)
            type_matrix[row, col] = "mcor"
        value[row, col] = output.value
        p_value[row, col] = output.p_value

    if y is None:
        lower = np.tril_indices(n_vars, -1)
        value[lower] = value.T[lower]
        p_value[lower] = p_value.T[lower]
        type_matrix[lower] = type_matrix.T[lower]

    return {"value": value, "p.value": p_value, "type": type_matrix}


assoc = association


def determ(model) -> np.ndarray:
    """Coefficients of determination of the regressors of a model.

    Function returns the coefficients of determination of the variables of a
    fitted :class:`~greybox.ALM` (the columns of its design matrix, without
    the intercept, as R's ``determ.alm``), or of the provided data (as R's
    ``determ.default``). See :func:`determination` for details.

    Parameters
    ----------
    model : ALM, np.ndarray or pd.DataFrame
        The fitted model, or the data for :func:`determination`.

    Returns
    -------
    np.ndarray
        Vector of determination coefficients, one for each variable.

    Examples
    --------
    >>> from greybox import ALM, formula, mtcars
    >>> y, X = formula("mpg ~ wt + hp + disp", mtcars)
    >>> result = determ(ALM().fit(X, y))
    """
    from .alm import ALM

    if isinstance(model, ALM):
        X = model._X_train_
        return determination(X[:, 1:] if np.all(X[:, 0] == 1) else X)
    return determination(model)
