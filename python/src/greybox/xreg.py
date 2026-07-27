"""Exogenous variables manipulation functions.

This module provides functions for transforming and expanding exogenous
variables for use in regression models.
"""

import re
from typing import Literal

import numpy as np


def xreg_transformer(
    xreg: np.ndarray,
    functions: list[str] | None = None,
    silent: bool = True,
) -> np.ndarray:
    """Transform exogenous variables.

    Function transforms each variable in the provided matrix or vector,
    producing non-linear values, depending on the selected pool of functions.

    Parameters
    ----------
    xreg : np.ndarray
        Vector / matrix containing variables that need to be transformed.
    functions : list of str, optional
        Vector of names for functions used. Options: "log", "exp", "inv",
        "sqrt", "square". If None, uses all functions.
    silent : bool, default=True
        If False, then progress is printed. Otherwise the function
        won't print anything.

    Returns
    -------
    np.ndarray
        Matrix with the transformed and the original variables.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> np.round(xreg_transformer(x, functions=["log", "sqrt"]), 4)
    array([[1.    , 0.    , 1.    ],
           [2.    , 0.6931, 1.4142],
           [3.    , 1.0986, 1.7321],
           [4.    , 1.3863, 2.    ],
           [5.    , 1.6094, 2.2361]])
    """
    if functions is None:
        functions = ["log", "exp", "inv", "sqrt", "square"]

    valid_funcs = ["log", "exp", "inv", "sqrt", "square"]
    functions = [f for f in functions if f in valid_funcs]

    if len(functions) == 0:
        raise ValueError(
            "functions parameter does not contain any valid function name. "
            "Please provide something from: log, exp, inv, sqrt, square"
        )

    # Dedupe while preserving the requested order (set() would make the
    # resulting column order nondeterministic across processes).
    functions = list(dict.fromkeys(functions))

    if not silent:
        print("Preparing matrices...    ")

    xreg = np.asarray(xreg, dtype=float)

    if xreg.ndim == 1:
        xreg = xreg.reshape(-1, 1)

    obs, n_vars = xreg.shape
    n_functions = len(functions)

    xreg_new = np.zeros((obs, (n_functions + 1) * n_vars))

    for j in range(n_vars):
        xreg_new[:, j] = xreg[:, j]
        col_start = n_vars + j * n_functions

        for func in functions:
            func_idx = functions.index(func)
            target_col = col_start + func_idx

            if func == "log":
                xreg_new[:, target_col] = np.log(np.maximum(xreg[:, j], 1e-10))
            elif func == "exp":
                xreg_new[:, target_col] = np.exp(np.minimum(xreg[:, j], 700))
            elif func == "inv":
                xreg_new[:, target_col] = 1.0 / (xreg[:, j] + 1e-10)
            elif func == "sqrt":
                xreg_new[:, target_col] = np.sqrt(np.maximum(xreg[:, j], 0))
            elif func == "square":
                xreg_new[:, target_col] = xreg[:, j] ** 2

    return xreg_new


def xreg_multiplier(xreg: np.ndarray, silent: bool = True) -> np.ndarray:
    """Generate cross-products of exogenous variables.

    Function generates the cross-products of the provided exogenous variables.
    This might be useful when introducing interactions between dummy and
    continuous variables.

    Parameters
    ----------
    xreg : np.ndarray
        Matrix containing variables that need to be expanded. Must have
        at least two columns.
    silent : bool, default=True
        If False, then progress is printed. Otherwise the function
        won't print anything.

    Returns
    -------
    np.ndarray
        Matrix with the cross-products and the original variables.

    Examples
    --------
    >>> x = np.array([[1, 2], [3, 4], [5, 6]])
    >>> xreg_multiplier(x)
    array([[ 1.,  2.,  2.],
           [ 3.,  4., 12.],
           [ 5.,  6., 30.]])
    """
    xreg = np.asarray(xreg, dtype=float)

    if xreg.ndim == 1:
        raise ValueError("xreg must be a matrix with at least 2 columns")

    n_obs, n_vars = xreg.shape

    if n_vars < 2:
        raise ValueError("xreg must have at least 2 columns")

    if not silent:
        print("Preparing matrices...    ")

    n_combinations = 0
    for i in range(n_vars):
        for j in range(i, n_vars):
            if i != j:
                n_combinations += 1

    xreg_new = np.zeros((n_obs, n_combinations + n_vars))

    xreg_new[:, :n_vars] = xreg

    k = 0
    for i in range(n_vars):
        for j in range(i, n_vars):
            if i == j:
                continue
            xreg_new[:, n_vars + k] = xreg[:, i] * xreg[:, j]
            k += 1

    return xreg_new


def xreg_expander(
    xreg: np.ndarray,
    lags: list[int] | None = None,
    silent: bool = True,
    gaps: Literal["auto", "NAs", "zero", "naive", "extrapolate"] = "auto",
) -> np.ndarray:
    """Expand exogenous variables with lags and leads.

    Function expands the provided matrix or vector of variables, producing
    values with lags and leads specified by lags parameter.

    Parameters
    ----------
    xreg : np.ndarray
        Vector / matrix containing variables that need to be expanded.
    lags : list of int, optional
        Vector of lags / leads. Negative values mean lags, positive ones
        mean leads. If None, uses -1, 0, 1 (seasonal lags if ts object).
    silent : bool, default=True
        If False, then progress is printed. Otherwise the function
        won't print anything.
    gaps : str, default="auto"
        Defines how to fill in the gaps in the data:
        - "NAs": leave missing values
        - "zero": substitute them by zeroes
        - "naive": use the last / first actual value
        - "extrapolate": use linear extrapolation
        - "auto": let the function select between "extrapolate" and "naive"

    Returns
    -------
    np.ndarray
        Matrix with the expanded variables.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> xreg_expander(x, lags=[-1, 0, 1])
    array([[1., 1., 2.],
           [2., 1., 3.],
           [3., 2., 4.],
           [4., 3., 5.],
           [5., 4., 5.]])
    """
    xreg = np.asarray(xreg, dtype=float)

    if xreg.ndim == 1:
        xreg = xreg.reshape(-1, 1)

    n_obs, n_vars = xreg.shape

    if lags is None:
        lags = [-1, 0, 1]

    lags = [lag for lag in lags if lag != 0]

    if len(lags) == 0:
        return xreg

    # Deduplicate while preserving intent, then sort deterministically:
    # negative lags by abs ascending (−1, −2, …), then positive leads (1, 2, …)
    seen: set = set()
    unique_lags = [
        v
        for v in lags
        if not (v in seen or seen.add(v))  # type: ignore[func-returns-value]
    ]
    neg_lags = sorted([v for v in unique_lags if v < 0], key=abs)
    pos_lags = sorted([v for v in unique_lags if v > 0])
    lags = neg_lags + pos_lags

    if not silent:
        print("Preparing matrices...    ")

    n_lags = len(lags)
    lag_mat = np.full((n_obs, n_vars * n_lags), np.nan)

    for c_idx in range(n_vars):
        for lag_idx, lag in enumerate(lags):
            t_col = c_idx * n_lags + lag_idx

            if lag < 0:
                shift = abs(lag)
                if shift < n_obs:
                    lag_mat[shift:, t_col] = xreg[: n_obs - shift, c_idx]
            else:
                if lag < n_obs:
                    lag_mat[: n_obs - lag, t_col] = xreg[lag:n_obs, c_idx]

    if gaps == "zero":
        lag_mat = np.nan_to_num(lag_mat, nan=0.0)
    elif gaps == "naive":
        for col_idx in range(lag_mat.shape[1]):
            col = lag_mat[:, col_idx]
            first_valid = np.argmax(~np.isnan(col))
            if first_valid > 0:
                col[:first_valid] = col[first_valid]
            last_valid = n_obs - 1 - np.argmax(~np.isnan(col[::-1]))
            if last_valid < n_obs - 1:
                col[last_valid + 1 :] = col[last_valid]
            lag_mat[:, col_idx] = col
    elif gaps in ("auto", "extrapolate"):
        for col_idx in range(lag_mat.shape[1]):
            col = lag_mat[:, col_idx]
            valid_mask = ~np.isnan(col)
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                if valid_indices[0] > 0:
                    first_val = col[valid_indices[0]]
                    col[: valid_indices[0]] = first_val
                if valid_indices[-1] < n_obs - 1:
                    last_val = col[valid_indices[-1]]
                    col[valid_indices[-1] + 1 :] = last_val

    # Prepend original columns (matches R's xregExpander output layout)
    return np.hstack([xreg, lag_mat])


def temporal_dummy(
    x,
    freq: int | None = None,
    h: int = 0,
) -> np.ndarray:
    """Generate dummy variables for temporal data.

    Function generates dummy variables for months, days, hours, etc.
    based on the provided time index.

    Parameters
    ----------
    x : array-like
        Vector of dates or time indices.
    freq : int, optional
        Frequency of the data. If not provided, attempts to infer.
    h : int, default=0
        Number of future observations to generate dummies for.

    Returns
    -------
    np.ndarray
        Matrix with dummy variables.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range('2020-01-01', periods=12, freq='ME')
    >>> dummies = temporal_dummy(dates, freq=12)
    >>> dummies.shape
    (12, 12)
    >>> dummies[0]
    array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """
    x = np.asarray(x)
    n_obs = len(x)

    if freq is None:
        freq = 12

    total_obs = n_obs + h

    if freq == 12:
        dummies = np.zeros((total_obs, 12))
        for i in range(total_obs):
            month = i % 12
            dummies[i, month] = 1
        return dummies
    elif freq == 4:
        dummies = np.zeros((total_obs, 4))
        for i in range(total_obs):
            quarter = i % 4
            dummies[i, quarter] = 1
        return dummies
    elif freq == 7:
        dummies = np.zeros((total_obs, 7))
        for i in range(total_obs):
            day = i % 7
            dummies[i, day] = 1
        return dummies
    elif freq == 24:
        dummies = np.zeros((total_obs, 24))
        for i in range(total_obs):
            hour = i % 24
            dummies[i, hour] = 1
        return dummies
    else:
        dummies = np.zeros((total_obs, freq))
        for i in range(total_obs):
            idx = i % freq
            dummies[i, idx] = 1
        return dummies


def B(
    x: np.ndarray,
    k: int,
    gaps: Literal["auto", "NAs", "zero", "naive", "extrapolate"] = "auto",
) -> np.ndarray:
    """Backshift operator: lag (k>0) or lead (k<0) of x.

    Positive k creates lag-k (past values); negative k creates lead-abs(k)
    (future values); k=0 returns x unchanged. Gaps at boundaries are
    filled per the gaps strategy.

    Parameters
    ----------
    x : array-like of shape (n,)
    k : int
        Lag order. Positive = lag (past values), negative = lead (future).
    gaps : str, default "auto"
        Boundary fill strategy passed to xreg_expander.

    Returns
    -------
    np.ndarray of shape (n,)
    """
    x = np.asarray(x, dtype=float)
    if k == 0:
        return x
    return xreg_expander(x, lags=[-k], gaps=gaps)[:, 1]


def _dyn_mult_calc(phi: np.ndarray, beta: np.ndarray, h: int) -> np.ndarray:
    """Compute dynamic multipliers for ARDL(p, m).

    Recurrence:
        m_0 = beta_0
        m_s = beta_s + sum_{i=1}^{min(p,s)} phi[i-1] * m[s-i]

    beta_s is treated as 0 for s >= len(beta).
    """
    p = len(phi)
    c = np.zeros(h)
    if h < 1:
        return c
    c[0] = beta[0] if len(beta) > 0 else 0.0
    for s in range(1, h):
        beta_s = beta[s] if s < len(beta) else 0.0
        acc = sum(phi[i - 1] * c[s - i] for i in range(1, min(p, s) + 1))
        c[s] = beta_s + acc
    return c


def _get_betas(model, parm: str) -> np.ndarray:
    """Extract sorted [beta_0, beta_1, ...] for variable parm from ALM model.

    beta_0 = coefficient for column named exactly `parm` (contemporaneous).
    beta_k = coefficient for column named `B(parm,k)` (distributed lag k).
    Results are sorted by lag order and returned as a numpy array.

    Raises
    ------
    ValueError
        If parm is not found as a contemporaneous or lagged term.
    """
    names = ["(Intercept)"] + list(
        model._feature_names
        if model._feature_names is not None
        else [f"x{i + 1}" for i in range(len(model._coef))]
    )
    coefs = np.concatenate([[model.intercept_], model._coef])
    coef_dict = dict(zip(names, coefs))

    lag_pattern = re.compile(rf"^B\({re.escape(parm)},\s*(\d+)\)$")
    betas: dict = {}

    if parm in coef_dict:
        betas[0] = coef_dict[parm]
    for name, val in coef_dict.items():
        m = lag_pattern.match(name)
        if m:
            betas[int(m.group(1))] = val

    if not betas:
        raise ValueError(f'The parameter "{parm}" is not found in the model.')

    return np.array([betas[k] for k in sorted(betas)])


def multipliers(model, parm: str, h: int = 10) -> dict:
    """Compute dynamic multipliers for an ARDL model.

    Combines distributed lag coefficients (B(parm, k) terms) with the
    ARI polynomial from the model to produce impulse-response multipliers
    over horizon h.

    Parameters
    ----------
    model : ALM
        Fitted ALM model containing parm (and optionally B(parm, k) columns
        and/or ARIMA orders).
    parm : str
        Variable name as it appears in the design matrix (column header).
    h : int, default 10
        Forecast horizon.

    Returns
    -------
    dict
        {"h1": m1, "h2": m2, ..., "hh": mh} of dynamic multipliers.

    Raises
    ------
    ValueError
        If parm is not found in the model.
    """
    names = ["(Intercept)"] + list(
        model._feature_names
        if model._feature_names is not None
        else [f"x{i + 1}" for i in range(len(model._coef))]
    )
    word_pat = re.compile(r"\b" + re.escape(parm) + r"\b")
    if not any(word_pat.search(n) for n in names):
        raise ValueError(f'The parameter "{parm}" is not found in the model.')

    phi = (
        np.array(list(model.arima_polynomial_.values()))
        if model.arima_polynomial_ is not None
        else np.array([0.0])
    )
    betas = _get_betas(model, parm)
    dm = _dyn_mult_calc(phi, betas, h)
    return {f"h{i + 1}": float(dm[i]) for i in range(h)}
