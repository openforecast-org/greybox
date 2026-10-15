"""Exogenous variables manipulation functions.

This module provides functions for transforming and expanding exogenous
variables for use in regression models.
"""

import importlib
import re
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd


def xreg_transformer(
    xreg: np.ndarray | pd.DataFrame | pd.Series,
    functions: list[str] | None = None,
    silent: bool = True,
) -> np.ndarray | pd.DataFrame:
    """Transform exogenous variables.

    Function transforms each variable in the provided matrix or vector,
    producing non-linear values, depending on the selected pool of functions.

    Parameters
    ----------
    xreg : np.ndarray, pd.DataFrame or pd.Series
        Vector / matrix containing variables that need to be transformed.
    functions : list of str, optional
        Vector of names for functions used. Options: "log", "exp", "inv",
        "sqrt", "square". If None, uses all functions.
    silent : bool, default=True
        If False, then progress is printed. Otherwise the function
        won't print anything.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Matrix with the transformed and the original variables: a DataFrame with
        R's column names (``x``, ``log_x``, ...) and the index of the input, if it
        is a DataFrame or a Series (named ``x`` if it has no name).

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
    if any(f not in valid_funcs for f in functions):
        warnings.warn(
            "An unknown function type specified. We will drop it from the list",
            stacklevel=2,
        )
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

    if isinstance(xreg, pd.Series):
        xreg = xreg.to_frame("x" if xreg.name is None else xreg.name)
    index = xreg.index if isinstance(xreg, pd.DataFrame) else None
    names = [str(name) for name in getattr(xreg, "columns", [])]
    xreg = np.asarray(xreg, dtype=float)

    if xreg.ndim == 1:
        xreg = xreg.reshape(-1, 1)

    obs, n_vars = xreg.shape
    n_functions = len(functions)

    xreg_new = np.zeros((obs, (n_functions + 1) * n_vars))

    # Each variable followed by its transformations, as R; -Inf, Inf and NaN as R,
    # which warns only on the NaNs
    with np.errstate(divide="ignore", over="ignore"):
        for j in range(n_vars):
            col_start = j * (n_functions + 1)
            xreg_new[:, col_start] = xreg[:, j]

            for func_idx, func in enumerate(functions):
                target_col = col_start + func_idx + 1

                if func == "log":
                    xreg_new[:, target_col] = np.log(xreg[:, j])
                elif func == "exp":
                    xreg_new[:, target_col] = np.exp(xreg[:, j])
                elif func == "inv":
                    xreg_new[:, target_col] = 1.0 / xreg[:, j]
                elif func == "sqrt":
                    xreg_new[:, target_col] = np.sqrt(xreg[:, j])
                elif func == "square":
                    xreg_new[:, target_col] = xreg[:, j] ** 2

    if index is None:
        return xreg_new
    columns = [n for name in names for n in [name] + [f"{f}_{name}" for f in functions]]
    return pd.DataFrame(xreg_new, index=index, columns=columns)


def xreg_multiplier(
    xreg: np.ndarray | pd.DataFrame, silent: bool = True
) -> np.ndarray | pd.DataFrame:
    """Generate cross-products of exogenous variables.

    Function generates the cross-products of the provided exogenous variables.
    This might be useful when introducing interactions between dummy and
    continuous variables.

    Parameters
    ----------
    xreg : np.ndarray or pd.DataFrame
        Matrix containing variables that need to be expanded. Must have
        at least two columns.
    silent : bool, default=True
        If False, then progress is printed. Otherwise the function
        won't print anything.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Matrix with the cross-products and the original variables: a DataFrame
        with R's column names (``a``, ``b``, ``a_b``, ...) and the index of the
        input, if it is a DataFrame.

    Examples
    --------
    >>> x = np.array([[1, 2], [3, 4], [5, 6]])
    >>> xreg_multiplier(x)
    array([[ 1.,  2.,  2.],
           [ 3.,  4., 12.],
           [ 5.,  6., 30.]])
    """
    index = xreg.index if isinstance(xreg, pd.DataFrame) else None
    names = [str(name) for name in getattr(xreg, "columns", [])]
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

    if index is None:
        return xreg_new
    columns = names + [
        f"{names[i]}_{names[j]}" for i in range(n_vars) for j in range(i + 1, n_vars)
    ]
    return pd.DataFrame(xreg_new, index=index, columns=columns)


def xreg_expander(
    xreg: np.ndarray | pd.DataFrame | pd.Series,
    lags: list[int] | None = None,
    silent: bool = True,
    gaps: Literal["auto", "NAs", "zero", "naive", "extrapolate"] = "auto",
) -> np.ndarray | pd.DataFrame:
    """Expand exogenous variables with lags and leads.

    Function expands the provided matrix or vector of variables, producing
    values with lags and leads specified by lags parameter.

    Parameters
    ----------
    xreg : np.ndarray, pd.DataFrame or pd.Series
        Vector / matrix containing variables that need to be expanded.
    lags : list of int, optional
        Vector of lags / leads. Negative values mean lags, positive ones
        mean leads. If None, uses -1, 0, 1, R's default for the data without
        frequency.
    silent : bool, default=True
        If False, then progress is printed. Otherwise the function
        won't print anything.
    gaps : str, default="auto"
        Defines how to fill in the gaps in the data:

        - "NAs": leave missing values
        - "zero": substitute them by zeroes
        - "naive": use the last / first actual value
        - "extrapolate": forecast the leads and backcast the lags with smooth's
          ``ADAM`` (``OM`` for the binary variables) if smooth is installed, and
          take the naive values otherwise, as R
        - "auto": "extrapolate", or "naive" for series longer than 10000

    Returns
    -------
    np.ndarray or pd.DataFrame
        Matrix with the expanded variables: a DataFrame with R's column names
        (``x``, ``xLag1``, ``xLead1``, ...) and the index of the input, if it is a
        DataFrame or a Series (named ``x`` if it has no name).

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> xreg_expander(x, lags=[-1, 0, 1], gaps="naive")
    array([[1., 1., 2.],
           [2., 1., 3.],
           [3., 2., 4.],
           [4., 3., 5.],
           [5., 4., 5.]])
    """
    if isinstance(xreg, pd.Series):
        xreg = xreg.to_frame("x" if xreg.name is None else xreg.name)
    if lags is None:
        lags = [-1, 0, 1]

    lags = [lag for lag in lags if lag != 0]

    if len(lags) == 0:
        warnings.warn("You have not specified any leads or lags.", stacklevel=2)
        if isinstance(xreg, pd.DataFrame):
            return xreg
        xreg = np.asarray(xreg, dtype=float)
        return xreg.reshape(-1, 1) if xreg.ndim == 1 else xreg

    index = xreg.index if isinstance(xreg, pd.DataFrame) else None
    names = [str(name) for name in getattr(xreg, "columns", [])]
    xreg = np.asarray(xreg, dtype=float)

    if xreg.ndim == 1:
        xreg = xreg.reshape(-1, 1)

    n_obs, n_vars = xreg.shape

    # The lags, then the leads, each in the given order, as R
    lags = [v for v in lags if v < 0] + [v for v in lags if v > 0]

    if not silent:
        print("Preparing matrices...    ")

    n_lags = len(lags)
    xreg_new = np.full((n_obs, n_vars * (n_lags + 1)), np.nan)
    max_lag = max([-lag for lag in lags if lag < 0], default=0)
    max_lead = max([lag for lag in lags if lag > 0], default=0)

    # As R, "auto" extrapolates with smooth's ADAM when smooth is installed, and takes
    # the first / last values for the long series
    if gaps == "auto" and n_obs > 10000:
        gaps = "naive"
    smooth: Any = None
    if gaps in ("auto", "extrapolate"):
        try:
            smooth = importlib.import_module("smooth")
        except ImportError:
            pass

    for c_idx in range(n_vars):
        x = xreg[:, c_idx]
        head = np.full(max_lag, np.nan)
        tail = np.full(max_lead, np.nan)
        if gaps == "zero":
            head[:], tail[:] = 0.0, 0.0
        elif gaps != "NAs":
            head[:], tail[:] = x[0], x[-1]
        if smooth is not None:
            # ADAM forecasts the leads and backcasts the lags on the reversed series,
            # with the model of the leads if there are any, as R; the binary variables
            # take the occurrence model OM, MNN with the inverse odds ratio
            binary = bool(np.all(np.isin(x, (0.0, 1.0))))
            if binary:
                model_class = smooth.OM
                kwargs: dict[str, Any] = {
                    "model": "MNN",
                    "occurrence": "inverse-odds-ratio",
                }
            else:
                model_class = smooth.ADAM
                kwargs = {"occurrence": "auto"}
            lead_model = None
            if max_lead > 0:
                lead_model = model_class(**kwargs).fit(x)
                tail = np.asarray(lead_model.predict(h=max_lead).mean, dtype=float)
            if max_lag > 0:
                if lead_model is not None:
                    kwargs = {
                        "model": lead_model.model_type,
                        "persistence": lead_model.persistence_vector,
                    }
                    if binary:
                        kwargs["occurrence"] = "inverse-odds-ratio"
                    else:
                        om = lead_model.om_model
                        kwargs["occurrence"] = "none" if om is None else om
                lag_model = model_class(**kwargs).fit(x[::-1])
                head = np.asarray(lag_model.predict(h=max_lag).mean, dtype=float)[::-1]
        extended = np.concatenate([head, x, tail])
        # Each variable followed by its lags and leads, as R
        block = c_idx * (n_lags + 1)
        xreg_new[:, block] = x
        for lag_idx, lag in enumerate(lags):
            xreg_new[:, block + 1 + lag_idx] = extended[
                max_lag + lag : max_lag + lag + n_obs
            ]

    if index is None:
        return xreg_new
    columns = [
        n
        for name in names
        for n in [name]
        + [f"{name}Lag{-lag}" if lag < 0 else f"{name}Lead{lag}" for lag in lags]
    ]
    return pd.DataFrame(xreg_new, index=index, columns=columns)


def temporal_dummy(
    x: Any,
    type: str = "month",
    of: str = "year",
    factors: bool = False,
    h: int = 0,
    freq: int | None = None,
) -> np.ndarray | pd.DataFrame | pd.Series | pd.Categorical | None:
    """Generate dummy variables for the provided seasonality type.

    Function generates the matrix of dummy variables for the months / weeks / days /
    hours / minutes / seconds of year / month / week / day / hour / minute, as R's
    ``temporaldummy()``.

    The dates are taken from a ``pd.DatetimeIndex``, a ``pd.Series`` of datetimes,
    a ``np.datetime64`` array (R's ``Date`` for the units of days and longer,
    ``POSIXct`` otherwise) or the ``DatetimeIndex`` of a ``pd.Series`` /
    ``pd.DataFrame`` (R's ``zoo``). Any other input is treated as R's numeric
    vector: the dummies follow the typical values (e.g. 30 days in a month), from
    the first one onwards. The weeks are those of ISO 8601, the hours / minutes of
    year, quarter, month and week count from 1, those of day / hour from 0, as R.

    Parameters
    ----------
    x : array-like, pd.DatetimeIndex, pd.Series or pd.DataFrame
        Vector of dates, or the object to take the dates (or the length) from.
    type : str, default="month"
        What type of frequency to produce: "month", "quarter", "week", "day",
        "hour", "halfhour", "minute" or "second".
    of : str, default="year"
        The frequency of what is needed: "year", "quarter", "month", "week",
        "day", "hour" or "minute". E.g. ``type="day"`` and ``of="month"`` give
        the days of month.
    factors : bool, default=False
        If True, return the categorical variable instead of the dummies.
    h : int, default=0
        Number of observations ahead to produce the dummies for as well.
    freq : int, optional
        Deprecated: the frequency of the data, mapped onto ``type`` and ``of`` as
        R's ``temporaldummy()`` of a ts object does (12 months of year, 4
        quarters of year, 52 weeks of year, 7 days of week, 24 hours of day, 48
        half-hours of day, 60 minutes of hour). Other values produce the cyclic
        dummies of that many periods as an np.ndarray.

    Returns
    -------
    np.ndarray, pd.DataFrame, pd.Categorical, pd.Series or None
        The dummies, with the columns of the observed values for the dates and of
        all the values otherwise: a DataFrame with R's column names (e.g.
        ``month1ofyear``) and the (extended) index of a pandas input. With
        ``factors=True``, the categorical variable: a ``pd.Categorical``, or a
        categorical ``pd.Series`` for a pandas input. None, with a warning, for
        the hours, minutes and seconds of days, as R's Date.

    Examples
    --------
    >>> temporal_dummy(np.arange(5), type="quarter")
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.],
           [1., 0., 0., 0.]])
    >>> dates = pd.date_range("2020-02-27", periods=4, freq="D")
    >>> dummies = temporal_dummy(dates, type="day", of="week", h=1)
    >>> list(dummies.columns)
    ['dayFriofweek', 'dayMonofweek', 'daySatofweek', 'daySunofweek', 'dayThuofweek']
    >>> dummies.index[-1]
    Timestamp('2020-03-02 00:00:00')
    """
    if freq is not None:
        warnings.warn(
            "freq is deprecated, use type and of instead",
            DeprecationWarning,
            stacklevel=2,
        )
        type_of = {
            12: ("month", "year"),
            4: ("quarter", "year"),
            52: ("week", "year"),
            7: ("day", "week"),
            24: ("hour", "day"),
            48: ("halfhour", "day"),
            60: ("minute", "hour"),
        }.get(freq)
        if type_of is None:
            return np.eye(freq)[np.arange(len(x) + h) % freq]
        type, of = type_of

    # The number of periods of the type in the of, for the data without dates
    periods: dict[str, tuple[dict[str, int], int]] = {
        "month": ({"year": 12}, 3),
        "quarter": ({}, 4),
        "week": ({"year": 52, "quarter": 13}, 4),
        "day": ({"year": 365, "quarter": 90, "month": 30}, 7),
        "hour": ({"year": 8760, "quarter": 2160, "month": 720, "week": 168}, 24),
        "halfhour": (
            {"year": 17520, "quarter": 4320, "month": 1440, "week": 336},
            48,
        ),
        "minute": (
            {
                "year": 525600,
                "quarter": 129600,
                "month": 43200,
                "week": 10080,
                "day": 1440,
            },
            60,
        ),
        "second": (
            {
                "year": 31536000,
                "quarter": 7776000,
                "month": 2592000,
                "week": 604800,
                "day": 86400,
                "hour": 3600,
            },
            60,
        ),
    }
    if type not in periods:
        raise ValueError(f"type must be one of: {', '.join(periods)}")
    if of not in ("year", "quarter", "month", "week", "day", "hour", "minute"):
        raise ValueError(
            "of must be one of: year, quarter, month, week, day, hour, minute"
        )

    # The dates of a datetime Series / array, else of the index, as R's zoo
    index = x.index if isinstance(x, (pd.Series, pd.DataFrame)) else None
    index = x if isinstance(x, pd.DatetimeIndex) else index
    if isinstance(x, pd.Series) and pd.api.types.is_datetime64_any_dtype(x.dtype):
        dates = pd.DatetimeIndex(x)
    elif isinstance(index, pd.DatetimeIndex):
        dates = index
    elif np.issubdtype(np.asarray(x).dtype, np.datetime64):
        dates = pd.DatetimeIndex(np.asarray(x))
        if np.datetime_data(np.asarray(x).dtype)[0] in ("Y", "M", "W", "D") and (
            type in ("hour", "halfhour", "minute", "second")
        ):
            warnings.warn(
                "The Date class does not support hours, minutes and seconds. "
                "Use POSIXt instead.",
                stacklevel=2,
            )
            return None
    else:
        dates = None

    # The h observations ahead continue with the last step, as R
    steps = np.arange(1, h + 1)
    if h > 0 and index is not None:
        index = index.append(pd.Index(index[-1] + steps * (index[-1] - index[-2])))
    if h > 0 and dates is not None:
        dates = dates.append(
            pd.DatetimeIndex(dates[-1] + steps * (dates[-1] - dates[-2]))
        )

    if dates is None:
        n_periods = periods[type][0].get(of, periods[type][1])
        values = np.arange(len(x) + h) % n_periods + 1
        levels = np.arange(1, n_periods + 1)
    else:
        # The clock time, with the days since the start of the year / quarter /
        # month / week
        d = dates if dates.tz is None else dates.tz_localize(None)
        days_of = {
            "year": d.dayofyear - 1,
            "quarter": (d.normalize() - d.to_period("Q").to_timestamp()).days,
            "month": d.day - 1,
            "week": d.dayofweek,
        }.get(of, np.zeros(len(d), dtype=int))
        abb = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        abb += ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if type == "quarter":
            values = np.array([f"Q{q}" for q in d.quarter])
        elif type == "month" and of == "quarter":
            values = np.asarray((d.month - 1) % 3 + 1)
        elif type == "month":
            values = np.array(abb)[d.month - 1]
        elif type == "week":
            values = np.asarray(d.isocalendar().week, dtype=int)
            if of != "year":
                warnings.warn(
                    f"The option of={of} is not yet implemented. "
                    "Returning weeks of year instead.",
                    stacklevel=2,
                )
        elif type == "day" and of in ("year", "quarter", "month"):
            values = np.asarray(days_of + 1)
        elif type == "day" and of == "week":
            values = np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])[
                d.dayofweek
            ]
        elif type == "hour" and of == "day":
            values = np.asarray(d.hour)
        elif type == "hour" and of in ("year", "quarter", "month", "week"):
            values = np.asarray(days_of * 24 + d.hour + 1)
        elif type == "halfhour":
            raise ValueError("Sorry, but the halfhour option is not available yet")
        elif type == "minute" and of == "hour":
            values = np.asarray(d.minute)
        elif type == "minute" and of != "minute":
            values = np.asarray(days_of * 1440 + d.hour * 60 + d.minute + 1)
        elif type == "second" and of == "minute":
            values = np.asarray(d.second)
        elif type == "second":
            raise ValueError(
                f"Sorry, but the seconds option is not available yet for the {of}"
            )
        else:
            raise ValueError(f'type="{type}" is not supported with of="{of}"')
        levels = np.unique(values)

    if factors:
        categorical = pd.Categorical(values, categories=levels)
        return categorical if index is None else pd.Series(categorical, index=index)
    dummies = (values[:, None] == levels).astype(float)
    if index is None:
        return dummies
    columns = [f"{type}{level}of{of}" for level in levels]
    return pd.DataFrame(dummies, index=index, columns=columns)


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
