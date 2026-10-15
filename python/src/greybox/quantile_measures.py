"""Quantile and interval scoring measures.

This module provides functions for:
1. Pinball cost function for quantile and expectile forecasts
2. Mean Interval Score (MIS) and related measures

References
----------
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. Journal of the American Statistical Association,
  102(477), 359-378.
"""

import warnings

import numpy as np


def pinball(
    holdout: np.ndarray,
    forecast: np.ndarray,
    level: float,
    loss: float = 1,
    na_rm: bool = True,
) -> float:
    """Pinball cost function.

    The pinball function measures the quality of quantile or expectile forecasts.
    It is used in quantile regression and forecast accuracy evaluation.

    For quantiles (loss=1):
        pinball = (1-level) * sum(abs(e) * I(e <= 0)) + level * sum(abs(e) * I(e > 0))

    where e = holdout - forecast.

    For expectiles (loss=2):
        Uses squared errors instead of absolute errors. Any other power of the
        absolute errors can be used as well.

    Parameters
    ----------
    holdout : np.ndarray
        Actual values.
    forecast : np.ndarray
        Forecasted values (quantile or expectile).
    level : float
        The level associated with the forecast (e.g., 0.5 for median,
        0.95 for 95th percentile).
    loss : float, default=1
        The power of the absolute errors:

        - 1: L1 loss (for quantiles)
        - 2: L2 loss (for expectiles)
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Pinball cost value.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> round(float(pinball(holdout, forecast, level=0.5)), 4)  # Median
    0.25
    >>> round(float(pinball(holdout, forecast, level=0.975)), 4)  # Upper
    0.1075
    >>> round(float(pinball(holdout, forecast, level=0.025)), 4)  # Lower
    0.3925

    References
    ----------
    Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
    prediction, and estimation. Journal of the American Statistical Association,
    102(477), 359-378.
    """
    holdout = np.asarray(holdout, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(holdout) != len(forecast):
        raise ValueError("Lengths of holdout and forecast must match")

    if na_rm:
        mask = ~(np.isnan(holdout) | np.isnan(forecast))
        holdout = holdout[mask]
        forecast = forecast[mask]

    errors = holdout - forecast
    result = (1 - level) * np.sum(np.abs(errors) ** loss * (errors <= 0)) + level * (
        np.sum(np.abs(errors) ** loss * (errors > 0))
    )

    return result


def mis(
    holdout: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    level: float = 0.95,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Mean Interval Score (MIS).

    The MIS evaluates the quality of prediction intervals. It rewards
    narrow intervals and penalizes misses (holdout values outside the interval).

    Formula:
        MIS = (upper - lower)
              + (2/alpha) * abs(lower - holdout), for holdout < lower
              + (2/alpha) * abs(holdout - upper), for holdout > upper

        where alpha = 1 - level

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    lower : np.ndarray
        Lower bound of prediction interval.
    upper : np.ndarray
        Upper bound of prediction interval.
    level : float, default=0.95
        Confidence level of the interval (e.g., 0.95 for 95% interval).
    na_rm : bool, default=True
        Remove NA values from the penalties for the values outside the interval.
        As in R, the widths of the interval are summed without removing NA values,
        and the sum is divided by the length of ``holdout``.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Mean Interval Score.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5])
    >>> lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    >>> upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> round(float(mis(holdout, lower, upper, level=0.95)), 4)
    1.0

    References
    ----------
    Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
    prediction, and estimation. Journal of the American Statistical Association,
    102(477), 359-378.
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout = np.asarray(holdout, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if level > 1:
        level = level / 100

    alpha = 1 - level
    lengths = [len(holdout), len(lower), len(upper)]

    if len(set(lengths)) > 1:
        raise ValueError("Lengths of holdout, lower, and upper must match")

    penalty_sum = np.nansum if na_rm else np.sum
    mis_value = np.sum(upper - lower) + (2 / alpha) * (
        penalty_sum((lower - holdout) * (holdout < lower))
        + penalty_sum((holdout - upper) * (holdout > upper))
    )

    return mis_value / len(holdout)


def smis(
    holdout: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    scale: float | np.floating | None = None,
    level: float = 0.95,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Scaled Mean Interval Score (sMIS).

    The sMIS scales the MIS by a scale parameter.

    Formula:
        sMIS = MIS / scale

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    lower : np.ndarray
        Lower bound of prediction interval.
    upper : np.ndarray
        Upper bound of prediction interval.
    scale : float
        Scale parameter.
    level : float, default=0.95
        Confidence level of the interval.
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Scaled Mean Interval Score.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5])
    >>> lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    >>> upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> round(float(smis(holdout, lower, upper, scale=3.0)), 4)
    0.3333

    References
    ----------
    See Gneiting & Raftery (2007).
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    if scale is None:
        raise TypeError("smis() missing required argument: 'scale'")

    return float(mis(holdout, lower, upper, level, na_rm) / scale)


def rmis(
    holdout: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    benchmark_lower: np.ndarray | None = None,
    benchmark_upper: np.ndarray | None = None,
    level: float = 0.95,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Relative Mean Interval Score (rMIS).

    The rMIS compares the MIS of a forecast to a benchmark forecast.

    Formula:
        rMIS = MIS(forecast) / MIS(benchmark)

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    lower : np.ndarray
        Lower bound of prediction interval.
    upper : np.ndarray
        Upper bound of prediction interval.
    benchmark_lower : np.ndarray
        Lower bound of benchmark prediction interval.
    benchmark_upper : np.ndarray
        Upper bound of benchmark prediction interval.
    level : float, default=0.95
        Confidence level of the interval.
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Relative Mean Interval Score.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5])
    >>> lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    >>> upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> benchmark_lower = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> benchmark_upper = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    >>> round(float(rmis(holdout, lower, upper, benchmark_lower, benchmark_upper)), 4)
    0.5

    References
    ----------
    See Gneiting & Raftery (2007).
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    return mis(holdout, lower, upper, level, na_rm) / mis(
        holdout, benchmark_lower, benchmark_upper, level, na_rm
    )
