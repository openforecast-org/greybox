"""Statistical error measures for model evaluation.

This module provides functions for calculating various statistical error measures
for point and interval forecasts. The functions include:

Error measures:

- ME - Mean Error
- MAE - Mean Absolute Error
- MSE - Mean Squared Error
- MRE - Mean Root Error (Kourentzes, 2014)
- MIS - Mean Interval Score (Gneiting & Raftery, 2007)
- MPE - Mean Percentage Error
- MAPE - Mean Absolute Percentage Error (See Svetunkov, 2017 for critique)
- MASE - Mean Absolute Scaled Error (Hyndman & Koehler, 2006)
- RMSSE - Root Mean Squared Scaled Error (used in M5 Competition)
- SAME - Scaled Absolute Mean Error (similar to MASE, but measures bias)
- rMAE - Relative Mean Absolute Error (Davydenko & Fildes, 2013)
- rRMSE - Relative Root Mean Squared Error
- rAME - Relative Absolute Mean Error
- rMIS - Relative Mean Interval Score
- sMSE - Scaled Mean Squared Error (Petropoulos & Kourentzes, 2015)
- sPIS - Scaled Periods-In-Stock (Wallstrom & Segerstedt, 2010)
- sCE - Scaled Cumulative Error
- sMIS - Scaled Mean Interval Score
- GMRAE - Geometric Mean Relative Absolute Error

References
----------
- Kourentzes N. (2014). The Bias Coefficient: a new metric for forecast bias.
  https://kourentzes.com/forecasting/2014/12/17/the-bias-coefficient-a-new-metric-for-forecast-bias/
- Svetunkov, I. (2017). Naughty APEs and the quest for the holy grail.
  https://openforecast.org/2017/07/29/naughty-apes-and-the-quest-for-the-holy-grail/
- Fildes R. (1992). The evaluation of extrapolative forecasting methods.
  International Journal of Forecasting, 8, pp.81-98.
- Hyndman R.J., Koehler A.B. (2006). Another look at measures of forecast accuracy.
  International Journal of Forecasting, 22, pp.679-688.
- Petropoulos F., Kourentzes N. (2015). Forecast combinations for intermittent demand.
  Journal of the Operational Research Society, 66, pp.914-924.
- Wallstrom P., Segerstedt A. (2010). Evaluation of forecasting error measurements
  and techniques for intermittent demand. International Journal of Production Economics,
  128, pp.625-636.
- Davydenko, A., Fildes, R. (2013). Measuring Forecasting Accuracy: The Case Of
  Judgmental Adjustments To Sku-Level Demand Forecasts. International Journal of
  Forecasting, 29(3), 510-522.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction,
  and estimation. Journal of the American Statistical Association, 102(477), 359-378.
"""

from typing import Any, Literal

import numpy as np

from .hm import asymmetry


def _validate_inputs(
    actual: np.ndarray, forecast: np.ndarray, na_rm: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and clean actual and forecast inputs."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError("Lengths of actual and forecast must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        actual = actual[mask]
        forecast = forecast[mask]

    return actual, forecast


def me(actual: np.ndarray, forecast: np.ndarray, na_rm: bool = True) -> float:
    """Mean Error (ME).

    The ME measures the average bias in forecasts. A positive value indicates
    forecasts are on average too high (over-forecasting), while a negative
    value indicates forecasts are on average too low (under-forecasting).

    Formula:
        ME = mean(actual - forecast)

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Mean Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> round(float(me(actual, forecast)), 2)
    -0.06

    References
    ----------
    Basic measure, see e.g. Hyndman & Koehler (2006).
    """
    actual, forecast = _validate_inputs(actual, forecast, na_rm)
    return np.mean(actual - forecast)


def mae(actual: np.ndarray, forecast: np.ndarray, na_rm: bool = True) -> float:
    """Mean Absolute Error (MAE).

    The MAE measures the average magnitude of errors without considering
    direction. It is robust to outliers and gives a direct interpretation
    in the same units as the data.

    Formula:
        ``MAE = mean(|actual - forecast|)``

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Mean Absolute Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> round(float(mae(actual, forecast)), 2)
    0.1

    References
    ----------
    See e.g. Hyndman & Koehler (2006).
    """
    actual, forecast = _validate_inputs(actual, forecast, na_rm)
    return np.mean(np.abs(actual - forecast))


def mse(actual: np.ndarray, forecast: np.ndarray, na_rm: bool = True) -> float:
    """Mean Squared Error (MSE).

    The MSE measures the average squared difference between actual and forecast.
    It penalizes larger errors more heavily than MAE.

    Formula:
        MSE = mean((actual - forecast)^2)

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Mean Squared Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> round(float(mse(actual, forecast)), 3)
    0.014

    References
    ----------
    See e.g. Hyndman & Koehler (2006).
    """
    actual, forecast = _validate_inputs(actual, forecast, na_rm)
    return np.mean((actual - forecast) ** 2)


def rmse(actual: np.ndarray, forecast: np.ndarray, na_rm: bool = True) -> float:
    """Root Mean Squared Error (RMSE).

    The RMSE is the square root of MSE. It is in the same units as the data,
    making it more interpretable than MSE while still penalizing larger errors.

    Formula:
        RMSE = sqrt(MSE) = sqrt(mean((actual - forecast)^2))

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Root Mean Squared Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> round(float(rmse(actual, forecast)), 4)
    0.1183

    References
    ----------
    See e.g. Hyndman & Koehler (2006).
    """
    return np.sqrt(mse(actual, forecast, na_rm))


def mpe(actual: np.ndarray, forecast: np.ndarray, na_rm: bool = True) -> float:
    """Mean Percentage Error (MPE).

    The MPE measures the average percentage error. Like ME, it can detect
    bias but expresses it as a fraction. However, it is undefined when
    actual values are zero.

    Formula:
        MPE = mean((actual - forecast) / actual)

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values and zero actuals.

    Returns
    -------
    float
        Mean Percentage Error (as a fraction).

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> round(float(mpe(actual, forecast)), 4)
    -0.0323

    References
    ----------
    See e.g. Hyndman & Koehler (2006).
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError("Lengths of actual and forecast must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(forecast) | (actual == 0))
        actual = actual[mask]
        forecast = forecast[mask]

    return np.mean((actual - forecast) / actual)


def mape(actual: np.ndarray, forecast: np.ndarray, na_rm: bool = True) -> float:
    """Mean Absolute Percentage Error (MAPE).

    The MAPE measures the average absolute percentage error. It is scale-independent
    and expressed as a fraction, making it easy to interpret across different
    contexts. However, it is undefined when actual values are zero and can be
    heavily influenced by small actual values.

    Formula:
        ``MAPE = mean(|actual - forecast| / actual)``

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values and zero actuals.

    Returns
    -------
    float
        Mean Absolute Percentage Error (as a fraction).

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> round(float(mape(actual, forecast)), 4)
    0.0423

    References
    ----------
    See Svetunkov, I. (2017) for a critique of MAPE:
    https://openforecast.org/2017/07/29/naughty-apes-and-the-quest-for-the-holy-grail/
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError("Lengths of actual and forecast must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(forecast) | (actual == 0))
        actual = actual[mask]
        forecast = forecast[mask]

    return np.mean(np.abs((actual - forecast) / actual))


def mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    scale: float | np.floating[Any] | None = None,
    na_rm: bool = True,
) -> float:
    """Mean Absolute Scaled Error (MASE).

    The MASE scales the MAE by the average error of a naive (random walk)
    forecast. This makes it scale-independent and allows comparison across
    different time series. A MASE < 1 means the forecast is better than
    the naive forecast; MASE > 1 means worse.

    Formula:
        MASE = MAE / scale
        where ``scale = mean(|actual[t] - actual[t-1]|)`` by default

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    scale : float, optional
        Scale parameter. If None, uses mean absolute difference of consecutive
        actual values (naive forecast error). Typical values include:
        - mean absolute deviation of in-sample one step ahead naive error
        - mean absolute value of the in-sample actuals
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Mean Absolute Scaled Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> round(float(mase(actual, forecast)), 2)
    0.15
    >>> # With custom scale
    >>> round(float(mase(actual, forecast, scale=1.0)), 2)
    0.15

    References
    ----------
    Hyndman R.J., Koehler A.B. (2006). Another look at measures of forecast accuracy.
    International Journal of Forecasting, 22, pp.679-688.
    """
    actual, forecast = _validate_inputs(actual, forecast, na_rm)

    if scale is None:
        scale = np.mean(np.abs(np.diff(actual)))

    if scale == 0:
        return np.nan

    return np.mean(np.abs(actual - forecast)) / scale


def rmsse(
    actual: np.ndarray,
    forecast: np.ndarray,
    scale: float | np.floating[Any] | None = None,
    na_rm: bool = True,
) -> float:
    """Root Mean Squared Scaled Error (RMSSE).

    The RMSSE is similar to MASE but uses MSE instead of MAE. It was
    developed for the M5 Competition.

    Formula:
        RMSSE = sqrt(MSE / scale)

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    scale : float, optional
        Scale parameter. If None, uses mean squared difference of consecutive
        actual values. Typical value: mean(diff(actual)^2).
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Root Mean Squared Scaled Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> round(float(rmsse(actual, forecast)), 4)
    0.1581

    References
    ----------
    Used in M5 Competition.
    """
    actual, forecast = _validate_inputs(actual, forecast, na_rm)

    if scale is None:
        scale = np.mean(np.diff(actual) ** 2)

    if scale == 0:
        return np.nan

    return np.sqrt(np.mean((actual - forecast) ** 2) / scale)


def same(
    actual: np.ndarray,
    forecast: np.ndarray,
    scale: float | np.floating[Any] | None = None,
    na_rm: bool = True,
) -> float:
    """Scaled Absolute Mean Error (SAME).

    SAME is similar to MASE but uses Mean Error (ME) instead of MAE.
    It measures bias rather than absolute error.

    Formula:
        ``SAME = |ME| / scale``

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    scale : float, optional
        Scale parameter. If None, uses mean absolute difference of consecutive
        actual values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Scaled Absolute Mean Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> round(float(same(actual, forecast)), 2)
    0.13

    References
    ----------
    See Petropoulos & Kourentzes (2015).
    """
    actual, forecast = _validate_inputs(actual, forecast, na_rm)

    if scale is None:
        scale = np.mean(np.abs(np.diff(actual)))

    if scale == 0:
        return np.nan

    return np.abs(np.mean(actual - forecast)) / scale


def rmae(
    actual: np.ndarray,
    forecast: np.ndarray,
    benchmark: np.ndarray,
    na_rm: bool = True,
) -> float:
    """Relative Mean Absolute Error (rMAE).

    The rMAE compares the MAE of a forecast to a benchmark forecast.
    Values < 1 indicate the forecast is better than the benchmark;
    values > 1 indicate worse.

    Formula:
        rMAE = MAE(forecast) / MAE(benchmark)

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    benchmark : np.ndarray
        Benchmark forecast values (e.g., naive or mean forecast).
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Relative Mean Absolute Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> benchmark = np.array([0.5, 1.5, 2.5, 3.5, 4.5,
    ...                      5.5, 6.5, 7.5, 8.5, 9.5])
    >>> round(float(rmae(actual, forecast, benchmark)), 2)
    0.3

    References
    ----------
    Davydenko, A., Fildes, R. (2013). Measuring Forecasting Accuracy:
    The Case Of Judgmental Adjustments To Sku-Level Demand Forecasts.
    International Journal of Forecasting, 29(3), 510-522.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)

    lengths = [len(actual), len(forecast), len(benchmark)]
    if len(set(lengths)) > 1:
        raise ValueError("Lengths of actual, forecast, and benchmark must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(forecast) | np.isnan(benchmark))
        actual = actual[mask]
        forecast = forecast[mask]
        benchmark = benchmark[mask]

    if np.all(forecast == benchmark):
        return 1.0

    return mae(actual, forecast, na_rm=False) / mae(actual, benchmark, na_rm=False)


def rrmse(
    actual: np.ndarray,
    forecast: np.ndarray,
    benchmark: np.ndarray,
    na_rm: bool = True,
) -> float:
    """Relative Root Mean Squared Error (rRMSE).

    The rRMSE compares the RMSE of a forecast to a benchmark forecast.

    Formula:
        rRMSE = RMSE(forecast) / RMSE(benchmark)

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    benchmark : np.ndarray
        Benchmark forecast values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Relative Root Mean Squared Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> benchmark = np.array([0.5, 1.5, 2.5, 3.5, 4.5,
    ...                      5.5, 6.5, 7.5, 8.5, 9.5])
    >>> round(float(rrmse(actual, forecast, benchmark)), 4)
    0.3162

    References
    ----------
    Davydenko, A., Fildes, R. (2013). Measuring Forecasting Accuracy.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)

    lengths = [len(actual), len(forecast), len(benchmark)]
    if len(set(lengths)) > 1:
        raise ValueError("Lengths of actual, forecast, and benchmark must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(forecast) | np.isnan(benchmark))
        actual = actual[mask]
        forecast = forecast[mask]
        benchmark = benchmark[mask]

    if np.all(forecast == benchmark):
        return 1.0

    return rmse(actual, forecast, na_rm=False) / rmse(actual, benchmark, na_rm=False)


def rame(
    actual: np.ndarray,
    forecast: np.ndarray,
    benchmark: np.ndarray,
    na_rm: bool = True,
) -> float:
    """Relative Absolute Mean Error (rAME).

    The rAME compares the absolute Mean Error (bias) of a forecast to
    a benchmark forecast. It measures relative bias.

    Formula:
        ``rAME = |ME(forecast)| / |ME(benchmark)|``

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    benchmark : np.ndarray
        Benchmark forecast values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Relative Absolute Mean Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> benchmark = np.array([0.5, 1.5, 2.5, 3.5, 4.5,
    ...                      5.5, 6.5, 7.5, 8.5, 9.5])
    >>> round(float(rame(actual, forecast, benchmark)), 2)
    0.26

    References
    ----------
    Davydenko, A., Fildes, R. (2013). Measuring Forecasting Accuracy.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)

    lengths = [len(actual), len(forecast), len(benchmark)]
    if len(set(lengths)) > 1:
        raise ValueError("Lengths of actual, forecast, and benchmark must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(forecast) | np.isnan(benchmark))
        actual = actual[mask]
        forecast = forecast[mask]
        benchmark = benchmark[mask]

    if np.all(forecast == benchmark):
        return 1.0

    me_benchmark = me(actual, benchmark, na_rm=False)
    if me_benchmark == 0:
        return np.inf

    return np.abs(me(actual, forecast, na_rm=False)) / np.abs(me_benchmark)


def smse(
    actual: np.ndarray,
    forecast: np.ndarray,
    scale: float | np.floating[Any],
    na_rm: bool = True,
) -> float:
    """Scaled Mean Squared Error (sMSE).

    The sMSE scales the MSE by a scale parameter. Note that the scale
    should be a squared value (e.g., mean of squared actuals).

    Formula:
        sMSE = MSE / scale

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    scale : float
        Scale parameter. Should typically be a squared value
        (e.g., mean(actual)^2 or mean(abs(actual[actual!=0]))^2).
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Scaled Mean Squared Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> scale = np.mean(np.abs(actual)) ** 2
    >>> round(float(smse(actual, forecast, scale)), 5)
    0.00083

    References
    ----------
    Petropoulos F., Kourentzes N. (2015). Forecast combinations for intermittent demand.
    Journal of the Operational Research Society, 66, pp.914-924.
    """
    actual, forecast = _validate_inputs(actual, forecast, na_rm)

    if scale == 0:
        return np.nan

    return np.mean((actual - forecast) ** 2) / scale


def spis(
    actual: np.ndarray,
    forecast: np.ndarray,
    scale: float | np.floating[Any],
    na_rm: bool = True,
) -> float:
    """Scaled Periods-In-Stock (sPIS).

    The sPIS measures the cumulative inventory performance over time.
    It is useful for intermittent demand forecasting.

    Formula:
        sPIS = sum(cumsum(forecast - actual)) / scale

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    scale : float
        Scale parameter (typically mean of non-zero actuals).
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Scaled Periods-In-Stock.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> scale = np.mean(np.abs(actual[actual != 0]))
    >>> round(float(spis(actual, forecast, scale)), 4)
    1.2364

    References
    ----------
    Wallstrom P., Segerstedt A. (2010). Evaluation of forecasting error measurements
    and techniques for intermittent demand. International Journal of Production
    Economics, 128, pp.625-636.
    """
    actual, forecast = _validate_inputs(actual, forecast, na_rm)

    if scale == 0:
        return np.nan

    cumsum_errors = np.cumsum(forecast - actual)
    return float(np.sum(cumsum_errors, axis=0) / scale)


def sce(
    actual: np.ndarray,
    forecast: np.ndarray,
    scale: float | np.floating[Any],
    na_rm: bool = True,
) -> float:
    """Scaled Cumulative Error (sCE).

    The sCE measures the cumulative forecast error scaled by a parameter.
    It can detect systematic bias over time.

    Formula:
        sCE = sum(actual - forecast) / scale

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    scale : float
        Scale parameter.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Scaled Cumulative Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> scale = np.mean(np.abs(actual[actual != 0]))
    >>> round(float(sce(actual, forecast, scale)), 4)
    -0.2364

    References
    ----------
    Petropoulos F., Kourentzes N. (2015). Forecast combinations for intermittent demand.
    """
    actual, forecast = _validate_inputs(actual, forecast, na_rm)

    if scale == 0:
        return np.nan

    return float(np.sum(actual - forecast, axis=0) / scale)


def gmrae(
    actual: np.ndarray,
    forecast: np.ndarray,
    benchmark: np.ndarray,
    na_rm: bool = True,
) -> float:
    """Geometric Mean Relative Absolute Error (GMRAE).

    The GMRAE compares the absolute errors of a forecast to a benchmark
    using geometric mean. It is robust to extreme values.

    Formula:
        ``GMRAE = exp(mean(log(|error| / |benchmark_error|)))``

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    benchmark : np.ndarray
        Benchmark forecast values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Geometric Mean Relative Absolute Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> benchmark = np.array([0.5, 1.5, 2.5, 3.5, 4.5,
    ...                      5.5, 6.5, 7.5, 8.5, 9.5])
    >>> round(float(gmrae(actual, forecast, benchmark)), 4)
    0.2828

    References
    ----------
    See Davydenko & Fildes (2013).
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)

    lengths = [len(actual), len(forecast), len(benchmark)]
    if len(set(lengths)) > 1:
        raise ValueError("Lengths of actual, forecast, and benchmark must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(forecast) | np.isnan(benchmark))
        actual = actual[mask]
        forecast = forecast[mask]
        benchmark = benchmark[mask]

    errors = actual - forecast
    benchmark_errors = actual - benchmark

    ratios = np.abs(errors) / np.abs(benchmark_errors)
    valid_mask = ~np.isnan(ratios) & (np.abs(benchmark_errors) > 0)
    ratios = ratios[valid_mask]

    if len(ratios) == 0:
        return np.nan

    return np.exp(np.mean(np.log(ratios)))


def measures(
    holdout: np.ndarray,
    forecast: np.ndarray,
    actual: np.ndarray,
    digits: int | None = None,
    benchmark: Literal["naive", "mean"] = "naive",
) -> dict:
    """Calculate comprehensive error measures for forecasts.

    This function calculates multiple error measures using holdout (test) data,
    forecasted values, and in-sample (training) data for scale calculation.

    Parameters
    ----------
    holdout : np.ndarray
        The vector of holdout (test) actual values.
    forecast : np.ndarray
        The vector of forecasts produced by a model.
    actual : np.ndarray
        The vector of actual in-sample (training) values.
    digits : int, optional
        Number of digits for rounding. If None, no rounding is done.
    benchmark : {"naive", "mean"}, default="naive"
        Benchmark method for relative measures:
        - "naive": Use last value repeated (random walk)
        - "mean": Use arithmetic mean of the series

    Returns
    -------
    dict
        Dictionary containing error measures:
        ME, MAE, MSE, MPE, MAPE, sCE, sMAE, sMSE, MASE,
        RMSSE, SAME, rMAE, rRMSE, rAME, asymmetry, sPIS

    Examples
    --------
    >>> import numpy as np
    >>> from greybox import measures
    >>> actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1, 6.0, 7.1, 8.0, 9.2])
    >>> result = measures(actual[5:], forecast[5:], actual[:5], digits=4)
    >>> print(result)
    {'ME': ..., 'MAE': ..., ...}

    References
    ----------
    See individual function references:
    - Hyndman & Koehler (2006) for MASE, RMSSE
    - Petropoulos & Kourentzes (2015) for sMSE, sCE, sPIS
    - Davydenko & Fildes (2013) for rMAE, rRMSE, rAME
    """
    holdout = np.asarray(holdout, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)

    h = len(holdout)

    if benchmark == "naive":
        benchmark_forecast = np.full(h, actual[-1])
    else:
        benchmark_forecast = np.full(h, np.mean(actual))

    nonzero_mask = actual != 0
    mean_abs_actual = (
        np.mean(np.abs(actual[nonzero_mask])) if np.any(nonzero_mask) else np.nan
    )
    mean_abs_diff = np.mean(np.abs(np.diff(actual)))

    error_measures = {
        "ME": me(holdout, forecast),
        "MAE": mae(holdout, forecast),
        "MSE": mse(holdout, forecast),
        "MPE": mpe(holdout, forecast),
        "MAPE": mape(holdout, forecast),
        "sCE": sce(holdout, forecast, mean_abs_actual),
        "sMAE": mase(holdout, forecast, np.mean(np.abs(actual))),
        "sMSE": smse(holdout, forecast, mean_abs_actual**2),
        "MASE": mase(holdout, forecast, mean_abs_diff),
        "RMSSE": rmsse(holdout, forecast, np.mean(np.diff(actual) ** 2)),
        "SAME": same(holdout, forecast, mean_abs_diff),
        "rMAE": rmae(holdout, forecast, benchmark_forecast),
        "rRMSE": rrmse(holdout, forecast, benchmark_forecast),
        "rAME": rame(holdout, forecast, benchmark_forecast),
        "asymmetry": asymmetry(holdout - forecast, 0),
        "sPIS": spis(holdout, forecast, mean_abs_actual),
    }

    if digits is not None:
        for key, value in error_measures.items():
            error_measures[key] = round(value, digits)

    return error_measures
