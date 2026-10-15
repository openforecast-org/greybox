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

import warnings
from typing import Any, Literal

import numpy as np

from .hm import asymmetry


def _validate_inputs(
    holdout: np.ndarray | None, forecast: np.ndarray | None, na_rm: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and clean holdout and forecast inputs."""
    holdout = np.asarray(holdout, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(holdout) != len(forecast):
        raise ValueError("Lengths of holdout and forecast must match")

    if na_rm:
        mask = ~(np.isnan(holdout) | np.isnan(forecast))
        holdout = holdout[mask]
        forecast = forecast[mask]

    return holdout, forecast


def me(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Mean Error (ME).

    The ME measures the average bias in forecasts. A positive value indicates
    forecasts are on average too high (over-forecasting), while a negative
    value indicates forecasts are on average too low (under-forecasting).

    Formula:
        ME = mean(holdout - forecast)

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Mean Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> round(float(me(holdout, forecast)), 2)
    -0.06

    References
    ----------
    Basic measure, see e.g. Hyndman & Koehler (2006).
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout, forecast = _validate_inputs(holdout, forecast, na_rm)
    return np.mean(holdout - forecast)


def mae(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Mean Absolute Error (MAE).

    The MAE measures the average magnitude of errors without considering
    direction. It is robust to outliers and gives a direct interpretation
    in the same units as the data.

    Formula:
        ``MAE = mean(|holdout - forecast|)``

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Mean Absolute Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> round(float(mae(holdout, forecast)), 2)
    0.1

    References
    ----------
    See e.g. Hyndman & Koehler (2006).
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout, forecast = _validate_inputs(holdout, forecast, na_rm)
    return np.mean(np.abs(holdout - forecast))


def mse(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Mean Squared Error (MSE).

    The MSE measures the average squared difference between holdout and forecast.
    It penalizes larger errors more heavily than MAE.

    Formula:
        MSE = mean((holdout - forecast)^2)

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Mean Squared Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> round(float(mse(holdout, forecast)), 3)
    0.014

    References
    ----------
    See e.g. Hyndman & Koehler (2006).
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout, forecast = _validate_inputs(holdout, forecast, na_rm)
    return np.mean((holdout - forecast) ** 2)


def rmse(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Root Mean Squared Error (RMSE).

    The RMSE is the square root of MSE. It is in the same units as the data,
    making it more interpretable than MSE while still penalizing larger errors.

    Formula:
        RMSE = sqrt(MSE) = sqrt(mean((holdout - forecast)^2))

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Root Mean Squared Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> round(float(rmse(holdout, forecast)), 4)
    0.1183

    References
    ----------
    See e.g. Hyndman & Koehler (2006).
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    return np.sqrt(mse(holdout, forecast, na_rm))


def mpe(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Mean Percentage Error (MPE).

    The MPE measures the average percentage error. Like ME, it can detect
    bias but expresses it as a fraction. However, it is undefined when
    holdout values are zero.

    Formula:
        MPE = mean((holdout - forecast) / holdout)

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Mean Percentage Error (as a fraction).

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> round(float(mpe(holdout, forecast)), 4)
    -0.0323

    References
    ----------
    See e.g. Hyndman & Koehler (2006).
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout, forecast = _validate_inputs(holdout, forecast, na_rm=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = (holdout - forecast) / holdout
    if na_rm:
        ratios = ratios[~np.isnan(ratios)]

    return np.mean(ratios)


def mape(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Mean Absolute Percentage Error (MAPE).

    The MAPE measures the average absolute percentage error. It is scale-independent
    and expressed as a fraction, making it easy to interpret across different
    contexts. However, it is undefined when holdout values are zero and can be
    heavily influenced by small holdout values.

    Formula:
        ``MAPE = mean(|holdout - forecast| / holdout)``

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Mean Absolute Percentage Error (as a fraction).

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> round(float(mape(holdout, forecast)), 4)
    0.0423

    References
    ----------
    See Svetunkov, I. (2017) for a critique of MAPE:
    https://openforecast.org/2017/07/29/naughty-apes-and-the-quest-for-the-holy-grail/
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout, forecast = _validate_inputs(holdout, forecast, na_rm=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.abs((holdout - forecast) / holdout)
    if na_rm:
        ratios = ratios[~np.isnan(ratios)]

    return np.mean(ratios)


def mase(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    scale: float | np.floating[Any] | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Mean Absolute Scaled Error (MASE).

    The MASE scales the MAE by the average error of a naive (random walk)
    forecast. This makes it scale-independent and allows comparison across
    different time series. A MASE < 1 means the forecast is better than
    the naive forecast; MASE > 1 means worse.

    Formula:
        MASE = MAE / scale
        where ``scale = mean(|holdout[t] - holdout[t-1]|)`` by default

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    scale : float, optional
        Scale parameter. If None, uses mean absolute difference of consecutive
        holdout values (naive forecast error). Typical values include:

        - mean absolute deviation of in-sample one step ahead naive error
        - mean absolute value of the in-sample actuals
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Mean Absolute Scaled Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> round(float(mase(holdout, forecast)), 2)
    0.15
    >>> # With custom scale
    >>> round(float(mase(holdout, forecast, scale=1.0)), 2)
    0.15

    References
    ----------
    Hyndman R.J., Koehler A.B. (2006). Another look at measures of forecast accuracy.
    International Journal of Forecasting, 22, pp.679-688.
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout, forecast = _validate_inputs(holdout, forecast, na_rm)

    if scale is None:
        scale = np.mean(np.abs(np.diff(holdout)))

    return np.mean(np.abs(holdout - forecast)) / scale


def rmsse(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    scale: float | np.floating[Any] | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Root Mean Squared Scaled Error (RMSSE).

    The RMSSE is similar to MASE but uses MSE instead of MAE. It was
    developed for the M5 Competition.

    Formula:
        RMSSE = sqrt(MSE / scale)

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    scale : float, optional
        Scale parameter. If None, uses mean squared difference of consecutive
        holdout values. Typical value: mean(diff(holdout)^2).
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Root Mean Squared Scaled Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> round(float(rmsse(holdout, forecast)), 4)
    0.1581

    References
    ----------
    Used in M5 Competition.
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout, forecast = _validate_inputs(holdout, forecast, na_rm)

    if scale is None:
        scale = np.mean(np.diff(holdout) ** 2)

    return np.sqrt(np.mean((holdout - forecast) ** 2) / scale)


def same(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    scale: float | np.floating[Any] | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Scaled Absolute Mean Error (SAME).

    SAME is similar to MASE but uses Mean Error (ME) instead of MAE.
    It measures bias rather than absolute error.

    Formula:
        ``SAME = |ME| / scale``

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    scale : float, optional
        Scale parameter. If None, uses mean absolute difference of consecutive
        holdout values.
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Scaled Absolute Mean Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> round(float(same(holdout, forecast)), 2)
    0.13

    References
    ----------
    See Petropoulos & Kourentzes (2015).
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout, forecast = _validate_inputs(holdout, forecast, na_rm)

    if scale is None:
        scale = np.mean(np.abs(np.diff(holdout)))

    return np.abs(np.mean(holdout - forecast)) / scale


def rmae(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    benchmark: np.ndarray | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Relative Mean Absolute Error (rMAE).

    The rMAE compares the MAE of a forecast to a benchmark forecast.
    Values < 1 indicate the forecast is better than the benchmark;
    values > 1 indicate worse.

    Formula:
        rMAE = MAE(forecast) / MAE(benchmark)

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    benchmark : np.ndarray
        Benchmark forecast values (e.g., naive or mean forecast).
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Relative Mean Absolute Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> benchmark = np.array([0.5, 1.5, 2.5, 3.5, 4.5,
    ...                      5.5, 6.5, 7.5, 8.5, 9.5])
    >>> round(float(rmae(holdout, forecast, benchmark)), 2)
    0.3

    References
    ----------
    Davydenko, A., Fildes, R. (2013). Measuring Forecasting Accuracy:
    The Case Of Judgmental Adjustments To Sku-Level Demand Forecasts.
    International Journal of Forecasting, 29(3), 510-522.
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout = np.asarray(holdout, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)

    lengths = [len(holdout), len(forecast), len(benchmark)]
    if len(set(lengths)) > 1:
        raise ValueError("Lengths of holdout, forecast, and benchmark must match")

    if na_rm:
        mask = ~(np.isnan(holdout) | np.isnan(forecast) | np.isnan(benchmark))
        holdout = holdout[mask]
        forecast = forecast[mask]
        benchmark = benchmark[mask]

    if np.all(forecast == benchmark):
        return 1.0

    return mae(holdout, forecast, na_rm=False) / mae(holdout, benchmark, na_rm=False)


def rrmse(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    benchmark: np.ndarray | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Relative Root Mean Squared Error (rRMSE).

    The rRMSE compares the RMSE of a forecast to a benchmark forecast.

    Formula:
        rRMSE = RMSE(forecast) / RMSE(benchmark)

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    benchmark : np.ndarray
        Benchmark forecast values.
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Relative Root Mean Squared Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> benchmark = np.array([0.5, 1.5, 2.5, 3.5, 4.5,
    ...                      5.5, 6.5, 7.5, 8.5, 9.5])
    >>> round(float(rrmse(holdout, forecast, benchmark)), 4)
    0.3162

    References
    ----------
    Davydenko, A., Fildes, R. (2013). Measuring Forecasting Accuracy.
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout = np.asarray(holdout, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)

    lengths = [len(holdout), len(forecast), len(benchmark)]
    if len(set(lengths)) > 1:
        raise ValueError("Lengths of holdout, forecast, and benchmark must match")

    if na_rm:
        mask = ~(np.isnan(holdout) | np.isnan(forecast) | np.isnan(benchmark))
        holdout = holdout[mask]
        forecast = forecast[mask]
        benchmark = benchmark[mask]

    if np.all(forecast == benchmark):
        return 1.0

    return rmse(holdout, forecast, na_rm=False) / rmse(holdout, benchmark, na_rm=False)


def rame(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    benchmark: np.ndarray | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Relative Absolute Mean Error (rAME).

    The rAME compares the absolute Mean Error (bias) of a forecast to
    a benchmark forecast. It measures relative bias.

    Formula:
        ``rAME = |ME(forecast)| / |ME(benchmark)|``

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    benchmark : np.ndarray
        Benchmark forecast values.
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Relative Absolute Mean Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> benchmark = np.array([0.5, 1.5, 2.5, 3.5, 4.5,
    ...                      5.5, 6.5, 7.5, 8.5, 9.5])
    >>> round(float(rame(holdout, forecast, benchmark)), 2)
    0.26

    References
    ----------
    Davydenko, A., Fildes, R. (2013). Measuring Forecasting Accuracy.
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout = np.asarray(holdout, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)

    lengths = [len(holdout), len(forecast), len(benchmark)]
    if len(set(lengths)) > 1:
        raise ValueError("Lengths of holdout, forecast, and benchmark must match")

    if na_rm:
        mask = ~(np.isnan(holdout) | np.isnan(forecast) | np.isnan(benchmark))
        holdout = holdout[mask]
        forecast = forecast[mask]
        benchmark = benchmark[mask]

    if np.all(forecast == benchmark):
        return 1.0

    return np.abs(me(holdout, forecast, na_rm=False)) / np.abs(
        me(holdout, benchmark, na_rm=False)
    )


def smse(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    scale: float | np.floating[Any] | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Scaled Mean Squared Error (sMSE).

    The sMSE scales the MSE by a scale parameter. Note that the scale
    should be a squared value (e.g., mean of squared actuals).

    Formula:
        sMSE = MSE / scale

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    scale : float
        Scale parameter. Should typically be a squared value
        (e.g., the squared mean absolute value of the in-sample actuals).
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Scaled Mean Squared Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> scale = np.mean(np.abs(holdout)) ** 2
    >>> round(float(smse(holdout, forecast, scale)), 5)
    0.00083

    References
    ----------
    Petropoulos F., Kourentzes N. (2015). Forecast combinations for intermittent demand.
    Journal of the Operational Research Society, 66, pp.914-924.
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout, forecast = _validate_inputs(holdout, forecast, na_rm)

    return np.mean((holdout - forecast) ** 2) / scale


def spis(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    scale: float | np.floating[Any] | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Scaled Periods-In-Stock (sPIS).

    The sPIS measures the cumulative inventory performance over time.
    It is useful for intermittent demand forecasting.

    Formula:
        sPIS = sum(cumsum(forecast - holdout)) / scale

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    scale : float
        Scale parameter (typically mean of non-zero actuals).
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Scaled Periods-In-Stock.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> scale = np.mean(np.abs(holdout[holdout != 0]))
    >>> round(float(spis(holdout, forecast, scale)), 4)
    1.2364

    References
    ----------
    Wallstrom P., Segerstedt A. (2010). Evaluation of forecasting error measurements
    and techniques for intermittent demand. International Journal of Production
    Economics, 128, pp.625-636.
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout, forecast = _validate_inputs(holdout, forecast, na_rm=False)
    cumsum_errors = np.cumsum(forecast - holdout)
    total = np.nansum(cumsum_errors) if na_rm else np.sum(cumsum_errors)

    return float(total / scale)


def sce(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    scale: float | np.floating[Any] | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Scaled Cumulative Error (sCE).

    The sCE measures the cumulative forecast error scaled by a parameter.
    It can detect systematic bias over time.

    Formula:
        sCE = sum(holdout - forecast) / scale

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    scale : float
        Scale parameter.
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Scaled Cumulative Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> scale = np.mean(np.abs(holdout[holdout != 0]))
    >>> round(float(sce(holdout, forecast, scale)), 4)
    -0.2364

    References
    ----------
    Petropoulos F., Kourentzes N. (2015). Forecast combinations for intermittent demand.
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout, forecast = _validate_inputs(holdout, forecast, na_rm)

    return float(np.sum(holdout - forecast, axis=0) / scale)


def gmrae(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    benchmark: np.ndarray | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> float:
    """Geometric Mean Relative Absolute Error (GMRAE).

    The GMRAE compares the absolute errors of a forecast to a benchmark
    using geometric mean. It is robust to extreme values.

    Formula:
        ``GMRAE = exp(mean(log(|error| / |benchmark_error|)))``

    Parameters
    ----------
    holdout : np.ndarray
        Holdout (observed) values.
    forecast : np.ndarray
        Forecasted values.
    benchmark : np.ndarray
        Benchmark forecast values.
    na_rm : bool, default=True
        Remove NA values.
    actual : np.ndarray, optional
        Deprecated alias of ``holdout``, kept for one release.

    Returns
    -------
    float
        Geometric Mean Relative Absolute Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.2, 3.2, 3.9, 5.1,
    ...                      6.2, 7.1, 8.2, 9.2, 10.1])
    >>> benchmark = np.array([0.5, 1.5, 2.5, 3.5, 4.5,
    ...                      5.5, 6.5, 7.5, 8.5, 9.5])
    >>> round(float(gmrae(holdout, forecast, benchmark)), 4)
    0.2828

    References
    ----------
    See Davydenko & Fildes (2013).
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout = np.asarray(holdout, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)

    lengths = [len(holdout), len(forecast), len(benchmark)]
    if len(set(lengths)) > 1:
        raise ValueError("Lengths of holdout, forecast, and benchmark must match")

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratios = np.log(np.abs((holdout - forecast) / (holdout - benchmark)))
        if na_rm:
            log_ratios = log_ratios[~np.isnan(log_ratios)]

        return np.exp(np.mean(log_ratios))


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

    if benchmark not in ("naive", "mean"):
        raise ValueError("benchmark must be 'naive' or 'mean'")
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
