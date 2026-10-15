"""Half-moment measures for data analysis.

This module provides functions for:
1. Half-moment based measures: hm, ham, asymmetry, extremity, cextremity
2. Mean Root Error (MRE)

References
----------
- Svetunkov I., Kourentzes N., Svetunkov S. "Half Central Moment for Data Analysis".
  Working Paper of Department of Management Science, Lancaster University, 2023:3, 1-21.
- Kourentzes N. (2014). The Bias Coefficient: a new metric for forecast bias.
"""

import warnings

import numpy as np


def hm(
    x: np.ndarray,
    C: float | None = None,
    *,
    center: float | None = None,
) -> complex:
    """Half Moment (HM).

    The half moment is a measure that captures the asymmetry of a distribution
    by considering only the positive or negative deviations from a center point.

    Formula:
        hm = mean(sqrt(x - C))

    where C is the centering parameter (default is mean of x).

    Parameters
    ----------
    x : np.ndarray
        Data vector.
    C : float, optional
        Centering parameter. If None, uses mean(x).
    center : float, optional
        Deprecated alias of ``C``, kept for one release.

    Returns
    -------
    complex
        Half moment (complex number when mean of sqrt is taken).

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> complex(hm(x))  # doctest: +ELLIPSIS
    (0.4828...+0.4828...j)

    References
    ----------
    Svetunkov I., Kourentzes N., Svetunkov S. "Half Central Moment for Data Analysis".
    """
    if center is not None:
        warnings.warn(
            "'center' is deprecated, use 'C' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        C = center

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if C is None:
        C = np.mean(x)

    return np.mean(np.sqrt(np.array(x - C, dtype=complex)))


def ham(
    x: np.ndarray,
    C: float | None = None,
    *,
    center: float | None = None,
) -> float:
    """Half Absolute Moment (HAM).

    The half absolute moment is the absolute value of the half moment.
    It captures the magnitude of deviations without direction.

    Formula:
        ham = mean(sqrt(abs(x - C)))

    Parameters
    ----------
    x : np.ndarray
        Data vector.
    C : float, optional
        Centering parameter. If None, uses mean(x).
    center : float, optional
        Deprecated alias of ``C``, kept for one release.

    Returns
    -------
    float
        Half absolute moment.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> round(float(ham(x)), 4)
    0.9657

    References
    ----------
    Svetunkov I., Kourentzes N., Svetunkov S. "Half Central Moment for Data Analysis".
    """
    if center is not None:
        warnings.warn(
            "'center' is deprecated, use 'C' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        C = center

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if C is None:
        C = np.mean(x)

    return float(np.mean(np.sqrt(np.abs(x - C))))


def asymmetry(
    x: np.ndarray,
    C: float | None = None,
    *,
    center: float | None = None,
) -> float:
    """Asymmetry coefficient.

    The asymmetry coefficient measures the asymmetry of a distribution
    around a center point based on the half moment.

    Formula:
        asymmetry = 1 - Arg(hm(x, C)) / (pi/4)

    Values:
        - 1: Maximum negative asymmetry (all values below center)
        - 0: Symmetric distribution
        - -1: Maximum positive asymmetry (all values above center)

    Parameters
    ----------
    x : np.ndarray
        Data vector.
    C : float, optional
        Centering parameter. If None, uses mean(x).
    center : float, optional
        Deprecated alias of ``C``, kept for one release.

    Returns
    -------
    float
        Asymmetry coefficient.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> round(float(asymmetry(x)), 4)
    0.0
    >>> x_skewed = np.array([1, 1, 1, 4, 5])
    >>> round(float(asymmetry(x_skewed)), 4)
    -0.1327

    References
    ----------
    Svetunkov I., Kourentzes N., Svetunkov S. "Half Central Moment for Data Analysis".
    """
    if center is not None:
        warnings.warn(
            "'center' is deprecated, use 'C' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        C = center

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if C is None:
        C = np.mean(x)

    return 1 - np.angle(hm(x, C)) / (np.pi / 4)


def extremity(
    x: np.ndarray,
    C: float | None = None,
    *,
    center: float | None = None,
) -> float:
    """Extremity coefficient.

    The extremity coefficient measures how extreme the values in a distribution
    are relative to the center.

    Formula:
        extremity = 2 * (ham(x, C) / mean((x - C)^2)^0.25)^a - 1

    where a = log(0.5)/log(2*3^(-0.75)). As in R, NA values in x give NA.

    Parameters
    ----------
    x : np.ndarray
        Data vector.
    C : float, optional
        Centering parameter. If None, uses mean(x).
    center : float, optional
        Deprecated alias of ``C``, kept for one release.

    Returns
    -------
    float
        Extremity coefficient.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> round(float(extremity(x)), 4)
    -0.3364

    References
    ----------
    Svetunkov I., Kourentzes N., Svetunkov S. "Half Central Moment for Data Analysis".
    """
    if center is not None:
        warnings.warn(
            "'center' is deprecated, use 'C' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        C = center

    x = np.asarray(x, dtype=float)

    if C is None:
        C = np.nanmean(x)

    exponent = np.log(0.5) / np.log(2 * 3**-0.75)
    with np.errstate(divide="ignore", invalid="ignore"):
        ch = ham(x, C) / np.mean((x - C) ** 2) ** 0.25

    return 2 * (ch**exponent) - 1


def cextremity(
    x: np.ndarray,
    C: float | None = None,
    *,
    center: float | None = None,
) -> complex:
    """Complex Extremity coefficient.

    The complex extremity coefficient extends the extremity coefficient
    to capture both magnitude and phase information.

    Formula:
        cextremity = 2 * (CH * 2)^(log(0.5)/log(2*3^(-0.75))) - 1

    where CH = hm(x, C) / mean((x - C)^2)^0.25. As in R, NA values in x
    give NA.

    Parameters
    ----------
    x : np.ndarray
        Data vector.
    C : float, optional
        Centering parameter. If None, uses mean(x).
    center : float, optional
        Deprecated alias of ``C``, kept for one release.

    Returns
    -------
    complex
        Complex extremity coefficient.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> cextremity(x)  # doctest: +ELLIPSIS
    (-0.3364...-0.3364...j)

    References
    ----------
    Svetunkov I., Kourentzes N., Svetunkov S. "Half Central Moment for Data Analysis".
    """
    if center is not None:
        warnings.warn(
            "'center' is deprecated, use 'C' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        C = center

    x = np.asarray(x, dtype=float)

    if C is None:
        C = np.nanmean(x)

    exponent = np.log(0.5) / np.log(2 * 3**-0.75)
    with np.errstate(divide="ignore", invalid="ignore"):
        ch = hm(x, C) / np.mean((x - C) ** 2) ** 0.25

    real_part = 2 * (np.real(ch) * 2) ** exponent - 1
    imag_part = 2 * (np.imag(ch) * 2) ** exponent - 1

    return complex(real_part, imag_part)


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


def mre(
    holdout: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    na_rm: bool = True,
    *,
    actual: np.ndarray | None = None,
) -> complex:
    """Mean Root Error (MRE).

    The MRE measures the average signed root error. It is used as a bias
    coefficient (Kourentzes, 2014) and can detect systematic over or
    under-forecasting.

    Formula:
        MRE = mean(sqrt(holdout - forecast))

    Note: When holdout < forecast, the roots are complex, so the result is
    a complex number, as in R.

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
    complex
        Mean Root Error.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> mre(holdout, forecast)  # doctest: +ELLIPSIS
    (0.0632...+0.2159...j)

    References
    ----------
    Kourentzes N. (2014). The Bias Coefficient: a new metric for forecast bias.
    https://kourentzes.com/forecasting/2014/12/17/the-bias-coefficient-a-new-metric-for-forecast-bias/
    """
    if actual is not None:
        warnings.warn(
            "'actual' is deprecated, use 'holdout' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        holdout = actual

    holdout, forecast = _validate_inputs(holdout, forecast, na_rm)
    errors = holdout - forecast
    complex_roots = np.sqrt(np.array(errors, dtype=complex))
    return complex(np.mean(complex_roots))
