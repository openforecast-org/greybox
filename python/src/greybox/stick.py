"""Seasonality, Trend, and Irregular Contribution Kit (STI).

This module is a Python port of R's ``greybox::stick``. It decomposes the
variance of a time series into seasonal, trend and irregular parts based on
an Analysis of Variance (ANOVA) of the series on the seasonal and trend
factors and measures the contribution of each component.

The function implements the Seasonality, Trend, and Irregular (STI)
classification of Hans Levenbach (Levenbach, H. (2021). Four P's in a Pod:
e-Commerce Forecasting and Planning for Supply Chain Practitioners.
Independently published. ISBN 979-8461733575).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


class StickResult:
    """Result of :func:`stick`. Mirrors R's ``stick`` S3 class.

    Attributes
    ----------
    y : numpy.ndarray
        The original time series.
    lags : numpy.ndarray
        The seasonal lags used in the analysis (unique, ascending).
    anova : pandas.DataFrame
        The ANOVA table, with one row per seasonal lag, a ``trend`` row
        (when a trend was fitted) and a ``Residuals`` row. Columns are
        ``Df``, ``Sum Sq``, ``Mean Sq``, ``F value`` and ``Pr(>F)``.
    strength : pandas.Series
        The strength of each component: one entry per seasonal lag, the
        ``trend`` and the ``irregular`` component. The values are the
        shares of the respective Sum of Squares in the total Sum of
        Squares and sum up to one.
    """

    __slots__ = ("anova", "lags", "strength", "y")

    def __init__(
        self,
        y: np.ndarray,
        lags: np.ndarray,
        anova: pd.DataFrame,
        strength: pd.Series,
    ):
        self.y = y
        self.lags = lags
        self.anova = anova
        self.strength = strength

    def __str__(self) -> str:
        digits = 4
        lags_str = ", ".join(str(int(lag)) for lag in self.lags)
        rounded = self.strength.round(digits)
        return (
            "Seasonality, Trend, and Irregular Contribution Kit\n"
            f"Seasonal lags: {lags_str}\n\n"
            "Strength of the components:\n"
            f"{rounded.to_string()}"
        )

    def __repr__(self) -> str:
        lags_str = ", ".join(str(int(lag)) for lag in self.lags)
        return f"StickResult(lags=[{lags_str}])"

    def _reshape_to_matrix(self, period: int) -> np.ndarray:
        """Reshape ``y`` into a ``[period x cycle]`` matrix.

        Columns are cycles, rows are positions within the cycle. The last
        (incomplete) cycle is padded with NaNs.
        """
        y_values = np.asarray(self.y, dtype=float)
        nobs = len(y_values)
        n_cycles = int(np.ceil(nobs / period))
        padded = np.full(n_cycles * period, np.nan)
        padded[:nobs] = y_values
        # C-order reshape gives row i = cycle i; transpose so that the
        # columns are cycles and the rows are positions within the cycle.
        return padded.reshape(n_cycles, period).T

    def _plot_seasonal(self, ax, period: int, **kwargs) -> None:
        """Draw the seasonal plot for one lag on the given axes."""
        matrix = self._reshape_to_matrix(period)
        n_row, n_col = matrix.shape
        x_idx = np.arange(1, n_row + 1)
        # One grey line per cycle, light-to-dark gradient.
        shades = 0.85 * np.arange(1, n_col + 1) / n_col
        for j in range(n_col):
            ax.plot(x_idx, matrix[:, j], color=str(shades[j]), **kwargs)
        # The average seasonal profile across the cycles.
        ax.plot(
            x_idx,
            np.nanmean(matrix, axis=1),
            color="black",
            linewidth=2,
            linestyle="--",
        )
        ax.set_title(f"Seasonal plot (lag {int(period)})")
        ax.set_xlabel("Period within the cycle")
        ax.set_ylabel("Value")

    def _plot_trend(self, ax, period: int, **kwargs) -> None:
        """Draw the trend plot (uses the longest lag) on the given axes."""
        matrix = self._reshape_to_matrix(period)
        n_row, n_col = matrix.shape
        x_idx = np.arange(1, n_col + 1)
        # One grey line per seasonal position, drawn across the cycles.
        shades = 0.85 * np.arange(1, n_row + 1) / n_row
        for i in range(n_row):
            ax.plot(x_idx, matrix[i, :], color=str(shades[i]), **kwargs)
        # The average level per cycle: the trend.
        ax.plot(
            x_idx,
            np.nanmean(matrix, axis=0),
            color="black",
            linewidth=2,
            linestyle="--",
        )
        ax.set_title("Trend plot")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Value")

    def plot(self, which=None, axes=None, **kwargs):
        """Plot the seasonal and trend components.

        Mirrors R's ``plot.stick()``. For every seasonal lag a "seasonal
        plot" is produced (the series reshaped into one grey line per
        cycle, with the average seasonal profile overlaid as a bold dashed
        black line); the final plot is the "trend plot" (one grey line per
        seasonal position drawn across the cycles, with the average level
        per cycle -- the trend -- overlaid as a bold dashed black line).

        Parameters
        ----------
        which : int or sequence of int, optional
            Which plots to produce: ``1, ..., k-1`` correspond to the
            seasonal plots for each of the seasonal lags (in ascending
            order), while ``k`` corresponds to the trend plot (the last
            one). If ``None`` (the default), all the plots are produced.
            Values outside the valid range are dropped with a warning.
        axes : matplotlib.axes.Axes or sequence of Axes, optional
            Axes to draw on, one per requested plot. A new figure with a
            row of subplots is created if ``None``.
        **kwargs
            Forwarded to :meth:`matplotlib.axes.Axes.plot` for the grey
            component lines.

        Returns
        -------
        numpy.ndarray of matplotlib.axes.Axes
            The axes containing the plots.
        """
        import matplotlib.pyplot as plt

        n_lags = len(self.lags)
        has_trend = "trend" in self.strength.index
        n_plots = n_lags + (1 if has_trend else 0)

        if which is None:
            which = list(range(1, n_plots + 1))
        elif np.isscalar(which):
            which = [int(which)]
        else:
            which = [int(w) for w in which]

        valid = [w for w in which if 1 <= w <= n_plots]
        if len(valid) != len(which):
            warnings.warn(
                f"Some values of 'which' are outside the range "
                f"[1, {n_plots}] and are dropped.",
                stacklevel=2,
            )
        which = valid
        if len(which) == 0:
            return np.array([], dtype=object)

        if axes is None:
            _, axes = plt.subplots(1, len(which))
        axes = np.atleast_1d(axes)

        lag_max = int(np.max(self.lags))
        for ax, w in zip(axes, which):
            if w <= n_lags:
                self._plot_seasonal(ax, int(self.lags[w - 1]), **kwargs)
            else:
                self._plot_trend(ax, lag_max, **kwargs)
        return axes


def _residual_sum_of_squares(y: np.ndarray, design: np.ndarray) -> float:
    """Residual sum of squares of an OLS fit of ``y`` on ``design``."""
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coef
    return float(residuals @ residuals)


def _factor_dummies(codes: np.ndarray) -> np.ndarray:
    """Treatment-coded dummies for a factor (first level dropped)."""
    levels = np.unique(codes)
    # Drop the first level for identifiability (treatment contrasts).
    return np.column_stack([(codes == level).astype(float) for level in levels[1:]])


def stick(y, lags=None) -> StickResult:
    """Seasonality, Trend, and Irregular Contribution Kit.

    Decomposes the variance of a time series into seasonal, trend and
    irregular parts based on an Analysis of Variance (ANOVA) of the series
    on the seasonal and trend factors, and measures the contribution of
    each component. This is a Python port of R's :func:`greybox::stick`.

    A data frame is formed internally with the response ``y``, a
    categorical (factor) variable for each of the provided seasonal
    ``lags`` and a "trend" factor. For a monthly series with ``lags=12``,
    the seasonal factor takes values ``1..12`` repeated throughout the
    sample (the month of the year), while the trend factor takes values
    ``1..ceil(T/12)``, each repeated 12 times (the year). The trend factor
    is constructed based on the longest of the provided lags. An ANOVA is
    then applied to ``y ~ seasonal + trend`` and the strength of each
    component is measured as the share of the respective Sum of Squares in
    the total Sum of Squares. The irregular component corresponds to the
    share of the residual Sum of Squares; the strengths sum up to one.

    The function implements the Seasonality, Trend, and Irregular (STI)
    classification of Hans Levenbach.

    Parameters
    ----------
    y : array-like
        The time series to analyse.
    lags : int or sequence of int
        The seasonal lags (periodicities) in the data, e.g.
        ``lags=[24, 168]`` for hourly data or ``lags=12`` for monthly
        data. Values not greater than one are dropped.

    Returns
    -------
    StickResult
        Object with the original data, the ``lags`` used, the ANOVA table
        and the ``strength`` of each component.

    References
    ----------
    Levenbach, H. (2021). Four P's in a Pod: e-Commerce Forecasting and
    Planning for Supply Chain Practitioners. Independently published.
    ISBN 979-8461733575.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(1, 121)
    >>> y = 100 + 0.3 * t + 20 * np.sin(2 * np.pi * t / 12)
    >>> result = stick(y, lags=12)
    >>> bool(np.isclose(result.strength.sum(), 1.0))
    True
    """
    if lags is None:
        raise ValueError("'lags' must be provided (e.g. lags=12).")

    y_values = np.asarray(y, dtype=float).ravel()
    nobs = len(y_values)

    # Only seasonal lags above one make sense; keep unique and sorted.
    lags = np.atleast_1d(np.asarray(lags))
    lags = np.unique(lags[lags > 1]).astype(int)
    if len(lags) == 0:
        raise ValueError(
            "No seasonal lags greater than one were provided. "
            "stick() needs at least one seasonal periodicity."
        )

    # Drop the lags that are too long to form at least two full cycles.
    too_long = lags >= nobs
    if np.any(too_long):
        warnings.warn(
            "The following lags are not smaller than the number of "
            "observations and are dropped: "
            f"{', '.join(str(int(lag)) for lag in lags[too_long])}.",
            stacklevel=2,
        )
        lags = lags[~too_long]
    if len(lags) == 0:
        raise ValueError("None of the provided lags is shorter than the sample.")

    positions = np.arange(nobs)

    # Build the ordered list of terms: a seasonal factor per lag, then the
    # trend ("year") factor based on the longest lag.
    term_names = [f"seasonal{int(lag)}" for lag in lags]
    term_codes = [positions % int(lag) for lag in lags]

    lag_max = int(np.max(lags))
    trend_codes = positions // lag_max
    has_trend = len(np.unique(trend_codes)) > 1
    if has_trend:
        term_names.append("trend")
        term_codes.append(trend_codes)

    # Sequential (Type I) ANOVA: add each term's dummies in order and
    # record the reduction in the residual sum of squares.
    intercept = np.ones((nobs, 1))
    design = intercept
    rss_prev = _residual_sum_of_squares(y_values, design)

    df = []
    sum_sq = []
    for codes in term_codes:
        dummies = _factor_dummies(codes)
        design = np.column_stack([design, dummies])
        rss_curr = _residual_sum_of_squares(y_values, design)
        df.append(dummies.shape[1])
        sum_sq.append(rss_prev - rss_curr)
        rss_prev = rss_curr

    # Residual ("irregular") row.
    df_residual = nobs - design.shape[1]
    sum_sq_residual = rss_prev

    rows = term_names + ["Residuals"]
    df_all = np.array(df + [df_residual], dtype=float)
    sum_sq_all = np.array(sum_sq + [sum_sq_residual], dtype=float)
    mean_sq = sum_sq_all / df_all

    # F values and p-values for the model terms (not for the residuals).
    mean_sq_residual = mean_sq[-1]
    f_value = np.full(len(rows), np.nan)
    p_value = np.full(len(rows), np.nan)
    if df_residual > 0:
        f_value[:-1] = mean_sq[:-1] / mean_sq_residual
        p_value[:-1] = stats.f.sf(f_value[:-1], df_all[:-1], df_residual)

    anova = pd.DataFrame(
        {
            "Df": df_all,
            "Sum Sq": sum_sq_all,
            "Mean Sq": mean_sq,
            "F value": f_value,
            "Pr(>F)": p_value,
        },
        index=rows,
    )

    # Strength of each component = its share of the total Sum of Squares.
    strength_values = sum_sq_all / sum_sq_all.sum()
    strength_names = term_names + ["irregular"]
    strength = pd.Series(strength_values, index=strength_names)

    return StickResult(y=y_values, lags=lags, anova=anova, strength=strength)
