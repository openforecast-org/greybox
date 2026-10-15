"""Regression for Multiple Comparison with the Best (RMCB).

This module is a Python port of R's ``greybox::rmcb``. RMCB is a
regression-based version of the Nemenyi / MCB test for comparing forecasting
methods (Demsar, 2006). It transforms the data into ranks and constructs a
regression on dummy variables of the type

    y = b' X + e,

where ``y`` is the vector of the ranks of the provided data, ``X`` is the
matrix of dummy variables for each column of the data (a forecasting method),
``b`` is the vector of coefficients and ``e`` is the error term. Because the
data is ranked, the test compares the medians of the methods and produces
plots based on the resulting confidence intervals.

The critical distances are by default calculated from the Studentised range
statistic, so the test gives exactly the same result as the Nemenyi test.
Several alternatives are available through the ``distribution`` argument:
``"dnorm"`` relies on the Student distribution and the covariance matrix of
the parameters; any other distribution (e.g. ``"dlnorm"``) fits a regression
on the original data with the assumed distribution and extracts the standard
errors from it.

The :class:`RMCBResult` returned by :func:`rmcb` provides a ``plot`` method
that produces either an ``"mcb"`` or a ``"lines"`` style of plot, mirroring
R's ``plot.rmcb``.

References
----------
Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data
Sets. Journal of Machine Learning Research, 7, 1-30.
https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .fitters import _lm_fit

# Colour palette shared with R's plot.rmcb for the "lines" plot.
_LINE_COLOURS = (
    "#0DA0DC",
    "#17850C",
    "#EA3921",
    "#E1C513",
    "#BB6ECE",
    "#5DAD9D",
)


class RMCBResult:
    """Result of :func:`rmcb`. Mirrors R's ``rmcb`` S3 class.

    Attributes
    ----------
    mean : pandas.Series
        Mean values (mean ranks for the default ``"tukey"`` distribution) for
        each method, sorted in ascending order and indexed by method name.
    interval : pandas.DataFrame
        Confidence intervals for each method, with the lower and upper bounds
        as columns (named after the corresponding percentiles) and the methods
        as rows (in the same ascending order as ``mean``).
    vlines : pandas.DataFrame
        Coordinates used for the ``"lines"`` plot, marking the groups of
        methods. Columns are ``"Group starts"`` and ``"Group ends"`` (1-based
        positions into the sorted methods); rows are ``Group1 .. GroupK``.
    groups : pandas.DataFrame
        Boolean table of group membership: ``True`` if a method belongs to a
        group, ``False`` otherwise. Rows are methods, columns are groups.
    methods : pandas.DataFrame
        Boolean pairwise table: ``True`` if the intervals of two methods
        overlap. Rows and columns are methods.
    p_value : float
        p-value of the significance test of the model (Friedman test for the
        default ``"tukey"`` distribution, an F test for ``"dnorm"`` and a
        chi-squared test otherwise).
    level : float
        Confidence level.
    model : object
        The fitted model used for the calculation of the intervals. A dict of
        OLS quantities for the rank-based distributions, or the fitted
        :class:`~greybox.ALM` instance for the general case.
    select : int
        1-based position (in the sorted order) of the highlighted method.
    distribution : str
        The distribution used for the calculations.
    correction : str
        The type of correction used (``"none"`` or ``"hac"``).
    factor : float
        The long-run variance factor (one if ``correction="none"``).
    """

    __slots__ = (
        "correction",
        "distribution",
        "factor",
        "groups",
        "interval",
        "level",
        "mean",
        "methods",
        "model",
        "p_value",
        "select",
        "vlines",
    )

    def __init__(
        self,
        mean: pd.Series,
        interval: pd.DataFrame,
        vlines: pd.DataFrame,
        groups: pd.DataFrame,
        methods: pd.DataFrame,
        p_value: float,
        level: float,
        model: Any,
        select: int,
        distribution: str,
        correction: str = "none",
        factor: float = 1.0,
    ):
        self.correction = correction
        self.factor = factor
        self.mean = mean
        self.interval = interval
        self.vlines = vlines
        self.groups = groups
        self.methods = methods
        self.p_value = p_value
        self.level = level
        self.model = model
        self.select = select
        self.distribution = distribution

    def __str__(self) -> str:
        n_methods = len(self.mean)
        # The model carries the total number of rows (obs * n_methods).
        n_total = _model_nobs(self.model)
        n_obs = n_total // n_methods if n_methods else 0
        out = (
            "Regression for Multiple Comparison with the Best\n"
            f"The significance level is {(1 - self.level) * 100:g}%\n"
            f"The number of observations is {n_obs}, "
            f"the number of methods is {n_methods}\n"
            f"Significance test p-value: {round(self.p_value, 5)}"
        )
        if self.correction != "none":
            out += (
                f"\nCritical distances corrected for dependence "
                f"({self.correction}), long-run variance factor: "
                f"{round(self.factor, 3)}"
            )
        return out

    def __repr__(self) -> str:
        return (
            f"RMCBResult(distribution={self.distribution!r}, "
            f"n_methods={len(self.mean)}, level={self.level})"
        )

    def plot(
        self,
        outplot: str = "mcb",
        select: int | str | None = None,
        ax=None,
        **kwargs,
    ):
        """Plot the RMCB result. Mirrors R's ``plot.rmcb``.

        Parameters
        ----------
        outplot : {"mcb", "lines"}
            The style of the plot. ``"mcb"`` draws one vertical interval per
            method with the selected method's bounds highlighted by horizontal
            dashed lines; ``"lines"`` draws the groups of indistinguishable
            methods as vertical segments.
        select : int or str, optional
            The method to highlight. Either a method name, a 0-based column
            index (into the sorted methods) or ``None`` to keep the method
            selected by :func:`rmcb` (the one with the lowest mean).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure and axes are created if ``None``.
        **kwargs
            Overrides for plot decorations: ``main``/``title``, ``xlab``,
            ``ylab``, ``xlim``, ``ylim``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt

        if outplot not in ("mcb", "lines"):
            raise ValueError(f"outplot must be 'mcb' or 'lines', got {outplot!r}.")

        names_methods = list(self.mean.index)
        n_methods = len(names_methods)

        # Resolve the selected method (1-based position in the sorted order).
        if select is None:
            select_pos = self.select
        elif isinstance(select, str):
            select_pos = names_methods.index(select) + 1
        else:
            select_pos = int(select) + 1
        select_idx = select_pos - 1

        if ax is None:
            _, ax = plt.subplots()

        intervals = self.interval.to_numpy()
        means = self.mean.to_numpy()
        n_groups = self.groups.shape[1]

        # Colours follow R: highlight overlapping methods if >1 group.
        if n_groups > 1:
            point_col = ["#0DA0DC"] * n_methods
            overlap = self.methods.to_numpy()[:, select_idx]
            for i in range(n_methods):
                if overlap[i]:
                    point_col[i] = "#0C6385"
            line_col = "#0DA0DC"
        else:
            point_col = ["darkgrey"] * n_methods
            line_col = "grey"

        main = kwargs.get(
            "main",
            kwargs.get(
                "title",
                f"The p-value from the significance test is "
                f"{self.p_value:.3f}.\n"
                f"{self.level * 100:g}% confidence intervals constructed.",
            ),
        )

        if outplot == "mcb":
            self._plot_mcb(
                ax,
                intervals,
                means,
                names_methods,
                point_col,
                line_col,
                select_idx,
                main,
                kwargs,
            )
        else:
            self._plot_lines(
                ax,
                names_methods,
                line_col,
                main,
                kwargs,
            )
        return ax

    def _plot_mcb(
        self,
        ax,
        intervals: np.ndarray,
        means: np.ndarray,
        names_methods: list,
        point_col: list,
        line_col: str,
        select_idx: int,
        main: str,
        kwargs: dict,
    ) -> None:
        """Draw the "mcb" style plot (intervals + points)."""
        n_methods = len(names_methods)
        positions = np.arange(1, n_methods + 1)

        xlim = kwargs.get("xlim", (0, n_methods + 1))
        if "ylim" in kwargs:
            ylim = kwargs["ylim"]
        else:
            lo = float(np.min(intervals))
            hi = float(np.max(intervals))
            ylim = (lo - 0.1, hi + 0.1)

        # One vertical line per method spanning its confidence interval.
        for i in range(n_methods):
            ax.plot(
                [positions[i], positions[i]],
                [intervals[i, 0], intervals[i, 1]],
                color=line_col,
                linewidth=2,
            )
        # Points at the mean of each method.
        for i in range(n_methods):
            ax.plot(
                positions[i],
                means[i],
                marker="o",
                color=point_col[i],
                linestyle="None",
            )

        # Horizontal dashed lines at the selected method's interval bounds.
        ax.axhline(intervals[select_idx, 0], linewidth=2, linestyle="--", color="grey")
        ax.axhline(intervals[select_idx, 1], linewidth=2, linestyle="--", color="grey")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(positions)
        rotation = 90 if n_methods > 5 else 0
        ax.set_xticklabels(names_methods, rotation=rotation)
        ax.set_xlabel(kwargs.get("xlab", ""))
        ax.set_ylabel(kwargs.get("ylab", ""))
        ax.set_title(main)

    def _plot_lines(
        self,
        ax,
        names_methods: list,
        line_col: str,
        main: str,
        kwargs: dict,
    ) -> None:
        """Draw the "lines" style plot (groups as vertical segments)."""
        n_methods = len(names_methods)
        vlines = self.vlines.to_numpy()
        k = vlines.shape[0]
        group_elements = self.groups.to_numpy().sum(axis=0)

        colours = [_LINE_COLOURS[i % len(_LINE_COLOURS)] for i in range(k)]

        xlim = kwargs.get("xlim", (0, k + 0.2))
        ylim = kwargs.get("ylim", (1 - 0.2, n_methods + 0.2))

        if k > 1:
            for i in range(k):
                if group_elements[i] > 1:
                    ax.plot(
                        [i + 1, i + 1],
                        [vlines[i, 0], vlines[i, 1]],
                        color=colours[i],
                        linewidth=2,
                    )
                    ax.plot(
                        [0, i + 1],
                        [vlines[i, 0], vlines[i, 0]],
                        color="gray",
                        linestyle="--",
                    )
                    ax.plot(
                        [0, i + 1],
                        [vlines[i, 1], vlines[i, 1]],
                        color="gray",
                        linestyle="--",
                    )
                else:
                    ax.plot(
                        [i + 1, i + 1],
                        [vlines[i, 0], vlines[i, 1]],
                        color=colours[i],
                        marker="o",
                        linestyle="None",
                    )
        else:
            ax.plot([1, 1], [1, n_methods], color=line_col, linewidth=2)
            ax.plot(
                [0, 1],
                [vlines[0, 0], vlines[0, 0]],
                color=line_col,
                linestyle="--",
            )
            ax.plot(
                [0, 1],
                [vlines[0, 1], vlines[0, 1]],
                color=line_col,
                linestyle="--",
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_yticks(np.arange(1, n_methods + 1))
        ax.set_yticklabels(names_methods)
        ax.set_xlabel(kwargs.get("xlab", ""))
        ax.set_ylabel(kwargs.get("ylab", ""))
        ax.set_title(main)


def _model_nobs(model: Any) -> int:
    """Total number of rows used by the fitted model (obs * n_methods)."""
    if isinstance(model, dict):
        return int(model.get("nobs", 0))
    nobs = getattr(model, "nobs", 0)
    return int(nobs() if callable(nobs) else nobs)


def _rank_rows(data: np.ndarray, na_last: bool, ties_method: str) -> np.ndarray:
    """Rank each row of ``data`` independently (mirrors R ``rank`` per row)."""
    ranked = np.empty_like(data, dtype=float)
    for i in range(data.shape[0]):
        row = data[i]
        finite = np.isfinite(row)
        if finite.all():
            ranked[i] = stats.rankdata(row, method=ties_method)
            continue
        # Rank the finite entries; place the NaNs first or last.
        order = stats.rankdata(row[finite], method=ties_method)
        out = np.full(row.shape, np.nan)
        out[finite] = order
        n_finite = int(finite.sum())
        n_missing = row.size - n_finite
        if na_last:
            out[~finite] = np.arange(n_finite + 1, n_finite + n_missing + 1)
        else:
            out[~finite] = np.arange(1, n_missing + 1)
            out[finite] = order + n_missing
        ranked[i] = out
    return ranked


def _ols_fit(x: np.ndarray, y: np.ndarray, df_residual: int) -> dict:
    """Ordinary least squares fit returning the quantities rmcb needs."""
    coef, residuals, _ = _lm_fit(x, y)
    fitted = y - residuals
    ssr = float(residuals @ residuals)
    sigma2 = ssr / df_residual
    xtx = x.T @ x
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    vcov = sigma2 * xtx_inv
    return {
        "coef": coef,
        "fitted": fitted,
        "residuals": residuals,
        "ssr": ssr,
        "sigma2": sigma2,
        "vcov": vcov,
        "df_residual": df_residual,
        "actuals": y,
        "nobs": len(y),
    }


def _long_run_factor(data: np.ndarray, log: bool = False) -> float:
    """Long-run variance factor of the columns (Newey-West, Bartlett kernel).

    The factor is calculated for each column, averaged over the columns and
    bounded below by one. Mirrors R's ``rmcbLongRunFactor()``.
    """
    data = np.asarray(data, dtype=float)
    if log:
        data = np.log(data)
    obs = data.shape[0]
    lag_max = int(np.floor(4 * (obs / 100) ** (2 / 9)))
    if lag_max < 1:
        return 1.0
    factors = []
    for j in range(data.shape[1]):
        x = data[:, j]
        x = x[~np.isnan(x)]
        n = len(x)
        if n - 1 < lag_max:
            factors.append(np.nan)
            continue
        xc = x - x.mean()
        denom = float(np.sum(xc**2))
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.array(
                [np.sum(xc[: n - k] * xc[k:]) / denom for k in range(1, lag_max + 1)]
            )
        if not np.all(np.isfinite(rho)):
            factors.append(np.nan)
            continue
        weights = 1 - np.arange(1, lag_max + 1) / (lag_max + 1)
        factors.append(1 + 2 * float(np.sum(weights * rho)))
    factors_arr = np.asarray(factors, dtype=float)
    if np.all(np.isnan(factors_arr)):
        return 1.0
    return float(max(1.0, np.nanmean(factors_arr)))


def rmcb(
    data,
    level: float = 0.95,
    outplot: str = "mcb",
    select: int | str | None = None,
    distribution: str = "tukey",
    na_last: bool = True,
    ties_method: str = "average",
    correction: str = "none",
) -> RMCBResult:
    """Regression for Multiple Comparison with the Best.

    This is a Python port of R's :func:`greybox::rmcb`. It performs a
    regression-based version of the Nemenyi / MCB test, comparing forecasting
    methods by ranking the data and fitting a dummy-variable regression on the
    ranks (or, for non-rank distributions, on the original data).

    Parameters
    ----------
    data : array-like or pandas.DataFrame
        Matrix with observations (series) in rows and methods in columns.
        Column names are used as method names when ``data`` is a DataFrame.
    level : float, default 0.95
        The width of the confidence interval.
    outplot : {"mcb", "lines", "none"}, default "mcb"
        Stored for parity with R and used as the default style of
        :meth:`RMCBResult.plot`. Unlike R, :func:`rmcb` does **not** draw the
        plot as a side effect; call :meth:`RMCBResult.plot` instead.
    select : int or str, optional
        The method to highlight on the plot. Either a method name or a 0-based
        column index. If ``None`` (the default), the method with the lowest
        mean is selected.
    distribution : str, default "tukey"
        ``"tukey"`` (the Nemenyi test, based on the Studentised range),
        ``"dnorm"`` (Student distribution and the covariance matrix),
        ``"dlnorm"`` (log-normal regression) or any other distribution
        supported by :class:`~greybox.ALM`.
    na_last : bool, default True
        Whether missing values are ranked last (passed through to the ranking,
        mirroring R's ``rank(na.last=)``).
    ties_method : str, default "average"
        Method for handling ties when ranking (mirrors R's
        ``rank(ties.method=)``).
    correction : {"none", "hac"}, default "none"
        Correction of the critical distances for the dependence between the
        observations (rows of ``data``). ``"none"`` assumes that the rows are
        independent, as in the original Nemenyi / MCB test. ``"hac"`` accounts
        for the autocorrelation in the rows (e.g. when they are successive
        time points of one time series): the long-run variance factor
        ``f = 1 + 2 * sum_{j=1}^{L} (1 - j/(L+1)) * rho_j`` (Newey-West,
        Bartlett kernel, ``L = floor(4 * (T/100)^(2/9))``) is calculated for
        each column of the data used in the test (ranks in case of ``"tukey"``
        and ``"dnorm"``), averaged over the columns and bounded below by one.
        The test is then done on the effective sample size ``T/f``, which
        widens the critical distances by ``sqrt(f)``.

    Returns
    -------
    RMCBResult
        The fitted RMCB object. Use its :meth:`~RMCBResult.plot` method to
        produce the ``"mcb"`` or ``"lines"`` plots.

    References
    ----------
    Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple
    Data Sets. Journal of Machine Learning Research, 7, 1-30.
    https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf

    Newey, W. K., West, K. D. (1994). Automatic Lag Selection in Covariance
    Matrix Estimation. The Review of Economic Studies, 61(4), 631-653.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.normal(size=(50, 4))
    >>> data[:, 1] += 4
    >>> data[:, 2] += 3
    >>> data[:, 3] += 2
    >>> result = rmcb(data, level=0.95)
    >>> bool((result.mean.to_numpy()[:-1] <= result.mean.to_numpy()[1:]).all())
    True
    """
    if correction not in ("none", "hac"):
        raise ValueError('correction must be one of "none" or "hac".')

    if isinstance(data, pd.DataFrame):
        names_methods = list(data.columns)
        data_arr = data.to_numpy(dtype=float)
    else:
        data_arr = np.asarray(data, dtype=float)
        names_methods = None

    if data_arr.ndim != 2:
        raise ValueError("data must be a 2-D matrix or DataFrame.")

    obs, n_methods = data_arr.shape
    obs_all = obs * n_methods
    if names_methods is None:
        names_methods = [f"Method{i + 1}" for i in range(n_methods)]
    names_methods = [str(name) for name in names_methods]

    # Friedman test on the original data for the default "tukey" distribution.
    friedman_p = None
    if distribution == "tukey":
        columns = [data_arr[:, j] for j in range(n_methods)]
        friedman_p = float(stats.friedmanchisquare(*columns).pvalue)

    # Use ranks for the rank-based distributions.
    if distribution in ("dnorm", "tukey"):
        data_arr = _rank_rows(data_arr, na_last, ties_method)

    # Long-run variance factor for the dependence between the rows of data.
    lr_factor = 1.0
    if correction == "hac":
        lr_factor = _long_run_factor(data_arr, log=(distribution == "dlnorm"))

    # Response in column-major order (R's c(data) / unlist(data.frame)).
    y = data_arr.flatten(order="F")

    # Treatment-coded dummy design: intercept then dummies for methods 2..N.
    dummies = np.zeros((obs_all, n_methods))
    dummies[:, 0] = 1.0
    for i in range(1, n_methods):
        dummies[obs * i : obs * (i + 1), i] = 1.0

    df_residual = obs_all - n_methods

    # Fit the model and extract the coefficients and the intercept SE.
    model: Any
    log_lik_full = None
    log_lik_null = None
    if distribution in ("dnorm", "tukey"):
        model = _ols_fit(dummies, y, df_residual)
        lm_coefs = model["coef"].copy()
        lm_se = float(np.sqrt(model["vcov"][0, 0]))
    elif distribution == "dlnorm":
        model = _ols_fit(dummies, np.log(y), df_residual)
        null_model = _ols_fit(dummies[:, [0]], np.log(y), obs_all - 1)
        lm_coefs = model["coef"].copy()
        lm_se = float(np.sqrt(model["vcov"][0, 0]))
        log_lik_full = _ols_log_lik(model)
        log_lik_null = _ols_log_lik(null_model)
    else:
        from .alm import ALM

        full = ALM(distribution=distribution).fit(dummies, y)
        null = ALM(distribution=distribution).fit(dummies[:, [0]], y)
        model = full
        lm_coefs = np.concatenate([[full.intercept_], full.coef])
        lm_se = float(np.sqrt(full.vcov()[0, 0]))
        # R uses alm's df.residual, which counts the scale as a parameter
        df_residual = full.nobs - full.nparam
        log_lik_full = full.log_lik
        log_lik_null = null.log_lik

    # Convert coefficients to per-method means: each dummy coef + intercept.
    lm_coefs[1:] = lm_coefs[0] + lm_coefs[1:]

    # Critical distance / half-width of the confidence intervals.
    if distribution == "tukey":
        q_stat = (
            stats.studentized_range.ppf(level, n_methods, np.inf)
            * np.sqrt(n_methods * (n_methods + 1) / (12 * obs / lr_factor))
            / 2
        )
    else:
        q_stat = (
            stats.t.ppf((1 + level) / 2, df=df_residual) * lm_se * np.sqrt(lr_factor)
        )

    intervals = np.column_stack([lm_coefs - q_stat, lm_coefs + q_stat])

    # Significance test p-value.
    p_value: float
    if distribution == "dnorm":
        actuals = model["actuals"]
        ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
        r2 = 1 - model["ssr"] / ss_tot
        f_value = r2 / (n_methods - 1) / ((1 - r2) / df_residual)
        p_value = float(stats.f.sf(f_value, n_methods - 1, df_residual))
    elif distribution == "tukey":
        assert friedman_p is not None
        p_value = friedman_p
    else:
        # Replicates R's exact (non-doubled) likelihood-ratio statistic.
        assert log_lik_full is not None and log_lik_null is not None
        stat = log_lik_full - log_lik_null
        p_value = float(stats.chi2.sf(stat, n_methods - 1))

    # Resolve the highlighted method on the *unsorted* coefficients.
    if select is None:
        select_name = names_methods[int(np.argmin(lm_coefs))]
    elif isinstance(select, str):
        select_name = select
    else:
        select_name = names_methods[int(select)]

    # Sort everything by ascending mean.
    order = np.argsort(lm_coefs, kind="stable")
    sorted_names = [names_methods[i] for i in order]
    lm_coefs = lm_coefs[order]
    intervals = intervals[order]
    # 1-based position of the selected method in the sorted order.
    select_pos = sorted_names.index(select_name) + 1

    if distribution == "dlnorm":
        intervals = np.exp(intervals)
        lm_coefs = np.exp(lm_coefs)

    low_label = f"{(1 - level) / 2 * 100:g}%"
    up_label = f"{(1 + level) / 2 * 100:g}%"

    mean_series = pd.Series(lm_coefs, index=sorted_names)
    interval_df = pd.DataFrame(
        intervals, index=sorted_names, columns=[low_label, up_label]
    )

    # Groups for the "lines" plot: contiguous runs of overlapping intervals.
    vlines_list = []
    for i in range(n_methods):
        intersecting = ~(
            (intervals[:, 1] < intervals[i, 0]) | (intervals[:, 0] > intervals[i, 1])
        )
        idx = np.where(intersecting)[0]
        vlines_list.append((int(idx.min()) + 1, int(idx.max()) + 1))

    # Deduplicate while preserving order.
    seen: set = set()
    vlines_unique = []
    for row in vlines_list:
        if row not in seen:
            seen.add(row)
            vlines_unique.append(row)
    vlines_arr = np.array(vlines_unique, dtype=int)
    n_groups = vlines_arr.shape[0]
    group_names = [f"Group{i + 1}" for i in range(n_groups)]
    vlines_df = pd.DataFrame(
        vlines_arr,
        index=group_names,
        columns=["Group starts", "Group ends"],
    )

    # Boolean membership table (methods x groups).
    groups_arr = np.zeros((n_methods, n_groups), dtype=bool)
    for i in range(n_groups):
        groups_arr[vlines_arr[i, 0] - 1 : vlines_arr[i, 1], i] = True
    groups_df = pd.DataFrame(groups_arr, index=sorted_names, columns=group_names)

    # Pairwise overlap table (methods x methods), matching R's logic.
    methods_arr = np.zeros((n_methods, n_methods), dtype=bool)
    for i in range(n_methods):
        for j in range(n_methods):
            methods_arr[i, j] = bool(
                (intervals[i, 1] >= intervals[j, 0])
                and (intervals[i, 0] <= intervals[j, 0])
                or (intervals[i, 1] >= intervals[j, 1])
                and (intervals[i, 0] <= intervals[j, 1])
                or (intervals[i, 1] <= intervals[j, 1])
                and (intervals[i, 0] >= intervals[j, 0])
            )
    methods_df = pd.DataFrame(methods_arr, index=sorted_names, columns=sorted_names)

    return RMCBResult(
        mean=mean_series,
        interval=interval_df,
        vlines=vlines_df,
        groups=groups_df,
        methods=methods_df,
        p_value=p_value,
        level=level,
        model=model,
        select=select_pos,
        distribution=distribution,
        correction=correction,
        factor=lr_factor,
    )


def _ols_log_lik(model: dict) -> float:
    """Gaussian log-likelihood of an OLS fit (matches R's logLik.lm)."""
    n = model["nobs"]
    ssr = model["ssr"]
    sigma2 = ssr / n
    return float(-0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1))
