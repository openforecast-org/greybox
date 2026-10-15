"""Plotting functions: spread(), graphmaker() and tableplot().

Python ports of R's :func:`greybox::spread`, :func:`greybox::graphmaker` and
:func:`greybox::tableplot`, and of R's ``plot.greybox()`` and
``plot.predict.greybox()`` for ALM and its predictions. Each function draws on
a new matplotlib figure and returns it, without calling ``plt.show()``. The
colours are those R uses with its default palette.
"""

import warnings

import numpy as np
import pandas as pd

from . import distributions as dist
from . import smoothers
from .diagnostics import outlier_dummy

_BLACK, _GREY41, _GREY, _LIGHTGREY = "#000000", "#696969", "#BEBEBE", "#D3D3D3"
_RED, _PURPLE, _BLUE, _DARKGREY = "#FF0000", "#A020F0", "#0000FF", "#A9A9A9"


def _pretty(lo, up, ndiv, min_n):
    """Breaks of R's ``pretty(c(lo, up), n=ndiv, min.n=min_n)``."""
    h, h5 = 1.5, 2.75
    if up - lo == 0 and up == 0:
        cell, i_small = 1.0, True
    else:
        cell = max(abs(lo), abs(up))
        bound = (1 + 1 / (1 + h)) * max(1, ndiv) * np.finfo(float).eps
        i_small = up - lo < cell * bound * 3
    if i_small:
        cell = (9 + cell / 10 if cell > 10 else cell) * 0.75
        cell = cell / min_n if min_n > 1 else cell
    else:
        cell = (up - lo) / ndiv if ndiv > 1 else up - lo

    base = 10.0 ** np.floor(np.log10(cell))
    unit = base
    if 2 * base - cell < h * (cell - unit):
        unit = 2 * base
        if 5 * base - cell < h5 * (cell - unit):
            unit = 5 * base
            if 10 * base - cell < h * (cell - unit):
                unit = 10 * base

    ns = np.floor(lo / unit + 1e-10)
    nu = np.ceil(up / unit - 1e-10)
    while ns * unit > lo + 1e-10 * unit:
        ns -= 1
    while nu * unit < up - 1e-10 * unit:
        nu += 1
    k = int(0.5 + nu - ns)
    if k < min_n:
        k = min_n - k
        if ns >= 0:
            nu, ns = nu + k // 2, ns - k // 2 - k % 2
        else:
            ns, nu = ns - k // 2, nu + k // 2 + k % 2
        ndiv = min_n
    else:
        ndiv = k

    lo, up = min(lo, ns * unit), max(up, nu * unit)
    breaks = np.linspace(lo, up, ndiv + 1)
    breaks[np.abs(breaks) < 1e-14 * (up - lo) / ndiv] = 0
    return breaks


def _hist(ax, values):
    """Draw R's hist() of a numeric variable or barplot(table()) of a factor.

    Returns the x positions (breaks or bar mid-points) and the heights.
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        codes = np.asarray(values.cat.codes)
        heights = np.bincount(codes[codes >= 0], minlength=len(values.cat.categories))
        positions = 0.7 + 1.2 * np.arange(heights.size)
        ax.bar(positions, heights, 1, color=_LIGHTGREY, edgecolor=_BLACK)
        ax.set_xticks(positions, [str(level) for level in values.cat.categories])
        x_range = np.array([0.2, 1.2 * heights.size])
    else:
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        breaks = _pretty(x.min(), x.max(), int(np.ceil(np.log2(x.size) + 1)), 1)
        widths = np.diff(breaks)
        if breaks.size > 5:
            diddle = np.median(widths)
        elif breaks.size <= 3:
            diddle = np.ptp(x)
        else:
            diddle = widths[widths > 0].min()
        fuzz = 1e-7 * diddle * np.r_[-1, np.ones(breaks.size - 1)]
        bins = np.searchsorted(breaks + fuzz, x, side="left") - 1
        heights = np.bincount(bins, minlength=breaks.size - 1)
        ax.bar(
            breaks[:-1],
            heights,
            widths,
            align="edge",
            color=_LIGHTGREY,
            edgecolor=_BLACK,
        )
        positions, x_range = breaks, breaks[[0, -1]]
    y_range = np.array([0, heights.max()])
    ax.set_xlim(*(x_range + np.array([-0.04, 0.04]) * np.ptp(x_range)))
    ax.set_ylim(*(y_range + np.array([-0.04, 0.04]) * np.ptp(y_range)))
    return positions, heights


def _tableplot(ax, x, y, labels, points, axes):
    """Draw the tableplot of the factors x and y on ax (see tableplot)."""
    x, y = pd.Categorical(x), pd.Categorical(y)
    valid = (x.codes >= 0) & (y.codes >= 0)
    counts = np.zeros((len(x.categories), len(y.categories)))
    np.add.at(counts, (x.codes[valid], y.codes[valid]), 1)
    shares = counts / counts.sum()

    # R: colorRampPalette(c("white", "black"))(1000)[findInterval(shares, ...)]
    grid = np.linspace(0, 1, 1000)
    greys = np.trunc(255 - 255 * grid[np.searchsorted(grid, shares, "right") - 1])
    for i, j in np.ndindex(shares.shape):
        colour = "#FFFFFF" if shares[i, j] > 0.5 else _BLACK
        ax.fill(
            [i + 0.5, i + 1.5, i + 1.5, i + 0.5],
            [j + 0.5, j + 0.5, j + 1.5, j + 1.5],
            facecolor=(greys[i, j] / 255,) * 3,
            edgecolor=_BLACK,
        )
        if labels:
            ax.text(
                i + 1,
                j + 1,
                f"{round(shares[i, j], 5):g}",
                ha="center",
                va="center",
                color=colour,
            )
        n = int(counts[i, j])
        if points and n > 0:
            ax.plot(
                np.full(n, i + 0.7),
                j + 1 + (np.arange(1, n + 1) - 0.5 - n / 2) / n * 0.75,
                "o",
                color=colour,
                mfc="none",
                ms=4,
            )

    for set_lim, k in ((ax.set_xlim, shares.shape[0]), (ax.set_ylim, shares.shape[1])):
        set_lim(0.5 - 0.04 * k, k + 0.5 + 0.04 * k)
    ax.set_xticks(np.arange(1, shares.shape[0] + 1), [str(v) for v in x.categories])
    ax.set_yticks(np.arange(1, shares.shape[1] + 1), [str(v) for v in y.categories])
    ax.tick_params(bottom=axes, labelbottom=axes, left=axes, labelleft=axes)


def _boxplot(ax, value, group, horizontal, lowess):
    """Draw R's boxplot(value ~ group) with the means as grey points (spread)."""
    values = value.to_numpy(dtype=float)
    codes = np.asarray(group.cat.codes)
    means = np.full(len(group.cat.categories), np.nan)
    for g in range(means.size):
        v = values[(codes == g) & ~np.isnan(values)]
        if v.size == 0:
            continue
        means[g] = v.mean()

        # R's boxplot.stats(): Tukey's hinges and 1.5 IQR whiskers
        s = np.sort(v)
        n4 = np.floor((s.size + 3) / 2) / 2
        d = np.array([1, n4, (s.size + 1) / 2, s.size + 1 - n4, s.size]) - 1
        stats = (s[np.floor(d).astype(int)] + s[np.ceil(d).astype(int)]) / 2
        iqr = stats[3] - stats[1]
        out = (v < stats[1] - 1.5 * iqr) | (v > stats[3] + 1.5 * iqr)
        if out.any():
            stats[[0, 4]] = v[~out].min(), v[~out].max()

        p = g + 1
        box = ([p - 0.4, p + 0.4, p + 0.4, p - 0.4], stats[[1, 1, 3, 3]])
        ax.fill(
            *(box[::-1] if horizontal else box), facecolor=_LIGHTGREY, edgecolor=_BLACK
        )
        for pos, val, style in (
            ([p - 0.4, p + 0.4], stats[[2, 2]], {"lw": 3}),
            ([p, p], stats[[0, 1]], {"ls": "--"}),
            ([p, p], stats[[3, 4]], {"ls": "--"}),
            ([p - 0.2, p + 0.2], stats[[0, 0]], {}),
            ([p - 0.2, p + 0.2], stats[[4, 4]], {}),
            (
                np.full(out.sum(), p),
                v[out],
                {"ls": "none", "marker": "o", "mfc": "none"},
            ),
        ):
            ax.plot(*((val, pos) if horizontal else (pos, val)), color=_BLACK, **style)

    mean_xy = (means, np.arange(1, means.size + 1))
    mean_xy = mean_xy if horizontal else mean_xy[::-1]
    if lowess:
        ax.plot(*mean_xy, color=_GREY, ls="--", lw=2)
    ax.plot(*mean_xy, "o", color=_GREY)


def spread(data, histograms=False, log=False, lowess=False, **kwargs):
    """Construct scatterplot / boxplots for the data.

    Python port of R's :func:`greybox::spread`. The plots depend on the
    types of the variables. If both variables are in metric scale, the
    classical scatterplot is constructed. If one of them is either integer
    (up to 10 values) or categorical, then boxplots (with grey dots
    corresponding to the mean values) are constructed. For two categorical
    variables the tableplot is drawn (see :func:`tableplot`). All of this is
    packed in a matrix of plots: the panel in row ``j`` and column ``i`` has
    the ``i``-th variable on the x axis and the ``j``-th one on the y axis.

    Parameters
    ----------
    data : pandas.DataFrame or array-like
        The data. Numeric columns with at most 10 distinct values and
        non-numeric columns are treated as categorical.
    histograms : bool, default=False
        If ``True``, the histograms and barplots are produced on the diagonal
        of the matrix. Otherwise the names of the variables are written there.
    log : bool, default=False
        If ``True``, the logarithms of all numerical variables are taken (of
        those with positive values only, with a warning otherwise).
    lowess : bool, default=False
        If ``True``, LOWESS lines are added to the scatterplots and the means
        are connected with lines on the boxplots, see
        :func:`greybox.smoothers.lowess`.
    **kwargs
        Only ``main`` (the title of the plot) is used.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the plots.

    Examples
    --------
    >>> from greybox import mtcars, spread
    >>> fig = spread(mtcars)
    >>> fig = spread(mtcars, histograms=True, log=True, lowess=True)
    """
    import matplotlib.pyplot as plt

    main = kwargs.get("main", "")
    if not isinstance(data, pd.DataFrame):
        values = np.asarray(data)
        values = values.reshape(len(values), -1)
        data = pd.DataFrame(
            values, columns=[f"V{i + 1}" for i in range(values.shape[1])]
        )

    n_variables = data.shape[1]
    if n_variables > 20:
        raise ValueError(
            "Too many variables! I can't work in such conditions! Too much "
            "pressure! Can you, please, reduce the number of variables at "
            "least to 20?"
        )

    names = [str(name) for name in data.columns]
    columns = [data.iloc[:, i] for i in range(n_variables)]
    numeric = np.zeros(n_variables, dtype=bool)
    for i, column in enumerate(columns):
        numeric[i] = pd.api.types.is_numeric_dtype(column) and not (
            pd.api.types.is_bool_dtype(column)
        )
        # Numeric variables with few values are treated as factors, as in R
        if numeric[i] and column.nunique(dropna=False) <= 10:
            numeric[i] = False
        if not numeric[i]:
            columns[i] = column.astype("category")

    if log:
        positive = numeric.copy()
        for i in np.flatnonzero(numeric):
            positive[i] = bool(np.all(columns[i].dropna() > 0))
        if np.any(positive != numeric):
            warnings.warn(
                "Some variables have non-positive data, so logarithms cannot "
                "be produced for them.",
                stacklevel=2,
            )
        for i in np.flatnonzero(positive):
            columns[i] = np.log(columns[i].astype(float))
            names[i] = f"log_{names[i]}"

    fig = plt.figure()
    if n_variables == 1:
        ax = fig.add_subplot()
        _hist(ax, columns[0])
        ax.set_title(main)
        return fig

    limits = []
    for i, column in enumerate(columns):
        if numeric[i]:
            limit = np.array([np.nanmin(column), np.nanmax(column)], dtype=float)
        else:
            limit = np.array([0.5, len(column.cat.categories) + 0.5])
        limits.append(limit + np.array([-0.04, 0.04]) * np.ptp(limit))

    axes = fig.subplots(
        n_variables, n_variables, squeeze=False, gridspec_kw={"wspace": 0, "hspace": 0}
    )
    for i in range(n_variables):
        for j in range(n_variables):
            ax = axes[j, i]
            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            if i == j and histograms:
                _hist(ax, columns[i])
            else:
                for k, set_ticks in ((i, ax.set_xticks), (j, ax.set_yticks)):
                    if not numeric[k]:
                        levels = columns[k].cat.categories
                        set_ticks(
                            np.arange(1, len(levels) + 1), [str(v) for v in levels]
                        )
                if i == j:
                    middle = np.mean(limits[i])
                    ax.text(
                        middle,
                        middle,
                        names[i],
                        fontsize="x-large",
                        ha="center",
                        va="center",
                    )
                elif numeric[i] and numeric[j]:
                    x = columns[i].to_numpy(dtype=float)
                    y = columns[j].to_numpy(dtype=float)
                    ax.plot(x, y, "o", color=_BLACK, mfc="none")
                    if lowess:
                        delta = 0.01 * (np.nanmax(x) - np.nanmin(x))
                        line = smoothers.lowess(x, y, delta=delta)
                        ax.plot(line["x"], line["y"], color=_GREY, ls="--", lw=2)
                elif numeric[i]:
                    _boxplot(ax, columns[i], columns[j], True, lowess)
                elif numeric[j]:
                    _boxplot(ax, columns[j], columns[i], False, lowess)
                else:
                    _tableplot(ax, columns[i], columns[j], False, True, False)
                ax.set_xlim(*limits[i])
                ax.set_ylim(*limits[j])

            if i == 0:
                if histograms:
                    ax.set_ylabel(names[j])
                else:
                    ax.tick_params(left=True, labelleft=True)
            if j == n_variables - 1:
                ax.tick_params(bottom=True, labelbottom=True)
            if i == n_variables - 1 and histograms:
                ax.tick_params(right=True, labelright=True)
            if j == 0 and histograms:
                ax.set_title(names[i])

    if main != "":
        fig.suptitle(main)
    return fig


def _time(x, default):
    """R's time(): the index of a pandas object, else the first values of default."""
    if isinstance(x, (pd.Series, pd.DataFrame)):
        index = x.index
        return index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
    return default[: len(x)]


def graphmaker(
    actuals,
    forecast,
    fitted=None,
    lower=None,
    upper=None,
    level=None,
    legend=True,
    vline=True,
    par_reset=True,
    **kwargs,
):
    """Linear graph construction function.

    Python port of R's :func:`greybox::graphmaker`: plots the actual values,
    the fitted values, the point forecasts and the prediction intervals,
    with a vertical line at the forecast origin and a legend underneath.

    The times of the values are the index of a pandas object (a
    ``PeriodIndex`` is converted to timestamps), the counterpart of R's
    ``ts`` objects. ``actuals`` without an index are at times ``1, ..., n``,
    and ``fitted`` without one starts with ``actuals``. A ``forecast``
    without an index, or one that starts at the same time as ``actuals``, is
    moved to the times following the last actual value (with the step of
    the index of ``actuals``, or its ``freq`` when it has one). ``lower`` and
    ``upper`` without an index (or when the forecast is moved) are at the
    times of the forecast, or of ``actuals`` when they have the length of
    ``fitted``.

    Parameters
    ----------
    actuals : array-like or pandas.Series
        The actual values.
    forecast : array-like or pandas.Series
        The point forecasts. Should start after the end of ``actuals``.
    fitted : array-like or pandas.Series, optional
        The fitted values.
    lower, upper : array-like, pandas.Series or pandas.DataFrame, optional
        The lower and upper bounds of the prediction intervals. Matrices
        (one column per level) are supported.
    level : float or sequence of float, optional
        The width of the prediction interval. Assumed 0.95 (with a warning
        if ``legend=True``) when the bounds are provided without it.
    legend : bool, default=True
        If ``True``, the legend is drawn under the plot.
    vline : bool, default=True
        Whether to draw the vertical line splitting the in-sample and the
        holdout sample.
    par_reset : bool, default=True
        R resets the graphical parameters after plotting when ``TRUE``; here
        it only sets the legend font to 0.75 of the default size, as R does.
        The returned figure can always be added to.
    **kwargs
        ``main``, ``xlab``, ``ylab``, ``xlim``, ``ylim`` and ``type`` (one
        of ``"l"``, ``"p"``, ``"b"``, ``"o"``, ``"n"`` as in R's ``plot()``)
        of the plot of the actual values.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the plot (and the legend axes, if drawn).

    Examples
    --------
    >>> import numpy as np
    >>> from greybox import graphmaker
    >>> fig = graphmaker(np.arange(1, 11), np.arange(11, 16))
    >>> fig = graphmaker(
    ...     np.arange(1, 11), np.arange(11, 16), lower=np.arange(9, 14),
    ...     upper=np.arange(13, 18), level=0.95,
    ... )
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    intervals = lower is not None or upper is not None
    if intervals and level is None:
        if legend:
            warnings.warn(
                "The width of prediction intervals is not provided to the "
                "function! Assuming 95%.",
                stacklevel=2,
            )
        level = 0.95

    actuals_v = np.asarray(actuals, dtype=float).ravel()
    actuals_t = _time(actuals, pd.Index(np.arange(1.0, actuals_v.size + 1)))
    forecast_v = np.asarray(forecast, dtype=float).ravel()
    fitted_v = (
        np.full(1, np.nan)
        if fitted is None
        else np.asarray(fitted, dtype=float).ravel()
    )
    fitted_t = _time(fitted, actuals_t) if fitted is not None else actuals_t[:1]
    h = forecast_v.size
    if np.all(np.isnan(forecast_v)):
        h = 0
        vline = False

    freq = getattr(actuals_t, "freq", None)
    if freq is None:
        freq = (actuals_t[-1] - actuals_t[0]) / max(actuals_t.size - 1, 1)
    forecast_t = _time(forecast, pd.Index([]))
    shifted = not isinstance(forecast, (pd.Series, pd.DataFrame)) or (
        forecast_t[0] == actuals_t[0]
    )
    if shifted:
        forecast_t = pd.Index(
            [actuals_t[-1] + freq * k for k in range(1, forecast_v.size + 1)]
        )

    if intervals:
        bounds = []
        for bound, other in ((lower, upper), (upper, lower)):
            values = np.asarray(bound if bound is not None else other, dtype=float)
            values = values if bound is not None else np.full(values.shape, np.nan)
            base = (
                actuals_t
                if fitted is not None and len(values) == fitted_v.size
                else forecast_t
            )
            times = _time(values if shifted or bound is None else bound, base)
            bounds.append((values, times))
        (lower_v, lower_t), (upper_v, upper_t) = bounds
    else:
        lower_v = upper_v = np.full(1, np.nan)

    fig = plt.figure()
    if legend:
        grid = fig.add_gridspec(2, 1, height_ratios=[0.86, 0.14])
        ax = fig.add_subplot(grid[0])
        ax_legend = fig.add_subplot(grid[1])
    else:
        ax = fig.add_subplot()

    finite = np.concatenate(
        [
            v[np.isfinite(v)]
            for v in (actuals_v, fitted_v, forecast_v, lower_v.ravel(), upper_v.ravel())
        ]
    )
    ylim = kwargs.get("ylim", (finite.min(), finite.max()))
    xlim = kwargs.get("xlim", (actuals_t[0], forecast_t[max(h, 1) - 1]))
    ax.set_ylim(
        ylim[0] - (ylim[1] - ylim[0]) * 0.04, ylim[1] + (ylim[1] - ylim[0]) * 0.04
    )
    ax.set_xlim(
        xlim[0] - (xlim[1] - xlim[0]) * 0.04, xlim[1] + (xlim[1] - xlim[0]) * 0.04
    )
    ax.set_title(kwargs.get("main", ""))
    ax.set_xlabel(kwargs.get("xlab", ""))
    ax.set_ylabel(kwargs.get("ylab", ""))
    plot_type = kwargs.get("type", "l")
    if plot_type not in ("l", "p", "b", "o", "n"):
        raise ValueError(
            f"type must be one of 'l', 'p', 'b', 'o', 'n', got {plot_type!r}."
        )
    actuals_style = {
        "color": _BLACK,
        "lw": 1,
        "ls": "-" if plot_type in "lbo" else "none",
        "marker": "o" if plot_type in "pbo" else None,
        "mfc": "none",
        "visible": plot_type != "n",
    }
    ax.plot(actuals_t, actuals_v, **actuals_style)

    if vline:
        ax.axvline(forecast_t[0] - freq, color=_RED, lw=2)

    if intervals and h != 1:
        lower_m, upper_m = (
            lower_v.reshape(len(lower_v), -1),
            upper_v.reshape(len(upper_v), -1),
        )
        if lower_m.shape[1] == upper_m.shape[1]:
            pairs = [(c, c, lower_m.shape[1] - c) for c in range(lower_m.shape[1])]
        else:
            pairs = [(-1, -1, 1)]
        for lc, uc, density in pairs:
            if np.all(np.isfinite(upper_m[:, uc])) and np.all(
                np.isfinite(lower_m[:, lc])
            ):
                ax.fill(
                    np.concatenate([np.asarray(upper_t), np.asarray(lower_t)[::-1]]),
                    np.concatenate([upper_m[:, uc], lower_m[::-1, lc]]),
                    facecolor="none",
                    edgecolor=_DARKGREY,
                    hatch="/" * density,
                    lw=0,
                )
        n_levels = max(lower_m.shape[1], upper_m.shape[1])
        ramp = np.trunc(169 + 42 * np.linspace(0, 1, n_levels))
        for values, times in ((lower_v, lower_t), (upper_v, upper_t)):
            if values.ndim == 2:
                for c in range(values.shape[1]):
                    colour = (ramp[c] / 255,) * 3
                    ax.plot(times, values[:, c], color=colour, lw=2, ls="--")
            else:
                ax.plot(times, values, color=_DARKGREY, lw=3, ls="--")
        ax.plot(forecast_t, forecast_v, color=_BLUE, lw=2)
        ax.plot(actuals_t, actuals_v, **actuals_style)
    elif intervals:
        for values, times in ((lower_v, lower_t), (upper_v, upper_t)):
            if values.size > 1:
                for c, value in enumerate(values.ravel()):
                    if np.isfinite(value):
                        grey = int(255 * (1 - (c + 1) / (values.size + 1)) + 0.5)
                        colour = (grey / 255,) * 3
                        ax.plot(times[:1], [value], "x", color=colour, mew=c + 2)
            else:
                ax.plot(times, values.ravel(), "x", color=_DARKGREY, mew=3)
        ax.plot(forecast_t, forecast_v, "x", color=_BLUE, mew=2)
    elif h != 1:
        ax.plot(forecast_t, forecast_v, color=_BLUE, lw=2)
    else:
        ax.plot(forecast_t, forecast_v, "x", color=_BLUE, mew=2)

    has_fitted = bool(np.any(~np.isnan(fitted_v)))
    if has_fitted:
        ax.plot(fitted_t, fitted_v, color=_PURPLE, lw=2, ls="--")

    if legend:
        levels = np.atleast_1d(level if intervals else [])
        level_label = ", ".join(f"{v * 100:g}" for v in levels)
        labels = [
            "Series",
            "Fitted values",
            "Point forecast",
            f"{level_label}% prediction interval{'s' if levels.size > 1 else ''}",
            "Forecast origin",
        ]
        colours = [_BLACK, _PURPLE, _BLUE, _DARKGREY, _RED]
        widths, styles = [1, 2, 2, 3, 2], ["-", "--", "-", "--", "-"]
        if intervals and h != 1 and lower_v.ndim == 2 and upper_v.ndim == 2:
            widths[3] = 2
        markers = [None, None, "x", "x", None] if h == 1 else [None] * 5
        if h == 1:
            styles[2] = styles[3] = "none"
        elements = [True, has_fitted, h != 0, intervals, h != 0]
        handles = [
            Line2D(
                [], [], color=colours[k], lw=widths[k], ls=styles[k], marker=markers[k]
            )
            for k in range(5)
            if elements[k]
        ]
        ax_legend.axis("off")
        ax_legend.legend(
            handles,
            [label for label, element in zip(labels, elements) if element],
            loc="center",
            ncol=3 if intervals else 2,
            fontsize=0.75 * plt.rcParams["font.size"] if par_reset else None,
            edgecolor=_BLACK,
            fancybox=False,
        )
    return fig


def tableplot(x, y=None, labels=True, legend=False, points=True, **kwargs):
    """Construct a plot for categorical variables.

    Python port of R's :func:`greybox::tableplot`. The plot is a grid of the
    cells of the contingency table of ``x`` and ``y``, shaded from white
    (no observations) to black (all of them) according to the share of the
    observations in each cell. The shares can be printed in the cells and
    the observations shown as points.

    Parameters
    ----------
    x : array-like, pandas.Series or pandas.DataFrame
        First categorical variable. If ``y`` is ``None`` and ``x`` is a
        matrix or a data frame with several columns, then the first two
        variables of the data are plotted against each other.
    y : array-like, optional
        Second categorical variable. If not provided, then only ``x`` is
        plotted.
    labels : bool, default=True
        Whether to print the shares inside the cells.
    legend : bool, default=False
        If ``True``, the legend of the shading is drawn next to the plot.
    points : bool, default=True
        Whether to plot the points in the cells. They help in understanding
        how many values lie in specific categories.
    **kwargs
        ``main``, ``xlab``, ``ylab`` and ``axes`` (whether to draw the axes)
        of the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the plot (and the legend axes, if drawn).

    Examples
    --------
    >>> from greybox import mtcars, tableplot
    >>> fig = tableplot(mtcars["am"], mtcars["gear"])
    >>> fig = tableplot(mtcars[["am", "gear"]], legend=True)
    """
    import matplotlib.pyplot as plt

    xlab, ylab = kwargs.get("xlab"), kwargs.get("ylab")
    y_provided = y is not None
    if y is None:
        if np.ndim(x) == 2 and np.shape(x)[1] > 1:
            if isinstance(x, pd.DataFrame):
                xlab, ylab = str(x.columns[0]), str(x.columns[1])
                x, y = x.iloc[:, 0], x.iloc[:, 1]
            else:
                x, y = np.asarray(x)[:, 0], np.asarray(x)[:, 1]
            y_provided = True
        else:
            x = x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x
            x = x if isinstance(x, pd.Series) else np.asarray(x).ravel()
            y = np.zeros(len(x))
    if xlab is None:
        xlab = "" if getattr(x, "name", None) is None else str(x.name)
    if ylab is None:
        ylab = "" if getattr(y, "name", None) is None else str(y.name)

    fig = plt.figure()
    if legend:
        grid = fig.add_gridspec(1, 2, width_ratios=[0.9, 0.1])
        ax = fig.add_subplot(grid[0])
    else:
        ax = fig.add_subplot()
    axes = kwargs.get("axes", True)
    _tableplot(ax, x, y, labels, points, axes)
    ax.tick_params(left=axes and y_provided, labelleft=axes and y_provided)
    ax.set_title(kwargs.get("main", ""))
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab if y_provided else "")

    if legend:
        ax_legend = fig.add_subplot(grid[1])
        ax_legend.set_xlim(1 - 0.02, 1.5 + 0.02)
        ax_legend.set_ylim(1 - 0.04, 2 + 0.04)
        ax_legend.axis("off")
        edges = np.linspace(1.25, 1.75, 6)
        for k, grey in enumerate(np.trunc(255 - 255 * np.linspace(0, 1, 5))):
            ax_legend.fill(
                [1, 1.25, 1.25, 1],
                [edges[k], edges[k], edges[k + 1], edges[k + 1]],
                facecolor=(grey / 255,) * 3,
                edgecolor=_BLACK,
            )
            ax_legend.text(1.3, edges[k + 1] - 0.05, f"{k / 4:g}", va="center")
    return fig


_COUNT = ("dpois", "dnbinom", "dbinom", "dgeom")
_LOG_RESIDUALS = ("dinvgauss", "dgamma", "dexp")
# R's text(pos=): 1 below, 2 to the left, 3 above the point
_TEXT_POSITIONS = {
    1: {"xytext": (0, -4), "ha": "center", "va": "top"},
    2: {"xytext": (-4, 0), "ha": "right", "va": "center"},
    3: {"xytext": (0, 4), "ha": "center", "va": "bottom"},
}


def _ppoints(n):
    """R's ppoints(n)."""
    a = 3 / 8 if n <= 10 else 1 / 2
    return (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)


def _finite_range(*values):
    """R's range() of the finite values."""
    finite = np.concatenate([np.ravel(np.asarray(v, dtype=float)) for v in values])
    finite = finite[np.isfinite(finite)]
    return np.array([finite.min(), finite.max()])


def _r_plot(ax, x, y, plot_type, labels, kwargs, xlim=None, ylim=None):
    """R's plot(x, y, type=): the limits (the finite ranges by default) widened
    by 4%, the titles (main, xlab, ylab) and the points ("p"), lines ("l"),
    both ("b", "o"), vertical bars ("h") or nothing ("n")."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    plot_type = kwargs.get("type", plot_type)
    if plot_type not in ("l", "p", "b", "o", "h", "n"):
        raise ValueError(
            f"type must be one of 'l', 'p', 'b', 'o', 'h', 'n', got {plot_type!r}."
        )
    for name, limit, values, set_limit in (
        ("xlim", xlim, x, ax.set_xlim),
        ("ylim", ylim, y, ax.set_ylim),
    ):
        limit = kwargs.get(name) if limit is None else limit
        lo, hi = _finite_range(values) if limit is None else limit
        set_limit(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
    for key, label, set_label in zip(
        ("main", "xlab", "ylab"), labels, (ax.set_title, ax.set_xlabel, ax.set_ylabel)
    ):
        set_label(kwargs.get(key, label))
    if plot_type == "h":
        ax.vlines(x, 0, y, color=_BLACK, gid="plot")
    elif plot_type != "n":
        ax.plot(
            x,
            y,
            color=_BLACK,
            lw=1,
            ls="-" if plot_type in "lbo" else "none",
            marker="o" if plot_type in "pbo" else None,
            mfc="none",
            gid="plot",
        )


def _r_points(ax, x, y, filled=False):
    """R's points(x, y), with pch=16 if filled."""
    ax.plot(x, y, "o", color=_BLACK, mfc=_BLACK if filled else "none", gid="points")


def _r_text(ax, x, y, labels, pos):
    """R's text(x, y, labels, pos)."""
    for xi, yi, label, p in zip(x, y, labels, np.broadcast_to(pos, len(labels))):
        ax.annotate(
            str(label),
            (xi, yi),
            textcoords="offset points",
            gid="text",
            **_TEXT_POSITIONS[int(p)],
        )


def _r_lowess(ax, x, y):
    """R's lines(lowess(x, y), col=2)."""
    line = smoothers.lowess(x, y)
    ax.plot(line["x"], line["y"], color=_RED, gid="lines")


def _r_bounds(ax, x, statistic):
    """R's hatched polygon between the bounds over x and the dashed red lines."""
    ax.fill(
        np.r_[x, x[::-1]],
        np.r_[np.full(len(x), statistic[0]), np.full(len(x), statistic[1])],
        facecolor="none",
        edgecolor=_LIGHTGREY,
        hatch="///",
        lw=0,
        gid="polygon",
    )


def _r_legend(ax, position, labels, colours, styles, markers, widths=None):
    """R's legend() inside the plot."""
    from matplotlib.lines import Line2D

    widths = [1] * len(labels) if widths is None else widths
    handles = [
        Line2D([], [], color=colour, ls=style, marker=marker, mfc=colour, lw=width)
        for colour, style, marker, width in zip(colours, styles, markers, widths)
    ]
    ax.legend(handles, labels, loc=position, edgecolor=_BLACK, fancybox=False)


def _legend_ylim(ylim, position):
    """R's room for the legend: ylim widened by 20% at the side of the legend."""
    widening = 0.2 * (ylim[1] - ylim[0])
    if position == "upper right":
        return np.array([ylim[0], ylim[1] + widening])
    return np.array([ylim[0] - widening, ylim[1]])


def _infinite_markers(ax, x, values, ylim):
    """Red triangles with the ids of the infinite values at the edges of ylim."""
    ids = np.flatnonzero(np.isinf(values))
    if ids.size == 0:
        return
    heights = np.asarray(ylim, dtype=float)[(values[ids] > 0).astype(int)] + 0.1
    ax.plot(x[ids], heights, "^", color=_BLACK, mfc=_RED, gid="points")
    _r_text(ax, x[ids], heights, ids + 1, 1)


def _alm_actuals_fitted(ax, model, lowess, kwargs):
    """Plot 1: actuals (likelihoods for count distributions) vs fitted."""
    response = model._response_name or "y"
    if model.distribution in _COUNT:
        from .pointlik import point_lik

        y = np.asarray(point_lik(model, log=False), dtype=float)
        x = 1 / np.asarray(model.fitted, dtype=float)
        labels = (
            "Probability vs Likelihood",
            "Probability",
            f"Likelihood of {response}",
        )
    else:
        y = np.asarray(model.actuals, dtype=float)
        if model.distribution in ("plogis", "pnorm"):
            y = (y != 0) * 1.0
        x = np.asarray(model.fitted, dtype=float)
        labels = ("Actuals vs Fitted", "Fitted", f"Actual {response}")

    # Without the zeros of an occurrence model and the NAs
    keep = ~np.isnan(x) & ~np.isnan(y)
    if model.occurrence_ is not None:
        keep &= y != 0
    x, y = x[keep], y[keep]
    _r_plot(ax, x, y, "p", labels, kwargs)
    ax.axline((0, 0), slope=1, color=_GREY, lw=2, ls="--", gid="abline")
    if lowess:
        _r_lowess(ax, x, y)


def _alm_residuals_fitted(ax, model, residuals, level, legend, lowess, kwargs):
    """Plots 2 and 3: standardised / studentised residuals vs fitted."""
    x = np.array(model.fitted, dtype=float)
    y = np.asarray(getattr(model, residuals)(), dtype=float)
    name = "Standardised" if residuals == "rstandard" else "Studentised"
    position = "lower right" if x[-1] > np.mean(x) else "upper right"

    outliers = outlier_dummy(model, level=level, type=residuals)
    ids, statistic = outliers.id, np.asarray(outliers.statistic, dtype=float)
    if model.distribution in _LOG_RESIDUALS:
        y, statistic = np.log(y), np.log(statistic)
    if model.occurrence_ is not None:
        x[np.asarray(model.occurrence_.actuals) == 0] = np.nan

    ylim = kwargs.get("ylim")
    if ylim is None:
        ylim = _finite_range(y, statistic) * 1.2
        if legend:
            ylim = _legend_ylim(ylim, position)
    x_range = np.array([np.nanmin(x), np.nanmax(x)])
    x_range += np.array([-1, 1]) * np.nanstd(x, ddof=1)

    labels = (f"{name} Residuals vs Fitted", "Fitted", f"{name} Residuals")
    _r_plot(ax, x, y, "p", labels, kwargs, ylim=ylim)
    ax.axhline(0, color=_GREY, ls="--", gid="abline")
    _r_bounds(ax, x_range, statistic)
    for value in statistic:
        ax.axhline(value, color=_RED, ls="--", gid="abline")
    if ids.size > 0:
        _r_points(ax, x[ids], y[ids], True)
        _r_text(ax, x[ids], y[ids], ids + 1, (y[ids] > 0) * 2 + 1)
    if lowess:
        _r_lowess(ax, x[~np.isnan(x)], y[~np.isnan(x)])
    _infinite_markers(ax, x, y, ylim)
    if legend:
        k = 3 if lowess else 2
        _r_legend(
            ax,
            position,
            [f"{round(level, 3) * 100:g}% bounds", "outside the bounds", "LOWESS line"][
                :k
            ],
            [_RED, _BLACK, _RED][:k],
            ["--", "none", "-"][:k],
            [None, "o", None][:k],
        )


def _alm_residuals_power(ax, model, standardised, absolute, lowess, kwargs):
    """Plots 4, 5, 13 and 14: |residuals| or residuals^2 vs fitted, of the
    residuals or of the standardised ones."""
    x = np.asarray(model.fitted, dtype=float)
    y = np.asarray(model.rstandard() if standardised else model.residuals, float)
    if model.distribution in _LOG_RESIDUALS:
        y = np.log(y)
    y = np.abs(y) if absolute else y**2
    if model.occurrence_ is not None:
        keep = np.asarray(model.occurrence_.actuals) != 0
        x, y = x[keep], y[keep]

    name = "Standardised Residuals" if standardised else "Residuals"
    ylab = f"|{name}|" if absolute else f"{name}^2"
    ylim = kwargs.get("ylim")
    if ylim is None and standardised:
        ylim = _finite_range(y)
    _r_plot(ax, x, y, "p", (f"{ylab} vs Fitted", "Fitted", ylab), kwargs, ylim=ylim)
    ax.axhline(0, color=_GREY, ls="--", gid="abline")
    if lowess:
        _r_lowess(ax, x[~np.isnan(y)], y[~np.isnan(y)])
    if standardised:
        _infinite_markers(ax, x, y, ylim)


def _alm_qq(ax, model, kwargs):
    """Plot 6: Q-Q plot of the residuals against the distribution of the model."""
    y = np.asarray(model.residuals, dtype=float)
    if model.occurrence_ is not None:
        y = y[np.asarray(model.occurrence_.actuals) != 0]
    labels = ("Theoretical Quantile", "Actual Quantile")
    distribution, scale, other = model.distribution, model.extract_scale(), model.other_
    p = _ppoints(200)

    # The empirical probabilities of the actuals below the quantiles of each
    # observation, without those of the zeros
    if distribution in _COUNT:
        actuals = np.asarray(model.actuals, dtype=float)
        mu, probabilities = np.asarray(model.mu_, dtype=float), p[:, None]
        if distribution == "dpois":
            name, quantiles = "Poisson", dist.qpois(probabilities, mu)
        elif distribution == "dnbinom":
            name = "Negative Binomial"
            quantiles = dist.qnbinom(probabilities, mu=mu, size=scale)
        elif distribution == "dbinom":
            name = "Binomial"
            quantiles = dist.qbinom(probabilities, model.size_, 1 / (1 + mu))
        else:
            name, quantiles = "Geometric", dist.qgeom(probabilities, 1 / (1 + mu))
        y = np.mean(actuals <= np.asarray(quantiles), axis=1)
        keep = y > np.mean(actuals == 0)
        _r_plot(
            ax,
            p[keep],
            y[keep],
            "p",
            (f"QQ-plot of {name} distribution", *labels),
            kwargs,
            xlim=kwargs.get("xlim", (0, 1)),
            ylim=kwargs.get("ylim", (0, 1)),
        )
        ax.axline((0, 0), slope=1, color=_BLACK, gid="abline")
        return

    if distribution in ("dnorm", "dlnorm", "dbcnorm", "dlogitnorm", "plogis", "pnorm"):
        main, quantile = "QQ plot of normal distribution", dist.qnorm
    elif distribution in ("dfnorm", "drectnorm"):
        main = (
            "QQ-plot of Folded Normal distribution"
            if distribution == "dfnorm"
            else "QQ-plot of Rectified Normal distribution"
        )
        q = dist.qfnorm if distribution == "dfnorm" else dist.qrectnorm

        def quantile(probability):
            return q(probability, 0, np.sqrt(scale))

    elif distribution in ("dgnorm", "dlgnorm"):
        main = "QQ-plot of Generalised Normal distribution"

        def quantile(probability):
            return dist.qgnorm(probability, 0, scale, other)

    elif distribution in ("dlaplace", "dllaplace"):
        main = "QQ-plot of Laplace distribution"

        def quantile(probability):
            return dist.qlaplace(probability, 0, scale)

    elif distribution == "dalaplace":
        main = (
            f"QQ-plot of Asymmetric Laplace distribution with alpha={round(other, 3):g}"
        )

        def quantile(probability):
            return dist.qalaplace(probability, 0, scale, other)

    elif distribution == "dlogis":
        main = "QQ-plot of Logistic distribution"

        def quantile(probability):
            return dist.qlogis(probability, 0, scale)

    elif distribution in ("ds", "dls"):
        main = "QQ-plot of S distribution"

        def quantile(probability):
            return dist.qs(probability, 0, scale)

    elif distribution == "dt":
        main = "QQ-plot of Student's distribution"

        def quantile(probability):
            return dist.qt(probability, scale)

    elif distribution == "dinvgauss":
        main = "QQ-plot of Inverse Gaussian distribution"

        def quantile(probability):
            return dist.qinvgauss(probability, mean=1, dispersion=scale)

    elif distribution == "dgamma":
        main = "QQ-plot of Gamma distribution"

        def quantile(probability):
            return dist.qgamma(probability, shape=1 / scale, scale=scale)

    elif distribution == "dexp":
        main = "QQ-plot of Exponential distribution"

        def quantile(probability):
            return dist.qexp(probability, rate=model.scale)

    else:
        name = "Chi-Squared" if distribution == "dchisq" else "Beta"
        warnings.warn(
            f"Sorry, but we don't produce QQ plots for the {name} distribution",
            stacklevel=4,
        )
        ax.axis("off")
        return

    if quantile is dist.qnorm:
        # R's qqnorm()
        x = np.asarray(dist.qnorm(_ppoints(y.size)))[np.argsort(np.argsort(y))]
    else:
        # R's qqplot() of the standardised residuals for some distributions,
        # interpolating the longer of the sorted samples to the shorter one
        if distribution in ("dfnorm", "drectnorm", "dgnorm", "dlgnorm", "dt"):
            y = y / np.std(y, ddof=1)
        x, y = np.sort(np.asarray(quantile(p), dtype=float)), np.sort(y[~np.isnan(y)])
        if y.size < x.size:
            x = np.interp(np.linspace(0, x.size - 1, y.size), np.arange(x.size), x)
        elif y.size > x.size:
            y = np.interp(np.linspace(0, y.size - 1, x.size), np.arange(y.size), y)
    _r_plot(ax, x, y, "p", (main, *labels), kwargs)

    # R's qqline(): through the quartiles
    quartiles = np.quantile(y[~np.isnan(y)], [0.25, 0.75])
    theoretical = np.asarray(quantile(np.array([0.25, 0.75])), dtype=float)
    slope = np.diff(quartiles)[0] / np.diff(theoretical)[0]
    ax.axline(
        (0, quartiles[0] - slope * theoretical[0]),
        slope=slope,
        color=_BLACK,
        gid="abline",
    )


def _alm_fit_time(ax, model, legend, kwargs):
    """Plot 7: actuals and fitted values over time."""
    values = np.asarray(model.actuals, dtype=float)
    if model.distribution in ("plogis", "pnorm"):
        values = (values != 0) * 1.0
    fitted = np.asarray(model.fitted, dtype=float)
    times = np.arange(1, values.size + 1)

    ylim = kwargs.get("ylim")
    position = "lower right" if fitted[-1] > np.mean(fitted) else "upper right"
    if legend and ylim is None:
        ylim = _finite_range(values, fitted)
        ylim = _legend_ylim(ylim, position)
    labels = ("Fit over time", "Time", model._response_name or "y")
    _r_plot(ax, times, values, "l", labels, kwargs, ylim=ylim)
    ax.plot(times, fitted, color=_RED, gid="lines")
    if legend:
        _r_legend(
            ax, position, ["Actuals", "Fitted"], [_BLACK, _RED], ["-", "-"], [None] * 2
        )


def _alm_residuals_time(ax, model, residuals, level, legend, lowess, kwargs):
    """Plots 8 and 9: standardised / studentised residuals over observations."""
    values = np.array(getattr(model, residuals)(), dtype=float)
    occurrence = model.occurrence_ is not None
    if occurrence:
        values[np.asarray(model.occurrence_.actuals) == 0] = np.nan
    name = "Standardised" if residuals == "rstandard" else "Studentised"

    outliers = outlier_dummy(model, level=level, type=residuals)
    ids, statistic = outliers.id, np.asarray(outliers.statistic, dtype=float)
    if model.distribution in _LOG_RESIDUALS:
        values, statistic = np.log(values), np.log(statistic)

    ylim = kwargs.get("ylim")
    ylim = _finite_range(values, statistic) * 1.2 if ylim is None else ylim
    if legend:
        ylim = np.array(ylim, dtype=float)
        ylim[1] += 0.2 * (ylim[1] - ylim[0])
        ylim[0] -= 0.2 * (ylim[1] - ylim[0])

    times = np.arange(1, values.size + 1)
    labels = (
        f"{name} Residuals over observations",
        "Observations",
        f"{name} Residuals",
    )
    _r_plot(ax, times, values, "l", labels, kwargs, ylim=ylim)
    # The points fill in the breaks of the line at the zeros of the occurrence
    if occurrence:
        _r_points(ax, times, values)
    if ids.size > 0:
        _r_points(ax, ids + 1, values[ids], True)
        _r_text(ax, ids + 1, values[ids], ids + 1, (values[ids] > 0) * 2 + 1)
    if lowess:
        filled = np.where(np.isnan(values), np.nanmean(values), values)
        _r_lowess(ax, times, filled)
    ax.axhline(0, color=_GREY, ls="--", gid="abline")
    for value in statistic:
        ax.axhline(value, color=_RED, ls="--", gid="abline")
    _r_bounds(ax, times, statistic)
    _infinite_markers(ax, times, values, ylim)
    if legend:
        _r_legend(
            ax,
            "upper right",
            ["Residuals", f"{level * 100:g}% prediction interval"],
            [_BLACK, _RED],
            ["-", "--"],
            [None] * 2,
        )


def _alm_acf(ax, model, partial, squared, level, kwargs):
    """Plots 10, 11, 15 and 16: ACF / PACF of the residuals or their squares."""
    values = np.asarray(model.residuals, dtype=float)
    values = values**2 if squared else values
    n = values.size
    lag_max = int(min(np.floor(10 * np.log10(n)), n - 1))

    # R's acf(na.action=na.pass): the sums of the products of the available
    # pairs over their number plus the lag, as correlations within [-1, 1]
    centred = values - np.nanmean(values)
    acf = np.empty(lag_max + 1)
    for lag in range(lag_max + 1):
        products = centred[lag:] * centred[: n - lag]
        available = ~np.isnan(products)
        acf[lag] = np.sum(products[available]) / (available.sum() + lag)
    acf = np.clip(acf / acf[0], -1, 1)
    if partial:
        # R's pacf(): the Durbin-Levinson recursion
        values, weights = np.empty(lag_max), np.empty(lag_max)
        weights[0] = values[0] = acf[1]
        for lag in range(1, lag_max):
            numerator = acf[lag + 1] - weights[:lag] @ acf[lag:0:-1]
            values[lag] = numerator / (1 - weights[:lag] @ acf[1 : lag + 1])
            weights[:lag] = weights[:lag] - values[lag] * weights[:lag][::-1]
            weights[lag] = values[lag]
    else:
        values = acf[1:]
    statistic = np.asarray(
        dist.qnorm([(1 - level) / 2, (1 + level) / 2], 0, np.sqrt(1 / model.nobs))
    )

    function = "Partial Autocorrelation" if partial else "Autocorrelation"
    of = "Squared Residuals" if squared else "Residuals"
    labels = (f"{function} Function of {of}", "Lags", "PACF" if partial else "ACF")
    lags = np.arange(1, values.size + 1)
    ylim = kwargs.get("ylim", (-1, 1))
    _r_plot(ax, lags, values, "h", labels, {**kwargs, "type": "h"}, ylim=ylim)
    ax.axhline(0, color=_BLACK, gid="abline")
    for value in statistic:
        ax.axhline(value, color=_RED, ls="--", gid="abline")
    outside = (values > statistic[1]) | (values < statistic[0])
    if np.any(outside):
        _r_points(ax, lags[outside], values[outside], True)
        _r_text(
            ax,
            lags[outside],
            values[outside],
            lags[outside],
            (values[outside] > 0) * 2 + 1,
        )


def _alm_cook(ax, model, legend, kwargs):
    """Plot 12: Cook's distance over observations, with the lines of 4/n,
    4/(n-k) and 1."""
    hat_values, nobs, nparam = model.hatvalues(), model.nobs, model.nparam
    values = np.abs(model.rstandard() ** 2 / nparam * hat_values / (1 - hat_values))
    thresholds = np.array([4 / nobs, 4 / (nobs - nparam), 1])
    styles, widths = [":", "--", (0, (8, 4))], [1, 1, 2]
    ids = np.flatnonzero(values >= thresholds[1])

    ylim = kwargs.get("ylim")
    ylim = _finite_range(values, thresholds) if ylim is None else ylim
    times = np.arange(1, values.size + 1)
    labels = ("Cook's distance over observations", "Observation", "Cook's distance")
    _r_plot(ax, times, values, "h", labels, kwargs, ylim=ylim)
    for threshold, style, width in zip(thresholds, styles, widths):
        ax.axhline(threshold, color=_RED, ls=style, lw=width, gid="abline")
    if ids.size > 0:
        _r_text(ax, ids + 1, values[ids], ids + 1, 2)
    _infinite_markers(ax, times, values, ylim)
    if legend:
        labels = [f"4/{nobs}", f"4/({nobs}-{nparam})", "1"]
        _r_legend(ax, "upper right", labels, [_RED] * 3, styles, [None] * 3, widths)


def _plot_alm(
    model,
    which=(1, 2, 4, 6),
    level=0.95,
    legend=False,
    ask=False,
    lowess=True,
    **kwargs,
):
    """Plots of the fit and residuals of an ALM.

    Python port of R's ``plot.greybox()`` for ``alm``. The panels are drawn on
    one figure, in the order of ``which``:

    1. Actuals vs Fitted values, with the grey line of the perfect fit (for
       the count distributions, the likelihoods of the actuals vs the inverse
       of the fitted values);
    2. Standardised residuals vs Fitted, with the bounds of ``level`` (red
       lines) and the ids of the observations outside of them;
    3. Studentised residuals vs Fitted, as (2) with the residuals divided by
       the scales without each observation;
    4. Absolute residuals vs Fitted, for the analysis of heteroscedasticity;
    5. Squared residuals vs Fitted;
    6. Q-Q plot of the residuals against the distribution of the model;
    7. Actuals (black line) and fitted values (red line) over time;
    8. Standardised residuals over observations, with the bounds of ``level``;
    9. Studentised residuals over observations;
    10. ACF of the residuals, with the bounds ``qnorm((1 +- level)/2, 0,
        1/sqrt(n))``;
    11. PACF of the residuals;
    12. Cook's distance over observations, with the lines of 4/n, 4/(n-k)
        and 1;
    13. Absolute standardised residuals vs Fitted;
    14. Squared standardised residuals vs Fitted;
    15. ACF of the squared residuals;
    16. PACF of the squared residuals.

    The plots 2-6, 8, 9, 13 and 14 of a model with a scale model
    (``scale_model``) are of the latter, as in R. The residuals of the
    distributions dinvgauss, dgamma and dexp are analysed in logarithms, the
    zeros of an occurrence model are excluded and the studentised residuals of
    the count distributions are replaced by the standardised ones (with a
    warning). The observations are numbered from 1, as in R.

    Parameters
    ----------
    model : ALM
        The fitted model.
    which : int or sequence of int, default=(1, 2, 4, 6)
        Which of the plots to produce.
    level : float, default=0.95
        The confidence level of the bounds of the plots 2, 3, 8, 9, 10, 11, 15
        and 16.
    legend : bool, default=False
        If ``True``, the legend is drawn on the plots 2, 3, 7, 8, 9 and 12.
    ask : bool, default=False
        If ``True``, the figure is shown and the user is asked to press Enter
        before each plot after the first one. R asks by default when the plots
        do not fit in the layout of the device; here they are all in one
        figure.
    lowess : bool, default=True
        If ``True``, the LOWESS lines (:func:`greybox.smoothers.lowess`) are
        drawn on the scatterplots.
    **kwargs
        ``main``, ``xlab``, ``ylab``, ``xlim``, ``ylim`` and ``type`` (as in
        R's ``plot()``) of every plot. Recommended with separate plots.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with one axes per plot.
    """
    import matplotlib.pyplot as plt

    which = [int(i) for i in np.atleast_1d(which)]
    if model.distribution in _COUNT and any(i in (3, 9) for i in which):
        warnings.warn(
            "Studentised residuals are not supported for count distributions. "
            "Switching to standardised.",
            stacklevel=3,
        )
        which = [{3: 2, 9: 8}.get(i, i) for i in which]
    which = [i for i in which if 1 <= i <= 16]

    # The diagnostics of the residuals of a model with a scale model are of the
    # latter
    residual_model = model
    if getattr(model, "scale_model", None) is not None:
        residual_model = model.scale_model
        if any(i in (2, 3, 4, 5, 6, 8, 9, 13, 14) for i in which):
            warnings.warn(
                "Note that residuals diagnostics plots are produced for scale model",
                stacklevel=3,
            )

    panels = {
        1: lambda ax: _alm_actuals_fitted(ax, model, lowess, kwargs),
        7: lambda ax: _alm_fit_time(ax, model, legend, kwargs),
        6: lambda ax: _alm_qq(ax, residual_model, kwargs),
        12: lambda ax: _alm_cook(ax, model, legend, kwargs),
    }
    for i, residuals in ((2, "rstandard"), (3, "rstudent")):
        panels[i] = lambda ax, r=residuals: _alm_residuals_fitted(
            ax, residual_model, r, level, legend, lowess, kwargs
        )
        panels[i + 6] = lambda ax, r=residuals: _alm_residuals_time(
            ax, residual_model, r, level, legend, lowess, kwargs
        )
    for i, standardised, absolute in (
        (4, False, True),
        (5, False, False),
        (13, True, True),
        (14, True, False),
    ):
        panels[i] = lambda ax, s=standardised, a=absolute: _alm_residuals_power(
            ax, residual_model, s, a, lowess, kwargs
        )
    for i, partial, squared in (
        (10, False, False),
        (11, True, False),
        (15, False, True),
        (16, True, True),
    ):
        panels[i] = lambda ax, p=partial, s=squared: _alm_acf(
            ax, model, p, s, level, kwargs
        )

    ncols = int(np.ceil(np.sqrt(max(len(which), 1))))
    nrows = int(np.ceil(max(len(which), 1) / ncols))
    fig = plt.figure(
        figsize=(max(6.4, 4.8 * ncols), max(4.8, 3.6 * nrows)), layout="constrained"
    )
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
    for ax in axes[len(which) :]:
        ax.remove()
    for k, (i, ax) in enumerate(zip(which, axes)):
        if ask and k > 0:
            plt.pause(0.001)
            input("Hit <Return> to see next plot: ")
        panels[i](ax)
    return fig


def _plot_prediction(prediction, model, y=None, newdata_provided=True, **kwargs):
    """Plot of the predictions of an ALM.

    Python port of R's ``plot.predict.greybox()``: :func:`graphmaker` of the
    actuals of the model (followed by ``y``, the actuals of the holdout, when
    provided without NaNs), the fitted values and the point forecasts with
    the bounds of the interval. The in-sample predictions
    (``newdata_provided=False``) are drawn as the fitted values with their
    bounds, without the forecast origin. The observations are at times 1, 2,
    ..., as in R.

    Parameters
    ----------
    prediction : PredictionResult
        The predictions of ``model``.
    model : ALM
        The fitted model.
    y : array-like, optional
        The actual values of the predicted observations.
    newdata_provided : bool, default=True
        Whether the predictions were produced for new data (``X``).
    **kwargs
        Passed to :func:`graphmaker`; ``legend=False`` (and ``par_reset=False``)
        unless provided, and the title names the response variable.

    Returns
    -------
    matplotlib.figure.Figure
        The figure of :func:`graphmaker`.
    """
    actuals = np.asarray(model.actuals, dtype=float)
    n = actuals.size
    if newdata_provided and y is not None:
        holdout = np.ravel(np.asarray(y, dtype=float))
        if not np.any(np.isnan(holdout)):
            actuals = np.concatenate([actuals, holdout])
    # The actuals of an occurrence model are its occurrence
    if prediction.distribution in ("plogis", "pnorm"):
        actuals = (actuals != 0) * 1.0

    mean = np.ravel(np.asarray(prediction.mean, dtype=float))
    if newdata_provided:
        fitted = pd.Series(np.asarray(model.fitted, dtype=float), np.arange(1.0, n + 1))
        times = np.arange(n + 1.0, n + mean.size + 1)
        forecast = pd.Series(mean, times)
    else:
        fitted = pd.Series(mean, np.arange(1.0, mean.size + 1))
        times = fitted.index.to_numpy()
        forecast = pd.Series([np.nan], [n + 1.0])

    # The infinite bounds are not drawn
    bounds = []
    for bound in (prediction.lower, prediction.upper):
        if bound is not None:
            bound = np.asarray(bound, dtype=float)
            bound = pd.DataFrame(np.where(np.isfinite(bound), bound, np.nan), times)
            bound = bound.iloc[:, 0] if np.ndim(prediction.lower) < 2 else bound
        bounds.append(bound)
    intervals = bounds[0] is not None or bounds[1] is not None

    options = {"legend": False, "par_reset": False} if "legend" not in kwargs else {}
    response = model._response_name or "y"
    kind = "Forecast" if newdata_provided else "Fitted values"
    options["main"] = f"{kind} for the variable {response}"
    return graphmaker(
        pd.Series(actuals, np.arange(1.0, actuals.size + 1)),
        forecast,
        fitted,
        *bounds,
        level=prediction.level if intervals else None,
        **{**options, **kwargs, "vline": newdata_provided},
    )
