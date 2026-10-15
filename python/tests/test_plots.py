"""Tests of spread(), graphmaker(), tableplot() and the plots of ALM.

The smoke tests draw every combination of the arguments on the Agg backend.
The comparison tests check the numbers that drive the plots (histogram breaks
and counts, LOWESS lines, boxplot statistics, table shares and colours, the
positions of the lines, polygons and the legend) against R. The R functions
are sourced from the R/ folder of the repository into an environment where
the graphics functions record their arguments.
"""

import itertools
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from numpy.testing import assert_allclose

from greybox import ALM, graphmaker, mtcars, spread, tableplot
from greybox.plots import _plot_alm, _plot_prediction, _pretty

from .conftest import has_rpy2

R_DIR = Path(__file__).resolve().parents[2] / "R"

requires_r = pytest.mark.skipif(not has_rpy2(), reason="Requires rpy2 and R")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def mixed_data():
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame(
        {
            "x": rng.normal(10, 2, n),
            "y": rng.normal(-1, 3, n),
            "level": rng.integers(1, 4, n),
            "colour": rng.choice(["red", "green", "blue"], n),
            "flag": rng.random(n) > 0.5,
        }
    )


# ---------------------------------------------------------------------------
# Smoke tests: every argument combination draws without error
# ---------------------------------------------------------------------------


class TestSpreadSmoke:
    @pytest.mark.parametrize(
        "histograms, log, lowess, main",
        list(
            itertools.product([False, True], [False, True], [False, True], [None, "T"])
        ),
    )
    def test_combinations(self, mixed_data, histograms, log, lowess, main):
        kwargs = {} if main is None else {"main": main}
        if log:
            # "y" has non-positive values, so R warns and skips it
            with pytest.warns(UserWarning, match="non-positive"):
                fig = spread(mixed_data, histograms, log, lowess, **kwargs)
        else:
            fig = spread(mixed_data, histograms, log, lowess, **kwargs)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == mixed_data.shape[1] ** 2
        fig.canvas.draw()

    def test_log_names(self, mixed_data):
        with pytest.warns(UserWarning):
            fig = spread(mixed_data, histograms=True, log=True)
        assert fig.axes[0].get_title() == "log_x"
        assert fig.axes[1].get_title() == "y"

    @pytest.mark.parametrize("column", ["x", "level", "colour"])
    def test_single_variable(self, mixed_data, column):
        fig = spread(mixed_data[[column]], main="One")
        assert len(fig.axes) == 1
        assert fig.axes[0].get_title() == "One"
        assert len(fig.axes[0].patches) > 0

    def test_array_input(self):
        rng = np.random.default_rng(1)
        fig = spread(rng.normal(size=(30, 3)), histograms=True, lowess=True)
        assert [ax.get_title() for ax in fig.axes[:3]] == ["V1", "V2", "V3"]

    def test_categorical_histogram_axis(self, mixed_data):
        # The barplot in the last diagonal panel keeps its frequencies axis
        fig = spread(mixed_data[["x", "colour"]], histograms=True)
        ax = np.array(fig.axes).reshape(2, 2)[1, 1]
        levels = ["blue", "green", "red"]
        assert [t.get_text() for t in ax.get_xticklabels()] == levels
        assert not {t.get_text() for t in ax.get_yticklabels()} & set(levels)

    def test_too_many_variables(self):
        with pytest.raises(ValueError, match="Too many variables"):
            spread(np.zeros((5, 21)))


class TestTableplotSmoke:
    @pytest.mark.parametrize(
        "labels, legend, points, axes",
        list(itertools.product([False, True], repeat=4)),
    )
    def test_combinations(self, labels, legend, points, axes):
        fig = tableplot(
            mtcars["am"],
            mtcars["gear"],
            labels=labels,
            legend=legend,
            points=points,
            axes=axes,
            main="Table",
        )
        assert len(fig.axes) == (2 if legend else 1)
        assert len(fig.axes[0].patches) == 6
        assert len(fig.axes[0].texts) == (6 if labels else 0)
        assert fig.axes[0].get_title() == "Table"
        fig.canvas.draw()

    def test_data_frame(self):
        fig = tableplot(mtcars[["am", "gear"]])
        ax = fig.axes[0]
        assert ax.get_xlabel() == "am" and ax.get_ylabel() == "gear"

    def test_matrix(self):
        fig = tableplot(mtcars[["am", "gear"]].to_numpy(), xlab="a", ylab="g")
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["3", "4", "5"]

    def test_single_variable(self):
        fig = tableplot(np.array(["a", "b", "b", "c"]))
        ax = fig.axes[0]
        assert len(ax.patches) == 3
        assert ax.get_ylabel() == ""
        assert [t.get_text() for t in ax.texts] == ["0.25", "0.5", "0.25"]

    def test_single_column_data_frame(self):
        fig = tableplot(mtcars[["gear"]])
        ax = fig.axes[0]
        assert ax.get_xlabel() == "gear" and len(ax.patches) == 3

    def test_dark_cell_colours(self):
        ax = tableplot(np.array([1, 1, 1, 2])).axes[0]
        assert [to_hex(t.get_color()) for t in ax.texts] == ["#ffffff", "#000000"]
        assert [to_hex(line.get_color()) for line in ax.lines] == [
            "#ffffff",
            "#000000",
        ]


class TestGraphmakerSmoke:
    @pytest.fixture
    def series(self):
        rng = np.random.default_rng(3)
        actuals = 100 + rng.normal(size=40).cumsum()
        forecast = np.full(8, actuals[-1])
        return actuals, forecast, actuals + rng.normal(size=40)

    @pytest.mark.parametrize(
        "legend, vline, par_reset, with_fitted, bounds, h",
        list(
            itertools.product(
                [False, True],
                [False, True],
                [False, True],
                [False, True],
                ["none", "vector", "matrix", "uneven", "mixed", "lower"],
                [1, 8],
            )
        ),
    )
    def test_combinations(
        self, series, legend, vline, par_reset, with_fitted, bounds, h
    ):
        actuals, forecast, fitted = series
        forecast = forecast[:h]
        width = np.c_[np.full(h, 5.0), np.full(h, 2.0)]
        lower = {
            "none": None,
            "vector": forecast - 5,
            "matrix": forecast[:, None] - width,
            "uneven": forecast[:, None] - width,
            "mixed": forecast[:, None] - width,
            "lower": forecast - 5,
        }[bounds]
        upper = {
            "none": None,
            "vector": forecast + 5,
            "matrix": forecast[:, None] + width,
            "uneven": forecast[:, None] + width[:, :1],
            "mixed": forecast + 5,
            "lower": None,
        }[bounds]
        fig = graphmaker(
            actuals,
            forecast,
            fitted=fitted if with_fitted else None,
            lower=lower,
            upper=upper,
            level=None if lower is None else [0.95, 0.8][: np.ndim(lower)],
            legend=legend,
            vline=vline,
            par_reset=par_reset,
            main="Graph",
            xlab="Time",
            ylab="Value",
        )
        assert len(fig.axes) == (2 if legend else 1)
        assert fig.axes[0].get_title() == "Graph"
        fig.canvas.draw()

    @pytest.mark.parametrize("plot_type", ["l", "p", "b", "o", "n"])
    def test_types(self, series, plot_type):
        actuals, forecast, _ = series
        graphmaker(actuals, forecast, type=plot_type).canvas.draw()

    def test_invalid_type(self, series):
        with pytest.raises(ValueError, match="type"):
            graphmaker(*series[:2], type="z")

    def test_level_warning(self, series):
        actuals, forecast, _ = series
        with pytest.warns(UserWarning, match="Assuming 95%"):
            fig = graphmaker(actuals, forecast, lower=forecast - 1, upper=forecast + 1)
        labels = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
        assert "95% prediction interval" in labels

    def test_all_nan_forecast(self, series):
        actuals, _, fitted = series
        fig = graphmaker(actuals, np.full(5, np.nan), fitted=fitted)
        labels = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
        assert labels == ["Series", "Fitted values"]
        assert all(line.get_color() != "#FF0000" for line in fig.axes[0].lines)

    def test_legend_without_fitted(self, series):
        actuals, forecast, _ = series
        fig = graphmaker(
            actuals, forecast, lower=forecast - 1, upper=forecast + 1, level=[0.95, 0.8]
        )
        legend = fig.axes[1].get_legend()
        assert [t.get_text() for t in legend.get_texts()] == [
            "Series",
            "Point forecast",
            "95, 80% prediction intervals",
            "Forecast origin",
        ]
        assert [to_hex(h.get_color()) for h in legend.legend_handles] == [
            "#000000",
            "#0000ff",
            "#a9a9a9",
            "#ff0000",
        ]

    def test_dates(self):
        index = pd.date_range("2020-01-01", periods=36, freq="MS")
        actuals = pd.Series(np.arange(36.0), index=index)
        fig = graphmaker(
            actuals,
            np.arange(36.0, 48),
            fitted=actuals.to_numpy() + 1,
            lower=np.arange(30.0, 42),
            upper=np.arange(42.0, 54),
            level=0.9,
        )
        ax = fig.axes[0]
        forecast_line = next(line for line in ax.lines if line.get_color() == "#0000FF")
        assert pd.Timestamp(forecast_line.get_xdata()[0]) == pd.Timestamp("2023-01-01")
        vline = next(line for line in ax.lines if line.get_color() == "#FF0000")
        assert pd.Timestamp(vline.get_xdata()[0]) == pd.Timestamp("2022-12-01")

    def test_single_forecast_origin(self):
        fig = graphmaker(np.arange(1.0, 11), np.array([11.0]), legend=False)
        vline = [line for line in fig.axes[0].lines if line.get_color() == "#FF0000"]
        assert_allclose(vline[0].get_xdata(), [10, 10])

    def test_forecast_with_index(self):
        actuals = pd.Series(np.arange(10.0))
        forecast = pd.Series(np.arange(10.0, 15), index=np.arange(10, 15))
        fig = graphmaker(actuals, forecast, legend=False)
        forecast_line = [
            line for line in fig.axes[0].lines if line.get_color() == "#0000FF"
        ]
        assert_allclose(forecast_line[0].get_xdata(), np.arange(10, 15))


# ---------------------------------------------------------------------------
# Comparison with R
# ---------------------------------------------------------------------------

R_HARNESS = """
library(greybox)
plotCalls <- new.env()
plotEnv <- new.env(parent=asNamespace("greybox"))
for (f in c("lines", "points", "polygon", "text", "abline", "legend", "hist",
            "boxplot", "plot", "rect", "axis", "title")) {
    local({
        name <- f
        fun <- get(f)
        assign(name, function(...) {
            out <- fun(...)
            plotCalls$calls[[length(plotCalls$calls) + 1]] <- list(
                name=name, args=list(...), out=out, usr=par("usr"))
            invisible(out)
        }, envir=plotEnv)
    })
}
for (file in c("tableplot.R", "spread.R", "graphmaker.R")) {
    sys.source(file.path("%s", file), envir=plotEnv)
}
plotRun <- function(expr) {
    plotCalls$calls <- list()
    pdf(NULL)
    eval(parse(text=expr), envir=plotEnv)
    dev.off()
    invisible(NULL)
}
plotOf <- function(name) {
    Filter(function(call) call$name == name, plotCalls$calls)
}
plotCoords <- function(name) {
    lapply(plotOf(name), function(call) {
        a <- call$args
        if ("x" %%in%% names(a)) {
            a <- list(a$x)
        } else {
            a <- a[if (is.null(names(a))) seq_along(a) else names(a) == ""]
        }
        if (inherits(a[[1]], "zoo") && length(a) == 1) {
            return(cbind(as.numeric(time(a[[1]])), as.numeric(a[[1]])))
        }
        xy <- xy.coords(a[[1]], if (length(a) > 1) a[[2]] else NULL)
        cbind(as.numeric(xy$x), as.numeric(xy$y))
    })
}
plotArg <- function(name, arg) {
    lapply(plotOf(name), function(call) call$args[[arg]])
}
plotHex <- function(col) {
    rgb(t(col2rgb(col)), maxColorValue=255)
}
"""


@pytest.fixture(scope="module")
def r():
    import rpy2.robjects as ro

    ro.r(R_HARNESS % R_DIR)
    return ro


def r_list(r, expr):
    """Evaluate an R expression returning a list of numeric matrices."""
    return [np.array(item) for item in r.r(expr)]


def r_mtcars(r, columns):
    return pd.DataFrame({c: np.array(r.r(f"mtcars${c}")) for c in columns})


@requires_r
class TestPrettyCompare:
    @pytest.mark.parametrize(
        "lo, up, n",
        [
            (0, 10, 5),
            (-3.3, 7.7, 8),
            (0.00123, 0.0458, 7),
            (1e5, 1e5, 4),
            (0, 0, 3),
            (-1e-8, 2.5e-7, 11),
            (10.4, 33.9, 6),
        ],
    )
    def test_pretty(self, r, lo, up, n):
        expected = np.array(r.r(f"pretty(c({lo!r}, {up!r}), n={n}, min.n=1)"))
        assert np.array_equal(_pretty(lo, up, n, 1), expected)


@requires_r
class TestSpreadCompare:
    @pytest.mark.parametrize(
        "r_data",
        [
            "mtcars$mpg",
            "mtcars$disp",
            "mtcars$qsec",
            "c(1:50)",
            "{set.seed(41); rnorm(100)}",
            "{set.seed(42); rexp(37) * 1000}",
            "{set.seed(43); round(rnorm(200, 0, 3), 1)}",
        ],
    )
    def test_histogram(self, r, r_data):
        values = np.array(r.r(r_data), dtype=float)
        r.r(f"plotHist <- hist({r_data}, plot=FALSE)")
        breaks = np.array(r.r("plotHist$breaks"))
        counts = np.array(r.r("plotHist$counts"))
        patches = spread(pd.DataFrame({"v": values})).axes[0].patches
        assert_allclose([p.get_x() for p in patches], breaks[:-1], rtol=1e-14)
        assert_allclose(
            [p.get_x() + p.get_width() for p in patches], breaks[1:], rtol=1e-14
        )
        assert [p.get_height() for p in patches] == list(counts)

    def test_panels(self, r):
        columns = ["mpg", "cyl", "disp", "hp", "am", "gear"]
        data = r_mtcars(r, columns)
        r.r(
            'plotRun(\'spread(mtcars[,c("mpg","cyl","disp","hp","am",'
            '"gear")], histograms=TRUE, lowess=TRUE)\')'
        )
        fig = spread(data, histograms=True, lowess=True)
        n = len(columns)
        numeric = [True, False, True, True, False, False]
        axes = np.array(fig.axes).reshape(n, n)

        r_hists = r.r("lapply(plotOf('hist'), function(call) call$out$counts)")
        r_lines = r_list(r, "plotCoords('lines')")
        r_boxes = r_list(r, "lapply(plotOf('boxplot'), function(call) call$out$stats)")
        r_colours = list(r.r("unlist(lapply(plotArg('polygon', 'col'), plotHex))"))
        r_means = [m for m in r_list(r, "plotCoords('points')") if m.shape[0] > 0]
        hists, lines, boxes, colours, means = [], [], [], [], []
        for i in range(n):
            for j in range(n):
                ax = axes[j, i]
                if i == j:
                    if numeric[i]:
                        hists.append([p.get_height() for p in ax.patches])
                elif numeric[i] and numeric[j]:
                    line = ax.lines[1]
                    lines.append(np.c_[line.get_xdata(), line.get_ydata()])
                elif numeric[i] or numeric[j]:
                    k = (len(ax.lines) - 2) // 6
                    value = 0 if numeric[i] else 1
                    stats = np.empty((5, k))
                    for g in range(k):
                        low, high = ax.lines[6 * g + 1], ax.lines[6 * g + 2]
                        median = ax.lines[6 * g].get_xydata()[0, value]
                        stats[:, g] = [
                            *low.get_xydata()[:, value],
                            median,
                            *high.get_xydata()[:, value],
                        ]
                    boxes.append(stats)
                    lines.append(ax.lines[6 * k].get_xydata())
                    means.append(ax.lines[6 * k + 1].get_xydata())
                else:
                    colours += [to_hex(p.get_facecolor()).upper() for p in ax.patches]
        assert len(hists) == len(r_hists)
        for python, expected in zip(hists, r_hists):
            assert python == list(expected)
        assert len(lines) == len(r_lines)
        for python, expected in zip(lines, r_lines):
            assert_allclose(python, expected, rtol=1e-12)
        assert len(boxes) == len(r_boxes)
        for python, expected in zip(boxes, r_boxes):
            assert_allclose(python, expected, rtol=1e-14)
        assert colours == r_colours
        # The last axis on the right is the frequencies axis of the barplot of "gear"
        r.r("plotAxis <- Filter(function(call) call$args[[1]] == 4, plotOf('axis'))")
        assert r.r("length(plotAxis)")[0] == n
        assert r.r(f"is.null(plotAxis[[{n}]]$args$labels)")[0]
        levels = {t.get_text() for t in axes[n - 1, n - 1].get_xticklabels()}
        assert not {t.get_text() for t in axes[n - 1, n - 1].get_yticklabels()} & levels
        # R's points() also draws the observations in the tableplots
        r_means = [m for m in r_means if m.shape[0] in [m.shape[0] for m in means]]
        for python in means:
            assert any(
                m.shape == python.shape and np.allclose(m, python, rtol=1e-12)
                for m in r_means
            )

    def test_log(self, r):
        data = r_mtcars(r, ["mpg", "disp", "wt"])
        data["wt"] = data["wt"] - 3
        r.r(
            'plotRun(\'{x <- mtcars[,c("mpg","disp","wt")]; x$wt <- x$wt - 3; '
            "suppressWarnings(spread(x, lowess=TRUE, log=TRUE))}')"
        )
        r_names = [str(v) for v in r.r("unlist(plotArg('text', 3))")]
        with pytest.warns(UserWarning, match="non-positive"):
            fig = spread(data, lowess=True, log=True)
        axes = np.array(fig.axes).reshape(3, 3)
        assert [axes[k, k].texts[0].get_text() for k in range(3)] == r_names
        r_lines = r_list(r, "plotCoords('lines')")
        lines = [
            axes[j, i].lines[1].get_xydata()
            for i in range(3)
            for j in range(3)
            if i != j
        ]
        for python, expected in zip(lines, r_lines, strict=True):
            assert_allclose(python, expected, rtol=1e-12)


@requires_r
class TestTableplotCompare:
    @pytest.mark.parametrize(
        "r_x, r_y",
        [
            ("mtcars$am", "mtcars$gear"),
            ("mtcars$cyl", "mtcars$carb"),
            (
                (
                    "{set.seed(5); factor(sample(c('a','b','c'), 50, TRUE),"
                    " levels=c('a','b','c','d'))}"
                ),
                "{set.seed(6); sample(1:3, 50, TRUE)}",
            ),
            ("mtcars$gear", "NULL"),
            ("c(1, 1, 1, 2, 1, 1, 3)", "NULL"),
            ("c(1, 1, 1, 2, 1, 1, 3)", "c(0, 0, 0, 0, 0, 0, 1)"),
        ],
    )
    def test_cells(self, r, r_x, r_y):
        r.r(f"plotX <- {r_x}; plotY <- {r_y}")
        r.r("plotRun('tableplot(plotX, plotY, legend=TRUE)')")
        x = r.r("plotX")
        if r.r("is.factor(plotX)")[0]:
            levels = [str(v) for v in r.r("levels(plotX)")]
            x = pd.Categorical.from_codes(np.array(x) - 1, levels)
        else:
            x = np.array(x)
        y = None if r_y == "NULL" else np.array(r.r("plotY"))
        fig = tableplot(x, y, legend=True)
        ax, ax_legend = fig.axes

        r_colours = list(r.r("unlist(lapply(plotArg('polygon', 'col'), plotHex))"))
        assert [to_hex(p.get_facecolor()).upper() for p in ax.patches] == r_colours
        r_labels = [f"{v:g}" for v in r.r("unlist(plotArg('text', 'labels'))")]
        assert [t.get_text() for t in ax.texts] == r_labels
        # R's coordinates of the cells start at 0.5, as in Python
        r_polygons = r_list(r, "plotCoords('polygon')")
        for patch, expected in zip(ax.patches, r_polygons, strict=True):
            assert_allclose(np.sort(patch.get_xy()[:-1], 0), np.sort(expected, 0))
        r_points = r_list(r, "plotCoords('points')")
        for line, expected in zip(ax.lines, r_points, strict=True):
            assert_allclose(line.get_xydata(), expected, rtol=1e-14)
        for python, name in ((ax.texts, "text"), (ax.lines, "points")):
            assert [to_hex(v.get_color()).upper() for v in python] == list(
                r.r(f"unlist(lapply(plotArg('{name}', 'col'), plotHex))")
            )
        r_legend = list(r.r("plotHex(plotOf('rect')[[1]]$args$col)"))
        assert [
            to_hex(p.get_facecolor()).upper() for p in ax_legend.patches
        ] == r_legend

    def test_single_column_data_frame(self, r):
        r.r("plotRun('tableplot(data.frame(gear=mtcars$gear))')")
        ax = tableplot(r_mtcars(r, ["gear"])).axes[0]
        r_labels = [f"{v:g}" for v in r.r("unlist(plotArg('text', 'labels'))")]
        assert [t.get_text() for t in ax.texts] == r_labels
        assert ax.get_xlabel() == r.r("plotOf('title')[[1]]$args$xlab")[0]


@requires_r
class TestGraphmakerCompare:
    def _lines(self, ax):
        """Lines in the order R draws them: without the first actuals and vline."""
        return [line for line in ax.lines[1:] if line.get_color() != "#FF0000"]

    @pytest.mark.parametrize(
        "r_args, python_args",
        [
            ("c(1:10), c(11:15)", lambda: (np.arange(1.0, 11), np.arange(11.0, 16))),
            (
                (
                    "c(1:10), c(11:15), fitted=c(1:10)+0.5, lower=c(9:13),"
                    " upper=c(13:17), level=0.9"
                ),
                lambda: (
                    np.arange(1.0, 11),
                    np.arange(11.0, 16),
                    np.arange(1.0, 11) + 0.5,
                    np.arange(9.0, 14),
                    np.arange(13.0, 18),
                    0.9,
                ),
            ),
            (
                (
                    "c(1:10), c(11:15), lower=cbind(c(9:13),c(10:14)),"
                    " upper=cbind(c(13:17),c(12:16)), level=c(0.95,0.8)"
                ),
                lambda: (
                    np.arange(1.0, 11),
                    np.arange(11.0, 16),
                    None,
                    np.c_[np.arange(9.0, 14), np.arange(10.0, 15)],
                    np.c_[np.arange(13.0, 18), np.arange(12.0, 17)],
                    [0.95, 0.8],
                ),
            ),
            (
                (
                    "c(1:10), c(11:15), lower=cbind(c(9:13),c(10:14)), upper=c(13:17),"
                    " level=c(0.95,0.8)"
                ),
                lambda: (
                    np.arange(1.0, 11),
                    np.arange(11.0, 16),
                    None,
                    np.c_[np.arange(9.0, 14), np.arange(10.0, 15)],
                    np.arange(13.0, 18),
                    [0.95, 0.8],
                ),
            ),
            (
                (
                    "ts(sin(1:20), start=c(2000,1), frequency=4),"
                    " ts(rep(0.5,6), start=c(2005,1), frequency=4),"
                    " fitted=ts(cos(1:20), start=c(2000,1), frequency=4),"
                    " lower=ts(rep(-1,6), start=c(2005,1), frequency=4),"
                    " upper=ts(rep(2,6), start=c(2005,1), frequency=4), level=0.95"
                ),
                lambda: (
                    pd.Series(np.sin(np.arange(1, 21)), index=2000 + np.arange(20) / 4),
                    pd.Series(np.full(6, 0.5), index=2005 + np.arange(6) / 4),
                    pd.Series(np.cos(np.arange(1, 21)), index=2000 + np.arange(20) / 4),
                    pd.Series(np.full(6, -1.0), index=2005 + np.arange(6) / 4),
                    pd.Series(np.full(6, 2.0), index=2005 + np.arange(6) / 4),
                    0.95,
                ),
            ),
            (
                (
                    "ts(1:10, start=c(2000,1), frequency=4),"
                    " ts(11, start=c(2002,3), frequency=4),"
                    " lower=ts(9, start=c(2002,3), frequency=4),"
                    " upper=ts(13, start=c(2002,3), frequency=4), level=0.95"
                ),
                lambda: (
                    pd.Series(np.arange(1.0, 11), index=2000 + np.arange(10) / 4),
                    pd.Series([11.0], index=[2002.5]),
                    None,
                    pd.Series([9.0], index=[2002.5]),
                    pd.Series([13.0], index=[2002.5]),
                    0.95,
                ),
            ),
            (
                "c(1:10), 11, lower=9, upper=13, level=0.9",
                lambda: (
                    np.arange(1.0, 11),
                    np.array([11.0]),
                    None,
                    np.array([9.0]),
                    np.array([13.0]),
                    0.9,
                ),
            ),
            (
                (
                    "ts(1:10, start=c(2000,1), frequency=4),"
                    " ts(11, start=c(2002,3), frequency=4)"
                ),
                lambda: (
                    pd.Series(np.arange(1.0, 11), index=2000 + np.arange(10) / 4),
                    pd.Series([11.0], index=[2002.5]),
                ),
            ),
            (
                (
                    "zoo::zoo(c(1:10), order.by=seq(1, 5.5, 0.5)), c(11:15),"
                    " fitted=zoo::zoo(c(1:10)+0.5, order.by=seq(1, 5.5, 0.5))"
                ),
                lambda: (
                    pd.Series(np.arange(1.0, 11), index=np.arange(1, 5.6, 0.5)),
                    np.arange(11.0, 16),
                    pd.Series(np.arange(1.0, 11) + 0.5, index=np.arange(1, 5.6, 0.5)),
                ),
            ),
            (
                "c(1:10), c(11:15), legend=FALSE, vline=FALSE",
                lambda: (
                    np.arange(1.0, 11),
                    np.arange(11.0, 16),
                    None,
                    None,
                    None,
                    None,
                    False,
                    False,
                ),
            ),
        ],
    )
    def test_graph(self, r, r_args, python_args):
        r.r(f"plotRun('graphmaker({r_args})')")
        fig = graphmaker(*python_args())
        ax = fig.axes[0]

        r_plot = r.r("plotOf('plot')")
        r_main = r_plot[len(r_plot) - 1].rx2("args")
        for limit, getter in (("xlim", ax.get_xlim), ("ylim", ax.get_ylim)):
            lo, hi = np.array(r_main.rx2(limit))
            assert_allclose(getter(), [lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)])

        r_vline = list(r.r("as.numeric(unlist(plotArg('abline', 'v')))"))
        vlines = [
            line.get_xdata()[0] for line in ax.lines if line.get_color() == "#FF0000"
        ]
        assert_allclose(vlines, r_vline, rtol=1e-12)

        lines = self._lines(ax)
        r_lines = r_list(r, "plotCoords('lines')")
        r_points = r_list(r, "plotCoords('points')")
        assert len(lines) == len(r_lines) + len(r_points)
        for python, expected in zip(lines, r_lines + r_points):
            assert_allclose(python.get_xydata(), expected, rtol=1e-12)
        r_line_colours = r.r(
            "unlist(lapply(c(plotArg('lines', 'col'), plotArg('points', 'col')),"
            " plotHex))"
        )
        assert [to_hex(line.get_color()).upper() for line in lines] == list(
            r_line_colours
        )
        r_polygons = r_list(r, "plotCoords('polygon')")
        for patch, expected in zip(ax.patches, r_polygons, strict=True):
            assert_allclose(patch.get_xy()[:-1], expected, rtol=1e-12)

        if len(fig.axes) == 1:
            assert r.r("length(plotOf('legend'))")[0] == 0
            return
        legend = fig.axes[1].get_legend()
        r.r("plotLegend <- plotOf('legend')[[1]]$args")
        assert [t.get_text() for t in legend.get_texts()] == list(
            r.r("plotLegend$legend")
        )
        handles = legend.legend_handles
        r_colours = list(r.r("plotHex(plotLegend$col)"))
        assert [to_hex(h.get_color()).upper() for h in handles] == r_colours
        assert [h.get_linewidth() for h in handles] == list(r.r("plotLegend$lwd"))
        styles = {1: "-", 2: "--", 0: "None"}
        assert [h.get_linestyle() for h in handles] == [
            styles[int(v)] for v in r.r("plotLegend$lty")
        ]
        markers = [h.get_marker() for h in handles]
        assert markers == ["x" if v == 4 else "None" for v in r.r("plotLegend$pch")]
        assert legend._ncols == int(r.r("plotLegend$ncol")[0])


# ---------------------------------------------------------------------------
# Plots of ALM and of its predictions
# ---------------------------------------------------------------------------

N = 60
_rng = np.random.default_rng(11)
X1, X2 = _rng.normal(size=N), _rng.normal(size=N)
Y_POSITIVE = 20 + 1.5 * X1 - X2 + _rng.normal(size=N)
# Two planted outliers
Y_POSITIVE[4] += 15
Y_POSITIVE[39] -= 8
Y_COUNT = _rng.poisson(np.exp(1 + 0.3 * X1)).astype(float)
Y_BINARY = (_rng.random(N) < 0.6).astype(float)
Y_UNIT = 1 / (1 + np.exp(-(0.3 * X1 + 0.3 * _rng.normal(size=N))))
_OCCURS = _rng.random(N) < 0.7
Y_OCCURRENCE = Y_POSITIVE * _OCCURS
Y_COUNT_OCCURRENCE = Y_COUNT * _OCCURS
Y_EXP = _rng.exponential(np.exp(1 + 0.2 * X1))

ALM_CASES = {
    "dnorm": (Y_POSITIVE, "dnorm", False),
    "dlaplace": (Y_POSITIVE, "dlaplace", False),
    "ds": (Y_POSITIVE, "ds", False),
    "dgnorm": (Y_POSITIVE, "dgnorm", False),
    "dlogis": (Y_POSITIVE, "dlogis", False),
    "dalaplace": (Y_POSITIVE, "dalaplace", False),
    "dt": (Y_POSITIVE, "dt", False),
    "dfnorm": (Y_POSITIVE, "dfnorm", False),
    "dlnorm": (Y_POSITIVE, "dlnorm", False),
    "dgamma": (Y_POSITIVE, "dgamma", False),
    "dinvgauss": (Y_POSITIVE, "dinvgauss", False),
    "dexp": (Y_EXP, "dexp", False),
    "dpois": (Y_COUNT, "dpois", False),
    "dnbinom": (Y_COUNT, "dnbinom", False),
    "dgeom": (Y_COUNT, "dgeom", False),
    "dbinom": (Y_BINARY, "dbinom", False),
    "dbeta": (Y_UNIT, "dbeta", False),
    "plogis": (Y_BINARY, "plogis", False),
    "pnorm": (Y_BINARY, "pnorm", False),
    "dnorm occurrence": (Y_OCCURRENCE, "dnorm", True),
    "dgamma occurrence": (Y_OCCURRENCE, "dgamma", True),
    "dlaplace occurrence": (Y_OCCURRENCE, "dlaplace", True),
    "dpois occurrence": (Y_COUNT_OCCURRENCE, "dpois", True),
    "dnorm scale": (Y_POSITIVE, "dnorm", False),
    "dgamma scale": (Y_POSITIVE, "dgamma", False),
}
_ALPHA = {"dalaplace": 0.3}


def _fit_python(case):
    y, distribution, occurrence = ALM_CASES[case]
    extra = {"alpha": _ALPHA[distribution]} if distribution in _ALPHA else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if case.endswith("scale"):
            extra["scale_formula"] = np.column_stack([np.ones(N), X1])
        return ALM(
            distribution=distribution,
            occurrence="plogis" if occurrence else "none",
            **extra,
        ).fit(np.column_stack([np.ones(N), X1, X2]), pd.Series(y, name="y"))


def _fit_r(r, case):
    y, distribution, occurrence = ALM_CASES[case]
    for name, values in (("y", y), ("x1", X1), ("x2", X2)):
        r.globalenv[name] = r.FloatVector(values)
    extra = f", alpha={_ALPHA[distribution]}" if distribution in _ALPHA else ""
    occurrence_r = ', occurrence="plogis"' if occurrence else ""
    occurrence_r += ", scale=~x1" if case.endswith("scale") else ""
    r.r(
        "m <- suppressWarnings(alm(y~x1+x2, data.frame(y=y,x1=x1,x2=x2), "
        f'distribution="{distribution}"{extra}{occurrence_r}))'
    )


def _artists(ax, gid):
    return [a for a in ax.get_children() if a.get_gid() == gid]


def _xy(artist):
    """The coordinates of a line, of the tops of vertical bars or of a text."""
    if isinstance(artist, matplotlib.collections.LineCollection):
        # The bars of the non-finite values are not drawn
        return np.array(
            [
                [s[0, 0], s[1, 1]] if s.size else [np.nan] * 2
                for s in artist.get_segments()
            ]
        )
    if isinstance(artist, matplotlib.text.Annotation):
        return np.array([artist.xy], dtype=float)
    return np.asarray(artist.get_xydata(), dtype=float)


def _abline(artist):
    """A horizontal line as (0, h), any other one as (1, intercept, slope)."""
    if isinstance(artist, matplotlib.lines.AxLine):
        (x1, y1), slope = artist.get_xy1(), artist.get_slope()
        return np.array([1, y1 - slope * x1, slope])
    return np.array([0, artist.get_ydata()[0]])


_TEXT_POS = {("center", "top"): 1, ("right", "center"): 2, ("center", "bottom"): 3}

R_ALM_HARNESS = """
for (f in c("qqnorm", "qqplot", "qqline")) {
    fun <- if (f == "qqnorm") getS3method("qqnorm", "default") else
        get(f, envir=asNamespace("stats"))
    environment(fun) <- plotEnv
    assign(f, fun, envir=plotEnv)
}
for (e in parse(file.path("%s", "methods.R"))) {
    if (is.call(e) && identical(e[[1]], as.name("<-")) &&
        as.character(e[[2]]) %%in%% c("plot.greybox", "plot.predict.greybox")) {
        eval(e, envir=plotEnv)
    }
}
plotXY <- function(a) {
    if (!is.null(names(a)) && "x" %%in%% names(a)) {
        x <- a$x
        y <- a$y
    } else {
        a <- a[if (is.null(names(a))) seq_along(a) else names(a) == ""]
        x <- a[[1]]
        y <- if (length(a) > 1) a[[2]] else NULL
    }
    xy <- xy.coords(x, y)
    cbind(as.numeric(xy$x), as.numeric(xy$y))
}
plotAll <- function(name) lapply(plotOf(name), function(call) plotXY(call$args))
plotAbline <- function() {
    out <- list()
    for (call in plotOf("abline")) {
        a <- call$args
        if (!is.null(a$h)) {
            for (h in a$h) out[[length(out) + 1]] <- c(0, h)
        } else {
            ab <- if (is.null(a$a)) c(a[[1]], a[[2]]) else c(a$a, a$b)
            out[[length(out) + 1]] <- c(1, ab)
        }
    }
    out
}
plotTexts <- function() {
    lapply(plotOf("text"), function(call) {
        xy <- plotXY(call$args)
        cbind(xy, call$args$labels, if (is.null(call$args$pos)) NA else call$args$pos)
    })
}
"""


@pytest.fixture(scope="module")
def r_alm(r):
    r.r(R_ALM_HARNESS % R_DIR)
    return r


def _assert_xy(python, expected, rtol=1e-5):
    """Equal coordinates, the non-finite values of R being missing in Python
    (R does not draw them either)."""
    assert python.shape == expected.shape
    finite = np.all(np.isfinite(expected), axis=1)
    assert_allclose(python[finite], expected[finite], rtol=rtol, atol=1e-10)
    assert not np.any(np.isfinite(python[~finite, 1]))


def _compare_panel(r, ax, rtol=1e-5):
    """Compare the numbers drawn on ax with those recorded by the R harness."""
    r_plots = r_list(r, "plotAll('plot')")
    python_plots = _artists(ax, "plot")
    assert len(python_plots) == len(r_plots) == 1
    _assert_xy(_xy(python_plots[0]), r_plots[0], rtol)
    usr = np.array(r.r("plotOf('plot')[[1]]$usr"))
    assert_allclose([*ax.get_xlim(), *ax.get_ylim()], usr, rtol=rtol, atol=1e-10)
    assert ax.get_title() == r.r("plotOf('plot')[[1]]$args$main")[0]

    for gid in ("lines", "points", "polygon"):
        expected = r_list(r, f"plotAll('{gid}')")
        python = [
            p.get_xy()[:-1] if gid == "polygon" else _xy(p) for p in _artists(ax, gid)
        ]
        assert len(python) == len(expected), gid
        for p, e in zip(python, expected):
            _assert_xy(p, e, rtol)

    expected = r_list(r, "plotAbline()")
    python = [_abline(a) for a in _artists(ax, "abline")]
    assert len(python) == len(expected)
    for p, e in zip(python, expected):
        assert_allclose(p, e, rtol=rtol, atol=1e-10)

    expected = r_list(r, "plotTexts()")
    python = _artists(ax, "text")
    assert len(python) == sum(len(e) for e in expected)
    expected = np.concatenate(expected) if expected else np.empty((0, 4))
    for text, (x, y, label, pos) in zip(python, expected):
        assert_allclose(_xy(text)[0], [x, y], rtol=rtol, atol=1e-10)
        assert text.get_text() == f"{label:g}"
        if not np.isnan(pos):
            assert _TEXT_POS[(text.get_ha(), text.get_va())] == pos

    r_legends = r.r("plotArg('legend', 'legend')")
    if len(r_legends) == 0:
        assert ax.get_legend() is None
    else:
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == list(r_legends[0])


@requires_r
class TestPlotAlmCompare:
    @pytest.mark.parametrize("case", list(ALM_CASES))
    def test_panels(self, r_alm, case):
        r = r_alm
        _fit_r(r, case)
        model = _fit_python(case)
        assert_allclose(model.coefficients, np.array(r.r("coef(m)")), rtol=1e-6)
        for which in range(1, 17):
            if case == "dbeta" and which == 12:
                # hatvalues() fails for dbeta in both languages
                continue
            r.r(
                f"plotRun('suppressWarnings(suppressMessages(plot.greybox(m, "
                f"which={which}, legend=TRUE)))')"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax = _plot_alm(model, which=which, legend=True).axes[0]
            if case == "dbeta" and which == 6:
                assert r.r("length(plotOf('plot'))")[0] == 0
                assert not ax.axison
                continue
            try:
                _compare_panel(r, ax)
            except AssertionError as error:
                raise AssertionError(f"which={which}: {error}") from error
            plt.close("all")

    @pytest.mark.parametrize("which", [1, 2, 4, 6, 8, 13])
    def test_without_lowess_and_legend(self, r_alm, which):
        r = r_alm
        _fit_r(r, "dlaplace")
        r.r(f"plotRun('plot.greybox(m, which={which}, lowess=FALSE)')")
        _compare_panel(
            r, _plot_alm(_fit_python("dlaplace"), which, lowess=False).axes[0]
        )

    def test_arguments(self, r_alm):
        r = r_alm
        _fit_r(r, "dnorm")
        r.r(
            'plotRun(\'plot.greybox(m, which=2, level=0.8, main="T", xlab="a", '
            'ylab="b", ylim=c(-3, 3))\')'
        )
        ax = _plot_alm(
            _fit_python("dnorm"),
            2,
            level=0.8,
            main="T",
            xlab="a",
            ylab="b",
            ylim=(-3, 3),
        ).axes[0]
        _compare_panel(r, ax)
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("a", "b")


def _prediction_cases():
    new = np.column_stack([np.ones(5), X1[:5] + 1, X2[:5]])
    y_new = Y_POSITIVE[:5] + 1.5
    return {
        "holdout": ("dnorm", new, y_new, {"interval": "prediction"}),
        "no holdout": ("dnorm", new, None, {"interval": "prediction"}),
        "levels": (
            "dlaplace",
            new,
            y_new,
            {"interval": "prediction", "level": [0.9, 0.8]},
        ),
        "upper": ("dnorm", new, y_new, {"interval": "prediction", "side": "upper"}),
        "lower dgamma": (
            "dgamma",
            new,
            y_new,
            {"interval": "prediction", "side": "lower"},
        ),
        "in-sample": ("dnorm", None, None, {"interval": "confidence"}),
        "no interval": ("dlnorm", new, y_new, {}),
        "plogis": ("plogis", new, Y_BINARY[:5], {"interval": "prediction"}),
        "plogis in-sample": ("plogis", None, None, {}),
    }


@requires_r
class TestPlotPredictionCompare:
    @pytest.mark.parametrize("case", list(_prediction_cases()))
    @pytest.mark.parametrize("legend", [False, True])
    def test_prediction(self, r_alm, case, legend):
        r = r_alm
        distribution, new, y_new, options = _prediction_cases()[case]
        data = {
            "dnorm": Y_POSITIVE,
            "dlaplace": Y_POSITIVE,
            "dgamma": Y_POSITIVE,
            "dlnorm": Y_POSITIVE,
            "plogis": Y_BINARY,
        }[distribution]
        ALM_CASES["prediction"] = (data, distribution, False)
        _fit_r(r, "prediction")
        model = _fit_python("prediction")
        del ALM_CASES["prediction"]

        r_options = "".join(
            f", {k}={v!r}".replace("[", "c(").replace("]", ")")
            if not isinstance(v, str)
            else f", {k}='{v}'"
            for k, v in options.items()
        )
        if new is None:
            r.r(f"p <- predict(m{r_options})")
            prediction = model.predict(**options)
        else:
            r.globalenv["xn1"] = r.FloatVector(new[:, 1])
            r.globalenv["xn2"] = r.FloatVector(new[:, 2])
            newdata = "data.frame(x1=xn1, x2=xn2)"
            if y_new is not None:
                r.globalenv["yn"] = r.FloatVector(y_new)
                newdata = "data.frame(x1=xn1, x2=xn2, y=yn)"
            r.r(f"p <- predict(m, {newdata}{r_options})")
            prediction = model.predict(new, **options)
        r.r(f"plotRun('plot.predict.greybox(p, legend={str(legend).upper()})')")
        kwargs = {"legend": True} if legend else {}
        fig = _plot_prediction(prediction, model, y_new, new is not None, **kwargs)
        ax = fig.axes[0]
        assert ax.get_title() == r.r("rev(plotOf('plot'))[[1]]$args$main")[0]
        usr = np.array(r.r("rev(plotOf('plot'))[[1]]$usr"))
        assert_allclose([*ax.get_xlim(), *ax.get_ylim()], usr, rtol=1e-5)

        # The lines and points in the order R draws them, after the actuals
        lines = [line for line in ax.lines[1:] if to_hex(line.get_color()) != "#ff0000"]
        expected = r_list(r, "plotAll('lines')") + r_list(r, "plotAll('points')")
        assert len(lines) == len(expected)
        for line, e in zip(lines, expected):
            _assert_xy(line.get_xydata(), e)
        r_vline = list(r.r("as.numeric(unlist(plotArg('abline', 'v')))"))
        vlines = [
            line.get_xdata()[0]
            for line in ax.lines
            if to_hex(line.get_color()) == "#ff0000"
        ]
        assert_allclose(vlines, r_vline)
        r_polygons = r_list(r, "plotAll('polygon')")
        for patch, e in zip(ax.patches, r_polygons, strict=True):
            assert_allclose(patch.get_xy()[:-1], e, rtol=1e-5)
        if legend:
            labels = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
            assert labels == list(r.r("plotOf('legend')[[1]]$args$legend"))
        else:
            assert len(fig.axes) == 1


# ---------------------------------------------------------------------------
# Smoke tests of the plots of ALM
# ---------------------------------------------------------------------------


class TestPlotAlmSmoke:
    @pytest.mark.parametrize("case", list(ALM_CASES))
    def test_all_panels(self, case):
        model = _fit_python(case)
        which = [i for i in range(1, 17) if not (case == "dbeta" and i == 12)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = _plot_alm(model, which=which, legend=True)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == len(which)
        fig.canvas.draw()

    @pytest.mark.parametrize("which", range(1, 17))
    @pytest.mark.parametrize("legend, lowess", [(False, False), (True, True)])
    def test_each_panel(self, which, legend, lowess):
        fig = _plot_alm(
            _fit_python("dnorm occurrence"), which, legend=legend, lowess=lowess
        )
        assert len(fig.axes) == 1
        fig.canvas.draw()

    def test_default(self):
        fig = _plot_alm(_fit_python("dnorm"))
        assert [ax.get_title() for ax in fig.axes] == [
            "Actuals vs Fitted",
            "Standardised Residuals vs Fitted",
            "|Residuals| vs Fitted",
            "QQ plot of normal distribution",
        ]
        assert fig.axes[0].get_ylabel() == "Actual y"

    def test_count_studentised(self):
        with pytest.warns(UserWarning, match="Switching to standardised"):
            fig = _plot_alm(_fit_python("dpois"), [3, 9])
        assert fig.axes[0].get_title() == "Standardised Residuals vs Fitted"
        assert fig.axes[1].get_title() == "Standardised Residuals over observations"

    def test_beta_qq(self):
        with pytest.warns(UserWarning, match="QQ plots for the Beta"):
            _plot_alm(_fit_python("dbeta"), 6)

    def test_kwargs(self):
        ax = _plot_alm(
            _fit_python("dnorm"), 7, main="M", xlab="t", ylab="v", type="p"
        ).axes[0]
        assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ("M", "t", "v")
        assert _artists(ax, "plot")[0].get_linestyle() == "None"
        with pytest.raises(ValueError, match="type"):
            _plot_alm(_fit_python("dnorm"), 1, type="z")

    def test_ask(self, monkeypatch):
        prompts = []
        monkeypatch.setattr("builtins.input", prompts.append)
        _plot_alm(_fit_python("dnorm"), [1, 2, 3], ask=True)
        assert prompts == ["Hit <Return> to see next plot: "] * 2

    def test_prediction(self):
        model = _fit_python("dnorm")
        new = np.column_stack([np.ones(5), X1[:5], X2[:5]])
        fig = _plot_prediction(
            model.predict(new, interval="prediction"), model, Y_POSITIVE[:5]
        )
        assert fig.axes[0].get_title() == "Forecast for the variable y"
        fig = _plot_prediction(model.predict(), model, newdata_provided=False)
        assert fig.axes[0].get_title() == "Fitted values for the variable y"
