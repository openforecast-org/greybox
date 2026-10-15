"""rpy2-based comparison tests for xreg_expander, xreg_transformer,
xreg_multiplier and temporal_dummy against R's xregExpander, xregTransformer,
xregMultiplier and temporaldummy."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError

from greybox.xreg import (
    temporal_dummy,
    xreg_expander,
    xreg_multiplier,
    xreg_transformer,
)

ro.r["library"]("greybox")
ro.r["library"]("zoo")
# The temporaldummy() methods of this tree, over those of the installed greybox
ro.r(
    f"""
    tdEnv <- new.env();
    sys.source("{Path(__file__).parents[2] / "R" / "temporaldummy.R"}", envir=tdEnv);
    for(tdMethod in setdiff(ls(tdEnv), "temporaldummy")){{
        registerS3method("temporaldummy", sub("temporaldummy.", "", tdMethod, fixed=TRUE),
                         get(tdMethod, tdEnv), envir=asNamespace("greybox"));
    }}
    """
)


def _r_matrix(expr):
    """Evaluate an R expression returning a matrix, as numpy."""
    return np.array(ro.r(f"unclass(as.matrix({expr}))"), dtype=float)


def _r_names(expr):
    """The column names of the R matrix."""
    return list(ro.r(f"colnames({expr})"))


@pytest.fixture(scope="module")
def xreg():
    ro.r(
        "set.seed(11); xr <- cbind(a=round(50+cumsum(rnorm(40)),3), "
        "c=round(rnorm(40,10,2),3))"
    )
    return _r_matrix("xr")


class TestXregExpanderVsR:
    @pytest.mark.parametrize("gaps", ["naive", "zero", "NAs"])
    def test_matrix_layout(self, xreg, gaps):
        r = _r_matrix(f"xregExpander(xr, lags=c(-2,1,-1,2), gaps='{gaps}')")
        py = xreg_expander(xreg, lags=[-2, 1, -1, 2], gaps=gaps)
        assert_allclose(py, r, equal_nan=True)

    def test_matrix_auto(self, xreg):
        pytest.importorskip("smooth")
        r = _r_matrix("xregExpander(xr, lags=c(-1,1), gaps='auto')")
        assert_allclose(xreg_expander(xreg, lags=[-1, 1], gaps="auto"), r, rtol=1e-6)

    def test_zero_lags_warn(self, xreg):
        r = _r_matrix("suppressWarnings(xregExpander(xr, lags=0))")
        with pytest.warns(UserWarning, match="leads or lags"):
            py = xreg_expander(xreg, lags=[0])
        assert_allclose(py, r)


class TestXregTransformerVsR:
    def test_values_without_clipping(self):
        r = _r_matrix(
            "suppressWarnings(xregTransformer(cbind(a=c(0.5,0,-1,4,800)), "
            "functions=c('log','exp','inv','sqrt','square')))"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            py = xreg_transformer(
                np.array([0.5, 0, -1, 4, 800]),
                functions=["log", "exp", "inv", "sqrt", "square"],
            )
        assert_allclose(py, r, equal_nan=True)

    def test_layout(self, xreg):
        r = _r_matrix("xregTransformer(xr, functions=c('square','log'))")
        py = xreg_transformer(xreg, functions=["square", "log"])
        assert_allclose(py, r)


class TestXregMultiplierVsR:
    def test_values(self):
        ro.r("xm <- cbind(a=c(1,2,3.5), b=c(2,-3,4), c=c(3,4,0.5))")
        r = _r_matrix("xregMultiplier(xm)")
        assert_allclose(xreg_multiplier(_r_matrix("xm")), r)


@pytest.fixture(scope="module")
def xreg_frame(xreg):
    return pd.DataFrame(xreg, columns=["a", "c"], index=pd.RangeIndex(40) * 2)


class TestXregNamesVsR:
    def test_expander(self, xreg_frame):
        expr = "xregExpander(as.data.frame(xr), lags=c(-2,1,-1,2), gaps='naive')"
        py = xreg_expander(xreg_frame, lags=[-2, 1, -1, 2], gaps="naive")
        assert list(py.columns) == _r_names(expr)
        assert py.index.equals(xreg_frame.index)
        assert_allclose(py.to_numpy(), _r_matrix(expr))

    def test_expander_series(self, xreg_frame):
        expr = "xregExpander(xr[,1], lags=c(-1,1), gaps='zero')"
        py = xreg_expander(xreg_frame["a"].rename(None), lags=[-1, 1], gaps="zero")
        assert list(py.columns) == _r_names(expr)
        assert_allclose(py.to_numpy(), _r_matrix(expr))

    def test_transformer(self, xreg_frame):
        expr = "xregTransformer(as.data.frame(xr), functions=c('square','log','inv'))"
        py = xreg_transformer(xreg_frame, functions=["square", "log", "inv"])
        assert list(py.columns) == _r_names(expr)
        assert py.index.equals(xreg_frame.index)
        assert_allclose(py.to_numpy(), _r_matrix(expr))

    def test_multiplier(self):
        ro.r("xm <- data.frame(a=c(1,2,3.5), b=c(2,-3,4), c=c(3,4,0.5))")
        frame = pd.DataFrame(
            {"a": [1, 2, 3.5], "b": [2, -3, 4], "c": [3, 4, 0.5]}, index=list("pqr")
        )
        py = xreg_multiplier(frame)
        assert list(py.columns) == _r_names("xregMultiplier(xm)")
        assert py.index.equals(frame.index)
        assert_allclose(py.to_numpy(), _r_matrix("xregMultiplier(xm)"))


TYPES = ["month", "quarter", "week", "day", "hour", "halfhour", "minute", "second"]
OFS = ["year", "quarter", "month", "week", "day", "hour", "minute"]


def _compare_dummies(r_expr, py_input, **kwargs):
    """Compare R's dummies and factor with those of temporal_dummy."""
    py = temporal_dummy(py_input, **kwargs)
    py_factor = temporal_dummy(py_input, factors=True, **kwargs)
    assert list(py.columns) == _r_names(r_expr)
    assert_allclose(py.to_numpy(), _r_matrix(r_expr))
    r_factor = r_expr[:-1] + ", factors=TRUE)"
    assert [str(v) for v in py_factor] == list(ro.r(f"as.character({r_factor})"))
    assert [str(v) for v in py_factor.cat.categories] == list(
        ro.r(f"levels({r_factor})")
    )
    return py


class TestTemporalDummyDateVsR:
    @pytest.mark.parametrize("start", ["2020-02-27", "2020-08-20", "2021-12-17"])
    @pytest.mark.parametrize(
        "type_of",
        [(t, o) for t in ["month", "quarter", "week"] for o in OFS]
        + [("day", o) for o in ["year", "quarter", "month", "week"]],
    )
    @pytest.mark.parametrize("h", [0, 5])
    def test_supported(self, start, type_of, h):
        type, of = type_of
        r_expr = (
            f"temporaldummy(as.Date('{start}')+0:39, type='{type}', of='{of}', h={h})"
        )
        dates = np.datetime64(start) + np.arange(40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            py = _compare_dummies(r_expr, pd.Series(dates), type=type, of=of, h=h)
            # numpy input gives the same dummies as an ndarray
            assert_allclose(temporal_dummy(dates, type=type, of=of, h=h), py.to_numpy())

    def test_week_of_month_warns(self):
        dates = np.datetime64("2020-02-27") + np.arange(40)
        with pytest.warns(UserWarning, match="of=month is not yet implemented"):
            temporal_dummy(dates, type="week", of="month")

    @pytest.mark.parametrize("type", ["hour", "halfhour", "minute", "second"])
    def test_hours_of_dates(self, type):
        assert ro.r(
            f"is.null(suppressWarnings(temporaldummy(as.Date('2020-02-27')+0:39, "
            f"type='{type}')))"
        )[0]
        with pytest.warns(UserWarning, match="Date class does not support"):
            assert (
                temporal_dummy(np.datetime64("2020-02-27") + np.arange(40), type=type)
                is None
            )

    @pytest.mark.parametrize("of", ["day", "hour", "minute"])
    def test_unsupported(self, of):
        with pytest.raises(RRuntimeError):
            ro.r(f"temporaldummy(as.Date('2020-02-27')+0:39, type='day', of='{of}')")
        with pytest.raises(ValueError):
            temporal_dummy(
                np.datetime64("2020-02-27") + np.arange(40), type="day", of=of
            )


class TestTemporalDummyPOSIXctVsR:
    @pytest.mark.parametrize(
        "start, by, n, tz",
        [
            ("2020-02-27 21:00:00", "h", 50, "UTC"),
            ("2020-03-30 20:00:00", "h", 60, "UTC"),
            ("2021-12-31 18:00:00", "h", 30, "America/New_York"),
            ("2020-03-31 23:50:00", "min", 30, "UTC"),
            ("2020-06-07 23:58:00", "min", 30, "Europe/London"),
        ],
    )
    @pytest.mark.parametrize(
        "type_of",
        [(t, o) for t in ["month", "quarter"] for o in OFS]
        + [("week", "year"), ("day", "year"), ("day", "quarter"), ("day", "month")]
        + [("day", "week")]
        + [("hour", o) for o in ["year", "quarter", "month", "week", "day"]]
        + [("minute", o) for o in ["year", "quarter", "month", "week", "day"]]
        + [("minute", "hour")],
    )
    @pytest.mark.parametrize("h", [0, 4])
    def test_supported(self, start, by, n, tz, type_of, h):
        type, of = type_of
        step = 3600 if by == "h" else 60
        r_expr = (
            f"temporaldummy(as.POSIXct('{start}', tz='{tz}')+{step}*(0:{n - 1}), "
            f"type='{type}', of='{of}', h={h})"
        )
        times = pd.date_range(start, periods=n, freq=by, tz=tz)
        _compare_dummies(r_expr, times, type=type, of=of, h=h)

    def test_seconds_of_minute(self):
        r_expr = (
            "temporaldummy(as.POSIXct('2020-02-27 23:59:30', tz='UTC')+0:69, "
            "type='second', of='minute', h=3)"
        )
        times = pd.date_range("2020-02-27 23:59:30", periods=70, freq="s")
        _compare_dummies(r_expr, times, type="second", of="minute", h=3)

    @pytest.mark.parametrize(
        "type_of",
        [("halfhour", "day"), ("second", "hour"), ("hour", "hour")]
        + [("minute", "minute"), ("day", "day")],
    )
    def test_unsupported(self, type_of):
        type, of = type_of
        with pytest.raises(RRuntimeError):
            ro.r(
                "temporaldummy(as.POSIXct('2020-02-27', tz='UTC')+3600*(0:9), "
                f"type='{type}', of='{of}')"
            )
        with pytest.raises(ValueError):
            temporal_dummy(
                pd.date_range("2020-02-27", periods=10, freq="h"), type=type, of=of
            )

    def test_zoo(self):
        r_expr = (
            "temporaldummy(zoo(1:30, as.POSIXct('2020-02-27 21:00:00', tz='UTC')+"
            "3600*(0:29)), type='hour', of='week', h=3)"
        )
        series = pd.Series(
            np.arange(1.0, 31),
            index=pd.date_range("2020-02-27 21:00", periods=30, freq="h"),
        )
        py = _compare_dummies(r_expr, series, type="hour", of="week", h=3)
        assert py.index[-1] == pd.Timestamp("2020-02-29 05:00")


class TestTemporalDummyTsVsR:
    @pytest.mark.parametrize(
        "frequency, type, of",
        [(4, "quarter", "year"), (12, "month", "year"), (7, "day", "week")]
        + [(24, "hour", "day"), (52, "week", "year"), (48, "halfhour", "day")]
        + [(60, "minute", "hour")],
    )
    @pytest.mark.parametrize("h", [0, 6])
    def test_frequency(self, frequency, type, of, h):
        r_expr = f"temporaldummy(ts(1:30, frequency={frequency}), h={h})"
        r = _r_matrix(r_expr)
        with pytest.warns(DeprecationWarning, match="freq is deprecated"):
            assert_allclose(temporal_dummy(np.arange(30), freq=frequency, h=h), r)
        py = temporal_dummy(pd.Series(np.arange(30.0)), type=type, of=of, h=h)
        assert list(py.columns) == _r_names(r_expr)
        assert_allclose(py.to_numpy(), r)
        assert py.index.equals(pd.RangeIndex(30 + h))
        # The ts of the factor holds its values, the levels are those of the default
        r_factor = r_expr[:-1] + ", factors=TRUE)"
        py_factor = temporal_dummy(np.arange(30), type=type, of=of, h=h, factors=True)
        assert [str(v) for v in py_factor] == list(ro.r(f"as.character({r_factor})"))
        assert [str(v) for v in py_factor.categories] == list(
            ro.r(f"levels(temporaldummy(1:30, '{type}', '{of}', TRUE, h={h}))")
        )

    @pytest.mark.parametrize("frequency, type", [(12, "month"), (4, "quarter")])
    def test_start_in_cycle(self, frequency, type):
        # The dummies of a ts follow its cycle: R's ts from the fifth / third period
        start = 5 if frequency == 12 else 3
        r = _r_matrix(
            f"temporaldummy(ts(1:20, frequency={frequency}, start=c(2000,{start})), h=2)"
        )
        py = temporal_dummy(np.arange(20 + start - 1), type=type, h=2)
        assert_allclose(py[start - 1 :], r)

    @pytest.mark.parametrize(
        "type_of",
        [("day", "quarter"), ("hour", "quarter"), ("minute", "day"), ("week", "day")],
    )
    def test_default(self, type_of):
        type, of = type_of
        r_expr = f"temporaldummy(1:10, type='{type}', of='{of}', h=2)"
        assert_allclose(
            temporal_dummy(np.arange(10), type=type, of=of, h=2), _r_matrix(r_expr)
        )
