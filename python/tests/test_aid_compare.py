"""rpy2-based comparison tests for aid() and aid_cat().

Compares Python's ``greybox.aid``/``aid_cat`` against R's
``greybox::aid``/``aidCat`` across the canonical demand categories.
Categorical decisions and integer indices must match exactly; numeric IC
values are compared with a coarse tolerance.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

import rpy2.robjects as ro
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter

from greybox import aid, aid_cat
from greybox.pointlik import point_lik_cumulative
from greybox.smoothers import lowess, supsmu
from greybox.alm import ALM
from greybox.formula import formula

ro.r["library"]("greybox")


# The numpy conversion only inside these tests, not in the later test files
@pytest.fixture(autouse=True)
def numpy_conversion():
    with localconverter(default_converter + numpy2ri.converter):
        yield


def _r_aid(y, **kwargs):
    """Run R's aid() and return its $name, $new, $obsolete, stockouts."""
    ro.globalenv["y_r"] = ro.FloatVector(np.asarray(y, dtype=float).tolist())
    extra = ""
    for k, v in kwargs.items():
        if isinstance(v, str):
            extra += f", {k}='{v}'"
        else:
            extra += f", {k}={v}"
    ro.r(f"res <- aid(y_r{extra})")
    name = str(ro.r("res$name")[0])
    is_new = bool(ro.r("res$new")[0])
    is_obs = bool(ro.r("res$obsolete")[0])
    start_r = ro.r("res$stockouts$start")
    end_r = ro.r("res$stockouts$end")
    start = (
        np.array([], dtype=int) if start_r is ro.NULL else np.array(start_r, dtype=int)
    )
    end = np.array([], dtype=int) if end_r is ro.NULL else np.array(end_r, dtype=int)
    type1 = str(ro.r("res$type$type1")[0])
    type2 = str(ro.r("res$type$type2")[0])
    type2a_r = ro.r("res$type$type2a")
    type2a = None if type2a_r is ro.NULL else str(type2a_r[0])
    return {
        "name": name,
        "new": is_new,
        "obsolete": is_obs,
        "start": start,
        "end": end,
        "type1": type1,
        "type2": type2,
        "type2a": type2a,
    }


def _assert_aid_matches(py_result, r_result):
    """Assert the categorical aid() decisions match."""
    assert py_result.name == r_result["name"], (
        f"name mismatch: py={py_result.name!r} r={r_result['name']!r}"
    )
    assert bool(py_result.new) == r_result["new"]
    assert bool(py_result.obsolete) == r_result["obsolete"]
    assert py_result.type["type1"] == r_result["type1"]
    assert py_result.type["type2"] == r_result["type2"]
    assert py_result.type["type2a"] == r_result["type2a"]
    py_start = (
        np.array([], dtype=int)
        if py_result.stockouts["start"] is None
        else np.asarray(py_result.stockouts["start"], dtype=int)
    )
    py_end = (
        np.array([], dtype=int)
        if py_result.stockouts["end"] is None
        else np.asarray(py_result.stockouts["end"], dtype=int)
    )
    assert_array_equal(py_start, r_result["start"])
    assert_array_equal(py_end, r_result["end"])


class TestSupsmuLowessVsR:
    """Native smoothers must match R bit-for-bit (machine precision)."""

    def test_lowess_matches_r(self):
        ro.r("set.seed(42); x <- as.numeric(1:30); y <- sin(x/3) + rnorm(30, 0, 0.2)")
        x_r = np.array(ro.r("x"))
        y_r = np.array(ro.r("y"))
        r_smoothed = np.array(ro.r("lowess(x, y)$y"))
        py = lowess(x_r, y_r)
        assert_allclose(py["y"], r_smoothed, rtol=1e-8, atol=1e-8)

    def test_supsmu_cv_matches_r(self):
        ro.r("set.seed(42); x <- as.numeric(1:40); y <- sin(x/4) + rnorm(40, 0, 0.2)")
        x_r = np.array(ro.r("x"))
        y_r = np.array(ro.r("y"))
        r_smoothed = np.array(ro.r("supsmu(x, y)$y"))
        py = supsmu(x_r, y_r)
        assert_allclose(py["y"], r_smoothed, rtol=1e-8, atol=1e-8)

    def test_supsmu_fixed_span_matches_r(self):
        ro.r("set.seed(11); x <- as.numeric(1:40); y <- cos(x/4) + rnorm(40, 0, 0.1)")
        x_r = np.array(ro.r("x"))
        y_r = np.array(ro.r("y"))
        r_smoothed = np.array(ro.r("supsmu(x, y, span=0.3)$y"))
        py = supsmu(x_r, y_r, span=0.3)
        assert_allclose(py["y"], r_smoothed, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize(
        "span, bass, weighted",
        [
            ("cv", 0, False),
            ("cv", 2, False),
            ("cv", 5, False),
            ("cv", 8, False),
            (0.1, 0, False),
            (0.3, 0, False),
            ("cv", 5, True),
            (0.3, 0, True),
        ],
    )
    def test_supsmu_ties_match_r(self, span, bass, weighted):
        """With tied x, R returns one point per distinct x, ordered by (x, y)."""
        ro.r(
            "set.seed(41); x <- round(runif(80, 0, 4), 1); "
            "y <- sin(x) + rnorm(80, 0, 0.3); wt <- runif(80, 0.5, 2)"
        )
        x_r = np.array(ro.r("x"))
        y_r = np.array(ro.r("y"))
        wt = np.array(ro.r("wt")) if weighted else None
        r_span = '"cv"' if span == "cv" else span
        r_wt = "wt" if weighted else "rep(1, length(y))"
        ro.r(f"rOut <- supsmu(x, y, wt={r_wt}, span={r_span}, bass={bass})")
        py = supsmu(x_r, y_r, wt=wt, span=span, bass=bass)
        assert len(py["x"]) == len(np.unique(x_r))
        assert_allclose(py["x"], np.array(ro.r("rOut$x")), rtol=0, atol=0)
        assert_allclose(py["y"], np.array(ro.r("rOut$y")), rtol=1e-13, atol=1e-13)


class TestPointLikCumulativeVsR:
    """pointLikCumulative parity for the discrete distributions used by aid().

    Compares the CDF formula directly. Each test fits the same model in R
    and Python, then computes ``pointLikCumulative`` on identical (R-fitted)
    ``mu`` values to isolate the CDF logic from optimizer convergence.
    """

    def _stub_dgeom(self, y, mu):
        """Build a fitted-style ALM whose attributes provide y and mu."""
        y_vec, X = formula("y ~ 1", {"y": y})
        model = ALM(distribution="dgeom").fit(X, y_vec)
        model.fitted_values_ = np.asarray(mu, dtype=float)
        model._y_train_ = np.asarray(y, dtype=float)
        return model

    def test_dgeom(self):
        ro.r("set.seed(0); y <- rpois(60, 1)")
        y_r = np.array(ro.r("y"))
        ro.r("m <- alm(y ~ 1, data=data.frame(y=y), distribution='dgeom')")
        mu_r = np.array(ro.r("m$mu"))
        r_cdf = np.array(ro.r("greybox:::pointLikCumulative(m)"))

        model = self._stub_dgeom(y_r, mu_r)
        py_cdf = point_lik_cumulative(model)
        assert_allclose(py_cdf, r_cdf, rtol=1e-10, atol=1e-12)

    def test_dpois(self):
        ro.r("set.seed(2); y <- rpois(50, 3)")
        y_r = np.array(ro.r("y"))
        ro.r("m <- alm(y ~ 1, data=data.frame(y=y), distribution='dpois')")
        mu_r = np.array(ro.r("m$mu"))
        r_cdf = np.array(ro.r("greybox:::pointLikCumulative(m)"))

        y_vec, X = formula("y ~ 1", {"y": y_r})
        model = ALM(distribution="dpois").fit(X, y_vec)
        model.fitted_values_ = mu_r
        model._y_train_ = y_r
        py_cdf = point_lik_cumulative(model)
        assert_allclose(py_cdf, r_cdf, rtol=1e-10, atol=1e-12)


class TestAidVsR:
    """Compare aid() decisions against R aid() on synthetic series."""

    def _round_trip(self, y, **kwargs):
        py_r = aid(y, **kwargs)
        r_r = _r_aid(y, **kwargs)
        _assert_aid_matches(py_r, r_r)
        return py_r, r_r

    def test_regular_normal(self):
        ro.r("set.seed(7); y <- rnorm(100, 10, 2)")
        y = np.array(ro.r("y"))
        py_r, r_r = self._round_trip(y)
        assert py_r.name == "regular fractional"

    def test_intermittent_poisson(self):
        ro.r("set.seed(42); y <- rpois(120, 0.7)")
        y = np.array(ro.r("y"))
        self._round_trip(y)

    def test_regular_count_poisson(self):
        ro.r("set.seed(3); y <- rpois(150, 8)")
        y = np.array(ro.r("y"))
        self._round_trip(y)

    def test_injected_stockout(self):
        ro.r("set.seed(0); y <- rpois(100, 3); y[41:50] <- 0")
        y = np.array(ro.r("y"))
        py_r, r_r = self._round_trip(y)
        assert len(py_r.stockouts["start"]) >= 1

    def test_new_product(self):
        ro.r("set.seed(5); y <- c(rep(0, 20), rpois(80, 5))")
        y = np.array(ro.r("y"))
        py_r, _ = self._round_trip(y)
        assert py_r.new is True

    def test_obsolete_product(self):
        ro.r("set.seed(11); y <- c(rpois(80, 5), rep(0, 20))")
        y = np.array(ro.r("y"))
        py_r, _ = self._round_trip(y)

    def test_binary(self):
        ro.r("set.seed(9); y <- rbinom(100, 1, 0.3)")
        y = np.array(ro.r("y"))
        self._round_trip(y)

    def test_low_volume(self):
        ro.r("set.seed(8); y <- rpois(120, 0.4)")
        y = np.array(ro.r("y"))
        self._round_trip(y)

    def test_fractional_intermittent(self):
        ro.r("set.seed(4); occ <- rbinom(150, 1, 0.6); y <- occ * rnorm(150, 5, 1.5)")
        y = np.array(ro.r("y"))
        self._round_trip(y)


class TestAidDummies:
    """Stockout / new / obsolete dummy vectors must match R element-by-element."""

    def test_stockout_dummy(self):
        ro.r(
            "set.seed(0); y <- rpois(100, 3); y[41:50] <- 0; "
            "r_dummy <- aid(y)$stockouts$dummy"
        )
        y = np.array(ro.r("y"))
        r_dummy = np.array(ro.r("r_dummy"), dtype=int)
        py_r = aid(y)
        assert_array_equal(py_r.stockouts["dummy"], r_dummy)

    def test_new_dummy(self):
        ro.r(
            "set.seed(5); y <- c(rep(0, 20), rpois(80, 5)); "
            "r_dummy <- aid(y)$stockouts$new"
        )
        y = np.array(ro.r("y"))
        r_dummy = np.array(ro.r("r_dummy"), dtype=int)
        py_r = aid(y)
        assert_array_equal(py_r.stockouts["new"], r_dummy)


class TestAidCatVsR:
    """aid_cat across multiple series should agree with R aidCat."""

    def test_matrix_input(self):
        ro.r(
            "set.seed(1); "
            "xreg <- cbind("
            "x1=rpois(80, 1), x2=rpois(80, 2), "
            "x3=rpois(80, 5), x4=rnorm(80, 10, 2)"
            "); "
            "r_res <- aidCat(xreg)"
        )
        xreg = np.array(ro.r("xreg"))
        r_anomalies = dict(
            zip(["New", "Stockouts", "Old"], list(ro.r("r_res$anomalies")))
        )

        py_r = aid_cat(xreg)
        assert int(py_r.anomalies["New"]) == int(r_anomalies["New"])
        assert int(py_r.anomalies["Stockouts"]) == int(r_anomalies["Stockouts"])
        assert int(py_r.anomalies["Old"]) == int(r_anomalies["Old"])

        r_categories = np.array(ro.r("as.character(r_res$categories)"))
        for py_cat, r_cat in zip(py_r.categories, r_categories):
            assert str(py_cat) == str(r_cat)

    def test_named_list_input(self):
        ro.r(
            "set.seed(2); "
            "sl <- list(p=rpois(70, 1), q=rnorm(90, 10, 2), r=rpois(80, 4)); "
            "r_res <- aidCat(sl); r_unnamed <- aidCat(unname(sl))"
        )
        series = {k: np.array(ro.r(f"sl${k}")) for k in ("p", "q", "r")}
        r_categories = list(ro.r("as.character(r_res$categories)"))
        assert list(ro.r("names(r_res$categories)")) == ["p", "q", "r"]
        assert ro.r("names(r_unnamed$categories)") is ro.NULL

        py_named = aid_cat(series)
        assert list(py_named.categories.index) == ["p", "q", "r"]
        assert [str(c) for c in py_named.categories] == r_categories

        py_unnamed = aid_cat(list(series.values()))
        assert list(py_unnamed.categories.index) == [0, 1, 2]
        assert [str(c) for c in py_unnamed.categories] == r_categories


class TestAidStockoutMuVsR:
    """The stockout model's mu is padded with 100 for new/obsolete products."""

    def test_new_and_obsolete_padding(self):
        ro.r(
            "set.seed(41); invisible(rpois(120, 0.7)); "
            "y <- c(rep(0,15), rpois(80,3), rep(0,25)); r_res <- aid(y)"
        )
        y = np.array(ro.r("y"))
        r_mu = np.array(ro.r("r_res$models$stockout$mu"))
        py_r = aid(y)
        assert py_r.new and py_r.obsolete
        assert len(py_r.models["stockout"].mu_) == len(r_mu) == 77
        assert_allclose(py_r.models["stockout"].mu_, r_mu, rtol=1e-6)
