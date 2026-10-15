"""Compare stick() and rmcb() between R and Python.

stick() of R is sourced from the R/ folder of the repository, so that the
comparison does not depend on the installed version of greybox.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

import rpy2.robjects as ro

from greybox import rmcb, stick

ro.r["library"]("greybox")
R_STICK = Path(__file__).resolve().parents[2] / "R" / "stick.R"
ro.r(f'stickFixed <- new.env(); sys.source("{R_STICK}", envir=stickFixed)')


@pytest.fixture(scope="module")
def series():
    ro.r(
        "set.seed(3); stickY <- 100 + rep(c(5,-3,2,-4), length.out=50) +"
        " 0.3*(1:50) + rep(sin(2*pi*(1:12)/12)*6, length.out=50) + rnorm(50)"
    )
    return np.array(ro.r("stickY"))


class TestStickCompare:
    def _compare(self, y_python, r_call):
        py = stick(y_python, lags=[int(lag) for lag in ro.r(f"{r_call}$lags")])
        r_anova = ro.r(f"as.matrix({r_call}$anova)")
        r_rows = [name.strip() for name in ro.r(f"rownames({r_call}$anova)")]
        assert list(py.anova.index) == r_rows
        # Without residual degrees of freedom R has no F value and Pr(>F)
        r_anova = np.array(r_anova).reshape(len(r_rows), -1, order="F")
        n_cols = r_anova.shape[1]
        assert_allclose(py.anova.iloc[:, :n_cols].to_numpy(dtype=float), r_anova)
        assert py.anova.iloc[:, n_cols:].isna().all().all()
        r_strength = ro.r(f"{r_call}$strength")
        assert list(py.strength.index) == list(r_strength.names)
        assert_allclose(py.strength.to_numpy(), np.array(r_strength), rtol=1e-8)
        return py

    def test_nested_lags(self, series):
        py = self._compare(series, "stickFixed$stick(stickY, lags=c(4,12))")
        # seasonal12 is partially aliased with seasonal4
        assert_allclose(py.anova["Df"].to_numpy(), [3, 8, 4, 34])

    def test_missing_values(self, series):
        ro.r("stickNA <- stickY; stickNA[c(7,20)] <- NA")
        y = series.copy()
        y[[6, 19]] = np.nan
        py = self._compare(y, "stickFixed$stick(stickNA, lags=12)")
        assert np.isfinite(py.strength.to_numpy()).all()

    def test_aliased_trend(self, series):
        # The second cycle is missing: the trend is not included
        ro.r("stickShort <- stickY[1:20]; stickShort[13:20] <- NA")
        y = series[:20].copy()
        y[12:] = np.nan
        py = self._compare(y, "stickFixed$stick(stickShort, lags=12)")
        assert "trend" not in py.anova.index


class TestRmcbCompare:
    @pytest.mark.parametrize("distribution", ["dlaplace", "dlnorm", "dnorm"])
    def test_intervals(self, distribution):
        ro.r(
            "set.seed(41); rmcbData <- matrix(abs(rnorm(160, 5, 2)), 40, 4,"
            " dimnames=list(NULL, paste0('M', 1:4))); rmcbData[,2] <- rmcbData[,2]+1"
        )
        data = np.array(ro.r("rmcbData"))
        py = rmcb(data, distribution=distribution)
        r = ro.r(f'rmcb(rmcbData, outplot="none", distribution="{distribution}")')
        assert_allclose(py.interval.to_numpy(), np.array(r.rx2("interval")), rtol=1e-6)
        assert py.p_value == pytest.approx(r.rx2("p.value")[0], rel=1e-6)
