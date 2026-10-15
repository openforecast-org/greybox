"""Tests comparing R and Python error measures.

These tests ensure that the point, interval and half-moment measures produce
the same values as the R greybox package, including NA, zero and Inf cases.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from greybox.hm import asymmetry, cextremity, extremity, ham, hm, mre
from greybox.point_measures import (
    gmrae,
    mape,
    mase,
    measures,
    mpe,
    rame,
    rmsse,
    same,
    sce,
    smse,
    spis,
)
from greybox.quantile_measures import mis, pinball, rmis, smis

ro = pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

ro.r["library"]("greybox")
ro.r(
    """msr <- list(h=c(5,0,3,NA,7,2), h2=c(5,0,3,4,7,2), f=c(4,1,3,2,5,3),
                   b=c(5,2,3,1,6,2), a=c(3,0,4,6,0,2,5),
                   lo=c(3,1,2,1,4,NA), lo2=c(3,1,2,1,4,1), up=c(6,2,4,3,6,4),
                   x=c(1,3,2,8,0,4,NA,5), xc=c(1,3,2,8,0,4,5))"""
)

nan = np.nan
h = np.array([5, 0, 3, nan, 7, 2.0])
h2 = np.array([5, 0, 3, 4, 7, 2.0])
f = np.array([4, 1, 3, 2, 5, 3.0])
b = np.array([5, 2, 3, 1, 6, 2.0])
a = np.array([3, 0, 4, 6, 0, 2, 5.0])
lo = np.array([3, 1, 2, 1, 4, nan])
lo2 = np.array([3, 1, 2, 1, 4, 1.0])
up = np.array([6, 2, 4, 3, 6, 4.0])
x = np.array([1, 3, 2, 8, 0, 4, nan, 5])
xc = x[~np.isnan(x)]

with np.errstate(divide="ignore", invalid="ignore"):
    CASES = [
        ("MPE(h, f)", mpe(h, f)),
        ("MAPE(h, f)", mape(h, f)),
        ("MPE(h, f, na.rm=FALSE)", mpe(h, f, na_rm=False)),
        ("MPE(c(0, 2), c(0, 1))", mpe(np.array([0, 2.0]), np.array([0, 1.0]))),
        ("MAPE(c(0, 2), c(0, 1))", mape(np.array([0, 2.0]), np.array([0, 1.0]))),
        ("sPIS(h, f, 2)", spis(h, f, 2)),
        ("sPIS(h, f, 2, na.rm=FALSE)", spis(h, f, 2, na_rm=False)),
        ("sCE(h, f, 2)", sce(h, f, 2)),
        ("GMRAE(h, f, b)", gmrae(h, f, b)),
        ("GMRAE(h2, f, b)", gmrae(h2, f, b)),
        ("GMRAE(h2, f, b, na.rm=FALSE)", gmrae(h2, f, b, na_rm=False)),
        (
            "GMRAE(c(1, 2, 3), c(1, 2.5, 3.5), c(2, 3, 4))",
            gmrae(
                np.array([1, 2, 3.0]), np.array([1, 2.5, 3.5]), np.array([2, 3, 4.0])
            ),
        ),
        ("MASE(h, f, 0)", mase(h, f, 0)),
        (
            "MASE(holdout=h, forecast=f, scale=2)",
            mase(holdout=h, forecast=f, scale=2),
        ),
        (
            "GMRAE(holdout=h2, forecast=f, benchmark=b)",
            gmrae(holdout=h2, forecast=f, benchmark=b),
        ),
        ("RMSSE(h, f, 0)", rmsse(h, f, 0)),
        ("SAME(h, f, 0)", same(h, f, 0)),
        ("SAME(f, f, 0)", same(f, f, 0)),
        ("sMSE(h, f, 0)", smse(h, f, 0)),
        ("sPIS(h, f, 0)", spis(h, f, 0)),
        ("sCE(h, f, 0)", sce(h, f, 0)),
        ("sMIS(h, lo2, up, 0)", smis(h, lo2, up, 0)),
        (
            "rAME(c(1, 2), c(1, 2), c(0, 3))",
            rame(np.array([1, 2.0]), np.array([1, 2.0]), np.array([0, 3.0])),
        ),
        (
            "rAME(c(1, 2), c(0, 2), c(0, 3))",
            rame(np.array([1, 2.0]), np.array([0, 2.0]), np.array([0, 3.0])),
        ),
        ("MIS(h, lo, up)", mis(h, lo, up)),
        ("MIS(h, lo2, up)", mis(h, lo2, up)),
        ("MIS(h, lo2, up, na.rm=FALSE)", mis(h, lo2, up, na_rm=False)),
        ("MIS(h2, lo2, up, level=80)", mis(h2, lo2, up, level=80)),
        ("sMIS(h, lo2, up, 3)", smis(h, lo2, up, 3)),
        ("rMIS(h, lo2, up, lo2 - 1, up + 1)", rmis(h, lo2, up, lo2 - 1, up + 1)),
        (
            "rMIS(holdout=h, lower=lo2, upper=up, benchmarkLower=lo2 - 1, "
            "benchmarkUpper=up + 1, level=0.9)",
            rmis(
                holdout=h,
                lower=lo2,
                upper=up,
                benchmark_lower=lo2 - 1,
                benchmark_upper=up + 1,
                level=0.9,
            ),
        ),
        ("pinball(h, f, 0.9)", pinball(h, f, 0.9)),
        ("pinball(h, f, 0.9, loss=2)", pinball(h, f, 0.9, loss=2)),
        ("pinball(h, f, 0.9, loss=3)", pinball(h, f, 0.9, loss=3)),
        ("pinball(h, f, 0.3, loss=1.5)", pinball(h, f, 0.3, loss=1.5)),
        ("pinball(h, f, 0.3, na.rm=FALSE)", pinball(h, f, 0.3, na_rm=False)),
        ("MRE(h, f)", mre(h, f)),
        ("MRE(h, f, na.rm=FALSE)", mre(h, f, na_rm=False)),
        ("hm(x)", hm(x)),
        ("hm(x, C=2)", hm(x, C=2)),
        ("ham(x)", ham(x)),
        ("ham(x, C=2)", ham(x, C=2)),
        ("asymmetry(x)", asymmetry(x)),
        ("asymmetry(x, C=2)", asymmetry(x, C=2)),
        ("extremity(x)", extremity(x)),
        ("extremity(x, C=2)", extremity(x, C=2)),
        ("extremity(xc)", extremity(xc)),
        ("extremity(xc, C=2)", extremity(xc, C=2)),
        ("extremity(5, C=2)", extremity(np.array([5.0]), C=2)),
        ("extremity(c(3, 3, 3))", extremity(np.array([3, 3, 3.0]))),
        ("cextremity(xc)", cextremity(xc)),
        ("cextremity(xc, C=2)", cextremity(xc, C=2)),
        ("cextremity(x, C=2)", cextremity(x, C=2)),
        ("measures(h2, f, a)", list(measures(h2, f, a).values())),
        (
            "measures(holdout=h2, forecast=f, actual=a)",
            list(measures(holdout=h2, forecast=f, actual=a).values()),
        ),
        (
            "measures(h2, f, a, benchmark='mean', digits=3)",
            list(measures(h2, f, a, benchmark="mean", digits=3).values()),
        ),
    ]


@pytest.mark.parametrize("r_code,py_result", CASES, ids=[c[0] for c in CASES])
def test_measure_matches_r(r_code, py_result):
    r_result = np.array(ro.r(f"with(msr, {r_code})"))
    assert_allclose(np.atleast_1d(py_result), r_result, rtol=1e-12, equal_nan=True)
