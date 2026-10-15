"""Compare formula() with R's model.frame() and model.matrix()."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

import rpy2.robjects as ro
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter

from greybox import formula

# The factors of R, the ordered one without ordering, as alm() does
ro.r(
    "formulaData <- mtcars; formulaData$cyl <- factor(formulaData$cyl);"
    "formulaData$gear <- factor(formulaData$gear);"
    "formulaData$carbc <- as.character(formulaData$carb);"
    "formulaData$amo <- factor(formulaData$am, levels=c(1,0), ordered=TRUE);"
    "formulaData$amo <- factor(formulaData$amo, ordered=FALSE)"
)

FORMULAS = [
    "mpg ~ wt + hp",
    "mpg ~ .",
    "log(mpg) ~ .",
    "mpg ~ . - cyl - wt",
    "mpg ~ log(wt) + .",
    "mpg ~ 0 + wt",
    "mpg ~ wt - 1",
    "mpg ~ wt + 0",
    "mpg ~ -1 + cyl + gear",
    "mpg ~ 0 + gear * cyl",
    "mpg ~ cyl",
    "mpg ~ carbc + amo",
    "mpg ~ wt:cyl",
    "mpg ~ cyl:wt + wt",
    "mpg ~ cyl * gear",
    "mpg ~ cyl:gear",
    "mpg ~ 0 + wt + cyl:gear",
    "mpg ~ wt:hp:cyl + hp:cyl",
    "mpg ~ wt * hp",
    "mpg ~ wt * hp * qsec",
    "mpg ~ hp:wt + wt + hp",
    "mpg ~ wt * cyl + hp",
    "mpg ~ wt * hp - wt",
    "mpg ~ I(wt^2) + I(wt*hp) + I(wt - 2*hp) + I(wt/2) + I(-wt)",
    "I(mpg^2) ~ log(wt) + exp(wt) + sqrt(hp)",
    "log(mpg) ~ log(wt) * cyl",
]


@pytest.fixture(scope="module")
def data(mtcars):
    frame = mtcars.copy()
    frame["cyl"] = pd.Categorical(frame["cyl"])
    frame["gear"] = frame["gear"].astype(int).astype(str)
    frame["carbc"] = frame["carb"].astype(int).astype(str).astype("string")
    frame["amo"] = pd.Categorical(frame["am"], categories=[1, 0], ordered=True)
    return frame


def _r(expression):
    with localconverter(default_converter + numpy2ri.converter):
        return ro.r(expression)


class TestFormulaCompare:
    @pytest.mark.parametrize("formula_str", FORMULAS)
    def test_model_matrix(self, data, formula_str):
        y, X = formula(formula_str, data)
        r_matrix = f"model.matrix({formula_str}, formulaData)"
        assert list(X.columns) == list(_r(f"colnames({r_matrix})"))
        assert_allclose(X.to_numpy(), np.asarray(_r(r_matrix)), rtol=1e-12)
        r_frame = f"model.frame({formula_str}, formulaData)"
        assert y.name == _r(f"names({r_frame})")[0]
        assert_allclose(y.to_numpy(), np.asarray(_r(f"model.response({r_frame})")))

    @pytest.mark.parametrize("formula_str", FORMULAS)
    def test_terms(self, data, formula_str):
        labels = _r(f'attr(terms({formula_str}, data=formulaData), "term.labels")')
        assert formula(formula_str, data, return_type="terms") == list(labels)

    def test_trend(self):
        ro.r("trendData <- data.frame(y=c(3, 1, 4, 1, 5), x=c(2, 7, 1, 8, 2))")
        ro.r("trendData$trend <- 1:5")
        _, X = formula("y ~ x * trend", {"y": [3, 1, 4, 1, 5], "x": [2, 7, 1, 8, 2]})
        r_matrix = "model.matrix(y ~ x * trend, trendData)"
        assert list(X.columns) == list(_r(f"colnames({r_matrix})"))
        assert_allclose(X.to_numpy(), np.asarray(_r(r_matrix)))
