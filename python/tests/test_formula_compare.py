"""Compare formula() with R's model.frame() and model.matrix()."""

import os
import subprocess

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

import rpy2.robjects as ro
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter

from greybox import ALM, CALM, formula
from greybox.predict import predict

# The factors of R, the ordered one without ordering, as alm() does
ro.r(
    "formulaData <- mtcars; formulaData$cyl <- factor(formulaData$cyl);"
    "formulaData$gear <- factor(formulaData$gear);"
    "formulaData$carbc <- as.character(formulaData$carb);"
    "formulaData$amo <- factor(formulaData$am, levels=c(1,0), ordered=TRUE);"
    "formulaData$amo <- factor(formulaData$amo, ordered=FALSE);"
    "formulaData$aml <- formulaData$am == 1"
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
    "mpg ~ wt^2",
    "mpg ~ wt + wt^3 + I(wt^2)",
    "mpg ~ (wt + hp)^2",
    "mpg ~ (wt + hp + qsec)^3",
    "mpg ~ (wt + hp + qsec)^2.5",
    "mpg ~ (wt + cyl):hp",
    "mpg ~ wt * (hp + cyl)",
    "mpg ~ (wt + hp + qsec)^2 - wt:hp",
    "mpg ~ (wt + hp - wt)^2",
    "mpg ~ (wt + hp)^2:cyl",
    "mpg ~ (wt + hp) * (qsec + gear)",
    "mpg ~ wt * hp - (wt + hp)",
    "mpg ~ (wt + cyl)^2 - 1",
    "mpg ~ (wt + 0) + cyl",
    "mpg ~ -wt:hp + qsec",
    "mpg ~ wt / hp",
    "mpg ~ (wt + hp) / cyl",
    "mpg ~ wt %in% cyl",
    "mpg ~ hp + wt %in% hp",
    "mpg ~ wt %in% (hp + qsec)",
    "mpg ~ wt * hp %in% cyl",
    "mpg ~ wt + offset(hp)",
    "mpg ~ offset(log(hp)) + wt * cyl + offset(hp):wt",
    "mpg ~ log(wt, 2) + log(hp, base=10) + log(qsec, 3) + log(drat, exp(1))",
    "log(mpg, 2) ~ wt",
    "mpg ~ aml",
    "mpg ~ 0 + aml",
    "mpg ~ aml * wt + aml:cyl",
    "mpg ~ (. - cyl - gear - carbc - amo - aml)^2",
]


@pytest.fixture(scope="module")
def data(mtcars):
    frame = mtcars.copy()
    frame["cyl"] = pd.Categorical(frame["cyl"])
    frame["gear"] = frame["gear"].astype(int).astype(str)
    frame["carbc"] = frame["carb"].astype(int).astype(str).astype("string")
    frame["amo"] = pd.Categorical(frame["am"], categories=[1, 0], ordered=True)
    frame["aml"] = frame["am"] == 1
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

    LEVELS = [
        "b",
        "A",
        "a",
        "B",
        "_x",
        "x y",
        "x-y",
        "xy",
        "10",
        "1a",
        "9",
        "e",
        "\u00e9",
        "ea",
        "\u00c9a",
        "ae",
        "\u00e6",
        "\u00c6",
        "ss",
        "\u00df",
        "o",
        "\u00f6",
        "\u00f8",
        "a b",
        "a_b",
        "a.b",
        "a1",
        "A1",
        "High",
        "high",
    ]

    def test_level_order(self):
        """The levels sorted as R in the C collation (that of R here), by codepoint."""
        assert _r('Sys.getlocale("LC_COLLATE")')[0] == "C"
        ro.r('caseData <- data.frame(y=1:8, g=rep(c("b", "A", "a", "B"), 2))')
        data = {"y": np.arange(1.0, 9), "g": ["b", "A", "a", "B"] * 2}
        assert list(formula("y ~ g", data)[1].columns) == list(
            _r("colnames(model.matrix(y ~ g, caseData))")
        )

    @pytest.mark.parametrize("locale", ["C", "C.UTF-8"])
    def test_level_order_locale(self, locale, tmp_path, monkeypatch):
        """The levels of a character column in the order of R's factor() in the
        locale: the code points in C, the Unicode collation (ICU's) in C.UTF-8. R in
        the C locale cannot represent the other characters (it sorts their escapes,
        "<U+00C6>"), so that case takes the ASCII levels only."""
        levels = [level for level in self.LEVELS if locale != "C" or level.isascii()]
        path = tmp_path / "levels.txt"
        path.write_text("\n".join(levels), encoding="utf-8")
        result = subprocess.run(
            [
                "Rscript",
                "-e",
                f'cat(levels(factor(readLines("{path}", '
                'encoding="UTF-8"))), sep="\\n")',
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            env={**os.environ, "LC_ALL": locale},
        )
        monkeypatch.setenv("LC_ALL", locale)
        data = {"y": np.arange(1.0, len(levels) + 1), "g": levels}
        columns = list(formula("y ~ 0 + g", data)[1].columns)
        assert columns == [
            "g" + level for level in result.stdout.rstrip("\n").split("\n")
        ]

    @pytest.mark.parametrize(
        "rows", [[2, 0], [3, 3, 1], [5, 0, 31, 2]], ids=["fewer", "reordered", "all"]
    )
    def test_training_levels(self, data, rows):
        """The new data gets the levels of the training data, as R's predict.lm()."""
        formula_str = "mpg ~ wt * cyl + carbc + aml"
        X_train = formula(formula_str, data, return_type="X")
        newdata = data.iloc[rows][["wt", "cyl", "carbc", "aml"]].astype(
            {"cyl": float, "carbc": object}
        )
        newdata.attrs["_xlevels"] = X_train.attrs["_xlevels"]
        X = formula(formula_str, newdata, return_type="X")
        ro.r(f"newRows <- formulaData[c({', '.join(str(i + 1) for i in rows)}), ]")
        ro.r(f"trainTerms <- delete.response(terms({formula_str}))")
        r_matrix = (
            "model.matrix(trainTerms, model.frame(trainTerms, newRows, "
            "xlev=.getXlevels(trainTerms, model.frame(trainTerms, formulaData))))"
        )
        assert list(X.columns) == list(_r(f"colnames({r_matrix})"))
        assert_allclose(X.to_numpy(), np.asarray(_r(r_matrix)))

    def test_predict_alm(self, data):
        """predict() of new data with fewer and reordered levels, as R's predict.alm()
        of the new data with the levels of the training data."""
        formula_str = "mpg ~ wt + cyl + aml + trend"
        y, X = formula(formula_str, data)
        model = ALM(distribution="dnorm").fit(X, y, formula=formula_str)
        newdata = pd.DataFrame(
            {"wt": [2.5, 3.0], "cyl": [8.0, 6.0], "aml": [True, True]}
        )
        prediction = predict(model, newdata, interval="prediction")
        ro.r("formulaData$trend <- 1:32")
        ro.r(f"m <- alm({formula_str}, formulaData, distribution='dnorm')")
        ro.r(
            "p <- predict(m, data.frame(wt=c(2.5, 3), aml=c(TRUE, TRUE), "
            "cyl=factor(c(8, 6))), "
            "interval='prediction')"
        )
        ro.r("formulaData$trend <- NULL")
        assert_allclose(prediction.mean, _r("p$mean"), rtol=1e-6)
        for bound in ("lower", "upper"):
            assert_allclose(
                np.ravel(getattr(prediction, bound)), _r(f"p${bound}").ravel()
            )
        with pytest.raises(ValueError, match="factor cyl has new levels 5"):
            predict(model, newdata.assign(cyl=[5.0, 6.0]))
        assert (
            "factor cyl has new level 5"
            in _r(
                "tryCatch(predict(m, data.frame(wt=2.5, aml=TRUE, cyl=factor(5))), "
                "error=function(e) conditionMessage(e))"
            )[0]
        )

    def test_predict_calm(self, data):
        """CALM's predict() of new data with fewer and reordered levels and the
        logical dummies, as R's predict() of calm."""
        frame = data[["mpg", "wt", "cyl", "aml"]]
        model = CALM(frame, bruteforce=True)
        newdata = pd.DataFrame({"wt": [2.5, 3.0], "cyl": [8, 6], "aml": [True, True]})
        ro.r("m <- calm(formulaData[, c('mpg', 'wt', 'cyl', 'aml')], bruteforce=TRUE)")
        assert model.coefficient_names == list(_r("names(coef(m))"))
        ro.r(
            "p <- predict(m, data.frame(wt=c(2.5, 3), aml=c(TRUE, TRUE), "
            "cyl=factor(c(8, 6))))"
        )
        assert_allclose(model.predict(newdata).mean, _r("p$mean"), rtol=1e-6)

    def test_trend(self):
        ro.r("trendData <- data.frame(y=c(3, 1, 4, 1, 5), x=c(2, 7, 1, 8, 2))")
        ro.r("trendData$trend <- 1:5")
        _, X = formula("y ~ x * trend", {"y": [3, 1, 4, 1, 5], "x": [2, 7, 1, 8, 2]})
        r_matrix = "model.matrix(y ~ x * trend, trendData)"
        assert list(X.columns) == list(_r(f"colnames({r_matrix})"))
        assert_allclose(X.to_numpy(), np.asarray(_r(r_matrix)))
