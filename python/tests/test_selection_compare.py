"""rpy2-based comparison tests for stepwise(), CALM() and rolling_origin().

Compares Python's ``greybox.stepwise``/``CALM``/``rolling_origin`` against R's
``stepwise``/``calm``/``ro`` on mtcars. The R functions are sourced from the
repository, so that the comparison is with the R code of this tree.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

import rpy2.robjects as ro

from greybox import ALM, CALM, rolling_origin, stepwise
from greybox.selection import CALMResult

R_DIR = Path(__file__).resolve().parents[2] / "R"
MTCARS = Path(__file__).parent / "mtcars.csv"

ro.r("suppressMessages(library(greybox))")
for name in ("stepwise", "calm", "ro"):
    ro.r(
        f"""local({{
        e <- new.env(); sys.source("{R_DIR / name}.R", envir=e)
        f <- get("{name}", envir=e); environment(f) <- asNamespace("greybox")
        utils::assignInNamespace("{name}", f, "greybox")
        assign("{name}", f, envir=globalenv())
        }})"""
    )
ro.r(f"""d0 <- read.csv("{MTCARS}"); d0 <- d0[, colnames(d0) != "X"]""")


def _r(expr):
    """Evaluate an R expression and return it as a float array."""
    return np.asarray(ro.r(expr), dtype=float)


@pytest.fixture(scope="module")
def d0():
    data = pd.read_csv(MTCARS)
    return data[[c for c in data.columns if c not in ("X", "Unnamed: 0")]]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _assert_stepwise(model, r_call, names=True):
    ro.r(f"m <- suppressWarnings({r_call})")
    assert_allclose(model.coefficients, _r("coef(m)"), rtol=1e-6)
    assert_allclose(list(model.ic_values.values()), _r("m$ICs"), rtol=1e-6)
    # R passes the names through make.names()
    if names:
        assert list(model.ic_values) == list(ro.r("names(m$ICs)"))
    assert model.nobs == int(_r("nobs(m)")[0])


class TestStepwiseCompare:
    @pytest.mark.parametrize("distribution", ["dnorm", "dlaplace", "dlnorm"])
    def test_distributions(self, d0, distribution):
        model = stepwise(d0, distribution=distribution)
        _assert_stepwise(model, f'stepwise(d0, distribution="{distribution}")')

    @pytest.mark.parametrize("method", ["kendall", "spearman"])
    def test_binary_regressors(self, d0, method):
        """The association with am and vs is the R² of a regression, as R."""
        model = stepwise(d0, method=method, ic="BIC")
        _assert_stepwise(model, f'stepwise(d0, method="{method}", ic="BIC")')

    def test_factor(self, d0):
        data = d0.assign(cyl=d0["cyl"].astype(int).astype(str))
        model = stepwise(data, method="kendall", ic="BIC")
        _assert_stepwise(
            model,
            'stepwise(transform(d0, cyl=factor(cyl)), method="kendall", ic="BIC")',
        )

    def test_missing_values(self, d0):
        data = d0.copy()
        data.loc[3, "wt"] = np.nan
        model = stepwise(data)
        _assert_stepwise(model, "stepwise(within(d0, wt[4] <- NA))")

    def test_plogis_binarised(self, d0):
        data = d0.assign(mpg=np.maximum(d0["mpg"] - 20, 0))
        model = stepwise(data, distribution="plogis")
        _assert_stepwise(
            model,
            'stepwise(transform(d0, mpg=pmax(mpg-20, 0)), distribution="plogis")',
        )

    def test_occurrence(self, d0):
        data = d0.assign(mpg=np.where(d0["am"] == 1, d0["mpg"], 0.0))
        model = stepwise(data, distribution="dlnorm", occurrence="plogis")
        _assert_stepwise(
            model,
            'stepwise(transform(d0, mpg=ifelse(am==1, mpg, 0)), distribution="dlnorm",'
            ' occurrence="plogis")',
        )
        assert isinstance(model.occurrence, ALM)
        assert_allclose(model.log_lik, _r("logLik(m)")[0], rtol=1e-6)
        assert_allclose(
            model.occurrence.coefficients, _r("coef(m$occurrence)"), rtol=1e-6
        )
        assert model.nparam == int(_r("nparam(m)")[0])

    def test_formula_and_subset(self, d0):
        model = stepwise(d0, formula="mpg ~ wt + log(hp) + qsec")
        _assert_stepwise(
            model, "stepwise(d0, formula=mpg ~ wt + log(hp) + qsec)", names=False
        )
        model = stepwise(d0, subset=np.arange(32) < 25)
        _assert_stepwise(model, "stepwise(d0, subset=1:32 < 26)")


def _assert_calm(model, r_call, vcov=True):
    ro.r(f"m <- suppressWarnings({r_call})")
    assert_allclose(model.coefficients, _r("coef(m)"), rtol=1e-6)
    if vcov:
        assert_allclose(model.vcov(), _r("vcov(m)"), rtol=1e-6, atol=1e-12)
    assert_allclose(model.importance, _r("m$importance"), rtol=1e-6, atol=1e-12)
    assert_allclose(model.IC, _r("m$IC")[0], rtol=1e-6)
    assert_allclose(model.combination, _r("m$combination"), rtol=1e-6, atol=1e-12)
    assert_allclose(model.fitted, _r("m$fitted"), rtol=1e-6)
    assert_allclose(model.scale, _r("m$scale")[0], rtol=1e-6)
    assert_allclose(
        [model.aic, model.aicc, model.bic, model.bicc],
        _r("c(AIC(m), AICc(m), BIC(m), BICc(m))"),
        rtol=1e-6,
    )


class TestCALMCompare:
    COLUMNS = ("mpg", "wt", "hp", "qsec", "am")
    R_DATA = 'd0[, c("mpg","wt","hp","qsec","am")]'

    @pytest.mark.parametrize("ic", ["AICc", "AIC", "BIC", "BICc"])
    def test_dnorm_bruteforce(self, d0, ic):
        model = CALM(d0[list(self.COLUMNS)], ic=ic, bruteforce=True)
        _assert_calm(model, f'calm({self.R_DATA}, ic="{ic}", bruteforce=TRUE)')

    @pytest.mark.parametrize("distribution", ["dlaplace", "dgnorm", "dbcnorm"])
    def test_distributions_bruteforce(self, d0, distribution):
        model = CALM(d0[list(self.COLUMNS)], bruteforce=True, distribution=distribution)
        _assert_calm(
            model,
            f'calm({self.R_DATA}, bruteforce=TRUE, distribution="{distribution}")',
        )

    def test_dgnorm_vcov(self, d0):
        model = CALM(d0[list(self.COLUMNS)], bruteforce=True, distribution="dgnorm")
        ro.r(f'm <- calm({self.R_DATA}, bruteforce=TRUE, distribution="dgnorm")')
        assert_allclose(model.vcov(), _r("vcov(m)"), rtol=1e-6)

    def test_not_bruteforce(self, d0):
        model = CALM(d0)
        _assert_calm(model, "calm(d0)")

    def test_stress(self):
        rng = np.random.default_rng(41)
        X = rng.normal(size=(60, 18))
        y = 5 + X[:, :14] @ np.full(14, 2.0) + rng.normal(size=60)
        data = pd.DataFrame(np.column_stack([y, X]))
        data.columns = ["y"] + [f"x{i + 1}" for i in range(18)]
        ro.globalenv["stress"] = ro.r["data.frame"](
            **{c: ro.FloatVector(data[c]) for c in data.columns}
        )
        model = CALM(data)
        assert len(model.combination) == 19
        _assert_calm(model, "calm(stress)")

    def test_intercept_only(self):
        rng = np.random.default_rng(0)
        data = pd.DataFrame(rng.normal(size=(30, 3)), columns=["y", "x1", "x2"])
        ro.globalenv["noise"] = ro.r["data.frame"](
            **{c: ro.FloatVector(data[c]) for c in data.columns}
        )
        model = CALM(data)
        ro.r("m <- calm(noise)")
        assert isinstance(model, ALM)
        assert list(ro.r("class(m)")) == ["alm", "greybox"]
        assert_allclose(model.coefficients, _r("coef(m)"), rtol=1e-6)

    def test_missing_values_and_factor(self, d0):
        data = d0[list(self.COLUMNS)].copy()
        data.loc[3, "wt"] = np.nan
        model = CALM(data, bruteforce=True)
        _assert_calm(
            model, f"calm(within({self.R_DATA}, wt[4] <- NA), bruteforce=TRUE)"
        )
        data = d0[["mpg", "wt", "cyl", "am"]].assign(
            cyl=d0["cyl"].astype(int).astype(str)
        )
        model = CALM(data, bruteforce=True)
        assert model.coefficient_names == ["(Intercept)", "wt", "cyl6", "cyl8", "am"]
        _assert_calm(
            model,
            'calm(transform(d0[, c("mpg","wt","cyl","am")], cyl=factor(cyl)),'
            " bruteforce=TRUE)",
        )

    def test_plogis_binarised(self, d0):
        data = d0[["mpg", "wt", "am"]].assign(mpg=np.maximum(d0["mpg"] - 20, 0))
        model = CALM(data, bruteforce=True, distribution="plogis")
        ro.r(
            'm <- calm(transform(d0[, c("mpg","wt","am")], mpg=pmax(mpg-20, 0)),'
            ' bruteforce=TRUE, distribution="plogis")'
        )
        assert_allclose(model.coefficients, _r("coef(m)"), rtol=1e-6)
        assert_allclose(model.fitted, _r("m$fitted"), rtol=1e-6)

    @pytest.mark.parametrize("distribution", ["dnorm", "dgamma", "dlnorm"])
    @pytest.mark.parametrize("interval", ["confidence", "prediction"])
    def test_predict(self, d0, distribution, interval):
        model = CALM(d0[list(self.COLUMNS)], bruteforce=True, distribution=distribution)
        ro.r(
            f'm <- calm({self.R_DATA}, bruteforce=TRUE, distribution="{distribution}")'
        )
        ro.r(f'p <- predict(m, interval="{interval}", level=c(0.9, 0.95))')
        prediction = model.predict(interval=interval, level=[0.9, 0.95])
        assert_allclose(prediction.mean, _r("p$mean"), rtol=1e-6)
        assert_allclose(prediction.lower, _r("p$lower"), rtol=1e-6)
        assert_allclose(prediction.upper, _r("p$upper"), rtol=1e-6)
        prediction = model.predict(model._X_train[:5], interval=interval, side="upper")
        ro.r(
            f'p <- predict(m, {self.R_DATA}[1:5, ], interval="{interval}",'
            ' side="upper")'
        )
        assert_allclose(prediction.upper, _r("p$upper")[:, 0], rtol=1e-6)

    def _fit_both(self, d0, distribution, bruteforce):
        if distribution == "dnbinom":
            data = d0[["carb"] + [c for c in d0.columns if c != "carb"]]
            r_data = 'd0[, c("carb", setdiff(colnames(d0), "carb"))]'
        elif distribution == "dlogitnorm":
            data, r_data = d0.assign(mpg=d0["mpg"] / 40), "transform(d0, mpg=mpg/40)"
        else:
            data, r_data = d0, "d0"
        if bruteforce:
            columns = [data.columns[0], "wt", "hp", "qsec", "am"]
            data = data[columns]
            r_data += f"[, c({', '.join(repr(c) for c in columns)})]".replace("'", '"')
        model = CALM(data, bruteforce=bruteforce, distribution=distribution)
        ro.r(
            f"m <- suppressWarnings(calm({r_data}, bruteforce={str(bruteforce).upper()},"
            f' distribution="{distribution}"))'
        )
        return model

    def _assert_predict(self, model, interval):
        ro.r(f'p <- predict(m, interval="{interval}", level=c(0.9, 0.95))')
        prediction = model.predict(interval=interval, level=[0.9, 0.95])
        assert_allclose(prediction.mean, _r("p$mean"), rtol=1e-6)
        if interval != "none":
            assert_allclose(prediction.lower, _r("p$lower"), rtol=1e-6)
            assert_allclose(prediction.upper, _r("p$upper"), rtol=1e-6)

    @pytest.mark.parametrize("bruteforce", [True, False])
    @pytest.mark.parametrize(
        "distribution",
        [
            "dlaplace",
            "ds",
            "dgnorm",
            "dlogis",
            "dt",
            "dalaplace",
            "dllaplace",
            "dls",
            "dlgnorm",
            "dbcnorm",
            "dinvgauss",
            "dexp",
            "dfnorm",
            "drectnorm",
            "dnbinom",
            "dlogitnorm",
        ],
    )
    def test_predict_distributions(self, d0, distribution, bruteforce):
        """The combined extra parameter is R's $other, and the predictions are R's."""
        model = self._fit_both(d0, distribution, bruteforce)
        if isinstance(model, CALMResult):
            if model.other_ is not None:
                assert_allclose(model.other_, _r("unlist(m$other)")[0], rtol=1e-6)
            assert_allclose(model.fitted, _r("m$fitted"), rtol=1e-6)
            assert_allclose(model.residuals, _r("m$residuals"), rtol=1e-6, atol=1e-10)
            assert_allclose(model.scale, _r("m$scale")[0], rtol=1e-6)
            assert_allclose(model.data, _r("as.matrix(m$data)"), rtol=1e-12)
            assert model.rank_ == int(_r("m$rank")[0])
        for interval in ("none", "confidence", "prediction"):
            self._assert_predict(model, interval)


class TestRollingOriginCompare:
    def test_matrix_output(self):
        y = np.array([5, 7, 6, 9, 8, 10, 12, 11, 13, 15, 14, 16.0])
        result = rolling_origin(
            y,
            h=3,
            origins=4,
            call=lambda data, h: np.column_stack(
                [np.full(h, data.mean()), np.full(h, np.median(data))]
            ),
        )
        ro.globalenv["y"] = ro.FloatVector(y)
        ro.r(
            'r <- ro(y, h=3, origins=4, call="list(m=cbind(rep(mean(data),h),'
            ' rep(median(data),h)))", value="m")'
        )
        assert result.mean.shape == (3, 4, 2)
        assert_allclose(result.mean, _r("r$m"))
        assert_allclose(result.holdout, _r("r$holdout"))
