"""R's alm() against Python's ALM: the estimates, the counts and the information
criteria, the losses, the checks of the regressors, the occurrence models, and the
predictions and summaries of every distribution."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

ro = pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from greybox import ALM, formula

FORMULA = "mpg ~ wt + hp"


@pytest.fixture(scope="module")
def mtcars():
    data = pd.read_csv(Path(__file__).parent / "mtcars.csv")
    data = data.drop(columns=[c for c in data.columns if c.startswith("Unnamed")])
    rng = np.random.default_rng(5)
    data["gear3"] = data["gear"] - 3
    data["mpgr"] = data["mpg"] / 40
    data["mpgb"] = data["mpg"] / data["mpg"].max()
    data["mpgz"] = np.maximum(data["mpg"] - 18, 0)
    data["carbz"] = np.where(data["wt"] > 3.3, 0, data["carb"])
    data["expl"] = np.cumprod(np.full(32, 1.04)) * 10 + rng.normal(size=32)
    data["noisy"] = np.cumsum(rng.normal(size=32)) + 30
    data["wt2"] = 2 * data["wt"]
    data["const"] = 1.0
    ro.r("suppressMessages(library(greybox))")
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["mt"] = data
    return data


def _r_fit(model, args):
    result = ro.r(
        f"""local({{m <- suppressWarnings(alm({model}, mt{args}));
        list(B=as.numeric(coef(m)), scale=as.numeric(m$scale)[1],
             ll=as.numeric(logLik(m)), np=nparam(m), dfr=m$df.residual, aic=AIC(m),
             aicc=AICc(m), bic=BIC(m), bicc=BICc(m), lv=m$lossValue,
             fit=as.numeric(fitted(m)), res=as.numeric(resid(m)), sigma=sigma(m))}})"""
    )
    return {k: np.array(result.rx2(k), dtype=float) for k in result.names}


def _python_fit(data, model, kwargs):
    y, X = formula(model, data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ALM(**kwargs).fit(X, y)
    missing = np.nan
    return {
        "B": fitted.coefficients,
        "scale": np.ravel(np.asarray(fitted.scale, dtype=float))[:1],
        "ll": missing if fitted.loglik is None else fitted.loglik,
        "np": fitted.nparam,
        "dfr": fitted.df_residual_,
        "aic": missing if fitted.aic is None else fitted.aic,
        "aicc": missing if fitted.aicc is None else fitted.aicc,
        "bic": missing if fitted.bic is None else fitted.bic,
        "bicc": missing if fitted.bicc is None else fitted.bicc,
        "lv": fitted.loss_value,
        "fit": fitted.fitted,
        "res": fitted.residuals,
        "sigma": fitted.sigma,
    }


def _compare(data, model, r_args, kwargs, rtol=1e-6):
    r = _r_fit(model, r_args)
    python = _python_fit(data, model, kwargs)
    for key, value in r.items():
        assert_allclose(
            np.ravel(np.asarray(python[key], dtype=float)),
            value,
            rtol=rtol,
            atol=1e-10,
            err_msg=key,
        )


CONTINUOUS = [
    "dnorm",
    "dlaplace",
    "ds",
    "dgnorm",
    "dlogis",
    "dt",
    "dalaplace",
    "dlnorm",
    "dllaplace",
    "dls",
    "dlgnorm",
    "dbcnorm",
    "dinvgauss",
    "dgamma",
    "dexp",
    "dfnorm",
    "drectnorm",
]


@pytest.mark.parametrize("distribution", CONTINUOUS)
def test_distributions(mtcars, distribution):
    """The estimates, counts, ICs, fitted values and errors of every distribution."""
    _compare(
        mtcars,
        FORMULA,
        f', distribution="{distribution}"',
        {"distribution": distribution},
    )


@pytest.mark.parametrize(
    "model, distribution",
    [
        ("carb ~ wt + hp", "dpois"),
        ("carb ~ wt + hp", "dnbinom"),
        ("carb ~ wt + hp", "dgeom"),
        ("am ~ wt + hp", "plogis"),
        ("am ~ wt + hp", "pnorm"),
        ("gear3 ~ wt + hp", "dbinom"),
        ("mpgr ~ wt + hp", "dbeta"),
        ("mpgb ~ wt + hp", "dlogitnorm"),
        ("mpgz ~ wt + hp", "plogis"),
    ],
)
def test_discrete_and_bounded(mtcars, model, distribution):
    """Counts, binary responses (also non-binary ones, made binary as R), dbinom with
    its size, dbeta with both shapes, and the boundary values of dlogitnorm."""
    _compare(
        mtcars,
        model,
        f', distribution="{distribution}"',
        {"distribution": distribution},
    )


@pytest.mark.parametrize(
    "loss, distribution",
    [
        ("MSE", "dnorm"),
        ("MAE", "dlaplace"),
        ("HAM", "ds"),
        ("MSE", "dlnorm"),
        ("MSE", "dgamma"),
        ("MAE", "dllaplace"),
        ("HAM", "dls"),
        ("MSE", "dalaplace"),
        ("ROLE", "dnorm"),
        ("MSE", "dfnorm"),
    ],
)
def test_losses(mtcars, loss, distribution):
    """The likelihood at the estimates of the losses that have one, the counts without
    the scale, and the extra parameter fixed at 0.5 by the other losses, as R."""
    _compare(
        mtcars,
        FORMULA,
        f', distribution="{distribution}", loss="{loss}"',
        {"distribution": distribution, "loss": loss},
    )


@pytest.mark.parametrize(
    "r_args, kwargs",
    [
        (', loss="LASSO", lambda=0.1', {"loss": "LASSO", "lambda_l1": 0.1}),
        (', loss="RIDGE", lambda=0.1', {"loss": "RIDGE", "lambda_l2": 0.1}),
        (
            ', loss="LASSO", lambda=0.05, distribution="dlnorm"',
            {"loss": "LASSO", "lambda_l1": 0.05, "distribution": "dlnorm"},
        ),
        (', loss="LASSO"', {"loss": "LASSO"}),
    ],
)
def test_lasso_ridge(mtcars, r_args, kwargs):
    """LASSO on the standardised regressors from the RIDGE start, RIDGE in its closed
    form, both in the space of the model, as R."""
    _compare(mtcars, FORMULA, r_args, kwargs)


@pytest.mark.parametrize(
    "model, r_args, kwargs",
    [
        ("expl ~ wt", ", orders=c(1,0,0)", {"orders": (1, 0, 0)}),
        (
            "expl ~ wt",
            ', distribution="dlaplace", orders=c(1,0,0)',
            {"distribution": "dlaplace", "orders": (1, 0, 0)},
        ),
        ("noisy ~ wt", ", orders=c(2,1,0)", {"orders": (2, 1, 0)}),
        ("noisy ~ wt", ", orders=c(2,2,0)", {"orders": (2, 2, 0)}),
        (
            "am ~ wt",
            ', distribution="plogis", orders=c(1,0,0)',
            {"distribution": "plogis", "orders": (1, 0, 0)},
        ),
        (
            "am ~ wt",
            ', distribution="pnorm", orders=c(1,1,0)',
            {"distribution": "pnorm", "orders": (1, 1, 0)},
        ),
    ],
)
def test_dynamic(mtcars, model, r_args, kwargs):
    """The penalty of a non-stationary AR polynomial, ARI(p>1, d>0), and the AR
    occurrence models with the lags of smooth's occurrence model, as R."""
    pytest.importorskip("smooth")
    _compare(mtcars, model, r_args, kwargs)


@pytest.mark.parametrize("model", ["mpg ~ wt + hp + wt2", "mpg ~ wt + hp + const"])
def test_regressor_checks(mtcars, model):
    """The perfectly correlated and constant regressors are dropped, as R."""
    _compare(mtcars, model, "", {})


@pytest.mark.parametrize(
    "model, distribution, occurrence",
    [
        ("mpgz ~ wt + hp", "dnorm", "plogis"),
        ("mpgz ~ wt + hp", "dlnorm", "pnorm"),
        ("mpgz ~ wt + hp", "dgamma", "plogis"),
        ("mpgz ~ wt + hp", "dbcnorm", "plogis"),
        ("carbz ~ wt + qsec", "dnbinom", "plogis"),
        ("carbz ~ wt + qsec", "dbinom", "plogis"),
    ],
)
def test_occurrence_string(mtcars, model, distribution, occurrence):
    """occurrence="plogis" / "pnorm" estimates the occurrence model, as R."""
    _compare(
        mtcars,
        model,
        f', distribution="{distribution}", occurrence="{occurrence}"',
        {"distribution": distribution, "occurrence": occurrence},
    )


PREDICT_CASES = [(FORMULA, d) for d in CONTINUOUS] + [
    ("carb ~ wt + hp", "dpois"),
    ("carb ~ wt + hp", "dnbinom"),
    ("carb ~ wt + hp", "dgeom"),
    ("am ~ wt + hp", "plogis"),
    ("am ~ wt + hp", "pnorm"),
    ("gear3 ~ wt + hp", "dbinom"),
    ("mpgr ~ wt + hp", "dbeta"),
    ("mpgr ~ wt + hp", "dlogitnorm"),
]


@pytest.mark.parametrize("model, distribution", PREDICT_CASES)
@pytest.mark.parametrize("interval", ["none", "confidence", "prediction"])
def test_predict(mtcars, model, distribution, interval):
    """The mean, bounds, variances, location and scale of predict(), as R. The
    confidence variances go through the numerical Hessian of the likelihood, which
    turns its last bits into differences of about 1e-6."""
    r = ro.r(
        f"""local({{m <- suppressWarnings(alm({model}, mt[1:28,],
                                              distribution="{distribution}"));
        p <- suppressWarnings(predict(m, mt[29:32,], interval="{interval}",
                                      level=c(0.8, 0.95)));
        list(mean=as.numeric(p$mean), lower=as.numeric(p$lower),
             upper=as.numeric(p$upper), var=as.numeric(p$variances),
             location=as.numeric(p$location), scale=as.numeric(p$scale))}})"""
    )
    r = {k: np.array(r.rx2(k), dtype=float) for k in r.names}
    y, X = formula(model, mtcars.iloc[:28])
    _, X_new = formula(model, mtcars.iloc[28:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ALM(distribution=distribution).fit(X, y)
        prediction = fitted.predict(X_new, interval=interval, level=[0.8, 0.95])
    assert_allclose(prediction.mean, r["mean"], rtol=1e-6)
    if distribution != "dbeta":
        assert_allclose(prediction.location, r["location"], rtol=1e-6)
    if interval == "none":
        return
    # The intervals of plogis / pnorm take the covariances of the parameters only
    numerical = interval == "confidence" or distribution in ("plogis", "pnorm")
    rtol = 1e-4 if numerical else 1e-6
    assert_allclose(np.ravel(prediction.lower, order="F"), r["lower"], rtol=rtol)
    assert_allclose(np.ravel(prediction.upper, order="F"), r["upper"], rtol=rtol)
    assert_allclose(prediction.variances, r["var"], rtol=rtol)
    if distribution not in ("dbeta", "dexp", "dgeom"):
        assert_allclose(np.ravel(prediction.scale), r["scale"], rtol=rtol)


@pytest.mark.parametrize(
    "model, distribution",
    [
        (FORMULA, "dnorm"),
        (FORMULA, "dlaplace"),
        (FORMULA, "dgamma"),
        ("mpgr ~ wt + hp", "dbeta"),
        ("carb ~ wt + hp", "dpois"),
        ("am ~ wt + hp", "plogis"),
        (FORMULA, "dlnorm"),
    ],
)
def test_summary(mtcars, model, distribution):
    """The standard errors, bounds at the level, significance and coefficients of
    determination of summary(), as R."""
    r = ro.r(
        f"""local({{m <- suppressWarnings(alm({model}, mt,
                                              distribution="{distribution}"));
        s <- summary(m, level=0.9);
        list(se=as.numeric(s$coefficients[,2]), lower=as.numeric(s$coefficients[,3]),
             r2=s$r.squared, ar2=s$adj.r.squared, sig=as.numeric(s$significance))}})"""
    )
    r = {k: np.array(r.rx2(k), dtype=float) for k in r.names}
    y, X = formula(model, mtcars)
    summary = ALM(distribution=distribution).fit(X, y).summary(level=0.9)
    assert_allclose(summary.se, r["se"], rtol=1e-4)
    assert_allclose(summary.lower_ci, r["lower"], rtol=1e-4, atol=1e-8)
    assert_allclose(summary.r_squared, r["r2"], rtol=1e-8)
    assert_allclose(summary.adj_r_squared, r["ar2"], rtol=1e-8)
    assert_allclose(summary.significance * 1.0, r["sig"])
    assert "Lower 5%" in str(summary)


def test_data(mtcars):
    """The data of the model, its parameters and in-sample predictions, as R."""
    r = ro.r(
        """local({m <- alm(mpg ~ wt + hp, mt, distribution="dlaplace");
        list(data=as.numeric(as.matrix(m$data)), df=m$df, B=m$B,
             fitted=as.numeric(predict(m)$mean))})"""
    )
    y, X = formula(FORMULA, mtcars)
    fitted = ALM(distribution="dlaplace").fit(X, y)
    assert_allclose(np.ravel(fitted.data, order="F"), np.array(r.rx2("data")))
    assert fitted.df_ == int(r.rx2("df")[0]) == fitted.rank_
    assert_allclose(fitted.B_, np.array(r.rx2("B")), rtol=1e-6)
    assert_allclose(fitted.predict().mean, np.array(r.rx2("fitted")), rtol=1e-6)
