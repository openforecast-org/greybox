"""R/Python parity of the scale model: ALM(scale_formula=...), sm() and implant().

The covariance matrices of the parameters estimated by the Hessian (those of the
scale model, and of the location model for most distributions) come from finite
differences with the step eps^(1/4), whose rounding errors (about 1e-6 of the
Fisher information at the same parameters in both languages) are amplified by its
inversion: they are compared with rtol=1e-3, the rest with rtol=1e-6.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

import rpy2.robjects as ro

from greybox import ALM, ScaleModel, formula, implant, point_lik, sm

R_DIR = Path(__file__).resolve().parents[2] / "R"
RTOL = 1e-6
RTOL_HESSIAN = 1e-3
DISTRIBUTIONS = [
    "dnorm",
    "dlnorm",
    "dlaplace",
    "ds",
    "dgnorm",
    "dgamma",
    "dinvgauss",
    "dlogis",
    "dalaplace",
    "dbcnorm",
]

ro.r["library"]("greybox")
# The functions of this branch, when the installed greybox does not have them yet
ro.r(
    f"""
if(!any(grepl("otherList", deparse(greybox:::scaler)))){{
    ns <- asNamespace("greybox");
    s3 <- ns[[".__S3MethodsTable__."]];
    for(f in c("scaler.R","alm.R","methods.R","pointLik.R")){{
        env <- new.env(parent=ns);
        sys.source(file.path("{R_DIR}", f), envir=env);
        for(nm in ls(env, all.names=TRUE)){{
            obj <- get(nm, env);
            if(is.function(obj)) environment(obj) <- ns;
            unlockBinding(nm, ns); assign(nm, obj, envir=ns); lockBinding(nm, ns);
            if(exists(nm, envir=s3, inherits=FALSE)) assign(nm, obj, envir=s3);
            pe <- as.environment("package:greybox");
            if(exists(nm, envir=pe, inherits=FALSE)){{
                unlockBinding(nm, pe); assign(nm, obj, envir=pe); lockBinding(nm, pe);
            }}
        }}
    }}
}}
set.seed(41);
smXreg <- cbind(rnorm(100,10,3),rnorm(100,50,5));
smXreg <- cbind(100+0.5*smXreg[,1]-0.75*smXreg[,2]+sqrt(exp(0.8+0.2*smXreg[,1]))*rnorm(100,0,1),
              smXreg,rnorm(100,300,10));
colnames(smXreg) <- c("y","x1","x2","Noise");
smXreg <- as.data.frame(smXreg);
smXregO <- smXreg;
smXregO$y[sample(100,20)] <- 0;
"""
)


def _r(code):
    return np.asarray(ro.r(code), dtype=float).ravel()


def _data(source):
    return pd.DataFrame(
        {name: _r(f"{source}${name}") for name in ("y", "x1", "x2", "Noise")}
    )


def _close_table(summary, expected):
    """The estimates, and the standard errors and bounds of finite differences: the
    bounds are as accurate as the standard errors."""
    expected = np.asarray(expected, dtype=float)
    _close(summary.coefficients, expected[:, 0])
    _close(summary.se, expected[:, 1], RTOL_HESSIAN)
    for bound, column in ((summary.lower_ci, 2), (summary.upper_ci, 3)):
        assert_allclose(
            bound,
            expected[:, column],
            rtol=0,
            atol=2 * RTOL_HESSIAN * np.max(summary.se),
        )


def _close(actual, expected, rtol=RTOL):
    assert_allclose(
        np.asarray(actual, dtype=float).ravel(),
        np.asarray(expected, dtype=float).ravel(),
        rtol=rtol,
        atol=1e-12,
    )


@pytest.fixture(scope="module", params=["none", "plogis"])
def occurrence(request):
    return request.param


def _fit(distribution, occurrence):
    """The models of both languages on the first 90 observations."""
    source = "smXreg" if occurrence == "none" else "smXregO"
    data = _data(source)
    ro.r(f"smIns <- {source}[1:90,]; smOut <- {source}[91:100,]")
    ro.r(
        f'm <- suppressWarnings(alm(y~., smIns, distribution="{distribution}", '
        f'occurrence="{occurrence}", scale=~x1+x2))'
    )
    y, X = formula("y ~ x1 + x2 + Noise", data.iloc[:90])
    model = ALM(
        distribution=distribution, occurrence=occurrence, scale_formula="~ x1 + x2"
    ).fit(X, y)
    new_data = formula("y ~ x1 + x2 + Noise", data.iloc[90:], return_type="X")
    return model, new_data


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
class TestScaleModelVsR:
    def test_estimates(self, distribution, occurrence):
        model, _ = _fit(distribution, occurrence)
        scale = model.scale_model
        _close(model.coefficients, _r("coef(m)"))
        _close(scale.coefficients, _r("coef(m$scale)"))
        _close(model.loglik, _r("logLik(m)"))
        _close(
            [model.aic, model.aicc, model.bic, model.bicc],
            _r("c(AIC(m), AICc(m), BIC(m), BICc(m))"),
        )
        assert model.nparam == _r("nparam(m)")[0]
        assert model.df_residual_ == _r("m$df.residual")[0]
        _close(model.sigma, _r("sigma(m)"))
        _close(model.extract_scale(), _r("extractScale(m)"))
        _close(model.extract_sigma(), _r("extractSigma(m)"))
        _close(point_lik(model), _r("pointLik(m)"))
        _close(np.nan_to_num(model.rstandard()), np.nan_to_num(_r("rstandard(m)")))
        # The scale model itself
        _close(scale.loglik, _r("logLik(m$scale)"))
        assert scale.nparam == _r("nparam(m$scale)")[0]
        _close(scale.fitted, _r("fitted(m$scale)"))
        _close(scale.residuals, _r("residuals(m$scale)"))
        _close(scale.actuals, _r("actuals(m$scale)"))
        _close([scale.aic, scale.bic], _r("c(AIC(m$scale), BIC(m$scale))"))

    def test_covariances_and_summaries(self, distribution, occurrence):
        model, _ = _fit(distribution, occurrence)
        rtol = (
            RTOL
            if distribution in ("dnorm", "dlnorm", "dbcnorm") and occurrence == "none"
            else RTOL_HESSIAN
        )
        _close(model.vcov(), _r("vcov(m)"), rtol)
        _close(model.scale_model.vcov(), _r("vcov(m$scale)"), RTOL_HESSIAN)
        # The Fisher information of the scale model at the same parameters
        ro.r("B <- coef(m$scale)")
        information = sm(model, "~ x1 + x2", parameters=_r("B"), FI=True)._FI  # type: ignore[union-attr]
        _close(
            information,
            _r("sm(m, ~x1+x2, parameters=B, FI=TRUE)$FI"),
            1e-5,
        )
        summary = model.summary()
        r_summary = np.array(ro.r("summary(m)$coefficients"))
        _close_table(summary, r_summary)
        assert list(summary.scale_parameters) == [False] * 4 + [True] * 3
        _close(model.confint(), np.column_stack([summary.lower_ci, summary.upper_ci]))
        _close_table(model.scale_model.summary(), ro.r("summary(m$scale)$coefficients"))

    @pytest.mark.parametrize("interval", ["none", "confidence", "prediction"])
    @pytest.mark.parametrize("new", [False, True])
    def test_predict(self, distribution, occurrence, interval, new):
        model, new_data = _fit(distribution, occurrence)
        prediction = model.predict(new_data if new else None, interval=interval)
        ro.r(f'p <- predict(m, {"smOut" if new else "NULL"}, interval="{interval}")')
        _close(prediction.mean, _r("p$mean"))
        if interval == "none":
            return
        # The confidence intervals take the covariances of the location model
        rtol = (
            RTOL
            if interval == "prediction"
            or (distribution in ("dnorm", "dlnorm", "dbcnorm") and occurrence == "none")
            else RTOL_HESSIAN
        )
        _close(prediction.lower, _r("p$lower"), rtol)
        _close(prediction.upper, _r("p$upper"), rtol)
        _close(
            np.broadcast_to(prediction.scale, prediction.mean.shape),
            np.broadcast_to(_r("p$scale"), prediction.mean.shape),
            rtol,
        )


class TestSmImplantVsR:
    def test_sm_and_implant(self, occurrence):
        source = "smXreg" if occurrence == "none" else "smXregO"
        data = _data(source)
        y, X = formula("y ~ x1 + x2 + Noise", data)
        location = ALM(distribution="dgamma" if occurrence == "none" else "dnorm")
        location.occurrence = occurrence
        location.fit(X, y)
        ro.r(
            f'm0 <- alm(y~., {source}, distribution="{location.distribution}", '
            f'occurrence="{occurrence}"); s <- sm(m0, ~x1+x2); mi <- implant(m0, s); '
            f'm <- alm(y~., {source}, distribution="{location.distribution}", '
            f'occurrence="{occurrence}", scale=~x1+x2)'
        )
        scale = sm(location, "~ x1 + x2")
        assert isinstance(scale, ScaleModel)
        _close(scale.coefficients, _r("coef(s)"))
        implanted = implant(location, scale)
        assert location.scale_model is None
        for code in ("mi", "m"):
            _close(implanted.loglik, _r(f"logLik({code})"))
            assert implanted.nparam == _r(f"nparam({code})")[0]
            _close(implanted.aicc, _r(f"AICc({code})"))
        # The default regressors of the scale are those of the location model
        _close(sm(location).coefficients, _r("coef(sm(m0))"))

    def test_scale_regressors_outside_location(self):
        data = _data("smXreg")
        ro.r("m <- alm(y~x1+Noise, smXreg[1:90,], scale=~x2)")
        y, X = formula("y ~ x1 + Noise", data.iloc[:90])
        scale_X = formula("~ x2", data.iloc[:90], return_type="X")
        model = ALM(scale_formula=scale_X).fit(X, y)
        _close(model.scale_model.coefficients, _r("coef(m$scale)"))
        # The new data have the regressors of both models
        new_data = formula("y ~ x1 + x2 + Noise", data.iloc[90:], return_type="X")
        prediction = model.predict(new_data, interval="prediction")
        ro.r('p <- predict(m, smXreg[91:100,], interval="prediction")')
        _close(prediction.lower, _r("p$lower"))
        _close(prediction.upper, _r("p$upper"))
        # The scale model predicts its scale and the interval of it
        prediction = model.scale_model.predict(data.iloc[90:], interval="confidence")
        ro.r('p <- predict(m$scale, smXreg[91:100,], interval="confidence")')
        _close(prediction.mean, _r("p$mean"))
        _close(prediction.lower, _r("p$lower"), RTOL_HESSIAN)
        _close(prediction.upper, _r("p$upper"), RTOL_HESSIAN)
