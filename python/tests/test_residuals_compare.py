"""R/Python parity of the residual diagnostics of ALM.

The standardised and studentised residuals, the hat values and the outlier
dummies of Python's ALM are compared with R's rstandard(), rstudent(),
hatvalues() and outlierdummy() for the distributions of alm(), with occurrence
and ARI models.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

import rpy2.robjects as ro

from greybox import ALM, outlier_dummy

pytestmark = pytest.mark.skipif_rpy2

N = 60
_rng = np.random.default_rng(11)
X1, X2 = _rng.normal(size=N), _rng.normal(size=N)
Y_POSITIVE = 20 + 1.5 * X1 - X2 + _rng.normal(size=N)
# Two planted outliers
Y_POSITIVE[4] += 15
Y_POSITIVE[39] -= 8
Y_EXP = _rng.exponential(np.exp(1 + 0.2 * X1))
Y_COUNT = _rng.poisson(np.exp(1 + 0.3 * X1)).astype(float)
Y_BINARY = (_rng.random(N) < 0.6).astype(float)
Y_UNIT = 1 / (1 + np.exp(-(0.3 * X1 + 0.3 * _rng.normal(size=N))))
Y_OCCURRENCE = Y_POSITIVE * (_rng.random(N) < 0.7)
Y_COUNT_OCCURRENCE = Y_COUNT * (_rng.random(N) < 0.7)
Y_AR = np.empty(N)
_previous = 20.0
for _t in range(N):
    _previous = Y_AR[_t] = 5 + 0.5 * X1[_t] + 0.7 * _previous + _rng.normal()
Y_AR_OCCURRENCE = Y_AR * (_rng.random(N) < 0.8)

_EXTRA = {"dalaplace": (", alpha=0.3", {"alpha": 0.3})}
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
    "dfnorm",
    "drectnorm",
    "dinvgauss",
    "dgamma",
]


def _case(label, y, distribution, occurrence=False, orders=(0, 0, 0), **kwargs):
    r_args, py_kwargs = _EXTRA.get(distribution, ("", {}))
    return pytest.param(
        y,
        distribution,
        r_args + kwargs.pop("r_args", ""),
        {**py_kwargs, **kwargs.pop("py_kwargs", {})},
        occurrence,
        orders,
        kwargs.pop("intercept", True),
        id=label,
        marks=kwargs.pop("marks", ()),
    )


CASES = (
    [_case(d, Y_POSITIVE, d) for d in CONTINUOUS]
    + [
        _case("dexp", Y_EXP, "dexp"),
        _case("dpois", Y_COUNT, "dpois"),
        _case("dnbinom", Y_COUNT, "dnbinom"),
        _case("dgeom", Y_COUNT, "dgeom"),
        _case("dbinom", Y_BINARY, "dbinom"),
        _case("dlogitnorm", Y_UNIT, "dlogitnorm"),
        _case("dbeta", Y_UNIT, "dbeta"),
        _case("plogis", Y_BINARY, "plogis"),
        _case("pnorm", Y_BINARY, "pnorm"),
        _case("dnorm no intercept", Y_POSITIVE, "dnorm", intercept=False),
        _case("dlaplace no intercept", Y_POSITIVE, "dlaplace", intercept=False),
        _case(
            "dnorm MSE",
            Y_POSITIVE,
            "dnorm",
            r_args=', loss="MSE"',
            py_kwargs={"loss": "MSE"},
        ),
        _case(
            "dlaplace MAE",
            Y_POSITIVE,
            "dlaplace",
            r_args=', loss="MAE"',
            py_kwargs={"loss": "MAE"},
        ),
    ]
    + [
        _case(f"{d} occurrence", Y_OCCURRENCE, d, occurrence=True)
        for d in [
            "dnorm",
            "dlnorm",
            "dlaplace",
            "ds",
            "dgnorm",
            "dalaplace",
            "dgamma",
            "dinvgauss",
            "dlogis",
            "dt",
            "dfnorm",
        ]
    ]
    + [
        _case(
            "dpois occurrence",
            Y_COUNT_OCCURRENCE,
            "dpois",
            occurrence=True,
        )
    ]
    + [
        _case(f"{d} ARI{orders}", Y_AR, d, orders=orders)
        for orders in [(1, 0, 0), (2, 0, 0), (1, 1, 0), (0, 1, 0)]
        for d in ["dnorm", "dlaplace", "dlnorm", "dgamma", "ds"]
    ]
    + [
        _case(f"{d} occurrence ARI", Y_AR_OCCURRENCE, d, True, (1, 0, 0))
        for d in ["dnorm", "dlnorm", "dlaplace"]
    ]
)


def _fit(y, distribution, r_args, py_kwargs, occurrence, orders, intercept):
    """Fit the model in R (as ``m``) and in Python."""
    ro.r("suppressMessages(library(greybox))")
    for name, values in (("y", y), ("x1", X1), ("x2", X2)):
        ro.globalenv[name] = ro.FloatVector(values)
    occurrence_r = ', occurrence="plogis"' if occurrence else ""
    ro.r(
        f"m <- suppressWarnings(alm(y~x1+x2{'' if intercept else '-1'}, "
        f'data.frame(y=y,x1=x1,x2=x2), distribution="{distribution}"{r_args}'
        f"{occurrence_r}, orders=c({orders[0]},{orders[1]},{orders[2]})))"
    )
    X = np.column_stack(([np.ones(N)] if intercept else []) + [X1, X2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ALM(
            distribution=distribution,
            occurrence="plogis" if occurrence else "none",
            orders=orders,
            **py_kwargs,
        ).fit(X, y)
    assert_allclose(model.coefficients, np.array(ro.r("coef(m)")), rtol=1e-6)
    return model


@pytest.mark.parametrize(
    "y, distribution, r_args, py_kwargs, occurrence, orders, intercept", CASES
)
def test_residuals_match_r(
    y, distribution, r_args, py_kwargs, occurrence, orders, intercept
):
    model = _fit(y, distribution, r_args, py_kwargs, occurrence, orders, intercept)
    for method in ("rstandard", "rstudent"):
        expected = np.array(ro.r(f"suppressWarnings(as.numeric({method}(m)))"))
        assert_allclose(getattr(model, method)(), expected, rtol=1e-6, atol=1e-12)
    for residuals in ("rstandard", "rstudent"):
        result = outlier_dummy(model, level=0.95, type=residuals)
        expected = ro.r(
            f'suppressWarnings(outlierdummy(m, level=0.95, type="{residuals}"))'
        )
        assert_allclose(
            result.statistic, np.array(expected.rx2("statistic")), rtol=1e-10
        )
        assert_allclose(result.id, np.array(expected.rx2("id"), dtype=int) - 1)
        if len(result.id):
            assert list(result.outliers.columns) == [
                f"outlier{i}" for i in range(1, len(result.id) + 1)
            ]


# The hat values of R: of the regressors for the Normal-based and count
# distributions, and from the covariance matrix of the parameters for the others,
# which R cannot compute for dbeta and with I(d) > 0
HAT_CASES = [
    case
    for case in CASES
    if case.values[1] != "dbeta"
    and not (
        case.values[5][1] > 0
        and case.values[1] not in ("dnorm", "dlnorm", "dt", "dbcnorm", "dlogitnorm")
    )
]


@pytest.mark.parametrize(
    "y, distribution, r_args, py_kwargs, occurrence, orders, intercept", HAT_CASES
)
def test_hatvalues_match_r(
    y, distribution, r_args, py_kwargs, occurrence, orders, intercept
):
    model = _fit(y, distribution, r_args, py_kwargs, occurrence, orders, intercept)
    expected = np.array(ro.r("suppressWarnings(as.numeric(hatvalues(m)))"))
    assert_allclose(model.hatvalues(), expected, rtol=1e-6)
