"""Tests of coefbootstrap(), R's bootstrap of the parameters of the models."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from greybox import ALM, BootstrapResult, coefbootstrap

N = 100
_rng = np.random.default_rng(42)
X1, X2 = _rng.normal(size=N), _rng.normal(size=N)
X = np.column_stack([np.ones(N), X1, X2])
Y = 10 + 2 * X1 - X2 + _rng.normal(size=N)
Y_AR = np.empty(N)
_previous = 20.0
for _t in range(N):
    _previous = Y_AR[_t] = 5 + 0.5 * X1[_t] + 0.7 * _previous + _rng.normal()


@pytest.fixture(scope="module")
def model():
    return ALM(distribution="dnorm").fit(X, Y)


def _bootstrap(model, seed=1, **kwargs):
    np.random.seed(seed)
    return coefbootstrap(model, **kwargs)


class TestCoefbootstrap:
    def test_result(self, model):
        result = _bootstrap(model, nsim=50)
        assert isinstance(result, BootstrapResult)
        assert result.coefficients.shape == (50, 3)
        assert result.vcov.shape == (3, 3)
        assert_allclose(result.vcov, np.cov(result.coefficients, rowvar=False))
        assert (result.method, result.nsim, result.size) == ("cr", 50, 75)
        assert result.replace is False and result.prob is None
        assert result.parallel is False and result.model == "ALM"
        assert result.time_elapsed >= 0
        assert str(result).startswith("Bootstrap for the ALM model with nsim=50")

    def test_seeded_reproducibility(self, model):
        first, second = _bootstrap(model, nsim=20), _bootstrap(model, nsim=20)
        assert_allclose(first.coefficients, second.coefficients)

    def test_parallel_matches_serial(self, model):
        serial = _bootstrap(model, nsim=20)
        parallel = _bootstrap(model, nsim=20, parallel=2)
        assert parallel.parallel is True
        assert_allclose(parallel.coefficients, serial.coefficients)

    def test_case_resampling_with_replacement_matches_analytic(self, model):
        """The classical bootstrap gives the covariance matrix of the parameters."""
        result = _bootstrap(model, nsim=1000, size=N, replace=True)
        assert_allclose(
            np.sqrt(np.diag(result.vcov)), np.sqrt(np.diag(model.vcov())), rtol=0.15
        )
        assert_allclose(result.coefficients.mean(axis=0), model.coefficients, atol=0.05)

    def test_prob(self, model):
        prob = np.zeros(N)
        prob[:50] = 1
        result = _bootstrap(model, nsim=5, size=40, prob=prob)
        assert result.size == 40 and result.prob is prob

    def test_dsr(self, model):
        result = _bootstrap(model, nsim=30, method="dsr", type="additive")
        assert result.method == "dsr"
        assert result.size is None and result.replace is None and result.prob is None
        assert np.all(np.isfinite(result.vcov))
        assert np.all(np.diag(result.vcov) > 0)

    @pytest.mark.parametrize(
        "kwargs",
        [{"kind": "parametric"}, {"type": "multiplicative"}, {"scale": False}],
    )
    def test_dsr_options(self, kwargs):
        model = ALM(distribution="dlnorm").fit(X, np.exp(Y / 5))
        result = _bootstrap(model, nsim=20, method="dsr", **kwargs)
        assert np.all(np.isfinite(result.coefficients))

    def test_dsr_intermittent(self):
        y = np.exp(Y / 5) * (_rng.random(N) < 0.7)
        model = ALM(distribution="dlnorm", occurrence="plogis").fit(X, y)
        for intermittent in (False, True):
            result = _bootstrap(model, nsim=20, method="dsr", intermittent=intermittent)
            assert np.all(np.isfinite(result.coefficients))

    def test_ari(self):
        model = ALM(orders=(1, 0, 0)).fit(X, Y_AR)
        result = _bootstrap(model, nsim=30)
        assert result.coefficients.shape == (30, 4)
        assert result.size == 75
        assert np.all(np.isfinite(result.vcov))
        assert np.all(np.isfinite(_bootstrap(model, nsim=20, method="dsr").vcov))

    def test_ari_differences(self):
        model = ALM(orders=(1, 1, 0)).fit(X, Y_AR)
        result = _bootstrap(model, nsim=20, size=60)
        assert result.coefficients.shape == (20, 4)

    def test_occurrence(self):
        y = np.exp(Y / 5) * (_rng.random(N) < 0.7)
        model = ALM(distribution="dlnorm", occurrence="plogis").fit(X, y)
        result = _bootstrap(model, nsim=20)
        assert result.coefficients.shape == (20, 3)

    def test_no_intercept(self):
        model = ALM().fit(X[:, 1:], Y)
        for method in ("cr", "dsr"):
            result = _bootstrap(model, nsim=20, method=method, type="additive")
            assert result.coefficients.shape == (20, 2)

    def test_dropped_regressor_is_zero(self):
        """A regressor without variability in a sample is dropped: its parameter is
        zero, as R."""
        dummy = np.zeros(N)
        dummy[0] = 1
        model = ALM().fit(np.column_stack([X, dummy]), Y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _bootstrap(model, nsim=20, size=50)
        assert np.any(result.coefficients[:, 3] == 0)
        assert np.all(result.coefficients[:, :3] != 0)

    def test_invalid_method(self, model):
        with pytest.raises(ValueError):
            coefbootstrap(model, nsim=5, method="bootstrap")


class TestBootstrapMethods:
    def test_vcov(self, model):
        expected = _bootstrap(model, nsim=50).vcov
        np.random.seed(1)
        assert_allclose(model.vcov(bootstrap=True, nsim=50), expected)

    def test_confint(self, model):
        expected = np.quantile(
            _bootstrap(model, nsim=50).coefficients, [0.05, 0.95], axis=0
        ).T
        np.random.seed(1)
        assert_allclose(model.confint(level=0.9, bootstrap=True, nsim=50), expected)
        np.random.seed(1)
        assert_allclose(
            model.confint(parm=[1], level=0.9, bootstrap=True, nsim=50), expected[[1]]
        )

    def test_summary(self, model):
        bootstrapped = _bootstrap(model, nsim=50)
        np.random.seed(1)
        summary = model.summary(bootstrap=True, nsim=50)
        assert_allclose(summary.se, np.sqrt(np.diag(bootstrapped.vcov)))
        lower, upper = np.quantile(bootstrapped.coefficients, [0.025, 0.975], axis=0)
        assert_allclose(summary.lower_ci, lower)
        assert_allclose(summary.upper_ci, upper)


@pytest.mark.skipif_rpy2
class TestCoefbootstrapR:
    """R and Python draw different random numbers, so the bootstrapped parameters
    are compared statistically: their means and standard deviations."""

    @pytest.mark.parametrize(
        "distribution, method, orders, nsim",
        [
            ("dnorm", "cr", (0, 0, 0), 1000),
            ("dlaplace", "cr", (0, 0, 0), 1000),
            ("dnorm", "cr", (1, 0, 0), 400),
            ("dnorm", "dsr", (0, 0, 0), 1000),
        ],
    )
    def test_matches_r(self, distribution, method, orders, nsim):
        import rpy2.robjects as ro

        y = Y_AR if orders[0] else Y
        ro.r("suppressMessages(library(greybox))")
        for name, values in (("y", y), ("x1", X1), ("x2", X2)):
            ro.globalenv[name] = ro.FloatVector(values)
        expected = ro.r(
            f"set.seed(1); coefbootstrap(alm(y~x1+x2, data.frame(y=y,x1=x1,x2=x2), "
            f'distribution="{distribution}", orders=c({orders[0]},0,0)), '
            f'nsim={nsim}, method="{method}", type="additive")'
        )
        model = ALM(distribution=distribution, orders=orders).fit(X, y)
        result = _bootstrap(model, nsim=nsim, method=method, type="additive")
        se = np.sqrt(np.diag(np.array(expected.rx2("vcov"))))
        assert_allclose(np.sqrt(np.diag(result.vcov)), se, rtol=0.2)
        difference = result.coefficients.mean(axis=0) - np.array(
            expected.rx2("coefficients")
        ).mean(axis=0)
        assert np.all(np.abs(difference) < 0.3 * se), difference / se
