"""Tests for greybox.point_measures module."""

import warnings

import numpy as np
import pytest

from greybox.point_measures import (
    mae,
    rrmse,
    rame,
    mse,
    rmse,
    mpe,
    mape,
    mase,
    me,
    rmsse,
    same,
    rmae,
    smse,
    spis,
    sce,
    gmrae,
    measures,
)
from greybox.association import (
    determination,
    association,
)
from greybox.hm import (
    mre,
    hm,
    ham,
    asymmetry,
    extremity,
    cextremity,
)
from greybox.quantile_measures import (
    pinball,
    mis,
    smis,
    rmis,
)

# Simple test arrays
actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
predicted = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
# errors: -0.1, -0.2, 0.2, -0.1, -0.3
# abs errors: 0.1, 0.2, 0.2, 0.1, 0.3  -> mean = 0.18
# sq errors: 0.01, 0.04, 0.04, 0.01, 0.09 -> mean = 0.038


class TestMAE:
    def test_basic(self):
        result = mae(actual, predicted)
        expected = np.mean([0.1, 0.2, 0.2, 0.1, 0.3])
        np.testing.assert_almost_equal(result, expected)

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            mae(np.array([1, 2]), np.array([1, 2, 3]))

    def test_with_nan(self):
        a = np.array([1.0, np.nan, 3.0])
        f = np.array([1.1, 2.2, 2.8])
        result = mae(a, f)
        np.testing.assert_almost_equal(result, np.mean([0.1, 0.2]))


class TestMSE:
    def test_basic(self):
        result = mse(actual, predicted)
        expected = np.mean([0.01, 0.04, 0.04, 0.01, 0.09])
        np.testing.assert_almost_equal(result, expected)


class TestRMSE:
    def test_basic(self):
        result = rmse(actual, predicted)
        expected = np.sqrt(np.mean([0.01, 0.04, 0.04, 0.01, 0.09]))
        np.testing.assert_almost_equal(result, expected)


class TestMPE:
    def test_basic(self):
        # (actual - forecast) / actual, returned as a fraction (matches R)
        errors = [
            (1 - 1.1) / 1,
            (2 - 2.2) / 2,
            (3 - 2.8) / 3,
            (4 - 4.1) / 4,
            (5 - 5.3) / 5,
        ]
        result = mpe(actual, predicted)
        np.testing.assert_almost_equal(result, np.mean(errors))

    def test_zero_actual(self):
        # As in R, a zero actual gives an infinite error, 0/0 is dropped
        a = np.array([0.0, 0.0, 2.0])
        f = np.array([0.1, 0.0, 2.2])
        assert mpe(a, f) == -np.inf
        np.testing.assert_almost_equal(mpe(a[1:], f[1:]), (2 - 2.2) / 2)


class TestMAPE:
    def test_basic(self):
        abs_pct = [
            abs((1 - 1.1) / 1),
            abs((2 - 2.2) / 2),
            abs((3 - 2.8) / 3),
            abs((4 - 4.1) / 4),
            abs((5 - 5.3) / 5),
        ]
        result = mape(actual, predicted)
        np.testing.assert_almost_equal(result, np.mean(abs_pct))


class TestMASE:
    def test_with_scale(self):
        result = mase(actual, predicted, scale=1.0)
        expected = mae(actual, predicted)
        np.testing.assert_almost_equal(result, expected)

    def test_auto_scale(self):
        # scale = mean(|diff(actual)|) = mean([1,1,1,1]) = 1.0
        result = mase(actual, predicted)
        expected = mae(actual, predicted) / 1.0
        np.testing.assert_almost_equal(result, expected)

    def test_zero_scale(self):
        a = np.array([3.0, 3.0, 3.0])
        f = np.array([3.1, 2.9, 3.0])
        with np.errstate(divide="ignore"):
            result = mase(a, f)
        assert result == np.inf


class TestDetermination:
    def test_single_variable(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = determination(x)
        np.testing.assert_almost_equal(result[0], 0.0)

    def test_multiple_variables(self):
        np.random.seed(42)
        x1 = np.random.normal(10, 3, 100)
        x2 = np.random.normal(50, 5, 100)
        x3 = 100 + 0.5 * x1 - 0.75 * x2 + np.random.normal(0, 3, 100)
        xreg = np.column_stack([x3, x1, x2])
        result = determination(xreg)
        assert len(result) == 3
        assert result[0] > result[1]
        assert result[0] > result[2]

    def test_perfect_correlation(self):
        x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x2 = 2 * x1
        xreg = np.column_stack([x1, x2])
        result = determination(xreg)
        np.testing.assert_almost_equal(result[0], 1.0)
        np.testing.assert_almost_equal(result[1], 1.0)

    def test_bruteforce_false(self):
        np.random.seed(42)
        x1 = np.random.normal(10, 3, 50)
        x2 = np.random.normal(50, 5, 50)
        xreg = np.column_stack([x1, x2])
        result = determination(xreg, bruteforce=False)
        assert len(result) == 2


class TestAssociation:
    def test_perfect_correlation(self):
        x = np.column_stack([np.array([1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10])])
        result = association(x, method="pearson")
        np.testing.assert_almost_equal(result["value"][0, 1], 1.0)
        np.testing.assert_almost_equal(result["value"][1, 0], 1.0)

    def test_returns_keys(self):
        x = np.column_stack([np.array([1, 2, 3, 4, 5]), np.array([5, 3, 4, 2, 1])])
        result = association(x)
        assert "value" in result
        assert "p.value" in result
        assert "type" in result

    def test_diagonal_is_one(self):
        x = np.column_stack([np.array([1, 2, 3, 4, 5]), np.array([5, 3, 4, 2, 1])])
        result = association(x)
        np.testing.assert_almost_equal(result["value"][0, 0], 1.0)
        np.testing.assert_almost_equal(result["value"][1, 1], 1.0)

    def test_with_y(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = association(x, y)
        assert result["value"].shape == (1, 1)


class TestME:
    def test_basic(self):
        result = me(actual, predicted)
        expected = np.mean(actual - predicted)
        np.testing.assert_almost_equal(result, expected)


class TestMIS:
    def test_basic(self):
        act = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = mis(act, lower, upper, level=0.95)
        assert result > 0


class TestRMAE:
    def test_basic(self):
        result = rmae(actual, predicted, actual - 0.1)
        assert result > 0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            rmae(np.array([1, 2]), np.array([1, 2]), np.array([1]))


class TestRMSSE:
    def test_with_scale(self):
        result = rmsse(actual, predicted, scale=1.0)
        expected = np.sqrt(mse(actual, predicted))
        np.testing.assert_almost_equal(result, expected)


class TestSAME:
    def test_with_scale(self):
        result = same(actual, predicted, scale=1.0)
        expected = np.abs(me(actual, predicted))
        np.testing.assert_almost_equal(result, expected)


class TestSMSE:
    def test_basic(self):
        result = smse(actual, predicted, scale=1.0)
        expected = mse(actual, predicted)
        np.testing.assert_almost_equal(result, expected)


class TestSPIS:
    def test_basic(self):
        result = spis(actual, predicted, scale=1.0)
        expected = np.sum(np.cumsum(predicted - actual))
        np.testing.assert_almost_equal(result, expected)


class TestSCE:
    def test_basic(self):
        result = sce(actual, predicted, scale=1.0)
        expected = np.sum(actual - predicted)
        np.testing.assert_almost_equal(result, expected)


class TestGMRAE:
    def test_different(self):
        result = gmrae(actual, predicted, actual - 0.1)
        assert result > 0


class TestMeasures:
    def test_basic(self):
        result = measures(actual, predicted, actual[:-2])
        assert "ME" in result
        assert "MAE" in result
        assert "MASE" in result
        assert "rMAE" in result

    def test_invalid_benchmark(self):
        with pytest.raises(ValueError):
            measures(actual, predicted, actual[:-2], benchmark="drift")


class TestPinball:
    def test_quantile(self):
        result = pinball(actual, predicted, level=0.5)
        assert result >= 0

    def test_expectile(self):
        result = pinball(actual, predicted, level=0.5, loss=2)
        assert result >= 0

    def test_any_loss(self):
        result = pinball(actual, predicted, level=0.5, loss=3)
        np.testing.assert_almost_equal(
            result, 0.5 * np.sum(np.abs(actual - predicted) ** 3)
        )


class TestHalfMoment:
    def test_hm(self):
        x = np.array([1, 2, 3, 4, 5])
        result = hm(x)
        assert isinstance(result, complex)

    def test_ham(self):
        x = np.array([1, 2, 3, 4, 5])
        result = ham(x)
        assert result >= 0

    def test_asymmetry(self):
        x = np.array([1, 2, 3, 4, 5])
        result = asymmetry(x)
        assert isinstance(result, (int, float, np.floating))

    def test_extremity(self):
        x = np.array([1, 2, 3, 4, 5])
        result = extremity(x)
        assert isinstance(result, (int, float, np.floating))

    def test_cextremity(self):
        x = np.array([1, 2, 3, 4, 5])
        result = cextremity(x)
        assert isinstance(result, complex)


lo = actual - 0.5
up = actual + 0.5
bench = {"benchmark": actual - 0.1}
OLD_ACTUAL = [
    (me, {}),
    (mae, {}),
    (mse, {}),
    (rmse, {}),
    (mpe, {}),
    (mape, {}),
    (mre, {}),
    (mase, {"scale": 2.0}),
    (rmsse, {"scale": 2.0}),
    (same, {"scale": 2.0}),
    (smse, {"scale": 2.0}),
    (spis, {"scale": 2.0}),
    (sce, {"scale": 2.0}),
    (rmae, bench),
    (rrmse, bench),
    (rame, bench),
    (gmrae, bench),
]
OLD_ACTUAL_INTERVALS = [
    (mis, {}),
    (smis, {"scale": 2.0}),
    (rmis, {"benchmark_lower": lo - 1, "benchmark_upper": up + 1}),
]


class TestDeprecatedArguments:
    @pytest.mark.parametrize(
        "fun,kw", OLD_ACTUAL, ids=[f.__name__ for f, _ in OLD_ACTUAL]
    )
    def test_actual(self, fun, kw):
        expected = fun(actual, predicted, **kw)
        with pytest.warns(DeprecationWarning, match="'actual' is deprecated"):
            result = fun(actual=actual, forecast=predicted, **kw)
        np.testing.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "fun,kw",
        OLD_ACTUAL_INTERVALS,
        ids=[f.__name__ for f, _ in OLD_ACTUAL_INTERVALS],
    )
    def test_actual_intervals(self, fun, kw):
        expected = fun(actual, lo, up, **kw)
        with pytest.warns(DeprecationWarning, match="'actual' is deprecated"):
            result = fun(actual=actual, lower=lo, upper=up, **kw)
        np.testing.assert_equal(result, expected)

    @pytest.mark.parametrize("fun", [hm, ham, asymmetry, extremity, cextremity])
    def test_center(self, fun):
        x = np.array([1.0, 3.0, 2.0, 8.0, 0.0, 4.0])
        expected = fun(x, C=2.0)
        with pytest.warns(DeprecationWarning, match="'center' is deprecated"):
            result = fun(x, center=2.0)
        np.testing.assert_equal(result, expected)

    def test_new_names_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mae(holdout=actual, forecast=predicted)
            mis(holdout=actual, lower=lo, upper=up)
            hm(actual, C=2.0)

    def test_unknown_keyword_still_fails(self):
        with pytest.raises(TypeError):
            mae(actual, predicted, actuals=actual)
