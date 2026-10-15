"""Tests for diagnostic functions.

These tests check the outlier_dummy function works correctly
and produces results consistent with R's outlierdummy.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from greybox.alm import ALM
from greybox.diagnostics import outlier_dummy
from greybox.formula import formula


class TestOutlierDummy:
    """Test outlier_dummy function."""

    def test_smoke_dnorm(self):
        """Smoke test: outlier_dummy runs without error."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 + 3 * x + np.random.randn(n) * 0.5
        # Add an outlier
        y[0] = 100.0
        data = {"y": y, "x": x}
        y_vec, X = formula("y ~ x", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y_vec)
        result = outlier_dummy(model)

        assert hasattr(result, "outliers")
        assert hasattr(result, "statistic")
        assert hasattr(result, "id")
        assert hasattr(result, "level")
        assert hasattr(result, "type")
        assert hasattr(result, "errors")
        assert result.level == 0.999

    def test_detects_outlier(self):
        """Test that a clear outlier is detected."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 + 3 * x + np.random.randn(n) * 0.5
        # Add a very extreme outlier
        y[0] = 200.0
        data = {"y": y, "x": x}
        y_vec, X = formula("y ~ x", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y_vec)
        result = outlier_dummy(model)

        # The outlier at index 0 should be detected
        assert 0 in result.id

    def test_no_outliers(self):
        """Test with data that has no outliers."""
        np.random.seed(42)
        n = 30
        x = np.linspace(0, 1, n)
        y = 2 + 3 * x  # Perfect linear relationship, no noise
        data = {"y": y, "x": x}
        y_vec, X = formula("y ~ x", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y_vec)
        result = outlier_dummy(model)

        assert result.outliers is None
        assert len(result.id) == 0

    def test_rstudent_type(self):
        """Test with rstudent residual type."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 + 3 * x + np.random.randn(n) * 0.5
        y[0] = 200.0
        data = {"y": y, "x": x}
        y_vec, X = formula("y ~ x", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y_vec)
        result = outlier_dummy(model, type="rstudent")

        assert result.type == "rstudent"
        assert 0 in result.id

    def test_custom_level(self):
        """Test with a lower confidence level (more outliers)."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 + 3 * x + np.random.randn(n) * 2.0
        # Add moderate outliers
        y[0] = 30.0
        y[1] = -20.0
        data = {"y": y, "x": x}
        y_vec, X = formula("y ~ x", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y_vec)

        result_strict = outlier_dummy(model, level=0.999)
        result_loose = outlier_dummy(model, level=0.95)

        # Lower confidence level should detect at least as many outliers
        assert len(result_loose.id) >= len(result_strict.id)

    def test_not_fitted_raises(self):
        """Test that unfitted model raises error."""
        model = ALM(distribution="dnorm", loss="likelihood")
        with pytest.raises((ValueError, AttributeError)):
            outlier_dummy(model)

    def test_outlier_matrix_shape(self):
        """Test that outlier matrix has correct shape."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 + 3 * x + np.random.randn(n) * 0.5
        y[0] = 200.0
        y[1] = -100.0
        data = {"y": y, "x": x}
        y_vec, X = formula("y ~ x", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y_vec)
        result = outlier_dummy(model)

        if result.outliers is not None:
            assert result.outliers.shape[0] == n
            assert result.outliers.shape[1] == len(result.id)

    def test_outlier_columns(self):
        """The dummies are named as R's: outlier1, outlier2, ..."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = 2 + 3 * x + np.random.randn(50) * 0.5
        y[[0, 1]] = [200.0, -100.0]
        y_vec, X = formula("y ~ x", {"y": y, "x": x})
        result = outlier_dummy(ALM().fit(X, y_vec))
        assert list(result.outliers.columns) == ["outlier1", "outlier2"]
        assert np.all(result.outliers.values[[0, 1], [0, 1]] == 1)
        assert result.outliers.values.sum() == 2

    def test_invalid_type_raises(self):
        y_vec, X = formula(
            "y ~ x", {"y": [1.0, 3, 2, 5, 4, 7], "x": [1, 2, 3, 4, 5, 6]}
        )
        with pytest.raises(ValueError):
            outlier_dummy(ALM().fit(X, y_vec), type="residuals")


@pytest.mark.skipif_rpy2
class TestOutlierDummyR:
    """The statistics and the outliers of R's outlierdummy(), with two planted
    outliers at the 5th and the 40th observations."""

    @pytest.mark.parametrize(
        "distribution", ["dnorm", "ds", "dalaplace", "dinvgauss", "dgamma", "dlogis"]
    )
    @pytest.mark.parametrize("type", ["rstandard", "rstudent"])
    def test_matches_r(self, distribution, type):
        import rpy2.robjects as ro

        rng = np.random.default_rng(7)
        n = 60
        x = rng.normal(size=n)
        y = 20 + 2 * x + rng.normal(size=n)
        y[4] += 12
        y[39] -= 7
        ro.r("suppressMessages(library(greybox))")
        ro.globalenv["y"], ro.globalenv["x"] = ro.FloatVector(y), ro.FloatVector(x)
        alpha = ", alpha=0.3" if distribution == "dalaplace" else ""
        expected = ro.r(
            f'outlierdummy(alm(y~x, data.frame(y=y, x=x), distribution="{distribution}"'
            f'{alpha}), level=0.99, type="{type}")'
        )
        y_vec, X = formula("y ~ x", {"y": y, "x": x})
        kwargs = {"alpha": 0.3} if distribution == "dalaplace" else {}
        model = ALM(distribution=distribution, **kwargs).fit(X, y_vec)
        result = outlier_dummy(model, level=0.99, type=type)
        assert_allclose(
            result.statistic, np.array(expected.rx2("statistic")), rtol=1e-6
        )
        assert_allclose(result.errors, np.array(expected.rx2("errors")), rtol=1e-6)
        assert_allclose(result.id, np.array(expected.rx2("id"), dtype=int) - 1)
