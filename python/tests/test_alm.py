"""Tests for formula parser and ALM model."""

import numpy as np
import pytest
from greybox.formula import formula, expand_formula
from greybox.alm import ALM
from scipy.special import erfc


def my_transform(x):
    """Custom transformation function for testing."""
    return x * 2


def transform_a(x):
    """Custom transformation function A for testing."""
    return x + 10


def transform_b(x):
    """Custom transformation function B for testing."""
    return x * 2


class TestFormula:
    """Tests for formula parser."""

    def test_formula_with_response(self):
        """Test formula with response variable."""
        data = {"y": [1, 2, 3], "x1": [4, 5, 6], "x2": [7, 8, 9]}
        y, X = formula("y ~ x1 + x2", data)

        assert np.array_equal(y, np.array([1, 2, 3]))
        assert X.shape == (3, 3)
        X_np = np.asarray(X)
        assert np.all(X_np[:, 0] == 1)
        assert np.array_equal(X_np[:, 1], np.array([4, 5, 6]))
        assert np.array_equal(X_np[:, 2], np.array([7, 8, 9]))

    def test_formula_without_response(self):
        """Test formula without response variable."""
        data = {"y": [1, 2, 3], "x1": [4, 5, 6], "x2": [7, 8, 9]}
        X = formula("~ x1 + x2", data, return_type="X")

        assert X.shape == (3, 3)
        X_np = np.asarray(X)
        assert np.all(X_np[:, 0] == 1)
        assert np.array_equal(X_np[:, 1], np.array([4, 5, 6]))
        assert np.array_equal(X_np[:, 2], np.array([7, 8, 9]))

    def test_formula_intercept_only(self):
        """Test formula with intercept only."""
        data = {"y": [1, 2, 3], "x1": [4, 5, 6]}
        y, X = formula("y ~ 1", data)

        assert np.array_equal(y, np.array([1, 2, 3]))
        assert X.shape == (3, 1)
        X_np = np.asarray(X)
        assert np.all(X_np == 1)

    def test_formula_terms(self):
        """Test formula terms extraction."""
        data = {"y": [1, 2, 3], "x1": [4, 5, 6]}
        terms = formula("y ~ x1", data, return_type="terms")

        assert terms == ["x1"]

    def test_formula_trend_not_in_data(self):
        """Test formula with trend not in data - should auto-generate."""
        data = {"y": [1, 2, 3, 4, 5], "x": [1, 2, 3, 4, 5]}
        y, X = formula("y ~ x + trend", data)

        assert y is not None
        assert X.shape == (5, 3)
        X_np = np.asarray(X)
        np.testing.assert_array_equal(X_np[:, 2], np.array([1, 2, 3, 4, 5]))

    def test_formula_trend_in_data(self):
        """Test formula with trend in data - should use provided values."""
        data = {
            "y": [1, 2, 3, 4, 5],
            "x": [1, 2, 3, 4, 5],
            "trend": [10, 20, 30, 40, 50],
        }
        y, X = formula("y ~ x + trend", data)

        assert X.shape == (5, 3)
        X_np = np.asarray(X)
        np.testing.assert_array_equal(X_np[:, 2], np.array([10, 20, 30, 40, 50]))

    def test_formula_log_y(self):
        """Test formula with log(y) on LHS."""
        data = {"y": [1, 2, 3, 4, 5], "x": [1, 2, 3, 4, 5]}
        y, X = formula("log(y) ~ x", data)

        expected_y = np.log([1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(y, expected_y)

    def test_formula_polynomial(self):
        """Test formula with polynomial terms."""
        data = {"y": [1, 2, 3, 4, 5], "x": [1, 2, 3, 4, 5]}
        y, X = formula("y ~ x + x^2 + x^3", data)

        assert X.shape == (5, 4)
        X_np = np.asarray(X)
        np.testing.assert_array_equal(X_np[:, 2], np.array([1, 4, 9, 16, 25]))  # x^2
        np.testing.assert_array_equal(X_np[:, 3], np.array([1, 8, 27, 64, 125]))  # x^3

    def test_formula_log_x(self):
        """Test formula with log(x) transformation."""
        data = {"y": [1, 2, 3, 4, 5], "x": [1, 2, 3, 4, 5]}
        y, X = formula("y ~ log(x)", data)

        expected_log_x = np.log([1, 2, 3, 4, 5])
        X_np = np.asarray(X)
        np.testing.assert_array_almost_equal(X_np[:, 1], expected_log_x)

    def test_formula_sqrt(self):
        """Test formula with sqrt transformation."""
        data = {"y": [1, 2, 3, 4, 5], "x": [1, 2, 3, 4, 5]}
        y, X = formula("y ~ sqrt(x)", data)

        expected_sqrt_x = np.sqrt([1, 2, 3, 4, 5])
        X_np = np.asarray(X)
        np.testing.assert_array_almost_equal(X_np[:, 1], expected_sqrt_x)

    def test_formula_I_protected(self):
        """Test formula with I() protected expression."""
        data = {"y": [1, 2, 3, 4, 5], "x": [1, 2, 3, 4, 5]}
        y, X = formula("y ~ I(x^2)", data)

        assert X.shape == (5, 2)
        X_np = np.asarray(X)
        np.testing.assert_array_equal(X_np[:, 1], np.array([1, 4, 9, 16, 25]))

    def test_expand_formula(self):
        """Test formula expansion."""
        expanded = expand_formula("y ~ x1 * x2")
        assert "x1:x2" in expanded

    def test_formula_custom_function(self):
        """Test formula with a user-defined custom function."""
        data = {"y": [1, 2, 3, 4, 5], "x": [1, 2, 3, 4, 5]}
        y, X = formula("y ~ my_transform(x)", data)

        expected = np.array([2, 4, 6, 8, 10])
        X_np = np.asarray(X)
        np.testing.assert_array_equal(X_np[:, 1], expected)

    def test_formula_imported_function(self):
        """Test formula with a function imported from scipy.special."""
        data = {"y": [1, 2, 3, 4, 5], "x": [0.5, 1.0, 1.5, 2.0, 2.5]}
        y, X = formula("y ~ erfc(x)", data)

        expected = erfc([0.5, 1.0, 1.5, 2.0, 2.5])
        X_np = np.asarray(X)
        np.testing.assert_array_almost_equal(X_np[:, 1], expected)

    def test_formula_custom_function_lhs(self):
        """Test formula with custom function on LHS (response variable)."""
        data = {"y": [1, 2, 3, 4, 5], "x": [1, 2, 3, 4, 5]}
        y, X = formula("my_transform(y) ~ x", data)

        expected_y = np.array([2, 4, 6, 8, 10])
        np.testing.assert_array_equal(y, expected_y)

    def test_formula_custom_function_I_wrapper(self):
        """Test formula with I() wrapper containing custom function."""
        data = {"y": [1, 2, 3, 4, 5], "x": [1, 2, 3, 4, 5]}
        y, X = formula("y ~ I(my_transform(x))", data)

        expected = np.array([2, 4, 6, 8, 10])
        X_np = np.asarray(X)
        np.testing.assert_array_equal(X_np[:, 1], expected)

    def test_formula_multiple_custom_functions(self):
        """Test formula with multiple custom functions."""
        data = {"y": [1, 2, 3, 4, 5], "x": [1, 2, 3, 4, 5]}
        y, X = formula("y ~ transform_a(x) + transform_b(x)", data)

        X_np = np.asarray(X)
        np.testing.assert_array_equal(X_np[:, 1], np.array([11, 12, 13, 14, 15]))
        np.testing.assert_array_equal(X_np[:, 2], np.array([2, 4, 6, 8, 10]))

    def test_formula_unknown_function_error(self):
        """Test that unknown functions raise a helpful error."""
        data = {"y": [1, 2, 3], "x": [1, 2, 3]}

        with pytest.raises(ValueError) as excinfo:
            formula("y ~ nonexistent_func(x)", data)

        assert "Unknown function 'nonexistent_func'" in str(excinfo.value)
        assert "not a callable function" not in str(excinfo.value)


class TestALM:
    """Tests for ALM model."""

    def test_alm_fit_predict(self):
        """Test ALM model fit and predict."""
        data = {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
        y, X = formula("y ~ x1 + x2", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        assert model.intercept_ is not None
        assert model.coef is not None
        assert model.scale is not None
        assert model.log_lik is not None
        assert model.aic is not None

        y_pred = model.predict(X)
        assert y_pred.mean.shape == y.shape

    def test_alm_get_params(self):
        """Test ALM get_params."""
        model = ALM(distribution="dgamma", loss="MSE")
        params = model.get_params()

        assert params["distribution"] == "dgamma"
        assert params["loss"] == "MSE"

    def test_alm_set_params(self):
        """Test ALM set_params."""
        model = ALM()
        model.set_params(distribution="dlogis", loss="MAE")

        assert model.distribution == "dlogis"
        assert model.loss == "MAE"

    def test_alm_score(self):
        """Test ALM score method."""
        data = {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
        y, X = formula("y ~ x1 + x2", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        score_mse = model.score(X, y, metric="MSE")
        score_r2 = model.score(X, y, metric="R2")

        assert score_mse >= 0
        assert score_r2 <= 1.1

    def test_alm_repr(self):
        """Test ALM repr."""
        model = ALM(distribution="dnorm", loss="likelihood")
        repr_str = repr(model)

        assert "ALM" in repr_str
        assert "dnorm" in repr_str
        assert "fitted=False" in repr_str

        # Fitted model repr
        data = {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        y, X = formula("y ~ x1", data)
        model.fit(X, y)
        repr_str = repr(model)
        assert "fitted=True" in repr_str
        assert "dnorm" in repr_str

    def test_alm_nlopt_kargs(self, mtcars):
        """Test ALM with custom nlopt_kargs."""
        data = mtcars.to_dict(orient="list")

        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm",
            loss="likelihood",
            nlopt_kargs={
                "algorithm": "NLOPT_LN_SBPLX",
                "maxeval": 100,
                "xtol_rel": 1e-4,
            },
        )
        model.fit(X, y)

        assert model.intercept_ is not None
        assert model.coef is not None
        assert model.log_lik is not None
        assert model.nlopt_result_ is not None


class TestALMProperties:
    """Tests for ALM properties."""

    @pytest.fixture
    def mtcars_data(self, mtcars):
        """Load mtcars data."""
        return mtcars.to_dict(orient="list")

    def test_alm_nobs(self, mtcars_data):
        """Test ALM nobs property."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        assert model.nobs == 32

    def test_alm_nparam(self, mtcars_data):
        """Test ALM nparam property."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        assert model.nparam == 3

    def test_alm_nparam_without_scale(self, mtcars_data):
        """Test ALM nparam for distributions without scale."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dpois", loss="likelihood")
        model.fit(X, y)

        assert model.nparam == 2

    def test_alm_sigma(self, mtcars_data):
        """Test ALM sigma property."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        n = len(y)
        k = model._n_features + 1
        expected_sigma = np.sqrt(np.sum(model.residuals_**2) / (n - k))
        np.testing.assert_allclose(model.sigma, expected_sigma)

    def test_alm_residuals(self, mtcars_data):
        """Test ALM residuals property."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        residuals = model.residuals
        assert residuals is not None
        assert len(residuals) == 32

    def test_alm_fitted(self, mtcars_data):
        """Test ALM fitted property."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        fitted = model.fitted
        assert fitted is not None
        assert len(fitted) == 32

    def test_alm_log_lik(self, mtcars_data):
        """Test ALM log_lik property."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        assert model.log_lik is not None
        np.testing.assert_allclose(model.log_lik, model.log_lik)

    def test_alm_actuals(self, mtcars_data):
        """Test ALM actuals property."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        actuals = model.actuals
        assert actuals is not None
        assert len(actuals) == 32

    def test_alm_formula(self, mtcars_data):
        """Test ALM formula property."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y, formula="mpg ~ wt")

        assert model.formula == "mpg ~ wt"

    def test_alm_formula_none(self, mtcars_data):
        """Test ALM formula property when not provided."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        assert model.formula is None


class TestALMSummary:
    """Tests for ALM summary method."""

    @pytest.fixture
    def mtcars_data(self, mtcars):
        """Load mtcars data."""
        return mtcars.to_dict(orient="list")

    def test_alm_summary(self, mtcars_data):
        """Test ALM summary method."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        summary = model.summary()

        assert summary is not None
        assert hasattr(summary, "coefficients")
        assert hasattr(summary, "se")
        assert hasattr(summary, "t_stat")
        assert hasattr(summary, "p_value")
        assert hasattr(summary, "lower_ci")
        assert hasattr(summary, "upper_ci")
        assert len(summary.coefficients) == 2

    def test_alm_summary_with_level(self, mtcars_data):
        """Test ALM summary with custom confidence level."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        summary = model.summary(level=0.99)

        assert summary is not None


class TestALMConfint:
    """Tests for ALM confint method."""

    @pytest.fixture
    def mtcars_data(self, mtcars):
        """Load mtcars data."""
        return mtcars.to_dict(orient="list")

    def test_alm_confint(self, mtcars_data):
        """Test ALM confint method."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        ci = model.confint()

        assert ci is not None
        assert ci.shape == (2, 2)

    def test_alm_confint_custom_level(self, mtcars_data):
        """Test ALM confint with custom level."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        ci = model.confint(level=0.99)

        assert ci is not None

    def test_alm_confint_specific_params(self, mtcars_data):
        """Test ALM confint for specific parameters."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        ci = model.confint(parm=1)

        assert ci is not None
        assert ci.shape == (1, 2)


class TestALMMtcars:
    """Tests comparing Python ALM with R alm() using mtcars dataset.

    Note: For normal distribution (dnorm), likelihood and MSE give the same
    results because they are equivalent for Gaussian errors.
    """

    @pytest.fixture
    def mtcars_data(self, mtcars):
        """Load mtcars data."""
        return mtcars.to_dict(orient="list")

    def test_alm_mtcars_dnorm(self, mtcars_data):
        """Test ALM with mtcars: mpg ~ wt, dnorm, likelihood.

        R results:
        - Intercept: 37.29
        - wt: -5.34
        - scale: 8.70 (sigma^2)
        - log-likelihood: -80.01
        """
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 37.29, rtol=1e-1, err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[0], -5.34, rtol=1e-1, err_msg="wt coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.scale, 8.70, rtol=2e-1, err_msg="Scale doesn't match R"
        )
        np.testing.assert_allclose(
            model.log_lik, -80.01, rtol=1e-1, err_msg="Log-likelihood doesn't match R"
        )

    def test_alm_mtcars_dnorm_two_predictors(self, mtcars_data):
        data = mtcars_data
        """Test ALM with mtcars: mpg ~ wt + hp, dnorm, likelihood.

        R results:
        - Intercept: 37.23
        - wt: -3.88
        - hp: -0.032
        - scale: 6.10 (sigma^2)
        - log-likelihood: -74.33
        """
        data = mtcars_data
        y, X = formula("mpg ~ wt + hp", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 37.23, rtol=1e-1, err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[0], -3.88, rtol=1e-1, err_msg="wt coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[1], -0.032, rtol=2e-1, err_msg="hp coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.scale, 6.10, rtol=2e-1, err_msg="Scale doesn't match R"
        )

    def test_alm_mtcars_dlaplace(self, mtcars_data):
        data = mtcars_data
        """Test ALM with mtcars: mpg ~ wt, dlaplace, likelihood.

        R results:
        - Intercept: 34.33
        - wt: -4.56
        - scale: 2.32
        """
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dlaplace", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 34.33, rtol=1e-1, err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[0], -4.56, rtol=1e-1, err_msg="wt coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.scale, 2.32, rtol=2e-1, err_msg="Scale doesn't match R"
        )

    def test_alm_mtcars_dnorm_mse(self, mtcars_data):
        data = mtcars_data
        """Test ALM with mtcars: mpg ~ wt, dnorm, MSE loss.

        For dnorm, likelihood and MSE are equivalent (Gaussian).
        R results:
        - Intercept: 37.29
        - wt: -5.34
        - scale: 8.70 (sigma^2)
        """
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="MSE", nlopt_kargs={"maxeval": 1000})
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 37.29, rtol=1e-1, err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[0], -5.34, rtol=1e-1, err_msg="wt coefficient doesn't match R"
        )

    def test_alm_mtcars_dnorm_mae(self, mtcars_data):
        data = mtcars_data
        """Test ALM with mtcars: mpg ~ wt, dnorm, MAE loss.

        R results (approximate):
        - Intercept: ~35-36
        - wt: ~-4.8 to -5.0
        """
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="MAE", nlopt_kargs={"maxeval": 1000})
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 35.6, rtol=2e-1, err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[0], -4.9, rtol=2e-1, err_msg="wt coefficient doesn't match R"
        )

    def test_alm_mtcars_dlogis(self, mtcars_data):
        data = mtcars_data
        """Test ALM with mtcars: mpg ~ wt, dlogis, likelihood.

        R results:
        - Intercept: 36.73
        - wt: -5.26
        - scale: 1.63
        """
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dlogis", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 36.73, rtol=1e-1, err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[0], -5.26, rtol=1e-1, err_msg="wt coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.scale, 1.63, rtol=2e-1, err_msg="Scale doesn't match R"
        )

    def test_alm_mtcars_dt(self, mtcars_data):
        data = mtcars_data
        """Test ALM with mtcars: mpg ~ wt, dt, likelihood.

        R results:
        - Intercept: ~37
        - wt: ~-5.3
        - scale: ~2.9
        """
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dt", loss="likelihood", nu=4, nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 37.0, rtol=1e-1, err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[0], -5.3, rtol=1e-1, err_msg="wt coefficient doesn't match R"
        )

    def test_alm_mtcars_dgnorm(self, mtcars_data):
        data = mtcars_data
        """Test ALM with mtcars: mpg ~ wt, dgnorm, likelihood.

        R results (shape=2):
        - Intercept: ~37
        - wt: ~-5.3
        - scale: ~2.9
        """
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dgnorm",
            loss="likelihood",
            shape=2.0,
            nlopt_kargs={"maxeval": 1000},
        )
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 37.0, rtol=1e-1, err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[0], -5.3, rtol=1e-1, err_msg="wt coefficient doesn't match R"
        )

    def test_alm_predict_intervals(self, mtcars_data):
        data = mtcars_data
        """Test ALM prediction intervals."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.arange(1, n + 1)])
        y = 2 + 0.5 * np.arange(1, n + 1) + np.random.normal(0, 1, n)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)

        result = model.predict(X, interval="confidence", level=0.95)

        assert result.mean is not None
        assert result.lower is not None
        assert result.upper is not None
        assert result.mean.shape == y.shape
        assert result.lower.shape == y.shape
        assert result.upper.shape == y.shape

        assert np.all(result.lower <= result.mean)
        assert np.all(result.mean <= result.upper)

        result_no_interval = model.predict(X)
        assert result_no_interval.lower is None
        assert result_no_interval.upper is None

        result_pred = model.predict(X, interval="prediction", level=0.95)
        assert result_pred.lower is not None
        assert result_pred.upper is not None
        assert np.all(result_pred.lower <= result_pred.upper)

        coverage = np.mean((y >= result_pred.lower) & (y <= result_pred.upper))
        assert 0.85 < coverage < 1.0

    def test_predict_result_level_and_variances(self):
        """Test that PredictionResult carries level, variances, side, interval."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = X @ [1, 2] + np.random.normal(0, 0.5, n)

        model = ALM(distribution="dnorm")
        model.fit(X, y)

        # With interval: level and variances populated
        result = model.predict(X[:5], interval="prediction", level=0.95)
        assert result.level == 0.95
        assert result.variances is not None
        assert result.variances.shape == (5,)
        assert np.all(result.variances > 0)
        assert result.side == "both"
        assert result.interval == "prediction"

        # Without interval: level stored, variances is None
        result_none = model.predict(X[:5])
        assert result_none.variances is None
        assert result_none.level == 0.95
        assert result_none.side == "both"
        assert result_none.interval == "none"

        # __repr__ returns DataFrame repr
        repr_str = repr(result)
        assert "mean" in repr_str

        # __len__ works
        assert len(result) == 5

        # to_dataframe works
        df = result.to_dataframe()
        assert "mean" in df.columns
        assert "lower" in df.columns
        assert "upper" in df.columns
        assert len(df) == 5

        # DataFrame-like properties
        assert result.shape == (5, 3)
        assert "mean" in result.columns

    def test_predict_result_loglik_and_distribution(self):
        """Test ALM loglik, distribution_, loss_ properties."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = X @ [1, 2] + np.random.normal(0, 0.5, n)

        model = ALM(distribution="dnorm")
        model.fit(X, y)

        assert model.loglik is not None
        assert model.loglik == model.log_lik
        assert model.distribution_ == "dnorm"
        assert model.loss_ == "likelihood"

    def test_alm_str_output(self):
        """Test that str(model) produces ADAM-style output."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = X @ [1, 2] + np.random.normal(0, 0.5, n)

        model = ALM(distribution="dnorm")
        model.fit(X, y)

        s = str(model)
        assert "Time elapsed:" in s
        assert "Model estimated: ALM(dnorm)" in s
        assert "Distribution assumed in the model: Normal" in s
        assert "Loss function type: likelihood" in s
        assert "Information criteria:" in s
        assert "AIC" in s
        assert "Sample size:" in s

    def test_alm_summary_has_time_elapsed(self):
        """Test that summary() output includes time elapsed."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = X @ [1, 2] + np.random.normal(0, 0.5, n)

        model = ALM(distribution="dnorm")
        model.fit(X, y)

        summary = model.summary()
        s = str(summary)
        assert "Time elapsed:" in s
        # Coefficient table still present
        assert "Coefficients:" in s
        assert "(Intercept)" in s

    def test_alm_predict_intervals_side_upper(self, mtcars_data):
        data = mtcars_data
        """Test ALM prediction intervals with side='upper'."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        result = model.predict(X, interval="confidence", level=0.95, side="upper")

        assert result.lower is None
        assert result.upper is not None
        assert np.all(result.mean <= result.upper)

    def test_alm_predict_intervals_side_lower(self, mtcars_data):
        data = mtcars_data
        """Test ALM prediction intervals with side='lower'."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        result = model.predict(X, interval="confidence", level=0.95, side="lower")

        assert result.lower is not None
        assert result.upper is None
        assert np.all(result.lower <= result.mean)

    def test_alm_predict_intervals_multiple_levels(self, mtcars_data):
        data = mtcars_data
        """Test ALM prediction intervals with multiple confidence levels."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        levels = [0.80, 0.90, 0.95]
        result = model.predict(X, interval="confidence", level=levels)

        assert result.lower is not None
        assert result.upper is not None
        assert result.lower.shape[1] == len(levels)
        assert result.upper.shape[1] == len(levels)

    def test_alm_predict_intervals_dlaplace(self, mtcars_data):
        data = mtcars_data
        """Test ALM prediction intervals with dlaplace distribution."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dlaplace", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        result = model.predict(X, interval="confidence", level=0.95)

        assert result.mean is not None
        assert result.lower is not None
        assert result.upper is not None
        assert np.all(result.lower <= result.mean)
        assert np.all(result.mean <= result.upper)

    def test_alm_predict_intervals_dlogis(self, mtcars_data):
        data = mtcars_data
        """Test ALM prediction intervals with dlogis distribution."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dlogis", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        result = model.predict(X, interval="prediction", level=0.95)

        assert result.mean is not None
        assert result.lower is not None
        assert result.upper is not None
        assert np.all(result.lower <= result.mean)
        assert np.all(result.mean <= result.upper)

    def test_alm_predict_mean_matches_fitted(self, mtcars_data):
        data = mtcars_data
        """Test that predict mean matches fitted values."""
        data = mtcars_data
        y, X = formula("mpg ~ wt + hp", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        result = model.predict(X)
        fitted = model.fitted_values_

        np.testing.assert_allclose(result.mean, fitted, rtol=1e-5)

    def test_alm_predict_intervals_wider_for_prediction(self, mtcars_data):
        data = mtcars_data
        """Test that prediction intervals are wider than confidence intervals."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        conf_result = model.predict(X, interval="confidence", level=0.95)
        pred_result = model.predict(X, interval="prediction", level=0.95)

        conf_width = np.mean(conf_result.upper - conf_result.lower)
        pred_width = np.mean(pred_result.upper - pred_result.lower)

        assert pred_width > conf_width

    def test_alm_vcov_compare_r(self, mtcars_data):
        data = mtcars_data
        """Test that vcov() matches R exactly."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        vcov_result = model.vcov()

        r_vcov = np.array([[3.647053, -1.0403720], [-1.040372, 0.3233731]])

        np.testing.assert_allclose(vcov_result, r_vcov, rtol=1e-5)

    def test_alm_predict_intervals_compare_r_confidence(self, mtcars_data):
        data = mtcars_data
        """Test confidence intervals match R exactly."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        result = model.predict(X, interval="confidence", level=0.95)

        r_mean = np.array(
            [
                23.282611,
                21.919770,
                24.885952,
                20.102650,
                18.900144,
                18.793255,
                18.205363,
                20.236262,
                20.450041,
                18.900144,
                18.900144,
                15.533127,
                17.350247,
                17.083024,
                9.226650,
                8.296712,
                8.718926,
                25.527289,
                28.653805,
                27.478021,
                24.111004,
                18.472586,
                18.926866,
                16.762355,
                16.735633,
                26.943574,
                25.847957,
                29.198941,
                20.343151,
                22.480940,
                18.205363,
                22.427495,
            ]
        )

        r_lower = np.array(
            [
                21.964642,
                20.731082,
                23.355101,
                18.982586,
                17.750512,
                17.638159,
                17.012529,
                19.115752,
                19.327252,
                17.750512,
                17.750512,
                14.037076,
                16.081323,
                15.785754,
                6.610581,
                5.496420,
                6.002595,
                23.898097,
                26.479618,
                25.518698,
                22.689744,
                17.298483,
                17.778531,
                15.428518,
                15.398629,
                25.078492,
                24.167406,
                26.922257,
                19.221743,
                21.245985,
                17.012529,
                21.197394,
            ]
        )

        r_upper = np.array(
            [
                24.60058,
                23.10846,
                26.41680,
                21.22271,
                20.04978,
                19.94835,
                19.39820,
                21.35677,
                21.57283,
                20.04978,
                20.04978,
                17.02918,
                18.61917,
                18.38029,
                11.84272,
                11.09700,
                11.43526,
                27.15648,
                30.82799,
                29.43734,
                25.53226,
                19.64669,
                20.07520,
                18.09619,
                18.07264,
                28.80866,
                27.52851,
                31.47562,
                21.46456,
                23.71589,
                19.39820,
                23.65760,
            ]
        )

        np.testing.assert_allclose(result.mean, r_mean, rtol=1e-5)
        np.testing.assert_allclose(result.lower, r_lower, rtol=1e-5)
        np.testing.assert_allclose(result.upper, r_upper, rtol=1e-5)

    def test_alm_predict_intervals_compare_r_prediction(self, mtcars_data):
        data = mtcars_data
        """Test prediction intervals match R exactly."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        result = model.predict(X, interval="prediction", level=0.95)

        r_lower = np.array(
            [
                16.810962,
                15.473207,
                18.367616,
                13.668387,
                12.460668,
                12.352801,
                11.758034,
                13.801921,
                14.015303,
                12.460668,
                12.460668,
                9.022876,
                10.888408,
                10.615559,
                2.371797,
                1.369459,
                1.825186,
                18.985159,
                21.955127,
                20.845967,
                17.617532,
                12.028696,
                12.487622,
                10.287456,
                10.260081,
                20.338748,
                19.292848,
                22.466297,
                13.908654,
                16.025685,
                11.758034,
                15.973167,
            ]
        )

        r_upper = np.array(
            [
                29.75426,
                28.36633,
                31.40429,
                26.53691,
                25.33962,
                25.23371,
                24.65269,
                26.67060,
                26.88478,
                25.33962,
                25.33962,
                22.04338,
                23.81209,
                23.55049,
                16.08150,
                15.22397,
                15.61267,
                32.06942,
                35.35248,
                34.11008,
                30.60448,
                24.91648,
                25.36611,
                23.23725,
                23.21119,
                33.54840,
                32.40307,
                35.93158,
                26.77765,
                28.93619,
                24.65269,
                28.88182,
            ]
        )

        np.testing.assert_allclose(result.lower, r_lower, rtol=1e-5)
        np.testing.assert_allclose(result.upper, r_upper, rtol=1e-5)

    def test_alm_predict_intervals_compare_r_multiple_levels(self, mtcars_data):
        data = mtcars_data
        """Test multiple confidence levels match R exactly."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        levels = [0.80, 0.90, 0.95]
        result = model.predict(X, interval="confidence", level=levels)

        r_lower_80 = np.array(
            [
                22.437508,
                21.157564,
                23.904346,
                19.384447,
                18.162982,
                18.052589,
                17.440499,
                19.517773,
                19.730091,
                18.162982,
                18.162982,
                14.573836,
                16.536593,
                16.251194,
                7.549185,
                6.501120,
                6.977171,
                24.482625,
                27.259682,
                26.221672,
                23.199669,
                17.719733,
                18.190536,
                15.907078,
                15.878325,
                25.747654,
                24.770361,
                27.739095,
                19.624086,
                21.689067,
                17.440499,
                21.638735,
            ]
        )

        r_lower_90 = np.array(
            [
                22.187675,
                20.932237,
                23.614158,
                19.172128,
                17.945058,
                17.833629,
                17.214385,
                19.305370,
                19.517255,
                17.945058,
                17.945058,
                14.290245,
                16.296056,
                16.005284,
                7.053284,
                5.970297,
                6.462264,
                24.173796,
                26.847543,
                25.850263,
                22.930256,
                17.497170,
                17.972857,
                15.654236,
                15.624883,
                25.394109,
                24.451796,
                27.307527,
                19.411512,
                21.454970,
                17.214385,
                21.405557,
            ]
        )

        r_lower_95 = np.array(
            [
                21.964642,
                20.731082,
                23.355101,
                18.982586,
                17.750512,
                17.638159,
                17.012529,
                19.115752,
                19.327252,
                17.750512,
                17.750512,
                14.037076,
                16.081323,
                15.785754,
                6.610581,
                5.496420,
                6.002595,
                23.898097,
                26.479618,
                25.518698,
                22.689744,
                17.298483,
                17.778531,
                15.428518,
                15.398629,
                25.078492,
                24.167406,
                26.922257,
                19.221743,
                21.245985,
                17.012529,
                21.197394,
            ]
        )

        np.testing.assert_allclose(result.lower[:, 0], r_lower_80, rtol=1e-5)
        np.testing.assert_allclose(result.lower[:, 1], r_lower_90, rtol=1e-5)
        np.testing.assert_allclose(result.lower[:, 2], r_lower_95, rtol=1e-5)

    def test_alm_predict_intervals_compare_r_side_upper(self, mtcars_data):
        data = mtcars_data
        """Test side='upper' matches R exactly."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        result = model.predict(X, interval="confidence", level=0.95, side="upper")

        r_upper = np.array(
            [
                24.37755,
                22.90730,
                26.15775,
                21.03317,
                19.85523,
                19.75288,
                19.19634,
                21.16715,
                21.38283,
                19.85523,
                19.85523,
                16.77601,
                18.40444,
                18.16076,
                11.40002,
                10.62313,
                10.97559,
                26.88078,
                30.46007,
                29.10578,
                25.29175,
                19.44800,
                19.88088,
                17.87047,
                17.84638,
                28.49304,
                27.24412,
                31.09035,
                21.27479,
                23.50691,
                19.19634,
                23.44943,
            ]
        )

        assert result.lower is None
        np.testing.assert_allclose(result.upper, r_upper, rtol=1e-5)

    def test_alm_predict_intervals_compare_r_side_lower(self, mtcars_data):
        data = mtcars_data
        """Test side='lower' matches R exactly."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000}
        )
        model.fit(X, y)

        result = model.predict(X, interval="confidence", level=0.95, side="lower")

        r_lower = np.array(
            [
                22.187675,
                20.932237,
                23.614158,
                19.172128,
                17.945058,
                17.833629,
                17.214385,
                19.305370,
                19.517255,
                17.945058,
                17.945058,
                14.290245,
                16.296056,
                16.005284,
                7.053284,
                5.970297,
                6.462264,
                24.173796,
                26.847543,
                25.850263,
                22.930256,
                17.497170,
                17.972857,
                15.654236,
                15.624883,
                25.394109,
                24.451796,
                27.307527,
                19.411512,
                21.454970,
                17.214385,
                21.405557,
            ]
        )

        np.testing.assert_allclose(result.lower, r_lower, rtol=1e-5)
        assert result.upper is None


class TestALMOrders:
    """Tests for ALM with ARIMA orders parameter.

    These tests compare Python ALM with R alm() function using ARIMA orders.

    Note: The orders parameter implements AR(p) terms. The d (differencing)
    parameter works differently between R and Python - R uses a polynomial
    formulation while Python uses a simpler lag-based approach.
    """

    @pytest.fixture
    def mtcars_data(self, mtcars):
        """Load mtcars data."""
        return mtcars.to_dict(orient="list")

    def test_alm_mtcars_orders_p1_d0(self, mtcars_data):
        """Test ALM with mtcars: mpg ~ wt, orders=(1,0,0), dnorm.

        R results (alm with orders=c(1,0,0)):
        - Intercept: 41.17
        - wt: -5.82
        - mpgLag1: -0.118
        - scale: 8.42 (sigma^2)
        - log-likelihood: -79.49
        """
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", orders=(1, 0, 0),
            nlopt_kargs={"maxeval": 5000}
        )
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 41.17, rtol=5e-2, err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[0], -5.82, rtol=5e-2, err_msg="wt coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[1], -0.118, rtol=1e-1, err_msg="mpgLag1 coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.scale, 8.42, rtol=5e-2, err_msg="Scale doesn't match R"
        )
        np.testing.assert_allclose(
            model.log_lik, -79.49, rtol=5e-2, err_msg="Log-likelihood doesn't match R"
        )

    def test_alm_mtcars_orders_p2_d0(self, mtcars_data):
        """Test ALM with mtcars: mpg ~ wt, orders=(2,0,0), dnorm.

        R results (alm with orders=c(2,0,0)):
        - Intercept: 42.78
        - wt: -5.78
        - mpgLag1: -0.023
        - mpgLag2: -0.181
        - scale: 7.59 (sigma^2)
        - log-likelihood: -77.83
        """
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", orders=(2, 0, 0),
            nlopt_kargs={"maxeval": 5000}
        )
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 42.78, rtol=5e-2, err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[0], -5.78, rtol=5e-2, err_msg="wt coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[1], -0.023, rtol=1, err_msg="mpgLag1 coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef[2], -0.181, rtol=1e-1, err_msg="mpgLag2 coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.scale, 7.59, rtol=5e-2, err_msg="Scale doesn't match R"
        )
        np.testing.assert_allclose(
            model.log_lik, -77.83, rtol=5e-2, err_msg="Log-likelihood doesn't match R"
        )

    def test_alm_orders_ma_not_implemented(self, mtcars_data):
        """Test that MA(q) raises NotImplementedError."""
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", orders=(0, 0, 1),
            nlopt_kargs={"maxeval": 100}
        )

        with pytest.raises(NotImplementedError, match="MA.*not implemented"):
            model.fit(X, y)

    def test_alm_orders_feature_names(self):
        """Test that orders parameter adds correct feature names."""
        data = {"y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        y, X = formula("y ~ x", data)

        model = ALM(distribution="dnorm", orders=(2, 0, 0))
        model.fit(X, y)

        assert "yLag1" in model._feature_names
        assert "yLag2" in model._feature_names

    def test_alm_orders_i1_pure_i(self, mtcars_data):
        """Test ALM with pure I(1) differencing: orders=(0,1,0).

        For a pure I(1) model (random walk with drift), the ARI polynomial
        is (1-B), giving a single constrained lag coefficient of 1.0.
        """
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", orders=(0, 1, 0),
            nlopt_kargs={"maxeval": 2000}
        )
        model.fit(X, y)

        assert model.arima_string_ == "ARIMA(0,1,0)"
        assert model.arima_polynomial_ is not None
        assert list(model.arima_polynomial_.keys()) == ["mpgLag1"]
        np.testing.assert_allclose(
            model.arima_polynomial_["mpgLag1"], 1.0, atol=1e-10,
            err_msg="Pure I(1) lag coefficient must be exactly 1.0"
        )

    def test_alm_orders_ari11(self, mtcars_data):
        """Test ALM with ARI(1,1): orders=(1,1,0).

        For ARI(1,1), the combined polynomial of AR(1)*(1-B) gives
        coefficients [1+phi1, -phi1] for [Lag1, Lag2], which always sum to 1.
        """
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", loss="likelihood", orders=(1, 1, 0),
            nlopt_kargs={"maxeval": 5000}
        )
        model.fit(X, y)

        assert model.arima_string_ == "ARIMA(1,1,0)"
        assert model.arima_polynomial_ is not None
        assert list(model.arima_polynomial_.keys()) == ["mpgLag1", "mpgLag2"]

        # Invariant: coefficients of ARI(1,d) polynomial always sum to 1
        poly_sum = sum(model.arima_polynomial_.values())
        np.testing.assert_allclose(
            poly_sum, 1.0, atol=1e-8,
            err_msg="ARI(1,1) polynomial coefficients must sum to 1.0"
        )

    def test_alm_orders_ari11_feature_names(self, mtcars_data):
        """Test feature names and arima_polynomial_ keys for ARI(1,1).

        For orders=(1,1,0): ar_order=1 free phi parameter → 1 lag name in
        _feature_names; all ari_order=2 lag names in arima_polynomial_.
        """
        data = mtcars_data
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm", orders=(1, 1, 0),
            nlopt_kargs={"maxeval": 2000}
        )
        model.fit(X, y)

        # Only the ar_order=1 free phi params appear in _feature_names
        assert "mpgLag1" in model._feature_names
        assert "mpgLag2" not in model._feature_names

        # arima_polynomial_ covers all ari_order=2 lags
        assert list(model.arima_polynomial_.keys()) == ["mpgLag1", "mpgLag2"]

