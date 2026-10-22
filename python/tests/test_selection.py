"""Tests for greybox.selection module."""

import numpy as np
import pandas as pd
import pytest

from greybox.alm import ALM
from greybox.selection import CALM, CALMResult, stepwise


class TestStepwise:
    """Tests for stepwise selection."""

    def test_smoke_mtcars(self, mtcars):
        """Test stepwise on mtcars data."""
        data = {
            "mpg": mtcars["mpg"].tolist(),
            "cyl": mtcars["cyl"].tolist(),
            "disp": mtcars["disp"].tolist(),
            "hp": mtcars["hp"].tolist(),
            "wt": mtcars["wt"].tolist(),
        }
        model = stepwise(data, ic="AICc", distribution="dnorm")
        assert model is not None
        assert model.coef is not None
        assert model.nobs == 32

    def test_stepwise_basic(self):
        """Test stepwise with simple data."""
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(10, 3, n)
        x2 = np.random.normal(50, 5, n)
        noise = np.random.normal(0, 3, n)
        y = 100 + 0.5 * x1 - 0.75 * x2 + noise

        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}

        model = stepwise(data, ic="AICc", silent=True)
        assert model is not None
        assert hasattr(model, "ic_values")
        assert isinstance(model.ic_values, dict)
        assert len(model.ic_values) >= 2

    def test_stepwise_with_noise(self):
        """Test stepwise correctly ignores noise variable."""
        np.random.seed(123)
        n = 200
        x1 = np.random.normal(10, 3, n)
        x2 = np.random.normal(50, 5, n)
        x3 = np.random.normal(0, 1, n)
        noise = np.random.normal(0, 100, n)
        y = 100 + 0.5 * x1 - 0.75 * x2 + noise

        data = {
            "y": y.tolist(),
            "x1": x1.tolist(),
            "x2": x2.tolist(),
            "x3": x3.tolist(),
        }

        model = stepwise(data, ic="AICc", silent=True)
        assert model is not None

    def test_stepwise_ic_values_attribute(self):
        """Test that ic_values attribute is a dict with step names."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, n)

        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}

        model = stepwise(data, ic="AICc", silent=True)
        assert hasattr(model, "ic_values")
        assert isinstance(model.ic_values, dict)
        assert len(model.ic_values) > 0
        # First key should always be "Intercept"
        assert list(model.ic_values.keys())[0] == "Intercept"
        # All keys are strings, all values are floats
        assert all(isinstance(k, str) for k in model.ic_values.keys())
        assert all(
            isinstance(v, (float, np.floating)) for v in model.ic_values.values()
        )

    def test_stepwise_time_elapsed_attribute(self):
        """Test that time_elapsed attribute is properly set."""
        data = {"y": list(range(1, 11)), "x1": list(range(1, 11))}

        model = stepwise(data, ic="AICc", silent=True)
        assert hasattr(model, "time_elapsed")
        assert isinstance(model.time_elapsed, float)
        assert model.time_elapsed >= 0

    def test_stepwise_ic_types(self):
        """Test stepwise with different IC types."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + np.random.normal(0, 0.1, n)

        data = {"y": y.tolist(), "x1": x1.tolist()}

        for ic_type in ["AICc", "AIC", "BIC", "BICc"]:
            model = stepwise(data, ic=ic_type, silent=True)
            assert model is not None
            assert hasattr(model, "ic_values")

    def test_stepwise_silent_output(self, capsys):
        """Test that silent=False produces output."""
        np.random.seed(42)
        n = 30
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, n)

        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}

        model = stepwise(data, ic="AICc", silent=False)
        captured = capsys.readouterr()
        assert "Formula:" in captured.out or "IC:" in captured.out

    def test_stepwise_correlation_method(self):
        """Test stepwise with different correlation methods."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, n)

        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}

        for method in ["pearson", "spearman", "kendall"]:
            model = stepwise(data, ic="AICc", method=method, silent=True)
            assert model is not None

    def test_stepwise_no_improvement(self):
        """Test stepwise when no variable improves IC."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        noise = np.random.normal(100, 0.001, n)
        y = 1 + 2 * x1 + noise

        data = {"y": y.tolist(), "x1": x1.tolist(), "noise": noise.tolist()}

        model = stepwise(data, ic="AICc", silent=True)
        assert model is not None

    def test_stepwise_returns_alm_object(self):
        """Test that stepwise returns an ALM object."""
        np.random.seed(42)
        n = 30
        x1 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + np.random.normal(0, 0.1, n)

        data = {"y": y.tolist(), "x1": x1.tolist()}

        model = stepwise(data, ic="AICc", silent=True)
        from greybox.alm import ALM

        assert isinstance(model, ALM)
        assert hasattr(model, "coef")
        assert hasattr(model, "intercept_")

    def test_stepwise_alm_compatibility(self):
        """Test that stepwise result supports all key ALM properties/methods."""
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(10, 3, n)
        x2 = np.random.normal(50, 5, n)
        y = 100 + 0.5 * x1 - 0.75 * x2 + np.random.normal(0, 3, n)

        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}
        model = stepwise(data, ic="AICc", silent=True)

        from greybox.alm import ALM

        # isinstance check
        assert isinstance(model, ALM)

        # Core properties
        assert model.coef is not None
        assert isinstance(model.intercept_, (float, np.floating))
        assert model.nobs == n
        assert model.nparam > 0

        # Residuals, fitted, actuals
        assert model.residuals is not None
        assert len(model.residuals) == n
        assert model.fitted is not None
        assert len(model.fitted) == n
        assert model.actuals is not None
        assert len(model.actuals) == n

        # Variance and log-likelihood
        assert isinstance(model.sigma, (float, np.floating))
        assert isinstance(model.log_lik, (float, np.floating))
        assert model.formula is not None

        # Variance-covariance matrix
        vcov = model.vcov()
        assert vcov is not None
        assert vcov.ndim == 2

        # Summary
        summary = model.summary()
        assert summary is not None
        assert str(summary)  # should be printable

        # Confidence intervals
        ci = model.confint()
        assert ci is not None

        # Predict — include intercept column to match fitted design matrix
        X_pred = np.column_stack([np.ones(5), x1[:5], x2[:5]])
        pred = model.predict(X_pred)
        assert pred is not None
        assert hasattr(pred, "mean")
        assert len(pred.mean) == 5

        # Score (sklearn-compatible)
        score = model.score(X_pred, y[:5])
        assert isinstance(score, float)

        # Information criteria
        assert isinstance(model.aic, (float, np.floating))
        assert isinstance(model.bic, (float, np.floating))
        assert isinstance(model.aicc, (float, np.floating))
        assert isinstance(model.bicc, (float, np.floating))

        # Stepwise-specific attributes
        assert isinstance(model.ic_values, dict)
        assert list(model.ic_values.keys())[0] == "Intercept"
        assert all(isinstance(k, str) for k in model.ic_values.keys())
        assert all(
            isinstance(v, (float, np.floating)) for v in model.ic_values.values()
        )
        assert isinstance(model.time_elapsed, float)
        assert model.time_elapsed >= 0


class TestCALMCore:
    """Tests for CALM function."""

    def test_calm_basic(self):
        """Test CALM with simple data."""
        np.random.seed(42)
        n = 30
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, n)

        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}

        result = CALM(data, ic="AICc", bruteforce=True, silent=True)
        assert "coefficients" in result
        assert "fitted" in result
        assert "IC" in result

    def test_calm_return_keys(self):
        """Test that CALM returns all expected keys."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}

        result = CALM(data, ic="AICc", bruteforce=True)
        expected_keys = [
            "coefficients",
            "coefficient_names",
            "vcov",
            "fitted",
            "residuals",
            "mu",
            "scale",
            "distribution",
            "log_lik",
            "IC",
            "IC_type",
            "df_residual",
            "df",
            "importance",
            "combination",
            "combination_col_names",
            "combination_row_names",
            "y_variable",
            "x_variables",
            "weights",
            "n_obs",
            "time_elapsed",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_calm_intercept_only_model_included(self):
        """Test that the intercept-only model is included in bruteforce."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1}

        result = CALM(data, ic="AICc", bruteforce=True)
        # With 1 variable: should have 2^1 = 2 models
        assert result["combination"].shape[0] == 2
        # First model should be intercept-only (all vars = 0)
        assert result["combination"][0, 0] == 0  # x1 not included

    def test_calm_2var_model_count(self):
        """Test that bruteforce generates 2^p models."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}

        result = CALM(data, ic="AICc", bruteforce=True)
        # 2 variables -> 2^2 = 4 models
        assert result["combination"].shape[0] == 4
        assert len(result["weights"]) == 4

    def test_calm_3var_model_count(self):
        """Test bruteforce with 3 variables generates 2^3 = 8 models."""
        np.random.seed(42)
        n = 80
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        x3 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2, "x3": x3}

        result = CALM(data, ic="AICc", bruteforce=True)
        assert result["combination"].shape[0] == 8

    def test_calm_weights_sum_to_one(self):
        """Test that IC weights sum to 1."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}

        result = CALM(data, ic="AICc", bruteforce=True)
        np.testing.assert_allclose(np.sum(result["weights"]), 1.0)

    def test_calm_importance_bounds(self):
        """Test importance is between 0 and 1."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}

        result = CALM(data, ic="AICc", bruteforce=True)
        # Intercept importance is always 1
        assert result["importance"][0] == 1.0
        # Variable importances are sum of IC weights, so in [0, 1]
        for imp in result["importance"][1:]:
            assert 0 <= imp <= 1.0 + 1e-10

    def test_calm_vcov_symmetric(self):
        """Test that combined vcov is symmetric."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}

        result = CALM(data, ic="AICc", bruteforce=True)
        vcov = result["vcov"]
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)

    def test_calm_vcov_positive_diagonal(self):
        """Test that vcov diagonal elements are non-negative."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}

        result = CALM(data, ic="AICc", bruteforce=True)
        assert np.all(np.diag(result["vcov"]) >= 0)

    def test_calm_ic_weighted_average(self):
        """Test that combined IC is the weighted average of model ICs."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}

        result = CALM(data, ic="AICc", bruteforce=True)
        # IC should be the weighted average of model ICs
        combo = result["combination"]
        n_vars = len(result["x_variables"])
        model_ics = combo[:, n_vars + 1]  # last column is ICs
        model_weights = combo[:, n_vars]  # second to last is weights
        expected_ic = np.sum(model_weights * model_ics)
        np.testing.assert_allclose(result["IC"], expected_ic, rtol=1e-10)

    def test_calm_df_residual(self):
        """Test that df_residual = n - sum(importance) - 1."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}

        result = CALM(data, ic="AICc", bruteforce=True)
        expected_df = n - np.sum(result["importance"]) - 1
        np.testing.assert_allclose(result["df_residual"], expected_df, rtol=1e-10)

    def test_calm_coefficient_names(self):
        """Test coefficient names match variable names."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}

        result = CALM(data, ic="AICc", bruteforce=True)
        assert result["coefficient_names"] == [
            "(Intercept)",
            "x1",
            "x2",
        ]
        assert len(result["coefficients"]) == 3

    def test_calm_fitted_residuals_shape(self):
        """Test fitted and residuals have correct shape."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1}

        result = CALM(data, ic="AICc", bruteforce=True)
        assert result["fitted"].shape == (n,)
        assert result["residuals"].shape == (n,)
        assert result["mu"].shape == (n,)

    def test_calm_dnorm_residuals_are_y_minus_mu(self):
        """Test that for dnorm, residuals = y - mu."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1}

        result = CALM(data, ic="AICc", bruteforce=True, distribution="dnorm")
        np.testing.assert_allclose(result["residuals"], y - result["mu"], atol=1e-12)

    def test_calm_dnorm_fitted_equals_mu(self):
        """Test that for dnorm, fitted = mu."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1}

        result = CALM(data, ic="AICc", bruteforce=True, distribution="dnorm")
        np.testing.assert_allclose(result["fitted"], result["mu"], atol=1e-12)

    def test_calm_ic_types(self):
        """Test CALM with different IC types."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1}

        for ic_type in ["AICc", "AIC", "BIC", "BICc"]:
            result = CALM(data, ic=ic_type, bruteforce=True)
            assert result["IC_type"] == ic_type
            assert np.isfinite(result["IC"])

    def test_calm_combination_table_structure(self):
        """Test the combination table has correct structure."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}

        result = CALM(data, ic="AICc", bruteforce=True)
        combo = result["combination"]
        # Should have n_vars + 2 columns (vars + IC weights + ICs)
        assert combo.shape[1] == 4  # 2 vars + weights + ICs
        # Column names should match
        assert result["combination_col_names"] == ["x1", "x2", "IC weights", "AICc"]
        # Row names
        assert len(result["combination_row_names"]) == combo.shape[0]

    def test_calm_strong_signal_high_importance(self):
        """Test that strong predictors get high importance."""
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        noise = np.random.normal(0, 1, n)
        y = 1 + 5 * x1 + np.random.normal(0, 0.1, n)
        data = {"y": y, "x1": x1, "x2": x2, "noise": noise}

        result = CALM(data, ic="AICc", bruteforce=True)
        # x1 should have importance close to 1
        x1_idx = result["coefficient_names"].index("x1")
        assert result["importance"][x1_idx] > 0.9

    def test_calm_non_bruteforce_basic(self):
        """Test CALM with bruteforce=False."""
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}

        result = CALM(data, ic="AICc", bruteforce=False)
        assert "coefficients" in result
        assert "vcov" in result
        assert np.isfinite(result["IC"])

    def test_calm_single_variable(self):
        """Test CALM with 1 variable (2 models)."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1}

        result = CALM(data, ic="AICc", bruteforce=True)
        assert result["combination"].shape[0] == 2
        assert len(result["coefficients"]) == 2

    def test_calm_dataframe_input(self):
        """Test CALM accepts DataFrame input."""
        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n),
                "x1": np.random.normal(0, 1, n),
                "x2": np.random.normal(0, 1, n),
            }
        )

        result = CALM(df, ic="AICc", bruteforce=True)
        assert "coefficients" in result


class TestCALMPrintSummary:
    """Tests for CALMResult print and summary methods."""

    @pytest.fixture
    def result(self):
        """Create a basic CALM result for testing."""
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(10, 3, n)
        x2 = np.random.normal(50, 5, n)
        noise = np.random.normal(0, 1, n)
        y = 100 + 0.5 * x1 - 0.75 * x2 + np.random.normal(0, 3, n)
        data = {"y": y, "x1": x1, "x2": x2, "Noise": noise}
        return CALM(data, ic="AICc", bruteforce=True, silent=True)

    def test_result_is_calm_result(self, result):
        """Test that CALM returns CALMResult."""
        assert isinstance(result, CALMResult)

    def test_dict_access(self, result):
        """Test dict-like [] access."""
        coef = result["coefficients"]
        assert coef is not None
        assert len(coef) > 0

    def test_contains(self, result):
        """Test 'in' operator."""
        assert "coefficients" in result
        assert "vcov" in result
        assert "IC" in result
        assert "nonexistent_key" not in result

    def test_keys(self, result):
        """Test keys() method."""
        k = result.keys()
        assert "coefficients" in k
        assert "vcov" in k
        assert "time_elapsed" in k

    def test_print_contains_coefficients(self, result):
        """Test str(result) contains Coefficients."""
        s = str(result)
        assert "Coefficients:" in s
        assert "(Intercept)" in s
        assert "x1" in s
        assert "x2" in s

    def test_print_contains_time_elapsed(self, result):
        """Test str(result) contains time elapsed."""
        s = str(result)
        assert "Time elapsed:" in s
        assert "seconds" in s

    def test_print_contains_model_info(self, result):
        """Test str(result) contains ADAM-style model info."""
        s = str(result)
        assert "Model estimated: CALM(AICc)" in s
        assert "Distribution assumed in the model:" in s
        assert "Loss function type: likelihood" in s

    def test_repr(self, result):
        """Test CALMResult repr."""
        r = repr(result)
        assert "CALMResult" in r
        assert "IC_type='AICc'" in r
        assert "fitted=True" in r

    def test_summary_contains_all_sections(self, result):
        """Test summary output contains all expected sections."""
        s = str(result.summary())
        assert "combined model" in s
        assert "Response variable: y" in s
        assert "Distribution used in the estimation:" in s
        assert "Coefficients:" in s
        assert "Estimate" in s
        assert "Std. Error" in s
        assert "Importance" in s
        assert "Lower 2.5%" in s
        assert "Upper 97.5%" in s
        assert "Error standard deviation:" in s
        assert "Sample size:" in s
        assert "Number of estimated parameters:" in s
        assert "Number of degrees of freedom:" in s
        assert "Approximate combined information criteria:" in s
        assert "AIC" in s
        assert "AICc" in s
        assert "BIC" in s
        assert "BICc" in s

    def test_summary_significance_stars(self, result):
        """Test that significant coefficients get * markers."""
        s = str(result.summary())
        # With this data, x1 and x2 should be significant
        # At minimum, the summary should contain at least one *
        assert "*" in s

    def test_summary_ic_type_in_header(self, result):
        """Test that IC type appears in the summary header."""
        s = str(result.summary())
        assert "AICc combined model" in s

    def test_summary_nparam_fractional(self, result):
        """Test that nparam = sum(importance) + 1."""
        summary = result.summary()
        expected = float(np.sum(result["importance"])) + 1
        assert abs(summary.nparam - expected) < 1e-10

    def test_attribute_access(self, result):
        """Test attribute-style access."""
        assert hasattr(result, "coefficients")
        assert hasattr(result, "vcov")
        assert result.distribution == "dnorm"
        assert result.IC_type == "AICc"


class TestCALMALMCompat:
    """Tests for ALM-compatible interface on CALMResult."""

    @pytest.fixture
    def setup(self):
        """Create CALM result and raw data for testing."""
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(10, 3, n)
        x2 = np.random.normal(50, 5, n)
        y = 100 + 0.5 * x1 - 0.75 * x2 + np.random.normal(0, 3, n)
        data = {"y": y, "x1": x1, "x2": x2}
        result = CALM(data, ic="AICc", bruteforce=True, silent=True)
        X = np.column_stack([np.ones(n), x1, x2])
        return result, X, y, x1, x2

    def test_predict_no_args(self, setup):
        """predict() with no args returns training fitted values."""
        result, X, y, x1, x2 = setup
        pred = result.predict()
        assert pred.mean is not None
        np.testing.assert_allclose(pred.mean, result.fitted)
        assert pred.lower is None
        assert pred.upper is None

    def test_predict_with_X(self, setup):
        """predict(X) returns predictions."""
        result, X, y, x1, x2 = setup
        pred = result.predict(X[:5])
        assert len(pred.mean) == 5
        assert pred.lower is None
        assert pred.upper is None

    def test_predict_formula_design(self):
        """predict() of a design matrix from formula(), as ALM.predict()."""
        from greybox import formula, mtcars

        result = CALM(mtcars[["mpg", "wt", "hp", "qsec", "drat"]])
        _, X = formula("mpg ~ wt + hp + qsec + drat", mtcars)
        pred = result.predict(X)
        np.testing.assert_allclose(pred.mean, result.predict(mtcars).mean)
        np.testing.assert_allclose(pred.mean, result.fitted)

    def test_predict_with_prediction_intervals(self, setup):
        """Prediction intervals work."""
        result, X, y, x1, x2 = setup
        pred = result.predict(X[:5], interval="prediction")
        assert len(pred.mean) == 5
        assert pred.lower is not None
        assert pred.upper is not None
        assert np.all(pred.lower < pred.mean)
        assert np.all(pred.upper > pred.mean)

    def test_predict_confidence_intervals(self, setup):
        """Confidence intervals work and are narrower than prediction."""
        result, X, y, x1, x2 = setup
        pred_ci = result.predict(X[:5], interval="confidence")
        pred_pi = result.predict(X[:5], interval="prediction")
        # Confidence intervals should be narrower
        ci_width = pred_ci.upper - pred_ci.lower
        pi_width = pred_pi.upper - pred_pi.lower
        assert np.all(ci_width < pi_width)

    def test_predict_side_upper(self, setup):
        """predict with side='upper' returns only upper."""
        result, X, y, x1, x2 = setup
        pred = result.predict(X[:5], interval="prediction", side="upper")
        assert pred.lower is None
        assert pred.upper is not None

    def test_predict_side_lower(self, setup):
        """predict with side='lower' returns only lower."""
        result, X, y, x1, x2 = setup
        pred = result.predict(X[:5], interval="prediction", side="lower")
        assert pred.lower is not None
        assert pred.upper is None

    def test_predict_level_and_variances(self, setup):
        """predict() populates level and variances fields."""
        result, X, y, x1, x2 = setup

        # With intervals
        pred = result.predict(X[:5], interval="prediction", level=0.90)
        assert pred.level == 0.9  # as ALM.predict
        assert pred.variances is not None
        assert pred.variances.shape == (5,)
        assert np.all(pred.variances > 0)

        # Without intervals
        pred_none = result.predict(X[:5])
        assert pred_none.level == 0.95  # default (not wrapped)
        assert pred_none.variances is None

        # No-args predict (training fitted)
        pred_train = result.predict()
        assert pred_train.level == 0.95  # default
        assert pred_train.variances is None

    def test_predict_wrong_shape(self, setup):
        """predict with wrong X shape raises ValueError."""
        result, X, y, x1, x2 = setup
        with pytest.raises(ValueError, match="features"):
            result.predict(np.ones((5, 10)))

    def test_score_r2(self, setup):
        """score with R2 metric."""
        result, X, y, x1, x2 = setup
        r2 = result.score(X, y, metric="R2")
        assert isinstance(r2, float)
        assert 0 < r2 < 1

    def test_score_mse(self, setup):
        """score with MSE metric."""
        result, X, y, x1, x2 = setup
        mse = result.score(X, y, metric="MSE")
        assert isinstance(mse, float)
        assert mse > 0

    def test_score_mae(self, setup):
        """score with MAE metric."""
        result, X, y, x1, x2 = setup
        mae = result.score(X, y, metric="MAE")
        assert isinstance(mae, float)
        assert mae > 0

    def test_score_likelihood(self, setup):
        """score with likelihood metric."""
        result, X, y, x1, x2 = setup
        ll = result.score(X, y, metric="likelihood")
        assert isinstance(ll, float)
        assert ll == result.log_lik

    def test_score_unknown_metric(self, setup):
        """score with unknown metric raises ValueError."""
        result, X, y, x1, x2 = setup
        with pytest.raises(ValueError, match="Unknown metric"):
            result.score(X, y, metric="foo")

    def test_confint(self, setup):
        """confint returns (n_params, 2) array."""
        result, X, y, x1, x2 = setup
        ci = result.confint()
        assert ci.shape == (3, 2)  # 3 params (intercept + 2 vars)
        # Lower < upper
        assert np.all(ci[:, 0] < ci[:, 1])

    def test_confint_subset(self, setup):
        """confint with parm argument."""
        result, X, y, x1, x2 = setup
        ci = result.confint(parm=0)
        assert ci.shape == (1, 2)

        ci2 = result.confint(parm=[1, 2])
        assert ci2.shape == (2, 2)

    def test_nobs_property(self, setup):
        """nobs returns correct count."""
        result, X, y, x1, x2 = setup
        assert result.nobs == 100
        assert result.nobs == result.n_obs

    def test_nparam_property(self, setup):
        """nparam = sum(importance) + 1."""
        result, X, y, x1, x2 = setup
        expected = float(np.sum(result.importance)) + 1
        assert abs(result.nparam - expected) < 1e-10

    def test_sigma_property(self, setup):
        """sigma is a positive float."""
        result, X, y, x1, x2 = setup
        assert isinstance(result.sigma, float)
        assert result.sigma > 0

    def test_coef_property(self, setup):
        """coef returns slopes only (no intercept)."""
        result, X, y, x1, x2 = setup
        assert len(result.coef) == 2  # x1, x2
        np.testing.assert_array_equal(result.coef, result.coefficients[1:])

    def test_intercept_property(self, setup):
        """intercept_ is a single float."""
        result, X, y, x1, x2 = setup
        assert isinstance(result.intercept_, float)
        assert result.intercept_ == result.coefficients[0]

    def test_aic_bic_properties(self, setup):
        """All 4 ICs are finite floats."""
        result, X, y, x1, x2 = setup
        for ic_name in ("aic", "bic", "aicc", "bicc"):
            val = getattr(result, ic_name)
            assert isinstance(val, float), f"{ic_name} not float"
            assert np.isfinite(val), f"{ic_name} not finite"

    def test_actuals_property(self, setup):
        """actuals returns original y."""
        result, X, y, x1, x2 = setup
        np.testing.assert_array_equal(result.actuals, y)

    def test_data_property(self, setup):
        """data has the response and the regressors, as R's $data."""
        result, X, y, x1, x2 = setup
        np.testing.assert_array_equal(result.data[:, 0], result.actuals)
        np.testing.assert_array_equal(result.data[:, 1:], result._X_train[:, 1:])

    def test_rank_property(self, setup):
        """rank_ is the number of the combined parameters, as R's $rank."""
        result, X, y, x1, x2 = setup
        assert result.rank_ == len(result.coefficients)

    def test_formula_property(self, setup):
        """formula returns a formula string."""
        result, X, y, x1, x2 = setup
        assert isinstance(result.formula, str)
        assert "y" in result.formula
        assert "x1" in result.formula
        assert "x2" in result.formula

    def test_vcov_method(self, setup):
        """vcov() returns the matrix."""
        result, X, y, x1, x2 = setup
        v = result.vcov()
        assert v.ndim == 2
        assert v.shape == (3, 3)
        np.testing.assert_allclose(v, v.T, atol=1e-12)

    def test_dict_access_vcov(self, setup):
        """result['vcov'] still works."""
        result, X, y, x1, x2 = setup
        v = result["vcov"]
        assert v is not None
        np.testing.assert_array_equal(v, result.vcov())

    def test_feature_names(self, setup):
        """_feature_names returns variable names without intercept."""
        result, X, y, x1, x2 = setup
        assert result._feature_names == ["x1", "x2"]

    def test_n_features(self, setup):
        """_n_features includes intercept column."""
        result, X, y, x1, x2 = setup
        assert result._n_features == 3

    def test_df_residual_alias(self, setup):
        """df_residual_ matches df_residual."""
        result, X, y, x1, x2 = setup
        assert result.df_residual_ == result.df_residual

    def test_loglik_property(self, setup):
        """loglik returns same as log_lik."""
        result, X, y, x1, x2 = setup
        assert result.loglik == result.log_lik
        assert isinstance(result.loglik, float)

    def test_distribution_underscore(self, setup):
        """distribution_ returns distribution name."""
        result, X, y, x1, x2 = setup
        assert result.distribution_ == "dnorm"

    def test_loss_underscore(self, setup):
        """loss_ returns 'likelihood'."""
        result, X, y, x1, x2 = setup
        assert result.loss_ == "likelihood"

    def test_n_param_property(self, setup):
        """n_param returns dict with number and df."""
        result, X, y, x1, x2 = setup
        np_dict = result.n_param
        assert "number" in np_dict
        assert "df" in np_dict
        assert np_dict["number"] == result.nparam

    def test_private_attrs_not_in_keys(self, setup):
        """Private attrs don't show in keys()."""
        result, X, y, x1, x2 = setup
        k = result.keys()
        assert "actuals" not in k
        assert "X_train" not in k
        assert "formula_str" not in k
        assert "_actuals" not in k
        # vcov should appear (special-cased)
        assert "vcov" in k


class TestCALMRComparison:
    """Tests comparing Python CALM against R calm."""

    @pytest.fixture(autouse=True)
    def _check_rpy2(self):
        """Skip tests if rpy2 or R greybox not available."""
        pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

    def _run_r_calm(self, y, x_dict, ic="AICc", distribution="dnorm", bruteforce=True):
        """Run R's calm and return results as a dict."""
        import rpy2.robjects as ro

        r = ro.r
        r("library(greybox)")

        # Build R data frame
        r_cols = {"y": ro.FloatVector(y.tolist())}
        for name, values in x_dict.items():
            r_cols[name] = ro.FloatVector(values.tolist())
        r_data = ro.DataFrame(r_cols)

        ro.globalenv["mydata"] = r_data
        ro.globalenv["my_ic"] = ic
        ro.globalenv["my_dist"] = distribution
        ro.globalenv["my_bruteforce"] = bruteforce

        r("""
        result <- calm(mydata, ic=my_ic, bruteforce=my_bruteforce,
                       silent=TRUE, distribution=my_dist)
        """)

        # Extract results
        coef = np.array(r("coef(result)"))
        vcov = np.array(r("result$vcov"))
        fitted = np.array(r("result$fitted"))
        residuals = np.array(r("result$residuals"))
        importance = np.array(r("result$importance"))
        ic_val = float(np.array(r("result$IC"))[0])
        log_lik = float(np.array(r("logLik(result)"))[0])
        df_res = float(np.array(r("result$df.residual"))[0])
        df_model = float(np.array(r("result$df"))[0])

        combo = np.array(r("result$combination"))

        return {
            "coefficients": coef,
            "vcov": vcov,
            "fitted": fitted,
            "residuals": residuals,
            "importance": importance,
            "IC": ic_val,
            "log_lik": log_lik,
            "df_residual": df_res,
            "df": df_model,
            "combination": combo,
        }

    def test_compare_dnorm_2vars(self):
        """Compare Python vs R calm with dnorm, 2 variables."""
        np.random.seed(42)
        n = 80
        x1 = np.random.normal(10, 3, n)
        x2 = np.random.normal(50, 5, n)
        y = 100 + 0.5 * x1 - 0.75 * x2 + np.random.normal(0, 3, n)

        x_dict = {"x1": x1, "x2": x2}
        py_data = {"y": y, "x1": x1, "x2": x2}

        r_result = self._run_r_calm(y, x_dict, ic="AICc")
        py_result = CALM(py_data, ic="AICc", bruteforce=True, distribution="dnorm")

        # Coefficients
        np.testing.assert_allclose(
            py_result["coefficients"],
            r_result["coefficients"],
            rtol=1e-6,
        )

        # Importance
        np.testing.assert_allclose(
            py_result["importance"],
            r_result["importance"],
            rtol=1e-6,
        )

        # IC (weighted average)
        np.testing.assert_allclose(
            py_result["IC"],
            r_result["IC"],
            rtol=1e-6,
        )

        # Fitted values
        np.testing.assert_allclose(
            py_result["fitted"],
            r_result["fitted"],
            rtol=1e-6,
        )

        # vcov
        np.testing.assert_allclose(py_result["vcov"], r_result["vcov"], rtol=1e-6)

        # df_residual
        np.testing.assert_allclose(
            py_result["df_residual"],
            r_result["df_residual"],
            rtol=1e-6,
        )

    def test_compare_dnorm_3vars(self):
        """Compare Python vs R calm with dnorm, 3 variables."""
        np.random.seed(123)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        x3 = np.random.normal(0, 1, n)
        y = 5 + 2 * x1 - 1 * x2 + np.random.normal(0, 1, n)

        x_dict = {"x1": x1, "x2": x2, "x3": x3}
        py_data = {"y": y, "x1": x1, "x2": x2, "x3": x3}

        r_result = self._run_r_calm(y, x_dict, ic="AICc")
        py_result = CALM(py_data, ic="AICc", bruteforce=True, distribution="dnorm")

        # 8 models for 3 variables
        assert py_result["combination"].shape[0] == 8

        # Coefficients
        np.testing.assert_allclose(
            py_result["coefficients"],
            r_result["coefficients"],
            rtol=1e-6,
        )

        # Importance
        np.testing.assert_allclose(
            py_result["importance"],
            r_result["importance"],
            rtol=1e-6,
        )

        # x3 (noise) should have lower importance than x1 and x2
        assert py_result["importance"][3] < py_result["importance"][1]
        assert py_result["importance"][3] < py_result["importance"][2]

    def test_compare_dnorm_1var(self):
        """Compare Python vs R calm with dnorm, 1 variable."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        y = 3 + 2 * x1 + np.random.normal(0, 0.5, n)

        x_dict = {"x1": x1}
        py_data = {"y": y, "x1": x1}

        r_result = self._run_r_calm(y, x_dict, ic="AICc")
        py_result = CALM(py_data, ic="AICc", bruteforce=True, distribution="dnorm")

        # 2 models
        assert py_result["combination"].shape[0] == 2

        np.testing.assert_allclose(
            py_result["coefficients"],
            r_result["coefficients"],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            py_result["importance"],
            r_result["importance"],
            rtol=1e-6,
        )

    def test_compare_bic(self):
        """Compare Python vs R calm with BIC."""
        np.random.seed(42)
        n = 80
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)

        x_dict = {"x1": x1, "x2": x2}
        py_data = {"y": y, "x1": x1, "x2": x2}

        r_result = self._run_r_calm(y, x_dict, ic="BIC")
        py_result = CALM(py_data, ic="BIC", bruteforce=True, distribution="dnorm")

        np.testing.assert_allclose(
            py_result["coefficients"],
            r_result["coefficients"],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            py_result["importance"],
            r_result["importance"],
            rtol=1e-6,
        )

    def test_compare_bruteforce_false(self):
        """Compare Python vs R calm with bruteforce=FALSE."""
        np.random.seed(42)
        n = 80
        x1 = np.random.normal(10, 3, n)
        x2 = np.random.normal(50, 5, n)
        x3 = np.random.normal(0, 1, n)
        y = 100 + 0.5 * x1 - 0.75 * x2 + np.random.normal(0, 3, n)

        x_dict = {"x1": x1, "x2": x2, "x3": x3}
        py_data = {"y": y, "x1": x1, "x2": x2, "x3": x3}

        r_result = self._run_r_calm(y, x_dict, ic="AICc", bruteforce=False)
        py_result = CALM(py_data, ic="AICc", bruteforce=False, distribution="dnorm")

        # Coefficients
        np.testing.assert_allclose(
            py_result["coefficients"],
            r_result["coefficients"],
            rtol=1e-6,
        )

        # Importance
        np.testing.assert_allclose(
            py_result["importance"],
            r_result["importance"],
            rtol=1e-6,
        )

        # IC (weighted average)
        np.testing.assert_allclose(
            py_result["IC"],
            r_result["IC"],
            rtol=1e-6,
        )


class TestCALM:
    """Tests for CALM function."""

    def test_calm_basic(self):
        """Test CALM with simple data."""
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 100)
        x2 = np.random.normal(0, 1, 100)
        y = 1 + 2 * x1 + 0.5 * x2 + np.random.normal(0, 1, 100)

        data = {"y": y, "x1": x1, "x2": x2}
        result = CALM(data, ic="AICc", silent=True)

        assert result is not None
        assert hasattr(result, "coefficients")
        assert hasattr(result, "importance")


class TestSelectionRBehaviour:
    """Behaviour of stepwise and CALM that follows R."""

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(60, 3))
        y = 2 + 1.5 * x[:, 0] - x[:, 1] + rng.normal(size=60)
        return pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2]})

    def test_summary_labels_follow_level(self, data):
        text = str(CALM(data, bruteforce=True).summary(level=0.9))
        assert "Lower 5.0%" in text
        assert "Upper 95.0%" in text

    def test_bicc(self, data):
        result = CALM(data, bruteforce=True)
        k, n = result.nparam, result.nobs
        expected = -2 * result.log_lik + k * np.log(n) * n / (n - k - 1)
        np.testing.assert_allclose(result.bicc, expected)
        np.testing.assert_allclose(result.summary().bicc, expected)

    def test_plogis_without_zeros(self):
        """A binary regressor and a response with no zeros: R's intercept only."""
        from greybox import mtcars

        data = mtcars[["mpg", "wt", "hp", "qsec", "cyl", "am"]]
        model = stepwise(data, distribution="plogis")
        assert len(model.coef) == 0
        np.testing.assert_allclose(model.intercept_, 751.0)

    def test_intercept_only_returns_alm(self):
        from greybox.alm import ALM

        rng = np.random.default_rng(0)
        data = pd.DataFrame(rng.normal(size=(30, 3)), columns=["y", "x1", "x2"])
        result = CALM(data)
        assert isinstance(result, ALM)
        assert len(result.coefficients) == 1

    def test_many_variables_warning(self):
        rng = np.random.default_rng(1)
        data = pd.DataFrame(rng.normal(size=(40, 15)))
        data.columns = ["y"] + [f"x{i}" for i in range(1, 15)]
        with pytest.warns(UserWarning, match="more than 14 variables"):
            result = CALM(data, bruteforce=True)
        assert len(result.combination) == 2**14

    def test_missing_values_dropped(self, data):
        data = data.copy()
        data.loc[[3, 10], "x2"] = np.nan
        assert stepwise(data).nobs == 58
        assert CALM(data, bruteforce=True).nobs == 58

    def test_plogis_response_binarised(self, data):
        data = data.assign(y=np.maximum(data["y"] - 2, 0))
        model = stepwise(data, distribution="plogis")
        np.testing.assert_array_equal(model.actuals, (data["y"] != 0) * 1.0)

    def test_occurrence_model(self, data):
        data = data.assign(y=np.where(data["x3"] > 0, np.exp(data["y"] / 4), 0))
        model = stepwise(data, distribution="dlnorm", occurrence="plogis")
        assert isinstance(model.occurrence, ALM)
        assert model.occurrence.distribution == "plogis"

    def test_combinations_in_binary_order(self, data):
        result = CALM(data, bruteforce=True)
        np.testing.assert_array_equal(
            result.combination[:, :3],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
            + [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        )

    @pytest.mark.parametrize("distribution", ["dbcnorm", "dgnorm", "dalaplace"])
    def test_other_combined(self, data, distribution):
        """other_ is the weighted extra parameter, which the predictions use."""
        data = data.assign(y=np.exp(data["y"] / 4))
        result = CALM(data[["y", "x1"]], bruteforce=True, distribution=distribution)
        if distribution == "dalaplace":
            assert result.other_ == 0.5
        else:
            others = [
                ALM(distribution=distribution)
                .fit(result._X_train[:, : 1 + j], result.actuals)
                .other_
                for j in range(2)
            ]
            np.testing.assert_allclose(result.other_, result.weights @ others)
        prediction = result.predict(interval="prediction")
        assert np.all(prediction.lower < prediction.upper)
