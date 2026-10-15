"""Tests for B() backshift operator and multipliers() dynamic multipliers."""

import numpy as np
import pytest

from greybox import ALM, B, multipliers
from greybox.formula import formula
from greybox.xreg import _dyn_mult_calc, _get_betas


# ---------------------------------------------------------------------------
# B() — backshift operator
# ---------------------------------------------------------------------------


class TestBackshiftOperator:
    def test_k_zero_returns_unchanged(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(B(x, 0), x)

    def test_k_positive_lags(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Lag 1: first value filled by boundary strategy (extrapolate → 1.0)
        result = B(x, 1)
        assert result.shape == (5,)
        # Values at positions 1..4 should be original x[0..3]
        np.testing.assert_array_equal(result[1:], x[:4])

    def test_k_negative_leads(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Lead 1: last value filled by boundary strategy
        result = B(x, -1)
        assert result.shape == (5,)
        # Values at positions 0..3 should be original x[1..4]
        np.testing.assert_array_equal(result[:4], x[1:])

    def test_output_shape(self):
        x = np.arange(10, dtype=float)
        for k in (-2, -1, 0, 1, 2):
            assert B(x, k).shape == (10,)

    def test_list_input(self):
        x = [10.0, 20.0, 30.0]
        result = B(x, 1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# formula() with B() terms
# ---------------------------------------------------------------------------


class TestFormulaWithB:
    def setup_method(self):
        np.random.seed(42)
        n = 20
        self.x = np.arange(1.0, n + 1)
        self.y = 2.0 * self.x + np.random.randn(n)
        self.data = {"y": self.y, "x": self.x}

    def test_column_names(self):
        y_out, X = formula("y ~ x + B(x,1)", self.data)
        assert "(Intercept)" in X.columns
        assert "x" in X.columns
        assert "B(x,1)" in X.columns

    def test_column_names_with_space(self):
        # B(x, 1) with space after comma should work the same
        y_out, X = formula("y ~ x + B(x, 1)", self.data)
        # Column name is normalised to B(x,1) (no space)
        assert "B(x,1)" in X.columns

    def test_b_column_values(self):
        y_out, X = formula("y ~ x + B(x,1)", self.data)
        # B(x,1) is lag-1 of x; values at rows 1..n-1 should equal x[0..n-2]
        b_col = X["B(x,1)"].values
        np.testing.assert_array_equal(b_col[1:], self.x[:-1])

    def test_negative_lag_lead(self):
        y_out, X = formula("y ~ x + B(x,-1)", self.data)
        assert "B(x,-1)" in X.columns
        # lead-1: values at rows 0..n-2 should equal x[1..n-1]
        b_col = X["B(x,-1)"].values
        np.testing.assert_array_equal(b_col[:-1], self.x[1:])

    def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="not found in data"):
            formula("y ~ B(z,1)", self.data)

    def test_dot_expansion_with_b(self):
        """formula with '.' and B() should not raise. Raw base var IS added by '.'."""
        import pandas as pd

        df = pd.DataFrame({"y": self.y, "x": self.x, "z": self.x * 0.1})
        y_out, X = formula("y ~ B(x,1)+.", df)
        assert "B(x,1)" in X.columns
        assert "z" in X.columns  # from "."
        assert "x" in X.columns  # raw x added by "." — matches R semantics


# ---------------------------------------------------------------------------
# _dyn_mult_calc — recurrence kernel
# ---------------------------------------------------------------------------


class TestDynMultCalc:
    def test_no_ar_pure_dl(self):
        # With phi=[0], m_s = beta_s for all s
        phi = np.array([0.0])
        beta = np.array([1.0, 0.5, 0.25])
        result = _dyn_mult_calc(phi, beta, 5)
        np.testing.assert_allclose(result[:3], [1.0, 0.5, 0.25])
        np.testing.assert_allclose(result[3:], [0.0, 0.0])

    def test_ar1_no_dl(self):
        # AR(1) with phi=0.8, only beta_0=1 (contemporaneous)
        phi = np.array([0.8])
        beta = np.array([1.0])
        result = _dyn_mult_calc(phi, beta, 5)
        # m_0=1, m_s = 0.8 * m_{s-1}
        expected = np.array([0.8**i for i in range(5)])
        expected[0] = 1.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_h_zero_returns_empty(self):
        phi = np.array([0.5])
        beta = np.array([1.0])
        result = _dyn_mult_calc(phi, beta, 0)
        assert len(result) == 0

    def test_ar2(self):
        # AR(2) with phi=[0.5, 0.3], beta_0=2
        phi = np.array([0.5, 0.3])
        beta = np.array([2.0])
        result = _dyn_mult_calc(phi, beta, 4)
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(0.5 * 2.0)
        assert result[2] == pytest.approx(0.5 * result[1] + 0.3 * result[0])
        assert result[3] == pytest.approx(0.5 * result[2] + 0.3 * result[1])


# ---------------------------------------------------------------------------
# _get_betas — coefficient extraction
# ---------------------------------------------------------------------------


class TestGetBetas:
    def setup_method(self):
        np.random.seed(0)
        n = 30
        # Not a trend: its lag would be perfectly correlated with it, and dropped
        x = np.random.randn(n).cumsum() + 10
        y = 0.5 * x + 0.3 * B(x, 1) + np.random.randn(n) * 0.5
        data = {"y": y, "x": x}
        y_out, X = formula("y ~ x + B(x,1)", data)
        self.model = ALM(distribution="dnorm")
        self.model.fit(X, y_out.values)

    def test_betas_shape(self):
        betas = _get_betas(self.model, "x")
        # Two terms: x (lag 0) and B(x,1) (lag 1)
        assert betas.shape == (2,)

    def test_betas_order(self):
        betas = _get_betas(self.model, "x")
        # betas[0] should be the coefficient for "x", betas[1] for "B(x,1)"
        names = list(self.model._feature_names)
        coefs = list(self.model._coef)
        x_idx = names.index("x")
        bx1_idx = names.index("B(x,1)")
        assert betas[0] == pytest.approx(coefs[x_idx])
        assert betas[1] == pytest.approx(coefs[bx1_idx])

    def test_unknown_parm_raises(self):
        with pytest.raises(ValueError, match="not found in the model"):
            _get_betas(self.model, "z")

    def test_contemporaneous_only(self):
        np.random.seed(1)
        n = 20
        x = np.arange(1.0, n + 1)
        y = 2.0 * x + np.random.randn(n)
        data = {"y": y, "x": x}
        y_out, X = formula("y ~ x", data)
        model = ALM(distribution="dnorm")
        model.fit(X, y_out.values)
        betas = _get_betas(model, "x")
        assert betas.shape == (1,)
        assert betas[0] == pytest.approx(model._coef[0])


# ---------------------------------------------------------------------------
# multipliers() — end-to-end
# ---------------------------------------------------------------------------


class TestMultipliers:
    def test_output_keys(self):
        np.random.seed(42)
        n = 30
        x = np.arange(1.0, n + 1)
        y = 2.0 * x + np.random.randn(n)
        data = {"y": y, "x": x}
        y_out, X = formula("y ~ x", data)
        model = ALM(distribution="dnorm")
        model.fit(X, y_out.values)
        dm = multipliers(model, "x", h=5)
        assert list(dm.keys()) == ["h1", "h2", "h3", "h4", "h5"]

    def test_ar1_decay(self):
        """AR(1) model: multipliers decay geometrically with phi."""
        np.random.seed(0)
        n = 60
        # True DGP: y_t = 1.5 * x_t + 0.7 * y_{t-1} + noise
        x = np.random.randn(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 1.5 * x[t] + 0.7 * y[t - 1] + np.random.randn() * 0.1

        data = {"y": y, "x": x}
        y_out, X = formula("y ~ x", data)
        model = ALM(distribution="dnorm", orders=(1, 0, 0))
        model.fit(X, y_out.values)

        dm = multipliers(model, "x", h=5)
        # Each multiplier should be smaller than or equal to the previous
        # (geometric decay with 0 < phi < 1)
        values = list(dm.values())
        for i in range(1, len(values)):
            assert abs(values[i]) <= abs(values[i - 1]) + 1e-6

    def test_distributed_lags(self):
        """With B(x,1) term, h1 and h2 reflect contemporaneous + lagged coef."""
        np.random.seed(7)
        n = 50
        x = np.random.randn(n)
        y = 2.0 * x + 0.5 * B(x, 1) + np.random.randn(n) * 0.1

        data = {"y": y, "x": x}
        y_out, X = formula("y ~ x + B(x,1)", data)
        model = ALM(distribution="dnorm")
        model.fit(X, y_out.values)

        dm = multipliers(model, "x", h=3)
        # h1 ≈ beta_0 (contemporaneous effect), h2 ≈ beta_1 (1-period lag effect)
        betas = _get_betas(model, "x")
        assert dm["h1"] == pytest.approx(betas[0], rel=1e-6)
        assert dm["h2"] == pytest.approx(betas[1], rel=1e-6)
        # h3 = 0 (no more lags, no AR component)
        assert dm["h3"] == pytest.approx(0.0, abs=1e-10)

    def test_unknown_parm_raises(self):
        np.random.seed(42)
        n = 20
        x = np.arange(1.0, n + 1)
        y = x + np.random.randn(n)
        data = {"y": y, "x": x}
        y_out, X = formula("y ~ x", data)
        model = ALM(distribution="dnorm")
        model.fit(X, y_out.values)
        with pytest.raises(ValueError, match="not found in the model"):
            multipliers(model, "z", h=3)

    def test_no_ar_pure_dl_cumulative(self):
        """Pure DL model: cumulative effect equals sum of betas."""
        np.random.seed(3)
        n = 50
        x = np.random.randn(n)
        y = 1.0 * x + 0.4 * B(x, 1) + 0.2 * B(x, 2) + np.random.randn(n) * 0.05

        data = {"y": y, "x": x}
        y_out, X = formula("y ~ x + B(x,1) + B(x,2)", data)
        model = ALM(distribution="dnorm")
        model.fit(X, y_out.values)

        dm = multipliers(model, "x", h=4)
        betas = _get_betas(model, "x")
        # Without AR, h4 should be 0 (no lag-3 coefficient)
        assert dm["h4"] == pytest.approx(0.0, abs=1e-10)
        # Total effect = sum of betas
        total = sum(dm.values())
        assert total == pytest.approx(betas.sum(), rel=1e-6)

    def test_ari_model(self):
        """ARI(1,1) model: integrated process, multipliers grow cumulatively."""
        np.random.seed(5)
        n = 60
        x = np.random.randn(n)
        # I(1) DGP: Δy_t = 0.8 * x_t + noise → y_t = cumsum(...)
        dy = 0.8 * x + np.random.randn(n) * 0.1
        y = np.cumsum(dy)

        data = {"y": y, "x": x}
        y_out, X = formula("y ~ x", data)
        model = ALM(distribution="dnorm", orders=(1, 1, 0))
        model.fit(X, y_out.values)

        dm = multipliers(model, "x", h=5)
        values = list(dm.values())
        # With integration, later multipliers should generally be >= earlier ones
        # (permanent effect accumulates)
        assert values[-1] >= values[0] - 1e-6

    def test_default_h(self):
        np.random.seed(0)
        n = 30
        x = np.arange(1.0, n + 1)
        y = x + np.random.randn(n)
        data = {"y": y, "x": x}
        y_out, X = formula("y ~ x", data)
        model = ALM(distribution="dnorm")
        model.fit(X, y_out.values)
        dm = multipliers(model, "x")
        assert len(dm) == 10
        assert list(dm.keys()) == [f"h{i}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# multipliers() as an ALM method
# ---------------------------------------------------------------------------


class TestMultipliersMethod:
    """Verify model.multipliers() delegates correctly to the standalone."""

    def setup_method(self):
        np.random.seed(42)
        n = 30
        self.x = np.arange(1.0, n + 1)
        self.y = 2.0 * self.x + np.random.randn(n)
        data = {"y": self.y, "x": self.x}
        y_out, X = formula("y ~ x", data)
        self.model = ALM(distribution="dnorm")
        self.model.fit(X, y_out.values)

    def test_method_output_keys(self):
        dm = self.model.multipliers("x", h=5)
        assert list(dm.keys()) == ["h1", "h2", "h3", "h4", "h5"]

    def test_method_equals_standalone(self):
        dm_method = self.model.multipliers("x", h=7)
        dm_standalone = multipliers(self.model, "x", h=7)
        assert dm_method == dm_standalone

    def test_method_unknown_parm_raises(self):
        with pytest.raises(ValueError, match="not found in the model"):
            self.model.multipliers("z")

    def test_method_default_h(self):
        dm = self.model.multipliers("x")
        assert len(dm) == 10
        assert list(dm.keys()) == [f"h{i}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# B re-exported from greybox.formula
# ---------------------------------------------------------------------------


class TestBFromFormulaModule:
    def test_b_importable_from_formula(self):
        from greybox.formula import B as B_formula

        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(B_formula(x, 0), x)

    def test_b_formula_same_object(self):
        from greybox.formula import B as B_formula
        from greybox import B as B_top

        assert B_formula is B_top
