"""Tests of formula(), the model.frame() and model.matrix() of R."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from greybox import formula


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "y": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
            "x": [2.0, 7.0, 1.0, 8.0, 2.0, 8.0],
            "z": [1.0, 4.0, 1.0, 4.0, 2.0, 1.0],
            "g": ["b", "a", "c", "a", "b", "c"],
        }
    )


class TestFormula:
    def test_dot(self, data):
        y, X = formula("y ~ .", data)
        assert list(X.columns) == ["(Intercept)", "x", "z", "gb", "gc"]
        assert_array_equal(y, data["y"])

    def test_dot_integer_names(self):
        frame = pd.DataFrame(np.arange(12.0).reshape(4, 3)).assign(y=[1, 2, 3, 5])
        _, X = formula("y ~ .", frame)
        assert list(X.columns) == ["(Intercept)", "0", "1", "2"]
        assert_array_equal(X["1"], frame[1])

    @pytest.mark.parametrize(
        "formula_str", ["y ~ 0 + x", "y ~ x - 1", "y ~ x + 0", "y ~ -1 + x"]
    )
    def test_no_intercept(self, data, formula_str):
        assert list(formula(formula_str, data)[1].columns) == ["x"]

    def test_intercept_back(self, data):
        assert list(formula("y ~ 0 + x + 1", data)[1].columns) == ["(Intercept)", "x"]

    def test_no_terms(self, data):
        _, X = formula("y ~ 0", data)
        assert X.shape == (6, 0)

    def test_factor_dummies(self, data):
        _, X = formula("y ~ g", data)
        assert list(X.columns) == ["(Intercept)", "gb", "gc"]
        assert_array_equal(X["gb"], [1, 0, 0, 0, 1, 0])
        assert_array_equal(X["gc"], [0, 0, 1, 0, 0, 1])

    def test_factor_without_intercept(self, data):
        _, X = formula("y ~ 0 + g", data)
        assert list(X.columns) == ["ga", "gb", "gc"]

    def test_categorical_levels(self, data):
        data["g"] = pd.Categorical(data["g"], categories=["c", "b", "a"], ordered=True)
        assert list(formula("y ~ g", data)[1].columns) == ["(Intercept)", "gb", "ga"]

    def test_float_levels(self, data):
        data["g"] = pd.Categorical([4.0, 6.0, 8.0, 4.0, 6.0, 8.5])
        _, X = formula("y ~ g", data)
        assert list(X.columns) == ["(Intercept)", "g6", "g8", "g8.5"]

    def test_factor_in_dict(self):
        data = {"y": [1.0, 2.0, 3.0], "g": np.array(["u", "v", "u"], dtype=object)}
        assert list(formula("y ~ g", data)[1].columns) == ["(Intercept)", "gv"]

    def test_factor_missing_value(self):
        data = {"y": [1.0, 2.0, 3.0, 4.0], "g": ["u", None, "v", "u"]}
        _, X = formula("y ~ g", data)
        assert_array_equal(X["gv"], [0, np.nan, 1, 0])

    def test_interactions(self, data):
        _, X = formula("y ~ x * g", data)
        assert list(X.columns) == ["(Intercept)", "x", "gb", "gc", "x:gb", "x:gc"]
        assert_array_equal(X["x:gc"], data["x"] * (data["g"] == "c"))
        _, X = formula("y ~ z:x", data)
        assert list(X.columns) == ["(Intercept)", "z:x"]
        assert_array_equal(X["z:x"], data["x"] * data["z"])

    def test_terms(self, data):
        terms = formula("y ~ x:z + g + x * z - z", data, return_type="terms")
        assert terms == ["g", "x", "x:z"]

    def test_protected_expressions(self, data):
        _, X = formula("y ~ I(x*z) + I(x - 2*z) + I(x^-1) + exp(x)", data)
        assert list(X.columns) == [
            "(Intercept)",
            "I(x * z)",
            "I(x - 2 * z)",
            "I(x^-1)",
            "exp(x)",
        ]
        assert_allclose(X["I(x - 2 * z)"], data["x"] - 2 * data["z"])
        assert_allclose(X["I(x^-1)"], 1 / data["x"])

    def test_dotted_names(self):
        data = {"y.1": [1.0, 2.0, 3.0], "x.1": [1.0, 4.0, 9.0]}
        y, X = formula("log(y.1) ~ sqrt(x.1)", data)
        assert y.name == "log(y.1)"
        assert_allclose(X["sqrt(x.1)"], [1, 2, 3])

    def test_trend_interaction(self, data):
        _, X = formula("y ~ x:trend", data)
        assert_array_equal(X["x:trend"], data["x"] * np.arange(1, 7))

    def test_unknown_variable(self, data):
        with pytest.raises(ValueError, match="Variable 'w' not found in data"):
            formula("y ~ log(w)", data)

    def test_return_types(self, data):
        y, X = formula("y ~ x + g", data, as_dataframe=False)
        assert isinstance(y, np.ndarray) and isinstance(X, np.ndarray)
        assert X.shape == (6, 4)
        assert formula("y ~ x + g", data, return_type="variables") == ["x", "gb", "gc"]
        assert_array_equal(formula("log(y) ~ x", data, return_type="y"), np.log(data.y))
