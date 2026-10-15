"""Tests of formula(), the model.frame() and model.matrix() of R."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from greybox import ALM, CALM, formula
from greybox.predict import predict


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

    def test_power_of_term(self, data):
        _, X = formula("y ~ x^2 + I(x^2)", data)
        assert list(X.columns) == ["(Intercept)", "x", "I(x^2)"]
        assert_array_equal(X["x"], data["x"])
        with pytest.raises(ValueError, match="Invalid power"):
            formula("y ~ x^1", data)

    @pytest.mark.parametrize(
        "formula_str, terms",
        [
            ("y ~ (x + z)^2", ["x", "z", "x:z"]),
            ("y ~ (x + z + g)^3 - x:z:g", ["x", "z", "g", "x:z", "x:g", "z:g"]),
            ("y ~ (x + z):g", ["x:g", "z:g"]),
            ("y ~ x * (z + g)", ["x", "z", "g", "x:z", "x:g"]),
            ("y ~ (x + z - x)^2", ["z"]),
            ("y ~ x * z - (x + z)", ["x:z"]),
            ("y ~ x %in% z", ["x:z"]),
            ("y ~ x %in% (z + g)", ["x:z:g"]),
            ("y ~ (x + z) / g", ["x", "z", "x:z:g"]),
            ("y ~ x + offset(z) + offset(z):x", ["x"]),
            ("y ~ (. - g)^2", ["x", "z", "x:z"]),
        ],
    )
    def test_operators(self, data, formula_str, terms):
        assert formula(formula_str, data, return_type="terms") == terms

    def test_intercept_in_group(self, data):
        assert list(formula("y ~ (x + z - 1)", data)[1].columns) == ["x", "z"]

    def test_log_base(self, data):
        _, X = formula("y ~ log(x, 2) + log(z,base=10) + log(x, 3)", data)
        assert list(X.columns)[1:] == ["log(x, 2)", "log(z, base = 10)", "log(x, 3)"]
        assert_array_equal(X["log(x, 2)"], np.log2(data["x"]))
        assert_array_equal(X["log(z, base = 10)"], np.log10(data["z"]))
        assert_allclose(X["log(x, 3)"], np.log(data["x"]) / np.log(3))

    def test_logical(self, data):
        data["l"] = [True, False, True, True, None, False]
        _, X = formula("y ~ l", data)
        assert list(X.columns) == ["(Intercept)", "lTRUE"]
        assert_array_equal(X["lTRUE"], [1, 0, 1, 1, np.nan, 0])
        assert list(formula("y ~ 0 + l", data)[1].columns) == ["lFALSE", "lTRUE"]

    def test_logical_one_value(self, data):
        _, X = formula("y ~ l", data.assign(l=True))
        assert list(X.columns) == ["(Intercept)", "lTRUE"]


class TestTrainingLevels:
    @pytest.fixture
    def model(self, data):
        data["l"] = [True, False, True, True, False, False]
        y, X = formula("y ~ x + g + l", data)
        return ALM(distribution="dnorm").fit(X, y, formula="y ~ x + g + l")

    def test_levels(self, data):
        X = formula("y ~ x * g", data, return_type="X")
        assert X.attrs["_xlevels"] == {"g": ["a", "b", "c"]}

    @pytest.mark.parametrize("levels", [["c", "b"], ["b", "b", "a"]])
    def test_new_data(self, data, levels):
        newdata = pd.DataFrame({"x": 1.0, "g": levels})
        newdata.attrs["_xlevels"] = {"g": ["a", "b", "c"]}
        X = formula("~ x + g", newdata, return_type="X")
        assert list(X.columns) == ["(Intercept)", "x", "gb", "gc"]
        assert_array_equal(X["gb"], [level == "b" for level in levels])

    def test_unseen_level(self, data):
        newdata = pd.DataFrame({"x": [1.0, 2.0], "g": ["a", "d"]})
        newdata.attrs["_xlevels"] = {"g": ["a", "b", "c"]}
        with pytest.raises(ValueError, match="factor g has new levels d"):
            formula("~ x + g", newdata)

    def test_predict(self, model):
        newdata = {"x": [1.0, 2.0], "g": ["c", "b"], "l": [True, True]}
        X = np.array([[1, 1.0, 0, 1, 1], [1, 2.0, 1, 0, 1]])
        assert model._xlevels == {"g": ["a", "b", "c"], "l": [False, True]}
        assert_allclose(predict(model, newdata).mean, model.predict(X).mean)
        with pytest.raises(ValueError, match="factor g has new levels d"):
            predict(model, {"x": [1.0], "g": ["d"], "l": [True]})

    def test_predict_trend(self, data):
        y, X = formula("y ~ x + trend", data)
        model = ALM(distribution="dnorm").fit(X, y, formula="y ~ x + trend")
        X_new = np.array([[1, 3.0, 7], [1, 4.0, 8]])
        assert_allclose(
            predict(model, {"x": [3.0, 4.0]}).mean, model.predict(X_new).mean
        )

    def test_calm_predict(self):
        rng = np.random.default_rng(1)
        data = pd.DataFrame(
            {
                "x": rng.normal(size=30),
                "g": rng.choice(["a", "b", "c"], 30),
                "l": rng.choice([True, False], 30),
            }
        )
        y = data["x"] + 2 * data["l"] + (data["g"] == "b") + rng.normal(size=30)
        model = CALM(data.assign(y=y)[["y", "x", "g", "l"]], bruteforce=True)
        newdata = {"x": [1.0, 2.0], "g": ["c", "b"], "l": [True, False]}
        X = np.array([[1, 1.0, 0, 1, 1], [1, 2.0, 1, 0, 0]])
        assert model.coefficient_names == ["(Intercept)", "x", "gb", "gc", "lTRUE"]
        assert_allclose(model.predict(newdata).mean, model.predict(X).mean)
