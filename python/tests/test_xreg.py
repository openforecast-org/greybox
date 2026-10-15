"""Tests for greybox.xreg module."""

import numpy as np
import pandas as pd
import pytest

from greybox.xreg import (
    temporal_dummy,
    xreg_expander,
    xreg_multiplier,
    xreg_transformer,
)


class TestXregExpander:
    def test_lag_minus_one(self):
        x = np.array([10, 20, 30, 40, 50], dtype=float)
        result = xreg_expander(x, lags=[-1])
        # col 0 = original, col 1 = lag-1
        assert result.shape == (5, 2)
        np.testing.assert_array_equal(result[:, 0], x)
        # lag=-1 shifts forward: row[1]=x[0], row[2]=x[1], ...
        np.testing.assert_array_equal(result[1:, 1], [10, 20, 30, 40])

    def test_lead_plus_one(self):
        x = np.array([10, 20, 30, 40, 50], dtype=float)
        result = xreg_expander(x, lags=[1])
        # col 0 = original, col 1 = lead-1
        assert result.shape == (5, 2)
        np.testing.assert_array_equal(result[:, 0], x)
        # lead=1: row[0]=x[1], row[1]=x[2], ...
        np.testing.assert_array_equal(result[:4, 1], [20, 30, 40, 50])

    def test_multiple_lags(self):
        x = np.array([10, 20, 30, 40, 50], dtype=float)
        result = xreg_expander(x, lags=[-1, -2])
        # col 0 = original, cols 1-2 = lag-1, lag-2
        assert result.shape == (5, 3)

    def test_zero_lag_only_returns_original(self):
        x = np.array([10, 20, 30], dtype=float)
        with pytest.warns(UserWarning, match="leads or lags"):
            result = xreg_expander(x, lags=[0])
        # lag=0 is removed, so returns original only
        assert result.shape == (3, 1)

    def test_gaps_zero(self):
        x = np.array([10, 20, 30, 40, 50], dtype=float)
        result = xreg_expander(x, lags=[-1], gaps="zero")
        # col 1 = lag-1; first row gap filled with 0
        assert result[0, 1] == 0.0

    def test_gaps_naive(self):
        x = np.array([10, 20, 30, 40, 50], dtype=float)
        result = xreg_expander(x, lags=[-1], gaps="naive")
        # col 1 = lag-1; first row should be filled with first valid value (10)
        assert result[0, 1] == 10.0

    def test_matrix_input(self):
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        result = xreg_expander(x, lags=[-1])
        # 2 original cols + 2 lag cols = 4 cols
        assert result.shape == (4, 4)

    def test_matrix_layout(self):
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        result = xreg_expander(x, lags=[1, -1], gaps="naive")
        # Each variable followed by its lags, then its leads, as R
        np.testing.assert_array_equal(result[:, 0], x[:, 0])
        np.testing.assert_array_equal(result[:, 1], [1, 1, 3, 5])
        np.testing.assert_array_equal(result[:, 2], [3, 5, 7, 7])
        np.testing.assert_array_equal(result[:, 3], x[:, 1])
        np.testing.assert_array_equal(result[:, 4], [2, 2, 4, 6])
        np.testing.assert_array_equal(result[:, 5], [4, 6, 8, 8])


class TestXregFrames:
    def test_expander(self):
        x = pd.DataFrame({"a": [1.0, 2, 3]}, index=list("pqr"))
        result = xreg_expander(x, lags=[1, -1], gaps="naive")
        assert list(result.columns) == ["a", "aLag1", "aLead1"]
        assert list(result.index) == list("pqr")
        np.testing.assert_array_equal(result["aLag1"], [1, 1, 2])

    def test_expander_zero_lags(self):
        x = pd.Series([1.0, 2, 3])
        with pytest.warns(UserWarning, match="leads or lags"):
            result = xreg_expander(x, lags=[0])
        assert list(result.columns) == ["x"]

    def test_transformer_series(self):
        result = xreg_transformer(pd.Series([1.0, 4.0]), functions=["sqrt"])
        assert list(result.columns) == ["x", "sqrt_x"]
        np.testing.assert_array_equal(result["sqrt_x"], [1, 2])

    def test_multiplier(self):
        x = pd.DataFrame({"a": [1.0, 2], "b": [3.0, 4], "c": [5.0, 6]})
        result = xreg_multiplier(x)
        assert list(result.columns) == ["a", "b", "c", "a_b", "a_c", "b_c"]
        np.testing.assert_array_equal(result["b_c"], [15, 24])

    def test_numpy_stays_ndarray(self):
        x = np.array([[1.0, 2], [3, 4]])
        assert isinstance(xreg_expander(x, lags=[-1], gaps="zero"), np.ndarray)
        assert isinstance(xreg_transformer(x, functions=["log"]), np.ndarray)
        assert isinstance(xreg_multiplier(x), np.ndarray)


class TestXregTransformer:
    def test_layout(self):
        x = np.array([[1, 3], [2, 4]], dtype=float)
        result = xreg_transformer(x, functions=["log", "square"])
        np.testing.assert_allclose(
            result,
            [[1, 0, 1, 3, np.log(3), 9], [2, np.log(2), 4, 4, np.log(4), 16]],
        )

    def test_no_clipping(self):
        x = np.array([0.0, -1.0])
        with pytest.warns(RuntimeWarning):
            result = xreg_transformer(x, functions=["log", "inv", "sqrt"])
        np.testing.assert_array_equal(result[0], [0, -np.inf, np.inf, 0])
        assert np.isnan(result[1, 1]) and np.isnan(result[1, 3])
        assert result[1, 2] == -1

    def test_unknown_function_warns(self):
        with pytest.warns(UserWarning, match="unknown function"):
            result = xreg_transformer(np.array([1.0, 2.0]), functions=["log", "cube"])
        assert result.shape == (2, 2)


class TestXregMultiplier:
    def test_basic(self):
        x = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        result = xreg_multiplier(x)
        # Should have original cols + 1 cross-product
        assert result.shape == (3, 3)
        # Cross product: col0 * col1
        np.testing.assert_array_equal(result[:, 2], [2, 12, 30])

    def test_three_cols(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        result = xreg_multiplier(x)
        # 3 original + C(3,2)=3 combinations = 6
        assert result.shape == (2, 6)
        # Originals preserved
        np.testing.assert_array_equal(result[:, 0], [1, 4])
        np.testing.assert_array_equal(result[:, 1], [2, 5])
        np.testing.assert_array_equal(result[:, 2], [3, 6])

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            xreg_multiplier(np.array([1, 2, 3]))

    def test_single_col_raises(self):
        with pytest.raises(ValueError):
            xreg_multiplier(np.array([[1], [2], [3]]))


class TestTemporalDummy:
    def test_monthly_shape(self):
        result = temporal_dummy(np.arange(24))
        assert result.shape == (24, 12)

    def test_monthly_one_hot(self):
        result = temporal_dummy(np.arange(12), type="month", of="year")
        np.testing.assert_array_equal(result, np.eye(12))

    def test_quarterly(self):
        result = temporal_dummy(np.arange(8), type="quarter")
        np.testing.assert_array_equal(result, np.eye(4)[[0, 1, 2, 3, 0, 1, 2, 3]])

    def test_weekly(self):
        result = temporal_dummy(np.arange(14), type="day", of="week")
        assert result.shape == (14, 7)

    def test_horizon(self):
        result = temporal_dummy(np.arange(6), type="quarter", h=4)
        assert result.shape == (10, 4)

    def test_cycling(self):
        result = temporal_dummy(np.arange(5), type="month", of="quarter")
        np.testing.assert_array_equal(result, np.eye(3)[[0, 1, 2, 0, 1]])

    def test_freq_deprecated(self):
        with pytest.warns(DeprecationWarning, match="freq is deprecated"):
            result = temporal_dummy(np.arange(6), freq=4, h=2)
        np.testing.assert_array_equal(
            result, temporal_dummy(np.arange(6), "quarter", h=2)
        )

    def test_freq_unmapped(self):
        with pytest.warns(DeprecationWarning, match="freq is deprecated"):
            result = temporal_dummy(pd.Series(np.arange(5.0)), freq=3)
        np.testing.assert_array_equal(result, np.eye(3)[[0, 1, 2, 0, 1]])

    def test_series_names_and_index(self):
        x = pd.Series(np.arange(5.0), index=pd.RangeIndex(10, 20, 2))
        result = temporal_dummy(x, type="quarter", h=2)
        assert list(result.columns) == [f"quarter{i}ofyear" for i in range(1, 5)]
        assert result.index.equals(pd.RangeIndex(10, 24, 2))

    def test_dates(self):
        dates = pd.date_range("2020-02-27", periods=4, freq="D")
        result = temporal_dummy(pd.Series(dates, index=list("abcd")), "day", "month")
        assert list(result.columns) == ["day1ofmonth", "day27ofmonth"] + [
            "day28ofmonth",
            "day29ofmonth",
        ]
        assert list(result.index) == list("abcd")
        np.testing.assert_array_equal(result.to_numpy(), np.eye(4)[[1, 2, 3, 0]])

    def test_factors(self):
        dates = np.datetime64("2020-03-30") + np.arange(4)
        result = temporal_dummy(dates, type="month", of="year", factors=True)
        assert isinstance(result, pd.Categorical)
        assert list(result) == ["Mar", "Mar", "Apr", "Apr"]
        assert list(result.categories) == ["Apr", "Mar"]
        index = pd.DatetimeIndex(dates)
        result = temporal_dummy(index, type="quarter", factors=True, h=1)
        assert isinstance(result.dtype, pd.CategoricalDtype)
        assert list(result) == ["Q1", "Q1", "Q2", "Q2", "Q2"]
        assert result.index[-1] == pd.Timestamp("2020-04-03")

    def test_hours_of_dates(self):
        with pytest.warns(UserWarning, match="Date class does not support"):
            result = temporal_dummy(np.datetime64("2020-03-30") + np.arange(4), "hour")
        assert result is None

    def test_hours(self):
        times = pd.date_range("2020-03-31 22:00", periods=3, freq="h", tz="Asia/Tokyo")
        result = temporal_dummy(times, type="hour", of="quarter", factors=True)
        assert list(result) == [91 * 24 - 1, 91 * 24, 1]

    def test_invalid(self):
        with pytest.raises(ValueError):
            temporal_dummy(np.arange(3), type="decade")
        with pytest.raises(ValueError):
            temporal_dummy(pd.date_range("2020-01-01", periods=3), "day", "day")
