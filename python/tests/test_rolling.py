"""Tests for rolling_origin() — R parity + flexible call API."""

import warnings
import numpy as np
import pytest

from greybox import rolling_origin, RollingOriginResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _naive_call(data, h):
    """Forecast: repeat last observed value h times."""
    return np.full(h, data[-1])


def _mean_call(data, h):
    return np.full(h, data.mean())


def _rng(seed=42, n=100):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n))


# ---------------------------------------------------------------------------
# 1. obs_in_sample formula
# ---------------------------------------------------------------------------


class TestObsInSample:
    """Verify initial window size from R formula."""

    def test_co_true(self):
        # obs_in_sample = n - (origins*step + (h-step)*co)
        # n=100, h=5, origins=10, step=1, co=True → 100 - (10 + 4) = 86
        y = _rng(n=100)
        sizes = []

        def call(data, h, counti):
            sizes.append(len(counti))
            return np.full(h, data.mean())

        rolling_origin(y, h=5, origins=10, step=1, ci=False, co=True, call=call)
        # origin 0: train size = obs_in_sample = 86
        assert sizes[0] == 86

    def test_co_false(self):
        # n=100, h=5, origins=10, step=1, co=False → 100 - (10 + 4*0) = 90
        y = _rng(n=100)
        sizes = []

        def call(data, h, counti):
            sizes.append(len(counti))
            return np.full(h, data.mean())

        rolling_origin(y, h=5, origins=10, step=1, ci=False, co=False, call=call)
        assert sizes[0] == 90


# ---------------------------------------------------------------------------
# 2. Expanding window (ci=False)
# ---------------------------------------------------------------------------


class TestExpandingWindow:
    def test_train_grows_by_step(self):
        y = _rng(n=100)
        sizes = []

        def call(data, h, counti):
            sizes.append(len(counti))
            return np.full(h, data.mean())

        rolling_origin(y, h=5, origins=10, step=1, ci=False, co=True, call=call)
        # Each origin adds step=1
        for i in range(1, len(sizes)):
            assert sizes[i] == sizes[i - 1] + 1

    def test_step_2_train_grows_by_2(self):
        y = _rng(n=120)
        sizes = []

        def call(data, h, counti):
            sizes.append(len(counti))
            return np.full(h, data.mean())

        rolling_origin(y, h=5, origins=8, step=2, ci=False, co=True, call=call)
        for i in range(1, len(sizes)):
            assert sizes[i] == sizes[i - 1] + 2


# ---------------------------------------------------------------------------
# 3. Sliding window (ci=True)
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    def test_train_size_constant(self):
        y = _rng(n=100)
        sizes = []

        def call(data, h, counti):
            sizes.append(len(counti))
            return np.full(h, data.mean())

        rolling_origin(y, h=5, origins=10, step=1, ci=True, co=True, call=call)
        # All training windows should be the same size
        assert len(set(sizes)) == 1

    def test_window_slides_by_step(self):
        y = _rng(n=100)
        starts = []

        def call(data, h, counti):
            starts.append(counti[0])
            return np.full(h, data.mean())

        rolling_origin(y, h=5, origins=5, step=1, ci=True, co=True, call=call)
        for i in range(1, len(starts)):
            assert starts[i] == starts[i - 1] + 1


# ---------------------------------------------------------------------------
# 4. Constant holdout (co=True)
# ---------------------------------------------------------------------------


class TestCoTrue:
    def test_holdout_no_nan(self):
        y = _rng(n=100)
        result = rolling_origin(y, h=5, origins=10, co=True, call=_mean_call)
        assert result.holdout.shape == (5, 10)
        assert not np.any(np.isnan(result.holdout))

    def test_mean_shape(self):
        y = _rng(n=100)
        result = rolling_origin(y, h=5, origins=10, co=True, call=_mean_call)
        assert result.mean.shape == (5, 10)

    def test_holdout_values_correct(self):
        """origin 0 holdout must equal data[obs_in_sample : obs_in_sample+h]."""
        y = _rng(n=100)
        # n=100, h=5, origins=10, step=1, co=True → obs_in_sample=86
        result = rolling_origin(y, h=5, origins=10, step=1, co=True, call=_mean_call)
        expected = y[86:91]
        np.testing.assert_array_equal(result.holdout[:, 0], expected)


# ---------------------------------------------------------------------------
# 5. Decreasing holdout (co=False)
# ---------------------------------------------------------------------------


class TestCoFalse:
    def test_late_origins_have_nan(self):
        """With co=False, origins near end forecast fewer than h steps."""
        y = _rng(n=100)
        # n=100, h=5, origins=10, step=1, co=False → obs_in_sample=90
        # origin 8: h_actual=min(5,1*2)=2 → rows 2..4 should be NaN
        # origin 9: h_actual=min(5,1*1)=1 → rows 1..4 should be NaN
        result = rolling_origin(y, h=5, origins=10, step=1, co=False, call=_mean_call)
        # Last origin (i=9) should have NaN beyond h_actual=1
        assert np.all(np.isnan(result.holdout[1:, -1]))

    def test_early_origins_full(self):
        y = _rng(n=100)
        result = rolling_origin(y, h=5, origins=10, step=1, co=False, call=_mean_call)
        # Origin 0: h_actual=min(5,10)=5, all filled
        assert not np.any(np.isnan(result.holdout[:, 0]))

    def test_str_says_decreasing(self):
        y = _rng(n=100)
        result = rolling_origin(y, h=5, origins=10, step=1, co=False, call=_mean_call)
        assert "decreasing" in str(result)


# ---------------------------------------------------------------------------
# 6. call returns array
# ---------------------------------------------------------------------------


class TestCallReturnsArray:
    def test_mean_shape(self):
        y = _rng(n=100)
        result = rolling_origin(y, h=5, origins=10, call=_naive_call)
        assert result.mean.shape == (5, 10)
        assert "mean" in result._fields
        assert not hasattr(result, "lower") or result.lower is None

    def test_forecast_values(self):
        y = np.arange(100, dtype=float)
        result = rolling_origin(
            y, h=3, origins=5, co=True, call=lambda data, h: np.full(h, data[-1])
        )
        # Each column: forecast = last training value
        # obs_in_sample = 100 - (5*1 + 2*1) = 93
        # origin 0 trains on [0..92], last value = 92.0
        assert result.mean[0, 0] == pytest.approx(92.0)


# ---------------------------------------------------------------------------
# 7. call returns dict
# ---------------------------------------------------------------------------


class TestCallReturnsDict:
    def test_lower_upper_present(self):
        y = _rng(n=100)

        def call_with_intervals(data, h):
            m = np.full(h, data.mean())
            s = data.std()
            return {"mean": m, "lower": m - 1.96 * s, "upper": m + 1.96 * s}

        result = rolling_origin(y, h=5, origins=10, call=call_with_intervals)
        assert hasattr(result, "lower")
        assert hasattr(result, "upper")
        assert result.lower.shape == (5, 10)
        assert result.upper.shape == (5, 10)
        assert set(result._fields) == {"mean", "lower", "upper"}


# ---------------------------------------------------------------------------
# 8. call with counti / counto kwargs
# ---------------------------------------------------------------------------


class TestCallWithCounti:
    def test_counti_injected(self):
        """counti contains the correct indices into the original series."""
        y = np.arange(100, dtype=float)
        counti_received = []

        def call(data, h, counti):
            counti_received.append(counti.copy())
            return np.full(h, data.mean())

        rolling_origin(y, h=5, origins=3, step=1, ci=False, co=True, call=call)
        # obs_in_sample = 100 - (3 + 4) = 93
        # origin 0: counti = [0, 1, ..., 92]
        assert counti_received[0][0] == 0
        assert counti_received[0][-1] == 92

    def test_counto_injected(self):
        y = np.arange(100, dtype=float)
        counto_received = []

        def call(data, h, counto):
            counto_received.append(counto.copy())
            return np.full(h, data.mean())

        rolling_origin(y, h=5, origins=3, step=1, co=True, call=call)
        # origin 0 holdout starts at obs_in_sample=93
        assert counto_received[0][0] == 93
        assert len(counto_received[0]) == 5

    def test_data_equals_y_at_counti(self):
        y = np.arange(100, dtype=float) * 2
        matches = []

        def call(data, h, counti):
            matches.append(np.allclose(data, y[counti]))
            return np.full(h, data.mean())

        rolling_origin(y, h=5, origins=5, call=call)
        assert all(matches)


# ---------------------------------------------------------------------------
# 9. step > 1
# ---------------------------------------------------------------------------


class TestStepGt1:
    def test_window_boundaries_step2(self):
        y = _rng(n=120)
        # n=120, h=5, origins=8, step=2, co=True
        # obs_in_sample = 120 - (8*2 + (5-2)*1) = 120 - 19 = 101
        starts = []
        ends = []

        def call(data, h, counti):
            starts.append(counti[0])
            ends.append(counti[-1])
            return np.full(h, data.mean())

        rolling_origin(y, h=5, origins=8, step=2, ci=False, co=True, call=call)
        assert ends[0] == 100  # obs_in_sample - 1 = 100
        assert ends[1] == 102  # obs_in_sample + step - 1 = 102
        assert starts[0] == 0


# ---------------------------------------------------------------------------
# 10. Backward compatibility (model_fn + predict_fn)
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_deprecation_warning(self):
        y = _rng(n=50)
        with pytest.warns(
            DeprecationWarning, match="model_fn and predict_fn are deprecated"
        ):
            result = rolling_origin(
                y,
                h=3,
                origins=5,
                model_fn=lambda d: d,
                predict_fn=lambda m, h: np.full(h, m.mean()),
            )
        assert result.mean.shape == (3, 5)

    def test_produces_correct_result(self):
        y = np.arange(80, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = rolling_origin(
                y,
                h=4,
                origins=6,
                model_fn=lambda d: d,
                predict_fn=lambda m, h: np.full(h, m.mean()),
            )
        assert result.origins == 6
        assert result.mean.shape == (4, 6)


# ---------------------------------------------------------------------------
# 11. __str__ / __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_constant_holdout_str(self):
        y = _rng(n=100)
        result = rolling_origin(y, h=5, origins=10, co=True, call=_mean_call)
        s = str(result)
        assert "constant" in s
        assert "Forecast horizon: 5" in s
        assert "Number of origins: 10" in s

    def test_repr_equals_str(self):
        y = _rng(n=100)
        result = rolling_origin(y, h=5, origins=10, call=_mean_call)
        assert repr(result) == str(result)


# ---------------------------------------------------------------------------
# 12. Defaults match R
# ---------------------------------------------------------------------------


class TestDefaultsMatchR:
    def test_default_h_origins_co(self):
        """Default h=10, origins=10, co=True (matches R's ro() defaults)."""
        y = _rng(n=200)
        calls = []

        def call(data, h):
            calls.append(h)
            return np.full(h, data.mean())

        result = rolling_origin(y, call=call)
        assert result.h == 10
        assert result.origins == 10
        # co=True → no NaN in holdout
        assert not np.any(np.isnan(result.holdout))
        # all calls requested h=10
        assert all(c == 10 for c in calls)


# ---------------------------------------------------------------------------
# 13. Matrix output and errors, as R
# ---------------------------------------------------------------------------


class TestMatrixAndErrors:
    def test_matrix_output_is_array(self):
        """An (h, k) matrix from the call gives an (h, origins, k) array."""
        y = _rng(n=50)

        def call(data, h):
            return np.column_stack([np.full(h, data.mean()), np.full(h, data[-1])])

        result = rolling_origin(y, h=4, origins=5, call=call, co=False)
        assert result.mean.shape == (4, 5, 2)
        assert np.all(np.isnan(result.mean[1:, -1, :]))
        np.testing.assert_allclose(result.mean[0, 0, 1], y[50 - 5 - 1])

    def test_one_column_matrix_is_vector(self):
        y = _rng(n=50)
        result = rolling_origin(
            y, h=3, origins=4, call=lambda data, h: np.full((h, 1), data.mean())
        )
        assert result.mean.shape == (3, 4)

    def test_call_error_raises(self):
        """A failing call is an error, not fewer origins, as R."""
        y = _rng(n=50)

        def call(data, h):
            if len(data) > 42:
                raise RuntimeError("model failed")
            return np.full(h, data.mean())

        with pytest.raises(RuntimeError, match="model failed"):
            rolling_origin(y, h=3, origins=5, call=call)
