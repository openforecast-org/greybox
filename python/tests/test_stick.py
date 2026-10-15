"""Unit tests for stick() / StickResult (no R needed).

These cover:
- The STI variance decomposition against values computed with R's
  ``greybox::stick`` (the ANOVA / strength reference values are hard-coded).
- The structure of the result (lags, ANOVA table, strength Series).
- Multiple seasonal lags and the trend based on the longest lag.
- ``StickResult.plot()`` returning usable matplotlib axes, the ``which``
  selector, and the out-of-range warning.
- Input validation and edge cases.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless testing

import matplotlib.pyplot as plt
import numpy as np
import pytest

from greybox import stick, StickResult


@pytest.fixture(scope="module")
def deterministic_series():
    """An exactly additive trend + seasonal series (irregular == 0)."""
    t = np.arange(1, 49)
    return 10 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 12)


def test_returns_stickresult(deterministic_series):
    result = stick(deterministic_series, lags=12)
    assert isinstance(result, StickResult)
    assert list(result.lags) == [12]


def test_strength_matches_r(deterministic_series):
    # Reference values from R: greybox::stick(y, lags=12)$strength
    result = stick(deterministic_series, lags=12)
    assert list(result.strength.index) == ["seasonal12", "trend", "irregular"]
    np.testing.assert_allclose(
        result.strength.to_numpy(),
        [0.1202181, 0.8797819, 0.0],
        atol=1e-6,
    )


def test_strengths_sum_to_one(deterministic_series):
    result = stick(deterministic_series, lags=12)
    assert result.strength.sum() == pytest.approx(1.0)


def test_anova_table_structure(deterministic_series):
    result = stick(deterministic_series, lags=12)
    anova = result.anova
    assert list(anova.index) == ["seasonal12", "trend", "Residuals"]
    assert list(anova.columns) == ["Df", "Sum Sq", "Mean Sq", "F value", "Pr(>F)"]
    # 48 obs: 11 df seasonal (12 levels), 3 df trend (4 cycles), 33 residual.
    np.testing.assert_array_equal(anova["Df"].to_numpy(), [11, 3, 33])
    # Sum Sq reference values from R.
    np.testing.assert_allclose(
        anova["Sum Sq"].to_numpy(), [295.1539, 2160.0, 0.0], atol=1e-3
    )


def test_multiple_lags():
    rng = np.random.default_rng(1)
    t = np.arange(1, 721)
    y = (
        50
        + 10 * np.sin(2 * np.pi * t / 24)
        + 5 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 2, 720)
    )
    result = stick(y, lags=[24, 168])
    assert list(result.lags) == [24, 168]
    assert list(result.strength.index) == [
        "seasonal24",
        "seasonal168",
        "trend",
        "irregular",
    ]
    assert result.strength.sum() == pytest.approx(1.0)
    # The dominant 24 cycle should carry the most seasonal strength.
    assert result.strength["seasonal24"] > result.strength["seasonal168"]


def test_lags_deduplicated_and_sorted():
    y = np.arange(1, 49, dtype=float)
    result = stick(y, lags=[12, 12, 6])
    assert list(result.lags) == [6, 12]


def test_lags_below_two_dropped():
    y = np.arange(1, 49, dtype=float)
    with pytest.raises(ValueError, match="greater than one"):
        stick(y, lags=1)


def test_lags_required():
    y = np.arange(1, 49, dtype=float)
    with pytest.raises(ValueError, match="must be provided"):
        stick(y)


def test_too_long_lag_warns_and_drops():
    y = np.arange(1, 49, dtype=float)
    with pytest.warns(UserWarning, match="not smaller than"):
        result = stick(y, lags=[12, 200])
    assert list(result.lags) == [12]


def test_plot_all(deterministic_series):
    result = stick(deterministic_series, lags=12)
    axes = result.plot()
    # One seasonal plot + one trend plot.
    assert len(axes) == 2
    assert axes[0].get_title() == "Seasonal plot (lag 12)"
    assert axes[1].get_title() == "Trend plot"
    plt.close("all")


def test_plot_which_subset(deterministic_series):
    result = stick(deterministic_series, lags=12)
    axes = result.plot(which=2)
    assert len(axes) == 1
    assert axes[0].get_title() == "Trend plot"
    plt.close("all")


def test_plot_on_supplied_axes(deterministic_series):
    result = stick(deterministic_series, lags=12)
    fig, axs = plt.subplots(1, 2)
    returned = result.plot(axes=axs)
    assert returned is axs
    plt.close(fig)


def test_plot_out_of_range_warns(deterministic_series):
    result = stick(deterministic_series, lags=12)
    with pytest.warns(UserWarning, match="outside the range"):
        result.plot(which=[1, 2, 5])
    plt.close("all")


def test_nested_lags_degrees_of_freedom():
    # R's aov: seasonal4 is nested in seasonal12, which only adds 8 Df
    rng = np.random.default_rng(3)
    y = 100 + 0.3 * np.arange(50) + rng.normal(size=50)
    result = stick(y, lags=[4, 12])
    np.testing.assert_array_equal(result.anova["Df"].to_numpy(), [3, 8, 4, 34])
    assert result.strength.sum() == pytest.approx(1.0)


def test_missing_values_dropped():
    # As R's aov, the observations with missing values are dropped
    rng = np.random.default_rng(3)
    y = 100 + 0.3 * np.arange(50) + rng.normal(size=50)
    y[[6, 19]] = np.nan
    result = stick(y, lags=12)
    assert np.isfinite(result.strength.to_numpy()).all()
    np.testing.assert_array_equal(result.anova["Df"].to_numpy(), [11, 4, 32])
    complete = stick(np.delete(y, [6, 19]), lags=12)
    assert result.anova["Sum Sq"].sum() == pytest.approx(complete.anova["Sum Sq"].sum())
