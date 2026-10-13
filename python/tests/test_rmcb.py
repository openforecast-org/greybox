"""Unit tests for rmcb() / RMCBResult (no R needed).

These cover:
- The structure of the result (mean, interval, vlines, groups, methods).
- The rank-based ("tukey", "dnorm") and regression ("dlnorm", general)
  distributions.
- Ascending ordering of the means and the default selection of the best
  (lowest-mean) method.
- ``RMCBResult.plot()`` returning usable matplotlib axes for both the "mcb"
  and "lines" styles, and rejecting an invalid ``outplot``.
- ``__str__`` containing the expected header lines.
- Input validation.

The numbers in the parity-flavoured assertions were reproduced with R's
``greybox::rmcb`` (see ``test_r_python_compare.py`` for the guarded
R-vs-Python comparison).
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from greybox import rmcb, RMCBResult


@pytest.fixture(scope="module")
def separated_data():
    """Four methods with clearly separated locations (B worst, A best)."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(50, 4))
    data[:, 1] += 4
    data[:, 2] += 3
    data[:, 3] += 2
    return pd.DataFrame(
        data,
        columns=["Method A", "Method B", "Method C - long name", "Method D"],
    )


@pytest.fixture(scope="module")
def positive_data():
    """Strictly positive data, suitable for the log-normal distribution."""
    rng = np.random.default_rng(7)
    data = np.abs(rng.normal(size=(40, 3))) + 1
    data[:, 1] += 2
    data[:, 2] += 1
    return pd.DataFrame(data, columns=["M1", "M2", "M3"])


def test_returns_rmcbresult(separated_data):
    result = rmcb(separated_data)
    assert isinstance(result, RMCBResult)
    assert result.distribution == "tukey"


def test_mean_is_sorted_ascending(separated_data):
    result = rmcb(separated_data)
    means = result.mean.to_numpy()
    assert np.all(means[:-1] <= means[1:])
    # Method A is the best (lowest mean rank) and should come first.
    assert result.mean.index[0] == "Method A"


def test_default_select_is_best(separated_data):
    result = rmcb(separated_data)
    # select is a 1-based position in the sorted order; the best is first.
    assert result.select == 1


def test_interval_structure(separated_data):
    result = rmcb(separated_data, level=0.95)
    assert list(result.interval.columns) == ["2.5%", "97.5%"]
    assert list(result.interval.index) == list(result.mean.index)
    # Lower bound below the mean, upper bound above it.
    lower = result.interval.iloc[:, 0].to_numpy()
    upper = result.interval.iloc[:, 1].to_numpy()
    assert np.all(lower <= result.mean.to_numpy())
    assert np.all(result.mean.to_numpy() <= upper)


def test_groups_and_vlines_consistency(separated_data):
    result = rmcb(separated_data)
    n_methods = len(result.mean)
    assert result.groups.shape[0] == n_methods
    assert result.groups.shape[1] == result.vlines.shape[0]
    assert list(result.vlines.columns) == ["Group starts", "Group ends"]
    # methods overlap table is square and symmetric on the diagonal True.
    assert result.methods.shape == (n_methods, n_methods)
    assert bool(np.all(np.diag(result.methods.to_numpy())))


def test_pvalue_in_unit_interval(separated_data):
    result = rmcb(separated_data)
    assert 0.0 <= result.p_value <= 1.0


def test_str_contains_header(separated_data):
    result = rmcb(separated_data)
    text = str(result)
    assert "Regression for Multiple Comparison with the Best" in text
    assert "the number of methods is 4" in text
    assert "The number of observations is 50" in text


@pytest.mark.parametrize("distribution", ["tukey", "dnorm"])
def test_rank_distributions(separated_data, distribution):
    result = rmcb(separated_data, distribution=distribution)
    assert result.distribution == distribution
    # Rank means for 4 methods lie within [1, 4].
    assert result.mean.min() >= 1.0
    assert result.mean.max() <= 4.0


def test_dlnorm_distribution(positive_data):
    result = rmcb(positive_data, distribution="dlnorm")
    assert result.distribution == "dlnorm"
    assert np.all(np.isfinite(result.mean.to_numpy()))
    # Log-normal means are exponentiated, hence strictly positive.
    assert np.all(result.mean.to_numpy() > 0)


def test_general_distribution(positive_data):
    # The general (alm-based) branch should produce a valid, finite result.
    result = rmcb(positive_data, distribution="dlaplace")
    assert result.distribution == "dlaplace"
    assert np.all(np.isfinite(result.mean.to_numpy()))
    assert 0.0 <= result.p_value <= 1.0


def test_numpy_array_input_default_names():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(30, 3))
    result = rmcb(data)
    assert set(result.mean.index) == {"Method1", "Method2", "Method3"}


def test_select_by_name(separated_data):
    result = rmcb(separated_data, select="Method B")
    # Method B is the worst, so it sits last in the sorted order.
    assert result.select == len(result.mean)


def test_invalid_data_shape():
    with pytest.raises(ValueError):
        rmcb(np.arange(10))


def test_plot_mcb_returns_axes(separated_data):
    result = rmcb(separated_data)
    ax = result.plot(outplot="mcb")
    assert ax is not None
    # x ticks: one per method.
    assert len(ax.get_xticks()) == len(result.mean)
    plt.close("all")


def test_plot_lines_returns_axes(separated_data):
    result = rmcb(separated_data)
    ax = result.plot(outplot="lines")
    assert ax is not None
    # y ticks: one per method.
    assert len(ax.get_yticks()) == len(result.mean)
    plt.close("all")


def test_plot_accepts_existing_axes(separated_data):
    result = rmcb(separated_data)
    _, ax = plt.subplots()
    returned = result.plot(outplot="mcb", ax=ax)
    assert returned is ax
    plt.close("all")


def test_plot_invalid_outplot(separated_data):
    result = rmcb(separated_data)
    with pytest.raises(ValueError):
        result.plot(outplot="bananas")
    plt.close("all")


def _ar1_matrix(phi=0.8, n=150, k=5, seed=1):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, k))
    x = np.zeros((n, k))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    x += np.arange(k) * 0.3
    return x


def test_correction_default_unchanged(separated_data):
    res = rmcb(separated_data, outplot="none")
    res_none = rmcb(separated_data, outplot="none", correction="none")
    assert res.correction == "none"
    assert res.factor == 1.0
    pd.testing.assert_frame_equal(res.interval, res_none.interval)
    assert "long-run" not in str(res)


def test_correction_hac_factor_at_least_one(separated_data):
    res = rmcb(separated_data, outplot="none", correction="hac")
    assert res.correction == "hac"
    assert res.factor >= 1.0
    assert "long-run variance factor" in str(res)


@pytest.mark.parametrize("distribution", ["tukey", "dnorm"])
def test_correction_hac_autocorrelated(distribution):
    x = _ar1_matrix()
    base = rmcb(x, distribution=distribution, outplot="none")
    hac = rmcb(x, distribution=distribution, outplot="none", correction="hac")
    assert hac.factor > 1.5
    w0 = (base.interval.iloc[:, 1] - base.interval.iloc[:, 0]).to_numpy()
    w1 = (hac.interval.iloc[:, 1] - hac.interval.iloc[:, 0]).to_numpy()
    assert np.all(w1 > w0)
    np.testing.assert_allclose(w1 / w0, np.sqrt(hac.factor), rtol=1e-8)


def test_correction_invalid_raises(separated_data):
    with pytest.raises(ValueError):
        rmcb(separated_data, correction="foo")
