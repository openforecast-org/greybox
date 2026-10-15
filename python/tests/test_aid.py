"""Unit tests for AidResult / AidCatResult helpers (no R needed).

These cover:
- R-style attribute access on the nested ``type`` and ``stockouts``
  dataclasses (with dict-style ``[...]`` backward compatibility).
- ``AidResult.plot()`` and ``AidCatResult.plot()`` returning a usable
  ``matplotlib.axes.Axes`` with the expected overlays.
"""

import dataclasses

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless testing

import matplotlib.pyplot as plt
import numpy as np
import pytest

from greybox import aid, aid_cat, AidType, Stockouts


@pytest.fixture(scope="module")
def stockout_result():
    """A series with one injected stockout window at 1-based positions 41..50."""
    rng = np.random.default_rng(0)
    y = rng.poisson(3, 100).astype(float)
    y[40:50] = 0
    return aid(y)


@pytest.fixture(scope="module")
def regular_result():
    """A plain Normal series — no stockouts, no new/obsolete."""
    rng = np.random.default_rng(7)
    y = rng.normal(10, 2, 80)
    return aid(y)


@pytest.fixture(scope="module")
def new_product_result():
    """20 leading zeros then Poisson(5) — should be flagged as new product."""
    rng = np.random.default_rng(5)
    y = np.concatenate([np.zeros(20), rng.poisson(5, 80).astype(float)])
    return aid(y)


class TestAidResultAttributes:
    """R-style attribute access on the nested dataclasses."""

    def test_type_is_dataclass(self, stockout_result):
        assert isinstance(stockout_result.type, AidType)
        names = {f.name for f in dataclasses.fields(stockout_result.type)}
        assert names == {"type1", "type2", "type2a"}

    def test_stockouts_is_dataclass(self, stockout_result):
        assert isinstance(stockout_result.stockouts, Stockouts)
        names = {f.name for f in dataclasses.fields(stockout_result.stockouts)}
        assert names == {"start", "end", "dummy", "new", "obsolete"}

    def test_attribute_access_stockouts(self, stockout_result):
        # R-style: result.stockouts.start
        assert stockout_result.stockouts.start.tolist() == [41]
        assert stockout_result.stockouts.end.tolist() == [50]
        assert int(stockout_result.stockouts.dummy.sum()) == 10

    def test_attribute_access_type(self, stockout_result):
        # R-style: result.type.type1
        assert stockout_result.type.type1 in ("count", "fractional")
        assert stockout_result.type.type2 in ("regular", "intermittent")

    def test_dict_style_backcompat_stockouts(self, stockout_result):
        # Legacy dict-style: result.stockouts["start"]
        assert stockout_result.stockouts["start"].tolist() == [41]
        assert "start" in stockout_result.stockouts
        assert list(stockout_result.stockouts.keys()) == [
            "start",
            "end",
            "dummy",
            "new",
            "obsolete",
        ]

    def test_dict_style_backcompat_type(self, stockout_result):
        assert stockout_result.type["type1"] == stockout_result.type.type1
        assert "type1" in stockout_result.type
        assert list(stockout_result.type.keys()) == [
            "type1",
            "type2",
            "type2a",
        ]

    def test_stockouts_get_method(self, stockout_result):
        # Dict-style .get() should also work (legacy code path).
        assert (
            stockout_result.stockouts.get("start").tolist()
            == stockout_result.stockouts.start.tolist()
        )
        assert stockout_result.stockouts.get("missing", "default") == "default"

    def test_regular_series_has_none_stockouts(self, regular_result):
        # No zeros at all → start/end are None (matches R's NULL).
        assert regular_result.stockouts.start is None
        assert regular_result.stockouts.end is None
        # Dummy is still a length-n vector of zeros.
        assert regular_result.stockouts.dummy.shape == (len(regular_result.y),)
        assert int(regular_result.stockouts.dummy.sum()) == 0


class TestAidPlot:
    """AidResult.plot() rendering and Axes contract."""

    def test_returns_axes(self, stockout_result):
        ax = stockout_result.plot()
        assert isinstance(ax, matplotlib.axes.Axes)
        assert len(ax.lines) >= 1  # the main series line
        plt.close(ax.figure)

    def test_uses_passed_ax(self, stockout_result):
        fig, ax_in = plt.subplots()
        ax_out = stockout_result.plot(ax=ax_in)
        assert ax_out is ax_in
        plt.close(fig)

    def test_overlays_for_stockout(self, stockout_result):
        ax = stockout_result.plot()
        # One light-grey axvspan per stockout.
        patches = [p for p in ax.patches]
        assert len(patches) == 1
        # Two vlines per stockout (red start + green-dashed end), plus the
        # main series line.
        n_vlines = sum(
            1
            for line in ax.lines
            if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]
        )
        assert n_vlines == 2
        plt.close(ax.figure)

    def test_no_overlays_for_regular(self, regular_result):
        ax = regular_result.plot()
        # No stockouts, no new, no obsolete → no patches at all.
        assert len(ax.patches) == 0
        plt.close(ax.figure)

    def test_overlay_for_new_product(self, new_product_result):
        ax = new_product_result.plot()
        # The leading-zeros block is shaded → at least one patch.
        assert len(ax.patches) >= 1
        plt.close(ax.figure)

    def test_title_carries_demand_name(self, stockout_result):
        ax = stockout_result.plot()
        assert stockout_result.name in ax.get_title()
        plt.close(ax.figure)


class TestAidCatPlot:
    """AidCatResult.plot() — the 2x3 category panel."""

    @pytest.fixture(scope="class")
    @staticmethod
    def cat_result():
        rng = np.random.default_rng(1)
        series = {
            "a": rng.poisson(1, 80).astype(float),
            "b": rng.poisson(5, 80).astype(float),
            "c": rng.normal(10, 2, 80),
        }
        return aid_cat(series)

    def test_returns_axes(self, cat_result):
        ax = cat_result.plot()
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)

    def test_renders_six_category_cells(self, cat_result):
        ax = cat_result.plot()
        texts = [t.get_text() for t in ax.texts]
        # Six category labels + 3 column marginals + 2 row marginals + total = 12
        assert len(texts) == 12
        for label in (
            "Regular Count",
            "Smooth Intermittent Count",
            "Lumpy Intermittent Count",
            "Regular Fractional",
            "Smooth Intermittent Fractional",
            "Lumpy Intermittent Fractional",
        ):
            assert any(label in t for t in texts), f"missing label {label!r}"
        plt.close(ax.figure)

    def test_total_matches_n_series(self, cat_result):
        ax = cat_result.plot()
        # The grand-total text is "Total\n(N)".
        totals = [t.get_text() for t in ax.texts if t.get_text().startswith("Total")]
        assert len(totals) == 1
        n_total = int(totals[0].split("(")[1].rstrip(")"))
        assert n_total == len(cat_result.results)
        plt.close(ax.figure)

    def test_categories_keep_names(self, cat_result):
        assert list(cat_result.categories.index) == ["a", "b", "c"]

    def test_list_input(self, cat_result):
        rng = np.random.default_rng(1)
        series = [
            rng.poisson(1, 80).astype(float),
            rng.poisson(5, 80).astype(float),
            rng.normal(10, 2, 80),
        ]
        result = aid_cat(series)
        assert list(result.categories.index) == [0, 1, 2]
        assert list(result.categories) == list(cat_result.categories)
