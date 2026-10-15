"""Unit tests of the measures of association that do not need R."""

import numpy as np
import pandas as pd
import pytest

from greybox import (
    ALM,
    CramerResult,
    McorResult,
    assoc,
    association,
    cramer,
    determ,
    determination,
    mcor,
    mtcars,
    pcor,
)


@pytest.fixture
def mtcars_na():
    data = mtcars.copy()
    data.loc[2, "cyl"] = np.nan
    data.loc[4, "mpg"] = np.nan
    return data


def test_assoc_alias():
    assert assoc is association


@pytest.mark.parametrize("function", [association, pcor])
def test_unknown_use(function):
    with pytest.raises(ValueError, match="Unknown use"):
        function(mtcars, use="pairwise.complete.obs")


def test_mcor_result(mtcars_na):
    result = mcor(mtcars["cyl"], mtcars["mpg"])
    assert isinstance(result, McorResult)
    assert (result.df, result.df_residual) == (2, 29)
    assert str(result).startswith("Multiple correlations value: 0.8558")
    assert np.isnan(mcor(mtcars_na["cyl"], mtcars_na["mpg"], use="everything").value)
    with pytest.raises(ValueError, match="Missing observations in mcor"):
        mcor(mtcars_na["cyl"], mtcars_na["mpg"], use="all.obs")


def test_cramer_result(mtcars_na):
    result = cramer(mtcars["am"], mtcars["gear"])
    assert isinstance(result, CramerResult)
    assert result.df == 2
    assert str(result).startswith("Cramer's V: ")
    assert np.isnan(cramer(mtcars_na["cyl"], mtcars["am"], use="everything").value)
    # Two observations: the bias correction divides by zero, as in R
    assert np.isnan(cramer([1, 2], [1, 2]).value)


def test_cramer_strings():
    x = pd.Series(["a", "b", "a", "b", "a", "c"] * 3)
    y = pd.Series(["u", "v", "u", "u", "v", "v"] * 3)
    codes = cramer(x.astype("category").cat.codes, y.astype("category").cat.codes)
    assert cramer(x, y).value == pytest.approx(codes.value, rel=1e-14)


def test_association_missing(mtcars_na):
    with pytest.raises(ValueError, match="variables: mpg, cyl"):
        association(mtcars_na, use="all.obs")
    sparse = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 1.0]})
    with pytest.raises(ValueError, match="Missing observations"):
        association(sparse, use="complete.obs")


def test_pcor_missing(mtcars_na):
    for use in ("everything", "all.obs"):
        with pytest.raises(ValueError):
            pcor(mtcars_na, use=use)
    complete = pcor(mtcars_na.dropna())
    np.testing.assert_allclose(pcor(mtcars_na)["value"], complete["value"])


def test_determ():
    X = np.column_stack([np.ones(len(mtcars)), mtcars[["wt", "hp", "disp"]]])
    model = ALM().fit(X, mtcars["mpg"].to_numpy())
    np.testing.assert_allclose(
        determ(model), determination(mtcars[["wt", "hp", "disp"]]), rtol=1e-14
    )
    np.testing.assert_allclose(
        determ(mtcars[["wt", "hp"]]), determination(mtcars[["wt", "hp"]])
    )
