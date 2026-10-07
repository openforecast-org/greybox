"""The log-densities of the Laplace, S and Generalised Normal distributions and
gamma() are R's to the bit (reference values from tests/R/log_densities.R)."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from greybox import _native_densities
from greybox.distributions import dgnorm, dlaplace, ds

SHAPES = [0.07, 0.3, 0.5, 0.93, 1, 1.37, 2, 2.6, 7.3, 31]


@pytest.fixture(scope="module")
def reference():
    rows = pd.read_csv(pathlib.Path(__file__).parent / "log_densities.csv")
    values = rows.value.map(float.fromhex)
    return {name: values[rows.name == name].to_numpy() for name in rows.name.unique()}


def test_the_laplace_and_s_log_densities_are_those_of_r(reference):
    q = reference["q"]
    np.testing.assert_array_equal(
        dlaplace(q, 0.37, 1.913, log=True), reference["dlaplace"]
    )
    np.testing.assert_array_equal(ds(q, 0.37, 1.913, log=True), reference["ds"])


@pytest.mark.parametrize("shape", SHAPES)
def test_the_generalised_normal_log_density_is_that_of_r(reference, shape):
    np.testing.assert_array_equal(
        dgnorm(reference["q"], 0.37, 1.913, shape, log=True),
        reference[f"dgnorm{shape}"],
    )


def test_gamma_is_that_of_r(reference):
    np.testing.assert_array_equal(
        _native_densities.gammafn(reference["x"]), reference["gamma"]
    )


def test_the_logs_stay_finite_in_the_tails():
    assert np.isfinite(dgnorm(1e3, 0, 1, 31, log=True))
    assert np.isfinite(dlaplace(1e5, 0, 1, log=True))
    assert np.isfinite(ds(1e8, 0, 1, log=True))
