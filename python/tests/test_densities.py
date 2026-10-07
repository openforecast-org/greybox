"""The densities and log-densities of the Laplace, S and Generalised Normal
distributions, the log-densities of their log-variants and gamma() are R's to
the bit (reference values from tests/R/densities.R)."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from greybox import _native_densities
from greybox.distributions import (
    dgnorm,
    dlaplace,
    dlgnorm,
    dllaplace,
    dls,
    ds,
)

# As R formats them in the names of the reference values
SHAPES = ["0.07", "0.3", "0.5", "0.93", "1", "1.37", "2", "2.6", "7.3", "31"]


@pytest.fixture(scope="module")
def reference():
    rows = pd.read_csv(pathlib.Path(__file__).parent / "densities.csv")
    values = rows.value.map(float.fromhex)
    return {name: values[rows.name == name].to_numpy() for name in rows.name.unique()}


def assert_r_equal(actual, desired):
    """Bit for bit, as R computes it on Linux, where the reference values come
    from. The densities go through the C library's exp(), log() and pow(), as
    R's do, and Windows' rounds differently from glibc in the last bit on some
    arguments (R for Windows has its own too), so there they agree to rounding.
    """
    if sys.platform != "win32":
        np.testing.assert_array_equal(actual, desired)
        return
    np.testing.assert_allclose(actual, desired, rtol=1e-13, atol=1e-300)


def test_the_laplace_and_s_log_densities_are_those_of_r(reference):
    q = reference["q"]
    assert_r_equal(dlaplace(q, 0.37, 1.913, log=True), reference["dlaplace"])
    assert_r_equal(ds(q, 0.37, 1.913, log=True), reference["ds"])


@pytest.mark.parametrize("shape", SHAPES)
def test_the_generalised_normal_log_density_is_that_of_r(reference, shape):
    assert_r_equal(
        dgnorm(reference["q"], 0.37, 1.913, float(shape), log=True),
        reference[f"dgnorm{shape}"],
    )


def test_gamma_is_that_of_r(reference):
    assert_r_equal(_native_densities.gammafn(reference["x"]), reference["gamma"])


def test_the_logs_stay_finite_in_the_tails():
    assert np.isfinite(dgnorm(1e3, 0, 1, 31, log=True))
    assert np.isfinite(dlaplace(1e5, 0, 1, log=True))
    assert np.isfinite(ds(1e8, 0, 1, log=True))


def test_the_log_laplace_and_log_s_log_densities_are_those_of_alm(reference):
    y = reference["y"]
    assert_r_equal(dllaplace(y, 0.37, 1.913, log=True), reference["dllaplace"])
    assert_r_equal(dls(y, 0.37, 1.913, log=True), reference["dls"])


@pytest.mark.parametrize("shape", SHAPES)
def test_the_log_generalised_normal_log_density_is_that_of_alm(reference, shape):
    assert_r_equal(
        dlgnorm(reference["y"], 0.37, 1.913, float(shape), log=True),
        reference[f"dlgnorm{shape}"],
    )


def test_the_log_variants_do_not_floor_their_logs_in_the_tails():
    """The logs used to be taken of the density floored at 1e-300."""
    assert dllaplace(1e300, 0, 1, log=True) < np.log(1e-300)
    assert dls(1e300, 0, 0.1, log=True) < np.log(1e-300)
    assert dlgnorm(1e300, 0, 1, 2, log=True) < np.log(1e-300)


def test_the_laplace_and_s_densities_are_those_of_r(reference):
    q = reference["q"]
    assert_r_equal(dlaplace(q, 0.37, 1.913), reference["pdf_dlaplace"])
    assert_r_equal(ds(q, 0.37, 1.913), reference["pdf_ds"])


@pytest.mark.parametrize("shape", SHAPES)
def test_the_generalised_normal_density_is_that_of_r(reference, shape):
    assert_r_equal(
        dgnorm(reference["q"], 0.37, 1.913, float(shape)),
        reference[f"pdf_dgnorm{shape}"],
    )
