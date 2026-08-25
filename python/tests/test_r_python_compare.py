"""Tests comparing R and Python distribution functions.

These tests ensure that the Python implementations produce identical results
to the R greybox package functions.
"""

import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

from greybox.distributions import (
    ds,
    ps,
    qs,
    rs,
    dalaplace,
    palaplace,
    qalaplace,
    ralaplace,
    dbcnorm,
    pbcnorm,
    qbcnorm,
    rbcnorm,
    dfnorm,
    pfnorm,
    qfnorm,
    rfnorm,
    dlogitnorm,
    plogitnorm,
    qlogitnorm,
    rlogitnorm,
    drectnorm,
    prectnorm,
    qrectnorm,
    rrectnorm,
    dgnorm,
    pgnorm,
    qgnorm,
    rgnorm,
    dllaplace,
    qllaplace,
    dls,
    qls,
    dlgnorm,
    qlgnorm,
    dlaplace,
    plaplace,
    qlaplace,
    dlnorm,
    plnorm,
    qlnorm,
    dlogis,
    qlogis,
    plogis,
    dt,
    pt,
    qt,
    dgamma,
    pgamma,
    qgamma,
    dexp,
    pexp,
    qexp,
    dbeta,
    pbeta,
    qbeta,
    dpois,
    ppois,
    qpois,
    dnbinom,
    pnbinom,
    qnbinom,
    dbinom,
    pbinom,
    qbinom,
    dgeom,
    pgeom,
    qgeom,
    dchi2,
    pchi2,
    qchi2,
    dinvgauss,
    pinvgauss,
    qinvgauss,
)


# Set up R environment
import rpy2.robjects as ro

ro.r["library"]("greybox")


def to_r_array(arr):
    """Convert numpy array to R vector."""
    return ro.FloatVector(arr.tolist())


def call_r_func(func_name, *args, **kwargs):
    """Call an R function from greybox package."""
    r_args = [to_r_array(a) if isinstance(a, np.ndarray) else a for a in args]
    r_kwargs = {
        k: (to_r_array(v) if isinstance(v, np.ndarray) else v)
        for k, v in kwargs.items()
    }
    return np.array(ro.r[func_name](*r_args, **r_kwargs))


class TestSvsR:
    """Compare S-distribution between R and Python."""

    def test_ds(self):
        """Test density function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, scale = 1.0, 1.0

        py_result = ds(q, loc=mu, scale=scale)
        r_result = call_r_func("ds", q, mu, scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_ds_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5])
        mu, scale = 1.0, 1.0

        py_result = ds(q, loc=mu, scale=scale, log=True)
        r_result = call_r_func("ds", q, mu, scale, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_ps(self):
        """Test CDF function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, scale = 1.0, 1.0

        py_result = ps(q, loc=mu, scale=scale)
        r_result = call_r_func("ps", q, mu, scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qs(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, scale = 1.0, 1.0

        py_result = qs(p, loc=mu, scale=scale)
        r_result = call_r_func("qs", p, mu, scale)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestGnormvsR:
    """Compare Generalized Normal distribution between R and Python."""

    def test_dgnorm(self):
        """Test density function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, scale, shape = 0.0, 1.0, 2.0

        py_result = dgnorm(q, loc=mu, scale=scale, shape=shape)
        r_result = call_r_func("dgnorm", q, mu, scale, shape)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dgnorm_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5])
        mu, scale, shape = 0.0, 1.0, 2.0

        py_result = dgnorm(q, loc=mu, scale=scale, shape=shape, log=True)
        r_result = call_r_func("dgnorm", q, mu, scale, shape, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_pgnorm(self):
        """Test CDF function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, scale, shape = 0.0, 1.0, 2.0

        py_result = pgnorm(q, loc=mu, scale=scale, shape=shape)
        r_result = call_r_func("pgnorm", q, mu, scale, shape)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qgnorm(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, scale, shape = 0.0, 1.0, 2.0

        py_result = qgnorm(p, loc=mu, scale=scale, shape=shape)
        r_result = call_r_func("qgnorm", p, mu, scale, shape)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestALapacevsR:
    """Compare Asymmetric Laplace distribution between R and Python."""

    def test_dalaplace(self):
        """Test density function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, scale, alpha = 0.0, 1.0, 0.5

        py_result = dalaplace(q, loc=mu, scale=scale, alpha=alpha)
        r_result = call_r_func("dalaplace", q, mu, scale, alpha)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dalaplace_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5])
        mu, scale, alpha = 0.0, 1.0, 0.5

        py_result = dalaplace(q, loc=mu, scale=scale, alpha=alpha, log=True)
        r_result = call_r_func("dalaplace", q, mu, scale, alpha, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_palaplace(self):
        """Test CDF function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, scale, alpha = 0.0, 1.0, 0.5

        py_result = palaplace(q, loc=mu, scale=scale, alpha=alpha)
        r_result = call_r_func("palaplace", q, mu, scale, alpha)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qalaplace(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, scale, alpha = 0.0, 1.0, 0.5

        py_result = qalaplace(p, loc=mu, scale=scale, alpha=alpha)
        r_result = call_r_func("qalaplace", p, mu, scale, alpha)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestBcnormvsR:
    """Compare Box-Cox Normal distribution between R and Python."""

    def test_dbcnorm(self):
        """Test density function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu, sigma, lambda_bc = 1.0, 1.0, 0.5

        py_result = dbcnorm(q, loc=mu, scale=sigma, lambda_bc=lambda_bc)
        r_result = call_r_func("dbcnorm", q, mu, sigma, lambda_bc)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dbcnorm_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5, 2.0])
        mu, sigma, lambda_bc = 1.0, 1.0, 0.5

        py_result = dbcnorm(q, loc=mu, scale=sigma, lambda_bc=lambda_bc, log=True)
        r_result = call_r_func("dbcnorm", q, mu, sigma, lambda_bc, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_pbcnorm(self):
        """Test CDF function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu, sigma, lambda_bc = 1.0, 1.0, 0.5

        py_result = pbcnorm(q, loc=mu, scale=sigma, lambda_bc=lambda_bc)
        r_result = call_r_func("pbcnorm", q, mu, sigma, lambda_bc)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qbcnorm(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, sigma, lambda_bc = 1.0, 1.0, 0.5

        py_result = qbcnorm(p, loc=mu, scale=sigma, lambda_bc=lambda_bc)
        r_result = call_r_func("qbcnorm", p, mu, sigma, lambda_bc)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestFnormvsR:
    """Compare Folded Normal distribution between R and Python."""

    def test_dfnorm(self):
        """Test density function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, sigma = 1.0, 1.0

        py_result = dfnorm(q, loc=mu, scale=sigma)
        r_result = call_r_func("dfnorm", q, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-9)

    def test_dfnorm_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5])
        mu, sigma = 1.0, 1.0

        py_result = dfnorm(q, loc=mu, scale=sigma, log=True)
        r_result = call_r_func("dfnorm", q, mu, sigma, log=True)

        assert_allclose(py_result, r_result, rtol=1e-9)

    def test_pfnorm(self):
        """Test CDF function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, sigma = 1.0, 1.0

        py_result = pfnorm(q, loc=mu, scale=sigma)
        r_result = call_r_func("pfnorm", q, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-9)

    def test_qfnorm(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, sigma = 1.0, 1.0

        py_result = qfnorm(p, loc=mu, scale=sigma)
        r_result = call_r_func("qfnorm", p, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-6)


class TestLogitnormvsR:
    """Compare Logit-Normal distribution between R and Python."""

    def test_dlogitnorm(self):
        """Test density function."""
        q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, sigma = 0.0, 1.0

        py_result = dlogitnorm(q, loc=mu, scale=sigma)
        r_result = call_r_func("dlogitnorm", q, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dlogitnorm_log(self):
        """Test density function with log=True."""
        q = np.array([0.1, 0.25, 0.5, 0.75])
        mu, sigma = 0.0, 1.0

        py_result = dlogitnorm(q, loc=mu, scale=sigma, log=True)
        r_result = call_r_func("dlogitnorm", q, mu, sigma, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_plogitnorm(self):
        """Test CDF function."""
        q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, sigma = 0.0, 1.0

        py_result = plogitnorm(q, loc=mu, scale=sigma)
        r_result = call_r_func("plogitnorm", q, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qlogitnorm(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, sigma = 0.0, 1.0

        py_result = qlogitnorm(p, loc=mu, scale=sigma)
        r_result = call_r_func("qlogitnorm", p, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestRectnormvsR:
    """Compare Rectified Normal distribution between R and Python."""

    def test_drectnorm(self):
        """Test density function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, sigma = 1.0, 1.0

        py_result = drectnorm(q, loc=mu, scale=sigma)
        r_result = call_r_func("drectnorm", q, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_drectnorm_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5])
        mu, sigma = 1.0, 1.0

        py_result = drectnorm(q, loc=mu, scale=sigma, log=True)
        r_result = call_r_func("drectnorm", q, mu, sigma, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_prectnorm(self):
        """Test CDF function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, sigma = 1.0, 1.0

        py_result = prectnorm(q, loc=mu, scale=sigma)
        r_result = call_r_func("prectnorm", q, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qrectnorm(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, sigma = 1.0, 1.0

        py_result = qrectnorm(p, loc=mu, scale=sigma)
        r_result = call_r_func("qrectnorm", p, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestLLapacevsR:
    """Compare Log-Laplace distribution between R and Python.

    Note: R greybox does not have dllaplace, qllaplace functions.
    These tests are for Python implementation verification only.
    """

    def test_dllaplace_python_only(self):
        """Test density function (R does not have this)."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu, scale = 1.0, 1.0

        py_result = dllaplace(q, loc=mu, scale=scale)
        assert py_result.shape == q.shape
        assert np.all(np.isfinite(py_result))

    def test_dllaplace_log_python_only(self):
        """Test density function with log=True (R does not have this)."""
        q = np.array([0.5, 1.0, 1.5, 2.0])
        mu, scale = 1.0, 1.0

        py_result = dllaplace(q, loc=mu, scale=scale, log=True)
        assert py_result.shape == q.shape
        assert np.all(np.isfinite(py_result))

    def test_qllaplace_python_only(self):
        """Test quantile function (R does not have this)."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, scale = 1.0, 1.0

        py_result = qllaplace(p, loc=mu, scale=scale)
        assert py_result.shape == p.shape
        assert np.all(np.isfinite(py_result))
        assert np.all(py_result > 0)


class TestLSvsR:
    """Compare Log-S distribution between R and Python.

    Note: R greybox does not have dls, qls functions.
    These tests are for Python implementation verification only.
    """

    def test_dls_python_only(self):
        """Test density function (R does not have this)."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu, scale = 1.0, 1.0

        py_result = dls(q, loc=mu, scale=scale)
        assert py_result.shape == q.shape
        assert np.all(np.isfinite(py_result))

    def test_dls_log_python_only(self):
        """Test density function with log=True (R does not have this)."""
        q = np.array([0.5, 1.0, 1.5, 2.0])
        mu, scale = 1.0, 1.0

        py_result = dls(q, loc=mu, scale=scale, log=True)
        assert py_result.shape == q.shape
        assert np.all(np.isfinite(py_result))

    def test_qls_python_only(self):
        """Test quantile function (R does not have this)."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, scale = 1.0, 1.0

        py_result = qls(p, loc=mu, scale=scale)
        assert py_result.shape == p.shape
        assert np.all(np.isfinite(py_result))
        assert np.all(py_result > 0)


class TestLGnormvsR:
    """Compare Log-Generalized Normal distribution between R and Python.

    Note: R greybox does not have dlgnorm, qlgnorm functions.
    These tests are for Python implementation verification only.
    """

    def test_dlgnorm_python_only(self):
        """Test density function (R does not have this)."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu, scale, shape = 1.0, 1.0, 2.0

        py_result = dlgnorm(q, loc=mu, scale=scale, shape=shape)
        assert py_result.shape == q.shape
        assert np.all(np.isfinite(py_result))

    def test_dlgnorm_log_python_only(self):
        """Test density function with log=True (R does not have this)."""
        q = np.array([0.5, 1.0, 1.5, 2.0])
        mu, scale, shape = 1.0, 1.0, 2.0

        py_result = dlgnorm(q, loc=mu, scale=scale, shape=shape, log=True)
        assert py_result.shape == q.shape
        assert np.all(np.isfinite(py_result))

    def test_qlgnorm_python_only(self):
        """Test quantile function (R does not have this)."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, scale, shape = 1.0, 1.0, 2.0

        py_result = qlgnorm(p, loc=mu, scale=scale, shape=shape)
        assert py_result.shape == p.shape
        assert np.all(np.isfinite(py_result))
        assert np.all(py_result > 0)


class TestEdgeCases:
    """Test edge cases and special parameter values."""

    def test_gnorm_shape_2(self):
        """Test gnorm with shape=2 (similar to normal)."""
        q = np.array([-1, 0, 1])
        mu, scale, shape = 0, 1, 2

        py_result = dgnorm(q, loc=mu, scale=scale, shape=shape)
        r_result = call_r_func("dgnorm", q, mu, scale, shape)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_bcnorm_lambda_zero(self):
        """Test bcnorm with lambda=0 (log transform)."""
        q = np.array([0.5, 1.0, 1.5, 2.0])
        mu, sigma, lambda_bc = 1.0, 1.0, 0.0

        py_result = dbcnorm(q, loc=mu, scale=sigma, lambda_bc=lambda_bc)
        r_result = call_r_func("dbcnorm", q, mu, sigma, lambda_bc)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_bcnorm_lambda_one(self):
        """Test bcnorm with lambda=1 (identity transform)."""
        q = np.array([0.5, 1.0, 1.5, 2.0])
        mu, sigma, lambda_bc = 1.0, 1.0, 1.0

        py_result = dbcnorm(q, loc=mu, scale=sigma, lambda_bc=lambda_bc)
        r_result = call_r_func("dbcnorm", q, mu, sigma, lambda_bc)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestLaplaceFuncvsR:
    """Compare Laplace distribution between R and Python."""

    def test_dlaplace(self):
        """Test density function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        loc, scale = 1.0, 1.0

        py_result = dlaplace(q, loc=loc, scale=scale)
        r_result = call_r_func("dlaplace", q, loc, scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dlaplace_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        loc, scale = 1.0, 1.0

        py_result = dlaplace(q, loc=loc, scale=scale, log=True)
        r_result = call_r_func("dlaplace", q, loc, scale, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_plaplace(self):
        """Test CDF function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        loc, scale = 1.0, 1.0

        py_result = plaplace(q, loc=loc, scale=scale)
        r_result = call_r_func("plaplace", q, loc, scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qlaplace(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        loc, scale = 1.0, 1.0

        py_result = qlaplace(p, loc=loc, scale=scale)
        r_result = call_r_func("qlaplace", p, loc, scale)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestLognormFuncvsR:
    """Compare Log-Normal distribution between R and Python."""

    def test_dlnorm(self):
        """Test density function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        meanlog, sdlog = 0.0, 1.0

        py_result = dlnorm(q, loc=meanlog, scale=sdlog)
        r_result = call_r_func("dlnorm", q, meanlog, sdlog)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dlnorm_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        meanlog, sdlog = 0.0, 1.0

        py_result = dlnorm(q, loc=meanlog, scale=sdlog, log=True)
        r_result = call_r_func("dlnorm", q, meanlog, sdlog, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_plnorm(self):
        """Test CDF function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        meanlog, sdlog = 0.0, 1.0

        py_result = plnorm(q, loc=meanlog, scale=sdlog)
        r_result = call_r_func("plnorm", q, meanlog, sdlog)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qlnorm(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        meanlog, sdlog = 0.0, 1.0

        py_result = qlnorm(p, loc=meanlog, scale=sdlog)
        r_result = call_r_func("qlnorm", p, meanlog, sdlog)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestLogisticFuncvsR:
    """Compare Logistic distribution between R and Python."""

    def test_dlogis(self):
        """Test density function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        loc, scale = 1.0, 1.0

        py_result = dlogis(q, loc=loc, scale=scale)
        r_result = call_r_func("dlogis", q, loc, scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dlogis_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        loc, scale = 1.0, 1.0

        py_result = dlogis(q, loc=loc, scale=scale, log=True)
        r_result = call_r_func("dlogis", q, loc, scale, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_plogis(self):
        """Test CDF function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        loc, scale = 1.0, 1.0

        py_result = plogis(q, loc=loc, scale=scale)
        r_result = call_r_func("plogis", q, loc, scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qlogis(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        loc, scale = 1.0, 1.0

        py_result = qlogis(p, loc=loc, scale=scale)
        r_result = call_r_func("qlogis", p, loc, scale)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestStudentTFuncvsR:
    """Compare Student's t-distribution between R and Python.

    Note: R's base dt() does not have loc/scale parameters,
    so we transform: dt((q-loc)/scale, df) / scale.
    """

    def test_dt(self):
        """Test density function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        df_val, loc, scale = 5.0, 1.0, 1.0

        py_result = dt(q, df=df_val, loc=loc, scale=scale)
        r_result = call_r_func("dt", (q - loc) / scale, df_val) / scale

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dt_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        df_val, loc, scale = 5.0, 1.0, 1.0

        py_result = dt(q, df=df_val, loc=loc, scale=scale, log=True)
        r_result = np.log(call_r_func("dt", (q - loc) / scale, df_val) / scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_pt(self):
        """Test CDF function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        df_val, loc, scale = 5.0, 1.0, 1.0

        py_result = pt(q, df=df_val, loc=loc, scale=scale)
        r_result = call_r_func("pt", (q - loc) / scale, df_val)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qt(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        df_val, loc, scale = 5.0, 1.0, 1.0

        py_result = qt(p, df=df_val, loc=loc, scale=scale)
        r_result = call_r_func("qt", p, df_val) * scale + loc

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestGammaFuncvsR:
    """Compare Gamma distribution between R and Python."""

    def test_dgamma(self):
        """Test density function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        shape, scale = 2.0, 1.0

        py_result = dgamma(q, shape=shape, scale=scale)
        r_result = call_r_func("dgamma", q, shape, scale=scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dgamma_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        shape, scale = 2.0, 1.0

        py_result = dgamma(q, shape=shape, scale=scale, log=True)
        r_result = call_r_func("dgamma", q, shape, scale=scale, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_pgamma(self):
        """Test CDF function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        shape, scale = 2.0, 1.0

        py_result = pgamma(q, shape=shape, scale=scale)
        r_result = call_r_func("pgamma", q, shape, scale=scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qgamma(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        shape, scale = 2.0, 1.0

        py_result = qgamma(p, shape=shape, scale=scale)
        r_result = call_r_func("qgamma", p, shape, scale=scale)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestExpFuncvsR:
    """Compare Exponential distribution between R and Python.

    Note: R uses rate parameterization, Python uses loc/scale.
    R: dexp(q, rate) = dexp(q-loc, rate=1/scale) for shifted version.
    """

    def test_dexp(self):
        """Test density function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        loc, scale = 0.0, 1.0

        py_result = dexp(q, loc=loc, scale=scale)
        r_result = call_r_func("dexp", q - loc, rate=1.0 / scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dexp_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        loc, scale = 0.0, 1.0

        py_result = dexp(q, loc=loc, scale=scale, log=True)
        r_result = call_r_func("dexp", q - loc, rate=1.0 / scale, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_pexp(self):
        """Test CDF function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        loc, scale = 0.0, 1.0

        py_result = pexp(q, loc=loc, scale=scale)
        r_result = call_r_func("pexp", q - loc, rate=1.0 / scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qexp(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        loc, scale = 0.0, 1.0

        py_result = qexp(p, loc=loc, scale=scale)
        r_result = call_r_func("qexp", p, rate=1.0 / scale) + loc

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestBetaFuncvsR:
    """Compare Beta distribution between R and Python."""

    def test_dbeta(self):
        """Test density function."""
        q = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        a, b = 2.0, 3.0

        py_result = dbeta(q, a=a, b=b)
        r_result = call_r_func("dbeta", q, a, b)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dbeta_log(self):
        """Test density function with log=True."""
        q = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        a, b = 2.0, 3.0

        py_result = dbeta(q, a=a, b=b, log=True)
        r_result = call_r_func("dbeta", q, a, b, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_pbeta(self):
        """Test CDF function."""
        q = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        a, b = 2.0, 3.0

        py_result = pbeta(q, a=a, b=b)
        r_result = call_r_func("pbeta", q, a, b)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qbeta(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        a, b = 2.0, 3.0

        py_result = qbeta(p, a=a, b=b)
        r_result = call_r_func("qbeta", p, a, b)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestPoissonFuncvsR:
    """Compare Poisson distribution between R and Python."""

    def test_dpois(self):
        """Test probability mass function."""
        q = np.array([0, 1, 2, 3, 5, 10])
        mu = 3.0

        py_result = dpois(q, loc=mu)
        r_result = call_r_func("dpois", q, mu)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dpois_log(self):
        """Test probability mass function with log=True."""
        q = np.array([0, 1, 2, 3, 5, 10])
        mu = 3.0

        py_result = dpois(q, loc=mu, log=True)
        r_result = call_r_func("dpois", q, mu, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_ppois(self):
        """Test CDF function."""
        q = np.array([0, 1, 2, 3, 5, 10])
        mu = 3.0

        py_result = ppois(q, loc=mu)
        r_result = call_r_func("ppois", q, mu)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qpois(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu = 3.0

        py_result = qpois(p, loc=mu)
        r_result = call_r_func("qpois", p, mu)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestNbinomFuncvsR:
    """Compare Negative Binomial distribution between R and Python."""

    def test_dnbinom(self):
        """Test probability mass function."""
        q = np.array([0, 1, 2, 3, 5, 10])
        mu, size = 3.0, 5.0

        py_result = dnbinom(q, loc=mu, size=size)
        r_result = call_r_func("dnbinom", q, mu=mu, size=size)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dnbinom_log(self):
        """Test probability mass function with log=True."""
        q = np.array([0, 1, 2, 3, 5, 10])
        mu, size = 3.0, 5.0

        py_result = dnbinom(q, loc=mu, size=size, log=True)
        r_result = call_r_func("dnbinom", q, mu=mu, size=size, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_pnbinom(self):
        """Test CDF function."""
        q = np.array([0, 1, 2, 3, 5, 10])
        mu, size = 3.0, 5.0

        py_result = pnbinom(q, loc=mu, size=size)
        r_result = call_r_func("pnbinom", q, mu=mu, size=size)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qnbinom(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, size = 3.0, 5.0

        py_result = qnbinom(p, loc=mu, size=size)
        r_result = call_r_func("qnbinom", p, mu=mu, size=size)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestBinomFuncvsR:
    """Compare Binomial distribution between R and Python."""

    def test_dbinom(self):
        """Test probability mass function."""
        q = np.array([0, 1, 2, 3, 5, 10])
        size, prob = 10, 0.3

        py_result = dbinom(q, size=size, prob=prob)
        r_result = call_r_func("dbinom", q, size=size, prob=prob)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dbinom_log(self):
        """Test probability mass function with log=True."""
        q = np.array([0, 1, 2, 3, 5, 10])
        size, prob = 10, 0.3

        py_result = dbinom(q, size=size, prob=prob, log=True)
        r_result = call_r_func("dbinom", q, size=size, prob=prob, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_pbinom(self):
        """Test CDF function."""
        q = np.array([0, 1, 2, 3, 5, 10])
        size, prob = 10, 0.3

        py_result = pbinom(q, size=size, prob=prob)
        r_result = call_r_func("pbinom", q, size=size, prob=prob)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qbinom(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        size, prob = 10, 0.3

        py_result = qbinom(p, size=size, prob=prob)
        r_result = call_r_func("qbinom", p, size=size, prob=prob)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestGeomFuncvsR:
    """Compare Geometric distribution between R and Python.

    Note: Both R and Python use the number of failures before
    first success parameterization. The Python implementation
    internally uses q+1 offset to map to scipy's 1-indexed geom.
    """

    def test_dgeom(self):
        """Test probability mass function."""
        q = np.array([0, 1, 2, 3, 5, 10])
        prob = 0.3

        py_result = dgeom(q, prob=prob)
        r_result = call_r_func("dgeom", q, prob=prob)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dgeom_log(self):
        """Test probability mass function with log=True."""
        q = np.array([0, 1, 2, 3, 5, 10])
        prob = 0.3

        py_result = dgeom(q, prob=prob, log=True)
        r_result = call_r_func("dgeom", q, prob=prob, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_pgeom(self):
        """Test CDF function."""
        q = np.array([0, 1, 2, 3, 5, 10])
        prob = 0.3

        py_result = pgeom(q, prob=prob)
        r_result = call_r_func("pgeom", q, prob=prob)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qgeom(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        prob = 0.3

        py_result = qgeom(p, prob=prob)
        r_result = call_r_func("qgeom", p, prob=prob)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestChisqFuncvsR:
    """Compare Chi-squared distribution between R and Python."""

    def test_dchi2(self):
        """Test density function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        df_val = 5.0

        py_result = dchi2(q, df=df_val)
        r_result = call_r_func("dchisq", q, df_val)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dchi2_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        df_val = 5.0

        py_result = dchi2(q, df=df_val, log=True)
        r_result = call_r_func("dchisq", q, df_val, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_pchi2(self):
        """Test CDF function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        df_val = 5.0

        py_result = pchi2(q, df=df_val)
        r_result = call_r_func("pchisq", q, df_val)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qchi2(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        df_val = 5.0

        py_result = qchi2(p, df=df_val)
        r_result = call_r_func("qchisq", p, df_val)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestInvGaussFuncvsR:
    """Compare Inverse Gaussian distribution between R and Python.

    Python's ``loc``/``scale`` are the mean and the dispersion, exactly as in
    R's ``statmod::dinvgauss(mean=, dispersion=)``, so the two are compared at
    the same arguments with no mapping in between.

    Tested away from ``loc=1, scale=1``: at that point several wrong
    parameterisations coincide with the right one, so it cannot tell them
    apart.
    """

    def test_dinvgauss(self):
        """Test density function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        mu, scale = 1.4, 0.6

        py_result = dinvgauss(q, loc=mu, scale=scale)
        ro.r["library"]("statmod")
        r_result = call_r_func("dinvgauss", q, mean=mu, dispersion=scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dinvgauss_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        mu, scale = 1.4, 0.6

        py_result = dinvgauss(q, loc=mu, scale=scale, log=True)
        ro.r["library"]("statmod")
        r_result = np.log(call_r_func("dinvgauss", q, mean=mu, dispersion=scale))

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_pinvgauss(self):
        """Test CDF function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        mu, scale = 1.4, 0.6

        py_result = pinvgauss(q, loc=mu, scale=scale)
        ro.r["library"]("statmod")
        r_result = call_r_func("pinvgauss", q, mean=mu, dispersion=scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qinvgauss(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, scale = 1.4, 0.6

        py_result = qinvgauss(p, loc=mu, scale=scale)
        ro.r["library"]("statmod")
        r_result = call_r_func("qinvgauss", p, mean=mu, dispersion=scale)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestRMCBvsR:
    """Compare rmcb() between R and Python for the rank-based branches."""

    def _fixture(self):
        """A fixed matrix shared by the comparison tests."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(50, 4))
        data[:, 1] += 4
        data[:, 2] += 3
        data[:, 3] += 2
        names = ["Method1", "Method2", "Method3", "Method4"]
        return data, names

    def _run(self, distribution):
        from greybox import rmcb
        import pandas as pd

        data, names = self._fixture()
        # Build the R matrix entirely in R (robust to numpy2ri global
        # conversion being active from other tests in the suite).
        ro.globalenv["flat"] = ro.FloatVector(data.flatten(order="F").tolist())
        ro.globalenv["nm"] = ro.StrVector(names)
        r_set = ro.r(
            f"d <- matrix(flat, nrow={data.shape[0]}, ncol={data.shape[1]});"
            "colnames(d) <- nm;"
            f'rmcb(d, level=0.95, outplot="none", distribution="{distribution}")'
        )
        r_mean = np.array(r_set.rx2("mean"))
        r_interval = np.array(r_set.rx2("interval"))
        r_pvalue = float(np.array(r_set.rx2("p.value"))[0])

        py = rmcb(
            pd.DataFrame(data, columns=names), distribution=distribution
        )
        return r_mean, r_interval, r_pvalue, py

    def test_rmcb_tukey(self):
        r_mean, r_interval, r_pvalue, py = self._run("tukey")
        assert_allclose(py.mean.to_numpy(), r_mean, rtol=1e-8)
        assert_allclose(py.interval.to_numpy(), r_interval, atol=1e-6)
        assert_allclose(py.p_value, r_pvalue, atol=1e-8)

    def test_rmcb_dnorm(self):
        r_mean, r_interval, r_pvalue, py = self._run("dnorm")
        assert_allclose(py.mean.to_numpy(), r_mean, rtol=1e-8)
        assert_allclose(py.interval.to_numpy(), r_interval, atol=1e-8)
        assert_allclose(py.p_value, r_pvalue, atol=1e-8)
