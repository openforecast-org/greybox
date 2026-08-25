"""Tests for custom distribution functions."""

import numpy as np
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
    plogis,
    pnorm,
    qnorm,
)


class TestS:
    """Tests for S-distribution."""

    def test_ds_basic(self):
        """Test basic density calculation."""
        y = np.array([0.0, 1.0, 2.0])
        result = ds(y, loc=1.0, scale=1.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_ds_log_false(self):
        """Test log=False returns density."""
        y = np.array([1.0])
        result = ds(y, loc=1.0, scale=1.0, log=False)
        assert result[0] >= 0

    def test_ds_log_true(self):
        """Test log=True returns log-density."""
        y = np.array([1.0])
        result = ds(y, loc=1.0, scale=1.0, log=True)
        assert np.isfinite(result[0])

    def test_ps_basic(self):
        """Test CDF."""
        y = np.array([0.0])
        result = ps(y, loc=0.0, scale=1.0)
        assert 0 <= result[0] <= 1

    def test_qs_basic(self):
        """Test quantile function."""
        p = np.array([0.5])
        result = qs(p, loc=0.0, scale=1.0)
        assert np.isfinite(result[0])

    def test_rs_basic(self):
        """Test random generation."""
        result = rs(5, loc=0.0, scale=1.0)
        assert len(result) == 5


class TestGnorm:
    """Tests for Generalized Normal distribution."""

    def test_dgnorm_basic(self):
        """Test basic density calculation."""
        y = np.array([0.0, 1.0, 2.0])
        result = dgnorm(y, loc=1.0, scale=1.0, shape=2.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_dgnorm_shape_one(self):
        """Test shape=1 (Laplace)."""
        y = np.array([0.0])
        result = dgnorm(y, loc=0.0, scale=1.0, shape=1.0)
        assert result[0] > 0

    def test_pgnorm_basic(self):
        """Test CDF."""
        y = np.array([0.0])
        result = pgnorm(y, loc=0.0, scale=1.0, shape=2.0)
        assert 0 <= result[0] <= 1

    def test_qgnorm_basic(self):
        """Test quantile function."""
        p = np.array([0.5])
        result = qgnorm(p, loc=0.0, scale=1.0, shape=2.0)
        assert np.isfinite(result[0])

    def test_rgnorm_basic(self):
        """Test random generation."""
        result = rgnorm(5, loc=0.0, scale=1.0, shape=2.0)
        assert len(result) == 5


class TestALaplace:
    """Tests for Asymmetric Laplace distribution."""

    def test_dalaplace_basic(self):
        """Test basic density calculation."""
        y = np.array([0.0, 1.0, 2.0])
        result = dalaplace(y, loc=1.0, scale=1.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_dalaplace_default_alpha(self):
        """Test default alpha=0.5."""
        y = np.array([1.0])
        result = dalaplace(y, loc=1.0, scale=1.0, alpha=0.5)
        assert result[0] > 0

    def test_dalaplace_asymmetric(self):
        """Test asymmetric alpha values."""
        y = np.array([1.0, 2.0])
        result_low = dalaplace(y, loc=1.0, scale=1.0, alpha=0.25)
        result_high = dalaplace(y, loc=1.0, scale=1.0, alpha=0.75)
        assert not np.array_equal(result_low, result_high)

    def test_palaplace_basic(self):
        """Test CDF."""
        y = np.array([0.0, 1.0, 2.0])
        result = palaplace(y, loc=1.0, scale=1.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_palaplace_at_mu(self):
        """Test CDF at mu equals alpha."""
        y = np.array([1.0])
        result = palaplace(y, loc=1.0, scale=1.0, alpha=0.5)
        assert result[0] == 0.5

    def test_qalaplace_basic(self):
        """Test quantile function."""
        p = np.array([0.25, 0.5, 0.75])
        result = qalaplace(p, loc=0.0, scale=1.0, alpha=0.5)
        assert result.shape == p.shape

    def test_qalaplace_bounds(self):
        """Test quantile at bounds."""
        result_0 = qalaplace(0.0, loc=0.0, scale=1.0, alpha=0.5)
        result_1 = qalaplace(1.0, loc=0.0, scale=1.0, alpha=0.5)
        assert np.isinf(result_1)
        assert result_0 == -np.inf

    def test_ralaplace_basic(self):
        """Test random generation."""
        result = ralaplace(5, loc=0.0, scale=1.0, alpha=0.5)
        assert len(result) == 5


class TestBcnorm:
    """Tests for Box-Cox Normal distribution."""

    def test_dbcnorm_basic(self):
        """Test basic density calculation."""
        y = np.array([1.0, 2.0, 3.0])
        result = dbcnorm(y, loc=0.0, scale=1.0, lambda_bc=0.5)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_dbcnorm_lambda_zero(self):
        """Test lambda=0 (log-normal case)."""
        y = np.array([1.0, 2.0, 3.0])
        result = dbcnorm(y, loc=0.0, scale=1.0, lambda_bc=0.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_dbcnorm_lambda_one(self):
        """Test lambda=1 (normal case)."""
        y = np.array([1.0, 2.0, 3.0])
        result = dbcnorm(y, loc=0.0, scale=1.0, lambda_bc=1.0)
        assert result.shape == y.shape

    def test_dbcnorm_log(self):
        """Test log=True returns log-density."""
        y = np.array([1.0, 2.0])
        result = dbcnorm(y, loc=0.0, scale=1.0, lambda_bc=0.5, log=True)
        assert np.all(np.isfinite(result))

    def test_pbcnorm_basic(self):
        """Test CDF."""
        y = np.array([1.0, 2.0, 3.0])
        result = pbcnorm(y, loc=0.0, scale=1.0, lambda_bc=0.5)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_pbcnorm_zero(self):
        """Test CDF at zero."""
        y = np.array([0.0])
        result = pbcnorm(y, loc=0.0, scale=1.0, lambda_bc=0.5)
        assert result[0] == 0

    def test_qbcnorm_basic(self):
        """Test quantile function."""
        p = np.array([0.25, 0.5, 0.75])
        result = qbcnorm(p, loc=0.0, scale=1.0, lambda_bc=0.5)
        assert result.shape == p.shape
        assert np.all(result >= 0)

    def test_rbcnorm_basic(self):
        """Test random generation."""
        result = rbcnorm(5, loc=0.0, scale=1.0, lambda_bc=0.5)
        assert len(result) == 5
        assert np.all(result >= 0)


class TestFnorm:
    """Tests for Folded Normal distribution."""

    def test_dfnorm_basic(self):
        """Test basic density calculation."""
        y = np.array([0.0, 1.0, 2.0])
        result = dfnorm(y, loc=1.0, scale=1.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_dfnorm_negative_input(self):
        """Test negative input returns 0."""
        y = np.array([-1.0])
        result = dfnorm(y, loc=1.0, scale=1.0)
        assert result[0] == 0

    def test_dfnorm_at_zero(self):
        """Test at y=0."""
        y = np.array([0.0])
        result = dfnorm(y, loc=1.0, scale=1.0)
        assert result[0] >= 0

    def test_pfnorm_basic(self):
        """Test CDF."""
        y = np.array([0.0, 1.0, 2.0])
        result = pfnorm(y, loc=1.0, scale=1.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_pfnorm_negative(self):
        """Test CDF for negative values."""
        y = np.array([-1.0])
        result = pfnorm(y, loc=1.0, scale=1.0)
        assert result[0] == 0

    def test_qfnorm_basic(self):
        """Test quantile function."""
        p = np.array([0.25, 0.5, 0.75])
        result = qfnorm(p, loc=1.0, scale=1.0)
        assert result.shape == p.shape
        assert np.all(result >= 0)

    def test_qfnorm_at_bounds(self):
        """Test quantile at bounds."""
        result_0 = qfnorm(0.0, loc=1.0, scale=1.0)
        result_1 = qfnorm(1.0, loc=1.0, scale=1.0)
        assert result_0 == 0
        assert np.isinf(result_1)

    def test_rfnorm_basic(self):
        """Test random generation."""
        result = rfnorm(5, loc=1.0, scale=1.0)
        assert len(result) == 5
        assert np.all(result >= 0)


class TestLogitnorm:
    """Tests for Logit-Normal distribution."""

    def test_dlogitnorm_basic(self):
        """Test basic density calculation."""
        y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = dlogitnorm(y, loc=0.0, scale=1.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_dlogitnorm_log(self):
        """Test log=True returns log-density."""
        y = np.array([0.3, 0.5, 0.7])
        result = dlogitnorm(y, loc=0.0, scale=1.0, log=True)
        assert np.all(np.isfinite(result))

    def test_dlogitnorm_at_extremes(self):
        """Test at extreme values."""
        y = np.array([0.001, 0.999])
        result = dlogitnorm(y, loc=0.0, scale=1.0)
        assert np.all(result >= 0)

    def test_plogitnorm_basic(self):
        """Test CDF."""
        y = np.array([0.1, 0.5, 0.9])
        result = plogitnorm(y, loc=0.0, scale=1.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_plogitnorm_bounds(self):
        """Test CDF at bounds."""
        y_low = np.array([-0.1])
        y_high = np.array([1.1])
        result_low = plogitnorm(y_low, loc=0.0, scale=1.0)
        result_high = plogitnorm(y_high, loc=0.0, scale=1.0)
        assert result_low[0] == 0
        assert result_high[0] == 1

    def test_qlogitnorm_basic(self):
        """Test quantile function."""
        p = np.array([0.25, 0.5, 0.75])
        result = qlogitnorm(p, loc=0.0, scale=1.0)
        assert result.shape == p.shape
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_rlogitnorm_basic(self):
        """Test random generation."""
        result = rlogitnorm(5, loc=0.0, scale=1.0)
        assert len(result) == 5
        assert np.all(result > 0)
        assert np.all(result < 1)


class TestRectnorm:
    """Tests for Rectified Normal distribution."""

    def test_drectnorm_basic(self):
        """Test basic density calculation."""
        y = np.array([-1.0, 0.0, 1.0])
        result = drectnorm(y, loc=1.0, scale=1.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_drectnorm_positive_only(self):
        """Test that output is non-negative."""
        y = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = drectnorm(y, loc=1.0, scale=1.0)
        assert np.all(result >= 0)

    def test_drectnorm_at_zero(self):
        """Test at y=0."""
        y = np.array([0.0])
        result = drectnorm(y, loc=1.0, scale=1.0)
        assert result[0] >= 0

    def test_prectnorm_basic(self):
        """Test CDF."""
        y = np.array([-1.0, 0.0, 1.0])
        result = prectnorm(y, loc=1.0, scale=1.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_prectnorm_at_zero(self):
        """Test CDF at zero."""
        y = np.array([0.0])
        result = prectnorm(y, loc=1.0, scale=1.0)
        assert result[0] > 0

    def test_qrectnorm_basic(self):
        """Test quantile function."""
        p = np.array([0.25, 0.5, 0.75])
        result = qrectnorm(p, loc=1.0, scale=1.0)
        assert result.shape == p.shape
        assert np.all(result >= 0)

    def test_rrectnorm_basic(self):
        """Test random generation."""
        result = rrectnorm(5, loc=1.0, scale=1.0)
        assert len(result) == 5
        assert np.all(result >= 0)


class TestHelperFunctions:
    """Tests for helper distribution functions."""

    def test_plogis(self):
        """Test logistic CDF."""
        y = np.array([0.0])
        result = plogis(y, loc=0.0, scale=1.0)
        assert result[0] == 0.5

    def test_pnorm(self):
        """Test normal CDF."""
        y = np.array([0.0])
        result = pnorm(y, loc=0.0, scale=1.0)
        assert result[0] == 0.5

    def test_pnorm_lower_tail_false(self):
        """Test upper tail."""
        y = np.array([0.0])
        result = pnorm(y, loc=0.0, scale=1.0, lower_tail=False)
        assert result[0] == 0.5

    def test_qnorm(self):
        """Test normal quantile function."""
        p = np.array([0.5])
        result = qnorm(p, loc=0.0, scale=1.0)
        np.testing.assert_array_almost_equal(result, np.array([0.0]))


class TestDgammaAccuracy:
    """`dgamma` uses Loader's saddle-point form, as R's stats::dgamma does.

    The direct expression `(a-1)*log(x) - x/s - lgamma(a) - a*log(s)` cancels
    once the shape grows: at a shape of 143 it loses about 1.4e-13, and at
    10000 about 1e-11. That is enough to move an optimiser off a boundary,
    which is how it was found -- an ADAM fit stopped short of beta = 0 where R
    reached it.
    """

    # Reference values from R 4.6.1: dgamma(q, shape=, scale=, log=TRUE)
    Q = np.array([0.5, 1.0, 2.5, 10.0])

    def test_matches_r_at_moderate_shape(self):
        from greybox import dgamma

        expected = np.array(
            [-1.8374107301096074, -1.4775968828829953,
             -1.5613061510088402, -5.1750117898889494]
        )
        got = dgamma(self.Q, shape=2.0, scale=1.5, log=True)
        np.testing.assert_allclose(got, expected, rtol=1e-13)

    def test_beats_the_direct_expression_at_large_shape(self):
        """Where it matters: a large shape is where the naive form loses bits."""
        from scipy import stats

        from greybox import dgamma

        shape, scale = 10000.0, 0.001
        q = np.array([9.0, 9.5, 10.0, 10.5, 11.0])
        # R 4.6.1: dgamma(q, shape=10000, scale=0.001, log=TRUE)
        expected = np.array(
            [-52.116157836149142, -11.498012354661739, 1.3836382264560427,
             -10.763510243393359, -45.60987391009968]
        )
        saddle = dgamma(q, shape=shape, scale=scale, log=True)
        direct = stats.gamma.logpdf(q, a=shape, scale=scale)
        # The saddle-point form tracks R to a few ulps; the direct expression
        # is orders of magnitude further away at this shape.
        err_saddle = np.max(np.abs((saddle - expected) / expected))
        err_direct = np.max(np.abs((direct - expected) / expected))
        assert err_saddle < 1e-13
        assert err_direct > 10 * err_saddle

    def test_shape_and_scale_may_vary_per_observation(self):
        """The scale model in `smooth` passes a shape that changes each period."""
        from greybox import dgamma

        q = np.array([1.0, 2.0, 3.0])
        shape = np.array([1.5, 2.0, 2.5])
        scale = np.array([1.0, 1.5, 2.0])
        got = dgamma(q, shape=shape, scale=scale, log=True)
        one_by_one = [
            float(dgamma(q[i], shape=shape[i], scale=scale[i], log=True))
            for i in range(3)
        ]
        np.testing.assert_allclose(got, one_by_one, rtol=0, atol=0)

    def test_log_and_density_agree(self):
        from greybox import dgamma

        q = np.array([0.5, 1.0, 4.0])
        np.testing.assert_allclose(
            dgamma(q, shape=2.0, scale=1.5),
            np.exp(dgamma(q, shape=2.0, scale=1.5, log=True)),
            rtol=1e-14,
        )

    def test_boundary_values_match_r(self):
        """R's conventions at 0, at negative q, and for non-finite input."""
        from greybox import dgamma

        # q = 0: +Inf for shape < 1, -log(scale) at shape == 1, -Inf above
        assert dgamma(0.0, shape=0.5, scale=2.0, log=True) == np.inf
        assert dgamma(0.0, shape=1.0, scale=2.0, log=True) == -np.log(2.0)
        assert dgamma(0.0, shape=2.0, scale=2.0, log=True) == -np.inf
        # outside the support, and non-finite
        assert dgamma(-1.0, shape=2.0, scale=1.0, log=True) == -np.inf
        assert dgamma(np.inf, shape=2.0, scale=1.0, log=True) == -np.inf
        assert np.isnan(dgamma(np.nan, shape=2.0, scale=1.0, log=True))
        # an invalid parameter is NaN, not an exception
        assert np.isnan(dgamma(1.0, shape=2.0, scale=-1.0, log=True))


class TestNormalRParity:
    """`dnorm`/`dlnorm` follow R's nmath term for term, not SciPy's grouping.

    SciPy computes its `log(sqrt(2*pi))` constant at import time and lands 1
    ULP below the correctly-rounded value R hard-codes, and `lognorm` routes
    `meanlog` through `exp()` and back. Neither matters on its own, but summed
    over a series they move a log-likelihood in the last digits -- which is how
    this surfaced, as an ADAM fit disagreeing with R in the 13th digit.

    All references are from R 4.6.1, printed at %.17g.
    """

    def test_dnorm_log_matches_r(self):
        from greybox import dnorm

        q = np.array([-3.0, -1.0, 0.0, 0.5, 2.0])
        expected = np.array(
            [
                -3.4495667842668429,
                -1.7886671302876045,
                -1.4772484451664971,
                -1.4512968880730717,
                -1.8924733586613067,
            ]
        )
        got = dnorm(q, loc=0.4, scale=1.7, log=True)
        np.testing.assert_array_equal(got, expected)

    def test_dnorm_density_matches_r(self):
        from greybox import dnorm

        q = np.array([-3.0, -1.0, 0.0, 0.5, 2.0])
        expected = np.array(
            [
                0.031759392066581217,
                0.16718285419212844,
                0.22826490848509695,
                0.234266273863697,
                0.15069861577989621,
            ]
        )
        np.testing.assert_array_equal(dnorm(q, loc=0.4, scale=1.7), expected)

    def test_dnorm_density_uses_rs_split_in_the_tail(self):
        """Past |z| = 5 R splits z so that z1*z1 is exact, and regroups the
        division by sigma. Both are needed to land on R's bits."""
        from greybox import dnorm

        q = np.array([6.0, 12.0, 20.0, 37.0, 38.5])
        expected = np.array(
            [
                6.0758828498232861e-09,
                2.1463837356630605e-32,
                5.5209483621597635e-88,
                2.1200065515246056e-298,
                5.434722104253712e-323,
            ]
        )
        np.testing.assert_array_equal(dnorm(q, loc=0.0, scale=1.0), expected)

    def test_dnorm_density_across_the_branch_boundary(self):
        from greybox import dnorm

        q = np.array([4.999, 5.0, 5.001])
        expected = np.array(
            [1.4941709802283082e-06, 1.4867195147342977e-06, 1.4793037305678625e-06]
        )
        np.testing.assert_array_equal(dnorm(q, loc=0.0, scale=1.0), expected)

    def test_dnorm_degenerate_scale(self):
        """R treats sigma == 0 as a point mass and sigma < 0 as an error."""
        from greybox import dnorm

        assert dnorm(0.0, 0.0, 0.0, log=True) == np.inf
        assert dnorm(1.0, 0.0, 0.0, log=True) == -np.inf
        assert dnorm(1.0, 0.0, 0.0) == 0.0
        assert np.isnan(dnorm(1.0, 0.0, -1.0))
        assert np.isnan(dnorm(1.0, 0.0, -1.0, log=True))
        assert dnorm(1.0, 0.0, np.inf) == 0.0

    def test_dlnorm_matches_r(self):
        from greybox import dlnorm

        q = np.array([0.25, 1.0, 3.0, 10.0])
        expected_log = np.array(
            [
                -0.95041934582064536,
                -1.1880656455541829,
                -2.3242948342724565,
                -4.5320788696191849,
            ]
        )
        expected_lin = np.array(
            [
                0.38657887922227269,
                0.30481030534500203,
                0.097852421902366177,
                0.010758287732064791,
            ]
        )
        np.testing.assert_array_equal(dlnorm(q, 0.5, 1.2, log=True), expected_log)
        np.testing.assert_array_equal(dlnorm(q, 0.5, 1.2), expected_lin)

    def test_dlnorm_does_not_split_the_exponent(self):
        """Unlike dnorm, R's dlnorm has no large-|z| branch."""
        from greybox import dlnorm

        q = np.array([1e-6, 1e6])
        expected = np.array([1.305609693717162e-160, 1.305609693717162e-172])
        np.testing.assert_array_equal(dlnorm(q, 0.0, 0.5), expected)

    def test_dlnorm_non_positive_support(self):
        from greybox import dlnorm

        assert dlnorm(0.0, 0.0, 1.0) == 0.0
        assert dlnorm(-1.0, 0.0, 1.0) == 0.0
        assert dlnorm(0.0, 0.0, 1.0, log=True) == -np.inf

    def test_broadcasting_and_scalar_returns(self):
        from greybox import dlnorm, dnorm

        q = np.array([0.5, 1.0])
        vec = dnorm(q, loc=np.array([0.0, 1.0]), scale=np.array([1.0, 2.0]), log=True)
        one_by_one = [
            float(dnorm(0.5, 0.0, 1.0, log=True)),
            float(dnorm(1.0, 1.0, 2.0, log=True)),
        ]
        np.testing.assert_array_equal(vec, one_by_one)
        assert np.ndim(dnorm(0.0, 0.0, 1.0, log=True)) == 0
        assert np.ndim(dlnorm(1.0, 0.0, 1.0, log=True)) == 0
