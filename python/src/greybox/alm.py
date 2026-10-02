"""Augmented Linear Model (ALM) for greybox.

This module provides the ALM estimator following scikit-learn principles.
Users should first use the formula module to get X and y, then pass them
to the fit() method.
"""

import time as time_module
from typing import ClassVar, Literal

import nlopt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma as _sp_gamma

from . import distributions as dist
from .cost_function import cf
from .fitters import (
    VARIANCE_SCALE_DISTRIBUTIONS,
    extractor_fitted,
    extractor_residuals,
    scale_sd,
    scaler_internal,
)
from .methods.summary import SummaryResult
from .transforms import bc_transform_inv as _bc_transform_inv
from .xreg import xreg_expander


def _numerical_hessian(f, x0, h=None):
    """Numerical Hessian via central finite differences.

    Matches R's pracma::hessian() with fixed step size.

    Parameters
    ----------
    f : callable
        Scalar function of a vector argument.
    x0 : np.ndarray
        Point at which to evaluate the Hessian.
    h : float, optional
        Step size. Default: eps^(1/4) matching R.

    Returns
    -------
    H : np.ndarray
        Hessian matrix.
    """
    if h is None:
        h = np.finfo(float).eps ** 0.25
    n = len(x0)
    H = np.empty((n, n))
    hh = np.diag(np.full(n, h))
    f0 = f(x0)
    for i in range(n):
        hi = hh[:, i]
        H[i, i] = (f(x0 - hi) - 2 * f0 + f(x0 + hi)) / h**2
        for j in range(i + 1, n):
            hj = hh[:, j]
            H[i, j] = (
                f(x0 + hi + hj) - f(x0 + hi - hj) - f(x0 - hi + hj) + f(x0 - hi - hj)
            ) / (4 * h**2)
            H[j, i] = H[i, j]
    return H


class PredictionResult:
    """Prediction result object with mean, interval bounds, and metadata.

    Supports DataFrame-like access: indexing by column name, len(),
    iteration, and conversion to pandas DataFrame via to_dataframe().

    Attributes
    ----------
    mean : np.ndarray
        Predicted values (point forecasts).
    lower : np.ndarray or None
        Lower prediction bounds.
    upper : np.ndarray or None
        Upper prediction bounds.
    level : float or list[float] or None
        Confidence level(s) used for the intervals.
    variances : np.ndarray or None
        Variance estimates for each observation.
    side : str
        Side of interval: "both", "upper", or "lower".
    interval : str
        Type of interval: "none", "confidence", or "prediction".
    """

    __slots__ = ("interval", "level", "lower", "mean", "side", "upper", "variances")

    def __init__(
        self,
        mean=None,
        lower=None,
        upper=None,
        level=None,
        variances=None,
        side="both",
        interval="none",
    ):
        self.mean = mean
        self.lower = lower
        self.upper = upper
        self.level = level
        self.variances = variances
        self.side = side
        self.interval = interval

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'mean' column and optional 'lower'/'upper'
            columns (or 'lower_0', 'upper_0', etc. for multiple levels).
        """
        data = {"mean": self.mean}
        if self.lower is not None:
            if self.lower.ndim == 1:
                data["lower"] = self.lower
            else:
                for i in range(self.lower.shape[1]):
                    data[f"lower_{i}"] = self.lower[:, i]
        if self.upper is not None:
            if self.upper.ndim == 1:
                data["upper"] = self.upper
            else:
                for i in range(self.upper.shape[1]):
                    data[f"upper_{i}"] = self.upper[:, i]
        return pd.DataFrame(data)

    def __repr__(self):
        return repr(self.to_dataframe())

    def __len__(self):
        if self.mean is not None:
            return len(self.mean)
        return 0

    def __getitem__(self, key):
        return self.to_dataframe()[key]

    @property
    def columns(self):
        """Column names of the DataFrame representation."""
        return self.to_dataframe().columns

    @property
    def shape(self):
        """Shape of the DataFrame representation."""
        return self.to_dataframe().shape

    @property
    def index(self):
        """Index of the DataFrame representation."""
        return self.to_dataframe().index

    @property
    def values(self):
        """Values of the DataFrame representation."""
        return self.to_dataframe().values


NLOPT_ALGORITHMS = {
    "NLOPT_LN_NELDERMEAD": nlopt.LN_NELDERMEAD,
    "NLOPT_LN_SBPLX": nlopt.LN_SBPLX,
    "NLOPT_LN_COBYLA": nlopt.LN_COBYLA,
    "NLOPT_LN_BOBYQA": nlopt.LN_BOBYQA,
    "NLOPT_LN_NEWUOA": nlopt.LN_NEWUOA,
    "NLOPT_LN_PRAXIS": nlopt.LN_PRAXIS,
    "NLOPT_LD_LBFGS": nlopt.LD_LBFGS,
    "NLOPT_LD_SLSQP": nlopt.LD_SLSQP,
    "NLOPT_LD_MMA": nlopt.LD_MMA,
}


class ALM:
    """Augmented Linear Model estimator.

    This estimator fits a linear model with various distributions and loss
    functions, following scikit-learn principles.

    Parameters
    ----------
    distribution : str, default="dnorm"
        Distribution name. Options: "dnorm", "dlaplace", "ds", "dgnorm",
        "dlogis", "dt", "dalaplace", "dlnorm", "dllaplace", "dls",
        "dlgnorm", "dbcnorm", "dfnorm", "drectnorm", "dinvgauss",
        "dgamma", "dexp", "dchisq", "dgeom", "dpois", "dnbinom",
        "dbinom", "dlogitnorm", "dbeta", "plogis", "pnorm".
    loss : str, default="likelihood"
        Loss function. Options: "likelihood", "MSE", "MAE", "HAM",
        "LASSO", "RIDGE", "ROLE".
    occurrence : str, default="none"
        Occurrence model for zero-inflated data. Options: "none",
        "plogis", "pnorm".
    scale_formula : array-like or None, default=None
        Formula for scale parameter. If None, scale is constant.
    orders : tuple, default=(0, 0, 0)
        ARIMA orders (p, d, q). Three integers: AR order (p), differencing
        order (d), and MA order (q). Only AR(p) and differencing (d) are
        supported. MA(q) raises NotImplementedError.

        - p (AR order): Number of lagged response variables to include.
        - d (Differencing): Order of differencing. Creates AR(p+d) terms
          internally but only uses first p in the model.
        - q (MA order): Not implemented.

        Examples:
            - ``orders=(1, 0, 0)``: AR(1) model
            - ``orders=(2, 0, 0)``: AR(2) model
            - ``orders=(1, 1, 0)``: ARIMA(1,1,0) with differencing
    alpha : float, optional
        Additional parameter for Asymmetric Laplace distribution.
    shape : float, optional
        Shape parameter for Generalized Normal distribution.
    lambda_bc : float, optional
        Box-Cox lambda parameter for Box-Cox Normal distribution.
    size : float, optional
        Size parameter for Negative Binomial/Binomial distributions.
    nu : float, optional
        Degrees of freedom for Student's t or Chi-squared distributions.
    trim : float, default=0.0
        Trim proportion for ROLE loss.
    lambda_l1 : float, optional
        L1 regularization parameter for LASSO.
    lambda_l2 : float, optional
        L2 regularization parameter for RIDGE.
    nlopt_kargs : dict, optional
        Dictionary of nlopt parameters. Options:
        - "algorithm": str, default="NLOPT_LN_NELDERMEAD"
        - "maxeval": int, default=40 per parameter
        - "maxtime": float, default=600 seconds
        - "xtol_rel": float, default=1e-6
        - "xtol_abs": float, default=1e-8
        - "ftol_rel": float, default=1e-4
        - "ftol_abs": float, default=0
        - "print_level": int, default=0 (0=none, 3=full)
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients (excluding intercept).
    intercept_ : float
        Estimated intercept.
    scale_ : float
        Estimated scale parameter. This is the variance sigma^2 for dnorm and
        dlnorm (of log(y) for the latter), following the ADAM monograph.
    other_ : dict
        Other estimated parameters (alpha, shape, etc.).
    fitted_values_ : ndarray of shape (n_samples,)
        Fitted values.
    residuals_ : ndarray of shape (n_samples,)
        Model residuals.
    loss_value_ : float
        Final value of the loss function.
    log_lik_ : float or None
        Log-likelihood (only for likelihood-based losses).
    aic_ : float or None
        Akaike Information Criterion.
    bic_ : float or None
        Bayesian Information Criterion.
    n_iter_ : int
        Number of optimization iterations.

    Examples
    --------
    >>> from greybox import formula, ALM
    >>> data = {'y': [2.1, 3.9, 6.2, 7.8, 10.1, 12.2],
    ...         'x1': [1, 2, 3, 4, 5, 6], 'x2': [2, 1, 4, 3, 6, 5]}
    >>> y, X = formula("y ~ x1 + x2", data)
    >>> model = ALM(distribution="dnorm", loss="likelihood")
    >>> _ = model.fit(X, y)
    >>> np.round(model.coef, 4)  # slope parameters (x1, x2)
    array([1.9354, 0.1021])

    >>> # Using nlopt with custom parameters (like R)
    >>> model = ALM(
    ...     distribution="dnorm",
    ...     loss="likelihood",
    ...     nlopt_kargs={
    ...         "algorithm": "NLOPT_LN_SBPLX",
    ...         "maxeval": 1000,
    ...         "xtol_rel": 1e-8,
    ...     }
    ... )
    >>> _ = model.fit(X, y)
    """

    DISTRIBUTIONS: ClassVar[list[str]] = [
        "dnorm",
        "dlaplace",
        "ds",
        "dgnorm",
        "dlogis",
        "dt",
        "dalaplace",
        "dlnorm",
        "dllaplace",
        "dls",
        "dlgnorm",
        "dbcnorm",
        "dinvgauss",
        "dgamma",
        "dexp",
        "dchisq",
        "dfnorm",
        "drectnorm",
        "dpois",
        "dnbinom",
        "dbinom",
        "dgeom",
        "dbeta",
        "dlogitnorm",
        "plogis",
        "pnorm",
    ]

    LOSS_FUNCTIONS: ClassVar[list[str]] = [
        "likelihood",
        "MSE",
        "MAE",
        "HAM",
        "LASSO",
        "RIDGE",
        "ROLE",
    ]

    def __init__(
        self,
        distribution="dnorm",
        loss="likelihood",
        occurrence="none",
        scale_formula=None,
        orders=(0, 0, 0),
        alpha=None,
        shape=None,
        lambda_bc=None,
        size=None,
        nu=None,
        trim=0.0,
        lambda_l1=None,
        lambda_l2=None,
        nlopt_kargs=None,
        verbose=0,
    ):
        self.distribution = distribution
        self.loss = loss
        self.occurrence = occurrence
        self.scale_formula = scale_formula
        self.orders = orders
        self.alpha = alpha
        self.shape = shape
        self.lambda_bc = lambda_bc
        self.size = size
        self.nu = nu
        self.trim = trim
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.nlopt_kargs = nlopt_kargs if nlopt_kargs is not None else {}
        self.verbose = verbose

        self._coef = None
        self._scale = None
        self.other_ = None
        self.fitted_values_ = None
        self.mu_: np.ndarray | None = None
        self.residuals_ = None
        self._loss_value = None
        self._log_lik = None
        self._aic = None
        self._bic = None
        self._aicc = None
        self._bicc = None
        self.n_iter_ = None
        self._time_elapsed = None
        self.ic_values = None
        self._result = None
        self._n_features = None
        self._n_features_exog = None
        self._X_train_ = None
        self._y_train_ = None
        self._formula_ = None
        self._feature_names = None
        self._response_name = None
        self._df_residual = None
        self._B_opt_ = None
        self._a_parameter_provided_ = False
        self._other_val_ = 1.0

    def _validate_params(self):
        """Validate input parameters."""
        if self.distribution not in self.DISTRIBUTIONS:
            raise ValueError(
                f"Invalid distribution: {self.distribution}. "
                f"Choose from: {self.DISTRIBUTIONS}"
            )

        if self.loss not in self.LOSS_FUNCTIONS:
            raise ValueError(
                f"Invalid loss: {self.loss}. Choose from: {self.LOSS_FUNCTIONS}"
            )

        if not isinstance(self.orders, (tuple, list)) or len(self.orders) != 3:
            raise ValueError("orders must be a tuple of 3 integers (p, d, q)")

        if isinstance(self.occurrence, ALM) and self.occurrence.fitted_values_ is None:
            raise ValueError(
                "Occurrence ALM must be fitted before being passed to ALM()"
            )

    def fit(self, X, y, formula=None, feature_names=None):
        """Fit the ALM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix. Should include intercept column (column of ones)
            as first column if you want an intercept.
        y : array-like of shape (n_samples,)
            Target values.
        formula : str, optional
            Formula string used to generate X and y. Stored for reference.
        feature_names : list of str, optional
            Names for feature columns. If provided, used in print output.

        Returns
        -------
        self : ALM
            Fitted estimator.
        """
        start_time = time_module.time()

        self._validate_params()

        X_is_dataframe = isinstance(X, pd.DataFrame)
        y_is_series = isinstance(y, pd.Series)

        if feature_names is None and X_is_dataframe:
            feature_names = list(X.columns)
            if "(Intercept)" in feature_names:
                feature_names.remove("(Intercept)")

        response_name = None
        if y_is_series and y.name is not None:
            response_name = y.name

        self._formula_ = formula
        self._feature_names = feature_names
        self._response_name = response_name

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self._n_features = n_features

        ar_order = self.orders[0]
        i_order = self.orders[1]
        ma_order = self.orders[2]

        if ma_order > 0:
            raise NotImplementedError(
                "MA(q) is not implemented in ALM. "
                "Only AR(p) and differencing (d) are supported."
            )

        ari_order = ar_order + i_order

        self._ar_order = ar_order
        self._i_order = i_order
        self._ari_order = ari_order

        # Save n_features before adding AR lag columns
        n_features_exog = n_features
        self._n_features_exog = n_features_exog

        ar_term_names = []
        if ari_order > 0:
            if response_name is None:
                response_name = "y"
                self._response_name = "y"

            ar_lags = list(range(-1, -ari_order - 1, -1))
            _log_scale_dists = ("dlnorm", "dllaplace", "dls", "dlgnorm")
            y_for_ar = np.log(y) if self.distribution in _log_scale_dists else y
            # [:, 1:] skips the original column prepended by xreg_expander
            ar_terms = xreg_expander(y_for_ar, lags=ar_lags, gaps="auto")[:, 1:]

            ar_term_names = [f"{response_name}Lag{i}" for i in range(1, ari_order + 1)]

            # Keep all ari_order lag columns in X (including differencing lags)
            # For i_order > 0, the polynomial constraint maps ar_order free phi params
            # to ari_order combined polynomial coefficients in the design matrix.

            X = np.hstack([X, ar_terms])

            # Feature names: for i_order > 0, only ar_order names are "free" phi
            # params in coef_, but all ari_order names appear in arima_polynomial_.
            free_ar_term_names = ar_term_names[:ar_order] if ar_order > 0 else []
            if feature_names is not None:
                feature_names = list(feature_names) + free_ar_term_names
            else:
                feature_names = [
                    f"x{i}" for i in range(n_features_exog)
                ] + free_ar_term_names

            n_samples, n_features = X.shape
            self._n_features = n_features
            self._feature_names = feature_names

        other_val = self._get_other_parameter()
        a_parameter_provided = other_val is not None

        if not a_parameter_provided:
            other_val = 1.0

        distributions_with_extra_param = (
            "dalaplace",
            "dnbinom",
            "dchisq",
            "dfnorm",
            "drectnorm",
            "dgnorm",
            "dlgnorm",
            "dbcnorm",
            "dt",
        )

        # n_params counts free optimizer parameters:
        # for ARI(p,d): n_features_exog + ar_order (not ari_order — d>0 adds
        # constrained polynomial columns, not free parameters).
        n_params_base = n_features_exog + ar_order
        if (
            self.distribution in distributions_with_extra_param
            and not a_parameter_provided
        ):
            n_params = n_params_base + 1
        elif self.distribution == "dbeta":
            n_params = 2 * n_params_base
        else:
            n_params = n_params_base

        # Store optimizer free-param count (excludes constrained ARI
        # differencing columns that appear in _X_train_ but are not free
        # parameters).  Used by the nparam property.
        self._n_params_ = n_params

        B_init = np.zeros(n_params)

        log_link_distributions = (
            "dinvgauss",
            "dgamma",
            "dexp",
            "dnbinom",
            "dbinom",
            "dgeom",
        )

        # ── Step 1: design matrix ─────────────────────────────────────────────
        # For ARI(d>0) the AR lag columns in X are the *level* lags. Differencing
        # them yields the correct regressors for the AR polynomial of the
        # differenced model, matching R's alm() initialisation.
        if i_order > 0:
            exog_init = X[i_order:, :n_features_exog]  # drop first i_order rows
            if ar_order > 0:
                ar_lag_cols = X[:, n_features_exog : n_features_exog + ar_order]
                ar_lag_diff = np.diff(
                    ar_lag_cols, n=i_order, axis=0
                )  # (n-i_order, ar_order)
                # R-style boundary: replace first i_order rows with column means
                ar_lag_diff[:i_order] = ar_lag_diff.mean(axis=0)
                X_init = np.hstack([exog_init, ar_lag_diff])
            else:
                X_init = exog_init
        else:
            X_init = X  # already n_params_base columns

        # ── Step 2: per-distribution response transformation and OLS ──────────
        def _ols_init(Xd, yr):
            try:
                return np.linalg.lstsq(Xd, yr, rcond=None)[0]
            except Exception:  # noqa: BLE001 - any OLS failure falls back to None
                return None

        B_ols = None
        if self.distribution in ("plogis", "pnorm"):
            from .transforms import bc_transform

            yr = bc_transform(y, 0.01)
            if i_order > 0:
                yr = np.diff(yr, n=i_order)
            B_ols = _ols_init(X_init, yr)
        elif self.distribution == "dpois":
            if i_order > 0:
                # ARI: log-link OLS (matches R's I(d) branch)
                yr = np.log(np.maximum(y, 1e-10))
                yr = np.diff(yr, n=i_order)
                B_ols = _ols_init(X_init, yr)
            else:
                # I(0): quasi-Newton step
                mu_insample = np.mean(y)
                try:
                    XtX_mu = X_init.T @ X_init * mu_insample
                    B_ols = np.linalg.solve(XtX_mu, X_init.T @ (y - mu_insample))
                except Exception:  # noqa: BLE001 - fall back to log-link OLS
                    yr = np.log(np.maximum(y, 1e-10))
                    B_ols = _ols_init(X_init, yr)
        elif self.distribution in (
            "dlnorm",
            "dllaplace",
            "dls",
            "dlgnorm",
            *log_link_distributions,
        ):
            if i_order > 0:
                yr = np.log(np.maximum(y, 1e-10))
                yr = np.diff(yr, n=i_order)
                B_ols = _ols_init(X_init, yr)
            else:
                mask = y > 0
                if mask.any():
                    B_ols = _ols_init(X_init[mask], np.log(y[mask]))
        elif self.distribution in ("dlogitnorm", "dbeta"):
            y_clip = np.clip(y, 1e-10, 1 - 1e-10)
            yr = np.log(y_clip / (1 - y_clip))
            if i_order > 0:
                yr = np.diff(yr, n=i_order)
            B_ols = _ols_init(X_init, yr)
        elif self.distribution == "dchisq":
            yr = np.sqrt(y)
            if i_order > 0:
                yr = np.diff(yr, n=i_order)
            B_ols = _ols_init(X_init, yr)
        elif self.distribution == "dbcnorm":
            from .transforms import bc_transform

            lam = self.lambda_bc if a_parameter_provided else 0.1
            yr = bc_transform(y, lam)
            if i_order > 0:
                yr = np.diff(yr, n=i_order)
            B_ols = _ols_init(X_init, yr)
        else:
            # default: dnorm, dt, dlaplace, dalaplace, ds, dfnorm, drectnorm,
            # dgnorm, dgeom, ...
            yr = y
            if i_order > 0:
                yr = np.diff(yr, n=i_order)
            # If occurrence is a fitted ALM (hurdle model), the size model
            # should fit non-zero observations only; initialise from that
            # subset so the optimiser doesn't get pulled towards mean(y)
            # which includes the zeros.
            if isinstance(self.occurrence, ALM) and i_order == 0:
                mask = y != 0
                if mask.any():
                    B_ols = _ols_init(X_init[mask], yr[mask])
                else:
                    B_ols = _ols_init(X_init, yr)
            else:
                B_ols = _ols_init(X_init, yr)

        # ── Step 3: unified placement ──────────────────────────────────────────
        if B_ols is None:
            pass  # keep B_init = zeros
        elif self.distribution == "dbeta":
            B_init[:n_params_base] = B_ols
            B_init[n_params_base:] = -B_ols
        elif (
            self.distribution in distributions_with_extra_param
            and not a_parameter_provided
        ):
            # Extra param in slot 0; OLS coefficients in slots 1:
            B_init[1:] = B_ols
            extra_inits = {
                "dfnorm": np.std(y, ddof=1),
                "drectnorm": np.std(y, ddof=1),
                "dalaplace": 0.5,
                "dgnorm": 2.0,
                "dlgnorm": 2.0,
                "dnbinom": np.var(y, ddof=1),
                "dchisq": 1.0,
                "dt": 2.0,
                "dbcnorm": 0.1,
            }
            B_init[0] = extra_inits.get(self.distribution, 0.0)
        else:
            B_init = B_ols

        # If occurrence is a pre-fitted ALM, switch to hurdle / mixture mode:
        # fit the size model on the non-zero subset, mirroring R's
        # `alm(y~., data, occurrence=<fitted alm>)`.
        occurrence_is_model = isinstance(self.occurrence, ALM)
        if occurrence_is_model:
            occ_fitted = np.asarray(self.occurrence.fitted_values_, dtype=float)
            if len(occ_fitted) != n_samples:
                raise ValueError(
                    "occurrence ALM has fitted_values_ of length "
                    f"{len(occ_fitted)}, expected {n_samples}"
                )
            otU_full = y != 0
            obs_nonzero = int(otU_full.sum())
            obs_zero = int(n_samples - obs_nonzero)
        else:
            occ_fitted = None
            otU_full = np.ones(n_samples, dtype=bool)
            obs_nonzero = n_samples
            obs_zero = 0

        print_level = self.nlopt_kargs.get("print_level", 0)
        iteration_count = [0]
        last_cf_value = [None]

        def objective_func(B, grad):
            if grad.size > 0:
                grad[:] = 0

            cf_value = cf(
                B,
                self.distribution,
                self.loss,
                y,
                X,
                ar_order=ar_order,
                i_order=i_order,
                lambda_val=(
                    self.lambda_l1
                    if self.loss == "LASSO"
                    else (self.lambda_l2 if self.loss == "RIDGE" else 0.0)
                ),
                other=other_val,
                a_parameter_provided=a_parameter_provided,
                trim=self.trim,
                lambda_bc=(
                    self.lambda_bc
                    if self.distribution == "dbcnorm" and a_parameter_provided
                    else 0.0
                ),
                size=self.size if self.distribution == "dbinom" else 1.0,
                otU=otU_full if occurrence_is_model else None,
                obs_zero=obs_zero,
                obs_nonzero=obs_nonzero,
                occurrence_model=occurrence_is_model,
            )

            if print_level > 0:
                iteration_count[0] += 1
                last_cf_value[0] = cf_value
                if iteration_count[0] % print_level == 0:
                    print(f"Iteration {iteration_count[0]}: B = {B}, CF = {cf_value}")

            return cf_value

        algorithm_name = self.nlopt_kargs.get("algorithm", "NLOPT_LN_NELDERMEAD")
        if algorithm_name not in NLOPT_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {algorithm_name}. "
                f"Choose from: {list(NLOPT_ALGORITHMS.keys())}"
            )

        algorithm = NLOPT_ALGORITHMS[algorithm_name]

        opt = nlopt.opt(algorithm, n_params)

        if (
            self.distribution in distributions_with_extra_param
            and not a_parameter_provided
        ):
            lower_bounds = [0.0] + [-np.inf] * n_params_base
            upper_bounds = [np.inf] * (n_params_base + 1)
            if self.distribution in ("dalaplace", "dbcnorm"):
                upper_bounds[0] = 1.0
        else:
            lower_bounds = [-np.inf] * n_params
            upper_bounds = [np.inf] * n_params

        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        opt.set_min_objective(objective_func)

        opt.set_xtol_rel(self.nlopt_kargs.get("xtol_rel", 1e-6))
        opt.set_xtol_abs(self.nlopt_kargs.get("xtol_abs", 1e-8))
        opt.set_ftol_rel(self.nlopt_kargs.get("ftol_rel", 1e-4))
        opt.set_ftol_abs(self.nlopt_kargs.get("ftol_abs", 0))

        maxeval = self.nlopt_kargs.get("maxeval")

        if maxeval is None:
            if self.distribution in ("dnorm", "dlnorm", "dlogitnorm") and self.loss in (
                "likelihood",
                "MSE",
            ):
                maxeval = 1
            else:
                maxeval = n_params * 40
            if self.loss in ("LASSO", "RIDGE"):
                maxeval = n_params * 200
                if self.lambda_l1 == 1.0 or self.loss == "RIDGE":
                    maxeval = 1

        opt.set_maxeval(maxeval)
        opt.set_maxtime(self.nlopt_kargs.get("maxtime", 600))

        try:
            B_opt = opt.optimize(B_init)
            self.n_iter_ = opt.get_numevals()
            self._result = opt.last_optimum_value()
            self.nlopt_result_ = opt.last_optimize_result()
        except Exception as e:  # noqa: BLE001 - report any optimiser failure, keep init
            if self.verbose > 0:
                print(f"Optimization failed: {e}")
            B_opt = B_init
            self.n_iter_ = 0
            self._result = None
            self.nlopt_result_ = None

        distributions_with_extra_param = (
            "dalaplace",
            "dnbinom",
            "dchisq",
            "dfnorm",
            "drectnorm",
            "dgnorm",
            "dlgnorm",
            "dbcnorm",
            "dt",
        )

        if n_features > 0:
            if (
                self.distribution in distributions_with_extra_param
                and not a_parameter_provided
            ):
                self.intercept_ = B_opt[1]
                self._coef = B_opt[2:] if len(B_opt) > 2 else np.array([])
            elif self.distribution == "dbeta":
                self.intercept_ = B_opt[0]
                self._coef = (
                    B_opt[1:n_params_base] if n_params_base > 1 else np.array([])
                )
            else:
                self.intercept_ = B_opt[0]
                self._coef = B_opt[1:] if len(B_opt) > 1 else np.array([])

        if (
            self.distribution in distributions_with_extra_param
            and not a_parameter_provided
        ):
            B_for_mu = B_opt[1:]
            other_val = B_opt[0]
        elif self.distribution == "dbeta":
            B_for_mu = B_opt[:n_params_base]
        else:
            B_for_mu = B_opt

        # For ARI models, apply polynomial expansion via fitter() to get mu.
        # For non-ARI models, compute mu directly from the linear predictor.
        if i_order > 0:
            from .fitters import fitter as _fitter_func

            fitter_result = _fitter_func(
                B_for_mu,
                self.distribution,
                y,
                X,
                other=other_val,
                a_parameter_provided=True,
                ar_order=ar_order,
                i_order=i_order,
            )
            mu_computed = fitter_result["mu"]
        else:
            linear_pred = X @ B_for_mu
            if (
                self.distribution
                in (
                    "dinvgauss",
                    "dgamma",
                    "dexp",
                    "dpois",
                    "dnbinom",
                    "dbinom",
                    "dgeom",
                )
                or self.distribution == "dbeta"
            ):
                mu_computed = np.exp(linear_pred)
            elif self.distribution == "dchisq":
                # The non-centrality, as in the fitter (R: (X B)^2)
                mu_computed = np.where(linear_pred < 0, 1e100, linear_pred**2)
            else:
                mu_computed = linear_pred

        fitter_return = {
            "mu": mu_computed,
            "scale": 1.0,
            "other": other_val,
        }

        scale = scaler_internal(
            B_opt,
            self.distribution,
            y,
            X,
            fitter_return["mu"],
            other_val,
            otU_full,
            obs_nonzero,
        )
        fitter_return["scale"] = scale
        self.other_ = other_val

        # Store ARI polynomial and ARIMA descriptor
        if ar_order > 0 or i_order > 0:
            # phi values are the last ar_order elements of B_for_mu
            ar_phi = B_for_mu[-ar_order:] if ar_order > 0 else np.array([])
            if ar_order > 0 and i_order > 0:
                # Combined ARI polynomial: convolve I(d) and AR(p) polynomials
                poly1_final = np.ones(ar_order + 1)
                poly1_final[1:] = -ar_phi
                poly2_final = np.array([1.0, -1.0])
                for _ in range(i_order - 1):
                    poly2_final = np.convolve(poly2_final, np.array([1.0, -1.0]))
                combined = -np.convolve(poly2_final, poly1_final)[1:]
            elif i_order > 0:
                # Pure I(d): polynomial is -poly2[1:] = ones (all +1)
                poly2_final = np.array([1.0, -1.0])
                for _ in range(i_order - 1):
                    poly2_final = np.convolve(poly2_final, np.array([1.0, -1.0]))
                combined = -poly2_final[1:]
            else:
                # Pure AR(p): coefficients are the phi values directly
                combined = ar_phi
            lag_names = [f"{response_name}Lag{i}" for i in range(1, ari_order + 1)]
            self.arima_polynomial_ = dict(zip(lag_names, combined))
            self.arima_string_ = f"ARIMA({ar_order},{i_order},0)"
        else:
            self.arima_polynomial_ = None
            self.arima_string_ = None

        lambda_bc_val = (
            other_val
            if self.distribution == "dbcnorm" and not a_parameter_provided
            else (self.lambda_bc if self.distribution == "dbcnorm" else 0.0)
        )

        # The location of the distribution, R's object$mu
        self.mu_ = np.asarray(fitter_return["mu"], dtype=float)
        self.fitted_values_ = extractor_fitted(
            self.distribution,
            fitter_return["mu"],
            scale,
            lambda_bc_val,
        )

        self.residuals_ = extractor_residuals(
            self.distribution,
            fitter_return["mu"],
            y,
            lambda_bc_val,
        )

        if occurrence_is_model:
            self.fitted_values_ = self.fitted_values_ * occ_fitted
            self.occurrence_ = self.occurrence

        self._loss_value = objective_func(B_opt, np.zeros(n_params))

        if self.loss == "likelihood":
            self._log_lik = -self._loss_value
            if self.distribution == "dbeta":
                n_params_calc = 2 * n_params_base
            else:
                n_params_calc = n_params_base
                if self.distribution not in (
                    "dexp",
                    "dpois",
                    "dgeom",
                    "dbinom",
                    "plogis",
                    "pnorm",
                ):
                    n_params_calc += 1
                if (
                    self.distribution in distributions_with_extra_param
                    and not a_parameter_provided
                    and self.distribution
                    not in (
                        "dfnorm",
                        "drectnorm",
                        "dt",
                        "dchisq",
                        "dnbinom",
                    )
                ):
                    n_params_calc += 1

            if occurrence_is_model:
                self._log_lik = self._log_lik + (self.occurrence.log_lik or 0.0)
                n_params_calc = n_params_calc + self.occurrence.nparam

            self._aic = 2 * n_params_calc - 2 * self._log_lik
            self._bic = n_params_calc * np.log(n_samples) - 2 * self._log_lik

            if n_samples - n_params_calc - 1 > 0:
                self._aicc = self._aic + (2 * n_params_calc * (n_params_calc + 1)) / (
                    n_samples - n_params_calc - 1
                )
                self._bicc = -2 * self._log_lik + (
                    n_params_calc * np.log(n_samples) * n_samples
                ) / (n_samples - n_params_calc - 1)
            else:
                self._aicc = np.nan
                self._bicc = np.nan

        self._df_residual = n_samples - n_params - 1
        self._X_train_ = X.copy()
        self._y_train_ = y.copy()

        scale = scaler_internal(
            B_opt,
            self.distribution,
            y,
            X,
            fitter_return["mu"],
            other_val,
            otU_full,
            obs_nonzero,
        )
        # The scale of the Normal-based distributions is the variance (ADAM monograph)
        if self.distribution in VARIANCE_SCALE_DISTRIBUTIONS:
            scale = scale**2
        self._scale = scale

        XtX = X.T @ X
        XtX += np.eye(XtX.shape[0]) * 1e-10
        self._XtX_inv_ = np.linalg.inv(XtX)

        # Store optimization state for Hessian-based vcov
        self._B_opt_ = B_opt.copy()
        self._a_parameter_provided_ = a_parameter_provided
        self._other_val_ = other_val

        self._time_elapsed = time_module.time() - start_time

        return self

    def vcov(self) -> np.ndarray:
        """Calculate variance-covariance matrix of parameter estimates.

        Uses distribution-specific methods matching R's vcov.alm():
        - Normal-like + likelihood/MSE: sigma^2 * (X'X)^-1
        - Poisson + likelihood: inverse Fisher information
        - Everything else: inverse numerical Hessian of cost function

        Returns
        -------
        vcov_matrix : np.ndarray
            Covariance matrix of shape (n_params, n_params)
        """
        if self._X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._X_train_
        n_vars = X.shape[1]
        distribution = self.distribution
        loss = self.loss

        # Path 1: Analytical for normal-like distributions with likelihood/MSE
        normal_like = ("dnorm", "dlnorm", "dbcnorm", "dlogitnorm")
        if (distribution in normal_like and loss == "likelihood") or loss == "MSE":
            XtX = X.T @ X
            try:
                L = np.linalg.cholesky(XtX)
                XtX_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n_vars)))
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv(XtX)
            return self.sigma**2 * XtX_inv

        # Path 2: Analytical for Poisson with likelihood
        if distribution == "dpois" and loss == "likelihood":
            mu = self.fitted
            FI = np.zeros((n_vars, n_vars))
            for i in range(len(mu)):
                xi = X[i]
                FI += np.outer(xi, xi) * mu[i]
            try:
                L = np.linalg.cholesky(FI)
                return np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n_vars)))
            except np.linalg.LinAlgError:
                return np.linalg.pinv(FI)

        # Path 3: Hessian-based for everything else
        return self._vcov_hessian(X, n_vars)

    def _vcov_hessian(self, X, n_vars):
        """Compute vcov via numerical Hessian of the cost function."""
        y = self._y_train_
        a_parameter_provided = self._a_parameter_provided_
        other_val = self._other_val_

        # B for the regression coefficients only (not extra param)
        if (
            self.distribution
            in (
                "dalaplace",
                "dnbinom",
                "dchisq",
                "dfnorm",
                "drectnorm",
                "dgnorm",
                "dlgnorm",
                "dbcnorm",
                "dt",
            )
            and not a_parameter_provided
        ):
            B_reg = self._B_opt_[1:]
        elif self.distribution == "dbeta":
            B_reg = self._B_opt_[:n_vars]
        else:
            B_reg = self._B_opt_

        lambda_val = (
            self.lambda_l1
            if self.loss == "LASSO"
            else (self.lambda_l2 if self.loss == "RIDGE" else 0.0)
        )
        lambda_bc_val = (
            self.lambda_bc
            if self.distribution == "dbcnorm" and a_parameter_provided
            else 0.0
        )

        def cf_wrapper(B):
            if (
                self.distribution
                in (
                    "dalaplace",
                    "dnbinom",
                    "dchisq",
                    "dfnorm",
                    "drectnorm",
                    "dgnorm",
                    "dlgnorm",
                    "dbcnorm",
                    "dt",
                )
                and not a_parameter_provided
            ):
                B_full = np.concatenate([[other_val], B])
            elif self.distribution == "dbeta":
                B_full = np.concatenate([B, self._B_opt_[n_vars:]])
            else:
                B_full = B
            return cf(
                B_full,
                self.distribution,
                self.loss,
                y,
                X,
                ar_order=0,
                i_order=0,
                lambda_val=lambda_val,
                other=other_val,
                a_parameter_provided=a_parameter_provided,
                trim=self.trim,
                lambda_bc=lambda_bc_val,
                size=self.size if self.distribution == "dbinom" else 1.0,
            )

        FI = _numerical_hessian(cf_wrapper, B_reg)

        # Check for broken variables (all-zero or NaN rows)
        broken = np.all(FI == 0, axis=1) | np.any(np.isnan(FI), axis=1)
        if np.any(broken):
            FI = _numerical_hessian(cf_wrapper, B_reg, h=np.finfo(float).eps ** (1 / 6))

        try:
            L = np.linalg.cholesky(FI)
            vcov = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n_vars)))
        except np.linalg.LinAlgError:
            try:
                vcov = np.linalg.inv(FI)
            except np.linalg.LinAlgError:
                vcov = np.linalg.pinv(FI)

        # Fix negative diagonal elements
        neg_diag = np.diag(vcov) < 0
        if np.any(neg_diag):
            diag_vals = np.diag(vcov).copy()
            diag_vals[neg_diag] = np.abs(diag_vals[neg_diag])
            np.fill_diagonal(vcov, diag_vals)

        return vcov

    def _calculate_variance(self, X, interval="confidence"):
        """Calculate conditional variance for prediction intervals.

        Parameters
        ----------
        X : np.ndarray
            Design matrix for predictions.
        interval : str
            Type of interval: "confidence" or "prediction".

        Returns
        -------
        variances : np.ndarray
            Variance values for each observation.
        """
        vcov_matrix = self.vcov()
        variances = np.diag(X @ vcov_matrix @ X.T)

        if interval == "prediction":
            return variances + (self.sigma**2)
        return variances

    def _calculate_quantiles(self, linear_pred, variances, interval, level, side):
        """Calculate lower and upper quantiles for prediction intervals.

        Accepts the raw linear predictor (before back-transformation) and
        branches by distribution to match R's predict.alm() behaviour:
        - symmetric distributions: t-interval in LP space
        - log-link distributions: exponentiate t-bounds for CI, use
          distribution quantile for PI
        - log-normal family: use the distribution quantile function directly
          for both CI and PI (se encodes the correct total uncertainty)
        - link-function distributions (plogis/pnorm): apply CDF to normal
          quantile of LP

        Parameters
        ----------
        linear_pred : np.ndarray
            Raw linear predictor values (before back-transformation).
        variances : np.ndarray
            Variance on LP scale.  For confidence intervals this is the
            parameter-estimation variance; for prediction intervals this
            also includes sigma² (the observation noise variance).
        interval : str
            Type of interval: "confidence" or "prediction".
        level : float or list
            Confidence level(s).
        side : str
            Side of interval: "both", "upper", or "lower".

        Returns
        -------
        lower : np.ndarray or None
            Lower bounds (in response scale).
        upper : np.ndarray or None
            Upper bounds (in response scale).
        """
        if interval == "none":
            return None, None

        if not isinstance(level, list):
            level = [level]

        n_levels = len(level)
        n_obs = len(linear_pred)
        # SE on linear predictor scale; for PI this includes sigma²
        se = np.sqrt(variances)

        if side == "upper":
            level_low = [0.0] * n_levels
            level_up = level
        elif side == "lower":
            level_low = [1 - lev for lev in level]
            level_up = [1.0] * n_levels
        else:
            level_low = [(1 - lev) / 2 for lev in level]
            level_up = [(1 + lev) / 2 for lev in level]

        t_low = stats.t.ppf(level_low, df=self.df_residual_)
        t_up = stats.t.ppf(level_up, df=self.df_residual_)

        lower = np.zeros((n_obs, n_levels))
        upper = np.zeros((n_obs, n_levels))

        d = self.distribution

        # Effective lambda_bc for dbcnorm (fitted or user-provided)
        lambda_bc_eff = None
        if d == "dbcnorm":
            lambda_bc_eff = (
                self.other_ if not self._a_parameter_provided_ else self.lambda_bc
            )

        for i in range(n_levels):
            ll = level_low[i]
            lu = level_up[i]
            tl = t_low[i]
            tu = t_up[i]

            # t-interval bounds in linear predictor space
            lp_l = linear_pred + tl * se
            lp_u = linear_pred + tu * se

            # ── Group A: symmetric, LP == response ──────────────────────────
            if d in ("dnorm", "dt"):
                lower[:, i] = lp_l
                upper[:, i] = lp_u

            # ── Group B: CI=t-based LP; PI=distribution quantile ────────────
            elif d == "dlaplace":
                if interval == "confidence":
                    lower[:, i] = lp_l
                    upper[:, i] = lp_u
                else:
                    # Laplace variance = 2*b²  →  b = sqrt(var/2)
                    b = np.sqrt(variances / 2)
                    lower[:, i] = dist.qlaplace(ll, loc=linear_pred, scale=b)
                    upper[:, i] = dist.qlaplace(lu, loc=linear_pred, scale=b)

            elif d == "dalaplace":
                if interval == "confidence":
                    lower[:, i] = lp_l
                    upper[:, i] = lp_u
                else:
                    alpha = self.other_
                    # ALaplace var = scale²*(α²+(1-α)²)/(α(1-α))²
                    scale_al = np.sqrt(
                        variances
                        * (alpha * (1 - alpha)) ** 2
                        / (alpha**2 + (1 - alpha) ** 2)
                    )
                    lower[:, i] = dist.qalaplace(
                        ll, loc=linear_pred, scale=scale_al, alpha=alpha
                    )
                    upper[:, i] = dist.qalaplace(
                        lu, loc=linear_pred, scale=scale_al, alpha=alpha
                    )

            elif d == "ds":
                if interval == "confidence":
                    lower[:, i] = lp_l
                    upper[:, i] = lp_u
                else:
                    # S-distribution variance = 120*scale⁴  →  scale=(var/120)^0.25
                    s_scale = (variances / 120) ** 0.25
                    lower[:, i] = dist.qs(ll, loc=linear_pred, scale=s_scale)
                    upper[:, i] = dist.qs(lu, loc=linear_pred, scale=s_scale)

            elif d == "dlogis":
                if interval == "confidence":
                    lower[:, i] = lp_l
                    upper[:, i] = lp_u
                else:
                    # Logistic variance = π²*scale²/3  →  scale=sqrt(var*3/π²)
                    logis_scale = np.sqrt(variances * 3 / np.pi**2)
                    lower[:, i] = dist.qlogis(ll, loc=linear_pred, scale=logis_scale)
                    upper[:, i] = dist.qlogis(lu, loc=linear_pred, scale=logis_scale)

            elif d == "dgnorm":
                if interval == "confidence":
                    lower[:, i] = lp_l
                    upper[:, i] = lp_u
                else:
                    # GNorm variance = Γ(3/s)/Γ(1/s)*scale²
                    s = self.other_
                    gn_scale = np.sqrt(variances * _sp_gamma(1 / s) / _sp_gamma(3 / s))
                    lower[:, i] = dist.qgnorm(
                        ll, loc=linear_pred, scale=gn_scale, shape=s
                    )
                    upper[:, i] = dist.qgnorm(
                        lu, loc=linear_pred, scale=gn_scale, shape=s
                    )

            elif d == "dfnorm":
                if interval == "confidence":
                    lower[:, i] = lp_l
                    upper[:, i] = lp_u
                else:
                    # qfnorm requires scalar mu/sigma; compute element-wise
                    lower[:, i] = np.array(
                        [
                            float(dist.qfnorm(ll, loc=float(mj), scale=float(sj)))
                            for mj, sj in zip(linear_pred, se)
                        ]
                    )
                    upper[:, i] = np.array(
                        [
                            float(dist.qfnorm(lu, loc=float(mj), scale=float(sj)))
                            for mj, sj in zip(linear_pred, se)
                        ]
                    )

            elif d == "drectnorm":
                if interval == "confidence":
                    lower[:, i] = lp_l
                    upper[:, i] = lp_u
                else:
                    lower[:, i] = dist.qrectnorm(ll, loc=linear_pred, scale=se)
                    upper[:, i] = dist.qrectnorm(lu, loc=linear_pred, scale=se)

            # ── Group C: log-link; CI=exp(t-bounds); PI=dist quantile ───────
            elif d == "dinvgauss":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    lower[:, i] = np.exp(linear_pred) * dist.qinvgauss(
                        ll, loc=1, scale=self.scale
                    )
                    upper[:, i] = np.exp(linear_pred) * dist.qinvgauss(
                        lu, loc=1, scale=self.scale
                    )

            elif d == "dgamma":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    lower[:, i] = np.exp(linear_pred) * dist.qgamma(
                        ll, shape=1 / self.scale, scale=self.scale
                    )
                    upper[:, i] = np.exp(linear_pred) * dist.qgamma(
                        lu, shape=1 / self.scale, scale=self.scale
                    )

            elif d == "dexp":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    # Exp(rate=1) has mean 1; scale by fitted mean
                    lower[:, i] = np.exp(linear_pred) * dist.qexp(ll)
                    upper[:, i] = np.exp(linear_pred) * dist.qexp(lu)

            elif d == "dpois":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    lower[:, i] = dist.qpois(ll, loc=np.exp(linear_pred))
                    upper[:, i] = dist.qpois(lu, loc=np.exp(linear_pred))

            elif d == "dnbinom":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    # self.scale holds the size parameter (abs(other))
                    lower[:, i] = dist.qnbinom(
                        ll, loc=np.exp(linear_pred), size=self.scale
                    )
                    upper[:, i] = dist.qnbinom(
                        lu, loc=np.exp(linear_pred), size=self.scale
                    )

            elif d == "dbinom":
                binom_size = self.size if self.size is not None else 1
                lower[:, i] = np.exp(lp_l) * binom_size
                upper[:, i] = np.exp(lp_u) * binom_size
                if interval == "prediction":
                    prob = 1.0 / (1.0 + np.exp(linear_pred))
                    lower[:, i] = dist.qbinom(ll, size=binom_size, prob=prob)
                    upper[:, i] = dist.qbinom(lu, size=binom_size, prob=prob)

            # ── Group D: log-transformed; CI=exp(t-bounds); PI=dist quantile ─
            elif d == "dllaplace":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    b = np.sqrt(variances / 2)
                    lower[:, i] = dist.qllaplace(ll, loc=linear_pred, scale=b)
                    upper[:, i] = dist.qllaplace(lu, loc=linear_pred, scale=b)

            elif d == "dls":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    s_scale = (variances / 120) ** 0.25
                    lower[:, i] = dist.qls(ll, loc=linear_pred, scale=s_scale)
                    upper[:, i] = dist.qls(lu, loc=linear_pred, scale=s_scale)

            elif d == "dlgnorm":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    beta = self.other_
                    lgn_scale = np.sqrt(
                        variances * _sp_gamma(1 / beta) / _sp_gamma(3 / beta)
                    )
                    lower[:, i] = dist.qlgnorm(
                        ll, loc=linear_pred, scale=lgn_scale, shape=beta
                    )
                    upper[:, i] = dist.qlgnorm(
                        lu, loc=linear_pred, scale=lgn_scale, shape=beta
                    )

            # ── Group E: log-normal family; use dist quantile for CI and PI ──
            elif d == "dlnorm":
                # qlnorm(p, meanlog=lp, sdlog=se) where se encodes the right
                # uncertainty: for CI se=sqrt(var_pred), for PI se=sqrt(var_pred+σ²)
                lower[:, i] = dist.qlnorm(ll, loc=linear_pred, scale=se)
                upper[:, i] = dist.qlnorm(lu, loc=linear_pred, scale=se)

            elif d == "dlogitnorm":
                lower[:, i] = dist.qlogitnorm(ll, loc=linear_pred, scale=se)
                upper[:, i] = dist.qlogitnorm(lu, loc=linear_pred, scale=se)

            # ── Group F: special transformations ────────────────────────────
            elif d == "dbcnorm":
                if interval == "confidence":
                    # Invert BC transform of t-bounds in LP space
                    lower[:, i] = _bc_transform_inv(lp_l, lambda_bc_eff)
                    upper[:, i] = _bc_transform_inv(lp_u, lambda_bc_eff)
                else:
                    lower[:, i] = dist.qbcnorm(
                        ll, loc=linear_pred, scale=se, lambda_bc=lambda_bc_eff
                    )
                    upper[:, i] = dist.qbcnorm(
                        lu, loc=linear_pred, scale=se, lambda_bc=lambda_bc_eff
                    )

            elif d == "dchisq":
                # CI: square the t-bounds; PI: non-central chi-squared
                lower[:, i] = lp_l**2
                upper[:, i] = lp_u**2
                if interval == "prediction":
                    lower[:, i] = stats.ncx2.ppf(ll, df=self.other_, nc=linear_pred**2)
                    upper[:, i] = stats.ncx2.ppf(lu, df=self.other_, nc=linear_pred**2)

            # ── Group G: link-function distributions (plogis/pnorm) ─────────
            elif d == "plogis":
                # Apply plogis to normal quantile of LP
                lower[:, i] = dist.plogis(
                    dist.qnorm(ll, loc=linear_pred, scale=se),
                    loc=0.0,
                    scale=1.0,
                )
                upper[:, i] = dist.plogis(
                    dist.qnorm(lu, loc=linear_pred, scale=se),
                    loc=0.0,
                    scale=1.0,
                )

            elif d == "pnorm":
                lower[:, i] = dist.pnorm(
                    dist.qnorm(ll, loc=linear_pred, scale=se), loc=0.0, scale=1.0
                )
                upper[:, i] = dist.pnorm(
                    dist.qnorm(lu, loc=linear_pred, scale=se), loc=0.0, scale=1.0
                )

            else:
                # Fallback: t-based (dgeom and any unhandled distributions)
                lower[:, i] = lp_l
                upper[:, i] = lp_u

        if n_levels == 1:
            lower = lower.ravel()
            upper = upper.ravel()

        if side == "upper":
            lower = None
        elif side == "lower":
            upper = None

        return lower, upper

    def predict(
        self,
        X: np.ndarray,
        interval: Literal["none", "confidence", "prediction"] = "none",
        level: float | list[float] = 0.95,
        side: Literal["both", "upper", "lower"] = "both",
    ) -> PredictionResult:
        """Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix. Should have same number of features as training data.
        interval : {"none", "confidence", "prediction"}, default="none"
            Type of interval to calculate:
            - "none": No intervals, return only point forecasts
            - "confidence": Confidence interval for the mean
            - "prediction": Prediction interval for new observations
        level : float or list of float, default=0.95
            Confidence level(s) for intervals. Can be a single float (e.g., 0.95)
            or a list of floats (e.g., [0.8, 0.9, 0.95]). Default is 0.95 (95%).
        side : {"both", "upper", "lower"}, default="both"
            Side of interval:
            - "both": Return both lower and upper bounds
            - "upper": Return only upper bounds
            - "lower": Return only lower bounds

        Returns
        -------
        PredictionResult
            Object with the following attributes:
            - mean : np.ndarray - Predicted values (point forecasts)
            - lower : np.ndarray or None - Lower prediction bounds
            - upper : np.ndarray or None - Upper prediction bounds
        """
        if self.fitted_values_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if interval not in ("none", "confidence", "prediction"):
            raise ValueError(
                f"interval must be 'none', 'confidence', "
                f"or 'prediction', got {interval!r}"
            )

        if side not in ("both", "upper", "lower"):
            raise ValueError(f"side must be 'both', 'upper', or 'lower', got {side!r}")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self._coef is not None and len(self._coef) > 0:
            B = np.concatenate([[self.intercept_], self._coef])
        else:
            B = np.array([self.intercept_])

        _log_link_distributions = (
            "dinvgauss",
            "dgamma",
            "dexp",
            "dpois",
            "dnbinom",
            "dbinom",
            "dgeom",
        )

        if self._ari_order > 0:
            # AR(p) or ARI(p,d): user passes exogenous X only; lag columns are
            # added internally via in-sample reuse or recursive forecasting.
            ar_coefs = np.array(list(self.arima_polynomial_.values()))
            n_exog = self._n_features_exog
            ari_order = self._ari_order
            lambda_bc_val = self.lambda_bc if self.distribution == "dbcnorm" else 0.0

            if X.shape[1] != n_exog:
                raise ValueError(
                    f"X has {X.shape[1]} features, expected {n_exog} "
                    f"(exogenous features only; ARI lag columns are added internally)"
                )

            if len(X) == len(self._X_train_):
                # In-sample: reuse training lag columns (matches fitted values exactly)
                X_lags = self._X_train_[:, n_exog:]
                linear_pred = X @ B[:n_exog] + X_lags @ ar_coefs
            else:
                # Out-of-sample: recursive forecasting
                h = len(X)
                X_lags = np.zeros((h, ari_order))
                y_train = self._y_train_

                # Seed lag matrix from tail of training data (raw y-scale)
                for j in range(1, ari_order + 1):  # j = lag number
                    for t in range(min(h, j)):  # t = forecast horizon step
                        X_lags[t, j - 1] = y_train[-(j - t)]

                linear_pred = np.empty(h)
                for i in range(h):
                    linear_pred[i] = float(X[i] @ B[:n_exog]) + float(
                        X_lags[i] @ ar_coefs
                    )
                    # Compute raw y-scale fitted value to feed into future lags
                    lp = linear_pred[i]
                    if (
                        self.distribution in _log_link_distributions
                        or self.distribution == "dbeta"
                    ):
                        mu_i = np.exp(lp)
                    elif self.distribution == "dchisq":
                        mu_i = 1e100 if lp < 0 else lp**2
                    else:
                        mu_i = lp
                    fitted_i = float(
                        extractor_fitted(
                            self.distribution,
                            np.array([mu_i]),
                            scale_sd(self.distribution, self.scale)
                            if self.scale is not None
                            else 1.0,
                            lambda_bc_val,
                        )[0]
                    )
                    for j in range(1, ari_order + 1):
                        if i + j < h:
                            X_lags[i + j, j - 1] = fitted_i

            X = np.hstack([X, X_lags])  # full X for variance; (h, n_exog + ari_order)
        else:
            if X.shape[1] != self._n_features:
                raise ValueError(
                    f"X has {X.shape[1]} features, "
                    f"but model was fitted with {self._n_features}"
                )
            linear_pred = X @ B
        if self.distribution in _log_link_distributions:
            mu = np.exp(linear_pred)
        elif self.distribution == "dchisq":
            mu = np.where(linear_pred < 0, 1e100, linear_pred**2)
        elif self.distribution == "dbeta":
            mu = np.exp(linear_pred)
        else:
            mu = linear_pred

        mean = extractor_fitted(
            self.distribution,
            mu,
            scale_sd(self.distribution, self.scale) if self.scale is not None else 1.0,
            self.lambda_bc if self.distribution == "dbcnorm" else 0.0,
        )

        variances = None
        if interval == "none":
            lower = None
            upper = None
        else:
            # Compute confidence variance once; add sigma² for prediction.
            variances_conf = self._calculate_variance(X, "confidence")
            if interval == "prediction":
                variances = variances_conf + self.sigma**2
            else:
                variances = variances_conf

            # For dfnorm/drectnorm, correct the mean using predictor SE
            # (R uses sqrt(var_pred) rather than self.scale for this correction)
            if self.distribution == "dfnorm":
                se_conf = np.sqrt(variances_conf)
                mean = np.sqrt(2 / np.pi) * se_conf * np.exp(
                    -(linear_pred**2) / (2 * se_conf**2)
                ) + linear_pred * (1 - 2 * dist.pnorm(-linear_pred / se_conf))
            elif self.distribution == "drectnorm":
                se_conf = np.sqrt(variances_conf)
                mean = linear_pred * (
                    1 - dist.pnorm(0, loc=linear_pred, scale=se_conf)
                ) + se_conf * dist.dnorm(0, loc=linear_pred, scale=se_conf)

            lower, upper = self._calculate_quantiles(
                linear_pred, variances, interval, level, side
            )

        return PredictionResult(
            mean=mean,
            lower=lower,
            upper=upper,
            level=level,
            variances=variances,
            side=side,
            interval=interval,
        )

    def score(self, X, y, metric="likelihood"):
        """Calculate model score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix.
        y : array-like of shape (n_samples,)
            True values.
        metric : str, optional
            Metric to use: "likelihood", "MSE", "MAE", or "R2".

        Returns
        -------
        score : float
            Score value.
        """
        y_pred_result = self.predict(X)
        y_pred = y_pred_result.mean

        if metric == "likelihood":
            if self.log_lik is not None:
                return self.log_lik
            y = np.asarray(y)
            log_lik = 0.0
            y_otU = (
                y[y != 0]
                if self.distribution not in ("dlnorm", "dllaplace", "dls", "dlgnorm")
                else y
            )
            mu_otU = (
                y_pred[y != 0]
                if self.distribution not in ("dlnorm", "dllaplace", "dls", "dlgnorm")
                else y_pred
            )
            if self.distribution == "dnorm":
                log_lik = np.sum(
                    dist.dnorm(
                        y_otU,
                        mean=mu_otU,
                        sd=scale_sd(self.distribution, self.scale),
                        log=True,
                    )
                )
            return log_lik
        elif metric == "MSE":
            return np.mean((y - y_pred) ** 2)
        elif metric == "MAE":
            return np.mean(np.abs(y - y_pred))
        elif metric == "R2":
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @property
    def nobs(self) -> int:
        """Number of observations.

        Returns
        -------
        nobs : int
            Number of observations used in the model.
        """
        if self._X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._X_train_.shape[0]

    @property
    def nparam(self) -> int:
        """Number of parameters.

        Returns
        -------
        nparam : int
            Number of parameters in the model (including intercept and scale).
        """
        if self._X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        n_params = self._n_params_
        if self.distribution not in (
            "dexp",
            "dpois",
            "dgeom",
            "dbinom",
            "plogis",
            "pnorm",
        ):
            n_params += 1
        return n_params

    @property
    def residuals(self) -> np.ndarray:
        """Model residuals.

        Returns
        -------
        residuals : np.ndarray
            Residuals (y - fitted values).
        """
        if self.residuals_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.residuals_

    @property
    def fitted(self) -> np.ndarray:
        """Fitted values.

        Returns
        -------
        fitted : np.ndarray
            Fitted values (predictions on training data).
        """
        if self.fitted_values_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.fitted_values_

    @property
    def actuals(self) -> np.ndarray:
        """Actual values (response variable).

        Returns
        -------
        actuals : np.ndarray
            Actual response values from training data.
        """
        if self._y_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._y_train_

    @property
    def data(self) -> np.ndarray:
        """Alias for actuals.

        Returns
        -------
        data : np.ndarray
            Original in-sample observations.
        """
        return self.actuals

    @property
    def n_param(self) -> dict:
        """Parameter count information.

        Returns
        -------
        n_param : dict
            Dictionary containing parameter count information.
        """
        return {
            "number": self.nparam,
            "df": self.df_residual_,
        }

    @property
    def formula(self) -> str | None:
        """Formula string used to fit the model.

        Returns
        -------
        formula : str or None
            Formula string if provided during fit.
        """
        return self._formula_

    @property
    def sigma(self) -> float:
        """Residual standard error (sigma).

        Returns
        -------
        sigma : float
            Residual standard error, computed as sqrt(sum(residuals^2) / (n - k))
            where n is the number of observations and k is the number of parameters
            (including the scale parameter).
            For dinvgauss/dgamma/dexp, uses (residuals - 1) since residuals are
            on a multiplicative scale (y/mu), matching R's sigma.alm().
        """
        if self.residuals_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        n = len(self.residuals_)
        k = (
            self._n_features + 1 - self._i_order
        )  # i_order constrained lags are not free params
        resid = self.residuals_
        if self.distribution in ("dinvgauss", "dgamma", "dexp"):
            resid = resid - 1
        return np.sqrt(np.sum(resid**2) / (n - k))

    @property
    def loglik(self) -> float | None:
        """Log-likelihood (ADAM-compatible name).

        Returns
        -------
        loglik : float or None
            Log-likelihood value.
        """
        return self._log_lik

    @property
    def log_lik(self) -> float | None:
        """Log-likelihood (backward-compatible alias for loglik)."""
        return self.loglik

    @property
    def aic(self) -> float | None:
        """Akaike Information Criterion."""
        return self._aic

    @property
    def aicc(self) -> float | None:
        """Corrected Akaike Information Criterion."""
        return self._aicc

    @property
    def bic(self) -> float | None:
        """Bayesian Information Criterion."""
        return self._bic

    @property
    def bicc(self) -> float | None:
        """Corrected Bayesian Information Criterion."""
        return self._bicc

    @property
    def loss_value(self) -> float | None:
        """Final value of the loss function."""
        return self._loss_value

    @property
    def scale(self) -> float | None:
        """Scale parameter: sigma^2 for dnorm and dlnorm, as in ADAM."""
        return self._scale

    @property
    def time_elapsed(self) -> float | None:
        """Time elapsed during model fitting (seconds)."""
        return self._time_elapsed

    @time_elapsed.setter
    def time_elapsed(self, value: float):
        self._time_elapsed = value

    @property
    def df_residual_(self) -> int | None:
        """Residual degrees of freedom."""
        return self._df_residual

    @property
    def distribution_(self) -> str:
        """Distribution name (ADAM convention with trailing _)."""
        return self.distribution

    @property
    def loss_(self) -> str:
        """Loss function name (ADAM convention with trailing _)."""
        return self.loss

    @property
    def coef(self) -> np.ndarray:
        """Estimated coefficients (slope parameters, excluding intercept).

        Returns
        -------
        coef : np.ndarray
            Coefficient vector (without intercept).
        """
        if self._coef is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._coef

    @property
    def coefficients(self) -> np.ndarray:
        """All coefficients including intercept as named vector.

        Returns
        -------
        coefficients : np.ndarray
            Full coefficient vector with names (intercept + slopes).
        """
        if self._coef is None:
            raise ValueError("Model not fitted. Call fit() first.")

        names = ["(Intercept)"]
        if self._feature_names is not None:
            names.extend(self._feature_names)
        else:
            for i in range(len(self._coef)):
                names.append(f"x{i + 1}")

        result = np.concatenate([[self.intercept_], self._coef])
        return result

    def summary(self, level: float = 0.95) -> "SummaryResult":
        """Model summary.

        Parameters
        ----------
        level : float, optional
            Confidence level for parameter intervals. Default is 0.95.

        Returns
        -------
        SummaryResult
            Summary of the model with coefficient estimates, standard errors,
            t-statistics, p-values, and confidence intervals.
        """
        from .methods.summary import SummaryResult

        if self._coef is None:
            raise ValueError("Model not fitted. Call fit() first.")

        coefficients = np.concatenate([[self.intercept_], self._coef])
        vcov_matrix = self.vcov()
        # For ARI models with i_order > 0, vcov is sized for all lag columns
        # (including constrained differencing lags). Truncate to free params.
        n_free = len(coefficients)
        if vcov_matrix.shape[0] > n_free:
            vcov_matrix = vcov_matrix[:n_free, :n_free]
        se = np.sqrt(np.diag(vcov_matrix))
        t_stat = coefficients / se
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=self.df_residual_))

        t_crit = stats.t.ppf((1 + level) / 2, df=self.df_residual_)
        lower_ci = coefficients - t_crit * se
        upper_ci = coefficients + t_crit * se

        response_var = "y"
        if self._response_name is not None:
            response_var = self._response_name
        elif self._formula_ is not None and "~" in self._formula_:
            response_var = self._formula_.split("~")[0].strip()

        return SummaryResult(
            coefficients=coefficients,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            scale=self.scale,
            sigma=self.sigma,
            df_residual=self.df_residual_,
            distribution=self.distribution,
            log_lik=self.log_lik,
            aic=self.aic,
            bic=self.bic,
            nobs=self.nobs,
            nparam=self.nparam,
            response_variable=response_var,
            loss=self.loss,
            aicc=self.aicc,
            bicc=self.bicc,
            feature_names=self._feature_names,
            time_elapsed=self.time_elapsed,
            arima_string=self.arima_string_,
        )

    def confint(
        self,
        parm: int | list[int] | None = None,
        level: float = 0.95,
    ) -> np.ndarray:
        """Confidence intervals for parameters.

        Parameters
        ----------
        parm : int or list of int, optional
            Which parameters to include. If None, all parameters are included.
            0 = intercept, 1, 2, ... = coefficients.
        level : float, optional
            Confidence level. Default is 0.95.

        Returns
        -------
        confint : np.ndarray
            Array with shape (n_params, 2) containing lower and upper bounds.
        """
        if self._coef is None:
            raise ValueError("Model not fitted. Call fit() first.")

        coefficients = np.concatenate([[self.intercept_], self._coef])
        vcov_matrix = self.vcov()
        se = np.sqrt(np.diag(vcov_matrix))

        t_crit = stats.t.ppf((1 + level) / 2, df=self.df_residual_)
        lower_ci = coefficients - t_crit * se
        upper_ci = coefficients + t_crit * se

        if parm is not None:
            if isinstance(parm, int):
                parm = [parm]
            lower_ci = lower_ci[parm]
            upper_ci = upper_ci[parm]

        return np.column_stack([lower_ci, upper_ci])

    def _get_other_parameter(self):
        """Get the additional parameter for distributions that require it."""
        if self.distribution == "dalaplace":
            return self.alpha
        elif self.distribution in ("dgnorm", "dlgnorm"):
            return self.shape
        elif self.distribution == "dbcnorm":
            return self.lambda_bc
        elif self.distribution in ("dt", "dchisq"):
            return self.nu
        elif self.distribution in ("dnbinom",):
            return self.size
        else:
            return None

    def get_params(self):
        """Get model parameters.

        Returns
        -------
        params : dict
            Dictionary of model parameters.
        """
        return {
            "distribution": self.distribution,
            "loss": self.loss,
            "occurrence": self.occurrence,
            "orders": self.orders,
            "alpha": self.alpha,
            "shape": self.shape,
            "lambda_bc": self.lambda_bc,
            "size": self.size,
            "nu": self.nu,
            "trim": self.trim,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "nlopt_kargs": self.nlopt_kargs,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        """Set model parameters.

        Parameters
        ----------
        **params : dict
            Parameters to set.

        Returns
        -------
        self : ALM
            Model with updated parameters.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def multipliers(self, parm: str, h: int = 10) -> dict:
        """Compute dynamic multipliers for an ARDL model.

        Combines distributed lag coefficients (B(parm, k) terms) with the
        ARI polynomial to produce impulse-response multipliers over horizon h.

        Parameters
        ----------
        parm : str
            Variable name as it appears in the design matrix.
        h : int, default 10
            Forecast horizon.

        Returns
        -------
        dict
            {"h1": m1, "h2": m2, ..., "hh": mh} of dynamic multipliers.

        Raises
        ------
        ValueError
            If parm is not found in the model.
        """
        from .xreg import multipliers as _multipliers

        return _multipliers(self, parm, h)

    def __str__(self):
        from .methods.summary import DISTRIBUTION_NAMES

        if self._coef is None:
            return repr(self)

        dist_name = DISTRIBUTION_NAMES.get(self.distribution, self.distribution)

        lines = []
        lines.append(f"Time elapsed: {self.time_elapsed:.2f} seconds")
        lines.append(f"Model estimated: ALM({self.distribution})")
        lines.append(f"Distribution assumed in the model: {dist_name}")
        lines.append(
            f"Loss function type: {self.loss}"
            + (
                f"; Loss function value: {self.loss_value:.4f}"
                if self.loss_value is not None
                else ""
            )
        )
        lines.append("")
        lines.append(f"Sample size: {self.nobs}")
        lines.append(f"Number of estimated parameters: {self.nparam}")
        lines.append(f"Number of degrees of freedom: {self.df_residual_}")

        if self.aic is not None:
            lines.append("")
            lines.append("Information criteria:")
            ic_header = f"{'AIC':>10}{'AICc':>10}{'BIC':>10}{'BICc':>10}"
            lines.append(ic_header)
            aicc = self.aicc if self.aicc is not None else np.nan
            bicc = self.bicc if self.bicc is not None else np.nan
            ic_vals = f"{self.aic:>10.4f}{aicc:>10.4f}{self.bic:>10.4f}{bicc:>10.4f}"
            lines.append(ic_vals)

        return "\n".join(lines)

    def __repr__(self):
        if self._coef is not None:
            return f"ALM(distribution={self.distribution!r}, fitted=True)"
        return f"ALM(distribution={self.distribution!r}, fitted=False)"
