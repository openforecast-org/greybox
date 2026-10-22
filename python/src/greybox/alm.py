"""Augmented Linear Model (ALM) for greybox.

This module provides the ALM estimator following scikit-learn principles.
Users should first use the formula module to get X and y, then pass them
to the fit() method.
"""

import importlib
import time as time_module
import warnings
from typing import ClassVar, Literal

import nlopt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import comb
from scipy.special import gamma as _sp_gamma

from . import distributions as dist
from .cost_function import cf
from .fitters import (
    ARI_FILTER_DISTRIBUTIONS,
    VARIANCE_SCALE_DISTRIBUTIONS,
    _lm_fit,
    alm_latent_variance,
    extractor_fitted,
    extractor_residuals,
    fitter_recursive,
    scale_sd,
    scaler_internal,
)
from .methods.summary import SummaryResult
from .transforms import bc_transform, mean_fast
from .transforms import bc_transform_inv as _bc_transform_inv
from .xreg import xreg_expander

# The distributions whose lags of the response are in logs (Box-Cox with zeroes), as R
_ARI_LOG_DISTRIBUTIONS = (
    "dexp",
    "dinvgauss",
    "dgamma",
    "dlnorm",
    "dllaplace",
    "dls",
    "dlgnorm",
    "dgeom",
    "dpois",
    "dnbinom",
    "dbinom",
)


# The responses of the distributions, as R's alm() checks them
_NON_NEGATIVE_DISTRIBUTIONS = (
    "dexp",
    "dlnorm",
    "dllaplace",
    "dls",
    "dbcnorm",
    "dchisq",
    "dfnorm",
    "drectnorm",
    "dgeom",
    "dpois",
    "dnbinom",
    "dbinom",
    "dinvgauss",
    "dgamma",
)
_POSITIVE_DISTRIBUTIONS = (
    "dinvgauss",
    "dgamma",
    "dlnorm",
    "dllaplace",
    "dls",
    "dlgnorm",
)


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
    occurrence : PredictionResult or None
        The predictions of the occurrence model, when the model has one and the
        occurrence is not provided, as R's ``$occurrence``.
    location : np.ndarray or None
        The location of the distribution (the linear predictor), as R.
    scale : np.ndarray or float or None
        The scale of the distribution of each prediction (R's ``$scale``), when an
        interval is produced.
    distribution : str or None
        The distribution of the model.
    model : ALM or None
        The model that produced the predictions, as R's ``$model``.
    newdata : array-like or None
        The data the predictions were produced for (``None`` in-sample), as R's
        ``$newdata``.
    newdata_provided : bool
        Whether the predictions were produced for new data, as R's
        ``$newdataProvided``.
    """

    __slots__ = (
        "distribution",
        "interval",
        "level",
        "location",
        "lower",
        "mean",
        "model",
        "newdata",
        "newdata_provided",
        "occurrence",
        "scale",
        "side",
        "upper",
        "variances",
    )

    def __init__(
        self,
        mean=None,
        lower=None,
        upper=None,
        level=None,
        variances=None,
        side="both",
        interval="none",
        occurrence=None,
        location=None,
        scale=None,
        distribution=None,
        model=None,
        newdata=None,
        newdata_provided=False,
    ):
        self.mean = mean
        self.lower = lower
        self.upper = upper
        self.level = level
        self.variances = variances
        self.side = side
        self.interval = interval
        self.occurrence = occurrence
        self.location = location
        self.scale = scale
        self.distribution = distribution
        self.model = model
        self.newdata = newdata
        self.newdata_provided = newdata_provided

    def plot(self, y: ArrayLike | None = None, **kwargs):
        """Plot of the predictions, as R's ``plot.predict.greybox()``.

        Parameters
        ----------
        y : array-like, optional
            The actual values of the predicted observations, drawn after the
            actuals of the model.
        **kwargs
            Passed to :func:`greybox.graphmaker`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.model is None:
            raise ValueError("The predictions do not keep the model to plot.")
        from .plots import _plot_prediction

        return _plot_prediction(
            self, self.model, y=y, newdata_provided=self.newdata_provided, **kwargs
        )

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
        "LASSO", "RIDGE", "ROLE", or a function ``loss(actual, fitted, B, xreg)``
        returning the value to minimise (R's custom loss).
    occurrence : str or ALM, default="none"
        Occurrence model for zero-inflated data: "none", a fitted ALM with
        "plogis" / "pnorm", or "plogis" / "pnorm" to estimate one on the same
        regressors, as R. The fitted occurrence model is ``occurrence_``.
    scale_formula : str, array-like, ScaleModel or None, default=None
        The regressors of the scale model (R's ``scale``), estimated by likelihood
        on the residuals of the location model (see :func:`greybox.sm`): a
        one-sided formula (e.g. ``"~ x1 + x2"``), evaluated with
        :func:`greybox.formula` on ``X`` of ``fit()`` (a DataFrame, or its columns
        named after ``feature_names``), or the design matrix of the scale (with the
        column of ones for the intercept), or a scale model estimated by
        :func:`greybox.sm`. The scale model is ``scale_model``; it needs the
        likelihood. If None, the scale is constant. For new data, ``predict()``
        takes the regressors of the scale from ``X``: a DataFrame with the columns
        of both models, unless the scale model has the regressors of the location
        model or a formula of them.
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
    trim : float, optional
        Trim proportion for ROLE loss; 0.05 when not provided, as in R.
    lambda_l1 : float, optional
        L1 regularization parameter for LASSO.
    lambda_l2 : float, optional
        L2 regularization parameter for RIDGE.
    nlopt_kargs : dict, optional
        Dictionary of nlopt parameters. Options:

        - "algorithm": str, default="NLOPT_LN_NELDERMEAD"
        - "maxeval": int, default=40 per parameter (200 for LASSO / RIDGE and the
          ARI models)
        - "maxtime": float, default=-1 (no limit), in seconds
        - "xtol_rel": float, default=1e-6
        - "xtol_abs": float, default=1e-8
        - "ftol_rel": float, default=1e-4 (1e-8 for LASSO / RIDGE and the ARI
          models)
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
        trim=None,
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
        self.size_: float | None = None
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
        self._xlevels = None
        self._feature_names = None
        self._response_name = None
        self._df_residual = None
        self._B_opt_ = None
        self._columns_kept = None
        self._intercept_needed = True
        self._n_features_input = None
        self.occurrence_: ALM | None = None
        self.scale_model: ALM | None = None
        self._a_parameter_provided_ = False
        self._other_val_ = 1.0

    def _validate_params(self):
        """Validate input parameters."""
        if self.distribution not in self.DISTRIBUTIONS:
            raise ValueError(
                f"Invalid distribution: {self.distribution}. "
                f"Choose from: {self.DISTRIBUTIONS}"
            )

        if not callable(self.loss) and self.loss not in self.LOSS_FUNCTIONS:
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

        # The scale model is estimated by likelihood only, as R
        self.scale_model = None
        if self.scale_formula is not None and self.loss_ not in ("likelihood", "ROLE"):
            warnings.warn(
                "Scale model can only be estimated via likelihood, your loss won't "
                "work. Switching to likelihood.",
                stacklevel=2,
            )
            self.loss = "likelihood"

        X_is_dataframe = isinstance(X, pd.DataFrame)
        # The data of a formula of the scale: the regressors with their names
        scale_data = X if X_is_dataframe else None
        y_is_series = isinstance(y, pd.Series)

        if feature_names is None and X_is_dataframe:
            feature_names = list(X.columns)
            if "(Intercept)" in feature_names:
                feature_names.remove("(Intercept)")

        response_name = None
        if y_is_series and y.name is not None:
            response_name = y.name

        self._formula_ = formula
        # The levels of the factors of formula(), for the new data, as R's xlevels
        self._xlevels = X.attrs.get("_xlevels") if X_is_dataframe else None
        self._feature_names = feature_names
        self._response_name = response_name

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._check_response(y)
        # The boundary values of dbeta / dlogitnorm are moved inside, and the response
        # of plogis / pnorm is the occurrence, as R
        if self.distribution in ("dbeta", "dlogitnorm") and np.any((y == 0) | (y == 1)):
            warnings.warn(
                "The response variable contains boundary values (either zero or one). "
                f'distribution="{self.distribution}" is not estimable in this case. So '
                "we used a minor correction for it, in order to overcome this "
                "limitation.",
                stacklevel=2,
            )
            y = y * (1 - 2 * 1e-10)
        if self.distribution in ("plogis", "pnorm"):
            y = (y != 0) * 1.0

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if isinstance(self.scale_formula, str) and scale_data is None:
            if feature_names is None:
                raise ValueError(
                    "A formula of the scale needs the names of the regressors: give "
                    "X as a DataFrame, or feature_names."
                )
            scale_data = pd.DataFrame(
                X,
                columns=(["(Intercept)"] if np.all(X[:, 0] == 1) else [])
                + list(feature_names),
            )

        # The occurrence model: a fitted ALM, or one estimated here for "plogis" /
        # "pnorm" after the checks of the regressors, as R
        self.occurrence_ = self.occurrence if isinstance(self.occurrence, ALM) else None
        occurrence_type = (
            self.occurrence if self.occurrence in ("plogis", "pnorm") else None
        )

        # The checks of R's alm(): the regressors without variability, perfectly
        # correlated with another one, or in a perfect combination with the others are
        # dropped (the first of them), on the observed sizes of an occurrence model
        from .association import determination

        self._n_features_input = X.shape[1]
        occurrence_requested = self.occurrence_ is not None or occurrence_type
        quiet = occurrence_requested or self.distribution in ("plogis", "pnorm")
        intercept_needed = bool(np.all(X[:, 0] == 1))
        X_observed = X[y != 0] if occurrence_requested else X
        columns = np.arange(int(intercept_needed), X.shape[1])
        no_variability = np.all(
            X_observed[:, columns] == X_observed[0, columns], axis=0
        )
        if np.all(no_variability) and len(columns) > 0:
            warnings.warn(
                "None of exogenous variables has variability. Fitting the straight "
                "line.",
                stacklevel=2,
            )
        elif np.any(no_variability) and not quiet:
            warnings.warn(
                "Some exogenous variables did not have any variability. We dropped "
                "them out.",
                stacklevel=2,
            )
        columns = columns[~no_variability]
        if self.loss not in ("LASSO", "RIDGE"):
            if len(columns) > 1:
                correlated = np.triu(
                    np.abs(np.corrcoef(X_observed[:, columns], rowvar=False)) >= 0.999,
                    k=1,
                )
                if np.any(correlated):
                    columns = np.delete(columns, np.unique(np.nonzero(correlated)[0]))
                    if not quiet:
                        warnings.warn(
                            "Some exogenous variables were perfectly correlated. "
                            "We've dropped them out.",
                            stacklevel=2,
                        )
            if len(columns) > 1 and intercept_needed:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    combined = determination(X_observed[:, columns]) >= 0.999
                    if np.any(combined):
                        while np.any(combined):
                            columns = np.delete(columns, np.flatnonzero(combined)[0])
                            combined = determination(X[:, columns]) >= 0.999
                        if not occurrence_requested:
                            warnings.warn(
                                "Some combinations of exogenous variables were "
                                "perfectly correlated. We've dropped them out.",
                                stacklevel=2,
                            )
        columns = columns[np.var(X_observed[:, columns], axis=0) != 0]
        if intercept_needed and self.loss not in ("LASSO", "RIDGE") and len(columns):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                combined = np.nan_to_num(determination(X_observed[:, columns]))
            if np.any(combined == 1):
                columns = np.delete(columns, np.flatnonzero(combined == 1)[0])
        if intercept_needed:
            columns = np.concatenate([[0], columns])
        self._columns_kept = columns
        self._intercept_needed = intercept_needed
        X = X[:, columns]
        if feature_names is not None:
            # The names do not include the intercept
            names_all = (["(Intercept)"] if intercept_needed else []) + list(
                feature_names
            )
            feature_names = [
                names_all[i] for i in columns if names_all[i] != "(Intercept)"
            ]
            self._feature_names = feature_names

        if occurrence_type is not None:
            self.occurrence_ = ALM(
                distribution=occurrence_type, orders=tuple(self.orders[:2]) + (0,)
            ).fit(X, (y != 0) * 1.0)

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
            # The response in the first column, transformed with its lags as in R. The
            # lags of plogis / pnorm are the log-odds of the occurrence model of smooth
            # (MNN with the direct probability), as R
            ar_source = y
            if self.distribution in ("plogis", "pnorm"):
                smooth = importlib.import_module("smooth")
                ar_source = np.asarray(
                    smooth.OM(model="MNN", occurrence="direct").fit(y).fitted,
                    dtype=float,
                )
                ar_source[np.isnan(ar_source)] = np.nanmin(ar_source)
                ar_source[ar_source == 0] = 1e-10
                with np.errstate(divide="ignore"):
                    ar_source = np.log(ar_source / (1 - ar_source))
                finite = ar_source[np.isfinite(ar_source)]
                ar_source[np.isposinf(ar_source)] = np.max(finite)
                ar_source[np.isneginf(ar_source)] = np.min(finite)
            ar_terms = np.asarray(
                xreg_expander(ar_source, lags=ar_lags, gaps="auto"), dtype=float
            )
            observed = (
                y != 0
                if isinstance(self.occurrence_, ALM)
                else np.ones(n_samples, dtype=bool)
            )
            if self.distribution in _ARI_LOG_DISTRIBUTIONS:
                if np.any(y[observed] == 0):
                    # Box-Cox if there are zeroes
                    ar_terms = bc_transform(ar_terms, 0.01)
                else:
                    with np.errstate(divide="ignore"):
                        ar_terms = np.log(np.maximum(ar_terms, 0))
                    ar_terms[np.isinf(ar_terms)] = 0
            elif self.distribution == "dbcnorm":
                ar_terms = bc_transform(ar_terms, 0.1)
            elif self.distribution == "dchisq":
                ar_terms = np.sqrt(ar_terms)
            self._ari_response = ar_terms[:, 0]
            ar_terms = ar_terms[:, 1:]
            # The sizes at the zeros of an occurrence model and before the sample are
            # not observed: their lags start from the mean of the observed ones and are
            # replaced by the recursion of the model. Without the occurrence model, the
            # zeros are observed values
            if isinstance(self.occurrence_, ALM):
                for i in range(ari_order):
                    latent_rows = np.concatenate(
                        [np.ones(i + 1, dtype=bool), ~observed]
                    )[:n_samples]
                    ar_terms[latent_rows, i] = np.mean(self._ari_response[observed])

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
        # The losses other than the likelihood cannot estimate the extra parameter of
        # the distribution, which is then 0.5, as R
        if (
            not a_parameter_provided
            and self.distribution in distributions_with_extra_param
            and self.loss not in ("likelihood", "ROLE")
        ):
            warnings.warn(
                "The chosen loss function does not allow optimisation of additional "
                f'parameters for the distribution="{self.distribution}". Use '
                "likelihood instead. We will use 0.5.",
                stacklevel=2,
            )
            other_val = 0.5
            a_parameter_provided = True
        if not a_parameter_provided:
            other_val = 1.0

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
        # With I(d), the differences are taken on the observed sizes of an occurrence
        # model, as R
        y_init = y
        if i_order > 0:
            observed_init = (
                y != 0 if isinstance(self.occurrence_, ALM) else np.ones(len(y), bool)
            )
            y_init = y[observed_init]
            exog_init = X[observed_init][i_order:, :n_features_exog]
            if ar_order > 0:
                ar_lag_cols = X[observed_init][
                    :, n_features_exog : n_features_exog + ar_order
                ]
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
            # R's .lm.fit(), which stops on non-finite values: the default start then
            if not (np.all(np.isfinite(Xd)) and np.all(np.isfinite(yr))) or not len(yr):
                return None
            return _lm_fit(Xd, yr)[0]

        B_ols = None
        if self.distribution in ("plogis", "pnorm"):
            yr = bc_transform(y_init, 0.01)
            if i_order > 0:
                yr = np.diff(yr, n=i_order)
            B_ols = _ols_init(X_init, yr)
        elif self.distribution == "dbinom" and i_order == 0:
            # R: least squares on the series smoothed by supsmu (alm.R, dbinom)
            from .smoothers import supsmu

            mask = (
                y != 0 if isinstance(self.occurrence_, ALM) else np.ones(len(y), bool)
            )
            smoothed = supsmu(np.flatnonzero(mask) + 1.0, y[mask])["y"]
            B_ols = _ols_init(X_init[mask], smoothed)
        elif self.distribution == "dpois":
            if i_order > 0:
                # ARI: log-link OLS (matches R's I(d) branch)
                yr = np.log(np.maximum(y_init, 1e-10))
                yr = np.diff(yr, n=i_order)
                B_ols = _ols_init(X_init, yr)
            else:
                # I(0): quasi-Newton step
                mu_insample = np.mean(y)
                try:
                    XtX_mu = X_init.T @ X_init * mu_insample
                    B_ols = np.linalg.solve(XtX_mu, X_init.T @ (y - mu_insample))
                    # The Newton step goes from the log of the mean
                    if intercept_needed:
                        B_ols[0] += np.log(mu_insample)
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
                yr = np.diff(np.log(y_init), n=i_order)
                B_ols = _ols_init(X_init, yr)
            else:
                # R: the log of the observed sizes, shifted by one if there are zeros
                # (any zero of the data for dgeom)
                mask = (
                    y != 0
                    if isinstance(self.occurrence_, ALM)
                    else np.ones(len(y), bool)
                )
                zeros = y if self.distribution == "dgeom" else y[mask]
                shift = 1.0 if np.any(zeros == 0) else 0.0
                B_ols = _ols_init(X_init[mask], np.log(y[mask] + shift))
        elif self.distribution in ("dlogitnorm", "dbeta"):
            y_clip = np.clip(y_init, 1e-10, 1 - 1e-10)
            yr = np.log(y_clip / (1 - y_clip))
            if i_order > 0:
                yr = np.diff(yr, n=i_order)
            B_ols = _ols_init(X_init, yr)
        elif self.distribution == "dchisq":
            yr = np.sqrt(y_init)
            if i_order > 0:
                yr = np.diff(yr, n=i_order)
            B_ols = _ols_init(X_init, yr)
        elif self.distribution == "dbcnorm":
            lam = self.lambda_bc if a_parameter_provided else 0.1
            if i_order > 0:
                B_ols = _ols_init(X_init, np.diff(bc_transform(y_init, lam), n=i_order))
            else:
                # R: on the observed sizes of an occurrence model
                mask = (
                    y != 0
                    if isinstance(self.occurrence_, ALM)
                    else np.ones(len(y), bool)
                )
                B_ols = _ols_init(X_init[mask], bc_transform(y[mask], lam))
        else:
            # default: dnorm, dt, dlaplace, dalaplace, ds, dfnorm, drectnorm,
            # dgnorm, dgeom, ...
            yr = y_init
            if i_order > 0:
                yr = np.diff(yr, n=i_order)
            # If occurrence is a fitted ALM (hurdle model), the size model
            # should fit non-zero observations only; initialise from that
            # subset so the optimiser doesn't get pulled towards mean(y)
            # which includes the zeros.
            if isinstance(self.occurrence_, ALM) and i_order == 0:
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
                "dnbinom": np.var(
                    y[y != 0] if isinstance(self.occurrence_, ALM) else y, ddof=1
                ),
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
        occurrence_is_model = isinstance(self.occurrence_, ALM)
        # R: the size of dbinom is the number of distinct values less one, and one
        # fewer with an occurrence model, whose zeroes are removed
        self.size_ = (
            len(np.unique(y)) - 1 - int(occurrence_is_model)
            if self.distribution == "dbinom"
            else self.size
        )
        if occurrence_is_model:
            occ_fitted = np.asarray(self.occurrence_.fitted_values_, dtype=float)
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

        # The sizes at the zeros of a dynamic occurrence model are latent: they go on
        # with the recursion of the model (the Kalman filter for the distributions
        # additive in the space of the lags), as R's fitterRecursive
        latent = None
        latent_variance = None
        if occurrence_is_model and ari_order > 0:
            latent = {
                "response": self._ari_response,
                "observed": otU_full,
                "n_exo": n_features_exog,
                "ari_order": ari_order,
                "filter": self.distribution in ARI_FILTER_DISTRIBUTIONS,
                "scale_power": 4 if self.distribution in ("ds", "dls") else 2,
            }
        # The lags of plogis / pnorm are refitted by the model, as R
        if self.distribution in ("plogis", "pnorm") and ari_order > 0:
            latent = {
                "cdf": True,
                "n_exo": n_features_exog,
                "ari_order": ari_order,
                "filter": False,
            }
        self._latent = latent

        # LASSO / RIDGE, as R: the response in the space of the model, the regressors
        # centred and scaled by their standard deviations for LASSO, which starts from
        # the RIDGE solution, and RIDGE in its closed form
        y_cf, X_cf, denominator = y, X, 1.0
        lambda_value = 0.0
        lower_bounds = upper_bounds = None
        penalised = self.loss in ("LASSO", "RIDGE")
        if penalised:
            lambda_value = self.lambda_l1 if self.loss == "LASSO" else self.lambda_l2
            if lambda_value is None:
                warnings.warn(
                    "You have not provided lambda parameter. We will set it to zero.",
                    stacklevel=2,
                )
                lambda_value = 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                denominator = np.std(X, axis=0, ddof=1)
                if self.distribution in ("dlnorm", "dllaplace", "dls", "dlgnorm"):
                    y_cf = np.log(y)
                elif self.distribution == "dlogitnorm":
                    y_cf = np.log(y / (1 - y))
                elif self.distribution == "dbcnorm":
                    y_cf = bc_transform(y, other_val)
            denominator[np.isinf(denominator)] = 1
            intercept_needed = bool(np.all(X[:, 0] == 1))
            slopes = slice(1, None) if intercept_needed else slice(None)
            x_means = np.mean(X[otU_full, slopes], axis=0) if intercept_needed else 0
            y_mean = np.mean(y_cf[otU_full]) if intercept_needed else 0.0
            X_scaled = (X[:, slopes] - x_means) / denominator[slopes]
            ridge = lambda_value / (1 - lambda_value) if lambda_value != 1 else 0.0
            if lambda_value != 1:
                B_slopes = np.linalg.solve(
                    X_scaled[otU_full].T @ X_scaled[otU_full]
                    + ridge * np.eye(X_scaled.shape[1]),
                    X_scaled[otU_full].T @ (y_cf[otU_full] - y_mean),
                )
            else:
                B_slopes = np.zeros(X_scaled.shape[1])
            if self.loss == "LASSO":
                y_cf = y_cf - y_mean
                X_cf = X_scaled
                B_init = B_slopes
                upper_bounds = list(np.abs(B_init) * 1000 + 1e-5)
                lower_bounds = [-bound - 1e-5 for bound in upper_bounds]
            else:
                B_slopes = B_slopes / denominator[slopes]
                B_init = (
                    np.concatenate([[y_mean - x_means @ B_slopes], B_slopes])
                    if intercept_needed
                    else B_slopes
                )
            n_params = len(B_init)

        print_level = self.nlopt_kargs.get("print_level", 0)
        iteration_count = [0]
        last_cf_value = [None]

        def objective_func(B, grad, loss=self.loss_):
            if grad.size > 0:
                grad[:] = 0

            cf_value = cf(
                B,
                self.distribution,
                loss,
                y_cf,
                X_cf,
                loss_function=self.loss if callable(self.loss) else None,
                ar_order=ar_order,
                i_order=i_order,
                lambda_val=lambda_value,
                denominator=denominator,
                other=other_val,
                a_parameter_provided=a_parameter_provided,
                trim=self._trim_value(),
                lambda_bc=(
                    self.lambda_bc
                    if self.distribution == "dbcnorm" and a_parameter_provided
                    else 0.0
                ),
                size=self.size_ if self.distribution == "dbinom" else 1.0,
                otU=otU_full if occurrence_is_model else None,
                obs_zero=obs_zero,
                obs_nonzero=obs_nonzero,
                occurrence_model=occurrence_is_model,
                latent=latent,
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

        if lower_bounds is not None:
            pass
        elif (
            self.distribution in distributions_with_extra_param
            and not a_parameter_provided
        ):
            # R resets the bounds of dt and dchisq to infinite ones
            lower_bounds = [
                -np.inf if self.distribution in ("dt", "dchisq") else 0.0
            ] + [-np.inf] * n_params_base
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
        # LASSO / RIDGE and the ARI models, whose likelihood is flat in the AR
        # parameters, need the accurate estimation, as in R
        accurate = ari_order > 0 or self.loss in ("LASSO", "RIDGE")
        opt.set_ftol_rel(self.nlopt_kargs.get("ftol_rel", 1e-8 if accurate else 1e-4))
        opt.set_ftol_abs(self.nlopt_kargs.get("ftol_abs", 0))

        maxeval = self.nlopt_kargs.get("maxeval")

        if maxeval is None:
            if (
                latent is None
                and self.scale_formula is None
                and self.distribution in ("dnorm", "dlnorm", "dlogitnorm")
                and self.loss in ("likelihood", "MSE")
            ):
                maxeval = 1
            else:
                maxeval = n_params * (200 if ari_order > 0 else 40)
            if penalised:
                maxeval = n_params * 200
                if lambda_value == 1 or self.loss == "RIDGE":
                    maxeval = 1

        opt.set_maxeval(maxeval)
        opt.set_maxtime(self.nlopt_kargs.get("maxtime", -1))

        try:
            B_opt = opt.optimize(B_init)
            self.n_iter_ = opt.get_numevals()
            self._result = opt.last_optimum_value()
            self.nlopt_result_ = opt.last_optimize_result()
            # The recursive models restart from the solution, keeping the better one (R)
            if latent is not None:
                result_first = self._result
                B_second = opt.optimize(B_opt)
                if opt.last_optimum_value() < result_first:
                    B_opt = B_second
                    self._result = opt.last_optimum_value()
                    self.nlopt_result_ = opt.last_optimize_result()
                self.n_iter_ += opt.get_numevals()
        except Exception as e:  # noqa: BLE001 - report any optimiser failure, keep init
            if self.verbose > 0:
                print(f"Optimization failed: {e}")
            B_opt = B_init
            self.n_iter_ = 0
            self._result = None
            self.nlopt_result_ = None

        # LASSO goes back to the original regressors, as R
        if penalised:
            loss_penalised = objective_func(B_opt, np.zeros(n_params))
            if self.loss == "LASSO":
                B_opt = (
                    np.zeros(len(B_opt))
                    if lambda_value == 1
                    else B_opt / denominator[slopes]
                )
                if intercept_needed:
                    B_opt = np.concatenate([[y_mean - x_means @ B_opt], B_opt])
            n_params = len(B_opt)

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

        if latent is not None:
            # The latent sizes at the zeros, with the lags of the recursion of the model
            recursive_return = fitter_recursive(
                B_opt,
                self.distribution,
                y,
                X,
                latent,
                other=other_val,
                a_parameter_provided=a_parameter_provided,
                ar_order=ar_order,
                i_order=i_order,
                otU=otU_full,
                trim=self._trim_value(),
            )
            X = recursive_return["matrix_xreg"]
            fitter_return["mu"] = recursive_return["mu"]
            scale = recursive_return["scale"]
            if latent["filter"]:
                latent_variance = recursive_return["variance"]
        else:
            scale = scaler_internal(
                B_opt,
                self.distribution,
                y,
                X,
                fitter_return["mu"],
                other_val,
                otU_full,
                obs_nonzero,
                self._trim_value(),
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

        # R complains about the dispersion of dexp, and puts the fitted values of the
        # negative locations of dbcnorm at zero
        if self.distribution == "dexp":
            dispersion = np.std(self.residuals_, ddof=1)
            if dispersion > 1.1 or dispersion < 0.9:
                warnings.warn(
                    "The distribution is "
                    f"{'over' if dispersion > 1.1 else 'under'}dispersed. Consider "
                    "using another one.",
                    stacklevel=2,
                )
        if self.distribution == "dbcnorm":
            self.fitted_values_ = np.where(self.mu_ < 0, 0.0, self.fitted_values_)
        # The fitted values of dbinom take its size, and the errors of dbinom and dbeta
        # are in the space of the data, as R
        if self.distribution == "dbinom":
            self.fitted_values_ = self.fitted_values_ * self.size_
        if self.distribution in ("dbinom", "dbeta"):
            self.residuals_ = y - self.fitted_values_
        # R's mu and scale of dbeta are the mean and the variance, from its shapes
        self._beta_shapes = None
        if self.distribution == "dbeta":
            shape1, shape2 = self.mu_, np.asarray(scale, dtype=float)
            self._beta_shapes = (shape1, shape2)
            self.mu_ = self.fitted_values_.copy()

        if occurrence_is_model:
            self.fitted_values_ = self.fitted_values_ * occ_fitted
            # The sizes of the zeros are not observed: their errors are zero, as in R
            self.residuals_ = np.where(otU_full, self.residuals_, 0.0)

        self._loss_value = (
            loss_penalised if penalised else objective_func(B_opt, np.zeros(n_params))
        )

        # The number of parameters, as R's nParam: the scale of the likelihood, the
        # extra parameter of the distribution, both parts of dbeta, and the non-zero
        # parameters of LASSO / RIDGE
        if self.loss in ("LASSO", "RIDGE"):
            n_params_model = int(np.sum(B_opt != 0))
        elif self.distribution == "dbeta":
            n_params_model = 2 * n_params_base
        else:
            n_params_model = (
                n_params_base
                + int(
                    self.loss in ("likelihood", "ROLE")
                    and self.distribution
                    not in ("dexp", "dpois", "dgeom", "dbinom", "plogis", "pnorm")
                )
                + int(
                    not a_parameter_provided
                    and self.distribution
                    in ("dchisq", "dt", "dbcnorm", "dgnorm", "dlgnorm", "dalaplace")
                )
            )
        self._n_params_ = n_params_model
        self._df_residual = n_samples - n_params_model

        # The likelihood, also at the estimates of the losses that correspond to one,
        # and multiplied by the smallest root of a non-stationary AR polynomial, which
        # divides the cost, as R
        self._log_lik = None
        if self.loss in ("likelihood", "ROLE"):
            self._log_lik = -self._loss_value
        elif (
            (
                self.loss == "MSE"
                and self.distribution in ("dnorm", "dlnorm", "dbcnorm", "dlogitnorm")
            )
            or (self.loss == "MAE" and self.distribution in ("dlaplace", "dllaplace"))
            or (self.loss == "HAM" and self.distribution in ("ds", "dls"))
        ):
            self._log_lik = -objective_func(
                B_opt, np.zeros(n_params), loss="likelihood"
            )
        if self._log_lik is not None and ar_order > 0:
            ar_phi = B_opt[-ar_order:]
            roots = np.abs(
                np.polynomial.polynomial.polyroots(np.concatenate([[1.0], -ar_phi]))
            )
            if np.any(roots < 1):
                self._log_lik = self._log_lik * np.min(roots)

        # The occurrence model adds its likelihood to the loss and to the likelihood
        if occurrence_is_model:
            occurrence_log_lik = self.occurrence_.log_lik or 0.0
            self._loss_value = self._loss_value - occurrence_log_lik
            if self._log_lik is not None:
                self._log_lik = self._log_lik + occurrence_log_lik

        # The information criteria of R's logLik(): the parameters of both models, and
        # the scale of the losses that do not estimate it
        if self._log_lik is not None:
            n_params_ic = n_params_model + (
                self.occurrence_.nparam
                if occurrence_is_model
                else int(
                    (self.loss, self.distribution)
                    in (("MSE", "dnorm"), ("MAE", "dlaplace"), ("HAM", "ds"))
                )
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                self._aic = 2 * n_params_ic - 2 * self._log_lik
                self._bic = n_params_ic * np.log(n_samples) - 2 * self._log_lik
                self._aicc = self._aic + np.divide(
                    2 * n_params_ic * (n_params_ic + 1), n_samples - n_params_ic - 1
                )
                self._bicc = -2 * self._log_lik + n_params_ic * np.log(
                    n_samples
                ) * n_samples / (n_samples - n_params_ic - 1)
        else:
            self._aic = self._bic = self._aicc = self._bicc = None
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
            self._trim_value(),
            variance=latent_variance,
        )
        # The scale of the Normal-based distributions is the variance (ADAM monograph)
        if self.distribution in VARIANCE_SCALE_DISTRIBUTIONS:
            scale = scale**2
        # R stores the size as the scale of dbinom, and the variance as the one of dbeta
        elif self.distribution == "dbinom":
            scale = float(self.size_)
        elif self.distribution == "dbeta":
            shape1, shape2 = self._beta_shapes
            scale = shape1 * shape2 / ((shape1 + shape2) ** 2 * (shape1 + shape2 + 1))
        self._scale = scale

        XtX = X.T @ X
        XtX += np.eye(XtX.shape[0]) * 1e-10
        self._XtX_inv_ = np.linalg.inv(XtX)

        # Store optimization state for Hessian-based vcov
        self._B_opt_ = B_opt.copy()
        self._a_parameter_provided_ = a_parameter_provided
        self._other_val_ = other_val

        # The scale model, estimated on the residuals of the location model, as R
        if self.scale_formula is not None:
            from .sm import ScaleModel, implant, sm

            scale_model = (
                self.scale_formula
                if isinstance(self.scale_formula, ScaleModel)
                else sm(self, formula=self.scale_formula, data=scale_data)
            )
            self.__dict__.update(implant(self, scale_model).__dict__)

        self._time_elapsed = time_module.time() - start_time

        return self

    def _check_response(self, y: np.ndarray) -> None:
        """Reject the responses the distribution cannot take, as R's alm()."""
        distribution = self.distribution
        if distribution in _NON_NEGATIVE_DISTRIBUTIONS and np.any(y < 0):
            raise ValueError(
                "Negative values are not allowed in the response variable for the "
                f"distribution '{distribution}'"
            )
        # The density of these is zero at zero (the log ones take log(y)), so the zeros
        # need an occurrence model
        if (
            distribution in _POSITIVE_DISTRIBUTIONS
            and np.any(y == 0)
            and not isinstance(self.occurrence, ALM)
            and self.occurrence not in ("plogis", "pnorm")
        ):
            raise ValueError(
                "Zero values are not allowed in the response variable for the "
                f"distribution '{distribution}'"
            )
        if distribution in ("dpois", "dnbinom", "dbinom", "dgeom") and np.any(
            y != np.trunc(y)
        ):
            raise ValueError(
                f"Count data is needed for the distribution '{distribution}', but you "
                "have fractional numbers. Maybe you should try some other distribution?"
            )
        if distribution in ("dbeta", "dlogitnorm") and np.any((y > 1) | (y < 0)):
            raise ValueError(
                "The response variable should lie between 0 and 1 in "
                f'distribution="{distribution}"'
            )

    def vcov(self, bootstrap: bool = False, **kwargs) -> np.ndarray:
        """Calculate variance-covariance matrix of parameter estimates.

        Uses distribution-specific methods matching R's vcov.alm():

        - Normal-like + likelihood/MSE: sigma^2 * (X'X)^-1
        - Poisson + likelihood: inverse Fisher information
        - Everything else: inverse numerical Hessian of cost function

        Parameters
        ----------
        bootstrap : bool, default=False
            Whether to take the covariance matrix of the bootstrapped parameters,
            from :func:`greybox.coefbootstrap`.
        **kwargs
            Passed to :func:`greybox.coefbootstrap` (``nsim``, ``size``, ...).

        Returns
        -------
        vcov_matrix : np.ndarray
            Covariance matrix of shape (n_params, n_params)
        """
        if self._X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if bootstrap:
            from .bootstrap import coefbootstrap

            return coefbootstrap(self, **kwargs).vcov

        X = self._X_train_
        n_vars = X.shape[1]
        distribution = self.distribution
        loss = self.loss

        # Path 1: Analytical for normal-like distributions with likelihood/MSE, without
        # the I part and the latent lags of an occurrence model, which depend on the
        # parameters: R takes those through the Hessian
        normal_like = ("dnorm", "dlnorm", "dbcnorm", "dlogitnorm")
        analytical = self._i_order == 0 and getattr(self, "_latent", None) is None
        if analytical and (
            (distribution in normal_like and loss == "likelihood") or loss == "MSE"
        ):
            XtX = X.T @ X
            try:
                L = np.linalg.cholesky(XtX)
                XtX_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n_vars)))
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv(XtX)
            return self.sigma**2 * XtX_inv

        # Path 2: Analytical for Poisson with likelihood
        if analytical and distribution == "dpois" and loss == "likelihood":
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
        else:
            # dbeta takes the parameters of both of its shapes, as R
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
        # The likelihood of the sizes of an occurrence model, with the latent lags of
        # the dynamic models recomputed for each parameter, as R's vcov() refits it
        occurrence_model = isinstance(self.occurrence_, ALM)
        otU = np.asarray(y) != 0 if occurrence_model else None
        latent = getattr(self, "_latent", None)

        def full_B(B):
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
                return np.concatenate([[other_val], B])
            return B

        def cf_wrapper(B):
            return cf(
                full_B(B),
                self.distribution,
                self.loss_,
                y,
                X,
                loss_function=self.loss if callable(self.loss) else None,
                ar_order=self._ar_order,
                i_order=self._i_order,
                lambda_val=lambda_val,
                other=other_val,
                a_parameter_provided=a_parameter_provided,
                trim=self._trim_value(),
                lambda_bc=lambda_bc_val,
                size=self.size_ if self.distribution == "dbinom" else 1.0,
                otU=otU,
                occurrence_model=occurrence_model,
                latent=latent,
            )

        if self.loss == "likelihood" and self.distribution in (
            "dlaplace",
            "dllaplace",
            "dalaplace",
        ):
            # The likelihood of the Laplace family is not smooth in the location, so its
            # Hessian is unreliable: the information is the expected one, from the
            # derivatives of the location (through the recursion of the latent sizes),
            # by pracma's central differences, as R
            from .fitters import fitter, fitter_recursive

            fitter_kwargs = {
                "other": other_val,
                "a_parameter_provided": a_parameter_provided,
                "ar_order": self._ar_order,
                "i_order": self._i_order,
                "loss": self.loss,
                "otU": otU,
                "trim": self._trim_value(),
            }

            def location(B):
                if latent is not None:
                    return fitter_recursive(
                        full_B(B), self.distribution, y, X, latent, **fitter_kwargs
                    )
                return fitter(full_B(B), self.distribution, y, X, **fitter_kwargs)

            observed = otU if otU is not None else np.ones(len(y), dtype=bool)
            step = np.finfo(float).eps ** (1 / 3)
            jacobian = np.column_stack(
                [
                    (
                        location(B_reg + step * e)["mu"]
                        - location(B_reg - step * e)["mu"]
                    )[observed]
                    / (2 * step)
                    for e in np.eye(len(B_reg))
                ]
            )
            # The variances of the scores: the scale with the filter after the latent
            # sizes
            variance = (
                location(B_reg)["variance"][observed] if latent is not None else 1.0
            )
            weights = np.ones(int(np.sum(observed))) / (self.scale**2 * variance)
            if self.distribution == "dalaplace":
                weights = weights * other_val * (1 - other_val)
            FI = jacobian.T @ (jacobian * weights[:, None])
        else:
            FI = _numerical_hessian(cf_wrapper, B_reg)

            # Check for broken variables (all-zero or NaN rows)
            broken = np.all(FI == 0, axis=1) | np.any(np.isnan(FI), axis=1)
            if np.any(broken):
                FI = _numerical_hessian(
                    cf_wrapper, B_reg, h=np.finfo(float).eps ** (1 / 6)
                )

        try:
            L = np.linalg.cholesky(FI)
            vcov = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(B_reg))))
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

    def _calculate_quantiles(
        self,
        linear_pred,
        variances,
        interval,
        level,
        side,
        normal_dnorm=False,
        scale=None,
    ):
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
        normal_dnorm : bool
            dnorm takes the Normal quantiles instead of Student's, as R with an
            occurrence model.
        scale : float or np.ndarray, optional
            The scale of dinvgauss and dgamma (those of the scale model for the new
            data); :attr:`scale` by default.

        Returns
        -------
        lower : np.ndarray or None
            Lower bounds (in response scale).
        upper : np.ndarray or None
            Upper bounds (in response scale).
        """
        if interval == "none":
            return None, None
        if scale is None:
            scale = self.scale

        if not isinstance(level, list):
            level = [level]

        n_levels = len(level)
        # dbeta takes its two shapes in place of the location
        beta_shapes = None
        if self.distribution == "dbeta":
            beta_shapes = linear_pred
            linear_pred = beta_shapes[0]
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
            if d == "dnorm" and normal_dnorm:
                lower[:, i] = stats.norm.ppf(ll, loc=linear_pred, scale=se)
                upper[:, i] = stats.norm.ppf(lu, loc=linear_pred, scale=se)
            elif d in ("dnorm", "dt"):
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
                    lower[:, i] = dist.qlaplace(ll, mu=linear_pred, scale=b)
                    upper[:, i] = dist.qlaplace(lu, mu=linear_pred, scale=b)

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
                        ll, mu=linear_pred, scale=scale_al, alpha=alpha
                    )
                    upper[:, i] = dist.qalaplace(
                        lu, mu=linear_pred, scale=scale_al, alpha=alpha
                    )

            elif d == "ds":
                if interval == "confidence":
                    lower[:, i] = lp_l
                    upper[:, i] = lp_u
                else:
                    # S-distribution variance = 120*scale⁴  →  scale=(var/120)^0.25
                    s_scale = (variances / 120) ** 0.25
                    lower[:, i] = dist.qs(ll, mu=linear_pred, scale=s_scale)
                    upper[:, i] = dist.qs(lu, mu=linear_pred, scale=s_scale)

            elif d == "dlogis":
                if interval == "confidence":
                    lower[:, i] = lp_l
                    upper[:, i] = lp_u
                else:
                    # Logistic variance = π²*scale²/3  →  scale=sqrt(var*3/π²)
                    logis_scale = np.sqrt(variances * 3 / np.pi**2)
                    lower[:, i] = dist.qlogis(
                        ll, location=linear_pred, scale=logis_scale
                    )
                    upper[:, i] = dist.qlogis(
                        lu, location=linear_pred, scale=logis_scale
                    )

            elif d == "dgnorm":
                if interval == "confidence":
                    lower[:, i] = lp_l
                    upper[:, i] = lp_u
                else:
                    # GNorm variance = Γ(3/s)/Γ(1/s)*scale²
                    s = self.other_
                    gn_scale = np.sqrt(variances * _sp_gamma(1 / s) / _sp_gamma(3 / s))
                    lower[:, i] = dist.qgnorm(
                        ll, mu=linear_pred, scale=gn_scale, shape=s
                    )
                    upper[:, i] = dist.qgnorm(
                        lu, mu=linear_pred, scale=gn_scale, shape=s
                    )

            elif d == "dfnorm":
                if interval == "confidence":
                    lower[:, i] = lp_l
                    upper[:, i] = lp_u
                else:
                    # qfnorm requires scalar mu/sigma; compute element-wise
                    lower[:, i] = np.array(
                        [
                            float(dist.qfnorm(ll, mu=float(mj), sigma=float(sj)))
                            for mj, sj in zip(linear_pred, se)
                        ]
                    )
                    upper[:, i] = np.array(
                        [
                            float(dist.qfnorm(lu, mu=float(mj), sigma=float(sj)))
                            for mj, sj in zip(linear_pred, se)
                        ]
                    )

            elif d == "drectnorm":
                if interval == "confidence":
                    lower[:, i] = lp_l
                    upper[:, i] = lp_u
                else:
                    lower[:, i] = dist.qrectnorm(ll, mu=linear_pred, sigma=se)
                    upper[:, i] = dist.qrectnorm(lu, mu=linear_pred, sigma=se)

            # ── Group C: log-link; CI=exp(t-bounds); PI=dist quantile ───────
            elif d == "dinvgauss":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    lower[:, i] = np.exp(linear_pred) * dist.qinvgauss(
                        ll, mean=1, dispersion=scale
                    )
                    upper[:, i] = np.exp(linear_pred) * dist.qinvgauss(
                        lu, mean=1, dispersion=scale
                    )

            elif d == "dgamma":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    lower[:, i] = np.exp(linear_pred) * dist.qgamma(
                        ll, shape=1 / scale, scale=scale
                    )
                    upper[:, i] = np.exp(linear_pred) * dist.qgamma(
                        lu, shape=1 / scale, scale=scale
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
                    lower[:, i] = dist.qpois(ll, lambda_=np.exp(linear_pred))
                    upper[:, i] = dist.qpois(lu, lambda_=np.exp(linear_pred))

            elif d == "dnbinom":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    # self.scale holds the size parameter (abs(other))
                    lower[:, i] = dist.qnbinom(
                        ll, mu=np.exp(linear_pred), size=self.scale
                    )
                    upper[:, i] = dist.qnbinom(
                        lu, mu=np.exp(linear_pred), size=self.scale
                    )

            elif d == "dgeom":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    prob = 1 / (1 + np.exp(linear_pred))
                    lower[:, i] = stats.geom.ppf(ll, prob) - 1
                    upper[:, i] = stats.geom.ppf(lu, prob) - 1

            elif d == "dbeta" and beta_shapes is not None:
                shape1, shape2 = beta_shapes
                if interval == "prediction":
                    lower[:, i] = stats.beta.ppf(ll, shape1, shape2)
                    upper[:, i] = stats.beta.ppf(lu, shape1, shape2)
                else:
                    mean_beta = shape1 / (shape1 + shape2)
                    se_beta = np.sqrt(variances / self.nobs)
                    lower[:, i] = mean_beta + tl * se_beta
                    upper[:, i] = mean_beta + tu * se_beta

            elif d == "dbinom":
                binom_size = self.size_ if self.size_ is not None else 1
                # The probability decreases in the location: the bounds swap
                lower[:, i] = binom_size / (1 + np.exp(lp_u))
                upper[:, i] = binom_size / (1 + np.exp(lp_l))
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
                    lower[:, i] = dist.qllaplace(ll, mu=linear_pred, scale=b)
                    upper[:, i] = dist.qllaplace(lu, mu=linear_pred, scale=b)

            elif d == "dls":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    s_scale = (variances / 120) ** 0.25
                    lower[:, i] = dist.qls(ll, mu=linear_pred, scale=s_scale)
                    upper[:, i] = dist.qls(lu, mu=linear_pred, scale=s_scale)

            elif d == "dlgnorm":
                lower[:, i] = np.exp(lp_l)
                upper[:, i] = np.exp(lp_u)
                if interval == "prediction":
                    beta = self.other_
                    lgn_scale = np.sqrt(
                        variances * _sp_gamma(1 / beta) / _sp_gamma(3 / beta)
                    )
                    lower[:, i] = dist.qlgnorm(
                        ll, mu=linear_pred, scale=lgn_scale, shape=beta
                    )
                    upper[:, i] = dist.qlgnorm(
                        lu, mu=linear_pred, scale=lgn_scale, shape=beta
                    )

            # ── Group E: log-normal family; use dist quantile for CI and PI ──
            elif d == "dlnorm":
                # qlnorm(p, meanlog=lp, sdlog=se) where se encodes the right
                # uncertainty: for CI se=sqrt(var_pred), for PI se=sqrt(var_pred+σ²)
                lower[:, i] = dist.qlnorm(ll, meanlog=linear_pred, sdlog=se)
                upper[:, i] = dist.qlnorm(lu, meanlog=linear_pred, sdlog=se)

            elif d == "dlogitnorm" and interval == "confidence":
                lower[:, i] = 1 / (1 + np.exp(-lp_l))
                upper[:, i] = 1 / (1 + np.exp(-lp_u))
            elif d == "dlogitnorm":
                lower[:, i] = dist.qlogitnorm(ll, mu=linear_pred, sigma=se)
                upper[:, i] = dist.qlogitnorm(lu, mu=linear_pred, sigma=se)

            # ── Group F: special transformations ────────────────────────────
            elif d == "dbcnorm":
                if interval == "confidence":
                    # Invert BC transform of t-bounds in LP space
                    lower[:, i] = _bc_transform_inv(lp_l, lambda_bc_eff)
                    upper[:, i] = _bc_transform_inv(lp_u, lambda_bc_eff)
                else:
                    lower[:, i] = dist.qbcnorm(
                        ll, mu=linear_pred, sigma=se, lambda_bc=lambda_bc_eff
                    )
                    upper[:, i] = dist.qbcnorm(
                        lu, mu=linear_pred, sigma=se, lambda_bc=lambda_bc_eff
                    )

            elif d == "dchisq":
                # CI: square the t-bounds; PI: non-central chi-squared
                lower[:, i] = lp_l**2
                upper[:, i] = lp_u**2
                if interval == "prediction":
                    lower[:, i] = dist.qchisq(ll, df=self.other_, ncp=linear_pred**2)
                    upper[:, i] = dist.qchisq(lu, df=self.other_, ncp=linear_pred**2)

            # ── Group G: link-function distributions (plogis/pnorm) ─────────
            elif d == "plogis":
                # Apply plogis to normal quantile of LP
                lower[:, i] = dist.plogis(
                    dist.qnorm(ll, mean=linear_pred, sd=se),
                    location=0.0,
                    scale=1.0,
                )
                upper[:, i] = dist.plogis(
                    dist.qnorm(lu, mean=linear_pred, sd=se),
                    location=0.0,
                    scale=1.0,
                )

            elif d == "pnorm":
                lower[:, i] = dist.pnorm(
                    dist.qnorm(ll, mean=linear_pred, sd=se), mean=0.0, sd=1.0
                )
                upper[:, i] = dist.pnorm(
                    dist.qnorm(lu, mean=linear_pred, sd=se), mean=0.0, sd=1.0
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
        X: np.ndarray | None = None,
        interval: Literal["none", "confidence", "prediction"] = "none",
        level: float | list[float] = 0.95,
        side: Literal["both", "upper", "lower"] = "both",
        occurrence: ArrayLike | None = None,
    ) -> PredictionResult:
        """Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Design matrix, with the same features as the training data. Without it,
            the in-sample values, as R's ``newdata=NULL``.
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
        occurrence : array-like, optional
            The future probabilities of occurrence (or its values), which scale the
            mean and the bounds of the confidence interval, as R's
            ``predict.alm(occurrence=)``; the prediction interval stays the one of the
            sizes, as in smooth's ``ADAM``. By default, a fitted occurrence model is
            predicted on ``X``: its probabilities scale the mean and the bounds of the
            confidence interval, and move the levels of the prediction interval to
            those of the sizes.

        Returns
        -------
        PredictionResult
            Object with the following attributes:

            - mean : np.ndarray - Predicted values (point forecasts)
            - lower : np.ndarray or None - Lower prediction bounds
            - upper : np.ndarray or None - Upper prediction bounds
            - occurrence : PredictionResult or None - The predictions of the
              occurrence model
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

        # Without X, the in-sample values, as R's newdata=NULL
        in_sample = X is None
        newdata = X
        if in_sample:
            X = self._X_train_[:, : self._n_features_exog]
        # The regressors of the scale model come from the same data: a DataFrame with
        # the columns of both models, or the columns of the location model
        scale_data = X
        if self.scale_model is not None:
            n_exog = self._n_features_exog
            names = (["(Intercept)"] if self._intercept_needed else []) + list(
                self._feature_names or [f"x{i}" for i in range(1, n_exog + 1)]
            )[: n_exog - int(self._intercept_needed)]
            if isinstance(X, pd.DataFrame) and set(names) <= set(X.columns):
                X = X[names]
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # The regressors that the checks of the fit kept
        if not in_sample and X.shape[1] == self._n_features_input:
            X = X[:, self._columns_kept]
        # The scales of the new data, from the scale model
        scale_new = None
        if self.scale_model is not None:
            if not isinstance(scale_data, pd.DataFrame):
                scale_data = pd.DataFrame(X[:, : self._n_features_exog], columns=names)
            scale_new = np.asarray(
                self.scale_model.predict(None if in_sample else scale_data).mean
            )

        # The fitted occurrence model gives the probabilities, unless they are provided
        occurrence_predict = None
        probability = None
        if occurrence is None and isinstance(self.occurrence_, ALM):
            occurrence_predict = self.occurrence_.predict(
                None if in_sample else X, interval=interval, level=level, side=side
            )
            probability = np.ravel(np.asarray(occurrence_predict.mean, dtype=float))

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

        ari_variance = 1.0
        if self._ari_order > 0:
            # AR(p) or ARI(p,d): user passes exogenous X only; lag columns are
            # added internally via in-sample reuse or recursive forecasting.
            ar_coefs = np.array(list(self.arima_polynomial_.values()))
            n_exog = self._n_features_exog
            if X.shape[1] != n_exog:
                raise ValueError(
                    f"X has {X.shape[1]} features, expected {n_exog} "
                    f"(exogenous features only; ARI lag columns are added internally)"
                )

            if in_sample:
                # In-sample: reuse training lag columns (matches fitted values exactly)
                X_lags = self._X_train_[:, n_exog:]
                linear_pred = X @ B[:n_exog] + X_lags @ ar_coefs
            else:
                # Out-of-sample: the lags start from the state after the last
                # observation (the filter over the latent sizes of an occurrence model,
                # the last values otherwise), and the forecasts go on with the recursion
                # of the model, with the variances of its errors over the horizon (the
                # Kalman filter for the distributions additive in the space of the
                # lags), as R
                from ._native_alm_latent import alm_latent

                h = len(X)
                y_train = np.asarray(self._y_train_, dtype=float)
                occurrence_model = isinstance(self.occurrence_, ALM)
                observed = (
                    y_train != 0
                    if occurrence_model
                    else np.ones(len(y_train), dtype=bool)
                )
                lags_in = self._X_train_[:, n_exog:]
                response = (
                    bc_transform(y_train, self.other_)
                    if self.distribution == "dbcnorm" and occurrence_model
                    else self._ari_response
                )
                mu_in = np.asarray(self.mu_, dtype=float)
                if (
                    self.distribution in _log_link_distributions
                    or self.distribution == "dbeta"
                ):
                    location_in = np.log(mu_in)
                elif self.distribution == "dchisq":
                    location_in = np.sqrt(mu_in)
                else:
                    location_in = mu_in
                ari_filter = self.distribution in ARI_FILTER_DISTRIBUTIONS
                latent_in = alm_latent(
                    location_in - lags_in @ ar_coefs,
                    ar_coefs,
                    response,
                    observed.astype(int),
                    lags_in[0],
                    np.zeros(0),
                    ari_filter,
                )
                latent_out = alm_latent(
                    X @ B[:n_exog],
                    ar_coefs,
                    np.zeros(h),
                    np.zeros(h, dtype=int),
                    latent_in["state_mean"],
                    np.ravel(latent_in["state_covariance"], order="F"),
                    ari_filter,
                )
                X_lags = latent_out["lags"]
                linear_pred = latent_out["mu"]
                ari_variance = latent_out["variance"]

            # The derivatives of the location by the AR parameters are the lags
            # differenced by the I part, as R
            i_order = self._i_order
            derivatives = np.zeros((self._ari_order, self._ar_order))
            for j in range(self._ar_order):
                derivatives[j : j + i_order + 1, j] = (-1.0) ** np.arange(
                    i_order + 1
                ) * comb(i_order, np.arange(i_order + 1))
            X = np.hstack([X, X_lags @ derivatives])
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

        # The shapes of dbeta, from the two parts of its parameters, as R
        shape1 = shape2 = None
        if self.distribution == "dbeta":
            shape1, shape2 = mu, np.exp(X @ self._B_opt_[len(B) :])
        scale_predicted = self.scale if scale_new is None else scale_new
        mean = extractor_fitted(
            self.distribution,
            mu,
            shape2
            if shape2 is not None
            else (
                scale_sd(self.distribution, scale_predicted)
                if scale_predicted is not None
                else 1.0
            ),
            self.other_ if self.distribution == "dbcnorm" else 0.0,
        )

        variances = None
        scale_values: float | np.ndarray | None = None
        if interval == "none":
            lower = None
            upper = None
        else:
            # Compute confidence variance once; add sigma² for prediction.
            # dbeta takes the variances of its shapes in place of those of the location
            variances_conf = (
                self._calculate_variance(X, "confidence")
                if self.distribution != "dbeta"
                else np.zeros(len(X))
            )
            variances_interval = variances = variances_conf
            d = self.distribution
            if interval == "prediction":
                # The variance of the error from the scales of the scale model, or
                # sigma², as R
                variance_error = np.asarray(self.sigma**2 * ari_variance)
                if scale_new is not None and self._ari_order == 0:
                    variance_error = {
                        "dlaplace": 2 * scale_new**2,
                        "dllaplace": 2 * scale_new**2,
                        "ds": 120 * scale_new**4,
                        "dls": 120 * scale_new**4,
                        "dlogis": scale_new**2 * np.pi**2 / 3,
                    }.get(d, scale_new)
                    if d in ("dgnorm", "dlgnorm"):
                        variance_error = (
                            scale_new**2
                            * _sp_gamma(3 / self.other_)
                            / _sp_gamma(1 / self.other_)
                        )
                    elif d == "dalaplace":
                        variance_error = (
                            scale_new**2
                            * ((1 - self.other_) ** 2 + self.other_**2)
                            / (self.other_ * (1 - self.other_)) ** 2
                        )
                variances_interval = variances = variances_conf + variance_error
                # R takes the estimated scale of dlnorm and dlogitnorm in place of
                # sigma² for the interval, without a scale model
                if d in ("dlnorm", "dlogitnorm") and scale_new is None:
                    variances_interval = (
                        variances - self.sigma**2 + np.asarray(self.scale, dtype=float)
                    )
            # The variances of dbeta are those of its shapes, as R
            if self.distribution == "dbeta":
                s1, s2 = np.asarray(shape1), np.asarray(shape2)
                variances = s1 * s2 / ((s1 + s2) ** 2 * (s1 + s2 + 1))

            # R: the levels of the sizes, given the probability of a zero
            levels = level if isinstance(level, list) else [level]
            if probability is not None and interval == "prediction":
                levels = [
                    np.maximum((lev - (1 - probability)) / probability, 0)
                    for lev in levels
                ]
            # The scale of the distribution of each prediction, as R's $scale
            scale_values = np.sqrt(variances_interval)
            if d in ("dlaplace", "dllaplace"):
                scale_values = np.sqrt(variances_interval / 2)
            elif d == "dalaplace":
                alpha = self.other_
                scale_values = np.sqrt(
                    variances_interval
                    * alpha**2
                    * (1 - alpha) ** 2
                    / (alpha**2 + (1 - alpha) ** 2)
                )
            elif d in ("ds", "dls"):
                scale_values = (variances_interval / 120) ** 0.25
            elif d in ("dgnorm", "dlgnorm"):
                shape = self.other_
                scale_values = np.sqrt(
                    variances_interval * _sp_gamma(1 / shape) / _sp_gamma(3 / shape)
                )
            elif d == "dlogis":
                scale_values = np.sqrt(variances_interval * 3 / np.pi**2)
            elif d in ("dchisq", "dnbinom", "plogis", "pnorm"):
                scale_values = self.scale
            elif d in ("dinvgauss", "dgamma") and interval == "prediction":
                scale_values = scale_predicted
            elif d == "dpois":
                scale_values = mean
            elif d == "dbinom":
                scale_values = self.size_
            lower, upper = self._calculate_quantiles(
                linear_pred if self.distribution != "dbeta" else (shape1, shape2),
                variances_interval if self.distribution != "dbeta" else variances,
                interval,
                levels,
                side,
                normal_dnorm=probability is not None,
                scale=scale_predicted,
            )

        # The occurrence scales the mean and the bounds of the confidence interval,
        # as R; the prediction interval stays the one of the sizes, as in smooth
        factor = None
        if probability is not None:
            factor = probability
        elif occurrence is not None:
            factor = np.ravel(np.asarray(occurrence, dtype=float))
        if factor is not None:
            mean = mean * factor
        if factor is not None and interval == "confidence":
            lower, upper = (
                None
                if bound is None
                else bound * factor.reshape(-1, *[1] * (bound.ndim - 1))
                for bound in (lower, upper)
            )

        return PredictionResult(
            mean=mean,
            lower=lower,
            upper=upper,
            level=level,
            variances=variances,
            side=side,
            interval=interval,
            occurrence=occurrence_predict,
            location=linear_pred,
            scale=scale_values,
            distribution=self.distribution,
            model=self,
            newdata=newdata,
            newdata_provided=not in_sample,
        )

    def plot(
        self,
        which=(1, 2, 4, 6),
        level: float = 0.95,
        legend: bool = False,
        ask: bool = False,
        lowess: bool = True,
        **kwargs,
    ):
        """Plots of the fit and residuals, as R's ``plot.greybox()``.

        Parameters
        ----------
        which : sequence of int, default=(1, 2, 4, 6)
            The panels to draw, from 1 to 16 (see R's ``plot.greybox()``): actuals
            vs fitted, standardised / studentised / absolute / squared residuals vs
            fitted, Q-Q plot, actuals and fitted over time, residuals over time,
            ACF / PACF of the residuals and of their squares, Cook's distance.
        level : float, default=0.95
            The confidence level of the bounds on the residuals.
        legend : bool, default=False
            Whether to draw the legends.
        ask : bool, default=False
            Whether to wait for the user before each panel after the first.
        lowess : bool, default=True
            Whether to draw the LOWESS lines.
        **kwargs
            ``main``, ``xlab``, ``ylab``, ``xlim``, ``ylim`` and ``type``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .plots import _plot_alm

        return _plot_alm(
            self,
            which=which,
            level=level,
            legend=legend,
            ask=ask,
            lowess=lowess,
            **kwargs,
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
            Number of parameters in the model (including intercept and scale), and in
            its occurrence model, as R's nparam().
        """
        if self._X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        n_params = self._n_params_
        if isinstance(self.occurrence_, ALM):
            n_params += self.occurrence_.nparam
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
        """The data of the model, as R's ``$data``: the response in the first column
        and the regressors of the model (without the intercept) in the others.

        Returns
        -------
        data : np.ndarray
            Array of shape (n_samples, 1 + n_regressors).
        """
        if self._X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        regressors = self._X_train_[:, int(self._intercept_needed) :]
        return np.column_stack([self._y_train_, regressors])

    @property
    def df_(self) -> int:
        """The number of the estimated parameters of the model, without its
        occurrence model, as R's ``$df`` (and ``$rank``)."""
        if self._X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._n_params_

    @property
    def rank_(self) -> int:
        """The rank of the model, as R's ``$rank``: its number of parameters."""
        return self.df_

    @property
    def B_(self) -> np.ndarray:
        """All the estimated parameters, the extra parameter of the distribution
        first, as R's ``$B``."""
        if self._B_opt_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.asarray(self._B_opt_, dtype=float)

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
            where n is the number of observations and k is :attr:`nparam`, as R's
            sigma.alm(). For dinvgauss/dgamma/dexp, uses (residuals - 1) since
            residuals are on a multiplicative scale (y/mu); for ROLE it is the
            trimmed mean of the squared residuals; for plogis/pnorm, the scale.
        """
        if self.residuals_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.distribution in ("plogis", "pnorm"):
            return float(self.scale) if self.scale is not None else np.nan
        resid = np.asarray(self.residuals_, dtype=float)
        occurrence_model = isinstance(self.occurrence_, ALM)
        n_param = self.nparam
        if self.distribution in ("dinvgauss", "dgamma", "dexp"):
            # R divides by all the observations here
            return float(np.sqrt(np.sum((resid - 1) ** 2) / (len(resid) - n_param)))
        elif self.loss == "ROLE":
            return float(
                np.sqrt(mean_fast(resid**2, trim=self._trim_value(), side="both"))
            )
        # The errors after the latent sizes of a dynamic occurrence model, standardised
        # by the variances of the Kalman filter, as R
        latent = getattr(self, "_latent", None)
        if latent is not None and latent["filter"]:
            polynomial = list(self.arima_polynomial_.values())
            resid = resid / np.sqrt(alm_latent_variance(polynomial, latent["observed"]))
        # With an occurrence model, R's sigma.alm() divides by the observed sizes less
        # the parameters of both models
        n_obs = (
            int(np.sum(np.asarray(self._y_train_) != 0))
            if occurrence_model
            else len(resid)
        )
        return float(np.sqrt(np.sum(resid**2) / (n_obs - n_param)))

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

    def extract_scale(self):
        """The scale of the distribution (R: ``extractScale``).

        The scales of the observations from the scale model (``scale_model``), when
        the model has one, and :attr:`scale` otherwise. It is the variance for the
        Normal-based distributions, as in the ADAM monograph.
        """
        if self.scale_model is not None:
            return np.asarray(self.scale_model.fitted, dtype=float)
        return self.scale

    def extract_sigma(self):
        """The standard deviation of the error term (R: ``extractSigma``).

        The one of each observation, implied by the scales of the scale model, when
        the model has one, and :attr:`sigma` otherwise.
        """
        d = self.distribution
        if self.scale_model is None or d not in (
            *VARIANCE_SCALE_DISTRIBUTIONS,
            "dinvgauss",
            "dgamma",
            "dlaplace",
            "dllaplace",
            "dalaplace",
            "ds",
            "dls",
            "dgnorm",
            "dlgnorm",
            "dlogis",
        ):
            return self.sigma
        scale = self.extract_scale()
        other = self.other_
        if d in ("dlaplace", "dllaplace"):
            return np.sqrt(2 * scale**2)
        if d in ("ds", "dls"):
            return np.sqrt(120 * scale**4)
        if d in ("dgnorm", "dlgnorm"):
            return np.sqrt(scale**2 * _sp_gamma(3 / other) / _sp_gamma(1 / other))
        if d == "dlogis":
            return scale * np.pi / np.sqrt(3)
        if d == "dalaplace":
            return scale * np.sqrt(
                (other**2 + (1 - other) ** 2) / (other**2 * (1 - other) ** 2)
            )
        return np.sqrt(scale)

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
        """Loss function name (ADAM convention with trailing _): "custom" for a
        function, as R."""
        return "custom" if callable(self.loss) else self.loss

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

        # dbeta has the parameters of both of its shapes, as R
        if self.distribution == "dbeta":
            return np.asarray(self._B_opt_, dtype=float)
        result = np.concatenate([[self.intercept_], self._coef])
        return result

    def summary(
        self, level: float = 0.95, bootstrap: bool = False, **kwargs
    ) -> "SummaryResult":
        """Model summary.

        Parameters
        ----------
        level : float, optional
            Confidence level for parameter intervals. Default is 0.95.
        bootstrap : bool, default=False
            Whether the standard errors and the intervals come from the bootstrapped
            parameters (their covariance matrix and quantiles), as R.
        **kwargs
            Passed to :func:`greybox.coefbootstrap` (``nsim``, ``size``, ...).

        Returns
        -------
        SummaryResult
            Summary of the model with coefficient estimates, standard errors,
            t-statistics, p-values, and confidence intervals.
        """
        from .methods.summary import SummaryResult

        if self._coef is None:
            raise ValueError("Model not fitted. Call fit() first.")

        coefficients = np.asarray(self.coefficients, dtype=float)
        if bootstrap:
            from .bootstrap import coefbootstrap

            bootstrapped = coefbootstrap(self, **kwargs)
            vcov_matrix = bootstrapped.vcov
        else:
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
        if bootstrap:
            lower_ci, upper_ci = np.quantile(
                bootstrapped.coefficients, [(1 - level) / 2, (1 + level) / 2], axis=0
            )

        # The parameters of the scale model follow, with the degrees of freedom of
        # the model, as R
        feature_names = self._feature_names
        scale_parameters = None
        if self.scale_model is not None:
            scale_coefficients = np.asarray(self.scale_model.coefficients, dtype=float)
            scale_se = np.sqrt(np.diag(self.scale_model.vcov()))
            coefficients = np.concatenate([coefficients, scale_coefficients])
            se = np.concatenate([se, scale_se])
            t_stat = coefficients / se
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=self.df_residual_))
            lower_ci = np.concatenate(
                [lower_ci, scale_coefficients - t_crit * scale_se]
            )
            upper_ci = np.concatenate(
                [upper_ci, scale_coefficients + t_crit * scale_se]
            )
            scale_parameters = np.arange(len(coefficients)) >= n_free
            feature_names = list(
                self._feature_names or [f"x{i}" for i in range(1, n_free)]
            ) + (["(Intercept)"] if self.scale_model._intercept_needed else [])
            feature_names += list(
                self.scale_model._feature_names
                or [f"x{i}" for i in range(1, len(scale_coefficients) + 1)]
            )

        # The significance and the coefficients of determination of R's summary()
        significance = ~((lower_ci <= 0) & (upper_ci >= 0))
        actuals = np.asarray(self.actuals, dtype=float)
        r_squared = 1 - np.sum(np.asarray(self.residuals) ** 2) / np.sum(
            (actuals - np.mean(actuals)) ** 2
        )
        adj_r_squared = 1 - (1 - r_squared) * (self.nobs - 1) / (
            self.nobs - self.nparam
        )

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
            feature_names=feature_names,
            time_elapsed=self.time_elapsed,
            arima_string=self.arima_string_,
            significance=significance,
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            scale_parameters=scale_parameters,
        )

    def confint(
        self,
        parm: int | list[int] | None = None,
        level: float = 0.95,
        bootstrap: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Confidence intervals for parameters.

        Parameters
        ----------
        parm : int or list of int, optional
            Which parameters to include. If None, all parameters are included.
            0 = intercept, 1, 2, ... = coefficients, followed by those of the scale
            model (``scale_model``), as R.
        level : float, optional
            Confidence level. Default is 0.95.
        bootstrap : bool, default=False
            Whether the bounds are the quantiles of the bootstrapped parameters, as R
            (of the location model only).
        **kwargs
            Passed to :func:`greybox.coefbootstrap` (``nsim``, ``size``, ...).

        Returns
        -------
        confint : np.ndarray
            Array with shape (n_params, 2) containing lower and upper bounds.
        """
        if self._coef is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if bootstrap:
            from .bootstrap import coefbootstrap

            lower_ci, upper_ci = np.quantile(
                coefbootstrap(self, **kwargs).coefficients,
                [(1 - level) / 2, (1 + level) / 2],
                axis=0,
            )
        else:
            coefficients = np.asarray(self.coefficients, dtype=float)
            se = np.sqrt(np.diag(self.vcov()))[: len(coefficients)]
            # The parameters of the scale model follow, as R
            if self.scale_model is not None:
                coefficients = np.concatenate(
                    [coefficients, self.scale_model.coefficients]
                )
                se = np.concatenate([se, np.sqrt(np.diag(self.scale_model.vcov()))])
            t_crit = stats.t.ppf((1 + level) / 2, df=self.df_residual_)
            lower_ci = coefficients - t_crit * se
            upper_ci = coefficients + t_crit * se

        if parm is not None:
            if isinstance(parm, int):
                parm = [parm]
            lower_ci = lower_ci[parm]
            upper_ci = upper_ci[parm]

        return np.column_stack([lower_ci, upper_ci])

    def hatvalues(self) -> np.ndarray:
        """Leverage of the observations, as R's hatvalues().

        The diagonal of the hat matrix of the regressors for the Normal-based and
        the count distributions, and of ``X V X' / sigma^2`` with the covariance
        matrix ``V`` of the parameters for the others.

        Returns
        -------
        np.ndarray
            The hat values, one per observation.
        """
        if self._X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X = self._X_train_
        if self.distribution in (
            "dt",
            "dnorm",
            "dlnorm",
            "dbcnorm",
            "dlogitnorm",
            "dnbinom",
            "dbinom",
            "dpois",
        ):
            # The squares of the orthonormal basis of the regressors, as R's hat()
            basis, singular, _ = np.linalg.svd(X, full_matrices=False)
            hat_values = np.sum(basis[:, singular > singular[0] * 1e-7] ** 2, axis=1)
        else:
            hat_values = np.sum(X @ self.vcov() * X, axis=1) / (
                np.asarray(self.extract_sigma()) ** 2
            )
        hat_values[hat_values == 1] = 1 - 1e-5
        return hat_values

    def rstandard(self) -> np.ndarray:
        """Standardised residuals, as R's rstandard().

        The residuals divided by their scale (with the leverage for the Normal-based
        distributions and dt), debiased for the number of parameters; for the count
        distributions, the standard Normal quantiles of the cumulative probabilities
        of the actuals. With an occurrence model, only the non-zero observations are
        standardised and the others are NaN.

        Returns
        -------
        np.ndarray
            The standardised residuals, one per observation.
        """
        from .pointlik import point_lik_cumulative

        obs = self.nobs
        df = obs - self.nparam
        errors = np.array(self.residuals, dtype=float)
        observed = (
            np.asarray(self.occurrence_.actuals) != 0
            if isinstance(self.occurrence_, ALM)
            else np.ones(obs, dtype=bool)
        )
        scale, shape = np.asarray(self.extract_scale(), dtype=float), self.other_
        # The scales of a scale model of the non-zeros, which are standardised
        scale_observed = scale[observed] if scale.size == obs else scale
        centred = errors[observed] - np.mean(errors[observed])
        if self.distribution in ("dnorm", "dlnorm", "dbcnorm", "dlogitnorm"):
            errors = errors / (np.sqrt(scale) * np.sqrt(1 - self.hatvalues()))
        elif self.distribution == "dt":
            errors = errors / (scale * np.sqrt(1 - self.hatvalues()))
        elif self.distribution in ("ds", "dls"):
            errors[observed] = centred / (scale_observed * obs / df) ** 2
        elif self.distribution in ("dgnorm", "dlgnorm"):
            errors[observed] = centred / (scale_observed**shape * obs / df) ** (
                1 / shape
            )
        elif self.distribution in ("dinvgauss", "dgamma", "dexp"):
            errors[observed] = errors[observed] / np.mean(errors[observed])
        elif self.distribution in ("dfnorm", "drectnorm"):
            errors[observed] = errors[observed] / np.sqrt(scale_observed * obs / df)
        elif self.distribution in ("dpois", "dnbinom", "dbinom", "dgeom"):
            errors[observed] = stats.norm.ppf(point_lik_cumulative(self)[observed])
        else:
            errors[observed] = centred / (scale_observed * obs / df)
        errors[~observed] = np.nan
        return errors

    def rstudent(self) -> np.ndarray:
        """Studentised residuals, as R's rstudent().

        Each residual divided by the scale estimated without it (with the leverage
        for the Normal-based distributions and dt); for the count distributions, the
        standard Normal quantiles of the cumulative probabilities of the actuals.
        With an occurrence model, only the non-zero observations are studentised and
        the others are NaN.

        Returns
        -------
        np.ndarray
            The studentised residuals, one per observation.
        """
        from .pointlik import point_lik_cumulative

        obs = self.nobs
        df = obs - self.nparam - 1
        errors = np.array(self.residuals, dtype=float)
        studentised = errors.copy()
        observed = (
            np.asarray(self.occurrence_.actuals) != 0
            if isinstance(self.occurrence_, ALM)
            else np.ones(obs, dtype=bool)
        )
        shape = alpha = self.other_
        # The scale without each observation, from the sums less its own term
        if self.distribution in ("dt", "dnorm", "dlnorm", "dlogitnorm", "dbcnorm"):
            errors = errors - np.mean(errors)
            terms = errors**2
            scale = np.sqrt((np.sum(terms) - terms) / df * (1 - self.hatvalues()))
        elif self.distribution in ("ds", "dls"):
            errors = errors - np.mean(errors)
            terms = np.sqrt(np.abs(errors))
            scale = ((np.sum(terms) - terms) / (2 * df)) ** 2
        elif self.distribution in ("dlaplace", "dllaplace"):
            errors = errors - np.mean(errors)
            terms = np.abs(errors)
            scale = (np.sum(terms) - terms) / df
        elif self.distribution == "dalaplace":
            terms = errors * (alpha - (errors <= 0))
            scale = (np.sum(terms) - terms) / df
        elif self.distribution in ("dgnorm", "dlgnorm"):
            terms = np.abs(errors) ** shape
            scale = ((np.sum(terms) - terms) * (shape / df)) ** (1 / shape)
        elif self.distribution == "dlogis":
            errors = errors - np.mean(errors)
            terms = errors**2
            scale = np.sqrt((np.sum(terms) - terms) / df) * np.sqrt(3) / np.pi
        elif self.distribution in ("dinvgauss", "dgamma", "dexp"):
            scale = (np.sum(errors) - errors) / (obs - 1)
        else:
            terms = errors**2
            scale = np.sqrt((np.sum(terms) - terms) / df)
        # The count distributions are not studentised
        if self.distribution in ("dpois", "dnbinom", "dbinom", "dgeom"):
            studentised[observed] = stats.norm.ppf(point_lik_cumulative(self)[observed])
        else:
            studentised[observed] = (errors / scale)[observed]
        studentised[~observed] = np.nan
        return studentised

    def _trim_value(self) -> float:
        """The trimming of ROLE: 0.05 unless provided, as in R; 0 for other losses."""
        if self.loss != "ROLE":
            return 0.0
        return 0.05 if self.trim is None else float(self.trim)

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
