"""Scale model of ALM: the Python port of R's ``sm()``, ``scaler()`` and
``implant()`` (``R/scaler.R``).

A scale model is a regression for the scale of the distribution of a fitted
location model, estimated by likelihood on its residuals (on the non-zeros of its
occurrence model), with the location fixed. ``ALM(scale_formula=...)`` estimates
it together with the location model; ``sm()`` estimates it for a fitted model and
``implant()`` puts it in the location model, as in R.
"""

import copy
import time as time_module

import nlopt
import numpy as np
import pandas as pd
from scipy import stats

from . import distributions as dist
from .alm import ALM, NLOPT_ALGORITHMS, PredictionResult, _numerical_hessian
from .fitters import VARIANCE_SCALE_DISTRIBUTIONS, _lm_fit, alm_latent_variance
from .methods.summary import SummaryResult


class ScaleModel(ALM):
    """The scale model of a location model, as R's class ``"scale"``.

    Built by :func:`sm` (or by ``ALM(scale_formula=...)``), not by ``fit()``. Its
    coefficients are those of the logarithm of the scale, its fitted values are the
    scales of the observations (the variance for the Normal-based distributions, as
    in the ADAM monograph), its residuals are those of the location model divided
    by them, and its actuals are the transformed residuals of the location model
    (R's ``$data[,1]``). Its likelihood is that of the location model with these
    scales. ``location_`` is the location model.
    """

    location_: ALM | None = None
    _FI: np.ndarray | None = None

    def fit(self, X, y, formula=None, feature_names=None):
        """Not available: a scale model is estimated by :func:`sm`."""
        raise TypeError("A scale model is estimated by sm(), not by fit().")

    def vcov(self, bootstrap: bool = False, **kwargs) -> np.ndarray:
        """Covariance matrix of the parameters: the inverse of the Hessian of the
        negative log-likelihood, as R's ``vcov.scale()`` (``bootstrap`` is ignored,
        as there)."""
        if self.location_ is None or self._B_opt_ is None:
            raise ValueError("Model not fitted. Call sm() first.")
        arguments = {"formula": self._X_train_, "parameters": self._B_opt_, "FI": True}
        FI = np.asarray(sm(self.location_, **arguments)._FI, dtype=float)
        # A variable that does not impact the likelihood needs a different step
        if np.any(np.all(FI == 0, axis=1) | np.any(np.isnan(FI), axis=1)):
            arguments["stepSize"] = np.finfo(float).eps ** (1 / 6)
            FI = np.asarray(sm(self.location_, **arguments)._FI, dtype=float)
        try:
            L = np.linalg.cholesky(FI)
            return np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(FI))))
        except np.linalg.LinAlgError:
            try:
                return np.linalg.inv(FI)
            except np.linalg.LinAlgError:
                return np.eye(len(FI)) * 1e100

    def predict(
        self,
        X=None,
        interval="none",
        level=0.95,
        side="both",
        occurrence=None,
    ) -> PredictionResult:
        """Predict the scale, as R's ``predict.scale()``.

        Parameters
        ----------
        X : DataFrame, dict or array-like, optional
            The new data: the data for the formula of the scale model, a DataFrame
            with the columns of its design matrix, or the design matrix itself.
            Without it, the in-sample values.
        interval : {"none", "confidence", "prediction"}, default="none"
            The interval for the logarithm of the scale, returned in the scale.
        level : float or list of float, default=0.95
            Confidence level(s).
        side : {"both", "upper", "lower"}, default="both"
            Side of the interval.
        occurrence : None
            Not used, as R.

        Returns
        -------
        PredictionResult
            The scales (``mean``) and the bounds of their intervals.
        """
        if self._B_opt_ is None:
            raise ValueError("Model not fitted. Call sm() first.")
        names = (["(Intercept)"] if self._intercept_needed else []) + list(
            self._feature_names or []
        )
        if X is None:
            Z = self._X_train_
        elif isinstance(self._formula_, str):
            from .formula import formula

            # With the levels of the factors of the training data, as R's xlevels
            frame = pd.DataFrame(X)
            frame.attrs["_xlevels"] = self._xlevels or {}
            Z = formula(self._formula_, frame, return_type="X")
        elif isinstance(X, pd.DataFrame) and self._feature_names is not None:
            regressors = list(self._feature_names)
            # The data with the columns of the scale model, with or without the
            # column of the intercept
            Z = np.empty((len(X), 0))
            if set(names) <= set(X.columns):
                Z = X[names]
            elif set(regressors) <= set(X.columns):
                Z = np.column_stack(
                    [np.ones((len(X), int(self._intercept_needed))), X[regressors]]
                )
        else:
            Z = X
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        if Z.shape[1] != len(self._B_opt_):
            raise ValueError(
                f"The scale model needs the regressors {names or len(self._B_opt_)} "
                "of the new data: give them as a DataFrame with these columns."
            )

        levels = level if isinstance(level, list) else [level]
        if side == "upper":
            level_low, level_up = [0.0] * len(levels), levels
        elif side == "lower":
            level_low, level_up = [1 - lev for lev in levels], [1.0] * len(levels)
        else:
            level_low = [(1 - lev) / 2 for lev in levels]
            level_up = [(1 + lev) / 2 for lev in levels]
        linear_pred = Z @ self._B_opt_
        variances = lower = upper = None
        if interval != "none":
            variances = np.abs(np.sum(Z @ self.vcov() * Z, axis=1))
            if interval == "prediction":
                variances = variances + self.sigma**2
            quantiles = stats.t.ppf(np.array([level_low, level_up]), self.df_residual_)
            lower, upper = (
                linear_pred[:, None] + q[None, :] * np.sqrt(variances)[:, None]
                for q in quantiles
            )
            if len(levels) == 1:
                lower, upper = lower.ravel(), upper.ravel()
        # The scale and the bounds of its interval, from their logarithms
        d = self.distribution
        power = {"dlogis": 0.5, "ds": 2.0, "dls": 2.0}.get(d, 1.0)
        if d in ("dgnorm", "dlgnorm"):
            power = 1 / self.other_
        mean, lower, upper = (
            None if bound is None else np.exp(bound) ** power
            for bound in (linear_pred, lower, upper)
        )
        return PredictionResult(
            mean=mean,
            lower=None if side == "upper" else lower,
            upper=None if side == "lower" else upper,
            level=level,
            variances=variances,
            side=side,
            interval=interval,
            location=linear_pred,
            distribution=d,
            model=self,
            newdata=X,
            newdata_provided=X is not None,
        )

    def summary(self, level: float = 0.95, bootstrap: bool = False, **kwargs):
        """Summary of the scale model, as R's ``summary.scale()``."""
        coefficients = np.asarray(self.coefficients, dtype=float)
        se = np.sqrt(np.diag(self.vcov()))
        t_crit = stats.t.ppf((1 + level) / 2, df=self.df_residual_)
        lower_ci, upper_ci = coefficients - t_crit * se, coefficients + t_crit * se
        return SummaryResult(
            coefficients=coefficients,
            se=se,
            t_stat=coefficients / se,
            p_value=2 * (1 - stats.t.cdf(np.abs(coefficients / se), self.df_residual_)),
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            distribution=self.distribution,
            nobs=self.nobs,
            nparam=self.nparam,
            df_residual=self.df_residual_,
            log_lik=self.loglik,
            aic=self.aic,
            bic=self.bic,
            aicc=self.aicc,
            bicc=self.bicc,
            response_variable=self._response_name or "y",
            feature_names=self._feature_names,
            significance=~((lower_ci <= 0) & (upper_ci >= 0)),
            scale_parameters=np.ones(len(coefficients), dtype=bool),
        )

    def extract_scale(self):
        """The scale of a scale model is 1, as R's ``extractScale()``: it *is* the
        scale."""
        return 1.0

    def rstandard(self) -> np.ndarray:
        """The residuals of a scale model are standardised already, as R: those of
        the zeros of an occurrence model are NaN."""
        errors = np.array(self.residuals, dtype=float)
        if isinstance(self.occurrence_, ALM):
            errors[np.asarray(self.occurrence_.actuals) == 0] = np.nan
        return errors

    def __str__(self):
        coefficients = np.asarray(self.coefficients, dtype=float)
        names = (["(Intercept)"] if self._intercept_needed else []) + list(
            self._feature_names or []
        )
        names += [f"x{i}" for i in range(len(names) + 1, len(coefficients) + 1)]
        regressors = [name for name in names if name != "(Intercept)"]
        formula = self._formula_ or "~ " + " + ".join(regressors) + (
            "" if self._intercept_needed else " - 1"
        )
        lines = [f"Formula: {formula}", ""]
        lines.append("Coefficients:")
        lines.append("".join(f"{name:>14}" for name in names))
        lines.append("".join(f"{value:>14.6f}" for value in coefficients))
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"ScaleModel(distribution={self.distribution!r}, "
            f"fitted={self._B_opt_ is not None})"
        )


def sm(object, formula=None, data=None, parameters=None, **kwargs) -> ScaleModel:
    """Scale model for a fitted location model, as R's ``sm()``.

    The scale of the distribution of the location model is ``exp(Z @ B)``, mapped
    to the scale of the distribution (its square root for ``dlogis``, its square
    for ``ds`` and ``dls``, its power ``1/shape`` for ``dgnorm`` and ``dlgnorm``),
    where ``Z`` are the regressors of the scale. The parameters ``B`` maximise the
    likelihood of the location model, with its location fixed, on the non-zeros of
    its occurrence model.

    Parameters
    ----------
    object : ALM
        The fitted location model.
    formula : str or array-like, optional
        The regressors of the scale: a one-sided formula (e.g. ``"~ x1 + x2"``),
        evaluated with :func:`greybox.formula` on ``data``, or their design matrix
        (with the column of ones for the intercept), as ``X`` of ``ALM.fit``.
        Without it, the regressors of the location model, as R.
    data : DataFrame or dict, optional
        The data of the formula. Without it, the design matrix of the location
        model, with its column names.
    parameters : array-like, optional
        The parameters of the scale model, to use as they are instead of being
        estimated.
    **kwargs
        The options of the optimiser, as R's ellipsis: ``algorithm`` (default
        ``"NLOPT_LN_NELDERMEAD"``), ``maxeval`` (200 per parameter), ``maxtime``,
        ``xtol_rel`` (1e-8), ``xtol_abs`` (1e-8), ``ftol_rel`` (1e-10), ``ftol_abs``
        (0), ``print_level``, and ``B``, the starting values. The likelihood of the
        scale is flat in its parameters, so the estimation is accurate by default.

    Returns
    -------
    ScaleModel
        The estimated scale model; put it in the location model with
        :func:`implant`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from greybox import ALM, formula, sm, implant
    >>> rng = np.random.default_rng(41)
    >>> x1, x2 = rng.normal(10, 3, 100), rng.normal(50, 5, 100)
    >>> y = 100 + 0.5 * x1 - 0.75 * x2 + np.sqrt(np.exp(0.8 + 0.2 * x1)) * rng.normal(
    ...     size=100
    ... )
    >>> data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    >>> y, X = formula("y ~ x1 + x2", data)
    >>> location = ALM().fit(X, y)
    >>> scale = sm(location, "~ x1")
    >>> location = implant(location, scale)
    """
    if not isinstance(object, ALM) or object.mu_ is None:
        raise TypeError("sm() needs a fitted ALM.")
    start_time = time_module.time()
    distribution = object.distribution
    if distribution not in (
        *VARIANCE_SCALE_DISTRIBUTIONS,
        "dlogis",
        "dlaplace",
        "dllaplace",
        "dalaplace",
        "ds",
        "dls",
        "dgnorm",
        "dlgnorm",
        "dinvgauss",
        "dgamma",
    ):
        raise ValueError(f"sm() does not support distribution {distribution!r}.")
    other = object.other_
    y = np.asarray(object.actuals, dtype=float)
    mu = np.asarray(object.mu_, dtype=float)
    residuals = np.asarray(object.residuals, dtype=float)

    # The non-zeros of an occurrence model, with non-zero residuals
    occurrence_model = isinstance(object.occurrence_, ALM)
    otU = y != 0 if occurrence_model else np.ones(len(y), dtype=bool)
    otU = otU & (residuals != 0)
    # The errors after the latent sizes of a dynamic occurrence model have the
    # variances of the Kalman filter: the scale takes the variance, or its root
    scale_filter = 1.0
    latent = getattr(object, "_latent", None)
    if occurrence_model and latent is not None and latent.get("filter"):
        filter_power = {"dnorm": 1, "dlnorm": 1, "dbcnorm": 1, "ds": 0.25, "dls": 0.25}
        scale_filter = alm_latent_variance(
            list(object.arima_polynomial_.values()), y != 0
        )[otU] ** filter_power.get(distribution, 0.5)

    # The regressors of the scale, those of the location model by default
    n_exog = object._n_features_exog
    names: list[str] | None = (
        ["(Intercept)"] if object._intercept_needed else []
    ) + list(object._feature_names or [f"x{i}" for i in range(1, n_exog + 1)])[
        : n_exog - int(object._intercept_needed)
    ]
    formula_string = formula if isinstance(formula, str) else None
    if formula is None:
        Z = object._X_train_[:, :n_exog]
    elif formula_string is not None:
        from .formula import formula as formula_parser

        if data is None:
            data = pd.DataFrame(object._X_train_[:, :n_exog], columns=names)
        Z = formula_parser(formula_string, data, return_type="X")
    else:
        Z, names = formula, None
    xlevels = Z.attrs.get("_xlevels") if isinstance(Z, pd.DataFrame) else None
    if isinstance(Z, pd.DataFrame):
        names = [str(name) for name in Z.columns]
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    intercept_needed = bool(np.all(Z[:, 0] == 1))
    n_variables = Z.shape[1]

    # The scale of the distribution from the parameters
    power = {"dlogis": 0.5, "ds": 2.0, "dls": 2.0}.get(distribution, 1.0)
    if distribution in ("dgnorm", "dlgnorm"):
        power = 1 / other

    y_observed, mu_observed = y[otU], mu[otU]
    log_y = np.log(y_observed) if distribution in ("dllaplace", "dls", "dlgnorm") else 0

    def cf_scale(B):
        scale = np.exp(Z[otU] @ B) ** power * scale_filter
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            densities = {
                "dnorm": lambda: dist.dnorm(
                    y_observed, mean=mu_observed, sd=np.sqrt(scale), log=True
                ),
                "dlaplace": lambda: dist.dlaplace(
                    y_observed, mu=mu_observed, scale=scale, log=True
                ),
                "ds": lambda: dist.ds(
                    y_observed, mu=mu_observed, scale=scale, log=True
                ),
                "dgnorm": lambda: dist.dgnorm(
                    y_observed, mu=mu_observed, scale=scale, shape=other, log=True
                ),
                "dlogis": lambda: dist.dlogis(
                    y_observed, location=mu_observed, scale=scale, log=True
                ),
                "dalaplace": lambda: dist.dalaplace(
                    y_observed, mu=mu_observed, scale=scale, alpha=other, log=True
                ),
                "dlnorm": lambda: dist.dlnorm(
                    y_observed, meanlog=mu_observed, sdlog=np.sqrt(scale), log=True
                ),
                "dllaplace": lambda: (
                    dist.dlaplace(log_y, mu=mu_observed, scale=scale, log=True) - log_y
                ),
                "dls": lambda: (
                    dist.ds(log_y, mu=mu_observed, scale=scale, log=True) - log_y
                ),
                "dlgnorm": lambda: (
                    dist.dgnorm(
                        log_y, mu=mu_observed, scale=scale, shape=other, log=True
                    )
                    - log_y
                ),
                "dbcnorm": lambda: dist.dbcnorm(
                    y_observed,
                    mu=mu_observed,
                    sigma=np.sqrt(scale),
                    lambda_bc=other,
                    log=True,
                ),
                "dfnorm": lambda: dist.dfnorm(
                    y_observed, mu=mu_observed, sigma=np.sqrt(scale), log=True
                ),
                "drectnorm": lambda: dist.drectnorm(
                    y_observed, mu=mu_observed, sigma=np.sqrt(scale), log=True
                ),
                "dinvgauss": lambda: dist.dinvgauss(
                    y_observed,
                    mean=mu_observed,
                    dispersion=scale / mu_observed,
                    log=True,
                ),
                "dgamma": lambda: dist.dgamma(
                    y_observed, shape=1 / scale, scale=scale * mu_observed, log=True
                ),
                "dlogitnorm": lambda: dist.dlogitnorm(
                    y_observed, mu=mu_observed, sigma=np.sqrt(scale), log=True
                ),
            }
            # Summed in long double, as R's sum()
            return -float(np.sum(densities[distribution]().astype(np.longdouble)))

    # The starting values: least squares on the logarithms of the residuals, in the
    # space of the scale of the distribution
    errors = np.abs(residuals[otU])
    if distribution in (*VARIANCE_SCALE_DISTRIBUTIONS, "dlogis"):
        response = 2 * np.log(errors)
    elif distribution in ("ds", "dls"):
        response = 0.5 * np.log(errors)
    elif distribution in ("dgnorm", "dlgnorm"):
        response = other + other * np.log(errors)
    elif distribution == "dgamma":
        response = 2 * np.log(np.abs(residuals[otU] - 1))
    elif distribution == "dinvgauss":
        response = np.log((residuals[otU] - 1) ** 2 / residuals[otU])
    else:
        response = np.log(errors)
    B = kwargs.get("B")
    if B is None:
        B = _lm_fit(Z[otU], response)[0]
    B = np.asarray(B if parameters is None else parameters, dtype=float)

    if parameters is None:
        opt = nlopt.opt(
            NLOPT_ALGORITHMS[kwargs.get("algorithm", "NLOPT_LN_NELDERMEAD")],
            n_variables,
        )
        opt.set_min_objective(lambda B, grad: cf_scale(B))
        opt.set_xtol_rel(kwargs.get("xtol_rel", 1e-8))
        opt.set_xtol_abs(kwargs.get("xtol_abs", 1e-8))
        opt.set_ftol_rel(kwargs.get("ftol_rel", 1e-10))
        opt.set_ftol_abs(kwargs.get("ftol_abs", 0))
        opt.set_maxeval(kwargs.get("maxeval", n_variables * 200))
        opt.set_maxtime(kwargs.get("maxtime", -1))
        B = opt.optimize(B)
        cf_value = opt.last_optimum_value()
        n_iter, nlopt_result = opt.get_numevals(), opt.last_optimize_result()
        if kwargs.get("print_level", 0) > 0:
            print(f"B = {B}, CF = {cf_value}")
    else:
        cf_value = cf_scale(B)
        n_iter, nlopt_result = 0, None

    scale = np.exp(Z @ B) ** power
    # The transformed residuals, as R's response of the scale model
    if distribution in (*VARIANCE_SCALE_DISTRIBUTIONS, "ds", "dls"):
        actuals = residuals**2
    elif distribution in ("dgnorm", "dlgnorm"):
        actuals = np.abs(residuals) ** (1 / other)
    else:
        actuals = np.abs(residuals)

    model = ScaleModel(distribution=distribution)
    model.location_ = object
    model.occurrence_ = object.occurrence_ if occurrence_model else None
    model.other_ = model._other_val_ = other
    model._a_parameter_provided_ = True
    model._ar_order = model._i_order = model._ari_order = 0
    model._latent = None
    model.arima_polynomial_ = model.arima_string_ = None
    model._formula_ = formula_string
    model._xlevels = xlevels
    model._response_name = object._response_name
    model._intercept_needed = intercept_needed
    model._feature_names = (
        None if names is None else [name for name in names if name != "(Intercept)"]
    )
    model._n_features = model._n_features_exog = model._n_features_input = n_variables
    model._columns_kept = np.arange(n_variables)
    model._B_opt_ = B
    model.intercept_ = B[0]
    model._coef = B[1:]
    model._X_train_ = Z
    model._y_train_ = actuals
    model.fitted_values_ = scale
    model.residuals_ = residuals / (
        np.sqrt(scale) if distribution in VARIANCE_SCALE_DISTRIBUTIONS else scale
    )
    model.n_iter_, model.nlopt_result_ = n_iter, nlopt_result
    model._loss_value = cf_value
    model._log_lik = -cf_value
    model._n_params_ = n_variables
    model._df_residual = len(y) - n_variables
    # The information criteria of R's logLik() of the scale model, whose parameters
    # include those of the occurrence model
    k, n = model.nparam, len(y)
    model._aic, model._aicc, model._bic, model._bicc = 2 * cf_value + k * np.array(
        [2, 2 * n / (n - k - 1), np.log(n), np.log(n) * n / (n - k - 1)]
    )
    if kwargs.get("FI", False):
        model._FI = _numerical_hessian(cf_scale, B, h=kwargs.get("stepSize"))
    model._time_elapsed = time_module.time() - start_time
    return model


def implant(location: ALM, scale: ScaleModel) -> ALM:
    """Implant the scale model in the location model, as R's ``implant()``.

    Parameters
    ----------
    location : ALM
        The fitted location model.
    scale : ScaleModel
        Its scale model, estimated by :func:`sm`.

    Returns
    -------
    ALM
        A copy of the location model with the scale model as its ``scale_model``:
        its parameters replace the scale of the location model in the number of
        parameters, and its likelihood (with that of the occurrence model) is the
        one of the model, as ``ALM(scale_formula=...)``. The scales of the
        prediction intervals and of the densities are those of the scale model.
    """
    if not isinstance(scale, ScaleModel):
        raise TypeError("sm is not a scale model. Cannot procede.")
    location = copy.copy(location)
    # The parameters of the scale model replace the scale of the location model (or
    # its previous scale model)
    replaced = (
        location.scale_model._n_params_
        if location.scale_model is not None
        else int(
            location.loss_ in ("likelihood", "ROLE")
            and location.distribution
            not in ("dexp", "dpois", "dgeom", "dbinom", "plogis", "pnorm")
        )
    )
    location.scale_model = scale
    location._n_params_ = location._n_params_ - replaced + scale._n_params_
    location._df_residual = location.nobs - location._n_params_
    # The likelihood of the scale model is that of the non-zeros: add the occurrence
    occurrence = location.occurrence_
    occurrence_model = isinstance(occurrence, ALM)
    log_lik = float(scale.loglik or 0.0)
    if isinstance(occurrence, ALM):
        log_lik += occurrence.loglik or 0.0
    location._log_lik = log_lik
    # The information criteria of R's logLik()
    k = location.nparam + (
        0
        if occurrence_model
        else int(
            (location.loss_, location.distribution)
            in (("MSE", "dnorm"), ("MAE", "dlaplace"), ("HAM", "ds"))
        )
    )
    n = location.nobs
    location._aic, location._aicc, location._bic, location._bicc = (
        -2 * log_lik
        + k * np.array([2, 2 * n / (n - k - 1), np.log(n), np.log(n) * n / (n - k - 1)])
    )
    return location
