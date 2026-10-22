"""Model selection and combination functions.

This module provides functions for stepwise regression, model combination,
and forecasting accuracy measures.
"""

import time
import warnings
from typing import Any, ClassVar, Literal, TypedDict

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats

from .alm import ALM, PredictionResult
from .fitters import _lm_fit
from .formula import formula as formula_func
from .methods.summary import DISTRIBUTION_NAMES
from .transforms import bc_transform, bc_transform_inv, mean_fast


class ModelInfo(TypedDict):
    vars: list[str]
    model: ALM
    ic: float
    coef: np.ndarray


class CALMSummary:
    """Summary of CALM result, matching R's summary.greyboxC."""

    def __init__(
        self,
        coefficients: np.ndarray,
        se: np.ndarray,
        importance: np.ndarray,
        lower_ci: np.ndarray,
        upper_ci: np.ndarray,
        coefficient_names: list[str],
        sigma: float,
        n_obs: int,
        nparam: float,
        df_residual: float,
        distribution: str,
        y_variable: str,
        ic_type: str,
        aic: float,
        aicc: float,
        bic: float,
        bicc: float,
    ):
        self.coefficients = coefficients
        self.se = se
        self.importance = importance
        self.lower_ci = lower_ci
        self.upper_ci = upper_ci
        self.coefficient_names = coefficient_names
        self.sigma = sigma
        self.n_obs = n_obs
        self.nparam = nparam
        self.df_residual = df_residual
        self.distribution = distribution
        self.y_variable = y_variable
        self.ic_type = ic_type
        self.aic = aic
        self.aicc = aicc
        self.bic = bic
        self.bicc = bicc

    def __str__(self) -> str:
        dist_name = DISTRIBUTION_NAMES.get(self.distribution, self.distribution)

        # The level of the intervals, from the Student quantile they were built with
        se = np.asarray(self.se, dtype=float)
        usable = np.flatnonzero(se > 0)
        level = np.nan
        if len(usable) > 0:
            j = usable[0]
            quantile = (self.upper_ci[j] - self.coefficients[j]) / se[j]
            level = 2 * stats.t.cdf(quantile, self.df_residual) - 1
        level_low = (1 - level) / 2 * 100
        level_high = (1 + level) / 2 * 100
        col_lo = f"Lower {level_low:.1f}%"
        col_hi = f"Upper {level_high:.1f}%"

        lines = []
        lines.append(f"The {self.ic_type} combined model")
        lines.append(f"Response variable: {self.y_variable}")
        lines.append(f"Distribution used in the estimation: {dist_name}")
        lines.append("Coefficients:")

        header = (
            f"{'':>12} {'Estimate':>12} {'Std. Error':>12} "
            f"{'Importance':>12} {col_lo:>12} {col_hi:>12}"
        )
        lines.append(header)

        for i, name in enumerate(self.coefficient_names):
            crosses_zero = self.lower_ci[i] <= 0 <= self.upper_ci[i]
            sig = " *" if not crosses_zero else ""
            lines.append(
                f"{name:>12} {self.coefficients[i]:>12.4f} "
                f"{self.se[i]:>12.4f} {self.importance[i]:>12.3f} "
                f"{self.lower_ci[i]:>12.4f} "
                f"{self.upper_ci[i]:>12.4f}{sig}"
            )

        lines.append("")
        lines.append(f"Error standard deviation: {self.sigma:.4f}")
        lines.append(f"Sample size: {self.n_obs}")
        lines.append(f"Number of estimated parameters: {self.nparam:.3f}")
        lines.append(f"Number of degrees of freedom: {self.df_residual:.3f}")
        lines.append("Approximate combined information criteria:")

        ic_header = f"{'AIC':>12}{'AICc':>12}{'BIC':>12}{'BICc':>12}"
        ic_vals = (
            f"{self.aic:>12.4f}{self.aicc:>12.4f}{self.bic:>12.4f}{self.bicc:>12.4f}"
        )
        lines.append(ic_header)
        lines.append(ic_vals)

        return "\n".join(lines)


class CALMResult:
    """Result of CALM, with print and summary support.

    Supports dict-like access for backwards compatibility and
    ALM-compatible interface (predict, score, confint, properties).
    """

    def __init__(self, **kwargs: Any):
        # Extract private attrs before storing the rest
        self._actuals = kwargs.pop("actuals", None)
        self._X_train = kwargs.pop("X_train", None)
        self._formula = kwargs.pop("formula_str", None)
        # The levels of the factors of the data, for the new data, as R's xlevels
        self._xlevels: dict[str, list] = {}
        self._vcov_matrix = kwargs.pop("vcov", None)
        self._fitted = kwargs.pop("fitted", None)
        self._residuals = kwargs.pop("residuals", None)
        self._coefficients = kwargs.pop("coefficients", None)
        self._time_elapsed = kwargs.pop("time_elapsed", 0.0)
        self._log_lik = kwargs.pop("log_lik", None)
        # Store commonly accessed attrs as private
        self._n_obs = kwargs.pop("n_obs", None)
        self._importance = kwargs.pop("importance", None)
        self._df_residual = kwargs.pop("df_residual", None)
        self._df = kwargs.pop("df", None)
        self._distribution = kwargs.pop("distribution", None)
        self._coefficient_names = kwargs.pop("coefficient_names", None)
        self._IC_type = kwargs.pop("IC_type", None)
        self._IC = kwargs.pop("IC", None)
        self._y_variable = kwargs.pop("y_variable", None)
        self._x_variables = kwargs.pop("x_variables", None)
        self._scale = kwargs.pop("scale", None)
        self._weights = kwargs.pop("weights", None)
        self._combination = kwargs.pop("combination", None)
        self._combination_col_names = kwargs.pop("combination_col_names", None)
        self._combination_row_names = kwargs.pop("combination_row_names", None)
        # The combined extra parameter of the distribution, as R's $other
        self.other_ = kwargs.pop("other_", None)
        for key, val in kwargs.items():
            setattr(self, key, val)

    # Map public dict keys to private attribute names
    _PRIVATE_MAP: ClassVar[dict[str, str]] = {
        "vcov": "_vcov_matrix",
        "fitted": "_fitted",
        "residuals": "_residuals",
        "coefficients": "_coefficients",
        "time_elapsed": "_time_elapsed",
        "log_lik": "_log_lik",
        "df": "_df",
    }

    def __getitem__(self, key: str) -> Any:
        """Dict-like access for backwards compat."""
        if key in self._PRIVATE_MAP:
            return getattr(self, self._PRIVATE_MAP[key])
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str) and key in self._PRIVATE_MAP:
            attr = self._PRIVATE_MAP[key]
            return getattr(self, attr) is not None
        return hasattr(self, str(key))

    def keys(self) -> list[str]:
        public = [k for k in self.__dict__ if not k.startswith("_")]
        # Add back keys that are stored privately but publicly accessible
        for pub_key, priv_attr in self._PRIVATE_MAP.items():
            if getattr(self, priv_attr, None) is not None:
                public.append(pub_key)
        return public

    # --- Properties for ALM compatibility ---

    @property
    def coefficients(self) -> np.ndarray:
        """Combined coefficients."""
        return self._coefficients

    @property
    def fitted(self) -> np.ndarray:
        """Fitted values."""
        return self._fitted

    @property
    def residuals(self) -> np.ndarray:
        """Model residuals."""
        return self._residuals

    @property
    def log_lik(self) -> float:
        """Combined log-likelihood."""
        return self._log_lik

    @property
    def loglik(self) -> float:
        """Log-likelihood (ADAM-compatible name)."""
        return self._log_lik

    @property
    def time_elapsed(self) -> float:
        """Time elapsed during computation (seconds)."""
        return self._time_elapsed

    @time_elapsed.setter
    def time_elapsed(self, value: float):
        self._time_elapsed = value

    @property
    def nobs(self) -> int:
        return self._n_obs

    @property
    def nparam(self) -> float:
        return float(np.sum(self.importance)) + 1

    @property
    def sigma(self) -> float:
        resid = self.residuals
        df_resid = self.df_residual
        if resid is None or df_resid is None or df_resid == 0:
            return 0.0
        return float(np.sqrt(np.sum(resid**2) / df_resid))

    @property
    def coef(self) -> np.ndarray:
        """Slope coefficients (excluding intercept)."""
        return self.coefficients[1:]

    @property
    def intercept_(self) -> float:
        return float(self.coefficients[0])

    @property
    def aic(self) -> float:
        k = self.nparam
        log_lik = self.log_lik
        if log_lik is None:
            return 0.0
        return -2 * log_lik + 2 * k

    @property
    def bic(self) -> float:
        k = self.nparam
        n = self._n_obs
        log_lik = self.log_lik
        if n is None or log_lik is None:
            return 0.0
        return -2 * log_lik + k * np.log(n)

    @property
    def aicc(self) -> float:
        k = self.nparam
        n = self._n_obs
        a = self.aic
        if n is None or n - k - 1 <= 0:
            return np.inf
        return a + (2 * k**2 + 2 * k) / (n - k - 1)

    @property
    def bicc(self) -> float:
        k = self.nparam
        n = self._n_obs
        if n is None or self.log_lik is None or n - k - 1 <= 0:
            return np.inf
        return -2 * self.log_lik + k * np.log(n) * n / (n - k - 1)

    @property
    def distribution_(self) -> str:
        """Distribution name (ADAM convention with trailing _)."""
        return self.distribution

    @property
    def loss_(self) -> str:
        """Loss function name (ADAM convention with trailing _)."""
        return "likelihood"

    @property
    def loss_value(self) -> float | None:
        """Loss function value."""
        if self._log_lik is None:
            return None
        return -self._log_lik

    @property
    def n_param(self) -> dict:
        """Parameter count information."""
        return {
            "number": self.nparam,
            "df": self.df_residual,
        }

    @property
    def actuals(self) -> np.ndarray:
        return self._actuals

    @property
    def data(self) -> np.ndarray:
        """The response and the regressors (without the intercept), as R's $data."""
        return np.column_stack([self._actuals, self._X_train[:, 1:]])

    @property
    def rank_(self) -> int:
        """The number of the combined parameters, as R's $rank."""
        return len(self.coefficients)

    @property
    def formula(self) -> str:
        return self._formula

    @property
    def _feature_names(self) -> list[str]:
        return self.coefficient_names[1:]

    @property
    def _n_features(self) -> int:
        return len(self.coefficients)

    @property
    def df_residual_(self) -> float:
        return self.df_residual

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self._n_obs

    @property
    def importance(self) -> np.ndarray:
        """Importance of variables (1 if included, 0 otherwise)."""
        return self._importance

    @property
    def df_residual(self) -> float:
        """Degrees of freedom for residuals."""
        return self._df_residual

    @property
    def df_model(self) -> float:
        """Degrees of freedom for model."""
        return self._df

    @property
    def distribution(self) -> str:
        """Distribution name."""
        return self._distribution

    @property
    def coefficient_names(self) -> list[str]:
        """Names of coefficients."""
        return self._coefficient_names

    @property
    def IC_type(self) -> str:
        """Information criterion type used."""
        return self._IC_type

    @property
    def IC(self) -> float:
        """Combined information criterion value."""
        return self._IC

    @property
    def y_variable(self) -> str:
        """Name of response variable."""
        return self._y_variable

    @property
    def x_variables(self) -> list[str]:
        """List of predictor variable names."""
        return self._x_variables

    @property
    def scale(self) -> float:
        """Scale parameter."""
        return self._scale

    @property
    def weights(self) -> np.ndarray:
        """Model weights based on IC."""
        return self._weights

    @property
    def combination(self) -> np.ndarray:
        """Combination matrix (variables × models with weights and ICs)."""
        return self._combination

    @property
    def combination_col_names(self) -> list[str]:
        """Column names for combination matrix."""
        return self._combination_col_names

    @property
    def combination_row_names(self) -> list[str]:
        """Row names for combination matrix."""
        return self._combination_row_names

    def vcov(self) -> np.ndarray:
        """Return variance-covariance matrix."""
        return self._vcov_matrix

    # --- Methods ---

    def predict(
        self,
        X: np.ndarray | DataFrame | dict | None = None,
        interval: str = "none",
        level: float | list[float] = 0.95,
        side: str = "both",
    ) -> PredictionResult:
        """Predict using the combined model.

        Parameters
        ----------
        X : array-like, dict, DataFrame, or None
            Design matrix (with intercept column), dict/DataFrame of new data,
            or None for the in-sample predictions, as R.
        interval : str, default="none"
            "none", "confidence", or "prediction".
        level : float or list of float, default=0.95
            Confidence level(s).
        side : str, default="both"
            "both", "upper", or "lower".

        Returns
        -------
        PredictionResult
        """
        if X is None:
            X = self._X_train

        # The design matrix of all the variables of the new data, with the levels of
        # the factors of the training data, and its columns of the parameters, as
        # R's predict.greybox() of y ~ . A column of the intercept, as in a design
        # matrix from formula(), is not a variable, and R's model.frame() drops it
        if isinstance(X, (dict, DataFrame)):
            frame = DataFrame(X).drop(columns="(Intercept)", errors="ignore")
            frame.attrs["_xlevels"] = self._xlevels
            X = formula_func(f"`{self.y_variable}` ~ .", frame, return_type="X")
            X = np.asarray(X[self.coefficient_names], dtype=float)
        else:
            X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != len(self.coefficients):
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but model expects {len(self.coefficients)}"
            )

        # An ALM with the combined parameters, as R's predict.alm() of a calm object
        model = ALM(distribution=self.distribution, loss="likelihood")
        model.intercept_ = self.coefficients[0]
        model._coef = self.coefficients[1:]
        model._n_features = len(self.coefficients)
        model._ari_order = 0
        model._X_train_ = self._X_train
        model._y_train_ = self._actuals
        model.fitted_values_ = self._fitted
        model.residuals_ = self._residuals
        model._scale = self._scale
        model.other_ = self.other_
        model._n_params_ = self.nparam
        model._df_residual = self.df_residual
        model.vcov = lambda *_, **__: self._vcov_matrix  # type: ignore[method-assign]
        return model.predict(
            X,
            interval=interval,  # type: ignore[arg-type]
            level=level,
            side=side,  # type: ignore[arg-type]
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
            Which parameters to include. If None, all.
        level : float, default=0.95
            Confidence level.

        Returns
        -------
        np.ndarray
            Shape (n_params, 2) with lower and upper bounds.
        """
        se = np.sqrt(np.abs(np.diag(self._vcov_matrix)))
        t_crit = stats.t.ppf((1 + level) / 2, df=self.df_residual)
        lower = self.coefficients - t_crit * se
        upper = self.coefficients + t_crit * se
        if parm is not None:
            if isinstance(parm, int):
                parm = [parm]
            lower = lower[parm]
            upper = upper[parm]
        return np.column_stack([lower, upper])

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = "likelihood",
    ) -> float:
        """Calculate model score.

        Parameters
        ----------
        X : array-like
            Design matrix.
        y : array-like
            True values.
        metric : str, default="likelihood"
            "likelihood", "MSE", "MAE", or "R2".

        Returns
        -------
        float
        """
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X).mean

        if metric == "likelihood":
            return float(self.log_lik)
        elif metric == "MSE":
            return float(np.mean((y - y_pred) ** 2))
        elif metric == "MAE":
            return float(np.mean(np.abs(y - y_pred)))
        elif metric == "R2":
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return float(1 - ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def __repr__(self) -> str:
        return f"CALMResult(IC_type={self.IC_type!r}, fitted=True)"

    def __str__(self) -> str:
        """Print method matching ADAM-style output."""
        dist_name = DISTRIBUTION_NAMES.get(self.distribution, self.distribution)

        lines = []
        lines.append(f"Time elapsed: {self.time_elapsed:.2f} seconds")
        lines.append(f"Model estimated: CALM({self.IC_type})")
        lines.append(f"Distribution assumed in the model: {dist_name}")
        lines.append("Loss function type: likelihood")
        lines.append("")

        coef = self.coefficients
        names = getattr(self, "coefficient_names", [])

        lines.append("Coefficients:")
        name_widths = [max(len(n), 12) for n in names]
        header = "  ".join(f"{n:>{w}}" for n, w in zip(names, name_widths))
        lines.append(header)
        vals = "  ".join(f"{c:>{w}.7f}" for c, w in zip(coef, name_widths))
        lines.append(vals)

        return "\n".join(lines)

    def summary(self, level: float = 0.95) -> CALMSummary:
        """Summary matching R's summary.greyboxC.

        Parameters
        ----------
        level : float, default=0.95
            Confidence level for intervals.

        Returns
        -------
        CALMSummary
        """
        coef = self.coefficients
        vcov = self._vcov_matrix
        importance = self.importance
        names = self.coefficient_names
        n_obs = self.n_obs
        df_res = self.df_residual
        residuals = self.residuals

        se = np.sqrt(np.abs(np.diag(vcov)))

        alpha = 1 - level
        t_crit = stats.t.ppf(1 - alpha / 2, df_res)
        lower_ci = coef - t_crit * se
        upper_ci = coef + t_crit * se

        sigma = float(np.sqrt(np.sum(residuals**2) / df_res))

        nparam = float(np.sum(importance)) + 1

        # Compute all 4 ICs from log_lik and nparam
        log_lik = self.log_lik
        k = nparam
        n = n_obs
        aic = -2 * log_lik + 2 * k
        bic = -2 * log_lik + k * np.log(n)
        if n - k - 1 > 0:
            aicc = aic + (2 * k**2 + 2 * k) / (n - k - 1)
            bicc = -2 * log_lik + k * np.log(n) * n / (n - k - 1)
        else:
            aicc = bicc = np.inf

        return CALMSummary(
            coefficients=coef,
            se=se,
            importance=importance,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            coefficient_names=names,
            sigma=sigma,
            n_obs=n_obs,
            nparam=nparam,
            df_residual=df_res,
            distribution=self.distribution,
            y_variable=self.y_variable,
            ic_type=self.IC_type,
            aic=aic,
            aicc=aicc,
            bic=bic,
            bicc=bicc,
        )


def _prepare_data(
    data: dict | DataFrame,
    formula_str: str | None = None,
    subset: Any | None = None,
) -> tuple[DataFrame, str, np.ndarray]:
    """Prepare data for stepwise selection.

    Parameters
    ----------
    data : dict or DataFrame
        Data containing variables.
    formula_str : str, optional
        Formula for variable selection after transformations.
    subset : array-like, optional
        Subset of observations to use.

    Returns
    -------
    tuple
        (data frame, response name, rows without missing values)
    """
    frame = DataFrame(data).reset_index(drop=True)
    frame.columns = [str(name) for name in frame.columns]

    # The variables of the formula after the transformations, as R's model.frame()
    if formula_str is not None:
        y_frame, X_frame = formula_func(
            formula_str, {k: frame[k].to_numpy() for k in frame}, as_dataframe=True
        )
        frame = pd.concat(
            [y_frame, X_frame.drop(columns="(Intercept)", errors="ignore")], axis=1
        )
    if subset is not None:
        subset = np.asarray(subset)
        rows = np.flatnonzero(subset) if subset.dtype == bool else subset
        frame = frame.iloc[rows].reset_index(drop=True)

    response_name = str(frame.columns[0])
    if not pd.api.types.is_numeric_dtype(frame[response_name]):
        warnings.warn(
            "The response variable is not numeric! We will make it numeric, but we "
            "cannot promise anything.",
            stacklevel=3,
        )
        frame[response_name] = frame[response_name].astype(float)

    rows_selected = ~frame.isna().any(axis=1).to_numpy()
    return frame, response_name, rows_selected


def _check_variability(data: DataFrame, var_names: list[str]) -> list[str]:
    """Check variability in variables and remove those without any.

    Parameters
    ----------
    data : DataFrame
        Data frame.
    var_names : list of str
        Variable names to check.

    Returns
    -------
    list of str
        Variable names that have variability.
    """
    return [var for var in var_names if data[var].nunique(dropna=False) > 1]


def _calculate_associations(
    residuals: np.ndarray,
    blocks: dict[str, DataFrame],
    numeric: dict[str, bool],
    method: str = "pearson",
) -> dict[str, float]:
    """Calculate association measures between residuals and variables.

    This is similar to R's assocFast(): the correlation with the numeric
    variables, and the square root of the coefficient of determination of the
    regression of the residuals on the categorical and the binary ones.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    blocks : dict
        The columns of the design matrix of each variable.
    numeric : dict
        Whether each variable is numeric and not binary.
    method : str
        Correlation method: "pearson", "kendall", or "spearman".

    Returns
    -------
    dict
        Dictionary mapping variable names to association values.
    """
    associations = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for var, block in blocks.items():
            values = block.to_numpy(dtype=float)
            if numeric[var]:
                if method == "spearman":
                    corr = stats.spearmanr(residuals, values[:, 0]).statistic
                elif method == "kendall":
                    corr = stats.kendalltau(residuals, values[:, 0]).statistic
                else:
                    corr = np.corrcoef(residuals, values[:, 0])[0, 1]
            else:
                design = np.column_stack([np.ones(len(residuals)), values])
                fit_residuals = _lm_fit(design, residuals)[1]
                # Constant residuals give 0/0, NaN in R
                corr = np.sqrt(
                    np.abs(
                        1
                        - np.divide(
                            mean_fast(fit_residuals**2, 1),
                            np.float64(
                                mean_fast((residuals - np.mean(residuals)) ** 2, 1)
                            ),
                        )
                    )
                )
            associations[var] = float(corr)

    return associations


def _calculate_ic(
    model: ALM,
    ic_type: str,
    df_add: int = 0,
) -> float:
    """Calculate information criterion from a model.

    Parameters
    ----------
    model : ALM
        Fitted model.
    ic_type : str
        Type of IC: "AIC", "BIC", "AICc", "BICc".
    df_add : int
        Additional degrees of freedom to add.

    Returns
    -------
    float
        The IC value.
    """
    n = model.nobs
    k = model.nparam + df_add
    log_lik = model.log_lik

    if n is None or log_lik is None:
        return np.inf

    ic_type = ic_type.upper()

    if ic_type == "AIC":
        return -2 * log_lik + 2 * k
    elif ic_type == "BIC":
        return -2 * log_lik + k * np.log(n)
    elif ic_type == "AICC":
        if n - k - 1 > 0:
            return -2 * log_lik + 2 * k + (2 * k**2 + 2 * k) / (n - k - 1)
        return np.inf
    elif ic_type == "BICC":
        if n - k - 1 > 0:
            return -2 * log_lik + (k * np.log(n) * n) / (n - k - 1)
        return np.inf
    return np.inf


def stepwise(
    data: dict | DataFrame,
    ic: Literal["AICc", "AIC", "BIC", "BICc"] = "AICc",
    silent: bool = True,
    df: int | None = None,
    formula: str | None = None,
    subset: Any | None = None,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    distribution: str = "dnorm",
    occurrence: Literal["none", "plogis", "pnorm"] | ALM = "none",
    **kwargs,
) -> ALM:
    """Stepwise selection of regressors.

    Function selects variables that give linear regression with the lowest
    information criterion. The selection is done stepwise (forward) based on
    partial correlations. This should be a simpler and faster implementation
    than step() function from stats package.

    The algorithm uses ALM to fit different models and correlation to select
    the next regressor in the sequence. The association with the categorical
    and the binary variables is measured via the square root of the coefficient
    of determination, as in R.

    Parameters
    ----------
    data : dict or DataFrame
        Data frame containing dependent variable in the first column and
        the others in the rest. The non-numeric columns are treated as
        categorical variables and expanded into dummies.
    ic : {"AICc", "AIC", "BIC", "BICc"}, default="AICc"
        Information criterion to use.
    silent : bool, default=True
        If False, then progress is printed.
    df : int, optional
        Number of degrees of freedom to add (should be used if stepwise is
        used on residuals).
    formula : str, optional
        If provided, then the selection will be done from the listed
        variables in the formula after all the necessary transformations.
    subset : array-like, optional
        An optional vector specifying a subset of observations to be used
        in the fitting process (boolean mask or integer positions).
    method : {"pearson", "kendall", "spearman"}, default="pearson"
        Method of correlations calculation. The default is Pearson's
        correlation, which should be applicable to a wide range of data
        in different scales.
    distribution : str, default="dnorm"
        Distribution to use for the ALM model. See ALM for details.
    occurrence : {"none", "plogis", "pnorm"} or ALM, default="none"
        What distribution to use for occurrence part, or a fitted occurrence
        model. The selection is done on the non-zero observations, and the
        final model is estimated with the occurrence model, as in R.

    Returns
    -------
    ALM
        The final fitted model with additional attributes:

        * ic_values: dict mapping step names to IC values
          (e.g. {"Intercept": 150.3, "x1": 140.2, "x2": 138.1}).
          Keys are in insertion order (Python 3.7+).
        * time_elapsed: float, seconds taken for calculation

    Examples
    --------
    >>> data = {'y': [1, 2, 3, 4, 5], 'x1': [1, 2, 3, 4, 5],
    ...         'x2': [2, 4, 6, 8, 10]}
    >>> model = stepwise(data)
    """
    start_time = time.time()

    if df is None:
        df = 0

    frame, response_name, rows_complete = _prepare_data(data, formula, subset)
    x_vars = [k for k in frame.columns if k != response_name]

    if len(x_vars) == 0:
        raise ValueError("Need at least one predictor variable")

    if distribution in ("plogis", "pnorm"):
        frame[response_name] = (frame[response_name] != 0).astype(float)

    variables = _check_variability(frame, x_vars)
    if len(variables) == 0:
        raise ValueError(
            "None of exogenous variables has variability. There's nothing to select!"
        )
    if len(variables) < len(x_vars):
        warnings.warn(
            "Some exogenous variables did not have any variability. "
            "We dropped them out.",
            stacklevel=2,
        )
    x_vars = variables

    # The observations without missing values, and the non-zero ones are used in the
    # selection with an occurrence model
    frame = frame.loc[rows_complete].reset_index(drop=True)
    y = frame[response_name].to_numpy(dtype=float)
    occurrence_model = isinstance(occurrence, ALM) or occurrence in ("plogis", "pnorm")
    rows_selected = y != 0 if occurrence_model else np.ones(len(y), dtype=bool)
    n_obs = int(np.sum(rows_selected))

    # The columns of each variable in the design matrix (dummies for the categorical
    # ones) and whether it is numeric and not binary, as in R
    blocks: dict[str, DataFrame] = {}
    numeric: dict[str, bool] = {}
    xlevels: dict[str, list] = {}
    for var in x_vars:
        column = frame[var]
        if pd.api.types.is_numeric_dtype(column) and column.dtype.kind != "b":
            blocks[var] = column.astype(float).to_frame()
            subset_column = column[rows_selected]
            numeric[var] = bool(((subset_column != 0) & (subset_column != 1)).any())
        else:
            design = formula_func(f"~ `{var}`", frame, return_type="X")
            blocks[var] = design.iloc[:, 1:]
            xlevels |= design.attrs["_xlevels"]
            numeric[var] = False
    blocks_selected = {var: block[rows_selected] for var, block in blocks.items()}

    selected_vars: list[str] = []
    all_ics: dict[str, float] = {}

    formula_str = f"{response_name} ~ 1"
    model = ALM(distribution=distribution, loss="likelihood", **kwargs)
    model.fit(np.ones((n_obs, 1)), y[rows_selected], formula=formula_str)

    current_ic = _calculate_ic(model, ic, df)
    best_ic = current_ic
    all_ics["Intercept"] = current_ic

    residuals = np.ravel(model.residuals)
    best_formula = formula_str

    if not silent:
        print(f"Formula: {formula_str}, IC: {current_ic:.4f}\n")

    m = 2
    while True:
        associations = _calculate_associations(
            residuals, blocks_selected, numeric, method
        )

        # The first variable with the highest absolute association among those not
        # in the model, as R
        absolute = np.abs([associations[var] for var in x_vars])
        candidates = [
            value
            for var, value in zip(x_vars, absolute)
            if var not in selected_vars and not np.isnan(value)
        ]
        if not candidates:
            break
        new_element = x_vars[int(np.flatnonzero(absolute == max(candidates))[0])]
        if new_element in selected_vars:
            break

        test_vars = [*selected_vars, new_element]
        test_formula = best_formula + f" + {new_element}"
        X_fit = pd.concat([blocks_selected[var] for var in test_vars], axis=1)
        X_fit.insert(0, "(Intercept)", 1.0)

        model = ALM(distribution=distribution, loss="likelihood", **kwargs)
        model.fit(
            X_fit.to_numpy(),
            y[rows_selected],
            formula=test_formula,
            feature_names=list(X_fit.columns[1:]),
        )

        if model.nparam + df >= n_obs + 1:
            if not silent:
                warnings.warn(
                    "Number of degrees of freedom is greater than number of "
                    "observations. Cannot proceed.",
                    stacklevel=2,
                )
            break

        current_ic = _calculate_ic(model, ic, df)

        if not silent:
            print(f"Step {m - 1}. Formula: {test_formula}, IC: {current_ic:.4f}")
            print("Correlations: ")
            for var in x_vars:
                print(f"  {var}: {associations[var]:.3f}")
            print()

        all_ics[new_element] = current_ic
        m += 1
        if current_ic >= best_ic:
            break
        best_ic = current_ic
        best_formula = test_formula
        selected_vars.append(new_element)
        residuals = np.ravel(model.residuals)

    # The final model on all the observations, with the occurrence model estimated
    # on the same variables, as R's alm()
    X_final = pd.concat(
        [DataFrame({"(Intercept)": np.ones(len(y))})]
        + [blocks[var] for var in selected_vars],
        axis=1,
    )
    feature_names_final = list(X_final.columns[1:])
    if occurrence in ("plogis", "pnorm"):
        occurrence = ALM(
            distribution=occurrence,
            loss="likelihood",
            orders=kwargs.get("orders", (0, 0, 0)),
        ).fit(
            X_final.to_numpy(),
            (y != 0).astype(float),
            formula=best_formula,
            feature_names=feature_names_final,
        )

    final_model = ALM(
        distribution=distribution, loss="likelihood", occurrence=occurrence, **kwargs
    )
    final_model.fit(
        X_final.to_numpy(),
        pd.Series(y, name=response_name),
        formula=best_formula,
        feature_names=feature_names_final,
    )

    final_model._xlevels = xlevels
    final_model.ic_values = all_ics
    final_model.time_elapsed = time.time() - start_time

    return final_model


def _get_ic_value(model: ALM, ic_type: str) -> float:
    """Get the IC value from a model."""
    if ic_type == "AIC":
        return model.aic if model.aic is not None else np.inf
    elif ic_type == "BIC":
        return model.bic if model.bic is not None else np.inf
    elif ic_type == "AICc":
        val = model.aicc
        if val is not None and not np.isnan(val):
            return val
        return np.inf
    elif ic_type == "BICc":
        val = model.bicc
        if val is not None and not np.isnan(val):
            return val
        return np.inf
    return np.inf


def _combine_mu(distribution: str, X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Compute mu from design matrix and combined coefficients.

    Applies distribution-specific link function matching R's lmCombine.
    """
    linear_pred = X @ coef
    if distribution in ("dinvgauss", "dgamma", "dexp", "dpois", "dnbinom"):
        return np.exp(linear_pred)
    elif distribution == "dchisq":
        return linear_pred**2
    else:
        return linear_pred


def _combine_scale(
    distribution: str,
    y: np.ndarray,
    mu: np.ndarray,
    other_combined: float = 0.0,
    alpha: float = 0.5,
    n_params: int = 1,
) -> float | np.ndarray:
    """Compute scale parameter for the combined model.

    Matches R lmCombine's switch(distribution, ...) block for scale.
    """
    # The scale of the Normal-based distributions is the variance, as in the
    # ADAM monograph; R's calm() divides by n
    if distribution in ("dnorm", "dfnorm", "drectnorm"):
        return float(np.mean((y - mu) ** 2))
    elif distribution == "dlaplace":
        return float(np.mean(np.abs(y - mu)))
    elif distribution == "ds":
        return float(np.mean(np.sqrt(np.abs(y - mu))) / 2)
    elif distribution == "dgnorm":
        beta = other_combined
        if beta > 0:
            return float((beta * np.mean(np.abs(y - mu) ** beta)) ** (1.0 / beta))
        return float(np.sqrt(np.mean((y - mu) ** 2)))
    elif distribution == "dlgnorm":
        beta = other_combined
        if beta > 0:
            return float(
                (beta * np.mean(np.abs(np.log(y) - mu) ** beta)) ** (1.0 / beta)
            )
        return float(np.sqrt(np.mean((np.log(y) - mu) ** 2)))
    elif distribution == "dlogis":
        return float(np.sqrt(np.mean((y - mu) ** 2) * 3 / np.pi**2))
    elif distribution == "dt":
        var_res = float(np.mean((y - mu) ** 2))
        if var_res > 1:
            return max(2.0, 2.0 / (1.0 - 1.0 / var_res))
        return 2.0
    elif distribution == "dalaplace":
        return float(np.mean((y - mu) * (alpha - (y <= mu) * 1.0)))
    elif distribution == "dlnorm":
        return float(np.mean((np.log(y) - mu) ** 2))
    elif distribution == "dllaplace":
        return float(np.mean(np.abs(np.log(y) - mu)))
    elif distribution == "dls":
        return float(np.mean(np.sqrt(np.abs(np.log(y) - mu))) / 2)
    elif distribution in ("dchisq", "dnbinom"):
        return float(other_combined)
    elif distribution == "dinvgauss":
        return float(np.mean((y / mu - 1) ** 2 / (y / mu)))
    elif distribution == "dgamma":
        return float(np.mean((y / mu - 1) ** 2))
    elif distribution == "dexp":
        return 1.0
    elif distribution == "dpois":
        return mu
    elif distribution == "dbcnorm":
        lambda_bc = other_combined
        if lambda_bc == 0:
            y_bc = np.log(y)
        else:
            y_bc = (y**lambda_bc - 1) / lambda_bc
        return float(np.mean((y_bc - mu) ** 2))
    elif distribution == "dlogitnorm":
        return float(np.mean((np.log(y / (1 - y)) - mu) ** 2))
    elif distribution == "pnorm":
        return float(
            np.sqrt(np.mean(stats.norm.ppf((y - stats.norm.cdf(mu) + 1) / 2) ** 2))
        )
    elif distribution == "plogis":
        return float(
            np.sqrt(
                np.mean(
                    np.log((1 + y * (1 + np.exp(mu))) / (1 + np.exp(mu) * (2 - y) - y))
                    ** 2
                )
            )
        )
    else:
        return float(np.sqrt(np.mean((y - mu) ** 2)))


def _combine_errors(distribution: str, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Compute residuals/errors for the combined model.

    Matches R lmCombine's switch(distribution, ...) block for errors.
    """
    if distribution in ("dinvgauss", "dgamma", "dexp"):
        return y / mu
    elif distribution == "dchisq":
        return np.sqrt(y) - np.sqrt(mu)
    elif distribution in ("dlnorm", "dllaplace", "dls", "dlgnorm"):
        return np.log(y) - mu
    elif distribution == "dlogitnorm":
        return np.log(y / (1 - y)) - mu
    elif distribution == "pnorm":
        return stats.norm.ppf((y - stats.norm.cdf(mu) + 1) / 2)
    elif distribution == "plogis":
        return np.log((1 + y * (1 + np.exp(mu))) / (1 + np.exp(mu) * (2 - y) - y))
    else:
        return y - mu


def _combine_fitted(
    distribution: str,
    mu: np.ndarray,
    scale: float | np.ndarray,
    other: float = 0.0,
) -> np.ndarray:
    """Compute fitted values for the combined model.

    Matches R lmCombine's switch(distribution, ...) block for yFitted.
    """
    if distribution == "dfnorm":
        # The scale of dfnorm is the variance
        sd = np.sqrt(scale)
        return np.sqrt(2 / np.pi) * sd * np.exp(-(mu**2) / (2 * scale)) + mu * (
            1 - 2 * stats.norm.cdf(-mu / sd)
        )
    elif distribution == "drectnorm":
        sd = np.sqrt(scale)
        return mu * (1 - stats.norm.cdf(0, mu, sd)) + sd * stats.norm.pdf(0, mu, sd)
    elif distribution == "dlogitnorm":
        return np.exp(mu) / (1 + np.exp(mu))
    elif distribution == "dbcnorm":
        return bc_transform_inv(mu, other)
    elif distribution == "dchisq":
        return mu + other
    elif distribution in ("dlnorm", "dllaplace", "dls", "dlgnorm"):
        return np.exp(mu)
    elif distribution == "pnorm":
        return stats.norm.cdf(mu)
    elif distribution == "plogis":
        return 1.0 / (1.0 + np.exp(-mu))
    else:
        return mu.copy()


def CALM(
    data: dict | DataFrame,
    ic: Literal["AICc", "AIC", "BIC", "BICc"] = "AICc",
    bruteforce: bool = False,
    silent: bool = True,
    distribution: str = "dnorm",
    **kwargs,
) -> CALMResult | ALM:
    """Combine ALM models based on information criteria.

    Function combines parameters of linear regressions of the first variable
    on all the other provided data. The algorithm uses ALM to fit different
    models and then combines the models based on the selected IC.

    Parameters not present in some models are assumed to be zero, creating
    a shrinkage effect in the combination.

    Parameters
    ----------
    data : dict or DataFrame
        Data frame containing dependent variable in the first column and
        the others in the rest. The observations with missing values are
        dropped, and the non-numeric columns are expanded into dummies.
    ic : {"AICc", "AIC", "BIC", "BICc"}, default="AICc"
        Information criterion to use.
    bruteforce : bool, default=False
        If True, all possible models are generated and combined.
        Otherwise the best model is found via stepwise, then if <14
        parameters recurses with bruteforce on selected vars, else
        does stress-testing (best +/- single vars).
    silent : bool, default=True
        If False, then progress is printed.
    distribution : str, default="dnorm"
        Distribution to use for the ALM model. With "dnorm", the models are
        estimated via least squares, as R's lmGreybox.
    **kwargs
        Additional arguments passed to ALM().

    Returns
    -------
    CALMResult or ALM
        The fitted model of stepwise when it has the intercept only, as R.
        Otherwise a CALMResult containing:

        - coefficients: combined parameters
        - vcov: combined variance-covariance matrix
        - fitted: fitted values (on original scale)
        - residuals: residuals (distribution-specific)
        - mu: location parameter
        - scale: scale parameter
        - ``other_``: the combined extra parameter of the distribution (shape,
          alpha, lambda of Box-Cox, nu or size), None for the others
        - distribution: distribution used
        - log_lik: combined log-likelihood
        - IC: combined information criterion value
        - IC_type: type of IC used
        - df_residual: residual degrees of freedom
        - df: model degrees of freedom
        - importance: importance of each parameter
        - combination: matrix of model combinations with weights/ICs
        - coefficient_names: names of coefficients
        - time_elapsed: computation time in seconds

    Examples
    --------
    >>> data = {'y': [1, 2, 3, 4, 5], 'x1': [1, 2, 3, 4, 5],
    ...         'x2': [2, 4, 6, 8, 10]}
    >>> result = CALM(data)
    """
    start_time = time.time()

    frame, y_var, rows_selected = _prepare_data(data)
    if frame.shape[1] < 2:
        raise ValueError("Need at least one predictor variable")
    if distribution in ("plogis", "pnorm"):
        frame[y_var] = (frame[y_var] != 0).astype(float).where(frame[y_var].notna())

    # R compares the number of columns of the data with the number of its rows
    if frame.shape[1] > frame.shape[0] and bruteforce:
        warnings.warn(
            "You have more variables than observations. We have to be smart here. "
            "Switching to 'bruteforce=FALSE'.",
            stacklevel=2,
        )
        bruteforce = False

    if frame.shape[1] > 14 and bruteforce:
        warnings.warn(
            "You have more than 14 variables. The computation might take a lot of "
            "time.",
            stacklevel=2,
        )

    if not bruteforce:
        if not silent:
            print("Selecting the best model...")
        best_model = stepwise(frame, ic=ic, distribution=distribution, **kwargs)
        # If the selected model does not contain variables, return it
        if len(best_model.coefficients) == 1:
            return best_model

    # The models of dalaplace are estimated with alpha=0.5 by default, as R
    if distribution == "dalaplace":
        kwargs.setdefault("alpha", 0.5)

    # The design matrix without the missing values, with the dummies of the
    # categorical variables, as R's model.matrix()
    frame = frame.loc[rows_selected].reset_index(drop=True)
    y = frame[y_var].to_numpy(dtype=float)
    exogenous = formula_func(f"`{y_var}` ~ .", frame, return_type="X")
    xlevels = exogenous.attrs["_xlevels"]
    exogenous = exogenous.drop(columns="(Intercept)")
    x_vars = [str(name) for name in exogenous.columns]
    n_vars = len(x_vars)
    n_obs = len(y)
    X_matrix = np.column_stack([np.ones(n_obs), exogenous.to_numpy()])

    if bruteforce:
        # All the combinations, counting in binary, as R
        n_combinations = 2**n_vars
        var_binary = (
            np.arange(n_combinations)[:, np.newaxis] >> np.arange(n_vars) & 1
        ).astype(int)
    else:
        selected_names = list(best_model._feature_names or [])

        # If <14 params, recurse with bruteforce on selected vars
        if best_model.nparam < 14:
            result = CALM(
                DataFrame(
                    {y_var: y, **{name: exogenous[name] for name in selected_names}}
                ),
                ic=ic,
                bruteforce=True,
                silent=silent,
                distribution=distribution,
                **kwargs,
            )
            result._xlevels = xlevels
            return result

        # Stress-testing mode: best model +/- each single variable
        best_vars = [name for name in selected_names if name in x_vars]
        excluded_vars = [v for v in x_vars if v not in best_vars]

        # Build binary combination matrix
        n_in_model = len(best_vars)
        n_combinations = 1 + len(excluded_vars) + n_in_model
        var_binary = np.zeros((n_combinations, n_vars), dtype=int)

        # Row 0: best model, copied to all rows
        for var in best_vars:
            var_binary[:, x_vars.index(var)] = 1

        # Rows 1..len(excluded_vars): add one excluded var
        for i, var in enumerate(excluded_vars):
            var_binary[1 + i, x_vars.index(var)] = 1

        # Rows after that: remove one included var
        for i in range(n_in_model):
            row = 1 + len(excluded_vars) + i
            included_indices = np.where(var_binary[row] == 1)[0]
            var_binary[row, included_indices[i]] = 0

    n_params = n_vars + 1  # intercept + variables

    # Storage arrays matching R's parameters, vcovValues, ICs, logLiks
    ics_array = np.full(n_combinations, np.nan)
    parameters = np.zeros((n_combinations, n_params))
    vcov_values = np.zeros((n_combinations, n_params, n_params))
    log_liks = np.full(n_combinations, np.nan)

    # Track "other" parameters for distributions that need them
    dists_with_other = (
        "dt",
        "dchisq",
        "dnbinom",
        "dalaplace",
        "dgnorm",
        "dlgnorm",
        "dbcnorm",
    )
    has_other = distribution in dists_with_other
    other_parameters = np.full(n_combinations, np.nan) if has_other else None

    if not silent:
        print(
            f"Estimation progress: {round(1 / n_combinations * 100)}%",
            end="",
        )

    for i in range(n_combinations):
        if not silent and i > 0:
            prev = f"{round(i / n_combinations * 100)}%"
            print("\b" * len(prev), end="")
            print(
                f"{round((i + 1) / n_combinations * 100)}%",
                end="",
            )

        model_indices = [0] + [j + 1 for j in range(n_vars) if var_binary[i, j]]
        X_fit = X_matrix[:, model_indices]

        if distribution == "dnorm":
            # The least squares of R's lmGreybox: logLik with the scale, the
            # corrected ICs and the variance of the errors without it
            k = len(model_indices)
            coef, fit_residuals, _ = _lm_fit(X_fit, y)
            sse = mean_fast(fit_residuals**2, 1)
            ll = -n_obs / 2 * (np.log(2 * np.pi) + 1 + np.log(sse / n_obs))
            denominator = np.float64(n_obs - k - 1)
            with np.errstate(divide="ignore"):
                ics_array[i] = {
                    "AIC": -2 * ll + 2 * (k + 1),
                    "BIC": -2 * ll + (k + 1) * np.log(n_obs),
                    "AICc": 2 * k - 2 * ll + 2 * k * (k + 1) / denominator,
                    "BICc": -2 * ll + k * np.log(n_obs) * n_obs / denominator,
                }[ic]
            log_liks[i] = ll
            model_vcov = sse / (n_obs - k) * np.linalg.inv(X_fit.T @ X_fit)
        else:
            model = ALM(distribution=distribution, loss="likelihood", **kwargs)
            model.fit(
                X_fit,
                y,
                feature_names=[x_vars[j - 1] for j in model_indices[1:]],
            )
            ics_array[i] = _get_ic_value(model, ic)
            log_liks[i] = model.log_lik
            coef = model.coefficients
            model_vcov = model.vcov()
            if has_other and model.other_ is not None:
                other_parameters[i] = model.other_  # type: ignore[index]

        parameters[i, model_indices] = coef
        vcov_values[i][np.ix_(model_indices, model_indices)] = model_vcov

    if not silent:
        print(" Done!")

    # Calculate IC weights
    models_good = np.isfinite(ics_array)
    if not np.any(models_good):
        raise ValueError("No models could be fitted")

    ic_weights = np.zeros(n_combinations)
    min_ic = np.min(ics_array[models_good])
    delta = ics_array[models_good] - min_ic
    ic_weights[models_good] = np.exp(-0.5 * delta)
    ic_weights[models_good] /= np.sum(ic_weights[models_good])

    # Combined parameters (weighted sum)
    combined_coef = np.sum(parameters * ic_weights[:, np.newaxis], axis=0)

    coefficient_names = ["(Intercept)"] + list(x_vars)

    # Compute mu (distribution-specific link)
    mu = _combine_mu(distribution, X_matrix, combined_coef)

    # Combined "other" parameter (weighted average)
    other_combined = 0.0
    if has_other and other_parameters is not None:
        valid = np.isfinite(other_parameters)
        if np.any(valid):
            other_combined = float(np.sum(ic_weights[valid] * other_parameters[valid]))

    # Scale (distribution-specific)
    alpha_val = kwargs.get("alpha", 0.5)
    scale = _combine_scale(distribution, y, mu, other_combined, alpha_val, n_params)

    # Fitted values (distribution-specific)
    if distribution == "dchisq" or distribution == "dbcnorm":
        other_for_fitted = other_combined
    else:
        other_for_fitted = 0.0
    fitted_values = _combine_fitted(distribution, mu, scale, other_for_fitted)

    # Residuals/errors (distribution-specific), of the Box-Cox transform for dbcnorm
    errors = _combine_errors(
        distribution,
        bc_transform(y, other_combined) if distribution == "dbcnorm" else y,
        mu,
    )

    # Importance: sum of IC weights for models containing each variable
    importance = np.zeros(n_params)
    importance[0] = 1.0  # intercept always included
    importance[1:] = ic_weights @ var_binary

    # Degrees of freedom (matching R: n - sum(importance) - 1)
    df_model = float(np.sum(importance)) + 1  # +1 for scale
    df_residual = float(n_obs - np.sum(importance) - 1)

    # Combined log-likelihood (weighted average)
    combined_log_lik = float(np.nansum(ic_weights * log_liks))

    # Combined IC (weighted average, not minimum)
    combined_ic = float(np.nansum(ic_weights * ics_array))

    # Combined vcov matrix
    vcov_combined = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(i, n_params):
            vcov_combined[i, j] = float(
                np.sum(
                    ic_weights**2
                    * (
                        vcov_values[:, i, j]
                        + (parameters[:, i] - combined_coef[i])
                        * (parameters[:, j] - combined_coef[j])
                    )
                )
            )
            vcov_combined[j, i] = vcov_combined[i, j]

    # Build combination table (variables × models with weights and ICs)
    combination = np.column_stack([var_binary, ic_weights, ics_array])
    combo_col_names = list(x_vars) + ["IC weights", ic]
    combo_row_names = [f"Model{i + 1}" for i in range(n_combinations)]

    elapsed_time = time.time() - start_time

    result = CALMResult(
        coefficients=combined_coef,
        coefficient_names=coefficient_names,
        vcov=vcov_combined,
        fitted=np.asarray(fitted_values).flatten(),
        residuals=np.asarray(errors).flatten(),
        mu=np.asarray(mu).flatten(),
        scale=scale,
        distribution=distribution,
        log_lik=combined_log_lik,
        IC=combined_ic,
        IC_type=ic,
        df_residual=df_residual,
        df=df_model,
        importance=importance,
        combination=combination,
        combination_col_names=combo_col_names,
        combination_row_names=combo_row_names,
        y_variable=y_var,
        x_variables=x_vars,
        weights=ic_weights,
        other_=other_combined if has_other else None,
        n_obs=n_obs,
        time_elapsed=elapsed_time,
        actuals=y,
        X_train=X_matrix,
        formula_str=f"{y_var} ~ " + " + ".join(x_vars),
    )
    result._xlevels = xlevels
    return result
