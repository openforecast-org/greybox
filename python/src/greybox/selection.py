"""Model selection and combination functions.

This module provides functions for stepwise regression, model combination,
and forecasting accuracy measures.
"""

import itertools
import time
from typing import Any, ClassVar, Literal, TypedDict

import numpy as np
from pandas import DataFrame
from scipy import stats

from .alm import ALM, PredictionResult
from .formula import formula as formula_func
from .methods.summary import DISTRIBUTION_NAMES
from .transforms import bc_transform_inv


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

        level_low = (1 - 0.95) / 2 * 100
        level_high = (1 + 0.95) / 2 * 100
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
        n = self._n_obs
        if n is None or n == 0:
            return 0.0
        return self.bic + self.nparam * np.log(n) ** 2 / n

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
        return self._actuals

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
            or None to return training fitted values.
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
            return PredictionResult(
                mean=self.fitted,
                level=level,
                side=side,
                interval=interval,
            )

        # Handle dict or DataFrame input - convert to design matrix using stored formula
        if isinstance(X, (dict, DataFrame)):
            # Get formula - use stored formula or build from coefficient names
            formula_str = self._formula
            if formula_str is None:
                # Build formula from coefficient names if not stored
                if hasattr(self, "coefficient_names") and self.coefficient_names:
                    formula_str = "y ~ " + " + ".join(
                        name for name in self.coefficient_names if name != "(Intercept)"
                    )
                else:
                    raise ValueError(
                        "Cannot convert dict/DataFrame to design "
                        "matrix: no formula stored in model. "
                        "Please pass a design matrix (np.ndarray)"
                        " with intercept as first column."
                    )

            # Convert dict to the format expected by formula
            data_dict = X if isinstance(X, dict) else X.to_dict(orient="list")
            # Add a dummy y variable for formula parsing (will be ignored)
            data_dict = {k: np.asarray(v) for k, v in data_dict.items()}
            _, X_design = formula_func(formula_str, data_dict, as_dataframe=True)
            X = np.asarray(X_design, dtype=float)
        else:
            X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != len(self.coefficients):
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but model expects {len(self.coefficients)}"
            )

        mu = _combine_mu(self.distribution, X, self.coefficients)
        mean = _combine_fitted(self.distribution, mu, self.scale)

        variances = None
        lower: np.ndarray | None = None
        upper: np.ndarray | None = None
        if interval != "none":
            vcov_matrix = self._vcov_matrix
            variances = np.diag(X @ vcov_matrix @ X.T)
            if interval == "prediction":
                variances = variances + self.sigma**2

            if not isinstance(level, list):
                level = [level]
            n_levels = len(level)
            n_obs = len(mean)
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

            q_low = stats.t.ppf(level_low, df=self.df_residual)
            q_up = stats.t.ppf(level_up, df=self.df_residual)

            lower = np.zeros((n_obs, n_levels))
            upper = np.zeros((n_obs, n_levels))
            for i in range(n_levels):
                lower[:, i] = mean + q_low[i] * se
                upper[:, i] = mean + q_up[i] * se

            if n_levels == 1:
                lower = lower.ravel()
                upper = upper.ravel()

            if side == "upper":
                lower = None
            elif side == "lower":
                upper = None

        return PredictionResult(
            mean=mean,
            lower=lower,
            upper=upper,
            level=level,
            variances=variances,
            side=side,
            interval=interval,
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
        else:
            aicc = np.inf
        bicc = bic + k * np.log(n) ** 2 / n

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


def _convert_to_dict(data: dict | DataFrame) -> dict:
    """Convert DataFrame to dict if needed."""
    if isinstance(data, DataFrame):
        return data.to_dict(orient="list")
    return data


def _prepare_data(
    data: dict | DataFrame,
    formula_str: str | None = None,
    subset: Any | None = None,
) -> tuple[dict, str, np.ndarray]:
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
        (data_dict, response_name, rows_selected)
    """
    data_dict = _convert_to_dict(data)

    if formula_str is not None or subset is not None:
        raise NotImplementedError(
            "formula and subset parameters are not yet implemented"
        )

    all_vars = list(data_dict.keys())
    response_name = all_vars[0]

    n_obs = len(data_dict[response_name])
    if subset is not None:
        rows_selected = np.array(subset, dtype=bool)
        if len(rows_selected) != n_obs:
            raise ValueError("subset length does not match data")
    else:
        rows_selected = np.ones(n_obs, dtype=bool)

    if any(isinstance(v, list) for v in data_dict.values()):
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key], dtype=float)

    return data_dict, response_name, rows_selected


def _check_variability(data_dict: dict, var_names: list[str]) -> list[str]:
    """Check variability in variables and remove those without any.

    Parameters
    ----------
    data_dict : dict
        Data dictionary.
    var_names : list of str
        Variable names to check.

    Returns
    -------
    list of str
        Variable names that have variability.
    """
    vars_with_variability = []
    for var in var_names:
        values = np.array(data_dict[var])
        if len(np.unique(values)) > 1:
            vars_with_variability.append(var)
    return vars_with_variability


def _calculate_associations(
    residuals: np.ndarray,
    data_dict: dict,
    var_names: list[str],
    rows_selected: np.ndarray,
    method: str = "pearson",
) -> dict[str, float]:
    """Calculate association measures between residuals and variables.

    This is similar to R's assocFast() function.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    data_dict : dict
        Data dictionary.
    var_names : list of str
        Variable names to calculate associations for.
    rows_selected : np.ndarray
        Boolean array indicating which rows to use.
    method : str
        Correlation method: "pearson", "kendall", or "spearman".

    Returns
    -------
    dict
        Dictionary mapping variable names to association values.
    """
    associations = {}

    for var in var_names:
        values = np.array(data_dict[var])[rows_selected]
        resid = residuals[rows_selected]

        try:
            if method == "pearson":
                corr, _ = stats.pearsonr(resid, values)
            elif method == "spearman":
                corr, _ = stats.spearmanr(resid, values)
            elif method == "kendall":
                corr, _ = stats.kendalltau(resid, values)
            else:
                corr = 0.0

            associations[var] = abs(corr) if not np.isnan(corr) else 0.0
        except Exception:  # noqa: BLE001 - unstable correlation falls back to 0.0
            associations[var] = 0.0

    return associations


def _get_ic_function(ic: str):
    """Get the IC function for calculation."""
    ic = ic.upper()
    if ic == "AIC":
        return lambda log_lik, df: -2 * log_lik + 2 * df
    elif ic == "BIC":
        return lambda log_lik, df: (
            -2 * log_lik
            + df * np.log(len(log_lik) if hasattr(log_lik, "__len__") else 1)
        )
    elif ic == "AICCC":
        n = 1
        return lambda log_lik, df: (
            -2 * log_lik + 2 * df + (2 * df * (df + 1)) / (n - df - 1)
            if n - df - 1 > 0
            else np.inf
        )
    else:
        return lambda log_lik, df: -2 * log_lik + 2 * df


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
    occurrence: Literal["none", "plogis", "pnorm"] = "none",
    **kwargs,
) -> ALM:
    """Stepwise selection of regressors.

    Function selects variables that give linear regression with the lowest
    information criterion. The selection is done stepwise (forward) based on
    partial correlations. This should be a simpler and faster implementation
    than step() function from stats package.

    The algorithm uses ALM to fit different models and correlation to select
    the next regressor in the sequence.

    Parameters
    ----------
    data : dict or DataFrame
        Data frame containing dependent variable in the first column and
        the others in the rest.
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
        in the fitting process.
    method : {"pearson", "kendall", "spearman"}, default="pearson"
        Method of correlations calculation. The default is Pearson's
        correlation, which should be applicable to a wide range of data
        in different scales.
    distribution : str, default="dnorm"
        Distribution to use for the ALM model. See ALM for details.
    occurrence : {"none", "plogis", "pnorm"}, default="none"
        What distribution to use for occurrence part. See ALM for details.

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

    data_dict, response_name, rows_selected = _prepare_data(data, formula, subset)

    data_dict = {k: np.array(v) for k, v in data_dict.items()}

    x_vars = [k for k in data_dict if k != response_name]

    if len(x_vars) == 0:
        raise ValueError("Need at least one predictor variable")

    x_vars = _check_variability(data_dict, x_vars)

    if len(x_vars) == 0:
        raise ValueError(
            "None of exogenous variables has variability. There's nothing to select!"
        )

    selected_vars: list[str] = []
    all_ics: dict[str, float] = {}

    formula_str = f"{response_name} ~ 1"
    y_fit, X_fit = formula_func(formula_str, data_dict, as_dataframe=True)
    feature_names = [c for c in X_fit.columns if c != "(Intercept)"]

    model = ALM(distribution=distribution, loss="likelihood", **kwargs)
    model.fit(X_fit, y_fit, formula=formula_str, feature_names=feature_names)

    current_ic = _calculate_ic(model, ic, df)
    best_ic = current_ic
    all_ics["Intercept"] = current_ic

    residuals = np.array(model.residuals)
    if len(residuals.shape) > 1:
        residuals = residuals.flatten()

    best_formula = formula_str

    if not silent:
        print(f"Formula: {formula_str}, IC: {current_ic:.4f}\n")

    m = 2
    best_ic_not_found = True

    while best_ic_not_found:
        associations = _calculate_associations(
            residuals, data_dict, x_vars, rows_selected, method
        )

        available_vars = [v for v in x_vars if v not in selected_vars]

        if not available_vars:
            best_ic_not_found = False
            break

        max_assoc: float = -1.0
        new_element: str | None = None
        for var in available_vars:
            if var in associations and associations[var] > max_assoc:
                max_assoc = associations[var]
                new_element = var

        if new_element is None:
            best_ic_not_found = False
            break

        test_formula = best_formula + f" + {new_element}"

        try:
            y_fit, X_fit = formula_func(test_formula, data_dict, as_dataframe=True)
            feature_names = [c for c in X_fit.columns if c != "(Intercept)"]

            model = ALM(distribution=distribution, loss="likelihood", **kwargs)
            model.fit(X_fit, y_fit, formula=test_formula, feature_names=feature_names)

            current_ic = _calculate_ic(model, ic, df)

            if not silent:
                print(f"Step {m - 1}. Formula: {test_formula}, IC: {current_ic:.4f}")
                print("Correlations: ")
                for var in available_vars:
                    print(f"  {var}: {associations.get(var, 0):.3f}")
                print()

            if current_ic >= best_ic:
                best_ic_not_found = False
            else:
                best_ic = current_ic
                best_formula = test_formula
                selected_vars.append(new_element)
                residuals = np.array(model.residuals)
                if len(residuals.shape) > 1:
                    residuals = residuals.flatten()

            all_ics[new_element] = current_ic
            m += 1

        except Exception as e:  # noqa: BLE001 - stop stepwise if a candidate fails
            if not silent:
                print(f"Error fitting model with {new_element}: {e}")
            best_ic_not_found = False

    y_final, X_final = formula_func(best_formula, data_dict, as_dataframe=True)
    feature_names_final = [c for c in X_final.columns if c != "(Intercept)"]

    final_model = ALM(distribution=distribution, loss="likelihood", **kwargs)
    final_model.fit(
        X_final, y_final, formula=best_formula, feature_names=feature_names_final
    )

    elapsed_time = time.time() - start_time

    final_model.ic_values = all_ics
    final_model.time_elapsed = elapsed_time

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
    if distribution in ("dinvgauss", "dgamma", "dpois", "dnbinom"):
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
    n = len(y)
    df = n - n_params  # residual degrees of freedom

    if distribution in ("dnorm", "dfnorm"):
        if df > 0:
            return float(np.sqrt(np.sum((y - mu) ** 2) / df))
        else:
            return float(np.sqrt(np.mean((y - mu) ** 2)))
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
        return float(np.sqrt(np.mean((np.log(y) - mu) ** 2)))
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
    elif distribution == "dpois":
        return mu
    elif distribution == "dbcnorm":
        lambda_bc = other_combined
        if lambda_bc == 0:
            y_bc = np.log(y)
        else:
            y_bc = (y**lambda_bc - 1) / lambda_bc
        return float(np.sqrt(np.mean((y_bc - mu) ** 2)))
    elif distribution == "dlogitnorm":
        return float(np.sqrt(np.mean((np.log(y / (1 - y)) - mu) ** 2)))
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
    if distribution in ("dinvgauss", "dgamma"):
        return y / mu
    elif distribution == "dchisq":
        return np.sqrt(y) - np.sqrt(mu)
    elif distribution in ("dlnorm", "dllaplace", "dls", "dlgnorm"):
        return np.log(y) - mu
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
        return np.sqrt(2 / np.pi) * scale * np.exp(-(mu**2) / (2 * scale**2)) + mu * (
            1 - 2 * stats.norm.cdf(-mu / scale)
        )
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
) -> CALMResult:
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
        the others in the rest.
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
        Distribution to use for the ALM model.
    **kwargs
        Additional arguments passed to ALM().

    Returns
    -------
    dict
        Dictionary containing:
        - coefficients: combined parameters
        - vcov: combined variance-covariance matrix
        - fitted: fitted values (on original scale)
        - residuals: residuals (distribution-specific)
        - mu: location parameter
        - scale: scale parameter
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

    data_dict = _convert_to_dict(data)
    data_dict = {k: np.array(v, dtype=float) for k, v in data_dict.items()}

    y_var = next(iter(data_dict))
    x_vars = list(data_dict.keys())[1:]

    if len(x_vars) == 0:
        raise ValueError("Need at least one predictor variable")

    n_vars = len(x_vars)
    y = data_dict[y_var]
    n_obs = len(y)

    # Check if bruteforce is feasible
    if n_vars >= n_obs and bruteforce:
        if not silent:
            print(
                "You have more variables than observations. "
                "Switching to bruteforce=False."
            )
        bruteforce = False

    if n_vars > 14 and bruteforce and not silent:
        print(
            "You have more than 14 variables. The computation might take a lot of time."
        )

    # Initialize for bruteforce=False case (non-bruteforce mode)
    selected_names: list[str] = []

    # --- Non-bruteforce mode ---
    if not bruteforce:
        if not silent:
            print("Selecting the best model...")
        best_model = stepwise(
            data_dict,
            ic=ic,
            silent=True,
            distribution=distribution,
            **kwargs,
        )

        # If intercept-only model selected, return it
        if best_model.coef is None or len(best_model.coef) == 0:
            elapsed = time.time() - start_time
            return CALMResult(
                coefficients=np.array([best_model.intercept_]),
                coefficient_names=["(Intercept)"],
                time_elapsed=elapsed,
            )

        # Get selected variable names
        if best_model._feature_names is not None:
            selected_names = list(best_model._feature_names)

        # If <14 params, recurse with bruteforce on selected vars
        if best_model.nparam < 14 and len(selected_names) > 0:
            subset_data = {y_var: data_dict[y_var]}
            for name in selected_names:
                if name in data_dict:
                    subset_data[name] = data_dict[name]
            return CALM(
                subset_data,
                ic=ic,
                bruteforce=True,
                silent=silent,
                distribution=distribution,
                **kwargs,
            )

        # Stress-testing mode: best model +/- each single variable
        best_vars = [name for name in selected_names if name in x_vars]
        excluded_vars = [v for v in x_vars if v not in best_vars]

        # Build binary combination matrix
        n_in_model = len(best_vars)
        n_combos = 1 + len(excluded_vars) + n_in_model
        var_binary = np.zeros((n_combos, n_vars), dtype=int)

        # Row 0: best model
        for var in best_vars:
            var_binary[0, x_vars.index(var)] = 1

        # Copy best model pattern to all rows
        for row in range(1, n_combos):
            var_binary[row] = var_binary[0].copy()

        # Rows 1..len(excluded_vars): add one excluded var
        for i, var in enumerate(excluded_vars):
            var_binary[1 + i, x_vars.index(var)] = 1

        # Rows after that: remove one included var
        for i in range(n_in_model):
            row = 1 + len(excluded_vars) + i
            included_indices = np.where(var_binary[row] == 1)[0]
            var_binary[row, included_indices[i]] = 0

        # Build all_combinations from var_binary
        all_combinations: list[list[str]] = []
        for row in range(n_combos):
            combo = [x_vars[j] for j in range(n_vars) if var_binary[row, j] == 1]
            all_combinations.append(combo)
    else:
        # Bruteforce mode: generate all 2^n combinations
        all_combinations: list[list[str]] = [[]]  # type: ignore[no-redef]
        for r in range(1, n_vars + 1):
            for comb in itertools.combinations(x_vars, r):
                all_combinations.append(list(comb))

    n_combinations = len(all_combinations)
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

    for i, combo in enumerate(all_combinations):
        if not silent and i > 0:
            prev = f"{round(i / n_combinations * 100)}%"
            print("\b" * len(prev), end="")
            print(
                f"{round((i + 1) / n_combinations * 100)}%",
                end="",
            )

        # Build formula
        if len(combo) == 0:
            formula_str = f"{y_var} ~ 1"
        else:
            formula_str = f"{y_var} ~ " + " + ".join(combo)

        try:
            y_fit, X_fit = formula_func(formula_str, data_dict, as_dataframe=True)
            feature_names = [c for c in X_fit.columns if c != "(Intercept)"]

            model = ALM(
                distribution=distribution,
                loss="likelihood",
                **kwargs,
            )
            model.fit(
                X_fit,
                y_fit,
                formula=formula_str,
                feature_names=feature_names,
            )

            ic_value = _get_ic_value(model, ic)
            ics_array[i] = ic_value
            log_liks[i] = model.log_lik

            # Build full coefficient vector
            full_coef = np.zeros(n_params)
            full_coef[0] = model.intercept_
            for j, var in enumerate(combo):
                idx = x_vars.index(var) + 1
                full_coef[idx] = model.coef[j]
            parameters[i] = full_coef

            # Build full vcov matrix
            model_vcov = model.vcov()
            model_indices = [0] + [x_vars.index(var) + 1 for var in combo]
            for mi, fi in enumerate(model_indices):
                for mj, fj in enumerate(model_indices):
                    vcov_values[i, fi, fj] = model_vcov[mi, mj]

            if has_other and model.other_ is not None:
                other_parameters[i] = model.other_  # type: ignore[index]

        except Exception:  # noqa: BLE001, S112 - skip models that fail to fit
            continue

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

    # Build design matrix with all variables
    formula_full = f"{y_var} ~ " + " + ".join(x_vars)
    _y_final, X_full = formula_func(formula_full, data_dict, as_dataframe=True)
    X_matrix = np.array(X_full)

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

    # Residuals/errors (distribution-specific)
    errors = _combine_errors(distribution, y, mu)

    # Importance: sum of IC weights for models containing each variable
    var_binary = np.zeros((n_combinations, n_vars))
    for i, combo in enumerate(all_combinations):
        for var in combo:
            idx = x_vars.index(var)
            var_binary[i, idx] = 1

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

    return CALMResult(
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
        n_obs=n_obs,
        time_elapsed=elapsed_time,
        actuals=y,
        X_train=X_matrix,
        formula_str=formula_full,
    )
