"""Summary results for ALM model."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

DISTRIBUTION_NAMES = {
    "dnorm": "Normal",
    "dlaplace": "Laplace",
    "ds": "S",
    "dgnorm": "Generalized Normal",
    "dlogis": "Logistic",
    "dt": "Student-t",
    "dalaplace": "Asymmetric Laplace",
    "dlnorm": "Log-Normal",
    "dllaplace": "Log-Laplace",
    "dls": "Log-S",
    "dlgnorm": "Log-Generalized Normal",
    "dbcnorm": "Box-Cox Normal",
    "dinvgauss": "Inverse Gaussian",
    "dgamma": "Gamma",
    "dexp": "Exponential",
    "dchisq": "Chi-Squared",
    "dfnorm": "Folded Normal",
    "drectnorm": "Rectified Normal",
    "dpois": "Poisson",
    "dnbinom": "Negative Binomial",
    "dbinom": "Binomial",
    "dgeom": "Geometric",
    "dbeta": "Beta",
    "dlogitnorm": "Logit-Normal",
    "plogis": "Logit",
    "pnorm": "Probit",
}


@dataclass
class SummaryResult:
    """Summary results for ALM model.

    Attributes
    ----------
    coefficients : np.ndarray
        Parameter estimates (including intercept).
    se : np.ndarray
        Standard errors of coefficients.
    t_stat : np.ndarray
        T-statistics for each coefficient.
    p_value : np.ndarray
        P-values for each coefficient.
    lower_ci : np.ndarray
        Lower confidence interval bounds.
    upper_ci : np.ndarray
        Upper confidence interval bounds.
    scale : float
        Scale parameter (sigma).
    df_residual : int
        Degrees of freedom for residuals.
    distribution : str
        Distribution used in the model.
    log_lik : float
        Log-likelihood of the model.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    nobs : int
        Number of observations.
    nparam : int
        Number of parameters.
    response_variable : str
        Name of the response variable.
    loss : str
        Loss function used in estimation.
    aicc : float
        Corrected AIC.
    bicc : float
        Corrected BIC.
    feature_names : list
        Names of the predictor variables.
    significance : np.ndarray
        Whether the confidence interval of each parameter excludes zero, as R.
    r_squared : float
        The coefficient of determination of the residuals, as R.
    adj_r_squared : float
        Its version adjusted for the number of parameters, as R.
    scale_parameters : np.ndarray
        Whether each parameter is one of the scale model, as R's
        ``scaleParameters``: the summary of a scale model has only those.
    """

    coefficients: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray
    distribution: str
    nobs: int
    nparam: int
    scale: float | None = None
    sigma: float | None = None
    df_residual: int | None = None
    log_lik: float | None = None
    aic: float | None = None
    bic: float | None = None
    response_variable: str = "y"
    loss: str = "likelihood"
    aicc: float | None = None
    bicc: float | None = None
    feature_names: list | None = None
    time_elapsed: float | None = None
    arima_string: str | None = None
    significance: np.ndarray | None = None
    r_squared: float | None = None
    adj_r_squared: float | None = None
    scale_parameters: np.ndarray | None = None

    def __str__(self) -> str:
        """String representation of summary."""
        dist_name = DISTRIBUTION_NAMES.get(self.distribution, self.distribution)

        names = ["(Intercept)"]
        if self.feature_names is not None:
            names.extend(self.feature_names)
        names.extend(f"x{i}" for i in range(len(names), len(self.coefficients)))

        # The level of the intervals, from their width in standard errors
        level = 0.95
        if self.df_residual is not None and self.se[0] > 0:
            t_crit = (self.upper_ci[0] - self.lower_ci[0]) / (2 * self.se[0])
            level = 2 * stats.t.cdf(t_crit, df=self.df_residual) - 1
        col1 = f"Lower {(1 - level) / 2 * 100:.3g}%"
        col2 = f"Upper {(1 + level) / 2 * 100:.3g}%"

        # The parameters of the scale model go in their own table, as R; the summary
        # of a scale model has only those
        scale_parameters = (
            np.zeros(len(self.coefficients), dtype=bool)
            if self.scale_parameters is None
            else np.asarray(self.scale_parameters, dtype=bool)
        )
        scale_only = bool(np.all(scale_parameters))

        lines = []
        if self.time_elapsed is not None:
            lines.append(f"Time elapsed: {self.time_elapsed:.2f} seconds")
            lines.append("")
        if scale_only:
            lines.append(f"Scale model for the variable: {self.response_variable}")
            lines.append(f"Distribution used in the estimation: {dist_name}")
        else:
            lines.append(f"Response variable: {self.response_variable}")
            lines.append(f"Distribution used in the estimation: {dist_name}")
            lines.append(f"Loss function used in estimation: {self.loss}")
        if self.arima_string is not None:
            lines.append(f"{self.arima_string} components were included in the model")
        header = f"{'':>12} {'Estimate':>12} {'Std. Error':>12} {col1:>12} {col2:>12}"
        for scale_table in (False, True):
            rows = np.flatnonzero(scale_parameters == scale_table)
            if len(rows) == 0:
                continue
            lines.append("")
            title = "Coefficients for scale:" if scale_table and not scale_only else ""
            lines.append(title or "Coefficients:")
            lines.append(header)
            lines.append("-" * 60)
            for i in rows:
                significant = (
                    self.significance[i]
                    if self.significance is not None
                    else self.p_value[i] < 0.05
                )
                sig = " *" if significant else ""
                lines.append(
                    f"{names[i]:>12} {self.coefficients[i]:>12.4f} "
                    f"{self.se[i]:>12.4f} {self.lower_ci[i]:>12.4f} "
                    f"{self.upper_ci[i]:>12.4f}{sig}"
                )

        lines.append("")
        if not scale_only:
            lines.append(f"Error standard deviation: {self.sigma:.4f}")
        lines.append(f"Sample size: {self.nobs}")
        lines.append(f"Number of estimated parameters: {self.nparam}")
        lines.append(f"Number of degrees of freedom: {self.df_residual}")
        lines.append("")
        lines.append("Information criteria:")

        ic_names = ["AIC", "AICc", "BIC", "BICc"]
        ic_values = [self.aic]
        ic_values.append(self.aicc if self.aicc is not None else np.nan)
        ic_values.append(self.bic)
        ic_values.append(self.bicc if self.bicc is not None else np.nan)

        ic_header = f"{'':>5}"
        ic_line = f"{'':>5}"
        for name, val in zip(ic_names, ic_values):
            ic_header += f"{name:>12}"
            ic_line += f"{val:>12.4f}"
        lines.append(ic_header)
        lines.append(ic_line)

        return "\n".join(lines)
