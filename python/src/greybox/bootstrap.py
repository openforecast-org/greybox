"""Bootstrap of the parameters of the models, as R's ``coefbootstrap()``."""

import os
import time as time_module
import warnings
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .alm import ALM
from .smoothers import supsmu


class BootstrapResult:
    """Result of :func:`coefbootstrap`, as R's class ``"bootstrap"``.

    Attributes
    ----------
    vcov : np.ndarray
        The covariance matrix of the parameters.
    coefficients : np.ndarray
        The matrix of the bootstrapped parameters, one row per iteration.
    method : str
        The bootstrap method used, ``"cr"`` or ``"dsr"``.
    nsim : int
        The number of iterations.
    size : int or None
        The sample size used in the bootstrap (None for ``"dsr"``).
    replace : bool or None
        Whether the sampling was done with replacement (None for ``"dsr"``).
    prob : np.ndarray or None
        The probability weights of the observations used in the sampling.
    parallel : bool
        Whether the calculations were done in parallel.
    model : str
        The name of the model.
    time_elapsed : float
        The time spent on the calculations, in seconds.
    """

    def __init__(
        self,
        vcov,
        coefficients,
        method,
        nsim,
        size,
        replace,
        prob,
        parallel,
        model,
        time_elapsed,
    ):
        self.vcov = vcov
        self.coefficients = coefficients
        self.method = method
        self.nsim = nsim
        self.size = size
        self.replace = replace
        self.prob = prob
        self.parallel = parallel
        self.model = model
        self.time_elapsed = time_elapsed

    def __str__(self):
        return (
            f"Bootstrap for the {self.model} model with nsim={self.nsim} using "
            f"{self.method} method\nTime elapsed: {self.time_elapsed:.2f} seconds"
        )


def _dsrboot(
    y,
    nsim=100,
    intermittent=False,
    type="multiplicative",
    kind="nonparametric",
    lag=1,
    sd=None,
    scale=True,
):
    """The bootstrapped series of R's dsrboot() (its ``boot``), in the columns."""
    y = np.asarray(y, dtype=float)
    obs = len(y)
    observed = np.ones(obs, dtype=bool)
    y_is_integer = False

    # Intermittent demand: the bootstrap of the demand sizes and of the intervals
    if np.any(y == 0):
        observed = y != 0
        if intermittent:
            sizes = y[observed]
            sizes_boot = _dsrboot(sizes, nsim, False, type, kind, 1, sd, scale)
            if np.all(sizes > 0):
                sizes_boot = np.abs(sizes_boot)
            if np.all(sizes == np.trunc(sizes)):
                sizes_boot = np.round(sizes_boot)
            intervals = np.diff(np.concatenate([[0], np.flatnonzero(observed) + 1]))
            intervals_boot = np.round(
                np.abs(_dsrboot(intervals, nsim, False, type, kind, 1, sd, scale))
            )
            intervals_boot[intervals_boot <= 0] = 1
            indices = np.cumsum(intervals_boot, axis=0).astype(int)
            y_new = np.zeros((max(indices.max(), obs), nsim))
            y_new[indices - 1, np.arange(nsim)] = sizes_boot
            return y_new[:obs]
        y_is_integer = bool(np.all(y == np.trunc(y)))

    # Binary data is not bootstrapped
    if np.all(np.isin(y / np.max(y), [0, 1])):
        warnings.warn(
            "The data is binary, so there is no point in bootstraping it.",
            stacklevel=2,
        )
        return np.tile(y[:, None], (1, nsim))

    y_transformed = y.copy()
    y_order = np.argsort(y, kind="stable")
    y_sorted = np.sort(y)
    if type == "multiplicative":
        with np.errstate(divide="ignore", invalid="ignore"):
            y_transformed = np.log(np.where(y_transformed == 0, np.nan, y_transformed))
            y_sorted = np.log(np.where(y_sorted == 0, np.nan, y_sorted))
    non_nan = ~np.isnan(y_sorted)
    obs_non_nan = int(np.sum(non_nan))

    # The smoothed sorted series
    y_smooth = y_sorted.copy()
    y_smooth[non_nan] = supsmu(np.arange(1.0, obs_non_nan + 1), y_sorted[non_nan])["y"]

    if kind == "nonparametric" and obs_non_nan > 1:
        # The differences of the smoothed data, taken from their smoothed quantiles
        differences = np.sort(np.diff(y_smooth[non_nan]))
        differences = differences[np.isfinite(differences)]
        cumulative = np.linspace(0, 1, len(differences) + 2)[1:-1]
        splined = supsmu(cumulative, differences)
        signs = np.random.choice([-1, 1], size=obs_non_nan * nsim, replace=True)
        differences_new = (
            signs
            * np.interp(
                np.random.uniform(0, 1, obs_non_nan * nsim), splined["x"], splined["y"]
            )
        ).reshape(obs_non_nan, nsim, order="F")
    elif kind == "nonparametric":
        differences_new = np.zeros((obs_non_nan, nsim))
    else:
        if sd is None:
            lag = 1 if lag > obs_non_nan else lag
            sd = 1.0
            if obs_non_nan > 1:
                finite = y_transformed[~np.isnan(y_transformed)]
                sd = np.nanmean(np.abs(finite[lag:] - finite[:-lag]))
        differences_new = np.random.normal(0, sd, (obs_non_nan, nsim))

    y_new = np.vstack(
        [
            np.full((obs - obs_non_nan, nsim), np.nan),
            y_smooth[non_nan, None] + differences_new,
        ]
    )

    # Scale the data to get a similar sd of the mean as in the sample
    if obs_non_nan > 1 and scale:
        sd_data = np.std(y, ddof=1) / np.sqrt(obs)
        means = y_new if type == "additive" else np.exp(y_new)
        sd_boot = np.std(np.nanmean(means, axis=0), ddof=1)
        y_new = y_smooth[:, None] + (sd_data / sd_boot) * (y_new - y_smooth[:, None])

    # Sort the values (NaN first) and put them in the original order
    y_sorted_new = np.sort(np.where(np.isnan(y_new), -np.inf, y_new), axis=0)
    y_sorted_new[np.isneginf(y_sorted_new)] = np.nan
    y_new[y_order] = y_sorted_new
    # Centre the points around the original data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y_new = y_transformed[:, None] + y_new - np.nanmean(y_new, axis=1)[:, None]

        # Make sure that the sd of the data is constant
        if obs_non_nan > 1:
            deviations = y_new - y_transformed[:, None]
            sd_new = np.sqrt(np.nanmean(deviations**2, axis=1))
            sd_new[sd_new == 0] = np.min(sd_new[sd_new != 0], initial=np.inf)
            y_new = (
                y_transformed[:, None]
                + (np.nanmean(sd_new) / sd_new)[:, None] * deviations
            )

    if type == "multiplicative":
        y_new = np.exp(y_new)
    # The zeros where they were in the original data, and integers if it was
    y_new[~observed] = 0
    if y_is_integer:
        y_new = np.ceil(y_new)
    return y_new


def coefbootstrap(
    model: ALM,
    nsim: int = 1000,
    size: int | None = None,
    replace: bool = False,
    prob=None,
    parallel: bool | int = False,
    method: str = "cr",
    **kwargs,
) -> BootstrapResult:
    """Bootstrap for parameters of models.

    The function applies the same model as in the provided object on a smaller
    sample in order to get the estimates of parameters and capture the
    uncertainty about them, as R's ``coefbootstrap()``. The ``"cr"`` (case
    resampling) method draws the observations at random, assuming that they are
    independent; for the models with ARI components, it takes continuous samples
    with a random start (and a random size, unless ``size`` is provided). The
    ``"dsr"`` (Data Shape Replication) method refits the model to the responses
    bootstrapped by R's ``dsrboot()``.

    Parameters
    ----------
    model : ALM
        The fitted model.
    nsim : int, default=1000
        Number of iterations (simulations) to run.
    size : int, optional
        The number of observations in each sample of ``"cr"``: 75% of the
        observations if not provided (random for the models with ARI components).
    replace : bool, default=False
        Whether the sampling of ``"cr"`` is done with replacement.
    prob : array-like, optional
        Probability weights of the observations in the sampling of ``"cr"``.
    parallel : bool or int, default=False
        Whether to do the calculations in parallel, or the number of processes to
        use for them.
    method : {"cr", "dsr"}, default="cr"
        The bootstrap method.
    **kwargs
        Passed to the bootstrap of the data in ``"dsr"`` (as R's ``dsrboot()``:
        ``intermittent``, ``type``, ``kind``, ``lag``, ``sd``, ``scale``).

    Returns
    -------
    BootstrapResult
        The covariance matrix and the matrix of the bootstrapped parameters, with
        the settings of the bootstrap.

    Examples
    --------
    >>> from greybox import ALM, coefbootstrap, formula, mtcars
    >>> y, X = formula("mpg ~ wt + hp", mtcars)
    >>> model = ALM(distribution="dlnorm").fit(X, y)
    >>> bootstrap = coefbootstrap(model, nsim=10)
    >>> bootstrap.vcov.shape
    (3, 3)
    """
    start_time = time_module.time()
    if method not in ("cr", "dsr"):
        raise ValueError("method should be either 'cr' or 'dsr'")
    n_cores = min((os.cpu_count() or 1) - 1, nsim) if parallel is True else parallel
    parallel = bool(parallel)

    coefficients_original = np.asarray(model.coefficients, dtype=float)
    n_variables = len(coefficients_original)
    obs = model.nobs
    n_exog = model._n_features_exog
    y = np.asarray(model.actuals, dtype=float)
    # The regressors without the lags of the response, which the fit rebuilds
    X = model._X_train_[:, :n_exog]
    arima_model = model.arima_polynomial_ is not None

    # The same model, with the extra parameter of the distribution fixed and the
    # occurrence model estimated on each sample, as R
    params = model.get_params()
    other = {
        "dt": "nu",
        "dchisq": "nu",
        "dnbinom": "size",
        "dalaplace": "alpha",
        "dbcnorm": "lambda_bc",
        "dgnorm": "shape",
        "dlgnorm": "shape",
    }.get(model.distribution)
    if other is not None:
        params[other] = model.other_
    if isinstance(model.occurrence_, ALM):
        params["occurrence"] = model.occurrence_.distribution

    if method == "cr":
        change_size = arima_model and size is None
        size = int(np.floor(0.75 * obs)) if size is None else size
        p = None if prob is None else np.asarray(prob, dtype=float) / np.sum(prob)
        subsets = []
        for _ in range(nsim):
            if arima_model:
                # Continuous samples with a random start, needed for dynamic models
                sample_size = (
                    int(np.floor(np.random.uniform(n_variables, obs))) + 1
                    if change_size
                    else size
                )
                start = int(np.floor(np.random.uniform(0, obs - sample_size)))
                subsets.append(start + np.arange(sample_size))
            else:
                subsets.append(np.random.choice(obs, size=size, replace=replace, p=p))
        Xs, ys = [X[subset] for subset in subsets], [y[subset] for subset in subsets]
    else:
        y_boot = _dsrboot(y, nsim=nsim, **kwargs)
        Xs, ys = [X] * nsim, list(y_boot.T)

    estimator = ALM(**params)
    coefficients = np.zeros((nsim, n_variables))
    fits: Iterable[ALM]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if parallel:
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                fits = list(executor.map(estimator.fit, Xs, ys))
        else:
            fits = map(estimator.fit, Xs, ys)
        for i, fitted in enumerate(fits):
            # The dropped regressors have zero parameters
            kept = fitted._columns_kept
            positions = np.concatenate(
                [kept, kept + n_exog]
                if model.distribution == "dbeta"
                else [kept, np.arange(n_exog, n_variables)]
            )
            coefficients[i, positions] = fitted.coefficients

    centred = coefficients - np.mean(coefficients, axis=0)
    return BootstrapResult(
        vcov=centred.T @ centred / (nsim - 1),
        coefficients=coefficients,
        method=method,
        nsim=nsim,
        size=None if method == "dsr" else size,
        replace=None if method == "dsr" else replace,
        prob=None if method == "dsr" else prob,
        parallel=parallel,
        model=type(model).__name__,
        time_elapsed=time_module.time() - start_time,
    )
