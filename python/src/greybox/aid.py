"""Automatic Identification of Demand (AID).

Python translation of R's :func:`greybox::aid` and :func:`greybox::aidCat`.
The :func:`aid` function fits several candidate models to a single time
series and picks the demand type that minimises the chosen information
criterion. The :func:`aid_cat` function applies :func:`aid` to a collection
of series and summarises the result as a 2x3 demand category table.

Functions
---------
- :func:`aid` -- Identify the demand type of a single series.
- :func:`aid_cat` -- Apply :func:`aid` across multiple series.

Result classes
--------------
- :class:`AidResult` -- Returned by :func:`aid`.
- :class:`AidCatResult` -- Returned by :func:`aid_cat`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .alm import ALM
from .formula import formula as build_formula
from .pointlik import point_lik_cumulative
from .smoothers import lowess, supsmu


@dataclass
class AidType:
    """Sub-categorisation of an AID classification (mirrors R's ``$type``).

    Both attribute access (``aid_type.type1``) and dict-style access
    (``aid_type["type1"]``) work; the latter exists for backward
    compatibility with code written against the earlier dict-based API.

    Attributes
    ----------
    type1 : str or None
        ``"count"`` or ``"fractional"``.
    type2 : str or None
        ``"regular"`` or ``"intermittent"``.
    type2a : str or None
        ``"smooth"`` or ``"lumpy"`` when intermittent, ``None`` otherwise.
    """

    type1: str | None
    type2: str | None
    type2a: str | None

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.keys()

    def keys(self):
        return ("type1", "type2", "type2a")

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class Stockouts:
    """Stockout / new / obsolete metadata (mirrors R's ``$stockouts``).

    Both attribute access (``stockouts.start``) and dict-style access
    (``stockouts["start"]``) work; the latter exists for backward
    compatibility with code written against the earlier dict-based API.

    Attributes
    ----------
    start : numpy.ndarray or None
        1-based positions where each stockout begins. ``None`` if the
        original series had no zeros at all; an empty array if zeros
        were present but none were flagged as stockouts.
    end : numpy.ndarray or None
        1-based positions where each stockout ends. Same length as
        ``start``.
    dummy : numpy.ndarray
        Length-``n`` binary indicator: ``1`` on observations inside a
        stockout window, ``0`` elsewhere.
    new : numpy.ndarray
        Length-``n`` binary indicator: ``1`` over the leading-zeros
        block when the series is flagged as a new product.
    obsolete : numpy.ndarray
        Length-``n`` binary indicator: ``1`` over the trailing-zeros
        block when the series is flagged as obsolete.
    """

    start: np.ndarray | None
    end: np.ndarray | None
    dummy: np.ndarray
    new: np.ndarray
    obsolete: np.ndarray

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.keys()

    def keys(self):
        return ("start", "end", "dummy", "new", "obsolete")

    def get(self, key: str, default=None):
        return getattr(self, key, default)


_VALID_ICS = ("AICc", "AIC", "BICc", "BIC")


def _get_ic(model: ALM, ic_name: str) -> float:
    """Return the requested IC value from a fitted ALM model."""
    attr = {"AICc": "aicc", "AIC": "aic", "BICc": "bicc", "BIC": "bic"}[ic_name]
    value = getattr(model, attr)
    if value is None:
        raise ValueError(f"Model has no {ic_name} value (was the loss 'likelihood'?)")
    return float(value)


def _fit_alm(formula_str: str, data: dict, **alm_kwargs) -> ALM | None:
    """Fit an ALM via the formula interface, returning None on failure.

    Mirrors R's ``suppressWarnings(alm(...))`` and the subsequent
    null-dropping at aid()'s line 307.
    """
    nlopt_kargs = alm_kwargs.pop("nlopt_kargs", None)
    maxeval = alm_kwargs.pop("maxeval", None)
    if maxeval is not None:
        nlopt_kargs = dict(nlopt_kargs or {})
        nlopt_kargs["maxeval"] = maxeval
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_vec, X = build_formula(formula_str, data)
            model = ALM(nlopt_kargs=nlopt_kargs, **alm_kwargs)
            model.fit(X, y_vec, formula=formula_str)
        return model
    except Exception:  # noqa: BLE001
        return None


def _interp_rule2(x_indexed: np.ndarray) -> np.ndarray:
    """Linear interpolation through NaNs, edge-clamped (R `approx(rule=2)`)."""
    n = len(x_indexed)
    idx = np.arange(n, dtype=float)
    mask = ~np.isnan(x_indexed)
    if not mask.any():
        return np.zeros(n)
    return np.interp(idx, idx[mask], x_indexed[mask])


class AidResult:
    """Result of :func:`aid`. Mirrors R's ``aid`` S3 class.

    The nested ``type`` and ``stockouts`` fields are typed dataclasses
    (:class:`AidType` and :class:`Stockouts`) that support both
    attribute access and dict-style access. The two patterns below are
    equivalent::

        result.stockouts.start    # Python idiom
        result.stockouts["start"] # R-list / dict idiom

    Attributes
    ----------
    y : numpy.ndarray
        The original time series with NaN values replaced by 0.
    models : dict[str, ALM]
        All fitted candidate models, keyed by demand-type name. If
        stockout detection ran, the stockout model is exposed under
        the key ``"stockout"``.
    name : str
        Name of the selected demand type (e.g. ``"smooth intermittent
        count"``).
    type : AidType
        Sub-categorisation of ``name`` with fields ``type1``,
        ``type2``, ``type2a``.
    stockouts : Stockouts
        Stockout / new / obsolete metadata. ``start`` and ``end`` are
        1-based positions matching R; ``dummy`` / ``new`` / ``obsolete``
        are binary numpy arrays of length ``len(y)``.
    new : bool
        ``True`` if the series starts with anomalous zeros (a new
        product).
    obsolete : bool
        ``True`` if the series ends with anomalous zeros (a
        discontinued product).
    ic : str
        Information criterion used to select between candidates.
    """

    __slots__ = (
        "ic",
        "models",
        "name",
        "new",
        "obsolete",
        "stockouts",
        "type",
        "y",
    )

    def __init__(
        self,
        y: np.ndarray,
        models: dict,
        name: str,
        type: AidType | dict,
        stockouts: Stockouts | dict,
        new: bool,
        obsolete: bool,
        ic: str = "AICc",
    ):
        self.y = y
        self.models = models
        self.name = name
        # Accept either the dataclass or a legacy dict for type/stockouts.
        self.type = type if isinstance(type, AidType) else AidType(**type)
        self.stockouts = (
            stockouts if isinstance(stockouts, Stockouts) else Stockouts(**stockouts)
        )
        self.new = new
        self.obsolete = obsolete
        self.ic = ic

    def __str__(self) -> str:
        lines = []
        starts = self.stockouts.start
        n_stk = 0 if starts is None else len(starts)
        if n_stk > 1:
            lines.append(f"There are {n_stk} potential stockouts in the data.")
        elif n_stk == 1:
            lines.append("There is 1 potential stockout in the data")
        if self.new:
            lines.append("The product is new (sales start later)")
        if self.obsolete:
            lines.append("The product has become obsolete")
        lines.append(f"The provided time series is {self.name}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"AidResult(name={self.name!r}, new={self.new}, obsolete={self.obsolete})"
        )

    def plot(self, ax=None, **kwargs):
        """Plot the series with stockout / new / obsolete overlays.

        Mirrors R's ``plot.aid()``:

        - Line plot of ``y`` against 1-based positions.
        - Solid red vertical line at each stockout start, dashed green
          vertical line at each stockout end.
        - Light-grey shaded rectangle covering each stockout span.
        - Light-blue shaded rectangle covering leading zeros when the
          series is flagged as a *new product*.
        - Light-orange shaded rectangle covering trailing zeros when
          the series is flagged as *obsolete*.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if ``None``.
        **kwargs
            Forwarded to :meth:`matplotlib.axes.Axes.plot` for the
            main series line.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        n = len(self.y)
        x_idx = np.arange(1, n + 1)  # 1-based to match R
        ax.plot(x_idx, self.y, **kwargs)

        starts = self.stockouts.start
        ends = self.stockouts.end
        if starts is not None and len(starts) > 0:
            for s, e in zip(starts, ends):
                ax.axvline(float(s) - 0.5, color="red")
                ax.axvline(float(e) + 0.5, color="green", linestyle="--")
                ax.axvspan(
                    float(s) - 0.5,
                    float(e) + 0.5,
                    color="lightgrey",
                    alpha=0.5,
                )
        if self.new:
            first_nz = int(np.where(self.y != 0)[0][0]) + 1
            ax.axvspan(0.5, first_nz - 0.5, color="lightblue", alpha=0.3)
        if self.obsolete:
            last_nz = int(np.where(self.y != 0)[0][-1]) + 1
            ax.axvspan(last_nz + 0.5, n + 0.5, color="lightsalmon", alpha=0.3)

        ax.set_xlabel("Time")
        ax.set_ylabel("y")
        ax.set_title(f"AID: {self.name}")
        return ax


class AidCatResult:
    """Result of :func:`aid_cat`. Mirrors R's ``aidCat`` S3 class.

    Attributes
    ----------
    categories : pandas.Categorical
        Categorical vector of demand types, one entry per input series,
        with the six possible levels: ``regular count``,
        ``smooth intermittent count``, ``lumpy intermittent count``,
        ``regular fractional``, ``smooth intermittent fractional``,
        ``lumpy intermittent fractional``.
    types : pandas.DataFrame
        2x3 frequency table with rows ``["Count", "Fractional"]`` and
        columns ``["Regular", "Smooth Intermittent", "Lumpy
        Intermittent"]``.
    anomalies : pandas.Series
        Counts of ``"New"``, ``"Stockouts"``, and ``"Old"`` (obsolete)
        across the input series.
    results : list[AidResult]
        The individual :class:`AidResult` objects, in the same order as
        the input columns.
    """

    __slots__ = ("anomalies", "categories", "results", "types")

    def __init__(
        self,
        categories: pd.Categorical,
        types: pd.DataFrame,
        anomalies: pd.Series,
        results: list[AidResult] | None = None,
    ):
        self.categories = categories
        self.types = types
        self.anomalies = anomalies
        self.results = results

    def __str__(self) -> str:
        return (
            "Demand categories:\n"
            + str(self.types)
            + "\n\nAnomalies:\n"
            + str(self.anomalies)
        )

    def __repr__(self) -> str:
        return f"AidCatResult(anomalies={dict(self.anomalies)})"

    def plot(self, ax=None, **kwargs):
        """Plot the 2x3 demand-category panel.

        Mirrors R's ``plot.aidCat()`` layout: six cells, one per
        demand category, each annotated with the count of series
        falling in it. Row marginals (Count / Fractional totals) are
        drawn along the right edge, column marginals (Regular /
        Smooth Intermittent / Lumpy Intermittent totals) along the
        bottom edge, and the grand total at the bottom-right.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if ``None``.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.subplots` when
            ``ax`` is ``None``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(**kwargs)

        t = self.types  # rows=["Count","Fractional"], cols=Regular/Smooth/Lumpy
        ax.set_xlim(-1, 2.2)
        ax.set_ylim(-1.2, 1)
        ax.set_axis_off()
        ax.set_xlabel("Demand intervals")
        ax.set_ylabel("Demand sizes type")

        # Horizontal and vertical grid lines.
        for y_h in (-1, 0, 1):
            ax.plot([-1, 2.2], [y_h, y_h], color="black", linewidth=0.7)
        for x_v in (-1, 0, 1, 2):
            ax.plot([x_v, x_v], [-1.2, 1], color="black", linewidth=0.7)

        # Category cells (top row = Count, bottom row = Fractional).
        labels = [
            (-0.5, 0.5, "Regular Count", t.iat[0, 0]),
            (0.5, 0.5, "Smooth Intermittent Count", t.iat[0, 1]),
            (1.5, 0.5, "Lumpy Intermittent Count", t.iat[0, 2]),
            (-0.5, -0.5, "Regular Fractional", t.iat[1, 0]),
            (0.5, -0.5, "Smooth Intermittent Fractional", t.iat[1, 1]),
            (1.5, -0.5, "Lumpy Intermittent Fractional", t.iat[1, 2]),
        ]
        for x, y, label, n in labels:
            ax.text(
                x,
                y,
                f"{label}\n({int(n)})",
                ha="center",
                va="center",
                fontsize=9,
            )

        # Column marginals along the bottom edge.
        col_totals = [int(t.iloc[:, c].sum()) for c in range(3)]
        col_labels = ["Regular", "Smooth Intermittent", "Lumpy Intermittent"]
        for x, label, n in zip((-0.5, 0.5, 1.5), col_labels, col_totals):
            ax.text(
                x,
                -1.1,
                f"{label}\n({n})",
                ha="center",
                va="center",
                fontsize=9,
            )

        # Row marginals along the right edge (rotated).
        row_totals = [int(t.iloc[r, :].sum()) for r in range(2)]
        for y, label, n in zip((0.5, -0.5), ("Count", "Fractional"), row_totals):
            ax.text(
                2.1,
                y,
                f"{label}\n({n})",
                ha="center",
                va="center",
                rotation=90,
                fontsize=9,
            )

        # Grand total in the bottom-right corner.
        ax.text(
            2.1,
            -1.1,
            f"Total\n({int(t.values.sum())})",
            ha="center",
            va="center",
            fontsize=9,
        )
        return ax


def aid(
    y,
    ic: str = "AICc",
    level: float = 0.99,
    loss: str = "likelihood",
    **alm_kwargs: Any,
) -> AidResult:
    """Automatic identification of demand type.

    Classifies a time series into one of six demand types and flags
    stockouts, new products, and obsolete products. This is a one-to-one
    Python port of R's :func:`greybox::aid`.

    The algorithm runs in two stages:

    1. **Stockout detection.** Inter-demand intervals are smoothed with
       :func:`~greybox.smoothers.lowess` and fitted with an
       :class:`~greybox.alm.ALM` Geometric model. Cumulative
       probabilities above ``level`` are flagged as potential
       stockouts, with leading/trailing zero runs interpreted as new
       or obsolete products.
    2. **Demand type classification.** Smoothed regressors built with
       :func:`~greybox.smoothers.supsmu` are used to fit up to four
       candidate models (rectified normal, normal + Bernoulli
       occurrence, negative binomial, negative binomial + Bernoulli
       occurrence). The model with the lowest information criterion
       names the demand type.

    Parameters
    ----------
    y : array_like
        The time series. ``NaN`` values are replaced by zero.
    ic : {"AICc", "AIC", "BICc", "BIC"}, default="AICc"
        Information criterion used to choose between candidate models.
    level : float, default=0.99
        Confidence level for stockout identification.
    loss : str, default="likelihood"
        Loss function passed to :class:`~greybox.alm.ALM`.
    **alm_kwargs
        Extra keyword arguments forwarded to :class:`~greybox.alm.ALM`.
        Useful for tuning the optimiser, e.g. ``nlopt_kargs={"maxeval":
        1000}``.

    Returns
    -------
    AidResult
        Object containing the identified demand type, fitted models,
        stockout indices, and new/obsolete flags.

    Notes
    -----
    The six possible demand types are:

    - ``regular count``
    - ``regular fractional``
    - ``smooth intermittent count``
    - ``smooth intermittent fractional``
    - ``lumpy intermittent count``
    - ``lumpy intermittent fractional``

    Examples
    --------
    Classify an intermittent count series:

    >>> import numpy as np
    >>> from greybox import aid
    >>> rng = np.random.default_rng(42)
    >>> y = rng.poisson(0.7, 120).astype(float)
    >>> res = aid(y)
    >>> res.name in {
    ...     "smooth intermittent count", "lumpy intermittent count"
    ... }
    True

    Inspect detected stockouts:

    >>> y2 = rng.poisson(3, 100).astype(float)
    >>> y2[40:50] = 0  # injected stockout
    >>> res2 = aid(y2)
    >>> res2.stockouts["start"]  # 1-based, matches R
    array([41])

    See Also
    --------
    aid_cat : Apply :func:`aid` to multiple series.
    """
    if ic not in _VALID_ICS:
        raise ValueError(f"ic must be one of {_VALID_ICS}, got {ic!r}")

    y = np.asarray(y, dtype=float).copy()
    obs_in_sample = len(y)
    if obs_in_sample == 0:
        raise ValueError("y must have at least one observation")
    # R: y[is.na(y)] <- 0
    y[np.isnan(y)] = 0.0

    # ---------- Stockout detection (R 62-170) ----------
    stockout_model: ALM | None = None
    product_new = False
    product_obsolete = False
    stockouts_start: np.ndarray | None = None
    stockouts_end: np.ndarray | None = None

    has_zeros = bool(np.any(y == 0))

    if has_zeros:
        # R: yIntervals <- diff(c(0, which(y != 0), obsInSample+1))
        nonzero_pos = np.where(y != 0)[0] + 1  # 1-based positions
        padded = np.concatenate(([0], nonzero_pos, [obs_in_sample + 1])).astype(float)
        y_intervals = np.diff(padded)
        n_int = len(y_intervals)

        # R: x = lowess(1:length(yIntervals), yIntervals)$y
        x_smoothed = lowess(np.arange(1, n_int + 1, dtype=float), y_intervals)["y"]
        # Build data, response is (yIntervals - 1) per R 'y-1~x'.
        df_intervals = {
            "y_minus_1": y_intervals - 1.0,
            "x": x_smoothed,
        }
        stockout_model = _fit_alm(
            "y_minus_1 ~ x",
            df_intervals,
            distribution="dgeom",
            loss=loss,
            **alm_kwargs,
        )
        if stockout_model is None:
            raise RuntimeError("Stockout model failed to fit.")
        probabilities = point_lik_cumulative(stockout_model)

        # R: productNew if first interval > 1.
        if y_intervals[0] != 1:
            product_new = True
        if probabilities[-1] > level and y_intervals[-1] != 1:
            product_obsolete = True

        # R 105-128: refit on trimmed data if head/tail anomalies detected.
        if product_new or product_obsolete:
            # Indices of yIntervals to keep (1-based in R, 0-based here).
            head_ids: list[int] = [] if product_new else [0]
            mid_ids = list(range(1, n_int - 1))
            tail_ids = [] if product_obsolete else [n_int - 1]
            y_intervals_ids = head_ids + mid_ids + tail_ids

            df_trim = {
                "y_minus_1": y_intervals[y_intervals_ids] - 1.0,
            }
            stockout_model = _fit_alm(
                "y_minus_1 ~ 1",
                df_trim,
                distribution="dgeom",
                loss=loss,
                **alm_kwargs,
            )
            if stockout_model is None:
                raise RuntimeError("Trimmed stockout model failed to fit.")
            probs_trim = point_lik_cumulative(stockout_model)
            # R: pad probabilities with 0 sentinels at head/tail.
            head_pad = [0.0] if product_new else []
            tail_pad = [0.0] if product_obsolete else []
            probabilities = np.concatenate([head_pad, probs_trim, tail_pad])
        else:
            y_intervals_ids = list(range(n_int))

        # R 138-140: outliers with interval==1 are not flagged.
        mask = (probabilities > level) & (y_intervals == 1)
        if np.any(mask):
            probabilities = probabilities.copy()
            probabilities[mask] = 0.0

        outliers = probabilities > level
        # 0-based indices into y_intervals
        outliers_id = np.where(outliers)[0]

        cum = np.cumsum(y_intervals).astype(int)  # 1-based positions

        if outliers_id.size > 0:
            # R 146-152: start = cumsum[outliersID-1]+1, with special case
            # for outliersID==1 → start = 1. All positions are 1-based to
            # match R's output exactly.
            stockouts_start = np.empty(outliers_id.size, dtype=int)
            for k, i in enumerate(outliers_id):
                if i == 0:
                    stockouts_start[k] = 1
                else:
                    stockouts_start[k] = cum[i - 1] + 1
            stockouts_end = (cum[outliers_id] - 1).astype(int)

            # 0-based stockout positions for internal filtering.
            stockout_zero_based: set[int] = set()
            for s, e in zip(stockouts_start, stockouts_end):
                stockout_zero_based.update(range(int(s) - 1, int(e)))
        else:
            stockouts_start = np.zeros(0, dtype=int)
            stockouts_end = np.zeros(0, dtype=int)
            stockout_zero_based = set()

        # R 158-163: yIDsToUse = (cumsum[head] .. cumsum[tail]) - 1, dropping
        # stockouts. R uses 1-based indices into y with index 0 silently dropped;
        # the equivalent 0-based range is below.
        first_kept_pos = int(cum[y_intervals_ids[0]])  # 1-based
        last_kept_pos = int(cum[y_intervals_ids[-1]])  # 1-based
        rng_start = max(0, first_kept_pos - 2)
        rng_stop = last_kept_pos - 1
        rng = np.arange(rng_start, rng_stop, dtype=int)
        y_ids_to_use = np.array(
            [p for p in rng if p not in stockout_zero_based], dtype=int
        )
    else:
        y_ids_to_use = np.arange(obs_in_sample)
        stockouts_start = None
        stockouts_end = None

    # ---------- Demand-type classification (R 172-326) ----------
    y_sub = y[y_ids_to_use]
    n_sub = len(y_sub)
    if n_sub == 0:
        raise ValueError(
            "All observations were classified as stockouts/new/obsolete; "
            "nothing left for the demand-type model."
        )

    # R: xregData = data.frame(y=y[yIDsToUse], x=supsmu(...))
    x_main = supsmu(np.arange(1, n_sub + 1, dtype=float), y_sub)["y"]

    # R 179-184: xregDataSizes — supsmu over non-zero values, interpolated.
    x_sizes = np.zeros(n_sub)
    nz_mask = y_sub != 0
    n_nz = int(nz_mask.sum())
    if n_nz > 0:
        smoothed_nz = supsmu(np.arange(1, n_nz + 1, dtype=float), y_sub[nz_mask])["y"]
        x_sizes[nz_mask] = smoothed_nz
    x_sizes_with_nan = x_sizes.copy()
    x_sizes_with_nan[~nz_mask] = np.nan
    x_sizes = _interp_rule2(x_sizes_with_nan)

    # R 187-189: fallback when main supsmu is degenerate.
    if np.all(x_main == 0) or np.all(x_main == 1):
        x_main = x_sizes.copy()

    # R 192-194: drop x if it's essentially constant 1.
    has_x = not np.all(np.round(x_main, 10) == 1)

    # R 198-201: binary / low-volume / zeroes-left checks.
    y_max = np.max(y_sub) if n_sub > 0 else 0
    if y_max == 0:
        y_is_binary = True
    else:
        scaled = y_sub / y_max
        y_is_binary = bool(np.all((scaled == 0) | (scaled == 1)))

    y_is_low_volume = bool(np.all(np.isin(y_sub, [0, 1, 2])))  # noqa: F841
    zeroes_left = bool(np.any(y_sub == 0))
    data_is_integer = bool(np.all(y == np.trunc(y)))

    # Default classification (will be overridden below).
    id_type = "regular fractional"
    id_type_detailed: dict[str, str | None] = {
        "type1": "count/fractional",
        "type2": "regular/intermittent",
        "type2a": "smooth/lumpy",
    }

    # Names exactly as R uses them (must round-trip via .split(" ")).
    id_models: dict[str, ALM | None] = {}

    if zeroes_left:
        # Build occurrence data.
        y_occ = (y_sub != 0).astype(float)
        x_occ_smooth = supsmu(np.arange(1, n_sub + 1, dtype=float), y_occ)["y"]
        occ_data = {"y": y_occ, "x": x_occ_smooth}

        if np.all(x_occ_smooth == x_occ_smooth[0]):
            model_occurrence = _fit_alm(
                "y ~ 1",
                occ_data,
                distribution="plogis",
                loss=loss,
                **alm_kwargs,
            )
        else:
            model_occ_fixed = _fit_alm(
                "y ~ 1",
                occ_data,
                distribution="plogis",
                loss=loss,
                **alm_kwargs,
            )
            model_occurrence = _fit_alm(
                "y ~ x",
                occ_data,
                distribution="plogis",
                loss=loss,
                **alm_kwargs,
            )
            if (
                model_occ_fixed is not None
                and model_occurrence is not None
                and _get_ic(model_occ_fixed, ic) < _get_ic(model_occurrence, ic)
            ):
                model_occurrence = model_occ_fixed

        # Main data for size models.
        if has_x:
            main_data = {"y": y_sub, "x": x_main}
            size_formula = "y ~ x"
        else:
            main_data = {"y": y_sub}
            size_formula = "y ~ 1"

        id_models["smooth intermittent fractional"] = None
        id_models["lumpy intermittent fractional"] = None
        id_models["smooth intermittent count"] = None
        id_models["lumpy intermittent count"] = None

        # Model 1: smooth intermittent fractional (drectnorm).
        id_models["smooth intermittent fractional"] = _fit_alm(
            size_formula,
            main_data,
            distribution="drectnorm",
            loss=loss,
            **alm_kwargs,
        )

        # Initial defaults for type.
        id_type = "smooth intermittent fractional"
        id_type_detailed["type2a"] = "smooth"
        id_type_detailed["type2"] = "intermittent"
        id_type_detailed["type1"] = "fractional"

        if y_is_binary:
            # Use modelOccurrence directly (Bernoulli).
            id_models["smooth intermittent count"] = model_occurrence
            id_type = "smooth intermittent count"
            id_type_detailed["type1"] = "count"
        else:
            # Model 2: lumpy intermittent fractional (mixture).
            if model_occurrence is not None:
                id_models["lumpy intermittent fractional"] = _fit_alm(
                    size_formula,
                    main_data,
                    distribution="dnorm",
                    occurrence=model_occurrence,
                    loss=loss,
                    **alm_kwargs,
                )

        if data_is_integer and not y_is_binary:
            # Model 3: smooth intermittent count via dnbinom.
            id_models["smooth intermittent count"] = _fit_alm(
                size_formula,
                main_data,
                distribution="dnbinom",
                loss=loss,
                maxeval=500,
                **alm_kwargs,
            )

            # Compare against binomial alternatives for very-low-volume data.
            # R's aid() does NOT pass `size` to dbinom; if the optimizer fails
            # the IC comes back effectively infinite and the alternative is
            # not preferred. We must mirror that — passing size would silently
            # produce a fit that beats dnbinom but doesn't match R.
            binom_alt1 = _fit_alm(
                "y ~ 1",
                main_data,
                distribution="dbinom",
                loss=loss,
                **alm_kwargs,
            )
            binom_alt2 = (
                _fit_alm(
                    size_formula,
                    main_data,
                    distribution="dbinom",
                    loss=loss,
                    **alm_kwargs,
                )
                if has_x
                else None
            )
            for alt in (binom_alt1, binom_alt2):
                if alt is None:
                    continue
                try:
                    alt_ic = _get_ic(alt, ic)
                except (ValueError, TypeError):
                    continue
                if not np.isfinite(alt_ic):
                    continue
                base = id_models["smooth intermittent count"]
                if base is None or alt_ic < _get_ic(base, ic):
                    id_models["smooth intermittent count"] = alt

            # Model 4: lumpy intermittent count via dnbinom + occurrence.
            if model_occurrence is not None:
                id_models["lumpy intermittent count"] = _fit_alm(
                    size_formula,
                    main_data,
                    distribution="dnbinom",
                    occurrence=model_occurrence,
                    loss=loss,
                    maxeval=500,
                    **alm_kwargs,
                )
    else:
        # No zeroes left: regular fractional vs regular count.
        if has_x:
            main_data = {"y": y_sub, "x": x_main}
            size_formula = "y ~ x"
        else:
            main_data = {"y": y_sub}
            size_formula = "y ~ 1"

        id_models["regular fractional"] = _fit_alm(
            size_formula,
            main_data,
            distribution="dnorm",
            loss=loss,
            **alm_kwargs,
        )
        if data_is_integer:
            id_models["regular count"] = _fit_alm(
                size_formula,
                main_data,
                distribution="dnbinom",
                loss=loss,
                maxeval=500,
                **alm_kwargs,
            )

    # Drop None / failed models (R 307).
    id_models_fit: dict[str, ALM] = {
        k: v for k, v in id_models.items() if v is not None
    }

    if not id_models_fit:
        raise RuntimeError("All candidate models failed to fit.")

    # R 312-326: pick best by IC unless binary.
    if not y_is_binary:
        best_name = min(
            id_models_fit.keys(), key=lambda k: _get_ic(id_models_fit[k], ic)
        )
        id_type = best_name
        parts = id_type.split(" ")
        if len(parts) == 3:
            id_type_detailed["type2a"] = parts[0]
            id_type_detailed["type2"] = parts[1]
            id_type_detailed["type1"] = parts[2]
        else:
            id_type_detailed["type2"] = parts[0]
            id_type_detailed["type1"] = parts[1]
            id_type_detailed["type2a"] = None

    # R 329-331: attach stockout model if there were any zeros.
    if has_zeros and stockout_model is not None:
        id_models_fit["stockout"] = stockout_model

    # ---------- Output dummies (R 333-349) ----------
    stockout_dummy = np.zeros(obs_in_sample, dtype=int)
    if (
        stockouts_start is not None
        and stockouts_end is not None
        and len(stockouts_start) > 0
    ):
        for s, e in zip(stockouts_start, stockouts_end):
            # 1-based inclusive in R → 0-based half-open in Python.
            stockout_dummy[int(s) - 1 : int(e)] = 1

    new_dummy = np.zeros(obs_in_sample, dtype=int)
    if product_new:
        # R: 1:(which(y!=0)[1]-1) ones.
        first_nz = int(np.where(y != 0)[0][0])
        new_dummy[:first_nz] = 1

    obsolete_dummy = np.zeros(obs_in_sample, dtype=int)
    if product_obsolete:
        last_nz = int(np.where(y != 0)[0][-1])
        obsolete_dummy[last_nz + 1 :] = 1

    stockouts = Stockouts(
        start=stockouts_start,
        end=stockouts_end,
        dummy=stockout_dummy,
        new=new_dummy,
        obsolete=obsolete_dummy,
    )

    return AidResult(
        y=y,
        models=id_models_fit,
        name=id_type,
        type=AidType(**id_type_detailed),
        stockouts=stockouts,
        new=bool(product_new),
        obsolete=bool(product_obsolete),
        ic=ic,
    )


_CATEGORY_LEVELS = [
    "regular count",
    "smooth intermittent count",
    "lumpy intermittent count",
    "regular fractional",
    "smooth intermittent fractional",
    "lumpy intermittent fractional",
]


def aid_cat(data, **aid_kwargs: Any) -> AidCatResult:
    """Apply :func:`aid` to multiple series and summarise the result.

    One-to-one port of R's :func:`greybox::aidCat`. Each column of
    ``data`` is treated as a separate series and passed through
    :func:`aid`; the per-series demand types are then tallied into a
    2x3 demand-category table plus a count of anomalies (new,
    stockouts, obsolete) across the collection.

    Parameters
    ----------
    data : dict, pandas.DataFrame, or 2-D numpy.ndarray
        The series to categorise. Each column / dict entry is one
        series.
    **aid_kwargs
        Keyword arguments forwarded to :func:`aid` (for example
        ``ic="BIC"`` or ``level=0.95``).

    Returns
    -------
    AidCatResult
        Object with the per-series categories, the 2x3 type matrix,
        the anomaly counts, and the individual :class:`AidResult`
        objects.

    Examples
    --------
    Apply to a small dictionary of series:

    >>> import numpy as np
    >>> from greybox import aid_cat
    >>> rng = np.random.default_rng(1)
    >>> series = {
    ...     "a": rng.poisson(1, 80).astype(float),
    ...     "b": rng.poisson(5, 80).astype(float),
    ...     "c": rng.normal(10, 2, 80),
    ... }
    >>> result = aid_cat(series)
    >>> result.types.shape
    (2, 3)

    See Also
    --------
    aid : Apply the classification to a single series.
    """
    if isinstance(data, pd.DataFrame):
        series = {col: data[col].to_numpy(dtype=float) for col in data.columns}
    elif isinstance(data, dict):
        series = {k: np.asarray(v, dtype=float) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("data must be 2-D when passed as ndarray")
        series = {f"V{i + 1}": data[:, i].astype(float) for i in range(data.shape[1])}
    else:
        raise TypeError("data must be a DataFrame, dict of arrays, or 2-D ndarray")

    results: list[AidResult] = []
    for name, vec in series.items():
        result = aid(vec, **aid_kwargs)
        results.append(result)

    cat_names = [r.name for r in results]
    categories = pd.Categorical(cat_names, categories=_CATEGORY_LEVELS, ordered=False)

    # 2x3 types matrix matching R's print.aidCat.
    columns = ["Regular", "Smooth Intermittent", "Lumpy Intermittent"]
    rows = ["Count", "Fractional"]
    types = pd.DataFrame(0, index=rows, columns=columns, dtype=int)
    label_map = {
        "regular count": ("Count", "Regular"),
        "smooth intermittent count": ("Count", "Smooth Intermittent"),
        "lumpy intermittent count": ("Count", "Lumpy Intermittent"),
        "regular fractional": ("Fractional", "Regular"),
        "smooth intermittent fractional": ("Fractional", "Smooth Intermittent"),
        "lumpy intermittent fractional": ("Fractional", "Lumpy Intermittent"),
    }
    for name in cat_names:
        if name in label_map:
            row, col = label_map[name]
            types.at[row, col] += 1

    n_new = sum(1 for r in results if r.new)
    n_old = sum(1 for r in results if r.obsolete)
    n_stockouts = 0
    for r in results:
        starts = r.stockouts.start
        if starts is not None:
            n_stockouts += len(starts)
    anomalies = pd.Series(
        [n_new, n_stockouts, n_old], index=["New", "Stockouts", "Old"], dtype=int
    )

    return AidCatResult(
        categories=categories, types=types, anomalies=anomalies, results=results
    )
