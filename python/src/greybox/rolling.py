"""Rolling origin evaluation.

This module provides functions for rolling origin cross-validation
for time series models.
"""

import inspect
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np


class RollingOriginResult:
    """Result of rolling origin evaluation.

    Attributes
    ----------
    actuals : np.ndarray
        The original data.
    holdout : np.ndarray
        Matrix of actual holdout values, shape (h, origins). NaN where
        h_actual < h (co=False, near end of series).
    mean : np.ndarray
        Point forecasts, shape (h, origins).
    lower : np.ndarray, optional
        Lower interval bounds, shape (h, origins). Present if call returned
        'lower' key/attribute.
    upper : np.ndarray, optional
        Upper interval bounds, shape (h, origins). Present if call returned
        'upper' key/attribute.
    origins : int
        Number of rolling origins evaluated.
    h : int
        Forecast horizon.
    _fields : list[str]
        Names of all forecast fields stored (always includes 'mean').
    """

    def __init__(
        self,
        actuals: np.ndarray,
        holdout: np.ndarray,
        origins: int,
        h: int,
        fields: dict,
    ):
        self.actuals = actuals
        self.holdout = holdout
        self.origins = origins
        self.h = h
        self._fields = list(fields.keys())
        for name, arr in fields.items():
            setattr(self, name, arr)

    def __str__(self) -> str:
        # Detect constant vs decreasing holdout from NaN pattern
        has_nan = np.any(np.isnan(self.holdout))
        window_type = "decreasing" if has_nan else "constant"
        lines = [
            f"Rolling origin with {window_type} holdout was done.",
            f" Forecast horizon: {self.h}",
            f" Number of origins: {self.origins}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


def _parse_call_result(result: Any, h_actual: int) -> dict:
    """Convert a call() return value to {field: array} dict.

    Supported return types:
    - np.ndarray or list           → {"mean": array}
    - dict with string keys        → one entry per key
    - object with .mean attr       → {"mean": arr, "lower": arr, ...}
    - 3-tuple (mean, lower, upper) → {"mean", "lower", "upper"}
    """

    def _to_arr(v):
        return np.asarray(v, dtype=float)

    if isinstance(result, (np.ndarray, list)):
        return {"mean": _to_arr(result)}

    if isinstance(result, tuple):
        if len(result) == 3:
            mean, lower, upper = result
            return {
                "mean": _to_arr(mean),
                "lower": _to_arr(lower),
                "upper": _to_arr(upper),
            }
        elif len(result) == 2:
            mean, lower = result
            return {"mean": _to_arr(mean), "lower": _to_arr(lower)}
        else:
            return {"mean": _to_arr(result[0])}

    if isinstance(result, dict):
        return {k: _to_arr(v) for k, v in result.items()}

    # Object with attributes
    out = {}
    for attr in ("mean", "lower", "upper"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if val is not None:
                out[attr] = _to_arr(val)
    if out:
        return out

    # Fallback: try to coerce directly
    return {"mean": _to_arr(result)}


def rolling_origin(
    data: np.ndarray,
    h: int = 10,
    origins: int = 10,
    step: int = 1,
    ci: bool = False,
    co: bool = True,
    call: Callable | None = None,
    silent: bool = True,
    # deprecated
    model_fn: Callable | None = None,
    predict_fn: Callable | None = None,
) -> RollingOriginResult:
    """Rolling Origin Evaluation.

    Produces rolling origin forecasts using a single callable that accepts
    training data and horizon h. This is the standard approach for
    evaluating forecast accuracy on time series.

    Parameters
    ----------
    data : np.ndarray
        Time series data (1-D).
    h : int, default=10
        Forecast horizon (number of steps ahead).
    origins : int, default=10
        Number of rolling origins.
    step : int, default=1
        Number of observations to advance between origins.
    ci : bool, default=False
        If False (default), expanding window — training set grows each
        origin. If True, sliding window — training set has constant size.
    co : bool, default=True
        If True (default), constant holdout: all origins forecast exactly
        h steps. If False, decreasing holdout: later origins may forecast
        fewer than h steps (when fewer observations remain).
    call : callable
        ``call(data, h, **optional) -> forecasts``

        ``data`` is the training slice (np.ndarray), ``h`` is the horizon.
        The function may optionally accept keyword arguments ``counti``,
        ``counto``, and/or ``countf`` (index arrays into the original
        series) — they are injected automatically if present in the
        signature.

        Return value can be:

        - ``np.ndarray`` of length h → stored as ``result.mean``
        - ``dict`` with string keys → one attribute per key
        - object with ``.mean`` / ``.lower`` / ``.upper`` attributes
        - 3-tuple ``(mean, lower, upper)``

    silent : bool, default=True
        If False, print progress information per origin.
    model_fn : callable, deprecated
        Old API: function ``train_data -> fitted_model``.
    predict_fn : callable, deprecated
        Old API: function ``(model, h) -> forecasts``.

    Returns
    -------
    RollingOriginResult
        - ``.actuals`` — original data
        - ``.holdout`` — actual values, shape ``(h, origins)``, NaN-padded
        - ``.mean`` — point forecasts, shape ``(h, origins)``
        - ``.lower`` / ``.upper`` — interval bounds (if returned by call)
        - ``.origins`` — number of origins evaluated
        - ``.h`` — forecast horizon
        - ``._fields`` — list of forecast field names

    Examples
    --------
    >>> import numpy as np
    >>> from greybox import rolling_origin
    >>> np.random.seed(42)
    >>> y = np.cumsum(np.random.randn(100))
    >>> result = rolling_origin(y, h=5, origins=10,
    ...     call=lambda data, h: np.full(h, data.mean()))
    >>> print(result)
    Rolling origin with constant holdout was done.
     Forecast horizon: 5
     Number of origins: 10
    >>> result.mean.shape
    (5, 10)
    """
    # --- backward compatibility ---
    if call is None and (model_fn is not None or predict_fn is not None):
        warnings.warn(
            "model_fn and predict_fn are deprecated. "
            "Use a single call=lambda data, h: ... instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if model_fn is None or predict_fn is None:
            raise ValueError("Both model_fn and predict_fn must be provided together.")
        _mfn = model_fn
        _pfn = predict_fn

        def call(data, h):
            return _pfn(_mfn(data), h)

    if call is None:
        raise ValueError(
            "call must be provided. "
            "Example: call=lambda data, h: np.full(h, data.mean())"
        )

    data = np.asarray(data, dtype=float)
    n = len(data)

    # R formula: obsInSample = obs - (origins*step + (h-step)*co)
    obs_in_sample = n - (origins * step + (h - step) * int(co))
    if obs_in_sample < 1:
        raise ValueError(
            f"obs_in_sample={obs_in_sample} < 1. Reduce origins, h, or step (n={n})."
        )

    # Inspect call signature for optional index kwargs
    try:
        sig = inspect.signature(call)
        _call_params = set(sig.parameters.keys())
    except (ValueError, TypeError):
        _call_params = set()

    _wants_counti = "counti" in _call_params
    _wants_counto = "counto" in _call_params
    _wants_countf = "countf" in _call_params

    # Allocate output matrices (h, origins), NaN-filled
    holdout_mat = np.full((h, origins), np.nan)
    # Forecast matrices allocated after first call to know fields
    fields_data: dict | None = None

    n_done = 0
    for i in range(origins):
        # Effective horizon for this origin
        if co:
            h_actual = h
        else:
            h_actual = min(h, step * (origins - i))

        # In-sample index range
        train_end = obs_in_sample + step * i
        if not ci:
            # expanding: [0, train_end)
            counti = np.arange(0, train_end)
        else:
            # sliding: constant window of size obs_in_sample
            counti = np.arange(step * i, train_end)

        # Out-of-sample index range
        counto = np.arange(train_end, train_end + h_actual)
        countf = np.concatenate([counti, counto])

        if train_end > n or (len(counto) > 0 and counto[-1] >= n):
            if not silent:
                print(f"Origin {i + 1}: out of bounds, stopping.")
            break

        train_data = data[counti]

        # Build kwargs for call
        extra_kwargs = {}
        if _wants_counti:
            extra_kwargs["counti"] = counti
        if _wants_counto:
            extra_kwargs["counto"] = counto
        if _wants_countf:
            extra_kwargs["countf"] = countf

        try:
            raw = call(train_data, h_actual, **extra_kwargs)
        except Exception as e:  # noqa: BLE001 - any model failure ends the loop
            if not silent:
                print(f"Origin {i + 1}/{origins} failed: {e}")
            break

        parsed = _parse_call_result(raw, h_actual)

        # Ensure all arrays have length h_actual; pad to h with NaN
        for key, arr in parsed.items():
            if len(arr) < h:
                padded = np.full(h, np.nan)
                padded[: len(arr)] = arr[:h_actual]
                parsed[key] = padded
            else:
                parsed[key] = arr[:h]

        # Initialise forecast matrices on first successful call
        if fields_data is None:
            fields_data = {k: np.full((h, origins), np.nan) for k in parsed}

        for key, arr in parsed.items():
            if key in fields_data:
                fields_data[key][:, i] = arr

        # Fill holdout
        actual_h = data[counto]
        holdout_mat[:h_actual, i] = actual_h

        if not silent:
            print(
                f"Origin {i + 1}/{origins}: "
                f"train=[{counti[0]}:{counti[-1] + 1}], "
                f"holdout=[{counto[0]}:{counto[-1] + 1}], "
                f"h_actual={h_actual}"
            )

        n_done += 1

    if fields_data is None:
        # No origins succeeded; return empty result with mean only
        fields_data = {"mean": np.full((h, origins), np.nan)}

    return RollingOriginResult(
        actuals=data,
        holdout=holdout_mat[:, :n_done],
        origins=n_done,
        h=h,
        fields={k: v[:, :n_done] for k, v in fields_data.items()},
    )
