"""Smoothers matching R's stats::lowess and stats::supsmu.

This module exposes two non-parametric smoothers that reproduce the
behaviour of R's :func:`stats::lowess` and :func:`stats::supsmu` to
machine precision. Both wrap native pybind11 extensions.

Functions
---------
- :func:`lowess` -- Cleveland's locally-weighted scatterplot smoother.
- :func:`supsmu` -- Friedman's variable-span super-smoother.

Both functions accept ``x`` and ``y`` as array-like inputs and return a
dict with two keys, ``"x"`` (sorted abscissa) and ``"y"`` (smoothed
ordinate).
"""

import numpy as np

from greybox import _native_lowess as _lowess_native  # type: ignore[attr-defined]
from greybox import _native_supsmu as _supsmu_native  # type: ignore[attr-defined]


def lowess(x, y=None, f=2 / 3, iter=3, delta=None):
    """LOWESS smoother (Locally Weighted Scatterplot Smoothing).

    Performs locally weighted polynomial regression using Cleveland's
    LOWESS algorithm with a tricube weight function and iterative
    reweighting for robustness to outliers. The implementation matches
    R's :func:`stats::lowess` to machine precision.

    Parameters
    ----------
    x : array_like
        The abscissa values. May also be a 2-D array with two columns,
        in which case the first column is used as ``x`` and the second
        as ``y``.
    y : array_like, optional
        The ordinate values. Required unless ``x`` already contains both
        columns.
    f : float, default=2/3
        Smoother span -- the fraction of points in the local
        neighbourhood used for each fit. Larger values produce smoother
        curves.
    iter : int, default=3
        Number of robustifying iterations. Each iteration downweights
        observations with large residuals from the previous fit.
    delta : float, optional
        Distance threshold for interpolation. Within ``delta`` of an
        evaluated point, the fit is linearly interpolated rather than
        recomputed. If ``None``, defaults to ``0.01 * (max(x) - min(x))``.

    Returns
    -------
    dict
        Dictionary with two keys:

        - ``"x"`` : :class:`numpy.ndarray` of sorted abscissa values.
        - ``"y"`` : :class:`numpy.ndarray` of smoothed ordinate values,
          aligned with ``"x"``.

    Notes
    -----
    The smoother fits a weighted linear regression at each point using
    a tricube weight function

    .. math::
        w(u) = (1 - |u|^3)^3

    applied to neighbours within the local span. Robustness iterations
    downweight outliers using a bisquare weight on the residuals.

    References
    ----------
    Cleveland, W. S. (1979). "Robust Locally Weighted Regression and
    Smoothing Scatterplots". *Journal of the American Statistical
    Association*, 74(368), 829-836.
    DOI: https://doi.org/10.1080/01621459.1979.10481038

    Examples
    --------
    Smooth a noisy sine wave:

    >>> import numpy as np
    >>> from greybox import lowess
    >>> rng = np.random.default_rng(0)
    >>> x = np.linspace(0, 6, 60)
    >>> y = np.sin(x) + rng.normal(0, 0.2, 60)
    >>> out = lowess(x, y, f=0.4)
    >>> out["y"].shape
    (60,)

    Pass both columns as a 2-D array:

    >>> xy = np.column_stack([x, y])
    >>> out = lowess(xy)
    """
    x = np.asarray(x, dtype=np.float64)
    if y is None:
        if x.ndim == 2 and x.shape[1] >= 2:
            y = x[:, 1].copy()
            x = x[:, 0].copy()
        else:
            raise ValueError(
                "If y is not provided, x must be a 2D array with at least 2 columns"
            )
    else:
        x = x.ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length, got {len(x)} and {len(y)}"
        )

    n = len(x)
    if n < 2:
        return {"x": x.copy(), "y": y.copy()}

    if delta is None:
        delta = -1.0

    order = np.argsort(x, kind="stable")
    x_sorted = np.ascontiguousarray(x[order])
    y_sorted = np.ascontiguousarray(y[order])

    smoothed = _lowess_native.lowess(x_sorted, y_sorted, f=f, nsteps=iter, delta=delta)
    return {"x": x_sorted, "y": np.asarray(smoothed)}


def supsmu(x, y, wt=None, span="cv", periodic=False, bass=0.0):
    """Friedman's variable-span super-smoother (SuperSmoother).

    Smooths a scatter-plot using Friedman's super-smoother algorithm.
    The smoother evaluates three running linear smoothers ("tweeters")
    with spans ``0.05``, ``0.2``, and ``0.5`` of the sample size, then
    chooses the best span at each abscissa value via cross-validated
    residuals. The implementation is a direct port of R's
    :func:`stats::supsmu` (FORTRAN ``supsmu`` from ``ppr.f``) and
    matches R to machine precision.

    Parameters
    ----------
    x : array_like
        The abscissa values. Will be sorted internally if not already
        ascending.
    y : array_like
        The ordinate values, same length as ``x``.
    wt : array_like, optional
        Per-observation weights. Default is uniform.
    span : float or {"cv"}, default="cv"
        Smoother span. Pass a float in ``(0, 1]`` for a fixed span;
        ``"cv"`` (or ``0``) selects the span automatically via
        leave-one-out cross-validation at each point.
    periodic : bool, default=False
        If ``True``, treat ``x`` as a periodic variable in ``[0, 1]``.
    bass : float, default=0.0
        Bass-tone control in ``[0, 10]``. Larger values shift the
        cross-validated span towards the smoother end of the range
        (more smoothing in noisy regions). Values outside the range
        disable the adjustment.

    Returns
    -------
    dict
        Dictionary with two keys:

        - ``"x"`` : :class:`numpy.ndarray` of the sorted distinct abscissa
          values (tied ``x`` values appear once, as in R).
        - ``"y"`` : :class:`numpy.ndarray` of smoothed ordinate values,
          aligned with ``"x"``.

    Notes
    -----
    For small samples (``n < 40``) or data with substantial serial
    correlation between observations close in ``x``, a prespecified
    fixed span (``span=0.2`` to ``span=0.4``) is recommended over
    cross-validated selection.

    The cross-validation step uses leave-one-out residuals from each of
    the three tweeters, smooths them with the medium-span smoother,
    then picks the smallest residual at each abscissa. A final smooth
    with the smallest span produces the output.

    References
    ----------
    Friedman, J. H. (1984). "A Variable Span Smoother". Technical
    Report 5 (SLAC-PUB-3477; STAN-LCS-005), Laboratory for
    Computational Statistics, Department of Statistics, Stanford
    University. https://www.osti.gov/biblio/1447470

    Examples
    --------
    Smooth count-data noise with the default cross-validated span:

    >>> import numpy as np
    >>> from greybox import supsmu
    >>> rng = np.random.default_rng(1)
    >>> x = np.arange(100, dtype=float)
    >>> y = 0.05 * x + rng.normal(0, 1, 100)
    >>> out = supsmu(x, y)
    >>> out["y"].shape
    (100,)

    Force a fixed span of 0.3:

    >>> out_fixed = supsmu(x, y, span=0.3)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length, got {len(x)} and {len(y)}"
        )

    n = len(x)
    if n < 2:
        return {"x": x.copy(), "y": y.copy()}

    if isinstance(span, str):
        if span.lower() == "cv":
            span_val = 0.0
        else:
            raise ValueError(f"span must be a float or 'cv', got {span!r}")
    else:
        span_val = float(span)

    # R: order(x, y), ties in x broken by y
    order = np.lexsort((y, x))
    x_sorted = np.ascontiguousarray(x[order])
    y_sorted = np.ascontiguousarray(y[order])
    if wt is None:
        wt_sorted = None
    else:
        wt = np.asarray(wt, dtype=np.float64).ravel()
        if len(wt) != n:
            raise ValueError(f"wt must have length {n}, got {len(wt)}")
        wt_sorted = np.ascontiguousarray(wt[order])

    smoothed = _supsmu_native.supsmu(
        x_sorted,
        y_sorted,
        wt=wt_sorted,
        span=span_val,
        bass=bass,
        periodic=bool(periodic),
    )
    # R returns one point per distinct x (the smoother averages over ties)
    unique = np.r_[True, np.diff(x_sorted) != 0]
    return {"x": x_sorted[unique], "y": np.asarray(smoothed)[unique]}
