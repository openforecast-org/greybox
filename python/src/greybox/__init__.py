"""Greybox - Toolbox for model building and forecasting.

All public functions and classes are re-exported at the top level so
users can write::

    from greybox import ALM, aid, lowess, formula, ham, association, ...

without having to remember which submodule each name lives in.

Submodules remain importable in the usual way (``import greybox.distributions``,
``from greybox.smoothers import lowess``), but the top-level package always
prefers the *function* with a given name over a submodule that shares it.
For example, ``from greybox import association`` returns the function
:func:`greybox.association.association`, not the submodule.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("greybox")
except PackageNotFoundError:
    __version__ = "unknown"

# Submodules that do not collide with a public function name and are kept
# importable as ``greybox.<submodule>``.
from . import data as data
from . import distributions as distributions
from . import point_measures as point_measures
from . import quantile_measures as quantile_measures
from . import rolling as rolling
from . import smoothers as smoothers

# Automatic Identification of Demand
from .aid import AidCatResult, AidResult, AidType, Stockouts, aid, aid_cat

# Core fitting and prediction
from .alm import ALM, PredictionResult

# Association / correlation
from .association import association, determination, mcor, pcor

# Datasets
from .data import mtcars

# Diagnostics
from .diagnostics import OutlierResult, outlier_dummy

# Re-export every distribution function (dnorm, pnorm, qnorm, rnorm, ds, dt,
# dgeom, dpois, ...) at the top level so users can do ``from greybox import
# dnorm`` directly.
from .distributions import *
from .distributions import __all__ as _distribution_names

# Formula interface
from .formula import expand_formula, formula

# Half-moment measures
from .hm import asymmetry, cextremity, extremity, ham, hm, mre

# Point accuracy measures
from .point_measures import (
    gmrae,
    mae,
    mape,
    mase,
    me,
    measures,
    mpe,
    mse,
    rame,
    rmae,
    rmse,
    rmsse,
    rrmse,
    same,
    sce,
    smse,
    spis,
)

# Point likelihoods
from .pointlik import point_lik, point_lik_cumulative

# Quantile / interval scoring measures
from .quantile_measures import mis, pinball, rmis, smis

# Regression for Multiple Comparison with the Best
from .rmcb import RMCBResult, rmcb

# Rolling-origin cross-validation
from .rolling import RollingOriginResult, rolling_origin

# Model selection
from .selection import CALM, CALMResult, stepwise

# Smoothers
from .smoothers import lowess, supsmu

# Seasonality, Trend, and Irregular Contribution Kit
from .stick import StickResult, stick

# Transforms
from .transforms import bc_transform, bc_transform_inv, mean_fast

# Variable processing
from .xreg import (
    B,
    multipliers,
    temporal_dummy,
    xreg_expander,
    xreg_multiplier,
    xreg_transformer,
)

__all__ = [
    "__version__",
    # Submodules
    "data",
    "distributions",
    "point_measures",
    "quantile_measures",
    "rolling",
    "smoothers",
    # Core fitting
    "ALM",
    "PredictionResult",
    # Formula
    "formula",
    "expand_formula",
    # Selection
    "stepwise",
    "CALM",
    "CALMResult",
    # Variable processing
    "B",
    "multipliers",
    "xreg_expander",
    "xreg_multiplier",
    "xreg_transformer",
    "temporal_dummy",
    # Transforms
    "bc_transform",
    "bc_transform_inv",
    "mean_fast",
    # Rolling
    "rolling_origin",
    "RollingOriginResult",
    # Datasets
    "mtcars",
    # Point likelihoods
    "point_lik",
    "point_lik_cumulative",
    # AID
    "aid",
    "aid_cat",
    "AidResult",
    "AidCatResult",
    "AidType",
    "Stockouts",
    # Smoothers
    "lowess",
    "supsmu",
    # Association
    "pcor",
    "mcor",
    "association",
    "determination",
    # Half-moments
    "hm",
    "ham",
    "asymmetry",
    "extremity",
    "cextremity",
    "mre",
    # Accuracy measures
    "measures",
    "me",
    "mae",
    "mse",
    "rmse",
    "mpe",
    "mape",
    "mase",
    "rmsse",
    "same",
    "rmae",
    "rrmse",
    "rame",
    "smse",
    "spis",
    "sce",
    "gmrae",
    # Quantile measures
    "pinball",
    "mis",
    "smis",
    "rmis",
    # Diagnostics
    "outlier_dummy",
    "OutlierResult",
    "stick",
    "StickResult",
    "rmcb",
    "RMCBResult",
] + list(_distribution_names)
