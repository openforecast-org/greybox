"""Compare the distribution functions of greybox between R and Python.

The R distributions with the fixes of this branch are sourced from the R/ folder
of the repository, so that the comparison does not depend on the installed
version of greybox. Each case is an R expression and the Python call that should
give the same numbers, including at the bounds and for degenerate parameters.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

import rpy2.robjects as ro

import greybox.distributions as gd

ro.r["library"]("greybox")
ro.r("suppressMessages(library(statmod))")
R_FOLDER = Path(__file__).resolve().parents[2] / "R"
ro.r("distFixed <- new.env(); distFixed$nan <- NaN; distFixed$inf <- Inf")
for file_name in (
    *("gnorm", "laplace", "svet", "alaplace"),
    *("fnorm", "bcnorm", "rectnorm", "logitnorm"),
):
    ro.r(f'sys.source("{R_FOLDER / file_name}.R", envir=distFixed)')

NAN, INF = np.nan, np.inf
X = np.array([-INF, -2.0, -1e-300, 0.0, 1e-300, 0.3, 0.5, 1.0, 2.0, INF, NAN])
P = np.array([-0.1, 0.0, 1e-300, 1e-17, 1e-10, 0.3, 0.5, 0.9, 1.0, 1.1, NAN])
GRID = np.linspace(-6, 9, 61)
POSITIVE = np.linspace(0.01, 12, 60)
CENTRAL = np.array([1e-4, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.9999])
GRID_R = "seq(-6, 9, length.out=61)"
POSITIVE_R = "seq(0.01, 12, length.out=60)"
X_R = "c(-Inf, -2, -1e-300, 0, 1e-300, 0.3, 0.5, 1, 2, Inf, NaN)"
P_R = "c(-0.1, 0, 1e-300, 1e-17, 1e-10, 0.3, 0.5, 0.9, 1, 1.1, NaN)"
CENTRAL_R = "c(1e-4, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.9999)"
DEGENERATE = [(0.5, 0.0), (0.5, -1.0), (NAN, 1.0), (0.5, NAN), (0.5, INF)]
# The far tails of the noncentral t and chi-squared: the arguments, the
# probabilities and their logs, and the (df, ncp) pairs. R's qt() does not return
# for a lower tail below 1e-100 at (df, ncp) = (1000, 10), so that pair is left out.
T_TAIL = np.array([-40, -5, -1, -1e-5, -1e-300, 0, 1e-300, 1e-5, 0.5, 2, 10, 100])
T_TAIL = np.r_[T_TAIL, 405.973, 1e5, 1e10]
CHISQ_TAIL = np.array([0, 1e-300, 1e-10, 0.01, 0.5, 3, 10, 50, 100, 400, 1e3, 1e4, INF])
P_TAIL = np.array([1e-300, 1e-100, 1e-20, 1e-10, 1e-4, 0.3, 0.9, 1 - 1e-10])
LOG_P_TAIL = np.array([-1000, -300, -50, -10, -1e-5, -1e-20])
T_TAIL_R, CHISQ_TAIL_R, P_TAIL_R, LOG_P_TAIL_R = (
    "c(" + ", ".join(repr(float(v)).replace("inf", "Inf") for v in values) + ")"
    for values in (T_TAIL, CHISQ_TAIL, P_TAIL, LOG_P_TAIL)
)
T_PARAMETERS = [(0.5, 1.2), (1, -2), (4.5, 1.2), (4.5, 10), (30, 39), (1e3, 50)]
T_PARAMETERS += [(1e6, 1.2), (4.5, 500)]
CHISQ_PARAMETERS = [(0, 1.2), (0.5, 1e-3), (1, 10), (3.5, 50), (30, 79.9), (30, 80)]
CHISQ_PARAMETERS += [(1e3, 500), (1e5, 5000)]
# The base wrappers with R's arguments: name, R's and Python's parameters, the
# grid of x and q and the tolerance. The noncentral t and chi-squared are R's nmath
# code, so they are held to the same tolerance as the others.
BASE = [
    ("norm", "0.3, 1.2", (0.3, 1.2), {}, GRID, GRID_R, 1e-12),
    ("logis", "0.3, 1.2", (0.3, 1.2), {}, GRID, GRID_R, 1e-12),
    ("lnorm", "0.2, 0.7", (0.2, 0.7), {}, POSITIVE, POSITIVE_R, 1e-12),
    ("gamma", "2.5, scale=1.3", (2.5, 1.3), {}, POSITIVE, POSITIVE_R, 1e-12),
    ("invgauss", "1.7, dispersion=0.4", (1.7, 0.4), {}, POSITIVE, POSITIVE_R, 1e-12),
    ("t", "4.5", (4.5,), {}, GRID, GRID_R, 1e-12),
    ("t", "4.5, 1.2", (4.5, 1.2), {}, GRID, GRID_R, 1e-12),
    ("chisq", "3.5", (3.5,), {}, POSITIVE, POSITIVE_R, 1e-12),
    ("chisq", "3.5, 2", (3.5, 2), {}, POSITIVE, POSITIVE_R, 1e-12),
    ("beta", "2, 3", (2, 3), {}, CENTRAL, CENTRAL_R, 1e-12),
    ("exp", "0.6", (0.6,), {}, POSITIVE, POSITIVE_R, 1e-12),
    ("pois", "3.7", (3.7,), {}, POSITIVE, POSITIVE_R, 1e-12),
    ("nbinom", "mu=3.7, size=1.3", (3.7, 1.3), {}, POSITIVE, POSITIVE_R, 1e-12),
    (
        "nbinom",
        "size=1.3, prob=0.3",
        (),
        {"size": 1.3, "prob": 0.3},
        POSITIVE,
        POSITIVE_R,
        1e-12,
    ),
    ("binom", "10, 0.3", (10, 0.3), {}, POSITIVE, POSITIVE_R, 1e-12),
    ("geom", "0.3", (0.3,), {}, POSITIVE, POSITIVE_R, 1e-12),
]

CASES = [
    # 1. pgnorm: the rate (1/scale)^shape, per element uniform, no clipped log
    *[
        (
            f"pgnorm({GRID_R}, 0.7, 1.3, {shape}, lower.tail={lt}, log.p={lg})",
            lambda shape=shape, lt=lt, lg=lg: gd.pgnorm(
                GRID, 0.7, 1.3, shape, lower_tail=lt == "TRUE", log_p=lg == "TRUE"
            ),
        )
        for shape in (0.3, 1, 2, 7.5, 150)
        for lt in ("TRUE", "FALSE")
        for lg in ("TRUE", "FALSE")
    ],
    (
        "pgnorm(c(1, 1, 3), 0, 1.3, c(1, 150, 2))",
        lambda: gd.pgnorm([1, 1, 3], 0, 1.3, [1, 150, 2]),
    ),
    # 2. qgnorm with log.p and the upper tail
    *[
        (
            f"qgnorm(log({CENTRAL_R}), 0.7, 1.3, {shape}, lower.tail=FALSE, log.p=TRUE)",
            lambda shape=shape: gd.qgnorm(
                np.log(CENTRAL), 0.7, 1.3, shape, lower_tail=False, log_p=True
            ),
        )
        for shape in (0.3, 1, 2, 7.5, 150)
    ],
    # 3. qgnorm, qlaplace and qs with degenerate scales
    *[
        (
            f"qgnorm({P_R}, {mu}, {scale}, 2)",
            lambda m=mu, s=scale: gd.qgnorm(P, m, s, 2),
        )
        for mu, scale in DEGENERATE
    ],
    *[
        (f"qlaplace({P_R}, {mu}, {scale})", lambda m=mu, s=scale: gd.qlaplace(P, m, s))
        for mu, scale in DEGENERATE
    ],
    *[
        (f"qs({P_R}, {mu}, {scale})", lambda m=mu, s=scale: gd.qs(P, m, s))
        for mu, scale in DEGENERATE
    ],
    # 4. The tails of qlaplace, qs, plaplace and ps without cancellation
    (f"qlaplace({P_R}, 0.7, 1.3)", lambda: gd.qlaplace(P, 0.7, 1.3)),
    (f"qs({P_R}, 0.7, 1.3)", lambda: gd.qs(P, 0.7, 1.3)),
    ("plaplace(c(-40, 40, -800, 0.7))", lambda: gd.plaplace([-40, 40, -800, 0.7])),
    ("ps(c(-1e4, 1e4, -1e6, 0))", lambda: gd.ps([-1e4, 1e4, -1e6, 0])),
    *[
        (f"plaplace({X_R}, {mu}, {scale})", lambda m=mu, s=scale: gd.plaplace(X, m, s))
        for mu, scale in DEGENERATE
    ],
    # 5. qfnorm as the root of pfnorm, for any mu, vectorised
    *[
        (
            f"qfnorm({CENTRAL_R}, {mu}, 1.3)",
            lambda m=mu: gd.qfnorm(CENTRAL, m, 1.3),
            1e-8,
        )
        for mu in (0.5, -1.2, 3, -20, 0)
    ],
    (
        (
            "qfnorm(c(0.05, 0.2, 0.4, 0.6, 0.8, 0.95), c(0.1, 0.5, 1, -0.3, 2, 0), "
            "c(0.5, 1, 1.5, 2, 0.8, 1.2))"
        ),
        lambda: gd.qfnorm(
            [0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
            [0.1, 0.5, 1, -0.3, 2, 0],
            [0.5, 1, 1.5, 2, 0.8, 1.2],
        ),
        1e-10,
    ),
    (
        "qfnorm(c(-0.1, 0, 0.5, 1, 1.1, NaN), 0.5, 1)",
        lambda: gd.qfnorm([-0.1, 0, 0.5, 1, 1.1, NAN], 0.5, 1),
        1e-10,
    ),
    # 6, 8, 9. No floors, the logs analytically, -Inf below the support
    *[
        (
            f"dbcnorm({X_R}, 0.4, 0.8, {lam}, log={lg})",
            lambda lam=lam, lg=lg: gd.dbcnorm(X, 0.4, 0.8, lam, log=lg == "TRUE"),
        )
        for lam in (0, 1, 0.3, 0.5, -0.4, 2)
        for lg in ("TRUE", "FALSE")
    ],
    (
        "dbcnorm(c(1e4, 1e6, 1e-30, 1e30), 0, 0.5, 0.3, log=TRUE)",
        lambda: gd.dbcnorm([1e4, 1e6, 1e-30, 1e30], 0, 0.5, 0.3, log=True),
    ),
    *[
        (
            f"dfnorm({X_R}, {mu}, 1.3, log={lg})",
            lambda mu=mu, lg=lg: gd.dfnorm(X, mu, 1.3, log=lg == "TRUE"),
        )
        for mu in (0.5, -1.2, 3, 0)
        for lg in ("TRUE", "FALSE")
    ],
    ("dfnorm(c(40, 60), log=TRUE)", lambda: gd.dfnorm([40, 60], log=True)),
    (
        f"dfnorm({POSITIVE_R}, -1.2, 1.3, log=TRUE)",
        lambda: gd.dfnorm(POSITIVE, -1.2, 1.3, log=True),
    ),
    *[
        (
            f"dalaplace({GRID_R}, 0.7, 1.3, {alpha}, log=TRUE)",
            lambda a=alpha: gd.dalaplace(GRID, 0.7, 1.3, a, log=True),
        )
        for alpha in (0.2, 0.5, 0.85)
    ],
    (
        "dalaplace(c(800, -2000), alpha=0.3, log=TRUE)",
        lambda: gd.dalaplace([800, -2000], alpha=0.3, log=True),
    ),
    *[
        (
            f"drectnorm({X_R}, {mu}, 1.3, log={lg})",
            lambda mu=mu, lg=lg: gd.drectnorm(X, mu, 1.3, log=lg == "TRUE"),
        )
        for mu in (0.5, -1.2, 40)
        for lg in ("TRUE", "FALSE")
    ],
    ("drectnorm(c(40, -1), log=TRUE)", lambda: gd.drectnorm([40, -1], log=True)),
    # 7. dlogitnorm without clipping
    *[
        (
            f"dlogitnorm(c({X_R}, 1e-6, 1e-12, 1 - 1e-12), {mu}, 1.3, log={lg})",
            lambda mu=mu, lg=lg: gd.dlogitnorm(
                np.r_[X, 1e-6, 1e-12, 1 - 1e-12], mu, 1.3, log=lg == "TRUE"
            ),
        )
        for mu in (0.5, -1.2)
        for lg in ("TRUE", "FALSE")
    ],
    # 11. pbcnorm and plogitnorm at the bounds and at NaN
    *[
        (
            f"pbcnorm({X_R}, 0.4, 0.8, {lam})",
            lambda lam=lam: gd.pbcnorm(X, 0.4, 0.8, lam),
        )
        for lam in (0, 0.5, -0.5, 2)
    ],
    (f"plogitnorm({X_R}, 0.4, 0.8)", lambda: gd.plogitnorm(X, 0.4, 0.8)),
    # 12, 13. qbcnorm and qlogitnorm at the bounds and with a zero scale
    *[
        (
            f"qbcnorm({P_R}, {mu}, {scale}, {lam})",
            lambda m=mu, s=scale, lam=lam: gd.qbcnorm(P, m, s, lam),
        )
        for lam in (0, 0.5, -0.5, 2)
        for mu, scale in [(0.4, 0.8), (-1, 1), *DEGENERATE]
    ],
    *[
        (
            f"qlogitnorm({P_R}, {mu}, {scale})",
            lambda m=mu, s=scale: gd.qlogitnorm(P, m, s),
        )
        for mu, scale in [(0.4, 0.8), (800, 1), *DEGENERATE]
    ],
    # 14. Degenerate parameters
    *[
        (
            f"{name}({X_R}, {mu}, {scale})",
            lambda name=name, m=mu, s=scale: getattr(gd, name)(X, m, s),
        )
        for name in ("drectnorm", "prectnorm", "pfnorm", "dfnorm", "dalaplace", "ps")
        for mu, scale in DEGENERATE
    ],
    *[
        (
            f"{name}({P_R}, {mu}, {scale})",
            lambda name=name, m=mu, s=scale: getattr(gd, name)(P, m, s),
        )
        for name in ("qrectnorm", "qalaplace")
        for mu, scale in DEGENERATE
    ],
    *[
        (
            f"dbcnorm({X_R}, {mu}, {scale}, {lam}, log={lg})",
            lambda m=mu, s=scale, lam=lam, lg=lg: gd.dbcnorm(
                X, m, s, lam, log=lg == "TRUE"
            ),
        )
        for lam in (0, 1, 0.5)
        for mu, scale in DEGENERATE
        for lg in ("TRUE", "FALSE")
    ],
    *[
        (
            f"dlnorm({X_R}, {mu}, {scale}, log={lg})",
            lambda m=mu, s=scale, lg=lg: gd.dlnorm(X, m, s, log=lg == "TRUE"),
        )
        for mu, scale in [(0.0, 0.0), *DEGENERATE]
        for lg in ("TRUE", "FALSE")
    ],
    # 15. The base wrappers at the bounds
    (
        "dinvgauss(c(1e-300, 1, Inf), mean=1.7, dispersion=0.4, log=TRUE)",
        lambda: gd.dinvgauss([1e-300, 1, INF], 1.7, 0.4, log=True),
    ),
    (
        "dinvgauss(c(1e-300, 1, Inf), mean=1.7, dispersion=0.4)",
        lambda: gd.dinvgauss([1e-300, 1, INF], 1.7, 0.4),
    ),
    ("dchisq(c(1, Inf), 3.5)", lambda: gd.dchisq([1, INF], 3.5)),
    ("dpois(c(1, Inf), 3.7, log=TRUE)", lambda: gd.dpois([1, INF], 3.7, log=True)),
    (
        "dnbinom(c(1, Inf), mu=3.7, size=1.3)",
        lambda: gd.dnbinom([1, INF], mu=3.7, size=1.3),
    ),
    (
        "dnbinom(c(0, 2, 2.5), mu=2, size=Inf)",
        lambda: gd.dnbinom([0, 2, 2.5], mu=2, size=INF),
    ),
    ("qpois(c(0, 0.5, 1), 3.7)", lambda: gd.qpois([0, 0.5, 1], 3.7)),
    (
        "qnbinom(c(0, 0.5, 1), mu=3.7, size=1.3)",
        lambda: gd.qnbinom([0, 0.5, 1], mu=3.7, size=1.3),
    ),
    ("qbinom(c(0, 0.5, 1), 10, 0.3)", lambda: gd.qbinom([0, 0.5, 1], 10, 0.3)),
    ("qgeom(c(0, 0.5), 0.3)", lambda: gd.qgeom([0, 0.5], 0.3)),
    ("dgeom(c(0, 1, 2.5), 0.3)", lambda: gd.dgeom([0, 1, 2.5], 0.3)),
    # 16. dnbinom in the mu form (R 4.3 itself is within 6e-10 at size=1e8)
    *[
        (
            f"dnbinom(0:39, mu=5, size={size}, log=TRUE)",
            lambda size=size: gd.dnbinom(np.arange(40.0), 5, size, log=True),
            1e-14 if size < 1e3 else 1e-9,
        )
        for size in (0.8, 20, 1e3, 1e6, 1e8)
    ],
    # 17. pnorm and plogis upper tails
    *[
        (
            f"{name}(c(seq(-40, 40, 1), NaN), 0.3, 1.2, lower.tail=FALSE, log.p={lg})",
            lambda name=name, lg=lg: getattr(gd, name)(
                np.r_[np.arange(-40.0, 41), NAN],
                0.3,
                1.2,
                lower_tail=False,
                log_p=lg == "TRUE",
            ),
        )
        for name in ("pnorm", "plogis")
        for lg in ("TRUE", "FALSE")
    ],
    # 18. A scalar q with a vector of mu
    *[
        (
            f"{name}(0.3, c(0.1, 0.5, 1, -0.3, 2, 0))",
            lambda name=name: getattr(gd, name)(0.3, [0.1, 0.5, 1, -0.3, 2, 0]),
        )
        for name in ("pbcnorm", "plogitnorm", "pgnorm", "ps", "plaplace", "prectnorm")
    ],
    # 19. The base wrappers with R's arguments: the densities with log, the
    # probabilities and quantiles with lower.tail and log.p
    *[
        (
            f"d{name}({x_r}, {args_r}, log={lg})",
            lambda name=name, a=args, k=kwargs, x=x, lg=lg: getattr(gd, "d" + name)(
                x, *a, log=lg == "TRUE", **k
            ),
            rtol,
        )
        for name, args_r, args, kwargs, x, x_r, rtol in BASE
        for lg in ("TRUE", "FALSE")
    ],
    *[
        (
            f"p{name}({x_r}, {args_r}, lower.tail={lt}, log.p={lg})",
            lambda name=name, a=args, k=kwargs, x=x, lt=lt, lg=lg: getattr(
                gd, "p" + name
            )(x, *a, lower_tail=lt == "TRUE", log_p=lg == "TRUE", **k),
            rtol,
        )
        for name, args_r, args, kwargs, x, x_r, rtol in BASE
        for lt in ("TRUE", "FALSE")
        for lg in ("TRUE", "FALSE")
    ],
    *[
        (
            f"q{name}({p_r}, {args_r}, lower.tail={lt}, log.p={lg})",
            lambda name=name, a=args, k=kwargs, p=p, lt=lt, lg=lg: getattr(
                gd, "q" + name
            )(p, *a, lower_tail=lt == "TRUE", log_p=lg == "TRUE", **k),
            rtol,
        )
        for name, args_r, args, kwargs, x, x_r, rtol in BASE
        for lt in ("TRUE", "FALSE")
        for lg, p, p_r in (
            ("FALSE", CENTRAL, CENTRAL_R),
            ("TRUE", np.log(CENTRAL), f"log({CENTRAL_R})"),
        )
    ],
    # 20. The noncentral Beta. R's pnbeta is accurate to 1e-9 only (it gives
    # 5.4e-10 for an upper tail of 9.7e-12), so the reference is the Poisson
    # mixture of R's central pbeta, its log from the other tail when the tail is
    # close to one, and the quantiles go back to their p
    *[
        (
            f"dbeta({CENTRAL_R}, 2, {shape2}, {ncp}, log={lg})",
            lambda shape2=shape2, ncp=ncp, lg=lg: gd.dbeta(
                CENTRAL, 2, shape2, ncp, log=lg == "TRUE"
            ),
        )
        for shape2, ncp in ((3, 1.5), (0.5, 30))
        for lg in ("TRUE", "FALSE")
    ],
    *[
        (
            (
                f"local({{tails <- sapply(c({lt}, !{lt}), function(lt) sapply("
                f"{CENTRAL_R}, function(q) sum(dpois(0:300, {ncp} / 2) * "
                "pbeta(q, 2 + 0:300, 3, lower.tail=lt)))); "
                f"if ({lg}) ifelse(tails[, 1] < 0.5, log(tails[, 1]), "
                "log1p(-tails[, 2])) else tails[, 1]})"
            ),
            lambda ncp=ncp, lt=lt, lg=lg: gd.pbeta(
                CENTRAL, 2, 3, ncp, lower_tail=lt == "TRUE", log_p=lg == "TRUE"
            ),
        )
        for ncp in (1.5, 30)
        for lt in ("TRUE", "FALSE")
        for lg in ("TRUE", "FALSE")
    ],
    *[
        (
            CENTRAL_R,
            lambda ncp=ncp, lt=lt: gd.pbeta(
                gd.qbeta(CENTRAL, 2, 3, ncp, lower_tail=lt), 2, 3, ncp, lower_tail=lt
            ),
        )
        for ncp in (1.5, 30)
        for lt in (True, False)
    ],
    # 21. The noncentral t and chi-squared in the far tails, R's nmath to the bit:
    # p down to 1e-300 and its log down to -1000, both tails, ncp up to 5000
    *[
        (
            f"d{name}({x_r}, {df}, {ncp}, log={lg})",
            lambda name=name, x=x, df=df, ncp=ncp, lg=lg: getattr(gd, "d" + name)(
                x, df, ncp, log=lg == "TRUE"
            ),
        )
        for name, x, x_r, parameters in (
            ("t", T_TAIL, T_TAIL_R, T_PARAMETERS),
            ("chisq", CHISQ_TAIL, CHISQ_TAIL_R, CHISQ_PARAMETERS),
        )
        for df, ncp in parameters
        for lg in ("TRUE", "FALSE")
    ],
    *[
        (
            f"p{name}({x_r}, {df}, {ncp}, lower.tail={lt}, log.p={lg})",
            lambda name=name, x=x, df=df, ncp=ncp, lt=lt, lg=lg: getattr(
                gd, "p" + name
            )(x, df, ncp, lower_tail=lt == "TRUE", log_p=lg == "TRUE"),
        )
        for name, x, x_r, parameters in (
            ("t", T_TAIL, T_TAIL_R, T_PARAMETERS),
            ("chisq", CHISQ_TAIL, CHISQ_TAIL_R, CHISQ_PARAMETERS),
        )
        for df, ncp in parameters
        for lt in ("TRUE", "FALSE")
        for lg in ("TRUE", "FALSE")
    ],
    *[
        (
            f"q{name}({p_r}, {df}, {ncp}, lower.tail={lt}, log.p={lg})",
            lambda name=name, p=p, df=df, ncp=ncp, lt=lt, lg=lg: getattr(
                gd, "q" + name
            )(p, df, ncp, lower_tail=lt == "TRUE", log_p=lg == "TRUE"),
        )
        for name, parameters in (("t", T_PARAMETERS), ("chisq", CHISQ_PARAMETERS))
        for df, ncp in parameters
        for lt in ("TRUE", "FALSE")
        for lg, p, p_r in (
            ("FALSE", P_TAIL, P_TAIL_R),
            ("TRUE", LOG_P_TAIL, LOG_P_TAIL_R),
        )
    ],
    (
        "qt(1e-10, 4.5, ncp=1.2, lower.tail=FALSE)",
        lambda: gd.qt(1e-10, 4.5, 1.2, lower_tail=False),
    ),
    *[
        (
            f"{name}(0.5, c(1, 4.5, 30), c(1.2, 10, 0), lower.tail=FALSE)",
            lambda name=name: getattr(gd, name)(
                0.5, [1, 4.5, 30], [1.2, 10, 0], lower_tail=False
            ),
        )
        for name in ("pt", "qt", "pchisq", "qchisq")
    ],
]


@pytest.mark.parametrize("case", CASES, ids=[case[0][:80] for case in CASES])
def test_the_distributions_are_those_of_r(case):
    expression, python, rtol = (*case, 1e-12)[:3]
    expected = np.asarray(ro.r(f"suppressWarnings(evalq({expression}, distFixed))"))
    assert_allclose(python(), expected, rtol=rtol, atol=0, equal_nan=True)
