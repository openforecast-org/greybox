"""Distribution functions for greybox.

This module contains implementations of various distributions
not available in scipy.stats, plus wrappers around scipy.stats.
"""

from .alaplace import dalaplace, palaplace, qalaplace, ralaplace
from .bcnorm import dbcnorm, pbcnorm, qbcnorm, rbcnorm
from .beta import dbeta, pbeta, qbeta, rbeta
from .binom import dbinom, pbinom, qbinom, rbinom
from .chi2 import dchi2, pchi2, qchi2, rchi2
from .exp import dexp, pexp, qexp, rexp
from .fnorm import dfnorm, pfnorm, qfnorm, rfnorm
from .gamma import dgamma, pgamma, qgamma, rgamma
from .geom import dgeom, pgeom, qgeom, rgeom
from .gnorm import dgnorm, pgnorm, qgnorm, rgnorm
from .helper import dnorm as dnorm_helper
from .helper import plogis as plogis_helper
from .helper import pnorm as pnorm_helper
from .helper import qnorm as qnorm_helper
from .invgauss import dinvgauss, pinvgauss, qinvgauss, rinvgauss
from .laplace import dlaplace, plaplace, qlaplace, rlaplace
from .lgnorm import dlgnorm, qlgnorm
from .llaplace import dllaplace, qllaplace
from .lnorm import dlnorm, plnorm, qlnorm
from .logis import dlogis, qlogis, rlogis
from .logitnorm import dlogitnorm, plogitnorm, qlogitnorm, rlogitnorm
from .ls import dls, qls
from .nbinom import dnbinom, pnbinom, qnbinom, rnbinom
from .pois import dpois, ppois, qpois, rpois
from .rectnorm import drectnorm, prectnorm, qrectnorm, rrectnorm
from .s import ds, ps, qs, rs
from .t import dt, pt, qt, rt

plogis = plogis_helper
pnorm = pnorm_helper
qnorm = qnorm_helper
dnorm = dnorm_helper

__all__ = [
    "dalaplace",
    "dbcnorm",
    "dbeta",
    "dbinom",
    "dchi2",
    "dexp",
    "dfnorm",
    "dgamma",
    "dgeom",
    "dgnorm",
    "dinvgauss",
    "dlaplace",
    "dlgnorm",
    "dllaplace",
    "dlnorm",
    "dlogis",
    "dlogitnorm",
    "dls",
    "dnbinom",
    "dnorm",
    "dpois",
    "drectnorm",
    "ds",
    "dt",
    "palaplace",
    "pbcnorm",
    "pbeta",
    "pbinom",
    "pchi2",
    "pexp",
    "pfnorm",
    "pgamma",
    "pgeom",
    "pgnorm",
    "pinvgauss",
    "plaplace",
    "plnorm",
    "plogis",
    "plogitnorm",
    "pnbinom",
    "pnorm",
    "ppois",
    "prectnorm",
    "ps",
    "pt",
    "qalaplace",
    "qbcnorm",
    "qbeta",
    "qbinom",
    "qchi2",
    "qexp",
    "qfnorm",
    "qgamma",
    "qgeom",
    "qgnorm",
    "qinvgauss",
    "qlaplace",
    "qlgnorm",
    "qllaplace",
    "qlnorm",
    "qlogis",
    "qlogitnorm",
    "qls",
    "qnbinom",
    "qnorm",
    "qpois",
    "qrectnorm",
    "qs",
    "qt",
    "ralaplace",
    "rbcnorm",
    "rbeta",
    "rbinom",
    "rchi2",
    "rexp",
    "rfnorm",
    "rgamma",
    "rgeom",
    "rgnorm",
    "rinvgauss",
    "rlaplace",
    "rlogis",
    "rlogitnorm",
    "rnbinom",
    "rpois",
    "rrectnorm",
    "rs",
    "rt",
]
