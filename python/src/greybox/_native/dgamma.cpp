// Gamma log-density matching R's stats::dgamma to near machine precision.
//
// Adapted from R's nmath sources:
//   src/nmath/dgamma.c, dpois.c, stirlerr.c, bd0.c
//
// SciPy's gamma.logpdf evaluates (a-1)*log(x) - x/s - lgamma(a) - a*log(s)
// directly, which cancels catastrophically once the shape grows: at a shape of
// 143 it loses about 1.4e-13, enough to move an optimiser off a boundary.
// R instead uses Loader's saddle-point form,
//
//     log f(x) = -0.5*log(2*pi*(a-1)) - stirlerr(a-1) - bd0(a-1, x/s) - log(s)
//
// where stirlerr is the Stirling series remainder and bd0 the "deviance part",
// both written so that the near-cancelling terms are never formed.
//
// References:
//   Loader, C. (2000) "Fast and Accurate Computation of Binomial
//   Probabilities". Available from the R sources.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

namespace {

constexpr double kLnSqrt2Pi = 0.918938533204672741780329736406;
constexpr double kTwoPi = 6.283185307179586476925286766559;

// Coefficients of the Stirling series: 1/12, 1/360, 1/1260, 1/1680, 1/1188.
constexpr double kS0 = 0.083333333333333333333;
constexpr double kS1 = 0.00277777777777777777778;
constexpr double kS2 = 0.00079365079365079365079365;
constexpr double kS3 = 0.000595238095238095238095238;
constexpr double kS4 = 0.0008417508417508417508417508;

// stirlerr(n) for n = 0, 0.5, 1.0, ..., 15.0, tabulated because the series is
// least accurate exactly where the arguments are smallest.
constexpr double kSferrHalves[31] = {
    0.0,
    0.1534264097200273452913848,   0.0810614667953272582196702,
    0.0548141210519176538961390,   0.0413406959554092940938221,
    0.03316287351993628748511048,  0.02767792568499833914878929,
    0.02374616365629749597132920,  0.02079067210376509311152277,
    0.01848845053267318523077934,  0.01664469118982119216319487,
    0.01513497322191737887351255,  0.01387612882307074799874573,
    0.01281046524292022692424986,  0.01189670994589177009505572,
    0.01110455975820691732662991,  0.010411265261972096497478567,
    0.009799416126158803298389475, 0.009255462182712732917728637,
    0.008768700134139385462952823, 0.008330563433362871256469318,
    0.007934114564314020547248100, 0.007573675487951840794972024,
    0.007244554301320383179543912, 0.006942840107209529865664152,
    0.006665247032707682442354394, 0.006408994188004207068439631,
    0.006171712263039457647532867, 0.005951370112758847735624416,
    0.005746216513010115682023589, 0.005554733551962801371038690};

// log(n!) - [log(sqrt(2*pi*n)) + n*log(n) - n], the error in Stirling's
// approximation. R/nmath/stirlerr.c.
inline double stirlerr(double n) {
    if (n <= 15.0) {
        const double nn = n + n;
        if (nn == static_cast<double>(static_cast<int>(nn))) {
            return kSferrHalves[static_cast<int>(nn)];
        }
        return std::lgamma(n + 1.0) - (n + 0.5) * std::log(n) + n - kLnSqrt2Pi;
    }
    const double nn = n * n;
    if (n > 500.0) return (kS0 - kS1 / nn) / n;
    if (n > 80.0) return (kS0 - (kS1 - kS2 / nn) / nn) / n;
    if (n > 35.0) return (kS0 - (kS1 - (kS2 - kS3 / nn) / nn) / nn) / n;
    return (kS0 - (kS1 - (kS2 - (kS3 - kS4 / nn) / nn) / nn) / nn) / n;
}

// x*log(x/np) + np - x, evaluated by a series when x and np are close so that
// the two nearly-equal terms are never subtracted. R/nmath/bd0.c.
inline double bd0(double x, double np) {
    if (!std::isfinite(x) || !std::isfinite(np) || np == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::fabs(x - np) < 0.1 * (x + np)) {
        double v = (x - np) / (x + np);
        double s = (x - np) * v;
        if (std::fabs(s) < std::numeric_limits<double>::min()) return s;
        double ej = 2.0 * x * v;
        v = v * v;
        for (int j = 1; j < 1000; ++j) {
            ej *= v;
            const double s1 = s + ej / ((j << 1) + 1);
            if (s1 == s) return s1;  // converged to the last bit
            s = s1;
        }
    }
    return x * std::log(x / np) + np - x;
}

// log of the Poisson density without the usual argument checks.
inline double dpois_raw_log(double x, double lambda) {
    if (lambda == 0.0) return (x == 0.0) ? 0.0 : -std::numeric_limits<double>::infinity();
    if (!std::isfinite(lambda)) return -std::numeric_limits<double>::infinity();
    if (x < 0.0) return -std::numeric_limits<double>::infinity();
    if (x <= lambda * std::numeric_limits<double>::min()) return -lambda;
    if (lambda < x * std::numeric_limits<double>::min()) {
        if (!std::isfinite(x)) return -std::numeric_limits<double>::infinity();
        return -lambda + x * std::log(lambda) - std::lgamma(x + 1.0);
    }
    return -0.5 * std::log(kTwoPi * x) - stirlerr(x) - bd0(x, lambda);
}

inline double dgamma_log_one(double x, double shape, double scale) {
    if (std::isnan(x) || std::isnan(shape) || std::isnan(scale)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (shape < 0.0 || scale <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x < 0.0) return -std::numeric_limits<double>::infinity();
    if (shape == 0.0) {
        return (x == 0.0) ? std::numeric_limits<double>::infinity()
                          : -std::numeric_limits<double>::infinity();
    }
    if (x == 0.0) {
        if (shape < 1.0) return std::numeric_limits<double>::infinity();
        if (shape > 1.0) return -std::numeric_limits<double>::infinity();
        return -std::log(scale);
    }
    if (!std::isfinite(x)) return -std::numeric_limits<double>::infinity();

    if (shape < 1.0) {
        const double pr = dpois_raw_log(shape, x / scale);
        const double ratio = shape / x;
        return pr + (std::isfinite(ratio) ? std::log(ratio)
                                          : std::log(shape) - std::log(x));
    }
    const double pr = dpois_raw_log(shape - 1.0, x / scale);
    return pr - std::log(scale);
}

// Length-1 inputs broadcast against the longest argument, matching numpy.
inline std::size_t pick(std::size_t n, std::size_t i) { return n == 1 ? 0 : i; }

py::array_t<double> dgamma_log(
    py::array_t<double, py::array::c_style | py::array::forcecast> q,
    py::array_t<double, py::array::c_style | py::array::forcecast> shape,
    py::array_t<double, py::array::c_style | py::array::forcecast> scale) {
    const auto bq = q.request();
    const auto bs = shape.request();
    const auto bc = scale.request();

    const std::size_t nq = static_cast<std::size_t>(bq.size);
    const std::size_t ns = static_cast<std::size_t>(bs.size);
    const std::size_t nc = static_cast<std::size_t>(bc.size);
    const std::size_t n = std::max(nq, std::max(ns, nc));

    for (std::size_t len : {nq, ns, nc}) {
        if (len != 1 && len != n) {
            throw std::invalid_argument(
                "q, shape and scale must be the same length or length 1");
        }
    }

    // Follow the shape of whichever argument is not a scalar.
    py::array_t<double> out(nq == n ? bq.shape
                                    : (ns == n ? bs.shape : bc.shape));
    auto bo = out.request();

    const double* pq = static_cast<const double*>(bq.ptr);
    const double* ps = static_cast<const double*>(bs.ptr);
    const double* pc = static_cast<const double*>(bc.ptr);
    double* po = static_cast<double*>(bo.ptr);

    for (std::size_t i = 0; i < n; ++i) {
        po[i] = dgamma_log_one(pq[pick(nq, i)], ps[pick(ns, i)], pc[pick(nc, i)]);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_native_dgamma, m) {
    m.doc() = "Gamma log-density via Loader's saddle-point form, as in R.";
    m.def("dgamma_log", &dgamma_log, py::arg("q"), py::arg("shape"),
          py::arg("scale"),
          "Log of the Gamma density. Arguments broadcast against each other "
          "when of length 1.");
}
