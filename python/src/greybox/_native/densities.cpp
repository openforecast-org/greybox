// The log-density of the Beta and the densities and log-densities of the logistic
// distributions as R's nmath computes them, those of the Laplace, S and Generalised
// Normal distributions as R's greybox computes them, the log-densities of their
// log-variants as R's alm() computes them, R's gamma function, and the noncentral t
// and chi-squared distributions as R's nmath computes them (nmath.h).
//
// Everything is in R's order of operations and through libm, as R evaluates
// log(), exp() and ^: NumPy's vectorised kernels round differently in the last
// bit (log on ~0.4% of the arguments, pow on ~5%), and the likelihood surfaces
// of smooth's models are flat enough for that bit to move the optimiser. The
// logs are analytical. lgamma(1/shape) and gamma(1/shape) are R's (nmath.h).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "nmath.h"

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// -log(2*scale) - abs(mu-q)/scale
inline double dlaplace_log_one(double q, double mu, double scale) {
    return -std::log(2 * scale) - std::fabs(mu - q) / scale;
}

// -2*log(2*scale) - sqrt(abs(mu-q))/scale
inline double ds_log_one(double q, double mu, double scale) {
    return -2 * std::log(2 * scale) - std::sqrt(std::fabs(mu - q)) / scale;
}

// log(shape) - log(2*scale) - lgamma(1/shape) - (abs(q-mu)/scale)^shape
inline double dgnorm_log_one(double q, double mu, double scale, double shape) {
    return std::log(shape) - std::log(2 * scale) - greybox_nmath::lgammafn(1 / shape) -
           greybox_nmath::r_pow(std::fabs(q - mu) / scale, shape);
}

// R's nmath dlogis(): x = |q-location|/scale, -(x + log(scale*(1+exp(-x))^2))
inline double dlogis_log_one(double q, double location, double scale) {
    double x = std::fabs((q - location) / scale);
    double f = 1.0 + std::exp(-x);
    return -(x + std::log(scale * f * f));
}

inline double dlogis_one(double q, double location, double scale) {
    double x = std::fabs((q - location) / scale);
    double e = std::exp(-x);
    double f = 1.0 + e;
    return e / (scale * f * f);
}

// R's nmath dbinom_raw(log=TRUE) and dbeta(), of R 4.3.3; bd0() and lbeta() are in
// nmath.h
inline double dbinom_raw_log(double x, double n, double p, double q) {
    if (p == 0) return x == 0 ? 0.0 : -INFINITY;
    if (q == 0) return x == n ? 0.0 : -INFINITY;
    if (x == 0) {
        if (n == 0) return 0.0;
        return p < 0.1 ? -greybox_nmath::bd0(n, n * q) - n * p : n * std::log(q);
    }
    if (x == n) return q < 0.1 ? -greybox_nmath::bd0(n, n * p) - n * q : n * std::log(p);
    if (x < 0 || x > n) return -INFINITY;
    double lc = greybox_nmath::stirlerr(n) - greybox_nmath::stirlerr(x) -
                greybox_nmath::stirlerr(n - x) - greybox_nmath::bd0(x, n * p) -
                greybox_nmath::bd0(n - x, n * q);
    double lf = 1.837877066409345483560659472811 + std::log(x) + std::log1p(-x / n);
    return lc - 0.5 * lf;
}

inline double dbeta_log_one(double x, double a, double b) {
    if (std::isnan(x) || std::isnan(a) || std::isnan(b)) return x + a + b;
    if (a < 0 || b < 0) return NAN;
    if (x < 0 || x > 1) return -INFINITY;
    if (a == 0 || b == 0 || !std::isfinite(a) || !std::isfinite(b)) {
        if (a == 0 && b == 0) return (x == 0 || x == 1) ? INFINITY : -INFINITY;
        if (a == 0 || a / b == 0) return x == 0 ? INFINITY : -INFINITY;
        if (b == 0 || b / a == 0) return x == 1 ? INFINITY : -INFINITY;
        return x == 0.5 ? INFINITY : -INFINITY;
    }
    if (x == 0) return a > 1 ? -INFINITY : (a < 1 ? INFINITY : std::log(b));
    if (x == 1) return b > 1 ? -INFINITY : (b < 1 ? INFINITY : std::log(a));
    if (a <= 2 || b <= 2) return (a - 1) * std::log(x) + (b - 1) * std::log1p(-x) -
                                   greybox_nmath::lbeta(a, b);
    return std::log(a + b - 1) + dbinom_raw_log(a - 1, a + b - 2, x, 1 - x);
}

// 1/(2*scale)*exp(-abs(mu-q)/scale)
inline double dlaplace_one(double q, double mu, double scale) {
    return 1 / (2 * scale) * std::exp(-std::fabs(mu - q) / scale);
}

// 1/(4*scale^2)*exp(-sqrt(abs(mu-q))/scale)
inline double ds_one(double q, double mu, double scale) {
    return 1 / (4 * greybox_nmath::r_pow(scale, 2)) *
           std::exp(-std::sqrt(std::fabs(mu - q)) / scale);
}

// exp(-(abs(q-mu)/scale)^shape)*shape/(2*scale*gamma(1/shape))
inline double dgnorm_one(double q, double mu, double scale, double shape) {
    return std::exp(-greybox_nmath::r_pow(std::fabs(q - mu) / scale, shape)) * shape /
           (2 * scale * greybox_nmath::gammafn(1 / shape));
}

// The log-variants, as alm() takes their log-likelihoods:
// dlaplace(log(y), mu, scale, log=TRUE) - log(y), and so on.
inline double dllaplace_log_one(double y, double mu, double scale) {
    return dlaplace_log_one(std::log(y), mu, scale) - std::log(y);
}

inline double dls_log_one(double y, double mu, double scale) {
    return ds_log_one(std::log(y), mu, scale) - std::log(y);
}

inline double dlgnorm_log_one(double y, double mu, double scale, double shape) {
    return dgnorm_log_one(std::log(y), mu, scale, shape) - std::log(y);
}

// Length-1 inputs broadcast against the longest argument, matching numpy.
inline std::size_t pick(std::size_t n, std::size_t i) { return n == 1 ? 0 : i; }

// f applied elementwise to the arguments, with the shape of the longest one.
template <std::size_t N, typename F>
py::array_t<double> elementwise(F f, const std::array<Array, N>& args) {
    std::array<py::buffer_info, N> buffers;
    std::size_t n = 0;
    for (std::size_t k = 0; k < N; ++k) {
        buffers[k] = args[k].request();
        n = std::max(n, static_cast<std::size_t>(buffers[k].size));
    }
    std::size_t longest = 0;
    for (std::size_t k = 0; k < N; ++k) {
        const auto size = static_cast<std::size_t>(buffers[k].size);
        if (size != 1 && size != n) {
            throw std::invalid_argument("the arguments must be the same length or length 1");
        }
        if (size == n) longest = k;
    }
    py::array_t<double> out(buffers[longest].shape);
    double* po = static_cast<double*>(out.request().ptr);
    for (std::size_t i = 0; i < n; ++i) {
        std::array<double, N> values;
        for (std::size_t k = 0; k < N; ++k) {
            const auto size = static_cast<std::size_t>(buffers[k].size);
            values[k] = static_cast<const double*>(buffers[k].ptr)[pick(size, i)];
        }
        po[i] = std::apply(f, values);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_native_densities, m) {
    m.doc() = "Densities and log-densities of the Laplace, S and Generalised Normal "
              "distributions and their log-variants, the gamma function and the "
              "noncentral t and chi-squared distributions, as in R.";
    m.def(
        "dlaplace_log",
        [](Array q, Array mu, Array scale) {
            return elementwise<3>(dlaplace_log_one, {q, mu, scale});
        },
        py::arg("q"), py::arg("mu"), py::arg("scale"));
    m.def(
        "ds_log",
        [](Array q, Array mu, Array scale) {
            return elementwise<3>(ds_log_one, {q, mu, scale});
        },
        py::arg("q"), py::arg("mu"), py::arg("scale"));
    m.def(
        "dgnorm_log",
        [](Array q, Array mu, Array scale, Array shape) {
            return elementwise<4>(dgnorm_log_one, {q, mu, scale, shape});
        },
        py::arg("q"), py::arg("mu"), py::arg("scale"), py::arg("shape"));
    m.def(
        "dlogis_log",
        [](Array q, Array location, Array scale) {
            return elementwise<3>(dlogis_log_one, {q, location, scale});
        },
        py::arg("q"), py::arg("location"), py::arg("scale"));
    m.def(
        "dlogis",
        [](Array q, Array location, Array scale) {
            return elementwise<3>(dlogis_one, {q, location, scale});
        },
        py::arg("q"), py::arg("location"), py::arg("scale"));
    m.def(
        "dbeta_log",
        [](Array x, Array shape1, Array shape2) {
            return elementwise<3>(dbeta_log_one, {x, shape1, shape2});
        },
        py::arg("x"), py::arg("shape1"), py::arg("shape2"));
    m.def(
        "dlaplace",
        [](Array q, Array mu, Array scale) {
            return elementwise<3>(dlaplace_one, {q, mu, scale});
        },
        py::arg("q"), py::arg("mu"), py::arg("scale"));
    m.def(
        "ds",
        [](Array q, Array mu, Array scale) {
            return elementwise<3>(ds_one, {q, mu, scale});
        },
        py::arg("q"), py::arg("mu"), py::arg("scale"));
    m.def(
        "dgnorm",
        [](Array q, Array mu, Array scale, Array shape) {
            return elementwise<4>(dgnorm_one, {q, mu, scale, shape});
        },
        py::arg("q"), py::arg("mu"), py::arg("scale"), py::arg("shape"));
    m.def(
        "dllaplace_log",
        [](Array y, Array mu, Array scale) {
            return elementwise<3>(dllaplace_log_one, {y, mu, scale});
        },
        py::arg("y"), py::arg("mu"), py::arg("scale"));
    m.def(
        "dls_log",
        [](Array y, Array mu, Array scale) {
            return elementwise<3>(dls_log_one, {y, mu, scale});
        },
        py::arg("y"), py::arg("mu"), py::arg("scale"));
    m.def(
        "dlgnorm_log",
        [](Array y, Array mu, Array scale, Array shape) {
            return elementwise<4>(dlgnorm_log_one, {y, mu, scale, shape});
        },
        py::arg("y"), py::arg("mu"), py::arg("scale"), py::arg("shape"));
    m.def(
        "gammafn",
        [](Array x) {
            return elementwise<1>([](double v) { return greybox_nmath::gammafn(v); }, {x});
        },
        py::arg("x"), "R's gamma function.");
    // R's noncentral t and chi-squared, elementwise over (x, df, ncp)
    using Density = double (*)(double, double, double, int);
    using Probability = double (*)(double, double, double, int, int);
    const std::pair<const char*, Density> densities[] = {{"dnt", greybox_nmath::dnt},
                                                         {"dnchisq", greybox_nmath::dnchisq}};
    const std::pair<const char*, Probability> probabilities[] = {
        {"pnt", greybox_nmath::pnt},
        {"qnt", greybox_nmath::qnt},
        {"pnchisq", greybox_nmath::pnchisq},
        {"qnchisq", greybox_nmath::qnchisq}};
    for (const auto& [name, density] : densities) {
        m.def(
            name,
            [density](Array x, Array df, Array ncp, bool give_log) {
                return elementwise<3>(
                    [&](double v, double d, double n) { return density(v, d, n, give_log); },
                    {x, df, ncp});
            },
            py::arg("x"), py::arg("df"), py::arg("ncp"), py::arg("give_log"));
    }
    for (const auto& [name, function] : probabilities) {
        m.def(
            name,
            [function](Array x, Array df, Array ncp, bool lower_tail, bool log_p) {
                return elementwise<3>(
                    [&](double v, double d, double n) {
                        return function(v, d, n, lower_tail, log_p);
                    },
                    {x, df, ncp});
            },
            py::arg("x"), py::arg("df"), py::arg("ncp"), py::arg("lower_tail"),
            py::arg("log_p"));
    }
}
