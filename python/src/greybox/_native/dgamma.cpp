// Gamma density and log-density matching R's stats::dgamma to the bit: the
// binding of R's nmath dgamma(), dpois_raw() and ebd0() (R 4.3), in nmath.h.
//
// SciPy's gamma.logpdf evaluates (a-1)*log(x) - x/s - lgamma(a) - a*log(s)
// directly, which cancels catastrophically once the shape grows: at a shape of
// 143 it loses about 1.4e-13, enough to move an optimiser off a boundary.
// R instead uses Loader's saddle-point form,
//
//     log f(x) = -0.5*log(2*pi*(a-1)) - stirlerr(a-1) - bd0(a-1, x/s) - log(s)
//
// where stirlerr is the Stirling series remainder and bd0 the "deviance part",
// both written so that the near-cancelling terms are never formed. Since R
// 4.2, bd0 is computed by ebd0(), which returns it in two parts, yh + yl,
// carried separately until the end.
//
// References:
//   Loader, C. (2000) "Fast and Accurate Computation of Binomial
//   Probabilities". Available from the R sources.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "nmath.h"

namespace py = pybind11;

namespace {

// R/nmath/dgamma.c, through dpois_raw() and ebd0() in nmath.h
template <bool give_log>
double dgamma_one(double x, double shape, double scale) {
    return greybox_nmath::dgamma(x, shape, scale, give_log);
}

// Length-1 inputs broadcast against the longest argument, matching numpy.
inline std::size_t pick(std::size_t n, std::size_t i) { return n == 1 ? 0 : i; }

template <bool give_log>
py::array_t<double> dgamma_array(
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
        po[i] = dgamma_one<give_log>(pq[pick(nq, i)], ps[pick(ns, i)], pc[pick(nc, i)]);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_native_dgamma, m) {
    m.doc() = "Gamma density via Loader's saddle-point form, as in R.";
    m.def("dgamma_log", &dgamma_array<true>, py::arg("q"), py::arg("shape"),
          py::arg("scale"),
          "Log of the Gamma density. Arguments broadcast against each other "
          "when of length 1.");
    m.def("dgamma", &dgamma_array<false>, py::arg("q"), py::arg("shape"),
          py::arg("scale"),
          "The Gamma density. Arguments broadcast against each other when of "
          "length 1.");
}
