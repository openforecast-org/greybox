// The latent values of the response of the dynamic models of ALM: see the header shared
// with R, src/headers/almLatent.h.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <vector>

#include "headers/almLatent.h"

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Flags = py::array_t<int, py::array::c_style | py::array::forcecast>;

template <typename T, typename A>
std::vector<T> to_vector(const A &values) {
    return std::vector<T>(values.data(), values.data() + values.size());
}

py::dict alm_latent(const Array &c, const Array &a, const Array &z, const Flags &observed,
                    const Array &initial_mean, const Array &initial_covariance,
                    bool filter) {
    const auto result = greybox::almLatent(
        to_vector<double>(c), to_vector<double>(a), to_vector<double>(z),
        to_vector<int>(observed), to_vector<double>(initial_mean),
        to_vector<double>(initial_covariance), filter);
    const auto n = static_cast<py::ssize_t>(c.size());
    const auto p = static_cast<py::ssize_t>(a.size());

    // The lags and the covariance come by columns, as in R
    py::array_t<double, py::array::f_style> lags({n, p});
    std::copy(result.lags.begin(), result.lags.end(), lags.mutable_data());
    py::array_t<double, py::array::f_style> covariance({p, p});
    std::copy(result.stateCovariance.begin(), result.stateCovariance.end(),
              covariance.mutable_data());

    py::dict out;
    out["lags"] = lags;
    out["mu"] = py::array_t<double>(n, result.mu.data());
    out["variance"] = py::array_t<double>(n, result.variance.data());
    out["state_mean"] = py::array_t<double>(p, result.stateMean.data());
    out["state_covariance"] = covariance;
    return out;
}

}  // namespace

PYBIND11_MODULE(_native_alm_latent, m) {
    m.doc() = "The latent values of the dynamic models of ALM (R's almLatentCpp)";
    m.def("alm_latent", &alm_latent, py::arg("c"), py::arg("a"), py::arg("z"),
          py::arg("observed"), py::arg("initial_mean"), py::arg("initial_covariance"),
          py::arg("filter"));
}
