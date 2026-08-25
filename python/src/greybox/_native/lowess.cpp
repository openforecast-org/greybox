// LOWESS smoother matching R's stats::lowess exactly.
//
// Adapted from
//   https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/lowess.c
// and from the sibling smooth-package implementation
// (smooth/src/headers/lowess.h), with Armadillo replaced by std::vector to
// avoid a heavy linear-algebra dependency.
//
// References:
//   Cleveland, W.S. (1979) "Robust Locally Weighted Regression and Smoothing
//   Scatterplots". JASA 74(368): 829-836.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace py = pybind11;

namespace {

void lowest(const std::vector<double>& x_sorted,
            const std::vector<double>& y_sorted,
            std::size_t n,
            double xs,
            std::size_t nleft,
            std::size_t nright,
            const std::vector<double>& rw_iter,
            double x_range,
            std::vector<double>& w,
            double& ys_out,
            bool& ok) {
    double h = std::max(xs - x_sorted[nleft], x_sorted[nright] - xs);
    double h9 = 0.999 * h;
    double h1 = 0.001 * h;

    double a = 0.0;
    std::fill(w.begin(), w.end(), 0.0);

    std::size_t j = nleft;
    while (j < n) {
        double r = std::abs(x_sorted[j] - xs);
        if (r <= h9) {
            if (r <= h1) {
                w[j] = 1.0;
            } else {
                double ratio = r / h;
                double cube = ratio * ratio * ratio;
                double omc = 1.0 - cube;
                w[j] = omc * omc * omc;
            }
            w[j] *= rw_iter[j];
            a += w[j];
        } else if (x_sorted[j] > xs) {
            break;
        }
        j++;
    }

    // R's "rightmost pt (may be greater than nright because of ties)": the
    // index the scan stopped at, not the last one that earned a weight. The
    // extra entries carry w == 0, so they contribute nothing, but tracking the
    // last accepted index instead would drop trailing ties.
    std::size_t nrt = j - 1;

    if (a <= 0.0) {
        ys_out = 0.0;
        ok = false;
        return;
    }

    for (std::size_t k = nleft; k <= nrt; ++k) {
        w[k] /= a;
    }

    if (h > 0.0) {
        a = 0.0;
        for (std::size_t k = nleft; k <= nrt; ++k) {
            a += w[k] * x_sorted[k];
        }
        double b = xs - a;
        double c = 0.0;
        for (std::size_t k = nleft; k <= nrt; ++k) {
            double diff = x_sorted[k] - a;
            // R writes this as w[j]*fsquare(x[j]-a), i.e. w * (d*d).  Grouping
            // it as (w*d)*d instead rounds differently, and the robustness
            // iterations amplify that single ULP into tens of them.
            c += w[k] * (diff * diff);
        }
        if (std::sqrt(c) > 0.001 * x_range) {
            b /= c;
            for (std::size_t k = nleft; k <= nrt; ++k) {
                w[k] *= (b * (x_sorted[k] - a) + 1.0);
            }
        }
    }

    ys_out = 0.0;
    for (std::size_t k = nleft; k <= nrt; ++k) {
        ys_out += w[k] * y_sorted[k];
    }
    ok = true;
}

std::vector<double> lowess_impl(const std::vector<double>& x,
                                const std::vector<double>& y,
                                double f,
                                int nsteps,
                                double delta) {
    std::size_t n = x.size();
    if (n < 2) {
        return y;
    }

    double x_min = *std::min_element(x.begin(), x.end());
    double x_max = *std::max_element(x.begin(), x.end());
    if (delta < 0.0) {
        delta = 0.01 * (x_max - x_min);
    }

    std::size_t ns = std::max(static_cast<std::size_t>(2),
                              std::min(n, static_cast<std::size_t>(f * n + 1e-7)));

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });

    std::vector<double> x_sorted(n), y_sorted(n);
    for (std::size_t i = 0; i < n; ++i) {
        x_sorted[i] = x[order[i]];
        y_sorted[i] = y[order[i]];
    }

    std::vector<double> ys(n, 0.0);
    std::vector<double> rw(n, 1.0);
    std::vector<double> res(n, 0.0);
    std::vector<double> w(n, 0.0);

    double x_range = x_sorted[n - 1] - x_sorted[0];

    int iteration = 0;
    while (iteration <= nsteps) {
        std::size_t nleft = 0;
        std::size_t nright = ns - 1;
        long last = -1;
        std::size_t i = 0;

        while (true) {
            if (nright < n - 1) {
                double d1 = x_sorted[i] - x_sorted[nleft];
                double d2 = x_sorted[nright + 1] - x_sorted[i];
                if (d1 > d2) {
                    nleft++;
                    nright++;
                    continue;
                }
            }

            double ys_i;
            bool ok;
            lowest(x_sorted, y_sorted, n, x_sorted[i], nleft, nright, rw, x_range,
                   w, ys_i, ok);
            ys[i] = ys_i;
            if (!ok) {
                ys[i] = y_sorted[i];
            }

            if (last < static_cast<long>(i) - 1) {
                double denom = x_sorted[i] - x_sorted[last];
                if (denom > 0.0) {
                    for (std::size_t j = static_cast<std::size_t>(last) + 1; j < i; ++j) {
                        double alpha = (x_sorted[j] - x_sorted[last]) / denom;
                        ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[last];
                    }
                }
            }

            last = static_cast<long>(i);

            double cut = x_sorted[last] + delta;
            i++;
            while (i < n) {
                if (x_sorted[i] > cut) {
                    break;
                }
                if (x_sorted[i] == x_sorted[last]) {
                    ys[i] = ys[last];
                    last = static_cast<long>(i);
                }
                i++;
            }

            std::size_t last_plus_one = static_cast<std::size_t>(last + 1);
            std::size_t i_minus_one = i > 0 ? i - 1 : 0;
            i = std::max(last_plus_one, i_minus_one);

            if (last >= static_cast<long>(n) - 1) {
                break;
            }
        }

        for (std::size_t k = 0; k < n; ++k) {
            res[k] = y_sorted[k] - ys[k];
        }

        double sum_abs = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            sum_abs += std::abs(res[k]);
        }
        double sc = sum_abs / n;

        if (iteration >= nsteps) {
            break;
        }

        std::vector<double> abs_res(n);
        for (std::size_t k = 0; k < n; ++k) {
            abs_res[k] = std::abs(res[k]);
        }
        std::sort(abs_res.begin(), abs_res.end());

        std::size_t m1 = n / 2;
        double cmad;
        if (n % 2 == 0) {
            std::size_t m2 = n - m1 - 1;
            cmad = 3.0 * (abs_res[m1] + abs_res[m2]);
        } else {
            cmad = 6.0 * abs_res[m1];
        }

        if (cmad < 1e-7 * sc) {
            break;
        }

        double c9 = 0.999 * cmad;
        double c1 = 0.001 * cmad;
        for (std::size_t k = 0; k < n; ++k) {
            double r = std::abs(res[k]);
            if (r <= c1) {
                rw[k] = 1.0;
            } else if (r <= c9) {
                double ratio = r / cmad;
                double omc = 1.0 - ratio * ratio;
                rw[k] = omc * omc;
            } else {
                rw[k] = 0.0;
            }
        }

        iteration++;
    }

    std::vector<double> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        result[order[i]] = ys[i];
    }
    return result;
}

py::array_t<double> lowess_wrapper(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> y,
                                   double f,
                                   int nsteps,
                                   double delta) {
    auto bx = x.request();
    auto by = y.request();
    if (bx.ndim != 1 || by.ndim != 1) {
        throw std::runtime_error("lowess: x and y must be 1-D arrays");
    }
    if (bx.size != by.size) {
        throw std::runtime_error("lowess: x and y must have the same length");
    }

    std::size_t n = static_cast<std::size_t>(bx.size);
    std::vector<double> xv(static_cast<double*>(bx.ptr),
                           static_cast<double*>(bx.ptr) + n);
    std::vector<double> yv(static_cast<double*>(by.ptr),
                           static_cast<double*>(by.ptr) + n);

    std::vector<double> out = lowess_impl(xv, yv, f, nsteps, delta);

    py::array_t<double> result(static_cast<py::ssize_t>(n));
    auto buf = result.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        buf(static_cast<py::ssize_t>(i)) = out[i];
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_native_lowess, m) {
    m.doc() = "LOWESS smoother matching R's stats::lowess (no Armadillo).";
    m.def(
        "lowess",
        &lowess_wrapper,
        py::arg("x"),
        py::arg("y"),
        py::arg("f") = 2.0 / 3.0,
        py::arg("nsteps") = 3,
        py::arg("delta") = -1.0,
        R"pbdoc(Cleveland LOWESS smoother. Mirrors R's stats::lowess.)pbdoc"
    );
}
