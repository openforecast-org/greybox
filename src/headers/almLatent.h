// The latent values of the response of the dynamic models of alm() (ARI orders): the
// sizes that are not observed (the zeros of an occurrence model, the horizon of a
// forecast) go on with the recursion of the model, as the conditional means given
// the observed values. Shared by R (src/almLatent.cpp) and Python
// (python/src/greybox/_native/alm_latent.cpp).
//
// The model in the space of the lags is z_t = c_t + a'(z_{t-1}, ..., z_{t-p}) + e_t,
// with c_t the part of the regressors. With filter=true it is the Kalman filter on the
// state (z_{t-1}, ..., z_{t-p}), in the units of the variance of e_t: the observed
// values are known exactly, the latent ones have the variances and covariances of the
// recursion, and the observations update the means of the latent values before them.
// Without the filter, the latent values are the recursion of the means, in the space of
// the lags: c_t + a'm_t, or its exponent where the lags are of the data and the model of
// its logarithm (link=1).

#ifndef GREYBOX_ALM_LATENT_H
#define GREYBOX_ALM_LATENT_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace greybox {

struct AlmLatentResult {
    // The lags at each observation (n x p, by columns): the state means
    std::vector<double> lags;
    // The one-step location of z_t, c_t + a'm_t
    std::vector<double> mu;
    // The variance of the error of z_t relative to that of e_t, 1 + a'P_t a
    std::vector<double> variance;
    // The state after the last observation: its mean (p) and covariance (p x p)
    std::vector<double> stateMean;
    std::vector<double> stateCovariance;
};

inline double almLatentLink(double value, int link) {
    return link == 1 ? std::exp(value) : value;
}

// c (n), a (p), z (n, used where observed), observed (n), initialMean (p), and
// initialCovariance (p x p, by columns), or empty for zeros
inline AlmLatentResult almLatent(const std::vector<double> &c, const std::vector<double> &a,
                                 const std::vector<double> &z, const std::vector<int> &observed,
                                 const std::vector<double> &initialMean,
                                 const std::vector<double> &initialCovariance,
                                 bool filter, int link) {
    const std::size_t n = c.size();
    const std::size_t p = a.size();
    AlmLatentResult result;
    result.lags.assign(n * p, 0.0);
    result.mu.assign(n, 0.0);
    result.variance.assign(n, 1.0);

    std::vector<double> m(initialMean);
    std::vector<double> P(p * p, 0.0);
    if (initialCovariance.size() == p * p) {
        P = initialCovariance;
    }
    std::vector<double> g(p, 0.0);
    std::vector<double> mNew(p, 0.0);
    std::vector<double> PNew(p * p, 0.0);

    for (std::size_t t = 0; t < n; ++t) {
        double location = c[t];
        for (std::size_t i = 0; i < p; ++i) {
            result.lags[t + i * n] = m[i];
            location += a[i] * m[i];
        }
        result.mu[t] = location;

        // g = P a, the covariances of z_t with the state, and F = 1 + a'P a
        double variance = 1.0;
        if (filter) {
            for (std::size_t i = 0; i < p; ++i) {
                g[i] = 0.0;
                for (std::size_t j = 0; j < p; ++j) {
                    g[i] += P[i + j * p] * a[j];
                }
                variance += a[i] * g[i];
            }
        }
        result.variance[t] = variance;

        if (p == 0) {
            continue;
        }
        // The new state (z_t, z_{t-1}, ..., z_{t-p+1})
        if (observed[t]) {
            const double innovation = z[t] - location;
            mNew[0] = z[t];
            for (std::size_t i = 1; i < p; ++i) {
                mNew[i] = m[i - 1] + (filter ? g[i - 1] * innovation / variance : 0.0);
            }
            std::fill(PNew.begin(), PNew.end(), 0.0);
            if (filter) {
                for (std::size_t i = 1; i < p; ++i) {
                    for (std::size_t j = 1; j < p; ++j) {
                        PNew[i + j * p] = P[(i - 1) + (j - 1) * p] -
                                          g[i - 1] * g[j - 1] / variance;
                    }
                }
            }
        }
        else {
            mNew[0] = filter ? location : almLatentLink(location, link);
            for (std::size_t i = 1; i < p; ++i) {
                mNew[i] = m[i - 1];
            }
            std::fill(PNew.begin(), PNew.end(), 0.0);
            if (filter) {
                PNew[0] = variance;
                for (std::size_t i = 1; i < p; ++i) {
                    PNew[i] = g[i - 1];
                    PNew[i * p] = g[i - 1];
                    for (std::size_t j = 1; j < p; ++j) {
                        PNew[i + j * p] = P[(i - 1) + (j - 1) * p];
                    }
                }
            }
        }
        m.swap(mNew);
        P.swap(PNew);
    }

    result.stateMean = m;
    result.stateCovariance = P;
    return result;
}

}  // namespace greybox

#endif
