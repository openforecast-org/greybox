#include <Rcpp.h>
#include "headers/almLatent.h"
using namespace Rcpp;

// The latent values of the response of the dynamic models of alm(): see headers/almLatent.h
// [[Rcpp::export]]
List almLatentCpp(NumericVector const &c, NumericVector const &a, NumericVector const &z,
                  LogicalVector const &observed, NumericVector const &initialMean,
                  NumericVector const &initialCovariance, bool filter){
    const int n = c.size();
    const int p = a.size();
    greybox::AlmLatentResult result =
        greybox::almLatent(as<std::vector<double>>(c), as<std::vector<double>>(a),
                           as<std::vector<double>>(z), as<std::vector<int>>(observed),
                           as<std::vector<double>>(initialMean),
                           as<std::vector<double>>(initialCovariance), filter);

    NumericMatrix lags(n, p);
    std::copy(result.lags.begin(), result.lags.end(), lags.begin());
    NumericMatrix stateCovariance(p, p);
    std::copy(result.stateCovariance.begin(), result.stateCovariance.end(), stateCovariance.begin());

    return List::create(Named("lags")=lags, Named("mu")=wrap(result.mu),
                        Named("variance")=wrap(result.variance),
                        Named("stateMean")=wrap(result.stateMean),
                        Named("stateCovariance")=stateCovariance);
}
