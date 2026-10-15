#' Box-Cox Normal Distribution
#'
#' Density, cumulative distribution, quantile functions and random number
#' generation for the distribution that becomes normal after the Box-Cox
#' transformation. Note that this is based on the original Box-Cox paper.
#'
#' The distribution has the following density function:
#'
#' f(y) = y^(lambda-1) 1/sqrt(2 pi) exp(-((y^lambda-1)/lambda -mu)^2 / (2 sigma^2))
#'
#' Both \code{pbcnorm} and \code{qbcnorm} are returned for the lower
#' tail of the distribution.
#'
#' In case of lambda=0, the values of the log normal distribution are returned.
#' In case of lambda=1, the values of the normal distribution are returned with mu=mu+1.
#'
#' All the functions are defined for non-negative values only.
#'
#' @template author
#' @keywords distribution
#'
#' @param q vector of quantiles.
#' @param p vector of probabilities.
#' @param n number of observations. Should be a single number.
#' @param mu vector of location parameters (means).
#' @param sigma vector of scale parameters.
#' @param lambda the value of the Box-Cox transform parameter.
#' @param log if \code{TRUE}, then probabilities are returned in
#' logarithms.
#'
#' @return Depending on the function, various things are returned
#' (usually either vector or scalar):
#' \itemize{
#' \item \code{dbcnorm} returns the density function value for the
#' provided parameters.
#' \item \code{pbcnorm} returns the value of the cumulative function
#' for the provided parameters.
#' \item \code{qbcnorm} returns quantiles of the distribution. Depending
#' on what was provided in \code{p}, \code{mu} and \code{sigma}, this
#' can be either a vector or a matrix, or an array.
#' \item \code{rbcnorm} returns a vector of random variables
#' generated from the bcnorm distribution. Depending on what was
#' provided in \code{mu} and \code{sigma}, this can be either a vector
#' or a matrix or an array.
#' }
#'
#' @examples
#' x <- dbcnorm(c(-1000:1000)/200, 0, 1, 1)
#' plot(c(-1000:1000)/200, x, type="l")
#'
#' x <- pbcnorm(c(-1000:1000)/200, 0, 1, 1)
#' plot(c(-1000:1000)/200, x, type="l")
#'
#' qbcnorm(c(0.025,0.975), 0, c(1,2), 1)
#'
#' x <- rbcnorm(1000, 0, 1, 1)
#' hist(x)
#'
#' @references \itemize{
#' \item Box, G. E., & Cox, D. R. (1964). An Analysis of Transformations.
#' Journal of the Royal Statistical Society. Series B (Methodological),
#' 26(2), 211-252.  Retrieved from https://www.jstor.org/stable/2984418
#' \item Granger, C. W. J., & Newbold, P. (1976).  Forecasting transformed
#' series.  Journal of the Royal Statistical Society. Series B
#' (Methodological), 38(2), 189-203.
#' \doi{10.1111/j.2517-6161.1976.tb01585.x}
#' \item Pankratz, A., & Dudley, U. (1987).  Forecasts of power-transformed
#' series.  Journal of Forecasting, 6(4), 239-248.
#' \doi{10.1002/for.3980060403}
#' \item Guerrero, V. M. (1993).  Time-series analysis supported by power
#' transformations.  Journal of Forecasting, 12(1), 37-48.
#' \doi{10.1002/for.3980120104}
#' }
#'
#' @seealso \code{\link[greybox]{Distributions}}
#'
#' @rdname BCNormal
#' @importFrom stats dnorm pnorm qnorm rnorm
#' @aliases BCNormal dbcnorm
#' @export dbcnorm
dbcnorm <- function(q, mu=0, sigma=1, lambda=0, log=FALSE){
    # The logs are taken analytically: the log of the density underflows in the tails
    if(lambda==0){
        bcnormReturn <- dlnorm(x=q, meanlog=mu, sdlog=sigma, log=log);
    }
    else if(lambda==1){
        bcnormReturn <- dnorm(x=q, mean=mu+1, sd=sigma, log=log);
    }
    else if(log){
        bcnormReturn <- suppressWarnings((lambda-1)*log(q) + dnorm((q^lambda-1)/lambda, mu, sigma, log=TRUE));
        bcnormReturn[q<=0] <- -Inf;
    }
    else{
        bcnormReturn <- suppressWarnings(q^(lambda-1) * dnorm((q^lambda-1)/lambda, mu, sigma));
        bcnormReturn[q<=0] <- 0;
    }

    return(bcnormReturn);
}

#' @rdname BCNormal
#' @export pbcnorm
#' @aliases pbcnorm
pbcnorm <- function(q, mu=0, sigma=1, lambda=0){
    if(lambda==0){
        bcnormReturn <- plnorm(q=q, meanlog=mu, sdlog=sigma);
    }
    else{
        # q=0 maps onto -1/lambda
        bcnormReturn <- suppressWarnings(pnorm(q=(q^lambda-1)/lambda, mean=mu, sd=sigma));
        bcnormReturn[q<0] <- 0;
    }
    return(bcnormReturn);
}

#' @rdname BCNormal
#' @export qbcnorm
#' @aliases qbcnorm
qbcnorm <- function(p, mu=0, sigma=1, lambda=0){
    if(lambda==0){
        bcnormReturn <- qlnorm(p=p, meanlog=mu, sdlog=sigma);
    }
    else if(lambda>0){
        # BC inverse support is x > -1/lambda (lower-bounded); the
        # underlying Normal can place mass below that boundary which
        # corresponds to (formally) negative y.  Naive
        # (qnorm(p)*lambda+1)^(1/lambda) becomes NaN for such p when
        # 1/lambda is non-integer (and positive for an even 1/lambda);
        # the standard convention here is to map that mass to y=0 (the
        # natural lower BC support edge).
        bcnormBase <- qnorm(p=p, mean=mu, sd=sigma)*lambda+1;
        bcnormReturn <- bcnormBase^(1/lambda);
        bcnormReturn[is.nan(bcnormReturn) | bcnormBase<0] <- 0;
    }
    else{
        # lambda < 0: BC inverse support is x < -1/lambda (upper-bounded).
        # The underlying Normal puts non-zero mass *above* -1/lambda,
        # which would correspond to y = +infinity (the mass at the
        # upper support boundary).  The unrenormalized BC CDF therefore
        # saturates at P_valid = pnorm(-1/lambda, mu, sigma) < 1, and
        # the naive quantile (qnorm(p)*lambda+1)^(1/lambda) becomes
        # NaN for any p > P_valid -- the previous behaviour was to
        # silently coerce those NaNs to 0, which is dimensionally
        # wrong (the model implies a *large* y, not zero).
        #
        # Renormalize to the truncated distribution on (0, +infinity)
        # by conditioning on the underlying Normal falling within the
        # valid BC inverse support, i.e. invert
        #   F_trunc(y) = pbcnorm(y, mu, sigma, lambda) / P_valid
        # which IS a proper CDF.  The renormalized quantile is finite
        # for every p in (0, 1) and is +infinity at p = 1.  See Granger
        # & Newbold (1976), Pankratz & Dudley (1987), Guerrero (1993)
        # for the standard truncated-distribution treatment of BC at
        # lambda < 0.
        P_valid <- pnorm(-1/lambda, mean=mu, sd=sigma);
        bcnormReturn <- (qnorm(p=p*P_valid, mean=mu, sd=sigma)*lambda+1)^(1/lambda);
        bcnormReturn[p==1] <- Inf;
    }
    return(bcnormReturn);
}

#' @rdname BCNormal
#' @export rbcnorm
#' @aliases rbcnorm
rbcnorm <- function(n=1, mu=0, sigma=1, lambda=0){
    if(lambda==0){
        bcnormReturn <- rlnorm(n=n, meanlog=mu, sdlog=sigma);
    }
    else{
        bcnormReturn <- qbcnorm(runif(n=n, 0, 1), mu=mu, sigma=sigma, lambda=lambda);
    }
    return(bcnormReturn);
}
