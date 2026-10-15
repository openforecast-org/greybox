#' Logit Normal Distribution
#'
#' Density, cumulative distribution, quantile functions and random number
#' generation for the distribution that becomes normal after the Logit
#' transformation.
#'
#' The distribution has the following density function:
#'
#' f(y) = 1/(sqrt(2 pi) y (1-y)) exp(-(logit(y) -mu)^2 / (2 sigma^2))
#'
#' where y is in (0, 1) and logit(y) =log(y/(1-y)).
#'
#' Both \code{plogitnorm} and \code{qlogitnorm} are returned for the lower
#' tail of the distribution.
#'
#' All the functions are defined for the values between 0 and 1.
#'
#' @template author
#' @keywords distribution
#'
#' @param q vector of quantiles.
#' @param p vector of probabilities.
#' @param n number of observations. Should be a single number.
#' @param mu vector of location parameters (means).
#' @param sigma vector of scale parameters.
#' @param log if \code{TRUE}, then probabilities are returned in
#' logarithms.
#'
#' @return Depending on the function, various things are returned
#' (usually either vector or scalar):
#' \itemize{
#' \item \code{dlogitnorm} returns the density function value for the
#' provided parameters.
#' \item \code{plogitnorm} returns the value of the cumulative function
#' for the provided parameters.
#' \item \code{qlogitnorm} returns quantiles of the distribution. Depending
#' on what was provided in \code{p}, \code{mu} and \code{sigma}, this
#' can be either a vector or a matrix, or an array.
#' \item \code{rlogitnorm} returns a vector of random variables
#' generated from the logitnorm distribution. Depending on what was
#' provided in \code{mu} and \code{sigma}, this can be either a vector
#' or a matrix or an array.
#' }
#'
#' @examples
#' x <- dlogitnorm(c(-1000:1000)/200, 0, 1)
#' plot(c(-1000:1000)/200, x, type="l")
#'
#' x <- plogitnorm(c(-1000:1000)/200, 0, 1)
#' plot(c(-1000:1000)/200, x, type="l")
#'
#' qlogitnorm(c(0.025,0.975), 0, c(1,2))
#'
#' x <- rlogitnorm(1000, 0, 1)
#' hist(x)
#'
#' @references \itemize{
#' \item Mead, R. (1965). A Generalised Logit-Normal Distribution.
#' Biometrics, 21 (3), 721–732. doi: 10.2307/2528553
#' }
#'
#' @seealso \code{\link[greybox]{Distributions}}
#'
#' @rdname LogitNormal
#' @importFrom stats dnorm pnorm qnorm rnorm
#' @aliases LogitNormal dlogitnorm
#' @export dlogitnorm
dlogitnorm <- function(q, mu=0, sigma=1, log=FALSE){
    # The log is taken analytically: the log of the density underflows in the tails
    if(log){
        return(-log(sigma*sqrt(2*pi)*q*(1-q))-(log(q/(1-q))-mu)^2/(2*sigma^2));
    }
    logitnormReturn <- 1 / (sigma*sqrt(2*pi)*q*(1-q))*exp(-(log(q/(1-q))-mu)^2/(2*sigma^2));
    # logitnormReturn[q<=0] <- -Inf;
    # logitnormReturn[q>=1] <- Inf;

    return(logitnormReturn);
}

#' @rdname LogitNormal
#' @export plogitnorm
#' @aliases plogitnorm
plogitnorm <- function(q, mu=0, sigma=1){
    logitnormReturn <- suppressWarnings(pnorm(q=log(q/(1-q)), mean=mu, sd=sigma));
    logitnormReturn[q<0] <- 0;
    logitnormReturn[q>=1] <- 1;
    return(logitnormReturn);
}

#' @rdname LogitNormal
#' @export qlogitnorm
#' @aliases qlogitnorm
qlogitnorm <- function(p, mu=0, sigma=1){
    logitnormReturn <- qnorm(p=p, mean=mu, sd=sigma);
    logitnormReturn[] <- 1/(1+exp(-logitnormReturn));
    logitnormReturn[is.nan(logitnormReturn)] <- 0;
    return(logitnormReturn);
}

#' @rdname LogitNormal
#' @export rlogitnorm
#' @aliases rlogitnorm
rlogitnorm <- function(n=1, mu=0, sigma=1){
    logitnormReturn <- qlogitnorm(runif(n=n, 0, 1), mu=mu, sigma=sigma);
    return(logitnormReturn);
}
