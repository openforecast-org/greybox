#' Rectified Normal Distribution
#'
#' Density, cumulative distribution, quantile functions and random number
#' generation for the Rectified Normal distribution.
#'
#' If x~N(mu, sigma^2) then y = max(0, x) follows Rectified Normal distribution:
#' y~RectN(mu, sigma^2), which can be written as:
#'
#' f_y = 1(x<=0) F_x(mu, sigma) + 1(x>0) f_x(x, mu, sigma),
#'
#' where F_x is the cumulative distribution function and f_x is the probability
#' density function of normal distribution.
#'
#' Both \code{prectnorm} and \code{qrectnorm} are returned for the lower
#' tail of the distribution.
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
#' @param log if \code{TRUE}, then probabilities are returned in
#' logarithms.
#'
#' @return Depending on the function, various things are returned
#' (usually either vector or scalar):
#' \itemize{
#' \item \code{drectnorm} returns the density function value for the
#' provided parameters.
#' \item \code{prectnorm} returns the value of the cumulative function
#' for the provided parameters.
#' \item \code{qrectnorm} returns quantiles of the distribution. Depending
#' on what was provided in \code{p}, \code{mu} and \code{sigma}, this
#' can be either a vector or a matrix, or an array.
#' \item \code{rrectnorm} returns a vector of random variables
#' generated from the RectN distribution. Depending on what was
#' provided in \code{mu} and \code{sigma}, this can be either a vector
#' or a matrix or an array.
#' }
#'
#' @seealso \code{\link[greybox]{Distributions}}
#'
#' @examples
#' x <- drectnorm(c(-1000:1000)/200, 0, 1)
#' plot(c(-1000:1000)/200, x, type="l")
#'
#' x <- prectnorm(c(-1000:1000)/200, 0, 1)
#' plot(c(-1000:1000)/200, x, type="l")
#'
#' qrectnorm(c(0.025,0.975), 0, c(1,2))
#'
#' x <- rrectnorm(1000, 0, 1)
#' hist(x)
#'
#' @rdname rectNormal
#' @aliases rectNormal drectnorm
#' @export drectnorm
drectnorm <- function(q, mu=0, sigma=1, log=FALSE){

    rectnormReturn <- (q>0) * dnorm(q, mu, sigma) +
        (q<=0)*pnorm(0, mu, sigma);

    # Return logs if needed. They are taken analytically: the log of the density
    # underflows in the tails
    if(log){
        qPositive <- rep_len(q>0, length(rectnormReturn));
        rectnormReturn[] <- ifelse(qPositive,
                                   rep_len(dnorm(q, mu, sigma, log=TRUE), length(qPositive)),
                                   rep_len(pnorm(0, mu, sigma, log.p=TRUE), length(qPositive)));
    }

    return(rectnormReturn);
}

#' @rdname rectNormal
#' @export prectnorm
#' @aliases prectnorm
prectnorm <- function(q, mu=0, sigma=1){

    rectnormReturn <- (q>0) * pnorm(q, mu, sigma) +
        (q<=0)*pnorm(0, mu, sigma);

    return(rectnormReturn);
}

#' @rdname rectNormal
#' @export qrectnorm
#' @aliases qrectnorm
qrectnorm <- function(p, mu=0, sigma=1){

    rectnormReturn <- pmax(qnorm(p, mu, sigma),0);

    return(rectnormReturn);
}

#' @rdname rectNormal
#' @export rrectnorm
#' @aliases rrectnorm
rrectnorm <- function(n=1, mu=0, sigma=1){

    rectnormReturn <- pmax(rnorm(n, mu, sigma),0);

    return(rectnormReturn);
}
