#' Laplace Distribution
#'
#' Density, cumulative distribution, quantile functions and random number
#' generation for the Laplace distribution with the location parameter mu
#' and the scale parameter (which is equal to Mean Absolute Error, aka
#' Mean Absolute Deviation).
#'
#' When mu=0 and scale=1, the Laplace distribution becomes standardized.
#' The distribution has the following density function:
#'
#' f(x) = 1/(2 scale) exp(-abs(x-mu) / scale)
#'
#' Both \code{plaplace} and \code{qlaplace} are returned for the lower
#' tail of the distribution.
#'
#' @template author
#' @keywords distribution
#'
#' @param q vector of quantiles.
#' @param p vector of probabilities.
#' @param n number of observations. Should be a single number.
#' @param mu vector of location parameters (means).
#' @param scale vector of mean absolute errors.
#' @param log if \code{TRUE}, then probabilities are returned in
#' logarithms.
#'
#' @return Depending on the function, various things are returned
#' (usually either vector or scalar):
#' \itemize{
#' \item \code{dlaplace} returns the density function value for the
#' provided parameters.
#' \item \code{plaplace} returns the value of the cumulative function
#' for the provided parameters.
#' \item \code{qlaplace} returns quantiles of the distribution. Depending
#' on what was provided in \code{p}, \code{mu} and \code{scale}, this
#' can be either a vector or a matrix, or an array.
#' \item \code{rlaplace} returns a vector of random variables
#' generated from the Laplace distribution. Depending on what was
#' provided in \code{mu} and \code{scale}, this can be either a vector
#' or a matrix or an array.
#' }
#'
#' @examples
#' x <- dlaplace(c(-100:100)/10, 0, 1)
#' plot(x, type="l")
#'
#' x <- plaplace(c(-100:100)/10, 0, 1)
#' plot(x, type="l")
#'
#' qlaplace(c(0.025,0.975), 0, c(1,2))
#'
#' x <- rlaplace(1000, 0, 1)
#' hist(x)
#'
#' @references \itemize{
#' \item Wikipedia page on Laplace distribution:
#' \url{https://en.wikipedia.org/wiki/Laplace_distribution}.
#' }
#'
#' @seealso \code{\link[greybox]{Distributions}}
#'
#' @rdname Laplace
#' @export dlaplace
#' @aliases Laplace dlaplace
dlaplace <- function(q, mu=0, scale=1, log=FALSE){
    # The log is taken analytically: the log of the density underflows in the tails
    if(log){
        return(-log(2*scale)-abs(mu-q)/scale);
    }
    return(1/(2*scale)*exp(-abs(mu-q)/scale));
}

#' @rdname Laplace
#' @export plaplace
#' @aliases plaplace
plaplace <- function(q, mu=0, scale=1){
    laplaceReturn <- 0.5 + 0.5*sign(q-mu)*(1-exp(-abs(q-mu)/scale));
    return(laplaceReturn);
}

#' @rdname Laplace
#' @export qlaplace
#' @aliases qlaplace
qlaplace <- function(p, mu=0, scale=1){
    return(qgnorm(p, mu=mu, scale=scale, shape=1))
    # p <- unique(p);
    # mu <- unique(mu);
    # scale <- unique(scale);
    # lengthMax <- max(length(p),length(mu),length(scale));
    # # If length of p, mu and scale differs, then go difficult. Otherwise do simple stuff
    # if(any(!c(length(p),length(mu),length(scale)) %in% c(lengthMax, 1))){
    #     laplaceReturn <- array(0,c(length(p),length(mu),length(scale)),
    #                            dimnames=list(paste0("p=",p),paste0("mu=",mu),paste0("scale=",scale)));
    #     laplaceReturn[p==0.5,,] <- 1;
    #     laplaceReturn[p==0,,] <- -Inf;
    #     laplaceReturn[p==1,,] <- Inf;
    #     probsToEstimate <- which(laplaceReturn[,1,1]==0);
    #     laplaceReturn[laplaceReturn==1] <- 0;
    #     for(k in 1:length(scale)){
    #         laplaceReturn[probsToEstimate,,k] <- (-scale[k] * sign(p[probsToEstimate]-0.5) *
    #                                                   log(1-2*abs(p[probsToEstimate]-0.5)));
    #     }
    #     laplaceReturn <- laplaceReturn + rep(mu,each=length(p));
    #     # Drop the redundant dimensions
    #     laplaceReturn <- laplaceReturn[,,];
    # }
    # else{
    #     laplaceReturn <- mu - scale * sign(p-0.5) * log(1-2*abs(p-0.5));
    # }
    # return(laplaceReturn);
}

#' @rdname Laplace
#' @export rlaplace
#' @aliases rlaplace
rlaplace <- function(n=1, mu=0, scale=1){
    laplaceReturn <- qlaplace(runif(n,0,1),mu,scale);
    return(laplaceReturn);
}
