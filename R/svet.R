#' S Distribution
#'
#' Density, cumulative distribution, quantile functions and random number
#' generation for the S distribution with the location parameter mu
#' and a scaling parameter scale.
#'
#' When mu=0 and ham=2, the S distribution becomes standardized with
#' scale=1 (this is because scale=ham/2). The distribution has the following
#' density function:
#'
#' f(x) = 1/(4 scale^2) exp(-sqrt(abs(x-mu)) / scale)
#'
#' The S distribution has fat tails and large excess.
#'
#' Both \code{ps} and \code{qs} are returned for the lower tail of
#' the distribution.
#'
#' @template author
#' @keywords distribution
#'
#' @param q vector of quantiles.
#' @param p vector of probabilities.
#' @param n number of observations. Should be a single number.
#' @param mu vector of location parameters (means).
#' @param scale vector of scaling parameter (which are equal to ham/2).
#' @param log if \code{TRUE}, then probabilities are returned in
#' logarithms.
#'
#' @return Depending on the function, various things are returned
#' (usually either vector or scalar):
#' \itemize{
#' \item \code{ds} returns the density function value for the
#' provided parameters.
#' \item \code{ps} returns the value of the cumulative function
#' for the provided parameters.
#' \item \code{qs} returns quantiles of the distribution. Depending
#' on what was provided in \code{p}, \code{mu} and \code{scale}, this
#' can be either a vector or a matrix, or an array.
#' \item \code{rs} returns a vector of random variables
#' generated from the S distribution. Depending on what was provided
#' in \code{mu} and \code{scale}, this can be either a vector or a matrix
#' or an array.
#' }
#'
#' @seealso \code{\link[greybox]{Distributions}}
#'
#' @examples
#' x <- ds(c(-1000:1000)/10, 0, 1)
#' plot(x, type="l")
#'
#' x <- ps(c(-1000:1000)/10, 0, 1)
#' plot(x, type="l")
#'
#' qs(c(0.025,0.975), 0, 1)
#'
#' x <- rs(1000, 0, 1)
#' hist(x)
#'
#' @rdname SDistribution
#' @importFrom stats optimize runif
#' @export ds
#' @aliases SDistribution ds
ds <- function(q, mu=0, scale=1, log=FALSE){
    # The log is taken analytically: the log of the density underflows in the tails
    if(log){
        return(-2*log(2*scale)-sqrt(abs(mu-q))/scale);
    }
    return(1/(4*scale^2)*exp(-sqrt(abs(mu-q))/scale));
}

#' @rdname SDistribution
#' @export ps
#' @aliases ps
ps <- function(q, mu=0, scale=1){
    svetReturn <- 0.5+0.5*sign(q-mu)*(1-1/scale*(sqrt(abs(mu-q))+scale)*exp(-sqrt(abs(mu-q))/scale))
    return(svetReturn);
}

#' @rdname SDistribution
#' @export qs
#' @aliases qs
qs <- function(p, mu=0, scale=1){
    # Square scale to get to the same scale as in GN
    return(qgnorm(p, mu=mu, scale=scale^2, shape=0.5))
    # p <- unique(p);
    # mu <- unique(mu);
    # scale <- unique(scale);
    # lengthMax <- max(length(p),length(mu),length(scale));
    # # If length of p, mu and scale differs, then go difficult. Otherwise do simple stuff
    # if(any(!c(length(p),length(mu),length(scale)) %in% c(lengthMax, 1))){
    #     svetReturn <- array(0,c(length(p),length(mu),length(scale)),
    #                         dimnames=list(paste0("p=",p),paste0("mu=",mu),paste0("scale=",scale)));
    #     svetReturn[p==0.5,,] <- 1;
    #     svetReturn[p==0,,] <- -Inf;
    #     svetReturn[p==1,,] <- Inf;
    #     probsToEstimate <- which(svetReturn[,1,1]==0);
    #     svetReturn[svetReturn==1] <- 0;
    #     if(length(probsToEstimate)!=0){
    #         for(i in 1:length(probsToEstimate)){
    #             j <- probsToEstimate[i];
    #             for(k in 1:length(scale)){
    #                 if(p[j]<0.5){
    #                     svetReturn[j,,k] <- (-scale[k]^2*lambertWm1(-2*p[j]/exp(1))^2 -
    #                                              2*scale[k]^2*lambertWm1(-2*p[j]/exp(1))-scale[k]^2);
    #                 }
    #                 else{
    #                     svetReturn[j,,k] <- (scale[k]^2*lambertWm1(2*(p[j]-1)/exp(1))^2 +
    #                                              2*scale[k]^2*lambertWm1(2*(p[j]-1)/exp(1))+scale[k]^2);
    #                 }
    #             }
    #         }
    #         svetReturn <- svetReturn + rep(mu,each=length(p));
    #     }
    #     # Drop the redundant dimensions
    #     svetReturn <- svetReturn[,,];
    # }
    # else{
    #     Ie <- (p < 0.5)*1;
    #     svetReturn <- rep(0,max(length(p),length(mu),length(scale)));
    #     svetReturn[] <- mu + ((-1)^Ie * scale^2*lambertWm1((-1)^Ie * 2*(p - 1 + Ie)/exp(1))^2 +
    #                             (-1)^Ie*2*scale^2*lambertWm1((-1)^Ie * 2*(p - 1 + Ie)/exp(1))+(-1)^Ie*scale^2);
    #     if(any(p==0)){
    #         svetReturn[p==0] <- -Inf;
    #     }
    #     if(any(p==1)){
    #         svetReturn[p==1] <- Inf;
    #     }
    # }
    # return(svetReturn);
}

#' @rdname SDistribution
#' @export rs
#' @aliases rs
rs <- function(n=1, mu=0, scale=1){
    svetReturn <- qs(runif(n,0,1),mu,scale);
    return(svetReturn);
}
