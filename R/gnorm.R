#' The generalized normal distribution
#'
#' Density, cumulative distribution, quantile functions and random number
#' generation for the Generalised Normal distribution with the location
#' mu, a scale and a shape parameters.
#'
#' A generalized normal random variable \eqn{x} with parameters location \eqn{\mu},
#' scale \eqn{s > 0} and shape \eqn{\beta > 0} has density:
#'
#' \deqn{p(x) = \beta exp{-(|x - \mu|/s)^\beta}/(2s \Gamma(1/\beta)).} \cr
#' The mean and variance of \eqn{x} are \eqn{\mu} and
#' \eqn{s^2 \Gamma(3/\beta)/\Gamma(1/\beta)}, respectively.
#'
#' The function are based on the functions from gnorm package of Maryclare Griffin
#' (package has been abandoned since 2018).
#'
#' The quantile and cumulative functions use uniform approximation for cases
#' \code{shape>100}. This is needed, because otherwise it is not possible to calculate
#' the values correctly due to \code{scale^(shape)=Inf} in R.
#'
#' @keywords distribution
#' @aliases dgnorm pgnorm qgnorm rgnorm
#' @param q vector of quantiles
#' @param p vector of probabilities
#' @param n number of observations
#' @param mu location parameter
#' @param scale scale parameter
#' @param shape shape parameter
#' @param log,log.p logical; if TRUE, probabilities p are given as log(p)
#' @param lower.tail logical; if TRUE (default), probabilities are \eqn{P[X\leq x]},
#' otherwise \eqn{P[X> x]}
#' @source \code{dgnorm}, \code{pgnorm}, \code{qgnorm} and\code{rgnorm} are all
#' parametrized as in Version 1 of the
#' \href{https://en.wikipedia.org/wiki/Generalized_normal_distribution}{Generalized
#' Normal Distribution Wikipedia page},
#' which uses the parametrization given by in Nadarajah (2005).
#' The same distribution was described much earlier by Subbotin (1923) and named
#' the exponential power distribution by Box and Tiao (1973). \cr \cr
#' @references
#' \itemize{
#' \item Box, G. E. P. and G. C. Tiao. "Bayesian inference in Statistical Analysis."
#' Addison-Wesley Pub. Co., Reading, Mass (1973).
#' \item Nadarajah, Saralees. "A generalized normal distribution." Journal of
#' Applied Statistics 32.7 (2005): 685-694.
#' \item Subbotin, M. T. "On the Law of Frequency of Error." Matematicheskii
#' Sbornik 31.2 (1923):  206-301.
#' }
#'
#' @author Maryclare Griffin and Ivan Svetunkov
#'
#' @seealso \code{\link[greybox]{Distributions}}
#'
#' @examples
#' # Density function values for standard normal distribution
#' x <- dgnorm(seq(-1, 1, length.out = 100), 0, sqrt(2), 2)
#' plot(x, type="l")
#'
#' #CDF of standard Laplace
#' x <- pgnorm(c(-100:100), 0, 1, 1)
#' plot(x, type="l")
#'
#' # Quantiles of S distribution
#' qgnorm(c(0.025,0.975), 0, 1, 0.5)
#'
#' # Random numbers from a distribution with shape=10000 (approximately uniform)
#' x <- rgnorm(1000, 0, 1, 1000)
#' hist(x)
#'
#' @rdname gnorm
#' @importFrom stats pgamma qgamma rbinom runif punif qunif
#' @export
dgnorm <- function(q, mu = 0, scale = 1, shape = 1,
                   log = FALSE) {
    # A failsafe for NaN / NAs of scale / shape
    if(any(is.nan(scale))){
        scale[is.nan(scale)] <- 0
    }
    if(any(is.na(scale))){
        scale[is.na(scale)] <- 0
    }
    if(any(scale<0)){
        scale[scale<0] <- 0
    }
    if(any(is.nan(shape))){
        shape[is.nan(shape)] <- 0
    }
    if(any(is.na(shape))){
        shape[is.na(shape)] <- 0
    }
    # The log is taken analytically: the log of the density underflows in the tails
    if(log){
        return(log(shape)-log(2*scale)-lgamma(1/shape)-(abs(q-mu)/scale)^shape)
    }
    gnormValues <- (exp(-(abs(q-mu)/ scale)^shape)* shape/(2*scale*gamma(1/shape)))

    return(gnormValues)
}

#' @rdname gnorm
#' @export
pgnorm <- function(q, mu = 0, scale = 1, shape = 1,
                   lower.tail = TRUE, log.p = FALSE) {
  # A failsafe for NaN / NAs of scale / shape
  if(any(is.nan(scale))){
    scale[is.nan(scale)] <- 0
  }
  if(any(is.na(scale))){
    scale[is.na(scale)] <- 0
  }
  if(any(scale<0)){
    scale[scale<0] <- 0
  }
  if(any(is.nan(shape))){
    shape[is.nan(shape)] <- 0
  }
  if(any(is.na(shape))){
    shape[is.na(shape)] <- 0
  }

  # Failsafe mechanism. If shape is too high, switch to uniform
  if(shape>100){
    return(punif(q, min=mu-scale, mu+scale, lower.tail=lower.tail, log.p=log.p))
  }
  else{
    p <- (1/2 + sign(q - mu[])* pgamma(abs(q - mu)^shape, shape = 1/shape, rate = (1/scale)^shape)/2)
    if (lower.tail) {
      if (!log.p) {
        return(p)
      } else {
        return(log(p))
      }
    } else if (!lower.tail) {
      if (!log.p) {
        return(1 - p)
      } else {
        return(log(1 - p))
      }
    }
  }
}

#' @rdname gnorm
#' @export
qgnorm <- function(p, mu = 0, scale = 1, shape = 1,
                   lower.tail = TRUE, log.p = FALSE) {
  # A failsafe for NaN / NAs of scale / shape
  if(any(is.nan(scale))){
    scale[is.nan(scale)] <- 0
  }
  if(any(is.na(scale))){
    scale[is.na(scale)] <- 0
  }
  if(any(scale<0)){
    scale[scale<0] <- 0
  }
  if(any(is.nan(shape))){
    shape[is.nan(shape)] <- 0
  }
  if(any(is.na(shape))){
    shape[is.na(shape)] <- 0
  }

  if (lower.tail & !log.p) {
    p <- p
  } else if (lower.tail & log.p) {
    p <- exp(p)
  } else if (!lower.tail & !log.p) {
    p <- 1 - p
  } else {
    p <- log(1 - p)
  }

  # Failsafe mechanism. If shape is too high, switch to uniform
  if(all(shape>100)){
    gnormValues <- qunif(p, min=mu-scale, mu+scale);
  }
  # If it is not too bad, scale the scale parameter
  else if(any((1/scale)^shape<1e-300)){
    lambdaScale <- ceiling(scale) / 10
    lambda <- (scale/lambdaScale)^(shape)
    gnormValues <- (sign(p-0.5)*(qgamma(abs(p - 0.5)*2, shape = 1/shape, scale = lambda))^(1/shape)*lambdaScale + mu)
  }
  else{
    lambda <- scale^(shape)
    gnormValues <- (sign(p-0.5)*qgamma(abs(p - 0.5)*2, shape = 1/shape, scale = lambda)^(1/shape) + mu)
  }

  return(gnormValues)
}

#' @rdname gnorm
#' @export
rgnorm <- function(n, mu = 0, scale = 1, shape = 1) {
  # A failsafe for NaN / NAs of scale / shape
  if(any(is.nan(scale))){
    scale[is.nan(scale)] <- 0
  }
  if(any(is.na(scale))){
    scale[is.na(scale)] <- 0
  }
  if(any(scale<0)){
    scale[scale<0] <- 0
  }
  if(any(is.nan(shape))){
    shape[is.nan(shape)] <- 0
  }
  if(any(is.na(shape))){
    shape[is.na(shape)] <- 0
  }

  gnormValues <- qgnorm(runif(n), mu=mu, scale=scale, shape=shape)
  # gnormValues <- qgamma(runif(n), shape = 1/shape, scale = scale^shape)^(1/shape)*((-1)^rbinom(n, 1, 0.5)) + mu
  return(gnormValues)
}
