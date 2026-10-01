#' Point likelihood values
#'
#' This function returns a vector of logarithms of likelihoods for each observation
#'
#' Instead of taking the expected log-likelihood for the whole series, this function
#' calculates the individual value for each separate observation. Note that these
#' values are biased, so you would possibly need to take number of degrees of freedom
#' into account in order to have an unbiased estimator.
#'
#' This value is based on the general likelihood (not its concentrated version), so
#' the sum of these values may slightly differ from the output of logLik.
#'
#' @aliases pointLik
#' @param object Time series model.
#' @param log Whether to take logarithm of likelihoods.
#' @param ...  Some stuff.
#' @return This function returns a vector.
#' @template author
#' @seealso \link[stats]{AIC}, \link[stats]{BIC}
#' @keywords htest
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' ourModel <- alm(y~x1+x2,as.data.frame(xreg))
#'
#' pointLik(ourModel)
#'
#' # Bias correction
#' pointLik(ourModel) - nparam(ourModel)
#'
#' # Bias correction in AIC style
#' 2*(nparam(ourModel)/nobs(ourModel) - pointLik(ourModel))
#'
#' # BIC calculation based on pointLik
#' log(nobs(ourModel))*nparam(ourModel) - 2*sum(pointLik(ourModel))
#'
#' @export pointLik
pointLik <- function(object, log=TRUE, ...) UseMethod("pointLik")

#' @export
pointLik.default <- function(object, log=TRUE, ...){
    likValues <- dnorm(residuals(object), mean=0, sd=sigma(object), log=log);

    return(likValues);
}

#' @export
pointLik.alm <- function(object, log=TRUE, ...){
    distribution <- object$distribution;
    y <- actuals(object);
    ot <- y!=0;
    occurrenceModel <- is.occurrence(object$occurrence);
    recursiveModel <- !is.null(object$other$arima);
    if(occurrenceModel){
        otU <- y!=0;
        y <- y[otU];
        mu <- object$mu[otU];
    }
    else{
        otU <- rep(TRUE, nobs(object));
        mu <- object$mu;
    }
    scale <- extractScale(object);
    # The scale of dnorm and dlnorm is the variance, while the densities need the standard deviation
    if(any(distribution==c("dnorm","dlnorm"))){
        scale <- sqrt(scale);
    }

    likValues <- vector("numeric",nobs(object));
    likValues[otU] <- switch(distribution,
                             "dnorm" = dnorm(y, mean=mu, sd=scale, log=log),
                             "dlnorm" = dlnorm(y, meanlog=mu, sdlog=scale, log=log),
                             "dgnorm" = dgnorm(y, mu=mu, scale=scale, shape=object$other$shape, log=log),
                             "dlgnorm" = dgnorm(log(y), mu=mu, scale=scale, shape=object$other$shape, log=log),
                             "dfnorm" = dfnorm(y, mu=mu, sigma=scale, log=log),
                             "drectnorm" = drectnorm(y, mu=mu, sigma=scale, log=log),
                             "dbcnorm" = dbcnorm(y, mu=mu, sigma=scale, lambda=object$other$lambdaBC, log=log),
                             "dlogitnorm" = dlogitnorm(y, mu=mu, sigma=scale, log=log),
                             "dexp" = dexp(y, rate=1/mu, log=log),
                             "dinvgauss" = dinvgauss(y, mean=mu, dispersion=scale/mu, log=log),
                             "dgamma" = dgamma(y, shape=1/scale, scale=scale*mu, log=log),
                             "dlaplace" = dlaplace(y, mu=mu, scale=scale, log=log),
                             "dllaplace" = dlaplace(log(y), mu=mu, scale=scale, log=log),
                             "dalaplace" = dalaplace(y, mu=mu, scale=scale, alpha=object$other$alpha, log=log),
                             "dlogis" = dlogis(y, location=mu, scale=scale, log=log),
                             "dt" = dt(y-mu, df=scale, log=log),
                             "ds" = ds(y, mu=mu, scale=scale, log=log),
                             "dls" = ds(log(y), mu=mu, scale=scale, log=log),
                             "dgeom" = dgeom(y, prob=1/(mu+1), log=log),
                             "dpois" = dpois(y, lambda=mu, log=log),
                             "dnbinom" = dnbinom(y, mu=mu, size=object$other$size, log=log),
                             "dbinom" = dbinom(y-occurrenceModel*1, prob=mu, size=object$other$size, log=log),
                             "dchisq" = dchisq(y, df=object$other$nu, ncp=mu, log=log),
                             "dbeta" = dbeta(y, shape1=mu, shape2=scale, log=log),
                             "plogis" = c(plogis(mu[ot], location=0, scale=1, log.p=TRUE),
                                          plogis(mu[!ot], location=0, scale=1, lower.tail=FALSE, log.p=TRUE)),
                             "pnorm" = c(pnorm(mu[ot], mean=0, sd=1, log.p=TRUE),
                                         pnorm(mu[!ot], mean=0, sd=1, lower.tail=FALSE, log.p=TRUE)),
                             0
    );
    if(any(distribution==c("dllaplace","dls","dlgnorm"))){
        likValues[otU] <- likValues[otU] - log(y);
    }

    # If this is a mixture model, take the respective probabilities into account
    if(occurrenceModel){
        # Add differential entropy. This should only be done for the recursive model
        if(recursiveModel){
            likValues[!otU] <- -switch(distribution,
                                       "dnorm" =,
                                       "dfnorm" =,
                                       "dbcnorm" =,
                                       "dlogitnorm" =,
                                       "dlnorm" = log(sqrt(2*pi)*scale)+0.5,
                                       "dexp" = 1,
                                       "dgnorm" =,
                                       "dlgnorm" = 1/object$other$shape -
                                           log(object$other$shape / (2*scale*gamma(1/object$other$shape))),
                                       "dinvgauss" = 0.5*(log(pi/2)+1+log(scale)),
                                       "dgamma" = 1/scale + log(scale) + log(gamma(1/scale)) + (1-1/scale)*digamma(1/scale),
                                       "dlaplace" =,
                                       "dllaplace" =,
                                       "dalaplace" = (1 + log(2*scale)),
                                       "dlogis" = 2,
                                       "dt" = ((scale+1)/2 *
                                                   (digamma((scale+1)/2)-digamma(scale/2)) +
                                                   log(sqrt(scale) * beta(scale/2,0.5))),
                                       "ds" = ,
                                       "dls" = (2 + 2*log(2*scale)),
                                       "dchisq" = (log(2)*gamma(scale/2)-
                                                       (1-scale/2)*digamma(scale/2)+
                                                       scale/2),
                                       "dbeta" = log(beta(mu,scale))-
                                           (mu-1)*
                                           (digamma(mu)-
                                                digamma(mu+scale))-
                                           (scale-1)*
                                           (digamma(scale)-
                                                digamma(mu+scale)),
                                       # This is a normal approximation of the real entropy
                                       "dpois" = 0.5*log(2*pi*scale)+0.5,
                                       "dnbinom" = log(sqrt(2*pi)*scale)+0.5,
                                       "dbinom" = 0.5*log(2*pi*object$other$size*object$mu[!otU]*(1-object$mu[!otU]))+0.5,
                                       0
            );
        }

        likValues <- likValues + pointLik(object$occurrence);
    }

    return(likValues);
}

#' @export
pointLik.ets <- function(object, log=TRUE, ...){

    likValues <- pointLik.default(object);
    if(errorType(object)=="M"){
        likValues[] <- likValues - log(abs(fitted(object)));
    }
    # This correction is needed so that logLik(object) ~ sum(pointLik(object))
    likValues[] <- likValues + 0.5 * (log(2 * pi) + 1 - log(nobs(object)));

    return(likValues);
}

#' @importFrom stats pgeom pnbinom ppois pbinom
pointLikCumulative <- function(object, ...){
    return(switch(object$distribution,
                  "dgeom"=pgeom(actuals(object), 1/(object$mu+1)),
                  "dpois"=ppois(actuals(object), lambda=object$mu),
                  "dnbinom"=pnbinom(actuals(object), mu=object$mu, size=object$other$size),
                  "dbinom"=pbinom(actuals(object), prob=1/(object$mu+1), size=object$other$size)));
}

#' Point AIC
#'
#' This function returns a vector of AIC values for the in-sample observations
#'
#' This is based on \link[greybox]{pointLik} function. The formula for this is:
#' pAIC_t = 2 * k - 2 * T * l_t ,
#' where k is the number of parameters, T is the number of observations and l_t is
#' the point likelihood. This way we preserve the property that AIC = mean(pAIC).
#'
#' @aliases pAIC
#' @param object Time series model.
#' @param ...  Some stuff.
#' @return The function returns the vector of point AIC values.
#' @template author
#' @seealso \link[greybox]{pointLik}
#' @keywords htest
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' ourModel <- alm(y~x1+x2,as.data.frame(xreg))
#'
#' pAICValues <- pAIC(ourModel)
#'
#' mean(pAICValues)
#' AIC(ourModel)
#'
#' @rdname pointIC
#' @export pAIC
pAIC <- function(object, ...) UseMethod("pAIC")

#' @export
pAIC.default <- function(object, ...){
    obs <- nobs(object);
    k <- nparam(object);
    return(2 * k - 2 * obs * pointLik(object));
}

#' @rdname pointIC
#' @export pAICc
pAICc <- function(object, ...) UseMethod("pAICc")

#' @export
pAICc.default <- function(object, ...){
    obs <- nobs(object);
    k <- nparam(object);
    return(2 * k - 2 * obs * pointLik(object) + 2 * k * (k + 1) / (obs - k - 1));
}

#' @rdname pointIC
#' @export pBIC
pBIC <- function(object, ...) UseMethod("pBIC")

#' @export
pBIC.default <- function(object, ...){
    obs <- nobs(object);
    k <- nparam(object);
    return(log(obs) * k - 2 * obs * pointLik(object));
}

#' @rdname pointIC
#' @export pBIC
pBICc <- function(object, ...) UseMethod("pBICc")

#' @export
pBICc.default <- function(object, ...){
    obs <- nobs(object);
    k <- nparam(object);
    return((k * log(obs) * obs) / (obs - k - 1)  - 2 * obs * pointLik(object));
}
