##### IC functions #####

#' Corrected Akaike's Information Criterion and Bayesian Information Criterion
#'
#' This function extracts AICc / BICc from models. It can be applied to wide
#' variety of models that use logLik() and nobs() methods (including the
#' popular lm, forecast, smooth classes).
#'
#' AICc was proposed by Nariaki Sugiura in 1978 and is used on small samples
#' for the models with normally distributed residuals. BICc was derived in
#' McQuarrie (1999) and is used in similar circumstances.
#'
#' IMPORTANT NOTE: both of the criteria can only be used for univariate models
#' (regression models, ARIMA, ETS etc) with normally distributed residuals!
#' In case of multivariate models, both criteria need to be modified. See
#' Bedrick & Tsai (1994) for details.
#'
#' @aliases AICc
#' @template author
#' @template AICRef
#'
#' @param object Time series model.
#' @param ...  Some stuff.
#' @return This function returns numeric value.
#' @seealso \link[stats]{AIC}, \link[stats]{BIC}
#' @references \itemize{
#' \item McQuarrie A.D., A small-sample correction for the Schwarz SIC
#' model selection criterion, Statistics & Probability Letters 44 (1999)
#' pp.79-86. \doi{10.1016/S0167-7152(98)00294-6}
#' \item Sugiura Nariaki (1978) Further analysts of the data by Akaike's
#' information criterion and the finite corrections, Communications in
#' Statistics - Theory and Methods, 7:1, 13-26,
#' \doi{10.1080/03610927808827599}
#' \item Bedrick, E. J., & Tsai, C.-L. (1994). Model Selection for
#' Multivariate Regression in Small Samples. Biometrics, 50(1), 226.
#' \doi{10.2307/2533213}
#' }
#' @keywords htest
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' ourModel <- stepwise(xreg)
#'
#' AICc(ourModel)
#' BICc(ourModel)
#'
#' @rdname InformationCriteria
#' @export AICc
AICc <- function(object, ...) UseMethod("AICc")

#' @rdname InformationCriteria
#' @aliases BICc
#' @export BICc
BICc <- function(object, ...) UseMethod("BICc")


#' @export
AICc.default <- function(object, ...){
    llikelihood <- logLik(object);
    nparamAll <- nparam(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    obs <- nobs(object);

    IC <- 2*nparamAll - 2*llikelihood + 2 * nparamAll * (nparamAll + 1) / (obs - nparamAll - 1);

    return(IC);
}

#' @export
BICc.default <- function(object, ...){
    llikelihood <- logLik(object);
    nparamAll <- nparam(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    obs <- nobs(object);

    IC <- - 2*llikelihood + (nparamAll * log(obs) * obs) / (obs - nparamAll - 1);

    return(IC);
}

#' @export
AICc.greybox <- function(object, ...){
    llikelihood <- logLik(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    obs <- nobs(object);

    # Correction is needed for the calculation of AIC in case of OLS et al (add scale).
    correction <- switch(object$loss,
                         "MSE"=switch(object$distribution,
                                      "dnorm"=1,
                                      0),
                         "MAE"=switch(object$distribution,
                                      "dlaplace"=1,
                                      0),
                         "HAM"=switch(object$distribution,
                                      "ds"=1,
                                      0),
                         0);

    nparamAll <- nparam(object) + correction;

    IC <- 2*nparamAll - 2*llikelihood + 2 * nparamAll * (nparamAll + 1) / (obs - nparamAll - 1);

    return(IC);
}

#' @export
BICc.greybox <- function(object, ...){
    llikelihood <- logLik(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    obs <- nobs(object);

    # Correction is needed for the calculation of AIC in case of OLS et al (add scale).
    correction <- switch(object$loss,
                         "MSE"=switch(object$distribution,
                                      "dnorm"=1,
                                      0),
                         "MAE"=switch(object$distribution,
                                      "dlaplace"=1,
                                      0),
                         "HAM"=switch(object$distribution,
                                      "ds"=1,
                                      0),
                         0);

    nparamAll <- nparam(object) + correction;

    IC <- - 2*llikelihood + (nparamAll * log(obs) * obs) / (obs - nparamAll - 1);

    return(IC);
}

#' @export
AICc.varest <- function(object, ...){
    llikelihood <- logLik(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    nSeries <- object$K;
    nparamAll <- nrow(coef(object)[[1]]);

    obs <- nobs(object);
    if(obs - (nparamAll + nSeries + 1) <=0){
        IC <- Inf;
    }
    else{
        IC <- -2*llikelihood + ((2*obs*(nparamAll*nSeries + nSeries*(nSeries+1)/2)) /
                                    (obs - (nparamAll + nSeries + 1)));
    }

    return(IC);
}

#' @export
BICc.varest <- function(object, ...){
    llikelihood <- logLik(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    nSeries <- object$K;
    nparamAll <- nrow(coef(object)[[1]]) + object$K;

    obs <- nobs(object);
    if(obs - (nparamAll + nSeries + 1) <=0){
        IC <- Inf;
    }
    else{
        IC <- -2*llikelihood + ((log(obs)*obs*(nparamAll*nSeries + nSeries*(nSeries+1)/2)) /
                                    (obs - (nparamAll + nSeries + 1)));
    }

    return(IC);
}

#' @export
extractAIC.alm <- function(fit, scale=NULL, k=2, ...){
    ellipsis <- list(...);

    # Correction is needed for the calculation of AIC in case of OLS et al (add scale).
    correction <- switch(fit$loss,
                         "MSE"=switch(fit$distribution,
                                      "dnorm"=1,
                                      0),
                         "MAE"=switch(fit$distribution,
                                      "dlaplace"=1,
                                      0),
                         "HAM"=switch(fit$distribution,
                                      "ds"=1,
                                      0),
                         0);

    if(!is.null(ellipsis$ic)){
        IC <- switch(ellipsis$ic,"AIC"=AIC,"BIC"=BIC,"BICc"=BICc,AICc);
        return(c(nparam(fit)+correction,IC(fit)));
    }
    else{
        return(c(nparam(fit)+correction,k*(nparam(fit)+correction)-2*logLik(fit)));
    }
}

#' Functions that extracts type of error from the model
#'
#' This function allows extracting error type from any model.
#'
#' \code{errorType} extracts the type of error from the model
#' (either additive or multiplicative).
#'
#' @template author
#' @template keywords
#'
#' @param object Model estimated using one of the functions of smooth package.
#' @param ... Currently nothing is accepted via ellipsis.
#' @return Either \code{"A"} for additive error or \code{"M"} for multiplicative.
#' All the other functions return strings of character.
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' ourModel <- alm(y~x1+x2,as.data.frame(xreg))
#'
#' errorType(ourModel)
#'
#' @export errorType
errorType <- function(object, ...) UseMethod("errorType")

#' @export
errorType.default <- function(object, ...){
    return("A");
}

#' @export
errorType.ets <- function(object, ...){
    if(substr(object$method,5,5)=="M"){
        return("M");
    }
    else{
        return("A");
    }
}

#' @importFrom stats logLik
#' @export
logLik.alm <- function(object, ...){
    if(is.occurrence(object$occurrence)){
        return(structure(object$logLik,nobs=nobs(object),df=nparam(object),class="logLik"));
    }
    else{
        # Correction is needed for the calculation of AIC in case of OLS et al (add scale).
        correction <- switch(object$loss,
                             "MSE"=switch(object$distribution,
                                          "dnorm"=1,
                                          0),
                             "MAE"=switch(object$distribution,
                                          "dlaplace"=1,
                                          0),
                             "HAM"=switch(object$distribution,
                                          "ds"=1,
                                          0),
                             0);
        return(structure(object$logLik,nobs=nobs(object),df=nparam(object)+correction,class="logLik"));
    }
}


#### Coefficients and extraction functions ####

#' Function extracts the actual values from the function
#'
#' This is a simple method that returns the values of the response variable of the model
#'
#' @template author
#'
#' @param object Model estimated using one of the functions of smooth package.
#' @param all If \code{FALSE}, then in the case of the occurrence model, only demand
#' sizes will be returned.
#' @param ... Other parameters to pass to the method. Currently nothing is supported here.
#' @return The vector of the response variable.
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' ourModel <- stepwise(xreg)
#'
#' actuals(ourModel)
#'
#' @rdname actuals
#' @export
actuals <- function(object, all=TRUE, ...) UseMethod("actuals")

#' @rdname actuals
#' @export
actuals.default <- function(object, all=TRUE, ...){
    return(object$y);
}

#' @rdname actuals
#' @export
actuals.lm <- function(object, all=TRUE, ...){
    return(object$model[,1]);
}

#' @rdname actuals
#' @export
actuals.alm <- function(object, all=TRUE, ...){
    if(all){
        return(object$data[,1])
    }
    else{
        return(object$data[object$data[,1]!=0,1]);
    }
}

#' @rdname actuals
#' @export
actuals.predict.greybox <- function(object, all=TRUE, ...){
    return(actuals(object$model, all=all, ...));
}

#' @export
formula.alm <- function(x, ...){
    return(x$call$formula);
}

#' Coefficients of the model and their statistics
#'
#' These are the basic methods for the alm and greybox models that extract coefficients,
#' their covariance matrix, confidence intervals or generating the summary of the model.
#' If the non-likelihood related loss was used in the process, then it is recommended to
#' use bootstrap (which is slow, but more reliable).
#'
#' The \code{coef()} method returns the vector of parameters of the model. If
#' \code{bootstrap=TRUE}, then the coefficients are calculated as the mean values of the
#' bootstrapped ones.
#'
#' The \code{vcov()} method returns the covariance matrix of parameters. If
#' \code{bootstrap=TRUE}, then the bootstrap is done using \link[greybox]{coefbootstrap}
#' function
#'
#' The \code{confint()} constructs the confidence intervals for parameters. Once again,
#' this can be done using \code{bootstrap=TRUE}.
#'
#' Finally, the \code{summary()} returns the table with parameters, their standard errors,
#' confidence intervals and general information about the model.
#'
#' @param object The model estimated using alm or other greybox function.
#' @param bootstrap The logical, which determines, whether to use bootstrap in the
#' process or not.
#' @param level The confidence level for the construction of the interval.
#' @param parm The parameters that need to be extracted.
#' @param ... Parameters passed to \link[greybox]{coefbootstrap} function.
#'
#' @return Depending on the used method, different values are returned.
#'
#' @template author
#' @template keywords
#'
#' @seealso \code{\link[greybox]{alm}, \link[greybox]{coefbootstrap}}
#'
#' @examples
#' # An example with ALM
#' ourModel <- alm(mpg~., mtcars, distribution="dlnorm")
#' coef(ourModel)
#' vcov(ourModel)
#' confint(ourModel)
#' summary(ourModel)
#'
#' @rdname coef.alm
#' @aliases coef.greybox coef.alm
#' @importFrom stats coef
#' @export
coef.greybox <- function(object, bootstrap=FALSE, ...){
    if(bootstrap){
        return(colMeans(coefbootstrap(object, ...)$coefficients));
    }
    else{
        return(object$coefficients);
    }
}

#' @export
coef.greyboxD <- function(object, ...){
    coefReturned <- list(coefficients=object$coefficients,
                         dynamic=object$coefficientsDynamic,importance=object$importance);
    return(structure(coefReturned,class="coef.greyboxD"));
}

#' @aliases confint.alm
#' @rdname coef.alm
#' @importFrom stats confint qt quantile
#' @export
confint.alm <- function(object, parm, level=0.95, bootstrap=FALSE, ...){

    confintNames <- c(paste0((1-level)/2*100,"%"),
                      paste0((1+level)/2*100,"%"));
    # Extract parameters
    parameters <- coef(object);
    if(!bootstrap){
        parametersSE <- sqrt(diag(vcov(object)));
        parametersNames <- names(parameters);
        # Add scale parameters if they were estimated
        if(is.scale(object$scale)){
            parameters <- c(parameters,coef(object$scale));
            parametersSE <- c(parametersSE, sqrt(diag(vcov(object$scale))));
            parametersNames <- names(parameters);
        }
        # Define quantiles using Student distribution
        paramQuantiles <- qt((1+level)/2,df=object$df.residual);

        # We can use normal distribution, because of the asymptotics of MLE
        confintValues <- cbind(parameters-paramQuantiles*parametersSE,
                               parameters+paramQuantiles*parametersSE);
        colnames(confintValues) <- confintNames;

        # Return S.E. as well, so not to repeat the thing twice...
        confintValues <- cbind(parametersSE, confintValues);
        # Give the name to the first column
        colnames(confintValues)[1] <- "S.E.";
        rownames(confintValues) <- parametersNames;
    }
    else{
        coefValues <- coefbootstrap(object, ...);
        confintValues <- cbind(sqrt(diag(coefValues$vcov)),
                               apply(coefValues$coefficients,2,quantile,probs=(1-level)/2),
                               apply(coefValues$coefficients,2,quantile,probs=(1+level)/2));
        colnames(confintValues) <- c("S.E.",confintNames);
    }

    # If parm was not provided, return everything.
    if(!exists("parm",inherits=FALSE)){
        parm <- c(1:length(parameters));
    }

    return(confintValues[parm,,drop=FALSE]);
}

#' @rdname coef.alm
#' @export
confint.scale <- confint.alm;

#' @export
confint.calm <- function(object, parm, level=0.95, ...){

    # Extract parameters
    parameters <- coef(object);
    # Extract SE
    parametersSE <- sqrt(abs(diag(vcov(object))));
    # Define quantiles using Student distribution
    paramQuantiles <- qt((1+level)/2,df=object$df.residual);
    # Do the stuff
    confintValues <- cbind(parameters-paramQuantiles*parametersSE,
                           parameters+paramQuantiles*parametersSE);
    confintNames <- c(paste0((1-level)/2*100,"%"),
                                 paste0((1+level)/2*100,"%"));
    colnames(confintValues) <- confintNames;
    # If parm was not provided, return everything.
    if(!exists("parm",inherits=FALSE)){
        parm <- names(parameters);
    }
    confintValues <- confintValues[parm,];
    if(!is.matrix(confintValues)){
        confintValues <- matrix(confintValues,1,2);
        colnames(confintValues) <- confintNames;
        rownames(confintValues) <- names(parameters);
    }
    return(confintValues);
}

#' @export
confint.greyboxD <- function(object, parm, level=0.95, ...){

    # Extract parameters
    parameters <- coef(object)$coefficients;
    # Extract SE
    parametersSE <- sqrt(abs(diag(vcov(object))));
    # Define quantiles using Student distribution
    paramQuantiles <- qt((1+level)/2,df=object$df.residual);
    # Do the stuff
    confintValues <- array(NA,c(length(parameters),2),
                           dimnames=list(dimnames(parameters)[[2]],
                                         c(paste0((1-level)/2*100,"%"),paste0((1+level)/2*100,"%"))));
    confintValues[,1] <- parameters-paramQuantiles*parametersSE;
    confintValues[,2] <- parameters+paramQuantiles*parametersSE;

    # If parm was not provided, return everything.
    if(!exists("parm",inherits=FALSE)){
        parm <- colnames(parameters);
    }
    return(confintValues[parm,]);
}

# This is needed for lmCombine and other functions, using fast regressions
#' @export
confint.lmGreybox <- function(object, parm, level=0.95, ...){
    # Extract parameters
    parameters <- coef(object);
    parametersSE <- sqrt(diag(vcov(object)));
    # Define quantiles using Student distribution
    paramQuantiles <- qt((1+level)/2,df=object$df.residual);

    # We can use normal distribution, because of the asymptotics of MLE
    confintValues <- cbind(parameters-qt((1+level)/2,df=object$df.residual)*parametersSE,
                           parameters+qt((1+level)/2,df=object$df.residual)*parametersSE);
    confintNames <- c(paste0((1-level)/2*100,"%"),
                                 paste0((1+level)/2*100,"%"));
    colnames(confintValues) <- confintNames;
    rownames(confintValues) <- names(parameters);

    if(!is.matrix(confintValues)){
        confintValues <- matrix(confintValues,1,2);
        colnames(confintValues) <- confintNames;
        rownames(confintValues) <- names(parameters);
    }

    # Return S.E. as well, so not to repeat the thing twice...
    confintValues <- cbind(parametersSE, confintValues);
    colnames(confintValues)[1] <- "S.E.";
    return(confintValues);
}

#' @importFrom stats nobs fitted
#' @export
nobs.alm <- function(object, ...){
    return(length(actuals(object, ...)));
}

#' @export
nobs.greybox <- function(object, ...){
    return(length(fitted(object)));
}

#' @export
nobs.varest <- function(object, ...){
    return(object$obs);
}

#' Number of parameters and number of variates in the model
#'
#' \code{nparam()} returns the number of estimated parameters in the model,
#' while \code{nvariate()} returns number of variates for the response
#' variable.
#'
#' \code{nparam()} is a very basic and a simple function which does what it says:
#' extracts number of estimated parameters in the model. \code{nvariate()} returns
#' number of variates (dimensions, columns) for the response variable (1 for the
#' univariate regression).
#'
#' @param object Time series model.
#' @param ... Some other parameters passed to the method.
#' @return Both functions return numeric values.
#' @template author
#' @seealso \link[stats]{nobs}, \link[stats]{logLik}
#' @keywords htest
#' @examples
#'
#' ### Simple example
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' ourModel <- lm(y~.,data=as.data.frame(xreg))
#'
#' nparam(ourModel)
#' nvariate(ourModel)
#'
#' @importFrom stats coef
#' @rdname nparam
#' @export nparam
nparam <- function(object, ...) UseMethod("nparam")

#' @export
nparam.default <- function(object, ...){
    # The length of the vector of parameters
    return(length(coef(object)));
}

#' @export
nparam.Arima <- function(object, ...){
    # If this is NA, the model was estimated via CSS
    if(is.na(logLik(object))){
        return(length(coef(object)))
    }
    else{
        # Get the df from the likelihood
        return(nparam(logLik(object)));
    }
}

#' @export
nparam.alm <- function(object, ...){
    # The number of parameters in the model + in the occurrence part
    if(is.occurrence(object$occurrence)){
        return(object$df+nparam(object$occurrence));
    }
    else{
        return(object$df);
    }
}

#' @export
nparam.logLik <- function(object, ...){
    # The length of the vector of parameters + variance
    return(attributes(object)$df);
}

#' @export
nparam.calm <- function(object, ...){
    # The length of the vector of parameters + variance
    return(sum(object$importance)+1);
}

#' @export
nparam.varest <- function(object, ...){
    ### This is the nparam per series
    # Parameters in all the matrices + the elements of the covariance matrix
    return(nrow(coef(object)[[1]])*object$K + 0.5*object$K*(object$K+1));
}

#' @rdname nparam
#' @export nvariate
nvariate <- function(object, ...) UseMethod("nvariate")

#' @export
nvariate.default <- function(object, ...){
    if(is.null(dim(actuals(object)))){
        return(1)
    }
    else{
        return(ncol(actuals(object)));
    }
}

#' @importFrom stats sigma
#' @export
sigma.greybox <- function(object, all=FALSE, ...){
    if(object$loss=="ROLE"){
        return(sqrt(meanFast(residuals(object)^2,
                             # df=nobs(object, all=all)-nparam(object),
                             trim=object$other$trim,
                             side="both")));
    }
    else{
        return(sqrt(sum(residuals(object)^2)/(nobs(object, all=all)-nparam(object))));
    }
}

#' @export
sigma.Arima <- function(object, ...){
    return(sqrt(object$sigma2));
}

#' @export
sigma.alm <- function(object, ...){
    if(any(object$distribution==c("plogis","pnorm"))){
        return(extractScale(object));
    }
    else if(any(object$distribution==c("dinvgauss","dgamma","dexp"))){
        return(sqrt(sum((residuals(object)-1)^2)/(nobs(object, ...)-nparam(object))));
    }
    else{
        return(sigma.greybox(object, ...));
    }
}

#' @export
sigma.ets <- function(object, ...){
    return(sqrt(object$sigma2));
}

#' @export
sigma.varest <- function(object, ...){
    # OLS estimate of Sigma, without the covariances
    return(t(residuals(object)) %*% residuals(object) / (nobs(object)-nparam(object)+object$K));
}

#' Functions to extract scale and standard error from a model
#'
#' Functions extract scale and the standard error of the residuals. Mainly needed for
#' the work with the model estimated via \link[greybox]{sm}.
#'
#' In case of a simpler model, the functions will return the scalar using
#' \code{sigma()} method. If the scale model was estimated, \code{extractScale()} and
#' \code{extractSigma()} will return the conditional scale and the conditional
#' standard error of the residuals respectively.
#'
#' The scale follows the ADAM monograph: it is the variance \eqn{\sigma^2} for the
#' Normal and Log-Normal distributions (\code{dnorm} and \code{dlnorm}), so that
#' \code{extractSigma()} returns its square root in these cases.
#'
#' @param object The model estimated using lm / alm / etc.
#' @param ... Other parameters (currently nothing).
#'
#' @return One of the two is returned, depending on the type of estimated model:
#' \itemize{
#' \item Scalar from \code{sigma()} method if the variance is assumed to be constant.
#' \item Vector of values if the scale model was estimated
#' }
#'
#' @template author
#' @template keywords
#'
#' @seealso \code{\link[greybox]{sm}}
#'
#' @examples
#' # Generate the data
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+sqrt(exp(0.8+0.2*xreg[,1]))*rnorm(100,0,1),
#'               xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' # Estimate the location and scale model
#' ourModel <- alm(y~., xreg, scale=~x1+x2)
#'
#' # Extract scale
#' extractScale(ourModel)
#' # Extract standard error
#' extractSigma(ourModel)
#'
#' @rdname extractScale
#' @export
extractScale <- function(object, ...) UseMethod("extractScale")

#' @rdname extractScale
#' @export
extractScale.default <- function(object, ...){
    return(sigma(object));
}

#' @rdname extractScale
#' @export
extractScale.greybox <- function(object, ...){
    if(is.scale(object$scale)){
        return(fitted(object$scale));
    }
    else{
        if(is.scale(object)){
            return(1);
            # return(fitted(object));
        }
        else{
            return(object$scale);
        }
    }
}

#' @rdname extractScale
#' @export
extractSigma <- function(object, ...) UseMethod("extractSigma")

#' @rdname extractScale
#' @export
extractSigma.default <- function(object, ...){
    return(sigma(object));
}

#' @rdname extractScale
#' @export
extractSigma.greybox <- function(object, ...){
    if(is.scale(object$scale)){
        return(switch(object$distribution,
                      "dnorm"=,
                      "dlnorm"=sqrt(extractScale(object)),
                      "dlogitnorm"=,
                      "dbcnorm"=,
                      "dfnorm"=,
                      "drectnorm"=,
                      "dinvgauss"=,
                      "dgamma"=extractScale(object),
                      "dlaplace"=,
                      "dllaplace"=sqrt(2*extractScale(object)),
                      "ds"=,
                      "dls"=sqrt(120*(extractScale(object)^4)),
                      "dgnorm"=,
                      "dlgnorm"=sqrt(extractScale(object)^2*gamma(3/object$other$shape) / gamma(1/object$other$shape)),
                      "dlogis"=extractScale(object)*pi/sqrt(3),
                      "dt"=1/sqrt(1-2/extractScale(object)),
                      "dalaplace"=extractScale(object)/sqrt((object$other$alpha^2*(1-object$other$alpha)^2)*
                          (object$other$alpha^2+(1-object$other$alpha)^2)),
                      # For now sigma is returned for: dpois, dnbinom, dchisq, dbeta and plogis, pnorm.
                      sigma(object)
                      ));
    }
    else{
        return(sigma(object));
    }
}

#' @aliases vcov.alm
#' @rdname coef.alm
#' @importFrom stats vcov
#' @export
vcov.alm <- function(object, bootstrap=FALSE, ...){
    nVariables <- length(coef(object));
    variablesNames <- names(coef(object));
    interceptIsNeeded <- any(variablesNames=="(Intercept)");
    ellipsis <- list(...);

    # Try the basic method, if not a bootstrap
    if(!bootstrap){
        # If the likelihood is not available, then this is a non-conventional loss
        if(is.na(logLik(object)) && (object$loss!="MSE")){
            warning(paste0("You used the non-likelihood compatible loss, so the covariance matrix might be incorrect. ",
                           "It is recommended to use bootstrap=TRUE option in this case."),
                    call.=FALSE);
        }

        # If there are ARIMA orders, define them.
        if(!is.null(object$other$arima)){
            arOrders <- object$other$orders[1];
            iOrders <- object$other$orders[2];
            maOrders <- object$other$orders[3];
        }
        else{
            arOrders <- iOrders <- maOrders <- 0;
        }

        # Analytical values for vcov
        if(iOrders==0 && maOrders==0 &&
           ((any(object$distribution==c("dnorm","dlnorm","dbcnorm","dlogitnorm")) & object$loss=="likelihood") ||
            object$loss=="MSE")){
            matrixXreg <- object$data;
            if(interceptIsNeeded){
                matrixXreg[,1] <- 1;
                colnames(matrixXreg)[1] <- "(Intercept)";
            }
            else{
                matrixXreg <- matrixXreg[,-1,drop=FALSE];
            }
            matrixXreg <- crossprod(matrixXreg);
            vcovMatrixTry <- try(chol2inv(chol(matrixXreg)), silent=TRUE);
            if(any(class(vcovMatrixTry)=="try-error")){
                warning(paste0("Choleski decomposition of covariance matrix failed, so we had to revert to the simple inversion.\n",
                               "The estimate of the covariance matrix of parameters might be inaccurate.\n"),
                        call.=FALSE);
                vcovMatrix <- try(solve(matrixXreg, diag(nVariables), tol=1e-20), silent=TRUE);

                # If the conventional approach failed, do bootstrap
                if(any(class(vcovMatrix)=="try-error")){
                    warning(paste0("Sorry, but the hessian is singular, so we could not invert it.\n",
                                   "Switching to bootstrap of covariance matrix of parameters.\n"),
                            call.=FALSE, immediate.=TRUE);
                    vcov <- coefbootstrap(object, ...)$vcov;
                }
                else{
                    # vcov <- object$scale^2 * vcovMatrix;
                    vcov <- sigma(object, all=FALSE)^2 * vcovMatrix;
                }
            }
            else{
                vcovMatrix <- vcovMatrixTry;
                # vcov <- object$scale^2 * vcovMatrix;
                vcov <- sigma(object, all=FALSE)^2 * vcovMatrix;
            }
            rownames(vcov) <- colnames(vcov) <- variablesNames;
        }
        # Analytical values in case of Poisson
        else if(iOrders==0 && maOrders==0 && object$distribution=="dpois"){
            matrixXreg <- object$data;
            if(interceptIsNeeded){
                matrixXreg[,1] <- 1;
                colnames(matrixXreg)[1] <- "(Intercept)";
            }
            else{
                matrixXreg <- matrixXreg[,-1,drop=FALSE];
            }
            obsInsample <- nobs(object);
            FIMatrix <- t(matrixXreg[1,,drop=FALSE]) %*% matrixXreg[1,,drop=FALSE] * object$mu[1];
            for(j in 2:obsInsample){
                FIMatrix[] <- FIMatrix + t(matrixXreg[j,,drop=FALSE]) %*% matrixXreg[j,,drop=FALSE] * object$mu[j];
            }

            # See if Choleski works... It sometimes fails, when we don't get to the max of likelihood.
            vcovMatrixTry <- try(chol2inv(chol(FIMatrix)), silent=TRUE);
            if(inherits(vcovMatrixTry,"try-error")){
                warning(paste0("Choleski decomposition of hessian failed, so we had to revert to the simple inversion.\n",
                               "The estimate of the covariance matrix of parameters might be inaccurate.\n"),
                        call.=FALSE, immediate.=TRUE);
                vcov <- try(solve(FIMatrix, diag(nVariables), tol=1e-20), silent=TRUE);

                # If the conventional approach failed, do bootstrap
                if(inherits(vcov,"try-error")){
                    warning(paste0("Sorry, but the hessian is singular, so we could not invert it.\n",
                                   "Switching to bootstrap of covariance matrix of parameters.\n"),
                            call.=FALSE, immediate.=TRUE);
                    vcov <- coefbootstrap(object, ...)$vcov;
                }
            }
            else{
                vcov <- vcovMatrixTry;
            }

            rownames(vcov) <- colnames(vcov) <- variablesNames;
        }
        # Fisher Information approach
        else{
            # Form the call for alm
            newCall <- object$call;
            # Tuning for srm, to call alm() instead
            # if(is.srm(object)){
            #     newCall[[1]] <- as.name("alm");
            #     newCall$folder <- NULL;
            # }
            if(interceptIsNeeded){
                newCall$formula <- as.formula(paste0("`",all.vars(newCall$formula)[1],"`~."));
            }
            else{
                newCall$formula <- as.formula(paste0("`",all.vars(newCall$formula)[1],"`~.-1"));
            }
            newCall$data <- object$data;
            newCall$subset <- object$subset;
            newCall$distribution <- object$distribution;
            if(object$loss=="custom"){
                newCall$loss <- object$lossFunction;
            }
            else{
                newCall$loss <- object$loss;
            }
            newCall$orders <- object$other$orders;
            newCall$parameters <- coef(object);
            newCall$scale <- object$scale;
            newCall$fast <- TRUE;
            if(any(object$distribution==c("dchisq","dt"))){
                newCall$nu <- object$other$nu;
            }
            else if(object$distribution=="dnbinom"){
                newCall$size <- object$other$size;
            }
            else if(object$distribution=="dalaplace"){
                newCall$alpha <- object$other$alpha;
            }
            else if(any(object$distribution==c("dfnorm","drectnorm"))){
                newCall$sigma <- object$other$sigma;
            }
            else if(object$distribution=="dbcnorm"){
                newCall$lambdaBC <- object$other$lambdaBC;
            }
            else if(any(object$distribution==c("dgnorm","dlgnorm"))){
                newCall$shape <- object$other$shape;
            }
            newCall$FI <- TRUE;
            # newCall$occurrence <- NULL;
            newCall$occurrence <- object$occurrence;
            # Include bloody ellipsis
            newCall <- as.call(c(as.list(newCall),substitute(ellipsis)));
            # Make sure that print_level is zero, not to print redundant things out
            newCall$print_level <- 0;

            # Recall alm to get hessian
            FIMatrix <- eval(newCall)$FI;
            # If any row contains all zeroes, then it means that the variable does not impact the likelihood
            brokenVariables <- apply(FIMatrix==0,1,all) | apply(is.nan(FIMatrix),1,any);
            # If there are issues, try the same stuff, but with a different step size for hessian
            if(any(brokenVariables)){
                newCall$stepSize <- .Machine$double.eps^(1/6);
                FIMatrix <- eval(newCall)$FI;
            }

            # See if Choleski works... It sometimes fails, when we don't get to the max of likelihood.
            vcovMatrixTry <- try(chol2inv(chol(FIMatrix)), silent=TRUE);
            if(inherits(vcovMatrixTry,"try-error")){
                warning(paste0("Choleski decomposition of hessian failed, so we had to revert to the simple inversion.\n",
                               "The estimate of the covariance matrix of parameters might be inaccurate.\n"),
                        call.=FALSE, immediate.=TRUE);
                vcovMatrixTry <- try(solve(FIMatrix, diag(nVariables), tol=1e-20), silent=TRUE);
                if(inherits(vcovMatrixTry,"try-error")){
                    vcov <- diag(1e+100,nVariables);
                }
                else{
                    vcov <- vcovMatrixTry;
                }
            }
            else{
                vcov <- vcovMatrixTry;
            }

            # If the conventional approach failed, do bootstrap
            if(inherits(vcovMatrixTry,"try-error")){
                warning(paste0("Sorry, but the hessian is singular, so we could not invert it.\n",
                               "Switching to bootstrap of covariance matrix of parameters.\n"),
                        call.=FALSE, immediate.=TRUE);
                vcov <- coefbootstrap(object, ...)$vcov;
            }
        }
    }
    else{
        vcov <- coefbootstrap(object, ...)$vcov;
    }

    if(nVariables>1){
        if(object$distribution=="dbeta"){
            dimnames(vcov) <- list(c(paste0("shape1_",variablesNames),paste0("shape2_",variablesNames)),
                                   c(paste0("shape1_",variablesNames),paste0("shape2_",variablesNames)));
        }
        else{
            dimnames(vcov) <- list(variablesNames,variablesNames);
        }
    }
    else{
        names(vcov) <- variablesNames;
    }

    # Sometimes the diagonal elements in the covariance matrix are negative because likelihood is not fully maximised...
    if(any(diag(vcov)<0)){
        diag(vcov) <- abs(diag(vcov));
    }
    return(vcov);
}

#' @export
vcov.calm <- function(object, ...){
    # xreg <- as.matrix(object$data);
    # xreg[,1] <- 1;
    # colnames(xreg)[1] <- "(Intercept)";
    #
    # vcovValue <- sigma(object)^2 * solve(t(xreg) %*% xreg) * object$importance %*% t(object$importance);
    # warning("The covariance matrix for combined models is approximate. Don't rely too much on that.",call.=FALSE);
    return(object$vcov);
}

#' @export
vcov.greyboxD <- function(object, ...){
    return(object$vcov);
    # xreg <- as.matrix(object$data);
    # xreg[,1] <- 1;
    # colnames(xreg)[1] <- "(Intercept)";
    # importance <- apply(object$importance,2,mean);
    #
    # vcovValue <- sigma(object)^2 * solve(t(xreg) %*% xreg) * importance %*% t(importance);
    # warning("The covariance matrix for combined models is approximate. Don't rely too much on that.",call.=FALSE);
    # return(vcovValue);
}

# This is needed for lmCombine and other functions, using fast regressions
#' @export
vcov.lmGreybox <- function(object, ...){
    vcov <- sigma(object)^2 * solve(crossprod(object$xreg));
    return(vcov);
}


#' @rdname coef.alm
#' @export
vcov.scale <- function(object, bootstrap=FALSE, ...){
    nVariables <- length(coef(object));
    variablesNames <- names(coef(object));
    interceptIsNeeded <- any(variablesNames=="(Intercept)");

    ellipsis <- list(...);

    # Form the call for scaler
    newCall <- object$call;
    newCall$data <- object$data[,-1,drop=FALSE];
    if(interceptIsNeeded){
        newCall$formula <- as.formula(paste0("~",paste(colnames(newCall$data),collapse="+")));
    }
    else{
        newCall$formula <- as.formula(paste0("~",paste(colnames(newCall$data),collapse="+"),"-1"));
    }
    newCall$subset <- object$subset;
    newCall$parameters <- coef(object);
    newCall$FI <- TRUE;
    # Include bloody ellipsis
    newCall <- as.call(c(as.list(newCall),substitute(ellipsis)));
    # Make sure that print_level is zero, not to print redundant things out
    newCall$print_level <- 0;

    # Recall alm to get hessian
    FIMatrix <- eval(newCall)$FI;
    # If any row contains all zeroes, then it means that the variable does not impact the likelihood
    brokenVariables <- apply(FIMatrix==0,1,all) | apply(is.nan(FIMatrix),1,any);
    # If there are issues, try the same stuff, but with a different step size for hessian
    if(any(brokenVariables)){
        newCall$stepSize <- .Machine$double.eps^(1/6);
        FIMatrix <- eval(newCall)$FI;
    }

    # See if Choleski works... It sometimes fails, when we don't get to the max of likelihood.
    vcovMatrixTry <- try(chol2inv(chol(FIMatrix)), silent=TRUE);
    if(inherits(vcovMatrixTry,"try-error")){
        warning(paste0("Choleski decomposition of hessian failed, so we had to revert to the simple inversion.\n",
                       "The estimate of the covariance matrix of parameters might be inaccurate.\n"),
                call.=FALSE, immediate.=TRUE);
        vcovMatrixTry <- try(solve(FIMatrix, diag(nVariables), tol=1e-20), silent=TRUE);
        if(inherits(vcovMatrixTry,"try-error")){
            vcov <- diag(1e+100,nVariables);
        }
        else{
            vcov <- FIMatrix;
        }
    }
    else{
        vcov <- vcovMatrixTry;
    }

    return(vcov);
}

#### Plot functions ####

#' Plots of the fit and residuals
#'
#' The function produces diagnostics plots for a \code{greybox} model
#'
#' The list of produced plots includes:
#' \enumerate{
#' \item Actuals vs Fitted values. Allows analysing, whether there are any issues in the fit.
#' Does the variability of actuals increase with the increase of fitted values? Is the relation
#' well captured? They grey line on the plot corresponds to the perfect fit of the model.
#' \item Standardised residuals vs Fitted. Plots the points and the confidence bounds
#' (red lines) for the specified confidence \code{level}. Useful for the analysis of outliers;
#' \item Studentised residuals vs Fitted. This is similar to the previous plot, but with the
#' residuals divided by the scales with the leave-one-out approach. Should be more sensitive
#' to outliers;
#' \item Absolute residuals vs Fitted. Useful for the analysis of heteroscedasticity;
#' \item Squared residuals vs Fitted - similar to (3), but with squared values;
#' \item Q-Q plot with the specified distribution. Can be used in order to see if the
#' residuals follow the assumed distribution. The type of distribution depends on the one used
#' in the estimation (see \code{distribution} parameter in \link[greybox]{alm});
#' \item Fitted over time. Plots actuals (black line), fitted values (purple line) and
#' prediction interval (red lines) of width \code{level}, but only in the case, when there
#' are some values lying outside of it. Can be used in order to make sure that the model
#' did not miss any important events over time;
#' \item Standardised residuals vs Time. Useful if you want to see, if there is autocorrelation or
#' if there is heteroscedasticity in time. This also shows, when the outliers happen;
#' \item Studentised residuals vs Time. Similar to previous, but with studentised residuals;
#' \item ACF of the residuals. Are the residuals autocorrelated? See \link[stats]{acf} for
#' details;
#' \item PACF of the residuals. No, really, are they autocorrelated? See \link[stats]{pacf}
#' for details;
#' \item Cook's distance over time. Shows influential observations. 0.5, 0.75 and 0.95 quantile
#' lines from Fisher's distribution are also plotted. If the value is above them then the
#' observation is influencial. This does not work well for non-normal distributions;
#' \item Absolute standardised residuals vs Fitted. Similar to the previous, but with absolute
#' values. This is more relevant to the models where scale is calculated as an absolute value of
#' something (e.g. Laplace);
#' \item Squared standardised residuals vs Fitted. This is an additional plot needed to diagnose
#' heteroscedasticity in a model with varying scale. The variance on this plot will be constant if
#' the adequate model for \code{scale} was constructed. This is more appropriate for normal and
#' the related distributions.
#' }
#' Which of the plots to produce, is specified via the \code{which} parameter. The plots 2, 3, 7,
#' 8 and 9 also use the parameters \code{level}, which specifies the confidence level for
#' the intervals.
#'
#' @param x Estimated greybox model.
#' @param which Which of the plots to produce. The possible options (see details for explanations):
#' \enumerate{
#' \item Actuals vs Fitted values;
#' \item Standardised residuals vs Fitted;
#' \item Studentised residuals vs Fitted;
#' \item Absolute residuals vs Fitted;
#' \item Squared residuals vs Fitted;
#' \item Q-Q plot with the specified distribution;
#' \item Fitted over time;
#' \item Standardised residuals over observations;
#' \item Studentised residuals over observations;
#' \item ACF of the residuals;
#' \item PACF of the residuals;
#' \item Cook's distance over observations with lines corresponding to heuristic values of
#' 4/n, 4/(n-k) and 1;
#' \item Absolute standardised residuals vs Fitted;
#' \item Squared standardised residuals vs Fitted;
#' \item ACF of the squared residuals;
#' \item PACF of the squared residuals.
#' }
#' @param level Confidence level. Defines width of confidence interval. Used in plots (2), (3), (7),
#' (8), (9), (10) and (11).
#' @param legend If \code{TRUE}, then the legend is produced on plots (2), (3) and (7).
#' @param ask Logical; if \code{TRUE}, the user is asked to press Enter before each plot.
#' @param lowess Logical; if \code{TRUE}, LOWESS lines are drawn on scatterplots, see \link[stats]{lowess}.
#' @param ... The parameters passed to the plot functions. Recommended to use with separate plots.
#' @return The function produces the number of plots, specified in the parameter \code{which}.
#'
#' @template author
#' @seealso \link[stats]{plot.lm}, \link[stats]{rstandard}, \link[stats]{rstudent}
#' @keywords ts univar
#' @examples
#'
#' xreg <- cbind(rlaplace(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rlaplace(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' ourModel <- alm(y~x1+x2, xreg, distribution="dnorm")
#'
#' par(mfcol=c(4,4))
#' plot(ourModel, c(1:14))
#'
#' @importFrom stats ppoints qqline qqnorm qqplot acf pacf lowess qf na.pass
#' @importFrom grDevices dev.interactive devAskNewPage grey
#' @aliases plot.alm
#' @export
plot.greybox <- function(x, which=c(1,2,4,6), level=0.95, legend=FALSE,
                         ask=prod(par("mfcol")) < length(which) && dev.interactive(),
                         lowess=TRUE, ...){

    # Define, whether to wait for the hit of "Enter"
    if(ask){
        oask <- devAskNewPage(TRUE);
        on.exit(devAskNewPage(oask));
    }

    # Special treatment for count distributions
    countDistribution <- any(x$distribution==c("dpois","dnbinom","dbinom","dgeom"));

    # Warn if the diagnostis will be done for scale
    if(is.scale(x$scale) && any(which %in% c(2:6,8,9,13:14))){
        message("Note that residuals diagnostics plots are produced for scale model");
    }

    if(countDistribution && any(which %in% c(3, 9))){
        warning("Studentised residuals are not supported for count distributions. Switching to standardised.",
                call.=FALSE);
        which[which==3] <- 2;
        which[which==9] <- 8;
    }

    # 1. Fitted vs Actuals values
    plot1 <- function(x, ...){
        ellipsis <- list(...);

        if(countDistribution){
            # Get the actuals and the fitted values
            ellipsis$y <- as.vector(pointLik(x, log=FALSE));
            if(is.occurrence(x)){
                if(any(x$distribution==c("plogis","pnorm"))){
                    ellipsis$y <- (ellipsis$y!=0)*1;
                }
            }
            ellipsis$x <- as.vector(1/fitted(x));

            # Title
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "Probability vs Likelihood";
            }
            if(!any(names(ellipsis)=="ylab")){
                ellipsis$ylab <- paste0("Likelihood of ",all.vars(x$call$formula)[1]);
            }
            if(!any(names(ellipsis)=="xlab")){
                ellipsis$xlab <- "Probability";
            }
        }
        else{
            # Get the actuals and the fitted values
            ellipsis$y <- as.vector(actuals(x));
            if(is.occurrence(x)){
                if(any(x$distribution==c("plogis","pnorm"))){
                    ellipsis$y <- (ellipsis$y!=0)*1;
                }
            }
            ellipsis$x <- as.vector(fitted(x));

            # Title
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "Actuals vs Fitted";
            }
            if(!any(names(ellipsis)=="ylab")){
                ellipsis$ylab <- paste0("Actual ",all.vars(x$call$formula)[1]);
            }
            if(!any(names(ellipsis)=="xlab")){
                ellipsis$xlab <- "Fitted";
            }
        }

        # If this is a mixture model, remove zeroes
        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[ellipsis$y!=0];
            ellipsis$y <- ellipsis$y[ellipsis$y!=0];
        }

        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$x)];
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
        }
        if(any(is.na(ellipsis$y))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$y)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }
        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "p";
        }
        # xlim and ylim
        # if(!any(names(ellipsis)=="xlim")){
        #     ellipsis$xlim <- range(c(ellipsis$x,ellipsis$y),na.rm=TRUE);
        # }
        # if(!any(names(ellipsis)=="ylim")){
        #     ellipsis$ylim <- range(c(ellipsis$x,ellipsis$y),na.rm=TRUE);
        # }

        # Start plotting
        do.call(plot,ellipsis);
        abline(a=0,b=1,col="grey",lwd=2,lty=2)
        if(lowess){
            lines(lowess(ellipsis$x, ellipsis$y), col=2);
        }
    }

    # 2 and 3: Standardised  / studentised residuals vs Fitted
    plot2 <- function(x, type="rstandard", ...){
        ellipsis <- list(...);

        # Amend to do analysis of residuals of scale model
        if(is.scale(x$scale)){
            x <- x$scale;
        }

        ellipsis$x <- as.vector(fitted(x));
        if(type=="rstandard"){
            ellipsis$y <- as.vector(rstandard(x));
            yName <- "Standardised";
        }
        else{
            ellipsis$y <- as.vector(rstudent(x));
            yName <- "Studentised";
        }

        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- paste0(yName," Residuals vs Fitted");
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- paste0(yName," Residuals");
        }

        if(legend){
            if(ellipsis$x[length(ellipsis$x)]>mean(ellipsis$x)){
                legendPosition <- "bottomright";
            }
            else{
                legendPosition <- "topright";
            }
        }

        # Get the IDs of outliers and statistic
        outliers <- outlierdummy(x, level=level, type=type);
        outliersID <- outliers$id;
        statistic <- outliers$statistic;
        # Analyse stuff in logarithms if the distribution is dinvgauss / dgamma / dexp
        if(any(x$distribution==c("dinvgauss","dgamma","dexp"))){
            ellipsis$y[] <- log(ellipsis$y);
            statistic[] <- log(statistic);
        }
        # Substitute zeroes with NAs if there was an occurrence
        if(is.occurrence(x$occurrence)){
            ellipsis$x[actuals(x$occurrence)==0] <- NA;
        }

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(c(ellipsis$y[is.finite(ellipsis$y)],statistic), na.rm=TRUE)*1.2;
            if(legend){
                if(legendPosition=="bottomright"){
                    ellipsis$ylim[1] <- ellipsis$ylim[1] - 0.2*diff(ellipsis$ylim);
                }
                else{
                    ellipsis$ylim[2] <- ellipsis$ylim[2] + 0.2*diff(ellipsis$ylim);
                }
            }
        }

        # If there are infinite values, mark them on the plot
        infiniteValues <- any(is.infinite(ellipsis$y));
        if(infiniteValues){
            infiniteMarkers <- ellipsis$y[is.infinite(ellipsis$y)];
            infiniteMarkersIDs <- which(is.infinite(ellipsis$y));
        }

        xRange <- range(ellipsis$x, na.rm=TRUE);
        xRange[1] <- xRange[1] - sd(ellipsis$x, na.rm=TRUE);
        xRange[2] <- xRange[2] + sd(ellipsis$x, na.rm=TRUE);

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        polygon(c(xRange,rev(xRange)),c(statistic[1],statistic[1],statistic[2],statistic[2]),
                col="lightgrey", border=NA, density=10);
        abline(h=statistic, col=2, lty=2);
        if(length(outliersID)>0){
            points(ellipsis$x[outliersID], ellipsis$y[outliersID], pch=16);
            text(ellipsis$x[outliersID], ellipsis$y[outliersID], labels=outliersID, pos=(ellipsis$y[outliersID]>0)*2+1);
        }
        if(lowess){
            # Remove NAs
            if(any(is.na(ellipsis$x))){
                ellipsis$y <- ellipsis$y[!is.na(ellipsis$x)];
                ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
            }
            lines(lowess(ellipsis$x, ellipsis$y), col=2);
        }
        # Markers for infinite values
        if(infiniteValues){
            points(ellipsis$x[infiniteMarkersIDs], ellipsis$ylim[(infiniteMarkers>0)+1]+0.1, pch=24, bg=2);
            text(ellipsis$x[infiniteMarkersIDs], ellipsis$ylim[(infiniteMarkers>0)+1]+0.1,
                 labels=infiniteMarkersIDs, pos=1);
        }

        if(legend){
            basicColour <- ellipsis$col;
            if(is.null(basicColour)){
                basicColour <- 1;
            }

            if(lowess){
                legend(legendPosition,
                       legend=c(paste0(round(level,3)*100,"% bounds"),"outside the bounds","LOWESS line"),
                       col=c(2, basicColour, 2), lwd=c(1,NA,1), lty=c(2,1,1), pch=c(NA,16,NA));
            }
            else{
                legend(legendPosition,
                       legend=c(paste0(round(level,3)*100,"% bounds"),"outside the bounds"),
                       col=c(2, basicColour), lwd=c(1,NA), lty=c(2,1), pch=c(NA,16));
            }
        }
    }

    # 4 and 5. Fitted vs |Residuals| or Fitted vs Residuals^2
    plot3 <- function(x, type="abs", ...){
        ellipsis <- list(...);

        # Amend to do analysis of residuals of scale model
        if(is.scale(x$scale)){
            x <- x$scale;
        }

        ellipsis$x <- as.vector(fitted(x));
        ellipsis$y <- as.vector(residuals(x));
        if(any(x$distribution==c("dinvgauss","dgamma","dexp"))){
            ellipsis$y[] <- log(ellipsis$y);
        }
        if(type=="abs"){
            ellipsis$y[] <- abs(ellipsis$y);
        }
        else{
            ellipsis$y[] <- ellipsis$y^2;
        }

        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[ellipsis$y!=0];
            ellipsis$y <- ellipsis$y[ellipsis$y!=0];
        }
        if(!any(names(ellipsis)=="main")){
            if(type=="abs"){
                ellipsis$main <- "|Residuals| vs Fitted";
            }
            else{
                ellipsis$main <- "Residuals^2 vs Fitted";
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="abs"){
                ellipsis$ylab <- "|Residuals|";
            }
            else{
                ellipsis$ylab <- "Residuals^2";
            }
        }

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        if(lowess){
            lines(lowess(ellipsis$x, ellipsis$y), col=2);
        }
    }

    # 6. Q-Q with the specified distribution
    plot4 <- function(x, ...){
        ellipsis <- list(...);

        # Amend to do analysis of residuals of scale model
        if(is.scale(x$scale)){
            x <- x$scale;
        }

        ellipsis$y <- as.vector(residuals(x));
        if(is.occurrence(x$occurrence)){
            ellipsis$y <- ellipsis$y[actuals(x$occurrence)!=0];
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Theoretical Quantile";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- "Actual Quantile";
        }

        # Number of points for pp-plots
        nsim <- 200;

        # For count distribution, we do manual construction
        if(countDistribution){
            if(!any(names(ellipsis)=="xlim")){
                ellipsis$xlim <- c(0,1);
            }
            if(!any(names(ellipsis)=="ylim")){
                ellipsis$ylim <- c(0,1);
            }
            ellipsis$x <- ppoints(nsim);

            if(x$distribution=="dpois"){
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Poisson distribution";
                }
                # ellipsis$x <- actuals(x)-qpois(ppoints(nobs(x)*100), lambda=x$mu);
                # ellipsis$x <- qpois(ppoints(nobs(x)*100), lambda=x$mu);
                # ellipsis$y[] <- actuals(x);

                # Produce matrix of quantiles
                yQuant <- matrix(qpois(ppoints(nsim), lambda=rep(x$mu, each=nsim)),
                                 nrow=nsim, ncol=nobs(x),
                                 dimnames=list(ppoints(nsim), NULL));
            }
            else if(x$distribution=="dnbinom"){
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Negative Binomial distribution";
                }
                # ellipsis$x <- actuals(x)-qnbinom(ppoints(nsim), mu=x$mu, size=extractScale(x));
                #
                # do.call(qqplot, ellipsis);
                # qqline(ellipsis$y, distribution=function(p) qnbinom(p, mu=x$mu, size=extractScale(x))-actuals(x));


                # Produce matrix of quantiles
                yQuant <- matrix(qnbinom(ppoints(nsim), mu=rep(x$mu, each=nsim),
                                         size=rep(extractScale(x), each=nsim)),
                                 nrow=nsim, ncol=nobs(x),
                                 dimnames=list(ppoints(nsim), NULL));
                # message("Sorry, but we don't produce QQ plots for the Negative Binomial distribution");
            }
            else if(x$distribution=="dbinom"){
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Binomial distribution";
                }

                # Produce matrix of quantiles
                yQuant <- matrix(qbinom(ppoints(nsim), prob=rep(1/(1+x$mu), each=nsim),
                                         size=rep(x$other$size, each=nsim)),
                                 nrow=nsim, ncol=nobs(x),
                                 dimnames=list(ppoints(nsim), NULL));
            }
            else if(x$distribution=="dgeom"){
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Geometric distribution";
                }

                # Produce matrix of quantiles
                yQuant <- matrix(qgeom(ppoints(nsim), prob=rep(1/(1+x$mu), each=nsim)),
                                 nrow=nsim, ncol=nobs(x),
                                 dimnames=list(ppoints(nsim), NULL));
            }
            # Get empirical probabilities
            ellipsis$y <- apply(matrix(actuals(x), nsim, nobs(x), byrow=T) <= yQuant, 1, sum) / nobs(x);

            # Remove zeroes not to contaminate the plot
            ellipsis$x <- ellipsis$x[ellipsis$y>mean(actuals(x)==0)];
            ellipsis$y <- ellipsis$y[ellipsis$y>mean(actuals(x)==0)];

            do.call(plot, ellipsis);
            abline(a=0, b=1);
        }
        # For the others, it is just a qqplot
        else{
            if(any(x$distribution==c("dnorm","dlnorm","dbcnorm","dlogitnorm","plogis","pnorm"))){
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ plot of normal distribution";
                }

                do.call(qqnorm, ellipsis);
                qqline(ellipsis$y);
            }
            else if(x$distribution=="dfnorm"){
                # Standardise residuals
                ellipsis$y[] <- ellipsis$y / sd(ellipsis$y);
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Folded Normal distribution";
                }
                ellipsis$x <- qfnorm(ppoints(nsim), mu=0, sigma=extractScale(x));

                do.call(qqplot, ellipsis);
                qqline(ellipsis$y, distribution=function(p) qfnorm(p, mu=0, sigma=extractScale(x)));
            }
            else if(x$distribution=="drectnorm"){
                # Standardise residuals
                ellipsis$y[] <- ellipsis$y / sd(ellipsis$y);
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Rectified Normal distribution";
                }
                ellipsis$x <- qrectnorm(ppoints(nsim), mu=0, sigma=extractScale(x));

                do.call(qqplot, ellipsis);
                qqline(ellipsis$y, distribution=function(p) qrectnorm(p, mu=0, sigma=extractScale(x)));
            }
            else if(any(x$distribution==c("dgnorm","dlgnorm"))){
                # Standardise residuals
                ellipsis$y[] <- ellipsis$y / sd(ellipsis$y);
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Generalised Normal distribution";
                }
                ellipsis$x <- qgnorm(ppoints(nsim), mu=0, scale=extractScale(x), shape=x$other$shape);

                do.call(qqplot, ellipsis);
                qqline(ellipsis$y, distribution=function(p) qgnorm(p, mu=0, scale=extractScale(x), shape=x$other$shape));
            }
            else if(any(x$distribution==c("dlaplace","dllaplace"))){
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Laplace distribution";
                }
                ellipsis$x <- qlaplace(ppoints(nsim), mu=0, scale=extractScale(x));

                do.call(qqplot, ellipsis);
                qqline(ellipsis$y, distribution=function(p) qlaplace(p, mu=0, scale=extractScale(x)));
            }
            else if(x$distribution=="dalaplace"){
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- paste0("QQ-plot of Asymmetric Laplace distribution with alpha=",round(x$other$alpha,3));
                }
                ellipsis$x <- qalaplace(ppoints(nsim), mu=0, scale=extractScale(x), alpha=x$other$alpha);

                do.call(qqplot, ellipsis);
                qqline(ellipsis$y, distribution=function(p) qalaplace(p, mu=0, scale=extractScale(x), alpha=x$other$alpha));
            }
            else if(x$distribution=="dlogis"){
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Logistic distribution";
                }
                ellipsis$x <- qlogis(ppoints(nsim), location=0, scale=extractScale(x));

                do.call(qqplot, ellipsis);
                qqline(ellipsis$y, distribution=function(p) qlogis(p, location=0, scale=extractScale(x)));
            }
            else if(any(x$distribution==c("ds","dls"))){
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of S distribution";
                }
                ellipsis$x <- qs(ppoints(nsim), mu=0, scale=extractScale(x));

                do.call(qqplot, ellipsis);
                qqline(ellipsis$y, distribution=function(p) qs(p, mu=0, scale=extractScale(x)));
            }
            else if(x$distribution=="dt"){
                # Standardise residuals
                ellipsis$y[] <- ellipsis$y / sd(ellipsis$y);
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Student's distribution";
                }
                ellipsis$x <- qt(ppoints(nsim), df=extractScale(x));

                do.call(qqplot, ellipsis);
                qqline(ellipsis$y, distribution=function(p) qt(p, df=extractScale(x)));
            }
            else if(x$distribution=="dinvgauss"){
                # Transform residuals for something meaningful
                # This is not 100% accurate, because the dispersion should change as well as mean...
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Inverse Gaussian distribution";
                }
                ellipsis$x <- qinvgauss(ppoints(nsim), mean=1, dispersion=extractScale(x));

                do.call(qqplot, ellipsis);
                qqline(ellipsis$y, distribution=function(p) qinvgauss(p, mean=1, dispersion=extractScale(x)));
            }
            else if(x$distribution=="dgamma"){
                # Transform residuals for something meaningful
                # This is not 100% accurate, because the scale should change together with mean...
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Gamma distribution";
                }
                ellipsis$x <- qgamma(ppoints(nsim), shape=1/extractScale(x), scale=extractScale(x));

                do.call(qqplot, ellipsis);
                qqline(ellipsis$y, distribution=function(p) qgamma(p, shape=1/extractScale(x), scale=extractScale(x)));
            }
            else if(x$distribution=="dexp"){
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Exponential distribution";
                }
                ellipsis$x <- qexp(ppoints(nsim), rate=x$scale);

                do.call(qqplot, ellipsis);
                qqline(ellipsis$y, distribution=function(p) qexp(p, rate=x$scale));
            }
            else if(x$distribution=="dchisq"){
                message("Sorry, but we don't produce QQ plots for the Chi-Squared distribution");
            }
            else if(x$distribution=="dbeta"){
                message("Sorry, but we don't produce QQ plots for the Beta distribution");
            }
        }
    }

    # 7. Linear graph,
    plot5 <- function(x, ...){
        ellipsis <- list(...);
        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- "Fit over time";
        }

        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "l";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- all.vars(x$call$formula)[1];
        }
        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Time";
        }

        # Get the actuals and the fitted values
        ellipsis$x <- actuals(x);
        if(is.alm(x)){
            if(any(x$distribution==c("plogis","pnorm"))){
                ellipsis$x <- (ellipsis$x!=0)*1;
            }
        }
        yFitted <- fitted(x);

        if(legend){
            if(yFitted[length(yFitted)]>mean(yFitted)){
                legendPosition <- "bottomright";
            }
            else{
                legendPosition <- "topright";
            }
            if(!any(names(ellipsis)=="ylim")){
                ellipsis$ylim <- range(c(actuals(x),yFitted),na.rm=TRUE);
                if(legendPosition=="bottomright"){
                    ellipsis$ylim[1] <- ellipsis$ylim[1] - 0.2*diff(ellipsis$ylim);
                }
                else{
                    ellipsis$ylim[2] <- ellipsis$ylim[2] + 0.2*diff(ellipsis$ylim);
                }
            }
        }

        # Start plotting
        do.call(plot,ellipsis);
        lines(yFitted, col=2);

        if(legend){
            basicColour <- ellipsis$col;
            if(is.null(basicColour)){
                basicColour <- 1;
            }
            legend(legendPosition,legend=c("Actuals","Fitted"),
                   col=c(basicColour,2), lwd=rep(1,2), lty=c(1,1));
        }
    }

    # 8 and 9. Standardised / Studentised residuals vs time
    plot6 <- function(x, type="rstandard", ...){
        ellipsis <- list(...);

        # Amend to do analysis of residuals of scale model
        if(is.scale(x$scale)){
            x <- x$scale;
        }

        if(type=="rstandard"){
            ellipsis$x <- rstandard(x);
            yName <- "Standardised";
        }
        else{
            ellipsis$x <- rstudent(x);
            yName <- "Studentised";
        }

        # If there is occurrence part, substitute zeroes with NAs
        if(is.occurrence(x$occurrence)){
            ellipsis$x[actuals(x$occurrence)==0] <- NA;
        }

        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- paste0(yName," Residuals over observations");
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Observations";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- paste0(yName," Residuals");
        }

        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "l";
        }

        # Get the IDs of outliers and statistic
        outliers <- outlierdummy(x, level=level, type=type);
        outliersID <- outliers$id;
        statistic <- outliers$statistic;
        # Analyse stuff in logarithms if the distribution is dinvgauss
        if(any(x$distribution==c("dinvgauss","dgamma","dexp"))){
            ellipsis$x[] <- log(ellipsis$x);
            statistic[] <- log(statistic);
        }

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(c(ellipsis$x[is.finite(ellipsis$x)],statistic),na.rm=TRUE)*1.2;
        }

        if(legend){
            legendPosition <- "topright";
            ellipsis$ylim[2] <- ellipsis$ylim[2] + 0.2*diff(ellipsis$ylim);
            ellipsis$ylim[1] <- ellipsis$ylim[1] - 0.2*diff(ellipsis$ylim);
        }

        # If there are infinite values, mark them on the plot
        infiniteValues <- any(is.infinite(ellipsis$x));
        if(infiniteValues){
            infiniteMarkers <- ellipsis$x[is.infinite(ellipsis$x)];
            infiniteMarkersIDs <- which(is.infinite(ellipsis$x));
        }

        # Start plotting
        do.call(plot,ellipsis);
        if(is.occurrence(x$occurrence)){
            points(ellipsis$x);
        }
        if(length(outliersID)>0){
            points(outliersID, ellipsis$x[outliersID], pch=16);
            text(outliersID, ellipsis$x[outliersID], labels=outliersID, pos=(ellipsis$x[outliersID]>0)*2+1);
        }
        # If there is occurrence model, plot points to fill in breaks
        if(is.occurrence(x$occurrence)){
            points(time(ellipsis$x), ellipsis$x);
        }
        if(lowess){
            # Substitute NAs with the mean
            if(any(is.na(ellipsis$x))){
                ellipsis$x[is.na(ellipsis$x)] <- mean(ellipsis$x, na.rm=TRUE);
            }
            lines(lowess(c(1:length(ellipsis$x)),ellipsis$x), col=2);
        }
        abline(h=0, col="grey", lty=2);
        abline(h=statistic[1], col=2, lty=2);
        abline(h=statistic[2], col=2, lty=2);
        polygon(c(1:nobs(x), c(nobs(x):1)),
                c(rep(statistic[1],nobs(x)), rep(statistic[2],nobs(x))),
                col="lightgrey", border=NA, density=10);

        # Markers for infinite values
        if(infiniteValues){
            points(infiniteMarkersIDs, ellipsis$ylim[(infiniteMarkers>0)+1]+0.1, pch=24, bg=2);
            text(infiniteMarkersIDs, ellipsis$ylim[(infiniteMarkers>0)+1]+0.1,
                 labels=infiniteMarkersIDs, pos=1);
        }
        if(legend){
            basicColour <- ellipsis$col;
            if(is.null(basicColour)){
                basicColour <- 1;
            }

            legend(legendPosition,legend=c("Residuals",paste0(level*100,"% prediction interval")),
                   col=c(basicColour,2), lwd=rep(1,3), lty=c(1,1,2));
        }
    }

    # 10 and 11. ACF and PACF
    plot7 <- function(x, type="acf", squared=FALSE, ...){
        ellipsis <- list(...);

        if(!any(names(ellipsis)=="main")){
            if(type=="acf"){
                if(squared){
                    ellipsis$main <- "Autocorrelation Function of Squared Residuals";
                }
                else{
                    ellipsis$main <- "Autocorrelation Function of Residuals";
                }
            }
            else{
                if(squared){
                    ellipsis$main <- "Partial Autocorrelation Function of Squared Residuals";
                }
                else{
                    ellipsis$main <- "Partial Autocorrelation Function of Residuals";
                }
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Lags";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="acf"){
                ellipsis$ylab <- "ACF";
            }
            else{
                ellipsis$ylab <- "PACF";
            }
        }

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- c(-1,1);
        }

        if(squared){
            if(type=="acf"){
                theValues <- acf(as.vector(residuals(x)^2), plot=FALSE, na.action=na.pass)
            }
            else{
                theValues <- pacf(as.vector(residuals(x)^2), plot=FALSE, na.action=na.pass);
            }
        }
        else{
            if(type=="acf"){
                theValues <- acf(as.vector(residuals(x)), plot=FALSE, na.action=na.pass)
            }
            else{
                theValues <- pacf(as.vector(residuals(x)), plot=FALSE, na.action=na.pass);
            }
        }
        ellipsis$x <- switch(type,
                             "acf"=theValues$acf[-1],
                             "pacf"=theValues$acf);
        statistic <- qnorm(c((1-level)/2, (1+level)/2),0,sqrt(1/nobs(x)));

        ellipsis$type <- "h"

        do.call(plot,ellipsis);
        abline(h=0, col=1, lty=1);
        abline(h=statistic, col=2, lty=2);
        if(any(ellipsis$x>statistic[2] | ellipsis$x<statistic[1])){
            outliers <- which(ellipsis$x >statistic[2] | ellipsis$x <statistic[1]);
            points(outliers, ellipsis$x[outliers], pch=16);
            text(outliers, ellipsis$x[outliers], labels=outliers, pos=(ellipsis$x[outliers]>0)*2+1);
        }
    }

    # 12. Cook's distance over time
    plot8 <- function(x, ...){
        ellipsis <- list(...);
        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- "Cook's distance over observations";
        }

        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "h";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- "Cook's distance";
        }
        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Observation";
        }

        # Get the cook's distance. Take abs() just in case... Not a very reasonable thing to do...
        ellipsis$x <- abs(cooks.distance(x));
        # thresholdsF <- qf(c(0.5,0.75,0.95), nparam(x), nobs(x)-nparam(x))
        thresholdsF <- c(4/nobs(x), 4/(nobs(x)-nparam(x)), 1);
        thresholdsColours <- c(2,2,2);
        thresholdsLty <- c(3,2,5);
        thresholdsLwd <- c(1,1,2);
        outliersID <- which(ellipsis$x>=thresholdsF[2]);

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(c(ellipsis$x[is.finite(ellipsis$x)], thresholdsF));
        }

        # If there are infinite values, mark them on the plot
        infiniteValues <- any(is.infinite(ellipsis$x));
        if(infiniteValues){
            infiniteMarkers <- ellipsis$x[is.infinite(ellipsis$x)];
            infiniteMarkersIDs <- which(is.infinite(ellipsis$x));
        }

        # Start plotting
        do.call(plot,ellipsis);
        for(i in 1:length(thresholdsF)){
            abline(h=thresholdsF[i], col=thresholdsColours[i], lty=thresholdsLty[i], lwd=thresholdsLwd[i]);
        }
        if(length(outliersID)>0){
            text(outliersID, ellipsis$x[outliersID], labels=outliersID, pos=2);
        }
        # Markers for infinite values
        if(infiniteValues){
            points(infiniteMarkersIDs, ellipsis$ylim[(infiniteMarkers>0)+1]+0.1, pch=24, bg=2);
            text(infiniteMarkersIDs, ellipsis$ylim[(infiniteMarkers>0)+1]+0.1,
                 labels=infiniteMarkersIDs, pos=1);
        }
        if(legend){
            legend("topright",
                   legend=paste0("F(",c(0.5,0.75,0.95),",",nparam(x),",",nobs(x)-nparam(x),")"),
                   col=thresholdsColours, lwd=thresholdsLwd, lty=thresholdsLty);
        }
    }

    # 13 and 14. Fitted vs (std. Residuals)^2 or Fitted vs |std. Residuals|
    plot9 <- function(x, type="abs", ...){
        ellipsis <- list(...);

        # Amend to do analysis of residuals of scale model
        if(is.scale(x$scale)){
            x <- x$scale;
        }

        ellipsis$x <- as.vector(fitted(x));
        ellipsis$y <- as.vector(rstandard(x));
        if(any(x$distribution==c("dinvgauss","dgamma","dexp"))){
            ellipsis$y[] <- log(ellipsis$y);
        }
        if(type=="abs"){
            ellipsis$y[] <- abs(ellipsis$y);
        }
        else{
            ellipsis$y[] <- ellipsis$y^2;
        }

        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[ellipsis$y!=0];
            ellipsis$y <- ellipsis$y[ellipsis$y!=0];
        }
        if(!any(names(ellipsis)=="main")){
            if(type=="abs"){
                ellipsis$main <- "|Standardised Residuals| vs Fitted";
            }
            else{
                ellipsis$main <- "Standardised Residuals^2 vs Fitted";
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="abs"){
                ellipsis$ylab <- "|Standardised Residuals|";
            }
            else{
                ellipsis$ylab <- "Standardised Residuals^2";
            }
        }

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(ellipsis$y[is.finite(ellipsis$y)]);
        }

        # If there are infinite values, mark them on the plot
        infiniteValues <- any(is.infinite(ellipsis$y));
        if(infiniteValues){
            infiniteMarkers <- ellipsis$y[is.infinite(ellipsis$y)];
            infiniteMarkersIDs <- which(is.infinite(ellipsis$y));
        }

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        if(lowess){
            lines(lowess(ellipsis$x[!is.na(ellipsis$y)], ellipsis$y[!is.na(ellipsis$y)]), col=2);
        }

        # Markers for infinite values
        if(infiniteValues){
            points(ellipsis$x[infiniteMarkersIDs], ellipsis$ylim[(infiniteMarkers>0)+1]+0.1, pch=24, bg=2);
            text(ellipsis$x[infiniteMarkersIDs], ellipsis$ylim[(infiniteMarkers>0)+1]+0.1,
                 labels=infiniteMarkersIDs, pos=1);
        }
    }

    for(i in which){
        if(any(i==1)){
            plot1(x, ...);
        }
        else if(any(i==2)){
            plot2(x, ...);
        }
        else if(any(i==3)){
            plot2(x, type="rstudent", ...);
        }
        else if(any(i==4)){
            plot3(x, ...);
        }
        else if(any(i==5)){
            plot3(x, type="squared", ...);
        }
        else if(any(i==6)){
            plot4(x, ...);
        }
        else if(any(i==7)){
            plot5(x, ...);
        }
        else if(any(i==8)){
            plot6(x, ...);
        }
        else if(any(i==9)){
            plot6(x, type="rstudent", ...);
        }
        else if(any(i==10)){
            plot7(x, type="acf", ...);
        }
        else if(any(i==11)){
            plot7(x, type="pacf", ...);
        }
        else if(any(i==12)){
            plot8(x, ...);
        }
        else if(any(i==13)){
            plot9(x, ...);
        }
        else if(any(i==14)){
            plot9(x, type="squared", ...);
        }
        else if(any(i==15)){
            plot7(x, type="acf", squared=TRUE, ...);
        }
        else if(any(i==16)){
            plot7(x, type="pacf", squared=TRUE, ...);
        }
    }

}

#' @export
plot.predict.greybox <- function(x, ...){
    yActuals <- actuals(x$model);
    yStart <- start(yActuals);
    yFrequency <- frequency(yActuals);
    yForecastStart <- time(yActuals)[length(yActuals)]+deltat(yActuals);

    if(!is.null(x$newdata)){
        yName <- all.vars(x$model$call$formula)[1];
        if(any(colnames(x$newdata)==yName)){
            yHoldout <- x$newdata[,yName];
            if(!any(is.na(yHoldout))){
                if(x$newdataProvided){
                    yActuals <- ts(c(yActuals,unlist(yHoldout)), start=yStart, frequency=yFrequency);
                }
                else{
                    yActuals <- ts(unlist(yHoldout), start=yForecastStart, frequency=yFrequency);
                }
                # If this is occurrence model, then transform actual to the occurrence
                if(any(x$distribution==c("pnorm","plogis"))){
                    yActuals <- (yActuals!=0)*1;
                }
            }
        }
    }

    # Change values of fitted and forecast, depending on whethere there was a newdata or not
    if(x$newdataProvided){
        yFitted <- ts(fitted(x$model), start=yStart, frequency=yFrequency);
        yForecast <- ts(x$mean, start=yForecastStart, frequency=yFrequency);
        vline <- TRUE;
    }
    else{
        yForecast <- ts(NA, start=yForecastStart, frequency=yFrequency);
        yFitted <- ts(x$mean, start=yStart, frequency=yFrequency);
        vline <- FALSE;
    }

    ellipsis <- list(...);
    ellipsis$actuals <- yActuals;
    ellipsis$forecast <- yForecast;
    ellipsis$fitted <- yFitted;
    ellipsis$vline <- vline;

    if(!is.null(x$lower) || !is.null(x$upper)){
        if(x$newdataProvided){
            yLower <- ts(x$lower, start=yForecastStart, frequency=yFrequency);
            yUpper <- ts(x$upper, start=yForecastStart, frequency=yFrequency);
        }
        else{
            yLower <- ts(x$lower, start=yStart, frequency=yFrequency);
            yUpper <- ts(x$upper, start=yStart, frequency=yFrequency);
        }

        if(is.matrix(x$level)){
            level <- x$level[1];
        }
        else{
            level <- x$level;
        }
        ellipsis$level <- level;
        ellipsis$lower <- yLower;
        ellipsis$upper <- yUpper;

        if((any(is.infinite(yLower)) & any(is.infinite(yUpper))) | (any(is.na(yLower)) & any(is.na(yUpper)))){
            ellipsis$lower[is.infinite(yLower) | is.na(yLower)] <- 0;
            ellipsis$upper[is.infinite(yUpper) | is.na(yUpper)] <- 0;
        }
        else if(any(is.infinite(yLower)) | any(is.na(yLower))){
            ellipsis$lower[is.infinite(yLower) | is.na(yLower)] <- 0;
        }
        else if(any(is.infinite(yUpper)) | any(is.na(yUpper))){
            ellipsis$upper <- NA;
        }
    }

    if(is.null(ellipsis$legend)){
        ellipsis$legend <- FALSE;
        ellipsis$parReset <- FALSE;
    }

    if(is.null(ellipsis$main)){
        if(x$newdataProvided){
            ellipsis$main <- paste0("Forecast for the variable ",colnames(x$model$data)[1]);
        }
        else{
            ellipsis$main <- paste0("Fitted values for the variable ",colnames(x$model$data)[1]);
        }
    }

    do.call(graphmaker,ellipsis);
}

#' @export
plot.coef.greyboxD <- function(x, ...){
    ellipsis <- list(...);
    # If type and ylab are not provided, set them...
    if(!any(names(ellipsis)=="type")){
        ellipsis$type <- "l";
    }
    if(!any(names(ellipsis)=="ylab")){
        ellipsis$ylab <- "Importance";
    }
    if(!any(names(ellipsis)=="ylim")){
        ellipsis$ylim <- c(0,1);
    }

    ourData <- x$importance;
    # We are not interested in intercept, so skip it in plot

    parDefault <- par(no.readonly=TRUE);
    on.exit(par(parDefault));

    pages <- ceiling((ncol(ourData)-1) / 8);
    perPage <- ceiling((ncol(ourData)-1) / pages);
    if(pages>1){
        parCols <- ceiling(perPage/4);
        perPage <- ceiling(perPage/parCols);
    }
    else{
        parCols <- 1;
    }

    parDims <- c(perPage,parCols);
    par(mfcol=parDims);

    if(pages>1){
        message(paste0("Too many variables. Ploting several per page, on ",pages," pages."));
    }

    for(i in 2:ncol(ourData)){
        ellipsis$x <- ourData[,i];
        ellipsis$main <- colnames(ourData)[i];
        do.call(plot,ellipsis);
    }
}

#' @importFrom grDevices rgb
#' @export
plot.rollingOrigin <- function(x, ...){
    y <- x$actuals;
    yDeltat <- deltat(y);

    # How many tables we have
    dimsOfHoldout <- dim(x$holdout);
    dimsOfThings <- lapply(x,dim);
    thingsToPlot <- 0;
    # 1 - actuals, 2 - holdout
    for(i in 3:length(dimsOfThings)){
        thingsToPlot <- thingsToPlot + all(dimsOfThings[[i]]==dimsOfHoldout)*1;
    }

    # Define basic parameters
    co <- !any(is.na(x$holdout[,ncol(x$holdout)]));
    h <- nrow(x$holdout);
    roh <- ncol(x$holdout);

    # Define the start of the RO
    roStart <- length(y)-h;
    roStart <- start(y)[1]+yDeltat*(roStart-roh+(h-1)*(!co));

    # Start plotting
    plot(y, ylab="Actuals", ylim=range(min(unlist(lapply(x,min,na.rm=T)),na.rm=T),
                                       max(unlist(lapply(x,max,na.rm=T)),na.rm=T),
                                       na.rm=TRUE),
         type="l", ...);
    abline(v=roStart, col=2, lwd=2);
    for(j in 1:thingsToPlot){
        colCurrent <- rgb((j-1)/thingsToPlot,0,(thingsToPlot-j+1)/thingsToPlot,1);
        for(i in 1:roh){
            points(roStart+i*yDeltat,x[[2+j]][1,i],col=colCurrent,pch=16);
            lines(c(roStart + (0:(h-1)+i)*yDeltat),c(x[[2+j]][,i]),col=colCurrent);
        }
    }
}

#### Print ####
#' @export
print.greybox <- function(x, ...){
    if(!is.null(x$timeElapsed)){
        cat("Time elapsed:",round(as.numeric(x$timeElapsed,units="secs"),2),"seconds\n");
    }
    cat("Call:\n");
    print(x$call);
    cat("\nCoefficients:\n");
    print(coef(x));
}

#' @export
print.occurrence <- function(x, ...){
    if(x$occurrence=="provided"){
        cat("The values for occcurrence part were provided by user\n");
    }
    else{
        print.greybox(x);
    }
}

#' @export
print.scale <- function(x, ...){
    cat("Formula:\n");
    print(formula(x));
    cat("\nCoefficients:\n");
    print(coef(x));
}

#' @export
print.coef.greyboxD <- function(x, ...){
    print(x$coefficients);
}

#' @export
print.association <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    cat("Associations: ")
    cat("\nvalues:\n"); print(round(x$value,digits));
    cat("\np-values:\n"); print(round(x$p.value,digits));
    cat("\ntypes:\n"); print(x$type);
    cat("\n");
}

#' @export
print.pcor <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    cat(paste0("Partial correlations using ",x$method," method: "))
    cat("\nvalues:\n"); print(round(x$value,digits));
    cat("\np-values:\n"); print(round(x$p.value,digits));
    cat("\n");
}

#' @export
print.cramer <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    cat("Cramer's V: "); cat(round(x$value,digits));
    cat("\nChi^2 statistics = "); cat(round(x$statistic,digits));
    cat(", df: "); cat(x$df);
    cat(", p-value: "); cat(round(x$p.value,digits));
    cat("\n");
}

#' @export
print.mcor <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    cat("Multiple correlations value: "); cat(round(x$value,digits));
    cat("\nF-statistics = "); cat(round(x$statistic,digits));
    cat(", df: "); cat(x$df);
    cat(", df resid: "); cat(x$df.residual);
    cat(", p-value: "); cat(round(x$p.value,digits));
    cat("\n");
}

#' @importFrom stats setNames
#' @export
print.summary.alm <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dgnorm" = paste0("Generalised Normal Distribution with shape=",round(x$other$shape,digits)),
                      "dlgnorm" = paste0("Log-Generalised Normal Distribution with shape=",round(x$other$shape,digits)),
                      "dlogis" = "Logistic",
                      "dlaplace" = "Laplace",
                      "dllaplace" = "Log-Laplace",
                      "dalaplace" = paste0("Asymmetric Laplace with alpha=",round(x$other$alpha,digits)),
                      "dt" = paste0("Student t with nu=",round(x$other$nu, digits)),
                      "ds" = "S",
                      "dls" = "Log-S",
                      "dfnorm" = "Folded Normal",
                      "drectnorm" = "Rectified Normal",
                      "dlnorm" = "Log-Normal",
                      "dbcnorm" = paste0("Box-Cox Normal with lambda=",round(x$other$lambdaBC,digits)),
                      "dlogitnorm" = "Logit Normal",
                      "dinvgauss" = "Inverse Gaussian",
                      "dgamma" = "Gamma",
                      "dexp" = "Exponential",
                      "dchisq" = paste0("Chi-Squared with nu=",round(x$other$nu,digits)),
                      "dgeom" = "Geometric",
                      "dpois" = "Poisson",
                      "dnbinom" = paste0("Negative Binomial with size=",round(x$other$size,digits)),
                      "dbinom" = paste0("Binomial with size=",x$other$size),
                      "dbeta" = "Beta",
                      "plogis" = "Cumulative logistic",
                      "pnorm" = "Cumulative normal"
    );
    if(is.occurrence(x$occurrence)){
        distribOccurrence <- switch(x$occurrence$distribution,
                                    "plogis" = "Cumulative logistic",
                                    "pnorm" = "Cumulative normal",
                                    "Provided values"
        );
        distrib <- paste0("Mixture of ", distrib," and ", distribOccurrence);
    }

    cat(paste0("Response variable: ", paste0(x$responseName,collapse="")));
    cat(paste0("\nDistribution used in the estimation: ", distrib));
    cat(paste0("\nLoss function used in estimation: ",x$loss));
    if(any(x$loss==c("LASSO","RIDGE"))){
        cat(paste0(" with lambda=",round(x$other$lambda,digits)));
    }
    if(!is.null(x$arima)){
        cat(paste0("\n",x$arima," components were included in the model"));
    }
    if(x$bootstrap){
        cat("\nBootstrap was used for the estimation of uncertainty of parameters");
    }

    cat("\nCoefficients:\n");
    stars <- setNames(vector("character",length(x$significance)),
                      names(x$significance));
    stars[x$significance] <- "*";
    print(data.frame(round(x$coefficients[!x$scaleParameters,,drop=FALSE],digits),stars[!x$scaleParameters],
                     check.names=FALSE,fix.empty.names=FALSE));

    if(any(x$scaleParameters)){
        cat("\nCoefficients for scale:\n");
        print(data.frame(round(x$coefficients[x$scaleParameters,,drop=FALSE],digits),stars[x$scaleParameters],
                         check.names=FALSE,fix.empty.names=FALSE));
    }

    cat("\nError standard deviation: "); cat(round(sqrt(x$s2),digits));
    cat("\nSample size: "); cat(x$dfTable[1]);
    cat("\nNumber of estimated parameters: "); cat(x$dfTable[2]);
    cat("\nNumber of degrees of freedom: "); cat(x$dfTable[3]);
    if(!is.null(x$ICs)){
        cat("\nInformation criteria:\n");
        print(round(x$ICs,digits));
    }
    cat("\n");
}

#' @export
print.summary.scale <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dgnorm" = paste0("Generalised Normal Distribution with shape=",round(x$other$shape,digits)),
                      "dlgnorm" = paste0("Log-Generalised Normal Distribution with shape=",round(x$other$shape,digits)),
                      "dlogis" = "Logistic",
                      "dlaplace" = "Laplace",
                      "dllaplace" = "Log-Laplace",
                      "dalaplace" = paste0("Asymmetric Laplace with alpha=",round(x$other$alpha,digits)),
                      "dt" = paste0("Student t with nu=",round(x$other$nu, digits)),
                      "ds" = "S",
                      "dls" = "Log-S",
                      "dfnorm" = "Folded Normal",
                      "drectnorm" = "Rectified Normal",
                      "dlnorm" = "Log-Normal",
                      "dbcnorm" = paste0("Box-Cox Normal with lambda=",round(x$other$lambdaBC,digits)),
                      "dlogitnorm" = "Logit Normal",
                      "dinvgauss" = "Inverse Gaussian",
                      "dgamma" = "Gamma",
                      "dexp" = "Exponential",
                      "dchisq" = paste0("Chi-Squared with nu=",round(x$other$nu,digits)),
                      "dgeom" = "Geometric",
                      "dpois" = "Poisson",
                      "dnbinom" = paste0("Negative Binomial with size=",round(x$other$size,digits)),
                      "dbinom" = paste0("Binomial with size=",x$other$size),
                      "dbeta" = "Beta",
                      "plogis" = "Cumulative logistic",
                      "pnorm" = "Cumulative normal"
    );

    cat(paste0("Scale model for the variable: ", paste0(x$responseName,collapse="")));
    cat(paste0("\nDistribution used in the estimation: ", distrib));
    if(x$bootstrap){
        cat("\nBootstrap was used for the estimation of uncertainty of parameters");
    }

    cat("\nCoefficients:\n");
    stars <- setNames(vector("character",length(x$significance)),
                      names(x$significance));
    stars[x$significance] <- "*";
    print(data.frame(round(x$coefficients,digits),stars,
                     check.names=FALSE,fix.empty.names=FALSE));

    cat("\nSample size: "); cat(x$dfTable[1]);
    cat("\nNumber of estimated parameters: "); cat(x$dfTable[2]);
    cat("\nNumber of degrees of freedom: "); cat(x$dfTable[3]);
    if(!is.null(x$ICs)){
        cat("\nInformation criteria:\n");
        print(round(x$ICs,digits));
    }
}

#' @export
print.summary.greybox <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dgnorm" = "Generalised Normal Distribution",
                      "dlgnorm" = "Log-Generalised Normal Distribution",
                      "dlogis" = "Logistic",
                      "dlaplace" = "Laplace",
                      "dllaplace" = "Log-Laplace",
                      "dalaplace" = "Asymmetric Laplace",
                      "dt" = "Student t",
                      "ds" = "S",
                      "ds" = "Log-S",
                      "dfnorm" = "Folded Normal",
                      "drectnorm" = "Rectified Normal",
                      "dlnorm" = "Log-Normal",
                      "dbcnorm" = "Box-Cox Normal",
                      "dlogitnorm" = "Logit Normal",
                      "dinvgauss" = "Inverse Gaussian",
                      "dgamma" = "Gamma",
                      "dexp" = "Exponential",
                      "dchisq" = "Chi-Squared",
                      "dgeom" = "Geometric",
                      "dpois" = "Poisson",
                      "dnbinom" = "Negative Binomial",
                      "dnbinom" = "Binomial",
                      "dbeta" = "Beta",
                      "plogis" = "Cumulative logistic",
                      "pnorm" = "Cumulative normal"
    );

    cat(paste0("Response variable: ", paste0(x$responseName,collapse=""),"\n"));
    cat(paste0("Distribution used in the estimation: ", distrib));
    if(!is.null(x$arima)){
        cat(paste0("\n",x$arima," components were included in the model"));
    }
    cat("\nCoefficients:\n");
    stars <- setNames(vector("character",length(x$significance)),
                      names(x$significance));
    stars[x$significance] <- "*";
    print(data.frame(round(x$coefficients,digits),stars,
                     check.names=FALSE,fix.empty.names=FALSE));
    cat("\nError standard deviation: "); cat(round(x$sigma,digits));
    cat("\nSample size: "); cat(x$dfTable[1]);
    cat("\nNumber of estimated parameters: "); cat(x$dfTable[2]);
    cat("\nNumber of degrees of freedom: "); cat(x$dfTable[3]);
    cat("Information criteria:\n");
    print(round(x$ICs,digits));
}

#' @export
print.summary.calm <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dgnorm" = "Generalised Normal Distribution",
                      "dlgnorm" = "Log-Generalised Normal Distribution",
                      "dlogis" = "Logistic",
                      "dlaplace" = "Laplace",
                      "dllaplace" = "Log-Laplace",
                      "dalaplace" = "Asymmetric Laplace",
                      "dt" = "Student t",
                      "ds" = "S",
                      "dls" = "Log-S",
                      "dfnorm" = "Folded Normal",
                      "drectnorm" = "Rectified Normal",
                      "dlnorm" = "Log-Normal",
                      "dbcnorm" = "Box-Cox Normal",
                      "dlogitnorm" = "Logit Normal",
                      "dinvgauss" = "Inverse Gaussian",
                      "dgamma" = "Gamma",
                      "dexp" = "Exponential",
                      "dchisq" = "Chi-Squared",
                      "dgeom" = "Geometric",
                      "dpois" = "Poisson",
                      "dnbinom" = "Negative Binomial",
                      "dnbinom" = "Binomial",
                      "dbeta" = "Beta",
                      "plogis" = "Cumulative logistic",
                      "pnorm" = "Cumulative normal"
    );

    # The name of the model used
    if(is.null(x$dynamic)){
        cat(paste0("The ",x$ICType," combined model\n"));
    }
    else{
        cat(paste0("The p",x$ICType," combined model\n"));
    }
    cat(paste0("Response variable: ", paste0(x$responseName,collapse=""),"\n"));
    cat(paste0("Distribution used in the estimation: ", distrib));
    cat("\nCoefficients:\n");
    stars <- setNames(vector("character",length(x$significance)),
                      names(x$significance));
    stars[x$significance] <- "*";
    print(data.frame(round(x$coefficients,digits),stars,
                     check.names=FALSE,fix.empty.names=FALSE));
    cat("\nError standard deviation: "); cat(round(x$sigma,digits));
    cat("\nSample size: "); cat(round(x$dfTable[1],digits));
    cat("\nNumber of estimated parameters: "); cat(round(x$dfTable[2],digits));
    cat("\nNumber of degrees of freedom: "); cat(round(x$dfTable[3],digits));
    cat("\nApproximate combined information criteria:\n");
    print(round(x$ICs,digits));
}

#' @export
print.predict.greybox <- function(x, ...){
    ourMatrix <- as.matrix(x$mean);
    colnames(ourMatrix) <- "Mean";
    if(!is.null(x$lower)){
        ourMatrix <- cbind(ourMatrix, x$lower, x$upper);
        if(is.matrix(x$level)){
            level <- colMeans(x$level)[-1];
        }
        else{
            level <- x$level;
        }
    }
    print(ourMatrix);
}

#' @export
print.rollingOrigin <- function(x, ...){
    co <- !any(is.na(x$holdout[,ncol(x$holdout)]));
    h <- nrow(x$holdout);
    roh <- ncol(x$holdout);

    if(co){
        cat(paste0("Rolling Origin with constant holdout was done.\n"));
    }
    else{
        cat(paste0("Rolling Origin with decreasing holdout was done.\n"));
    }
    cat(paste0("Forecast horizon is ",h,"\n"));
    cat(paste0("Number of origins is ",roh,"\n"));
}

#### Regression diagnostics ####

#' @importFrom stats hatvalues hat
#' @export
hatvalues.greybox <- function(model, ...){
    # Prepare the hat values
    if(any(names(coef(model))=="(Intercept)")){
        xreg <- model$data;
        xreg[,1] <- 1;
    }
    else{
        xreg <- model$data[,-1,drop=FALSE];
    }
    # Hatvalues for different distributions
    if(any(model$distribution==c("dt","dnorm","dlnorm","dbcnorm","dlogitnorm","dnbinom","dbinom","dpois"))){
        hatValue <- hat(xreg);
    }
    else{
        vcovValues <- vcov(model);
        # Remove the scale parameters covariance
        if(is.scale(model$scale)){
            vcovValues <- vcovValues[1:length(coef(model)),1:length(coef(model)),drop=FALSE];
        }
        hatValue <- diag(xreg %*% vcovValues %*% t(xreg))/extractSigma(model)^2;
    }
    names(hatValue) <- names(actuals(model));

    # Substitute extreme values by something very big.
    if(any(hatValue==1)){
        hatValue[hatValue==1] <- 1 -1E5;
    }
    return(hatValue);
}

#' @export
residuals.greybox <- function(object, ...){
    errors <- object$residuals;
    names(errors) <- names(actuals(object));
    ellipsis <- list(...);
    # Remove zeroes if they are not needed
    if(!is.null(ellipsis$all) && (!ellipsis$all)){
        if(is.occurrence(object$occurrence)){
            errors <- errors[actuals(object)!=0];
        }
    }
    return(errors)
}

#' @importFrom stats rstandard
#' @export
rstandard.greybox <- function(model, ...){
    obs <- nobs(model);
    df <- obs - nparam(model);
    errors <- residuals(model, ...);
    # If this is an occurrence model, then only modify the non-zero obs
    if(is.occurrence(model$occurrence)){
        residsToGo <- (actuals(model$occurrence)!=0);
    }
    else{
        residsToGo <- rep(TRUE,obs);
    }

    # If it is scale model, there's no need to divide by scale anymore
    if(!is.scale(model)){
        # The proper residuals with leverage are currently done only for normal-based distributions
        if(any(model$distribution==c("dnorm","dlnorm"))){
            errors[] <- errors / (sqrt(extractScale(model))*sqrt(1-hatvalues(model)));
        }
        else if(any(model$distribution==c("dt","dbcnorm","dlogitnorm"))){
            errors[] <- errors / (extractScale(model)*sqrt(1-hatvalues(model)));
        }
        else if(any(model$distribution==c("ds","dls"))){
            errors[residsToGo] <- (errors[residsToGo] - mean(errors[residsToGo])) / (extractScale(model) * obs / df)^2;
        }
        else if(any(model$distribution==c("dgnorm","dlgnorm"))){
            errors[residsToGo] <- ((errors[residsToGo] - mean(errors[residsToGo])) /
                                       (extractScale(model)^model$other$shape * obs / df)^{1/model$other$shape});
        }
        else if(any(model$distribution==c("dinvgauss","dgamma","dexp"))){
            errors[residsToGo] <- errors[residsToGo] / mean(errors[residsToGo]);
        }
        else if(any(model$distribution==c("dfnorm","drectnorm"))){
            errors[residsToGo] <- (errors[residsToGo]) / sqrt(extractScale(model)^2 * obs / df);
        }
        else if(any(model$distribution==c("dpois","dnbinom","dbinom","dgeom"))){
            errors[residsToGo] <- qnorm(pointLikCumulative(model));
        }
        else{
            errors[residsToGo] <- (errors[residsToGo] - mean(errors[residsToGo])) / (extractScale(model) * obs / df);
        }
    }

    # Fill in values with NAs if there is occurrence model
    if(is.occurrence(model$occurrence)){
        errors[!residsToGo] <- NA;
    }

    return(errors);
}

#' @importFrom stats rstudent
#' @export
rstudent.greybox <- function(model, ...){
    obs <- nobs(model);
    df <- obs - nparam(model) - 1;
    rstudentised <- errors <- residuals(model, ...);
    # If this is an occurrence model, then only modify the non-zero obs
    if(is.occurrence(model$occurrence)){
        residsToGo <- (actuals(model$occurrence)!=0);
    }
    else{
        residsToGo <- rep(TRUE,obs);
    }

    # The proper residuals with leverage are currently done only for normal-based distributions
    if(any(model$distribution==c("dt","dnorm","dlnorm","dlogitnorm","dbcnorm"))){
        # Prepare the hat values
        if(any(names(coef(model))=="(Intercept)")){
            xreg <- model$data;
            xreg[,1] <- 1;
        }
        else{
            xreg <- model$data[,-1,drop=FALSE];
        }
        hatValues <- hat(xreg);
        errors[] <- errors - mean(errors);
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / sqrt(sum(errors[-i]^2) / df * (1-hatValues[i]));
        }
    }
    else if(any(model$distribution==c("ds","dls"))){
        # This is an approximation from the vcov matrix
        # hatValues <- diag(xreg %*% vcov(model) %*% t(xreg))/sigma(model)^2;
        errors[] <- errors - mean(errors);
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] /  (sum(sqrt(abs(errors[-i]))) / (2*df))^2;
        }
    }
    else if(any(model$distribution==c("dlaplace","dllaplace"))){
        errors[] <- errors - mean(errors);
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / (sum(abs(errors[-i])) / df);
        }
    }
    else if(model$distribution=="dalaplace"){
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / (sum(errors[-i] * (model$other$alpha - (errors[-i]<=0)*1)) / df);
        }
    }
    else if(any(model$distribution==c("dgnorm","dlgnorm"))){
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] /  (sum(abs(errors[-i])^model$other$shape) * (model$other$shape/df))^{1/model$other$shape};
        }
    }
    else if(model$distribution=="dlogis"){
        errors[] <- errors - mean(errors);
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / (sqrt(sum(errors[-i]^2) / df) * sqrt(3) / pi);
        }
    }
    else if(any(model$distribution==c("dinvgauss","dgamma","dexp"))){
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / mean(errors[-i]);
        }
    }
    # We don't do studentised residuals for count distributions
    else if(any(model$distribution==c("dpois","dnbinom","dbinom","dgeom"))){
        rstudentised[residsToGo] <- qnorm(pointLikCumulative(model));
    }
    else{
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / sqrt(sum(errors[-i]^2) / df);
        }
    }

    # Fill in values with NAs if there is occurrence model
    if(is.occurrence(model$occurrence)){
        rstudentised[!residsToGo] <- NA;
    }

    return(rstudentised);
}

#' @importFrom stats cooks.distance
#' @export
cooks.distance.greybox <- function(model, ...){
    # Number of parameters
    nParam <- nparam(model);
    # Hat values
    hatValues <- hatvalues(model);
    # Standardised residuals
    errors <- rstandard(model);

    return(errors^2 / nParam * hatValues/(1-hatValues));
}

#### Summary ####
#' @importFrom stats summary.lm
#' @export
summary.greybox <- function(object, level=0.95, ...){
    ourReturn <- summary.lm(object, ...);
    errors <- residuals(object);
    obs <- nobs(object, all=TRUE);

    # Collect parameters and their standard errors
    parametersTable <- ourReturn$coefficients[,1:2];
    parametersTable <- cbind(parametersTable,confint(object, level=level));
    rownames(parametersTable) <- names(coef(object));
    colnames(parametersTable) <- c("Estimate","Std. Error",
                                   paste0("Lower ",(1-level)/2*100,"%"),
                                   paste0("Upper ",(1+level)/2*100,"%"));
    ourReturn$coefficients <- parametersTable;
    # Mark those that are significant on the selected level
    ourReturn$significance <- !(parametersTable[,3]<=0 & parametersTable[,4]>=0);

    ICs <- c(AIC(object),AICc(object),BIC(object),BICc(object));
    names(ICs) <- c("AIC","AICc","BIC","BICc");
    ourReturn$ICs <- ICs;
    ourReturn$distribution <- object$distribution;
    ourReturn$responseName <- formula(object)[[2]];

    # Table with degrees of freedom
    dfTable <- c(obs,nparam(object),obs-nparam(object));
    names(dfTable) <- c("n","k","df");

    ourReturn$r.squared <- 1 - sum(errors^2) / sum((actuals(object)-mean(actuals(object)))^2);
    ourReturn$adj.r.squared <- 1 - (1 - ourReturn$r.squared) * (obs - 1) / (dfTable[3]);

    ourReturn$dfTable <- dfTable;
    ourReturn$arima <- object$other$arima;

    ourReturn$bootstrap <- FALSE;

    ourReturn <- structure(ourReturn,class="summary.greybox");
    return(ourReturn);
}

#' @aliases summary.alm
#' @rdname coef.alm
#' @export
summary.alm <- function(object, level=0.95, bootstrap=FALSE, ...){
    errors <- residuals(object);
    obs <- nobs(object, all=TRUE);
    scaleModel <- is.scale(object$scale);

    # Collect parameters and their standard errors
    parametersConfint <- confint(object, level=level, bootstrap=bootstrap, ...);
    parameters <- coef(object);
    scaleParameters <- rep(FALSE,length(parameters));
    if(scaleModel){
        parameters <- c(parameters, coef(object$scale));
        scaleParameters <- c(scaleParameters,rep(TRUE,length(coef(object$scale))));
    }
    parametersTable <- cbind(parameters,parametersConfint);
    rownames(parametersTable) <- names(parameters);
    colnames(parametersTable) <- c("Estimate","Std. Error",
                                   paste0("Lower ",(1-level)/2*100,"%"),
                                   paste0("Upper ",(1+level)/2*100,"%"));
    ourReturn <- list(coefficients=parametersTable);
    # Mark those that are significant on the selected level
    ourReturn$significance <- !(parametersTable[,3]<=0 & parametersTable[,4]>=0);

    # If there is a likelihood, then produce ICs
    if(!is.na(logLik(object))){
        ICs <- c(AIC(object),AICc(object),BIC(object),BICc(object));
        names(ICs) <- c("AIC","AICc","BIC","BICc");
        ourReturn$ICs <- ICs;
    }
    ourReturn$distribution <- object$distribution;
    ourReturn$loss <- object$loss;
    ourReturn$occurrence <- object$occurrence;
    ourReturn$other <- object$other;
    ourReturn$responseName <- formula(object)[[2]];

    # Table with degrees of freedom
    dfTable <- c(obs,nparam(object),obs-nparam(object));
    names(dfTable) <- c("n","k","df");

    ourReturn$r.squared <- 1 - sum(errors^2) / sum((actuals(object)-mean(actuals(object)))^2);
    ourReturn$adj.r.squared <- 1 - (1 - ourReturn$r.squared) * (obs - 1) / (dfTable[3]);

    ourReturn$dfTable <- dfTable;
    ourReturn$arima <- object$other$arima;
    ourReturn$s2 <- sigma(object)^2;

    ourReturn$bootstrap <- bootstrap;
    ourReturn$scaleParameters <- scaleParameters;

    ourReturn <- structure(ourReturn,class=c("summary.alm","summary.greybox"));
    return(ourReturn);
}

#' @export
summary.scale <- function(object, level=0.95, bootstrap=FALSE, ...){
    obs <- nobs(object, all=TRUE);

    # Collect parameters and their standard errors
    parametersConfint <- confint(object, level=level, bootstrap=bootstrap, ...);
    parameters <- coef(object);
    parametersTable <- cbind(parameters,parametersConfint);
    rownames(parametersTable) <- names(parameters);
    colnames(parametersTable) <- c("Estimate","Std. Error",
                                   paste0("Lower ",(1-level)/2*100,"%"),
                                   paste0("Upper ",(1+level)/2*100,"%"));
    ourReturn <- list(coefficients=parametersTable);
    # Mark those that are significant on the selected level
    ourReturn$significance <- !(parametersTable[,3]<=0 & parametersTable[,4]>=0);

    # If there is a likelihood, then produce ICs
    if(!is.na(logLik(object))){
        ICs <- c(AIC(object),AICc(object),BIC(object),BICc(object));
        names(ICs) <- c("AIC","AICc","BIC","BICc");
        ourReturn$ICs <- ICs;
    }
    ourReturn$distribution <- object$distribution;
    ourReturn$loss <- object$loss;
    ourReturn$other <- object$other;
    ourReturn$responseName <- formula(object)[[2]];

    # Table with degrees of freedom
    dfTable <- c(obs,nparam(object),obs-nparam(object));
    names(dfTable) <- c("n","k","df");

    ourReturn$dfTable <- dfTable;

    ourReturn$bootstrap <- bootstrap;

    ourReturn <- structure(ourReturn,class="summary.scale");
    return(ourReturn);
}

#' @export
summary.calm <- function(object, level=0.95, ...){

    # Extract the values from the object
    errors <- residuals(object);
    obs <- nobs(object);
    parametersTable <- cbind(coef(object),sqrt(abs(diag(vcov(object)))),object$importance);

    # Calculate the quantiles for parameters and add them to the table
    parametersTable <- cbind(parametersTable,confint(object, level=level));
    rownames(parametersTable) <- names(coef(object));
    colnames(parametersTable) <- c("Estimate","Std. Error","Importance",
                                   paste0("Lower ",(1-level)/2*100,"%"),
                                   paste0("Upper ",(1+level)/2*100,"%"));
    # Mark those that are significant on the selected level
    significance <- !(parametersTable[,4]<=0 & parametersTable[,5]>=0);

    # Extract degrees of freedom
    df <- c(object$df, object$df.residual, object$rank);
    # Calculate s.e. of residuals
    residSE <- sqrt(sum(errors^2)/df[2]);

    ICs <- c(AIC(object),AICc(object),BIC(object),BICc(object));
    names(ICs) <- c("AIC","AICc","BIC","BICc");

    # Table with degrees of freedom
    dfTable <- c(obs, nparam(object), object$df.residual);
    names(dfTable) <- c("n","k","df");

    R2 <- 1 - sum(errors^2) / sum((actuals(object)-mean(actuals(object)))^2);
    R2Adj <- 1 - (1 - R2) * (obs - 1) / (dfTable[3]);

    ourReturn <- structure(list(coefficients=parametersTable, significance=significance, sigma=residSE,
                                ICs=ICs, ICType=object$ICType, df=df, r.squared=R2, adj.r.squared=R2Adj,
                                distribution=object$distribution, responseName=formula(object)[[2]],
                                dfTable=dfTable, bootstrap=FALSE),
                           class=c("summary.calm","summary.greybox"));
    return(ourReturn);
}

#' @export
summary.greyboxD <- function(object, level=0.95, ...){

    # Extract the values from the object
    errors <- residuals(object);
    obs <- nobs(object);
    parametersTable <- cbind(coef.greybox(object),sqrt(abs(diag(vcov(object)))),apply(object$importance,2,mean));

    parametersConfint <- confint(object, level=level);
    # Calculate the quantiles for parameters and add them to the table
    parametersTable <- cbind(parametersTable,parametersConfint);

    rownames(parametersTable) <- names(coef.greybox(object));
    colnames(parametersTable) <- c("Estimate","Std. Error","Importance",
                                   paste0("Lower ",(1-level)/2*100,"%"),
                                   paste0("Upper ",(1+level)/2*100,"%"));
    # Mark those that are significant on the selected level
    significance <- !(parametersTable[,4]<=0 & parametersTable[,5]>=0);

    # Extract degrees of freedom
    df <- c(object$df, object$df.residual, object$rank);
    # Calculate s.e. of residuals
    residSE <- sqrt(sum(errors^2)/df[2]);

    ICs <- c(AIC(object),AICc(object),BIC(object),BICc(object));
    names(ICs) <- c("AIC","AICc","BIC","BICc");

    R2 <- 1 - sum(errors^2) / sum((actuals(object)-mean(actuals(object)))^2)
    R2Adj <- 1 - (1 - R2) * (obs - 1) / (obs - df[1]);

    # Table with degrees of freedom
    dfTable <- c(nobs(object), nparam(object), object$df.residual);
    names(dfTable) <- c("n","k","df");

    ourReturn <- structure(list(coefficients=parametersTable, significance=significance, sigma=residSE,
                                dynamic=coef(object)$dynamic,
                                ICs=ICs, ICType=object$ICType, df=df, r.squared=R2, adj.r.squared=R2Adj,
                                distribution=object$distribution, responseName=formula(object)[[2]],
                                nobs=nobs(object), nparam=nparam(object), dfTable=dfTable, bootstrap=FALSE),
                           class=c("summary.calm","summary.greybox"));
    return(ourReturn);
}

#' @export
summary.lmGreybox <- function(object, level=0.95, ...){
    parametersTable <- cbind(coef(object),sqrt(diag(vcov(object))));
    parametersTable <- cbind(parametersTable, parametersTable[,1]+qnorm((1-level)/2,0,parametersTable[,2]),
                             parametersTable[,1]+qnorm((1+level)/2,0,parametersTable[,2]));
    colnames(parametersTable) <- c("Estimate","Std. Error",
                                   paste0("Lower ",(1-level)/2*100,"%"),
                                   paste0("Upper ",(1+level)/2*100,"%"));
    return(parametersTable)
}


#' @export
as.data.frame.summary.greybox <- function(x, ...){
    stars <- setNames(vector("character",length(x$significance)),
                      names(x$significance));
    stars[x$significance] <- "*";
    return(data.frame(x$coefficients,stars,
                      row.names=rownames(x$coefficients),
                      check.names=FALSE,fix.empty.names=FALSE));
}

#' @importFrom texreg extract createTexreg
extract.greybox <- function(model, ...){
    summaryALM <- summary(model, ...);
    tr <- extract(summaryALM);
    return(tr)
}

extract.summary.greybox <- function(model, ...){
    gof <- c(model$dfTable, model$ICs)
    gof.names <- c("Num.\\ obs.", "Num.\\ param.", "Num.\\ df", names(model$ICs))

    tr <- createTexreg(
        coef.names=rownames(model$coefficients),
        coef=model$coef[, 1],
        se=model$coef[, 2],
        ci.low=model$coef[, 3],
        ci.up=model$coef[, 4],
        gof.names=gof.names,
        gof=gof
    )
    return(tr)
}

#' @importFrom methods setMethod
setMethod("extract", signature=className("alm","greybox"), definition=extract.greybox)
setMethod("extract", signature=className("greybox","greybox"), definition=extract.greybox)
setMethod("extract", signature=className("calm","greybox"), definition=extract.greybox)
setMethod("extract", signature=className("greyboxD","greybox"), definition=extract.greybox)
setMethod("extract", signature=className("summary.alm","greybox"), definition=extract.summary.greybox)
setMethod("extract", signature=className("summary.greybox","greybox"), definition=extract.summary.greybox)
setMethod("extract", signature=className("summary.calm","greybox"), definition=extract.summary.greybox)

#' @importFrom xtable xtable
#' @export
xtable::xtable

#' @export
xtable.greybox <- function(x, caption = NULL, label = NULL, align = NULL, digits = NULL,
                           display = NULL, auto = FALSE, ...){
    greyboxSummary <- summary(x);
    return(do.call("xtable", list(x=greyboxSummary,
                                  caption=caption, label=label, align=align, digits=digits,
                                  display=display, auto=auto, ...)));
}

#' @export
xtable.summary.greybox <- function(x, caption = NULL, label = NULL, align = NULL, digits = NULL,
                           display = NULL, auto = FALSE, ...){
    # Substitute class with lm
    class(x) <- "summary.lm";
    return(do.call("xtable", list(x=x,
                                  caption=caption, label=label, align=align, digits=digits,
                                  display=display, auto=auto, ...)));
}

#### Predictions and forecasts ####

#' @param occurrence If occurrence was provided, then a user can provide a vector of future
#' values via this variable.
#' @rdname predict.greybox
#' @importFrom stats predict qchisq qlnorm qlogis qpois qnbinom qbeta qgamma qexp qgeom qbinom
#' @importFrom statmod qinvgauss
#' @export
predict.alm <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                            level=0.95, side=c("both","upper","lower"),
                        occurrence=NULL, ...){
    if(is.null(newdata)){
        newdataProvided <- FALSE;
    }
    else{
        newdataProvided <- TRUE;
    }
    interval <- match.arg(interval);
    side <- match.arg(side);
    if(!is.null(newdata)){
        h <- nrow(newdata);
    }
    else{
        h <- nobs(object);
    }
    levelOriginal <- level;
    nLevels <- length(level);

    ariOrderNone <- is.null(object$other$polynomial);
    if(ariOrderNone){
        greyboxForecast <- predict.greybox(object, newdata, interval, level, side=side, ...);
    }
    else{
        greyboxForecast <- predict_almari(object, newdata, interval, level, side=side, ...);
    }
    greyboxForecast$location <- greyboxForecast$mean;
    if(interval!="none"){
        greyboxForecast$scale <- sqrt(greyboxForecast$variances);
    }
    greyboxForecast$distribution <- object$distribution;

    # If there is an occurrence part of the model, use it
    if(is.null(occurrence) && is.occurrence(object$occurrence) &&
       is.list(object$occurrence) &&
       (is.null(object$occurrence$occurrence) || object$occurrence$occurrence!="provided")){
        occurrencePredict <- predict(object$occurrence, newdata, interval=interval, level=level, side=side, ...);
        # Reset horizon, just in case
        h <- length(occurrencePredict$mean);
        # Create a matrix of levels for each horizon and level
        level <- matrix(level, h, nLevels, byrow=TRUE);
        # The probability of having zero should be subtracted from that thing...
        if(interval=="prediction"){
            level[] <- (level - (1 - occurrencePredict$mean)) / occurrencePredict$mean;
        }
        level[level<0] <- 0;
        greyboxForecast$occurrence <- occurrencePredict;
    }
    else{
        # Create a matrix of levels for each horizon and level
        level <- matrix(level, h, nLevels, byrow=TRUE);
        if(is.null(occurrence) &&
           is.list(object$occurrence) &&
           !is.null(object$occurrence$occurrence) &&
           object$occurrence$occurrence=="provided"){
            warning("occurrence is not provided for the new data. Using a vector of ones.", call.=FALSE);
            occurrence <- rep(1, h);
        }
    }

    scaleModel <- is.scale(object$scale);

    # levelLow and levelUp are matrices here...
    levelLow <- levelUp <- matrix(level, h, nLevels, byrow=TRUE);
    if(side=="upper"){
        levelLow[] <- 0;
        levelUp[] <- level;
    }
    else if(side=="lower"){
        levelLow[] <- 1-level;
        levelUp[] <- 1;
    }
    else{
        levelLow[] <- (1-level)/2;
        levelUp[] <- (1+level)/2;
    }

    levelLow[levelLow<0] <- 0;
    levelUp[levelUp<0] <- 0;

    if(object$distribution=="dnorm"){
        if(is.null(occurrence) && is.occurrence(object$occurrence) && interval!="none"){
            greyboxForecast$lower[] <- qnorm(levelLow,greyboxForecast$mean,greyboxForecast$scale);
            greyboxForecast$upper[] <- qnorm(levelUp,greyboxForecast$mean,greyboxForecast$scale);
        }
    }
    else if(object$distribution=="dlaplace"){
        # Use the connection between the variance and MAE in Laplace distribution
        if(interval!="none"){
            scaleValues <- sqrt(greyboxForecast$variances/2);
            greyboxForecast$scale <- scaleValues;
        }
        if(interval=="prediction"){
            for(i in 1:nLevels){
                greyboxForecast$lower[,i] <- qlaplace(levelLow[,i],greyboxForecast$mean,scaleValues);
                greyboxForecast$upper[,i] <- qlaplace(levelUp[,i],greyboxForecast$mean,scaleValues);
            }
        }
    }
    else if(object$distribution=="dllaplace"){
        # Use the connection between the variance and MAE in Laplace distribution
        if(interval!="none"){
            scaleValues <- sqrt(greyboxForecast$variances/2);
            greyboxForecast$scale <- scaleValues;
        }
        if(interval=="prediction"){
            for(i in 1:nLevels){
                greyboxForecast$lower[,i] <- exp(qlaplace(levelLow[,i],greyboxForecast$mean,scaleValues));
                greyboxForecast$upper[,i] <- exp(qlaplace(levelUp[,i],greyboxForecast$mean,scaleValues));
            }
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
        greyboxForecast$mean[] <- exp(greyboxForecast$mean);
        greyboxForecast$scale <- scaleValues;
    }
    else if(object$distribution=="dalaplace"){
        # Use the connection between the variance and scale in ALaplace distribution
        alpha <- object$other$alpha;
        if(interval!="none"){
            scaleValues <- sqrt(greyboxForecast$variances * alpha^2 * (1-alpha)^2 / (alpha^2 + (1-alpha)^2));
            greyboxForecast$scale <- scaleValues;
        }
        if(interval=="prediction"){
            for(i in 1:nLevels){
                greyboxForecast$lower[,i] <- qalaplace(levelLow[,i],greyboxForecast$mean,scaleValues,alpha);
                greyboxForecast$upper[,i] <- qalaplace(levelUp[,i],greyboxForecast$mean,scaleValues,alpha);
            }
        }
    }
    else if(object$distribution=="ds"){
        # Use the connection between the variance and scale in S distribution
        if(interval!="none"){
            scaleValues <- (greyboxForecast$variances/120)^0.25;
            greyboxForecast$scale <- scaleValues;
        }
        if(interval=="prediction"){
            for(i in 1:nLevels){
                greyboxForecast$lower[,i] <- qs(levelLow[,i],greyboxForecast$mean,scaleValues);
                greyboxForecast$upper[,i] <- qs(levelUp[,i],greyboxForecast$mean,scaleValues);
            }
        }
    }
    else if(object$distribution=="dls"){
        # Use the connection between the variance and scale in S distribution
        if(interval!="none"){
            scaleValues <- (greyboxForecast$variances/120)^0.25;
            greyboxForecast$scale <- scaleValues;
        }
        if(interval=="prediction"){
            for(i in 1:nLevels){
                greyboxForecast$lower[,i] <- exp(qs(levelLow[,i],greyboxForecast$mean,scaleValues));
                greyboxForecast$upper[,i] <- exp(qs(levelUp[,i],greyboxForecast$mean,scaleValues));
            }
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
        greyboxForecast$mean <- exp(greyboxForecast$mean);
    }
    else if(object$distribution=="dgnorm"){
        # Use the connection between the variance and scale in Generalised Normal distribution
        if(interval!="none"){
            if(scaleModel){
                scaleValues <- greyboxForecast$variances;
            }
            else{
                scaleValues <- sqrt(greyboxForecast$variances*(gamma(1/object$other$shape)/gamma(3/object$other$shape)));
            }
            greyboxForecast$scale <- scaleValues;
        }
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qgnorm(levelLow,greyboxForecast$mean,scaleValues,object$other$shape);
            greyboxForecast$upper[] <- qgnorm(levelUp,greyboxForecast$mean,scaleValues,object$other$shape);
        }
    }
    else if(object$distribution=="dlgnorm"){
        # Use the connection between the variance and scale in Generalised Normal distribution
        if(interval!="none"){
            scaleValues <- sqrt(greyboxForecast$variances*(gamma(1/object$other$beta)/gamma(3/object$other$beta)));
            greyboxForecast$scale <- scaleValues;
        }
        if(interval=="prediction"){
            greyboxForecast$lower[] <- exp(qgnorm(levelLow,greyboxForecast$mean,scaleValues,object$other$beta));
            greyboxForecast$upper[] <- exp(qgnorm(levelUp,greyboxForecast$mean,scaleValues,object$other$beta));
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
        greyboxForecast$mean <- exp(greyboxForecast$mean);
    }
    else if(object$distribution=="dt"){
        # Use df estimated by the model and then construct conventional intervals. df=2 is the minimum in this model.
        df <- nobs(object) - nparam(object);
        if(interval=="prediction"){
            greyboxForecast$lower[] <- greyboxForecast$mean + sqrt(greyboxForecast$variances) * qt(levelLow,df);
            greyboxForecast$upper[] <- greyboxForecast$mean + sqrt(greyboxForecast$variances) * qt(levelUp,df);
        }
    }
    else if(object$distribution=="dfnorm"){
        if(interval=="prediction"){
            for(i in 1:nLevels){
                greyboxForecast$lower[,i] <- qfnorm(levelLow[,i],greyboxForecast$mean,sqrt(greyboxForecast$variances));
                greyboxForecast$upper[,i] <- qfnorm(levelUp[,i],greyboxForecast$mean,sqrt(greyboxForecast$variances));
            }
        }
        if(interval!="none"){
            # Correct the mean value
            greyboxForecast$mean <- (sqrt(2/pi)*sqrt(greyboxForecast$variances)*exp(-greyboxForecast$mean^2 /
                                                                                       (2*greyboxForecast$variances)) +
                                         greyboxForecast$mean*(1-2*pnorm(-greyboxForecast$mean/sqrt(greyboxForecast$variances))));
        }
        else{
            warning("The mean of Folded Normal distribution was not corrected. ",
                    "We only do that, when interval!='none'.",
                    call.=FALSE)
        }
    }
    else if(object$distribution=="drectnorm"){
        if(interval=="prediction"){
            for(i in 1:nLevels){
                greyboxForecast$lower[,i] <- qrectnorm(levelLow[,i],greyboxForecast$mean,sqrt(greyboxForecast$variances));
                greyboxForecast$upper[,i] <- qrectnorm(levelUp[,i],greyboxForecast$mean,sqrt(greyboxForecast$variances));
            }
        }
        if(interval!="none"){
            # Correct the mean value
            greyboxForecast$mean <- greyboxForecast$mean *
                                        (1-pnorm(0, greyboxForecast$mean, sqrt(greyboxForecast$variances))) +
                                    sqrt(greyboxForecast$variances) *
                                        dnorm(0, greyboxForecast$mean, sqrt(greyboxForecast$variances));
        }
        else{
            warning("The mean of Folded Normal distribution was not corrected. ",
                    "We only do that, when interval!='none'.",
                    call.=FALSE)
        }
    }
    else if(object$distribution=="dchisq"){
        greyboxForecast$mean <- greyboxForecast$mean^2;
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qchisq(levelLow,df=object$other$nu,ncp=greyboxForecast$mean);
            greyboxForecast$upper[] <- qchisq(levelUp,df=object$other$nu,ncp=greyboxForecast$mean);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- (greyboxForecast$lower)^2;
            greyboxForecast$upper[] <- (greyboxForecast$upper)^2;
        }
        if(interval!="none"){
            greyboxForecast$mean <- greyboxForecast$mean + extractScale(object);
            greyboxForecast$scale <- extractScale(object);
        }
        else{
            warning("The mean of Chi Squared distribution was not corrected. ",
                    "We only do that, when interval!='none'.",
                    call.=FALSE)
        }
    }
    else if(object$distribution=="dlnorm"){
        if(interval=="prediction"){
            if(scaleModel){
                sdlog <- sqrt(greyboxForecast$variances);
            }
            else{
                sdlog <- sqrt(greyboxForecast$variances - sigma(object)^2 + extractScale(object));
            }
            greyboxForecast$scale <- sdlog;
        }
        else if(interval=="confidence"){
            sdlog <- sqrt(greyboxForecast$variances);
            greyboxForecast$scale <- sdlog;
        }
        if(interval!="none"){
            greyboxForecast$lower[] <- qlnorm(levelLow,greyboxForecast$mean,sdlog);
            greyboxForecast$upper[] <- qlnorm(levelUp,greyboxForecast$mean,sdlog);
        }
        greyboxForecast$mean <- exp(greyboxForecast$mean);
    }
    else if(object$distribution=="dbcnorm"){
        if(interval!="none"){
            sigma <- sqrt(greyboxForecast$variances);
            greyboxForecast$scale <- sigma;
        }
        # If negative values were produced, zero them out
        if(any(greyboxForecast$mean<0)){
            greyboxForecast$mean[greyboxForecast$mean<0] <- 0;
        }
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qbcnorm(levelLow,greyboxForecast$mean,sigma,object$other$lambdaBC);
            greyboxForecast$upper[] <- qbcnorm(levelUp,greyboxForecast$mean,sigma,object$other$lambdaBC);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- (greyboxForecast$lower*object$other$lambdaBC+1)^{1/object$other$lambdaBC};
            greyboxForecast$upper[] <- (greyboxForecast$upper*object$other$lambdaBC+1)^{1/object$other$lambdaBC};
        }
        if(object$other$lambdaBC==0){
            greyboxForecast$mean[] <- exp(greyboxForecast$mean)
        }
        else{
            greyboxForecast$mean[] <- (greyboxForecast$mean*object$other$lambdaBC+1)^{1/object$other$lambdaBC};
        }
    }
    else if(object$distribution=="dlogitnorm"){
        if(interval=="prediction"){
            if(scaleModel){
                sigma <- sqrt(greyboxForecast$variances);
            }
            else{
                sigma <- sqrt(greyboxForecast$variances - sigma(object)^2 + extractScale(object)^2);
            }
            greyboxForecast$scale <- sigma;
        }
        else if(interval=="confidence"){
            sigma <- sqrt(greyboxForecast$variances);
            greyboxForecast$scale <- sigma;
        }
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qlogitnorm(levelLow,greyboxForecast$mean,sigma);
            greyboxForecast$upper[] <- qlogitnorm(levelUp,greyboxForecast$mean,sigma);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower)/(1+exp(greyboxForecast$lower));
            greyboxForecast$upper[] <- exp(greyboxForecast$upper)/(1+exp(greyboxForecast$upper));
        }
        greyboxForecast$mean[] <- exp(greyboxForecast$mean)/(1+exp(greyboxForecast$mean));
    }
    else if(object$distribution=="dinvgauss"){
        greyboxForecast$mean <- exp(greyboxForecast$mean);
        if(interval=="prediction"){
            greyboxForecast$scale <- extractScale(object);
            greyboxForecast$lower[] <- greyboxForecast$mean*qinvgauss(levelLow,mean=1,
                                                                      dispersion=greyboxForecast$scale);
            greyboxForecast$upper[] <- greyboxForecast$mean*qinvgauss(levelUp,mean=1,
                                                                      dispersion=greyboxForecast$scale);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
    }
    else if(object$distribution=="dgamma"){
        greyboxForecast$mean <- exp(greyboxForecast$mean);
        if(interval=="prediction"){
            greyboxForecast$scale <- extractScale(object);
            greyboxForecast$lower[] <- greyboxForecast$mean*qgamma(levelLow, shape=1/greyboxForecast$scale,
                                                                   scale=greyboxForecast$scale);
            greyboxForecast$upper[] <- greyboxForecast$mean*qgamma(levelUp, shape=1/greyboxForecast$scale,
                                                                   scale=greyboxForecast$scale);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
    }
    else if(object$distribution=="dexp"){
        greyboxForecast$mean <- exp(greyboxForecast$mean);
        if(interval=="prediction"){
            greyboxForecast$lower[] <- greyboxForecast$mean*qexp(levelLow, rate=1);
            greyboxForecast$upper[] <- greyboxForecast$mean*qexp(levelUp, rate=1);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
    }
    else if(object$distribution=="dlogis"){
        # Use the connection between the variance and scale in logistic distribution
        if(interval!="none"){
            scale <- sqrt(greyboxForecast$variances * 3 / pi^2);
            greyboxForecast$scale <- scale;
        }
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qlogis(levelLow,greyboxForecast$mean,scale);
            greyboxForecast$upper[] <- qlogis(levelUp,greyboxForecast$mean,scale);
        }
    }
    else if(object$distribution=="dpois"){
        greyboxForecast$mean <- exp(greyboxForecast$mean);
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qpois(levelLow,greyboxForecast$mean);
            greyboxForecast$upper[] <- qpois(levelUp,greyboxForecast$mean);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
        greyboxForecast$scale <- greyboxForecast$mean;
    }
    else if(object$distribution=="dnbinom"){
        greyboxForecast$mean <- exp(greyboxForecast$mean);
        if(interval!="none" && is.null(extractScale(object))){
            # This is a very approximate thing in order for something to work...
            greyboxForecast$scale <- abs(greyboxForecast$mean^2 / (greyboxForecast$variances - greyboxForecast$mean));
        }
        else{
            greyboxForecast$scale <- extractScale(object);
        }
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qnbinom(levelLow,mu=greyboxForecast$mean,size=greyboxForecast$scale);
            greyboxForecast$upper[] <- qnbinom(levelUp,mu=greyboxForecast$mean,size=greyboxForecast$scale);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
    }
    else if(object$distribution=="dbinom"){
        greyboxForecast$mean <- 1/(1+exp(greyboxForecast$mean));
        greyboxForecast$scale <- object$other$size;
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qbinom(levelLow,prob=greyboxForecast$mean,size=greyboxForecast$scale);
            greyboxForecast$upper[] <- qbinom(levelUp,prob=greyboxForecast$mean,size=greyboxForecast$scale);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
        greyboxForecast$mean <- greyboxForecast$mean * object$other$size;
    }
    else if(object$distribution=="dbeta"){
        greyboxForecast$shape1 <- greyboxForecast$mean;
        greyboxForecast$shape2 <- greyboxForecast$variances;
        greyboxForecast$mean <- greyboxForecast$shape1 / (greyboxForecast$shape1 + greyboxForecast$shape2);
        greyboxForecast$variances <- (greyboxForecast$shape1 * greyboxForecast$shape2 /
                                          ((greyboxForecast$shape1+greyboxForecast$shape2)^2 *
                                               (greyboxForecast$shape1 + greyboxForecast$shape2 + 1)));
        if(interval=="prediction"){
            greyboxForecast$lower <- qbeta(levelLow,greyboxForecast$shape1,greyboxForecast$shape2);
            greyboxForecast$upper <- qbeta(levelUp,greyboxForecast$shape1,greyboxForecast$shape2);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower <- (greyboxForecast$mean + qt(levelLow,df=object$df.residual)*
                                          sqrt(greyboxForecast$variances/nobs(object)));
            greyboxForecast$upper <- (greyboxForecast$mean + qt(levelUp,df=object$df.residual)*
                                          sqrt(greyboxForecast$variances/nobs(object)));
        }
    }
    else if(object$distribution=="plogis"){
        # The intervals are based on the assumption that a~N(0, sigma^2), and p=exp(a) / (1 + exp(a))
        if(interval!="none"){
            greyboxForecast$scale <- extractScale(object);
        }
        greyboxForecast$mean <- plogis(greyboxForecast$location, location=0, scale=1);

        if(interval!="none"){
            greyboxForecast$lower[] <- plogis(qnorm(levelLow, greyboxForecast$location, sqrt(greyboxForecast$variances)),
                                            location=0, scale=1);
            greyboxForecast$upper[] <- plogis(qnorm(levelUp, greyboxForecast$location, sqrt(greyboxForecast$variances)),
                                            location=0, scale=1);
        }
    }
    else if(object$distribution=="pnorm"){
        # The intervals are based on the assumption that a~N(0, sigma^2), and pnorm link
        if(interval!="none"){
            greyboxForecast$scale <- extractScale(object);
        }

        greyboxForecast$mean <- pnorm(greyboxForecast$location, mean=0, sd=1);

        if(interval!="none"){
            greyboxForecast$lower[] <- pnorm(qnorm(levelLow, greyboxForecast$location, sqrt(greyboxForecast$variances)),
                                            mean=0, sd=1);
            greyboxForecast$upper[] <- pnorm(qnorm(levelUp, greyboxForecast$location, sqrt(greyboxForecast$variances)),
                                            mean=0, sd=1);
        }
    }

    # If there is an occurrence part of the model, use it
    if(is.null(occurrence) && is.occurrence(object$occurrence) &&
       is.list(object$occurrence) &&
       (is.null(object$occurrence$occurrence) || object$occurrence$occurrence!="provided")){
        greyboxForecast$mean <- greyboxForecast$mean * greyboxForecast$occurrence$mean;
        #### This is weird and probably wrong. But I don't know yet what the confidence intervals mean in case of occurrence model.
        if(interval=="confidence"){
            greyboxForecast$lower[] <- greyboxForecast$lower * greyboxForecast$occurrence$mean;
            greyboxForecast$upper[] <- greyboxForecast$upper * greyboxForecast$occurrence$mean;
        }
    }
    else{
        # If occurrence was provided, modify everything
        if(!is.null(occurrence)){
            greyboxForecast$mean[] <- greyboxForecast$mean * occurrence;
            greyboxForecast$lower[] <- greyboxForecast$lower * occurrence;
            greyboxForecast$upper[] <- greyboxForecast$upper * occurrence;
        }
    }

    greyboxForecast$level <- cbind(levelLow, levelUp);
    colnames(greyboxForecast$level) <- c(paste0("Lower",c(1:nLevels)),paste0("Upper",c(1:nLevels)));
    greyboxForecast$level <- rbind(switch(side,
                                          "both"=c((1-levelOriginal)/2,(1+levelOriginal)/2),
                                          "lower"=c(levelOriginal,rep(0,nLevels)),
                                          "upper"=c(rep(0,nLevels),levelOriginal)),
                                   greyboxForecast$level);
    rownames(greyboxForecast$level) <- c("Original",paste0("h",c(1:h)));
    greyboxForecast$newdataProvided <- newdataProvided;
    return(structure(greyboxForecast,class="predict.greybox"));
}

#' Forecasting using greybox functions
#'
#' The functions allow producing forecasts based on the provided model and newdata.
#'
#' \code{predict} produces predictions for the provided model and \code{newdata}. If
#' \code{newdata} is not provided, then the data from the model is extracted and the
#' fitted values are reproduced. This might be useful when confidence / prediction
#' intervals are needed for the in-sample values.
#'
#' \code{forecast} function produces forecasts for \code{h} steps ahead. There are four
#' scenarios in this function:
#' \enumerate{
#' \item If the \code{newdata} is  not provided, then it will produce forecasts of the
#' explanatory variables to the horizon \code{h} (using \code{es} from smooth package
#' or using Naive if \code{smooth} is not installed) and use them as \code{newdata}.
#' \item If \code{h} and \code{newdata} are provided, then the number of rows to use
#' will be regulated by \code{h}.
#' \item If \code{h} is \code{NULL}, then it is set equal to the number of rows in
#' \code{newdata}.
#' \item If both \code{h} and \code{newdata} are not provided, then it will use the
#' data from the model itself, reproducing the fitted values.
#' }
#' After forming the \code{newdata} the \code{forecast} function calls for
#' \code{predict}, so you can provide parameters \code{interval}, \code{level} and
#' \code{side} in the call for \code{forecast}.
#'
#' @aliases forecast.greybox forecast.alm
#' @param object Time series model for which forecasts are required.
#' @param newdata The new data needed in order to produce forecasts.
#' @param interval Type of intervals to construct: either "confidence" or
#' "prediction". Can be abbreviated
#' @param level Confidence level. Defines width of prediction interval.
#' @param side What type of interval to produce: \code{"both"} - produces both
#' lower and upper bounds of the interval, \code{"upper"} - upper only, \code{"lower"}
#' - respectively lower only. In the \code{"both"} case the probability is split into
#' two parts: ((1-level)/2, (1+level)/2). When \code{"upper"} is specified, then
#' the intervals for (0, level) are constructed Finally, with \code{"lower"} the interval
#' for (1-level, 1) is returned.
#' @param h The forecast horizon.
#' @param ...  Other arguments passed to \code{vcov} function (see \link[greybox]{coef.alm}
#' for details).
#' @return \code{predict.greybox()} returns object of class "predict.greybox",
#' which contains:
#' \itemize{
#' \item \code{model} - the estimated model.
#' \item \code{mean} - the expected values.
#' \item \code{fitted} - fitted values of the model.
#' \item \code{lower} - lower bound of prediction / confidence intervals.
#' \item \code{upper} - upper bound of prediction / confidence intervals.
#' \item \code{level} - confidence level.
#' \item \code{newdata} - the data provided in the call to the function.
#' \item \code{variances} - conditional variance for the holdout sample.
#' In case of \code{interval="prediction"} includes variance of the error.
#' }
#'
#' \code{predict.alm()} is based on \code{predict.greybox()} and returns
#' object of class "predict.alm", which in addition contains:
#' \itemize{
#' \item \code{location} - the location parameter of the distribution.
#' \item \code{scale} - the scale parameter of the distribution.
#' \item \code{distribution} - name of the fitted distribution.
#' }
#'
#' \code{forecast()} functions return the same "predict.alm" and
#' "predict.greybox" classes, with the same set of output variables.
#'
#' @template author
#' @seealso \link[stats]{predict.lm}
#' @keywords ts univar
#' @examples
#'
#' xreg <- cbind(rlaplace(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rlaplace(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' inSample <- xreg[1:80,]
#' outSample <- xreg[-c(1:80),]
#'
#' ourModel <- alm(y~x1+x2, inSample, distribution="dlaplace")
#'
#' predict(ourModel,outSample)
#' predict(ourModel,outSample,interval="c")
#'
#' plot(predict(ourModel,outSample,interval="p"))
#' plot(forecast(ourModel,h=10,interval="p"))
#'
#' @rdname predict.greybox
#' @export
predict.greybox <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                            level=0.95, side=c("both","upper","lower"), ...){
    interval <- match.arg(interval);
    side <- match.arg(side);

    scaleModel <- is.scale(object$scale);
    parameters <- coef.greybox(object);
    parametersNames <- names(parameters);

    nLevels <- length(level);
    levelLow <- levelUp <- vector("numeric",nLevels);
    if(side=="upper"){
        levelLow[] <- 0;
        levelUp[] <- level;
    }
    else if(side=="lower"){
        levelLow[] <- 1-level;
        levelUp[] <- 1;
    }
    else{
        levelLow[] <- (1-level) / 2;
        levelUp[] <- (1+level) / 2;
    }
    paramQuantiles <- qt(c(levelLow, levelUp),df=object$df.residual);

    if(is.null(newdata)){
        matrixOfxreg <- object$data;
        newdataProvided <- FALSE;
        # The first column is the response variable. Either substitute it by ones or remove it.
        if(any(parametersNames=="(Intercept)")){
            matrixOfxreg[,1] <- 1;
        }
        else{
            matrixOfxreg <- matrixOfxreg[,-1,drop=FALSE];
        }
    }
    else{
        newdataProvided <- TRUE;

        if(!is.data.frame(newdata)){
            if(is.vector(newdata)){
                newdataNames <- names(newdata);
                newdata <- matrix(newdata, nrow=1, dimnames=list(NULL, newdataNames));
            }
            newdata <- as.data.frame(newdata);
        }
        else{
            dataOrders <- unlist(lapply(newdata,is.ordered));
            # If there is an ordered factor, remove the bloody ordering!
            if(any(dataOrders)){
                newdata[dataOrders] <- lapply(newdata[dataOrders],function(x) factor(x, levels=levels(x), ordered=FALSE));
            }
        }

        # The gsub is needed in order to remove accidental special characters
        colnames(newdata) <- make.names(colnames(newdata), unique=TRUE);

        # Extract the formula and get rid of the response variable
        testFormula <- formula(object);

        # If the user asked for trend, but it's not in the data, add it
        if(any(all.vars(testFormula)=="trend") && all(colnames(newdata)!="trend")){
            newdata <- cbind(newdata,trend=nobs(object)+c(1:nrow(newdata)));
        }

        testFormula[[2]] <- NULL;
        # Expand the data frame
        newdataExpanded <- model.frame(testFormula, newdata);
        interceptIsNeeded <- attr(terms(newdataExpanded),"intercept")!=0;
        matrixOfxreg <- model.matrix(newdataExpanded,data=newdataExpanded);
        matrixOfxreg <- matrixOfxreg[,parametersNames,drop=FALSE];
    }

    h <- nrow(matrixOfxreg);

    if(object$distribution=="dbeta"){
        parametersNames <- substr(parametersNames[1:(length(parametersNames)/2)],8,nchar(parametersNames));
    }

    if(any(is.calm(object),is.greyboxD(object))){
        if(ncol(matrixOfxreg)==2){
            colnames(matrixOfxreg) <- parametersNames;
        }
        else{
            colnames(matrixOfxreg)[1] <- parametersNames[1];
        }
        matrixOfxreg <- matrixOfxreg[,parametersNames,drop=FALSE];
    }

    if(!is.matrix(matrixOfxreg)){
        matrixOfxreg <- as.matrix(matrixOfxreg);
        h <- nrow(matrixOfxreg);
    }

    if(h==1){
        matrixOfxreg <- matrix(matrixOfxreg, nrow=1);
    }

    if(object$distribution=="dbeta"){
        # We predict values for shape1 and shape2 and write them down in mean and variance.
        ourForecast <- as.vector(exp(matrixOfxreg %*% parameters[1:(length(parameters)/2)]));
        vectorOfVariances <- as.vector(exp(matrixOfxreg %*% parameters[-c(1:(length(parameters)/2))]));
        # ourForecast <- ourForecast / (ourForecast + as.vector(exp(matrixOfxreg %*% parameters[-c(1:(length(parameters)/2))])));

        yLower <- NULL;
        yUpper <- NULL;
    }
    else{
        ourForecast <- as.vector(matrixOfxreg %*% parameters);
        vectorOfVariances <- NULL;

        if(interval!="none"){
            ourVcov <- vcov(object, ...);
            # abs is needed for some cases, when the likelihood was not fully optimised
            vectorOfVariances <- abs(diag(matrixOfxreg %*% ourVcov %*% t(matrixOfxreg)));
            yUpper <- yLower <- matrix(NA, h, nLevels);
            if(interval=="confidence"){
                for(i in 1:nLevels){
                    yLower[,i] <- ourForecast + paramQuantiles[i] * sqrt(vectorOfVariances);
                    yUpper[,i] <- ourForecast + paramQuantiles[i+nLevels] * sqrt(vectorOfVariances);
                }
            }
            else if(interval=="prediction"){
                if(scaleModel){
                    sigmaValues <- predict.scale(object$scale, newdata, interval="none", ...)$mean;
                    # Get variance from the scale
                    sigmaValues[] <- switch(object$distribution,
                                            "dnorm"=,
                                            "dlnorm"=sigmaValues,
                                            "dbcnorm"=,
                                            "dlogitnorm"=,
                                            "dfnorm"=,
                                            "drectnorm"=sigmaValues^2,
                                            "dlaplace"=,
                                            "dllaplace"=2*sigmaValues^2,
                                            "dalaplace"=sigmaValues^2*
                                                ((1-object$other$alpha)^2+object$other$alpha^2)/
                                                (object$other$alpha*(1-object$other$alpha))^2,
                                            "ds"=,
                                            "dls"=120*sigmaValues^4,
                                            "dgnorm"=,
                                            "dlgnorm"=sigmaValues^2*gamma(3/object$other$shape)/
                                                gamma(1/object$other$shape),
                                            "dlogis"=sigmaValues*pi/sqrt(3),
                                            "dgamma"=,
                                            "dinvgauss"=,
                                            sigmaValues);
                }
                else{
                    sigmaValues <- sigma(object)^2;
                }
                vectorOfVariances[] <- vectorOfVariances + sigmaValues;
                for(i in 1:nLevels){
                    yLower[,i] <- ourForecast + paramQuantiles[i] * sqrt(vectorOfVariances);
                    yUpper[,i] <- ourForecast + paramQuantiles[i+nLevels] * sqrt(vectorOfVariances);
                }
            }

            colnames(yLower) <- switch(side,
                                       "both"=,
                                       "lower"=paste0("Lower bound (",levelLow*100,"%)"),
                                       "upper"=rep("Lower 0%",nLevels));

            colnames(yUpper) <- switch(side,
                                       "both"=,
                                       "upper"=paste0("Upper bound (",levelUp*100,"%)"),
                                       "lower"=rep("Upper 100%",nLevels));
        }
        else{
            yLower <- NULL;
            yUpper <- NULL;
        }
    }

    ourModel <- list(model=object, mean=ourForecast, lower=yLower, upper=yUpper, level=c(levelLow, levelUp), newdata=newdata,
                     variances=vectorOfVariances, newdataProvided=newdataProvided);
    return(structure(ourModel,class="predict.greybox"));
}

# The internal function for the predictions from the model with ARI
predict_almari <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                            level=0.95, side=c("both","upper","lower"), ...){
    interval <- match.arg(interval);
    side <- match.arg(side);

    y <- actuals(object, all=FALSE);

    # Write down the AR order
    if(is.null(object$other$arima)){
        arOrder <- 0;
    }
    else{
        arOrder <- object$other$orders[1];
    }

    ariOrder <- length(object$other$polynomial);
    ariParameters <- object$other$polynomial;
    ariNames <- names(ariParameters);

    parameters <- coef.greybox(object);
    # Split the parameters into normal and polynomial (for ARI)
    if(arOrder>0){
        parameters <- parameters[-c(length(parameters)+(1-arOrder):0)];
    }
    nonariParametersNumber <- length(parameters);
    parametersNames <- names(parameters);

    nLevels <- length(level);
    levelLow <- levelUp <- vector("numeric",nLevels);
    if(side=="upper"){
        levelLow[] <- 0;
        levelUp[] <- level;
    }
    else if(side=="lower"){
        levelLow[] <- 1-level;
        levelUp[] <- 1;
    }
    else{
        levelLow[] <- (1-level) / 2;
        levelUp[] <- (1+level) / 2;
    }
    paramQuantiles <- qt(c(levelLow, levelUp),df=object$df.residual);

    if(is.null(newdata)){
        matrixOfxreg <- object$data[,-1,drop=FALSE];
        newdataProvided <- FALSE;
        interceptIsNeeded <- any(names(coef(object))=="(Intercept)");
        if(interceptIsNeeded){
            matrixOfxreg <- cbind(1,matrixOfxreg);
        }
    }
    else{
        newdataProvided <- TRUE;

        if(!is.data.frame(newdata)){
            if(is.vector(newdata)){
                newdataNames <- names(newdata);
                newdata <- matrix(newdata, nrow=1, dimnames=list(NULL, newdataNames));
            }
            newdata <- as.data.frame(newdata);
        }
        else{
            dataOrders <- unlist(lapply(newdata,is.ordered));
            # If there is an ordered factor, remove the bloody ordering!
            if(any(dataOrders)){
                newdata[dataOrders] <- lapply(newdata[dataOrders],function(x) factor(x, levels=levels(x), ordered=FALSE));
            }
        }

        colnames(newdata) <- make.names(colnames(newdata), unique=TRUE);
        # Extract the formula and get rid of the response variable
        testFormula <- formula(object);
        testFormula[[2]] <- NULL;
        # Expand the data frame
        newdataExpanded <- model.frame(testFormula, newdata);
        interceptIsNeeded <- attr(terms(newdataExpanded),"intercept")!=0;
        matrixOfxreg <- model.matrix(newdataExpanded,data=newdataExpanded);

        matrixOfxreg <- matrixOfxreg[,parametersNames,drop=FALSE];
    }

    if(object$distribution=="dbeta"){
        parametersNames <- substr(parametersNames[1:(length(parametersNames)/2)],8,nchar(parametersNames));
    }

    if(any(is.calm(object),is.greyboxD(object))){
        matrixOfxreg <- as.matrix(cbind(rep(1,nrow(newdata)),newdata[,-1]));
        if(ncol(matrixOfxreg)==2){
            colnames(matrixOfxreg) <- parametersNames;
        }
        else{
            colnames(matrixOfxreg)[1] <- parametersNames[1];
        }
        matrixOfxreg <- matrixOfxreg[,parametersNames,drop=FALSE];
    }

    if(!is.matrix(matrixOfxreg)){
        matrixOfxreg <- matrix(matrixOfxreg,ncol=1);
    }
    h <- nrow(matrixOfxreg);

    if(h==1){
        matrixOfxreg <- matrix(matrixOfxreg, nrow=1);
    }

    # Add ARI polynomials to the parameters
    parameters <- c(parameters,ariParameters);

    # If the newdata is provided, do the recursive thingy
    if(newdataProvided){
    # Fill in the tails with the available data
        if(any(object$distribution==c("plogis","pnorm"))){
            matrixOfxregFull <- cbind(matrixOfxreg, matrix(NA,h,ariOrder,dimnames=list(NULL,ariNames)));
            matrixOfxregFull <- rbind(matrix(NA,ariOrder,ncol(matrixOfxregFull)),matrixOfxregFull);
            if(interceptIsNeeded){
                matrixOfxregFull[1:ariOrder,-1] <- tail(object$data[,-1,drop=FALSE],ariOrder);
                matrixOfxregFull[1:ariOrder,1] <- 1;
            }
            else{
                matrixOfxregFull[1:ariOrder,] <- tail(object$data[,-1,drop=FALSE],ariOrder);
            }
            h <- h+ariOrder;
        }
        else{
            matrixOfxregFull <- cbind(matrixOfxreg, matrix(NA,h,ariOrder,dimnames=list(NULL,ariNames)));
            for(i in 1:ariOrder){
                matrixOfxregFull[1:min(h,i),nonariParametersNumber+i] <- tail(y,i)[1:min(h,i)];
            }
        }

        # Transform the lagged response variables
        if(any(object$distribution==c("dlnorm","dllaplace","dls","dpois","dnbinom","dbinom"))){
            if(any(y==0) & !is.occurrence(object$occurrence)){
                # Use Box-Cox if there are zeroes
                matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)] <- (matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)]^0.01-1)/0.01;
                colnames(matrixOfxregFull)[nonariParametersNumber+c(1:ariOrder)] <- paste0(ariNames,"Box-Cox");
            }
            else{
                matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)] <- log(matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)]);
                colnames(matrixOfxregFull)[nonariParametersNumber+c(1:ariOrder)] <- paste0(ariNames,"Log");
            }
        }
        else if(object$distribution=="dbcnorm"){
            # Use Box-Cox if there are zeroes
            matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)] <- (matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)]^
                                                                            object$other$lambdaBC-1)/object$other$lambdaBC;
            colnames(matrixOfxregFull)[nonariParametersNumber+c(1:ariOrder)] <- paste0(ariNames,"Box-Cox");
        }
        else if(object$distribution=="dlogitnorm"){
            matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)] <-
                log(matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)]/
                        (1-matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)]));
            colnames(matrixOfxregFull)[nonariParametersNumber+c(1:ariOrder)] <- paste0(ariNames,"Logit");
        }
        else if(object$distribution=="dchisq"){
            matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)] <- sqrt(matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)]);
            colnames(matrixOfxregFull)[nonariParametersNumber+c(1:ariOrder)] <- paste0(ariNames,"Sqrt");
        }

        # if(object$distribution=="dbeta"){
        #     # We predict values for shape1 and shape2 and write them down in mean and variance.
        #     ourForecast <- as.vector(exp(matrixOfxregFull %*% parameters[1:(length(parameters)/2)]));
        #     vectorOfVariances <- as.vector(exp(matrixOfxregFull %*% parameters[-c(1:(length(parameters)/2))]));
        #     # ourForecast <- ourForecast / (ourForecast + as.vector(exp(matrixOfxregFull %*% parameters[-c(1:(length(parameters)/2))])));
        #
        #     lower <- NULL;
        #     upper <- NULL;
        # }
        # else{

        # Produce forecasts iteratively
        ourForecast <- vector("numeric", h);
        for(i in 1:h){
            ourForecast[i] <- matrixOfxregFull[i,] %*% parameters;
            for(j in 1:ariOrder){
                if(i+j-1==h){
                    break;
                }
                matrixOfxregFull[i+j,nonariParametersNumber+j] <- ourForecast[i];
            }
        }

        if(any(object$distribution==c("plogis","pnorm"))){
            matrixOfxreg <- matrixOfxregFull[-c(1:ariOrder),1:(nonariParametersNumber+arOrder),drop=FALSE];
            ourForecast <- ourForecast[-c(1:ariOrder)];
        }
        else{
            matrixOfxreg <- matrixOfxregFull[,1:(nonariParametersNumber+arOrder),drop=FALSE];
        }
    }
    else{
        ourForecast <- object$mu;
    }
    vectorOfVariances <- NULL;

    if(interval!="none"){
        ourVcov <- vcov(object, ...);
        # abs is needed for some cases, when the likelihoond was not fully optimised
        vectorOfVariances <- abs(diag(matrixOfxreg %*% ourVcov %*% t(matrixOfxreg)));

        yUpper <- yLower <- matrix(NA, h, nLevels);
        if(interval=="confidence"){
            for(i in 1:nLevels){
                yLower[,i] <- ourForecast + paramQuantiles[i] * sqrt(vectorOfVariances);
                yUpper[,i] <- ourForecast + paramQuantiles[i+nLevels] * sqrt(vectorOfVariances);
            }
        }
        else if(interval=="prediction"){
            vectorOfVariances[] <- vectorOfVariances + sigma(object)^2;
            for(i in 1:nLevels){
                yLower[,i] <- ourForecast + paramQuantiles[i] * sqrt(vectorOfVariances);
                yUpper[,i] <- ourForecast + paramQuantiles[i+nLevels] * sqrt(vectorOfVariances);
            }
        }

        colnames(yLower) <- switch(side,
                                   "both"=,
                                   "lower"=paste0("Lower bound (",levelLow*100,"%)"),
                                   "upper"=rep("Lower 0%",nLevels));

        colnames(yUpper) <- switch(side,
                                   "both"=,
                                   "upper"=paste0("Upper bound (",levelUp*100,"%)"),
                                   "lower"=rep("Upper 100%",nLevels));
    }
    else{
        yLower <- NULL;
        yUpper <- NULL;
    }
    # }

    ourModel <- list(model=object, mean=ourForecast, lower=yLower, upper=yUpper, level=c(levelLow, levelUp), newdata=newdata,
                     variances=vectorOfVariances, newdataProvided=newdataProvided);
    return(structure(ourModel,class="predict.greybox"));
}

#' @rdname predict.greybox
#' @export
predict.scale <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                          level=0.95, side=c("both","upper","lower"), ...){
    scalePredicted <- predict.greybox(object, newdata, interval, level, side, ...);

    # Fix the predicted scale
    scalePredicted$mean[] <- exp(scalePredicted$mean);
    scalePredicted$mean[] <- switch(object$distribution,
                                    "dnorm"=,
                                    "dlnorm"=scalePredicted$mean,
                                    "dbcnorm"=,
                                    "dlogitnorm"=,
                                    "dfnorm"=,
                                    "drectnorm"=,
                                    "dlogis"=sqrt(scalePredicted$mean),
                                    "dlaplace"=,
                                    "dllaplace"=,
                                    "dalaplace"=scalePredicted$mean,
                                    "ds"=,
                                    "dls"=scalePredicted$mean^2,
                                    "dgnorm"=,
                                    "dlgnorm"=scalePredicted$mean^{1/object$other},
                                    "dgamma"=sqrt(scalePredicted$mean)+1,
                                    # This is based on polynomial from y = (x-1)^2/x
                                    "dinvgauss"=(scalePredicted$mean+2+
                                                     sqrt(scalePredicted$mean^2+
                                                              4*scalePredicted$mean))/2,
                                    scalePredicted$mean);
    return(scalePredicted);
}

#' @importFrom generics forecast
#' @export
generics::forecast

#' @rdname predict.greybox
#' @export
forecast.greybox <- function(object, newdata=NULL, h=NULL, ...){
    if(!is.null(newdata) & is.null(h)){
        h <- nrow(newdata);
    }

    if(!is.null(newdata) & !is.null(h)){
        if(nrow(newdata)>h){
            newdata <- head(newdata, h);
        }
        # If not enough values in the newdata, use naive
        else if(nrow(newdata)<h){
            warning("Not enough observations in the newdata. Using Naive in order to fill in the values.", call.=FALSE);
            newdata <- rbind(newdata,newdata[rep(nrow(newdata),h-nrow(newdata)),]);
        }
    }
    else if(is.null(newdata) & !is.null(h)){
        warning("No newdata provided, the values will be forecasted", call.=FALSE, immediate.=TRUE);
        if(ncol(object$data)>1){

            # If smooth is installed, use ADAM
            if(requireNamespace("smooth", quietly=TRUE) && (packageVersion("smooth")>="3.0.0")){
                newdata <- matrix(NA, h, ncol(object$data)-1, dimnames=list(NULL, colnames(object$data)[-1]));

                # If the user asked for trend, but it's not in the data, add it
                if(any(all.vars(formula(object))=="trend")){
                    newdata[,"trend"] <- nobs(object)+c(1:h);
                }
                for(i in 1:ncol(newdata)){
                    if(colnames(newdata)[i]!="trend"){
                        newdata[,i] <- smooth::adam(object$data[,i+1], occurrence="i", h=h)$forecast;
                    }
                }
            }
            # Otherwise use Naive
            else{
                newdata <- matrix(object$data[nobs(object),], h, ncol(object$data), byrow=TRUE,
                                  dimnames=list(NULL, colnames(object$data)));
                # If the user asked for trend, but it's not in the data, add it
                if(any(all.vars(formula(object))=="trend")){
                    newdata[,"trend"] <- nobs(object)+c(1:h);
                }
            }
        }
        else{
            newdata <- matrix(NA, h, 1, dimnames=list(NULL, colnames(object$data)[1]));
        }
    }
    return(predict(object, newdata, ...));
}

#' @rdname predict.greybox
#' @export
forecast.alm <- function(object, newdata=NULL, h=NULL, ...){
    if(!is.null(newdata) & is.null(h)){
        h <- nrow(newdata);
    }

    if(!is.null(newdata) & !is.null(h)){
        if(nrow(newdata)>h){
            newdata <- head(newdata, h);
        }
        # If not enough values in the newdata, use naive
        else if(nrow(newdata)<h){
            warning("Not enough observations in the newdata. Using Naive in order to fill in the values.", call.=FALSE);
            newdata <- rbind(newdata,newdata[rep(nrow(newdata),h-nrow(newdata)),]);
        }
    }
    else if(is.null(newdata) & !is.null(h)){
        warning("No newdata provided, the values will be forecasted", call.=FALSE, immediate.=TRUE);
        if(ncol(object$data)>1){
            # If smooth is installed, use ADAM
            if(requireNamespace("smooth", quietly=TRUE) && (packageVersion("smooth")>="3.0.0")){
                if(!is.null(object$other$polynomial)){
                    ariLength <- length(object$other$polynomial);
                    newdata <- matrix(NA, h, ncol(object$data)-ariLength-1,
                                      dimnames=list(NULL, colnames(object$data)[-c(1, (ncol(object$data)-ariLength+1):ncol(object$data))]));
                }
                else{
                    newdata <- matrix(NA, h, ncol(object$data)-1, dimnames=list(NULL, colnames(object$data)[-1]));
                }

                # If the user asked for trend, but it's not in the data, add it
                if(any(all.vars(formula(object))=="trend")){
                    newdata[,"trend"] <- nobs(object)+c(1:h);
                }
                for(i in 1:ncol(newdata)){
                    if(colnames(newdata)[i]!="trend"){
                        newdata[,i] <- smooth::adam(object$data[,i+1], occurrence="i", h=h)$forecast;
                    }
                }
            }
            # Otherwise use Naive
            else{
                newdata <- matrix(object$data[nobs(object),], h, ncol(object$data), byrow=TRUE,
                                  dimnames=list(NULL, colnames(object$data)));
                # If the user asked for trend, but it's not in the data, add it
                if(any(all.vars(formula(object))=="trend")){
                    newdata[,"trend"] <- nobs(object)+c(1:h);
                }
            }
        }
        else{
            newdata <- matrix(NA, h, 1, dimnames=list(NULL, colnames(object$data)[1]));
        }
    }
    return(predict(object, newdata, ...));
}

#' @importFrom generics accuracy
#' @export
generics::accuracy

#' Error measures for an estimated model
#'
#' Function produces error measures for the provided object and the holdout values of the
#' response variable. Note that instead of parameters \code{x}, \code{test}, the function
#' accepts the vector of values in \code{holdout}. Also, the parameters \code{d} and \code{D}
#' are not supported - MASE is always calculated via division by first differences.
#'
#' The function is a wrapper for the \link[greybox]{measures} function and is implemented
#' for convenience.
#'
#' @template author
#'
#' @param object The estimated model or a forecast from the estimated model generated via
#' either \code{predict()} or \code{forecast()} functions.
#' @param holdout The vector of values of the response variable in the holdout (test) set.
#' If not provided, then the function will return the in-sample error measures.
#' @param ... Other variables passed to the \code{forecast()} function (e.g. \code{newdata}).
#' @examples
#'
#' xreg <- cbind(rlaplace(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rlaplace(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' ourModel <- alm(y~x1+x2+trend, xreg, subset=c(1:80), distribution="dlaplace")
#'  accuracy(predict(ourModel,xreg[-c(1:80),]), xreg[-c(1:80),"y"])
#' @rdname accuracy
#' @export
accuracy.greybox <- function(object, holdout=NULL, ...){
    if(is.null(holdout)){
        return(measures(actuals(object), fitted(object), actuals(object)));
    }
    else{
        h <- length(holdout);
        return(measures(holdout, forecast(object, h=h, ...)$mean, actuals(object)));
    }
}

#' @rdname accuracy
#' @export
accuracy.predict.greybox <- function(object, holdout=NULL, ...){
    if(is.null(holdout)){
        return(accuracy(object$model));
    }
    else{
        return(measures(holdout, object$mean, actuals(object)));
    }
}
