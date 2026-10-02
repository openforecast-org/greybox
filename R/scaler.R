#' Scale Model
#'
#' This method produces a model for scale of distribution for the provided pre-estimated model.
#' The model can be estimated either via \code{lm} or \code{alm}.
#'
#' This function is useful, when you suspect a heteroscedasticity in your model and want to
#' fit a model for the scale of the pre-specified distribution. This function is complementary
#' for \code{lm} or \code{alm}.
#'
#' @template author
#' @template keywords
#'
#' @param object The pre-estimated \code{alm} or \code{lm} model.
#' @param formula The formula for scale. It should start with ~ and contain all variables
#' that should impact the scale.
#' @param data The data, on which the scale model needs to be estimated. If not provided,
#' then the one used in the \code{object} is used.
#' @param parameters The parameters to use in the model. Only needed if you know the parameters
#' in advance or want to test yours.
#' @param ... Other parameters to pass to the method, including those explained in
#' \link[greybox]{alm} (e.g. parameters for optimiser).
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+sqrt(exp(0.8+0.2*xreg[,1]))*rnorm(100,0,1),
#'               xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' # Estimate the location model
#' ourModel <- alm(y~.,xreg)
#' # Estimate the scale model
#' ourScale <- sm(ourModel,formula=~x1+x2)
#' # Summary of the scale model
#' summary(ourScale)
#'
#' @rdname sm
#' @export
sm <- function(object, ...) UseMethod("sm")

#' @rdname sm
#' @export
sm.default <- function(object, formula=NULL, data=NULL, parameters=NULL, ...){
    # The function creates a scale model for the provided model
    distribution <- "dnorm";
    if(is.null(data)){
        if(is.matrix(object$model)){
            data <- object$model;
        }
        else if(is.matrix(object$data)){
            data <- object$data;
        }
    }
    if(is.null(formula)){
        formula <- formula(object);
        formula[[2]] <- NULL;
    }
    return(do.call("scaler",list(formula, data, object$call$subset, object$call$na.action,
                                 distribution, fitted(object), actuals(object), residuals(object),
                                 parameters, cl=object$call, ...)));
}

#' @rdname sm
#' @export
sm.lm <- function(object, formula=NULL, data=NULL,
                  parameters=NULL, ...){
    # The function creates a scale model for the provided model
    distribution <- "dnorm";
    if(is.null(data)){
        data <- object$model;
    }
    if(is.null(formula)){
        formula <- formula(object);
        formula[[2]] <- NULL;
    }
    return(do.call("scaler",list(formula, data, object$call$subset, object$call$na.action,
                                 distribution, fitted(object), actuals(object), residuals(object),
                                 parameters, cl=object$call, ...)));
}

#' @rdname sm
#' @export
sm.alm <- function(object, formula=NULL, data=NULL,
                   # orders=c(0,0,0),
                   parameters=NULL, ...){
    # The function creates a scale model for the provided model
    distribution <- object$distribution;
    cl <- object$call;
    if(is.null(data)){
        data <- object$call$data;
    }
    if(is.null(formula)){
        formula <- formula(object);
        formula[[2]] <- NULL;
    }
    return(do.call("scaler",list(formula, data, object$call$subset, object$call$na.action,
                                 distribution, object$mu, actuals(object), residuals(object),
                                 parameters, object$occurrence, object$other,
                                 recursive=!is.null(object$other$arima), cl=cl, ...)));
}

# The distributions built on the Normal one: their scale is the variance sigma^2, as in the
# ADAM monograph, while their densities take the standard deviation, sqrt(scale).
scaleNormalBased <- c("dnorm","dlnorm","dbcnorm","dlogitnorm","dfnorm","drectnorm");

scaler <- function(formula, data, subset=NULL, na.action=NULL, distribution, mu, y, residuals,
                   parameters=NULL, occurrence=NULL, other=NULL, recursive=FALSE, ...){
    # The function estimates the scale model

    # Start measuring the time of calculations
    startTime <- Sys.time();

    cl <- match.call();

    obsInsample <- length(y);

    # Extract the other value
    if(!is.null(other)){
        other <- switch(distribution,
                        "dgnorm"=,
                        "dlgnorm"=other$shape,
                        "dalaplace"=other$alpha,
                        "dnbinom"=other$size,
                        "dchisq"=other$nu,
                        "dbcnorm"=other$lambdaBC,
                        "dt"=other$nu,
                        NULL);
    }

    #### Ellipsis values ####
    ellipsis <- match.call(expand.dots = FALSE)$`...`;

    # Fisher Information
    if(is.null(ellipsis$FI)){
        FI <- FALSE;
    }
    else{
        FI <- ellipsis$FI;
    }
    # Starting values for the optimiser
    if(is.null(ellipsis$B)){
        B <- NULL;
    }
    else{
        B <- ellipsis$B;
    }
    # Parameters for the nloptr from the ellipsis
    if(is.null(ellipsis$xtol_rel)){
        xtol_rel <- 1E-6;
    }
    else{
        xtol_rel <- ellipsis$xtol_rel;
    }
    if(is.null(ellipsis$algorithm)){
        # if(recursiveModel){
        # algorithm <- "NLOPT_LN_BOBYQA";
        # }
        # else{
        algorithm <- "NLOPT_LN_SBPLX";
        # }
    }
    else{
        algorithm <- ellipsis$algorithm;
    }
    if(is.null(ellipsis$maxtime)){
        maxtime <- -1;
    }
    else{
        maxtime <- ellipsis$maxtime;
    }
    if(is.null(ellipsis$xtol_abs)){
        xtol_abs <- 1E-8;
    }
    else{
        xtol_abs <- ellipsis$xtol_abs;
    }
    if(is.null(ellipsis$ftol_rel)){
        ftol_rel <- 1E-6;
    }
    else{
        ftol_rel <- ellipsis$ftol_rel;
    }
    if(is.null(ellipsis$ftol_abs)){
        ftol_abs <- 0;
    }
    else{
        ftol_abs <- ellipsis$ftol_abs;
    }
    if(is.null(ellipsis$print_level)){
        print_level <- 0;
    }
    else{
        print_level <- ellipsis$print_level;
    }
    print_level_hidden <- print_level;
    if(print_level==41){
        print_level[] <- 0;
    }
    if(is.null(ellipsis$stepSize)){
        stepSize <- .Machine$double.eps^(1/4);
    }
    else{
        stepSize <- ellipsis$stepSize;
    }

    if(is.null(ellipsis$cl)){
        responseName <- "y";
    }
    else{
        responseName <- formula(ellipsis$cl)[[2]];
    }

    occurrenceModel <- FALSE;
    obsZero <- 0;
    otU <- rep(TRUE,obsInsample);
    if(is.occurrence(occurrence)){
        occurrenceModel[] <- TRUE;
        otU[] <- actuals(occurrence)!=0;
        obsZero[] <- sum(!otU);
    }
    # Deal with subset
    if(is.null(subset)){
        subset <- c(1:obsInsample);
    }
    else if(is.logical(subset)){
        otU <- otU & subset;
    }
    else{
        otU <- which(otU) == subset;
    }

    if(any(residuals[otU]==0)){
        otU <- otU & residuals[subset]!=0;
    }

    # Prepare the data
    mf <- match.call(expand.dots = FALSE);
    m <- match(c("formula", "data", "subset", "na.action"), names(mf), 0L);
    mf <- mf[c(1L, m)];
    mf$drop.unused.levels <- TRUE;
    if(is.name(mf$data)){
        mf$data <- eval(mf$data);
    }
    if(!is.data.frame(mf$data)){
        mf$data <- data.frame(mf$data);
    }
    if(is.logical(mf$subset)){
        mf$subset <- which(mf$subset)
    }
    mf[[1L]] <- quote(stats::model.frame);
    dataWork <- eval(mf, parent.frame());
    dataTerms <- terms(dataWork);
    matrixXregScale <- model.matrix(dataWork,data=dataWork);
    # Extract number of variables and their names
    nVariables <- ncol(matrixXregScale);
    variablesNames <- colnames(matrixXregScale);

    fitterScale <- function(B, distribution){
        scale <- exp(matrixXregScale %*% B);
        # The scale of the Normal-based distributions is the variance, as in the ADAM monograph
        scale[] <- switch(distribution,
                          "dnorm"=,
                          "dlnorm"=,
                          "dbcnorm"=,
                          "dlogitnorm"=,
                          "dfnorm"=,
                          "drectnorm"=scale,
                          "dlogis"=sqrt(scale),
                          "dlaplace"=,
                          "dllaplace"=,
                          "dalaplace"=scale,
                          "ds"=,
                          "dls"=scale^2,
                          "dgnorm"=,
                          "dlgnorm"=scale^{1/other},
                          "dgamma"=,
                          "dinvgauss"=scale,
                          # "dgamma"=sqrt(scale)+1,
                          # This is based on polynomial from y = (x-1)^2/x
                          # "dinvgauss"=(scale+2+sqrt(scale^2+4*scale))/2,
                          scale);
        return(scale);
    }

    #### The function estimates parameters of scale model ####
    CFScale <- function(B){
        scaleAll <- fitterScale(B, distribution);
        # The densities are evaluated on the non-zeroes, with their scales
        scale <- scaleAll[otU];
        CFValue <- -sum(switch(distribution,
                               "dnorm" = dnorm(y[otU], mean=mu[otU], sd=sqrt(scale), log=TRUE),
                               "dlaplace" = dlaplace(y[otU], mu=mu[otU], scale=scale, log=TRUE),
                               "ds" = ds(y[otU], mu=mu[otU], scale=scale, log=TRUE),
                               "dgnorm" = dgnorm(y[otU], mu=mu[otU], scale=scale,
                                                 shape=other, log=TRUE),
                               "dlogis" = dlogis(y[otU], location=mu[otU], scale=scale, log=TRUE),
                               "dt" = dt(y[otU]-mu[otU], df=scale, log=TRUE),
                               "dalaplace" = dalaplace(y[otU], mu=mu[otU], scale=scale,
                                                       alpha=other, log=TRUE),
                               "dlnorm" = dlnorm(y[otU], meanlog=mu[otU], sdlog=sqrt(scale), log=TRUE),
                               "dllaplace" = dlaplace(log(y[otU]), mu=mu[otU],
                                                      scale=scale, log=TRUE)-log(y[otU]),
                               "dls" = ds(log(y[otU]), mu=mu[otU], scale=scale, log=TRUE)-log(y[otU]),
                               "dlgnorm" = dgnorm(log(y[otU]), mu=mu[otU], scale=scale,
                                                  shape=other, log=TRUE)-log(y[otU]),
                               "dbcnorm" = dbcnorm(y[otU], mu=mu[otU], sigma=sqrt(scale),
                                                   lambda=other, log=TRUE),
                               "dfnorm" = dfnorm(y[otU], mu=mu[otU], sigma=sqrt(scale), log=TRUE),
                               "drectnorm" = drectnorm(y[otU], mu=mu[otU], sigma=sqrt(scale), log=TRUE),
                               "dinvgauss" = dinvgauss(y[otU], mean=mu[otU],
                                                       dispersion=scale/mu[otU], log=TRUE),
                               "dgamma" = dgamma(y[otU], shape=1/scale,
                                                 scale=scale*mu[otU], log=TRUE),
                               "dchisq" = dchisq(y[otU], df=scale, ncp=mu[otU], log=TRUE),
                               "dpois" = dpois(y[otU], lambda=mu[otU], log=TRUE),
                               "dnbinom" = dnbinom(y[otU], mu=mu[otU], size=scale, log=TRUE),
                               "dlogitnorm" = dlogitnorm(y[otU], mu=mu[otU], sigma=sqrt(scale), log=TRUE)
                               # "dbeta" = dbeta(y[otU], shape1=mu[otU], shape2=scale, log=TRUE),
                               # "pnorm" = c(pnorm(mu[otU][ot], mean=0, sd=1, log.p=TRUE),
                               #             pnorm(mu[otU][!ot], mean=0, sd=1, lower.tail=FALSE, log.p=TRUE)),
                               # "plogis" = c(plogis(mu[otU][ot], location=0, scale=1, log.p=TRUE),
                               #              plogis(mu[otU][!ot], location=0, scale=1, lower.tail=FALSE, log.p=TRUE))
        ));

        # The differential entropy of the zeros, when the location model is recursive, as in alm()
        if(occurrenceModel && recursive){
            CFValue[] <- CFValue + sum(entropyZero(distribution,
                                                   switch(distribution,
                                                          "dnorm"=,"dlnorm"=,"dbcnorm"=,"dlogitnorm"=,
                                                          "dfnorm"=,"drectnorm"=sqrt(scaleAll[!otU]),
                                                          scaleAll[!otU]),
                                                   mu[!otU], other));
        }
        return(CFValue);
    }

    # Prepare parameters
    if(is.null(B)){
        if(any(distribution==c("dnorm","dlnorm","dbcnorm","dlogitnorm","dfnorm","drectnorm","dlogis"))){
            B <- .lm.fit(matrixXregScale[otU,,drop=FALSE],2*log(abs(residuals[otU])))$coefficients;
        }
        else if(any(distribution==c("dlaplace","dllaplace","dalaplace"))){
            B <- .lm.fit(matrixXregScale[otU,,drop=FALSE],log(abs(residuals[otU])))$coefficients;
        }
        else if(any(distribution==c("ds","dls"))){
            B <- .lm.fit(matrixXregScale[otU,,drop=FALSE],0.5*log(abs(residuals[otU])))$coefficients;
        }
        else if(any(distribution==c("dgnorm","dlgnorm"))){
            B <- .lm.fit(matrixXregScale[otU,,drop=FALSE],other+other*log(abs(residuals[otU])))$coefficients;
        }
        else if(distribution=="dgamma"){
            B <- .lm.fit(matrixXregScale[otU,,drop=FALSE],2*log(abs(residuals[otU]-1)))$coefficients;
        }
        else if(distribution=="dinvgauss"){
            B <- .lm.fit(matrixXregScale[otU,,drop=FALSE],log((residuals[otU]-1)^2/residuals[otU]))$coefficients;
        }
        # Other distributions: dt, dchisq, dnbinom, dpois, pnorm, plogis, dbeta
        else{
            B <- .lm.fit(matrixXregScale[otU,,drop=FALSE],log(abs(residuals[otU])))$coefficients;
        }
    }
    names(B) <- variablesNames;

    BLower <- rep(-Inf,nVariables);
    BUpper <- rep(Inf,nVariables);

    #### Define what to do with the maxeval ####
    if(is.null(ellipsis$maxeval)){
        maxeval <- length(B) * 40;
    }
    else{
        maxeval <- ellipsis$maxeval;
    }

    res <- nloptr(B, CFScale,
                  opts=list(algorithm=algorithm, xtol_rel=xtol_rel, maxeval=maxeval, print_level=print_level,
                            maxtime=maxtime, xtol_abs=xtol_abs, ftol_rel=ftol_rel, ftol_abs=ftol_abs),
                  lb=BLower, ub=BUpper);

    B[] <- res$solution;
    CFValue <- res$objective;

    if(print_level_hidden>0){
        print(res);
    }

    scale <- fitterScale(B, distribution);

    # Extract the actual values from the model
    errors <- switch(distribution,
                     "dnorm"=,
                     "dlnorm"=,
                     "dbcnorm"=,
                     "dlogitnorm"=,
                     "dfnorm"=,
                     "drectnorm"=residuals[subset]^2,
                     "dlogis"=,
                     "dlaplace"=,
                     "dllaplace"=,
                     "dalaplace"=abs(residuals[subset]),
                     "ds"=,
                     "dls"=residuals[subset]^2,
                     "dgnorm"=,
                     "dlgnorm"=abs(residuals[subset])^{1/other},
                     "dgamma"=,
                     "dinvgauss"=abs(residuals[subset]));

    #### Produce Fisher Information ####
    if(FI){
        # Only vcov is needed, no point in redoing the occurrenceModel
        occurrenceModel <- FALSE;
        FI <- hessian(CFScale, B, h=stepSize);

        if(any(is.nan(FI))){
            warning("Something went wrong and we failed to produce the covariance matrix of the parameters.\n",
                    "Obviously, it's not our fault. Probably Russians have hacked your computer...\n",
                    "Try a different distribution maybe?", call.=FALSE);
            FI <- diag(1e+100,nVariables);
        }
        dimnames(FI) <- list(variablesNames,variablesNames);
    }

    # Write the original residuals as the response variable
    interceptIsNeeded <- attr(dataTerms,"intercept")!=0;
    if(interceptIsNeeded){
        matrixXregScale[,1] <- errors;
    }
    else{
        matrixXregScale <- cbind(errors,matrixXregScale);
    }
    colnames(matrixXregScale)[1] <- "residuals";
    if(any(distribution==scaleNormalBased)){
        errors[] <- residuals[subset] / sqrt(scale);
    }
    else{
        errors[] <- residuals[subset] / scale;
    }

    # If formula does not have response variable, update it.
    # This is mainly needed for the proper plots and outputs
    if(length(formula)==2){
        cl$formula <- update.formula(formula,paste0(responseName,"~."))
    }

    # Form the scale object
    finalModel <- structure(list(formula=formula, coefficients=B, fitted=scale, residuals=errors,
                                 df.residual=obsInsample-nVariables, df=nVariables, call=cl, rank=nVariables,
                                 data=matrixXregScale, terms=dataTerms, logLik=-CFValue,
                                 occurrence=occurrence, subset=subset, other=ellipsis, B=B, FI=FI,
                                 distribution=distribution, other=other, loss="likelihood",
                                 timeElapsed=Sys.time()-startTime),
                            class=c("scale","alm","greybox"));
    return(finalModel);
}


#' Implant the scale model in the location model
#'
#' The function implants the scale model into the location model. It currently works
#' with \link[greybox]{alm} / \link[smooth]{adam} and \code{sm()} method.
#'
#' The function is needed in order to treat the scale of model correctly in the methods
#' like \code{forecast()}.
#'
#' @template author
#' @template keywords
#'
#' @param location Model estimated using either \code{alm} or \code{adam}.
#' @param scale The scale model, estimate with \code{sm} method.
#' @param ... Currently nothing. Implemented for flexibility.
#' @return The model of the same class as the location model, but with scale
#' from the estimated model via \code{sm()}. This is needed to produce
#' appropriate forecasts in case of scale model and to take  into account
#' the correct number of estimated parameters.
#' @seealso \link[greybox]{alm}, \link[smooth]{adam}, \link[greybox]{sm}
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+sqrt(exp(0.8+0.2*xreg[,1]))*rnorm(100,0,1),
#'               xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' # Estimate the location model
#' ourModel <- alm(y~.,xreg)
#' # Estimate the scale model
#' ourScale <- sm(ourModel,formula=~x1+x2)
#' # Implant scale into location
#' ourModel <- implant(ourModel, ourScale)
#'
#' @rdname implant
#' @export implant
implant <- function(location, scale, ...) UseMethod("implant")

#' @export
implant.alm <- function(location, scale, ...){
    if(!is.scale(scale)){
        stop("sm is not a scale model. Cannot procede.",
             call.=FALSE)
    }
    nParam <- nparam(location) + nparam(scale);
    obsInsample <- nobs(location);

    location$scale <- scale;
    location$logLik <- logLik(scale);
    location$df.residual <- obsInsample-nParam;
    location$df <- nParam;
    location$rank <- nParam;
    location$call$scale <- formula(scale);

    return(location);
}
