#' Bootstrap for parameters of models
#'
#' The function does the bootstrap for parameters of models and returns covariance matrix
#' together with the original bootstrapped data.
#'
#' The function applies the same model as in the provided object on a smaller sample in
#' order to get the estimates of parameters and capture the uncertainty about them. This is
#' a simple implementation of the case resampling, which assumes that the observations are
#' independent.
#'
#' @param object The model estimated using either lm, or alm, or glm.
#' @param nsim Number of iterations (simulations) to run.
#' @param size A non-negative integer giving the number of items to choose (the sample size),
#' passed to \link[base]{sample} function in R. If not provided and model contains ARIMA
#' components, this value will be selected at random on each iteration. This is only used for
#' \code{method="cr"}.
#' @param replace Should sampling be with replacement? Also, passed to \link[base]{sample}
#' function in R. Only used in \code{method="cr"}.
#' @param prob A vector of probability weights for obtaining the elements of the vector
#' being sampled. This is passed to the \link[base]{sample} as well. Only used with
#' \code{method="cr"}.
#' @param parallel Either a logical, specifying whether to do the calculations in parallel,
#' or the number, specifying the number of cores to use for the parallel calculation.
#' @param method Which bootstrap method to use. Currently two options are supported:
#' \code{"dsr"} - "Data Shape Replication, implemented in \link[greybox]{dsrboot};
#' \code{"cr"} - "Case Resampling", basic bootstrap that assumes that observations are
#' independent (not suitable for models with ARIMA elements).
#' @param ... Parameters passed to the \link[greybox]{dsrboot} function.
#'
#' @return Class "bootstrap" is returned, which contains:
#' \itemize{
#' \item vcov - the covariance matrix of parameters;
#' \item coefficients - the matrix with the bootstrapped coefficients.
#' \item nsim - number of runs done;
#' \item size - the sample size used in the bootstrap;
#' \item replace - whether the sampling was done with replacement;
#' \item prob - a vector of probability weights used in the process;
#' \item parallel - whether the calculations were done in parallel;
#' \item model - the name of the model used (the name of the function);
#' \item timeElapsed - the time that was spend on the calculations.
#' }
#'
#' @template author
#' @template keywords
#'
#' @seealso \code{\link[greybox]{alm}}
#'
#' @examples
#' # An example with ALM
#' ourModel <- alm(mpg~., mtcars, distribution="dlnorm", loss="HAM")
#' # A fast example with 10 iterations. Use at least 100 to get better results
#' coefbootstrap(ourModel, nsim=10)
#'
#' @rdname coefbootstrap
#' @export
coefbootstrap <- function(object, nsim=1000, size=floor(0.75*nobs(object)),
                          replace=FALSE, prob=NULL, parallel=FALSE,
                          method=c("cr","dsr"), ...) UseMethod("coefbootstrap")

#' @export
coefbootstrap.default <- function(object, nsim=1000, size=floor(0.75*nobs(object)),
                                  replace=FALSE, prob=NULL, parallel=FALSE,
                                  method=c("cr","dsr"), ...){

    startTime <- Sys.time();

    # Form the call for the function
    if(!is.null(object$call)){
        newCall <- object$call;
    }
    else{
        stop("In order for the function to work, the object should have the variable 'call' in it. Cannot proceed.",
             call.=FALSE);
    }

    method <- match.arg(method);

    if(is.numeric(parallel)){
        nCores <- parallel;
        parallel <- TRUE;
    }
    else if(is.logical(parallel) && parallel){
        # Detect number of cores for parallel calculations
        nCores <- min(parallel::detectCores() - 1, nsim);
    }

    # Define the variable, which is only needed for doParallel
    cluster <- NULL;

    # If they asked for parallel, make checks and try to do that
    if(parallel){
        if(!requireNamespace("foreach", quietly = TRUE)){
            stop("In order to run the function in parallel, 'foreach' package must be installed.", call. = FALSE);
        }
        if(!requireNamespace("parallel", quietly = TRUE)){
            stop("In order to run the function in parallel, 'parallel' package must be installed.", call. = FALSE);
        }

        # Check the system and choose the package to use
        if(Sys.info()['sysname']=="Windows"){
            if(requireNamespace("doParallel", quietly = TRUE)){
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need 'doParallel' package.",
                     call. = FALSE);
            }
        }
        else{
            if(requireNamespace("doMC", quietly = TRUE)){
                doMC::registerDoMC(nCores);
                cluster <- NULL;
            }
            else if(requireNamespace("doParallel", quietly = TRUE)){
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need either 'doMC' (prefered) or 'doParallel' packages.",
                     call. = FALSE);
            }
        }
    }

    # Coefficients of the model
    coefficientsOriginal <- coef(object);
    nVariables <- length(coefficientsOriginal);
    variablesNames <- names(coefficientsOriginal);
    interceptIsNeeded <- any(variablesNames=="(Intercept)");
    obsInsample <- nobs(object);

    # The matrix with coefficients
    coefBootstrap <- matrix(0, nsim, nVariables, dimnames=list(NULL, variablesNames));
    # Indices for the observations to use and the vector of subsets
    indices <- c(1:obsInsample);

    if(method=="cr"){
        if(!parallel){
            for(i in 1:nsim){
                newCall$subset <- sample(indices,size=size,replace=replace,prob=prob);
                testModel <- suppressWarnings(eval(newCall));
                coefBootstrap[i,names(coef(testModel))] <- coef(testModel);
            }
        }
        else{
            # We don't do rbind for security reasons - in order to deal with skipped variables
            coefBootstrapParallel <- foreach::`%dopar%`(foreach::foreach(i=1:nsim),{
                newCall$subset <- sample(indices,size=size,replace=replace,prob=prob);
                testModel <- eval(newCall);
                return(coef(testModel));
            })
            # Prepare the matrix with parameters
            for(i in 1:nsim){
                coefBootstrap[i,names(coefBootstrapParallel[[i]])] <- coefBootstrapParallel[[i]];
            }
        }
    }
    else{
        # Set all redundant variables to NA
        replace <- prob <- size <- NA;
        # Amend the data
        newCall$data <- model.matrix(object);
        newCall$data[,1] <- model.frame(object)[[1]];
        colnames(newCall$data)[1] <- all.vars(formula(object))[1];
        # Create a new dataset
        newData <- replicate(nsim, as.data.frame(newCall$data), simplify=FALSE);
        # Create formula for the used data
        newCall$formula <- as.formula(paste0(colnames(newCall$data)[1],"~."));
        # Bootstrap the data
        # dataBoot <- suppressWarnings(apply(as.data.frame(newCall$data), 2, dsrboot,
        #                                    nsim=nsim, intermittent=FALSE));
        dataBoot <- dsrboot(actuals(object), nsim=nsim, ...)$boot;
        # nLevels <- length(dataBoot);
        # Fill in the list of data
        for(i in 1:nsim){
            newData[[i]][,1] <- dataBoot[,i];
            # for(j in 1:nLevels){
            #     newData[[i]][,j] <- dataBoot[[j]]$boot[,i];
            # }
        }
        if(!parallel){
            for(i in 1:nsim){
                newCall$data[] <- newData[[i]];
                testModel <- suppressWarnings(eval(newCall));
                coefBootstrap[i,names(coef(testModel))] <- coef(testModel);
            }
        }
        else{
            # We don't do rbind for security reasons - in order to deal with skipped variables
            coefBootstrapParallel <- foreach::`%dopar%`(foreach::foreach(i=1:nsim),{
                newCall$data[] <- newData[[i]];
                testModel <- eval(newCall);
                return(coef(testModel));
            })
            # Prepare the matrix with parameters
            for(i in 1:nsim){
                coefBootstrap[i,names(coefBootstrapParallel[[i]])] <- coefBootstrapParallel[[i]];
            }
        }
    }
    # Get rid of NAs. They mean "zero"
    coefBootstrap[is.na(coefBootstrap)] <- 0;

    # Centre the coefficients for the calculation of the vcov
    coefvcov <- coefBootstrap - matrix(coefficientsOriginal, nsim, nVariables, byrow=TRUE);

    # Check if the clusters have been made
    if(!is.null(cluster)){
        parallel::stopCluster(cluster);
    }

    return(structure(list(vcov=(t(coefvcov) %*% coefvcov)/nsim,
                          coefficients=coefBootstrap, method=method,
                          nsim=nsim, size=size, replace=replace, prob=prob, parallel=parallel,
                          model=object$call[[1]], timeElapsed=Sys.time()-startTime),
                     class="bootstrap"));
}

#' @rdname coefbootstrap
#' @export
coefbootstrap.lm <- function(object, nsim=1000, size=floor(0.75*nobs(object)),
                             replace=FALSE, prob=NULL, parallel=FALSE,
                             method=c("cr","dsr"), ...){
    return(coefbootstrap.default(object, nsim, size, replace, prob, parallel, method, ...));
}

#' @rdname coefbootstrap
#' @export
coefbootstrap.alm <- function(object, nsim=1000, size=floor(0.75*nobs(object)),
                              replace=FALSE, prob=NULL, parallel=FALSE,
                              method=c("cr","dsr"), ...){

    startTime <- Sys.time();

    cl <- match.call();

    method <- match.arg(method);

    if(is.numeric(parallel)){
        nCores <- parallel;
        parallel <- TRUE;
    }
    else if(is.logical(parallel) && parallel){
        # Detect number of cores for parallel calculations
        nCores <- min(parallel::detectCores() - 1, nsim);
    }

    # If they asked for parallel, make checks and try to do that
    if(parallel){
        if(!requireNamespace("foreach", quietly = TRUE)){
            stop("In order to run the function in parallel, 'foreach' package must be installed.", call. = FALSE);
        }
        if(!requireNamespace("parallel", quietly = TRUE)){
            stop("In order to run the function in parallel, 'parallel' package must be installed.", call. = FALSE);
        }

        # Check the system and choose the package to use
        if(Sys.info()['sysname']=="Windows"){
            if(requireNamespace("doParallel", quietly = TRUE)){
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need 'doParallel' package.",
                     call. = FALSE);
            }
        }
        else{
            if(requireNamespace("doMC", quietly = TRUE)){
                doMC::registerDoMC(nCores);
                cluster <- NULL;
            }
            else if(requireNamespace("doParallel", quietly = TRUE)){
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need either 'doMC' (prefered) or 'doParallel' packages.",
                     call. = FALSE);
            }
        }

        # Ьake the list of the needed packages to pass to doParallel. Bloody Windows!
        callenvir <- globalenv();
        callpackages <- search();
        callpackages <- callpackages[c(-1,-length(callpackages))];
        callpackages <- callpackages[substring(callpackages,1,7)=="package"];
        callpackages <- substring(callpackages,9,nchar(callpackages));
        callpackages <- callpackages[callpackages!="timeDate"];
        callpackages <- callpackages[callpackages!="zoo"];
        callpackages <- callpackages[callpackages!="stats"];
        callpackages <- callpackages[callpackages!="graphics"];
        callpackages <- callpackages[callpackages!="grDevices"];
        callpackages <- callpackages[callpackages!="utils"];
        callpackages <- callpackages[callpackages!="datasets"];
        callpackages <- callpackages[callpackages!="methods"];
    }

    # Coefficients of the model
    coefficientsOriginal <- coef(object);
    nVariables <- length(coefficientsOriginal);
    variablesNames <- names(coefficientsOriginal);
    interceptIsNeeded <- any(variablesNames=="(Intercept)");
    variablesNamesMade <- make.names(variablesNames);
    if(interceptIsNeeded){
        variablesNamesMade[1] <- variablesNames[1];
    }
    obsInsample <- nobs(object);

    # The matrix with coefficients
    coefBootstrap <- matrix(0, nsim, nVariables, dimnames=list(NULL, variablesNames));
    # Indices for the observations to use and the vector of subsets
    indices <- c(1:obsInsample);

    # Form the call for alm
    newCall <- object$call;
    # Tuning for srm, to call alm() instead
    # if(is.srm(object)){
    #     newCall[[1]] <- as.name("alm");
    #     newCall$folder <- NULL;
    # }
    # This is based on the expanded data, so that we don't need to redo everything
    if(interceptIsNeeded){
        newCall$formula <- as.formula(paste0("`",colnames(object$data)[1],"`~."));
    }
    else{
        newCall$formula <- as.formula(paste0("`",colnames(object$data)[1],"`~.-1"));
    }
    newCall$data <- substitute(object$data);
    newCall$distribution <- object$distribution;
    if(object$loss=="custom"){
        newCall$loss <- object$lossFunction;
    }
    else{
        newCall$loss <- object$loss;
    }

    # Deal with ari components. Remove the columns with lagged variables
    arimaModel <- !is.null(object$other$polynomial);
    if(arimaModel){
        # The lags are the last columns of the data (under their transformed names)
        newCall$data <- object$data[,1:(ncol(object$data)-length(object$other$polynomial)),drop=FALSE];
    }

    # If this is ARIMA, and the size wasn't specified, make it changeable
    if(arimaModel && is.null(cl$size)){
        changeSize <- TRUE;
    }
    else{
        changeSize <- FALSE;
    }

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
    else if(object$distribution=="dfnorm"){
        newCall$sigma <- object$other$sigma;
    }
    else if(object$distribution=="dbcnorm"){
        newCall$lambdaBC <- object$other$lambdaBC;
    }
    else if(any(object$distribution==c("dgnorm","dlgnorm"))){
        newCall$shape <- object$other$shape;
    }
    newCall$occurrence <- object$occurrence;
    # Only pre-initialise the parameters if non-normal stuff is used
    if(all(object$distribution!=c("dnorm","dlnorm","dlogitnorm"))){
        newCall$B <- coef(object);
    }
    # If there is scale model, remove it
    if(is.scale(object$scale)){
        newCall$scale <- NULL;
    }

    # Function creates a random sample. Needed for dynamic models
    sampler <- function(indices,size,replace,prob,arimaModel=FALSE,changeSize=FALSE){
        if(arimaModel){
            if(changeSize){
                size[] <- floor(runif(1,nVariables,obsInsample))+1;
            }
            # This way we return the continuos sample, but with random starting point
            return(floor(runif(1,0,obsInsample-size))+c(1:size));
        }
        else{
            return(sample(indices,size=size,replace=replace,prob=prob));
        }
    }

    if(method=="cr"){
        if(!parallel){
            for(i in 1:nsim){
                newCall$subset <- sampler(indices,size,replace,prob,arimaModel,changeSize);
                testModel <- suppressWarnings(eval(newCall));
                coefBootstrap[i,variablesNamesMade %in% names(coef(testModel))] <- coef(testModel);
            }
        }
        else{
            # We don't do rbind for security reasons - in order to deal with skipped variables
            coefBootstrapParallel <- foreach::`%dopar%`(foreach::foreach(i=1:nsim,
                                                                         .packages=callpackages,
                                                                         .export=ls(envir=callenvir)),{
                newCall$subset <- sampler(indices,size,replace,prob,arimaModel,changeSize);
                testModel <- eval(newCall);
                return(coef(testModel));
            })
            # Prepare the matrix with parameters
            for(i in 1:nsim){
                coefBootstrap[i,variablesNamesMade %in% names(coefBootstrapParallel[[i]])] <- coefBootstrapParallel[[i]];
            }
        }
    }
    else{
        # Set all redundant variables to NA
        replace <- prob <- size <- NA;
        # Create a new dataset
        newData <- replicate(nsim, object$data, simplify=FALSE);
        newCall$data <- object$data;
        newCall$formula <- as.formula(paste0(colnames(object$data)[1],"~."));
        # Bootstrap the data
        # dataBoot <- suppressWarnings(apply(newCall$data, 2, dsrboot,
        #                                    nsim=nsim, intermittent=FALSE));
        dataBoot <- dsrboot(actuals(object), nsim=nsim, ...)$boot;
        # nLevels <- length(dataBoot);
        # Fill in the list of data
        for(i in 1:nsim){
            newData[[i]][,1] <- dataBoot[,i];
            # for(j in 1:nLevels){
            #     newData[[i]][,j] <- dataBoot[[j]]$boot[,i];
            # }
        }
        if(!parallel){
            for(i in 1:nsim){
                newCall$data[] <- newData[[i]];
                testModel <- suppressWarnings(eval(newCall));
                coefBootstrap[i,names(coef(testModel))] <- coef(testModel);
            }
        }
        else{
            # We don't do rbind for security reasons - in order to deal with skipped variables
            coefBootstrapParallel <- foreach::`%dopar%`(foreach::foreach(i=1:nsim,
                                                                         .packages=callpackages,
                                                                         .export=ls(envir=callenvir)),{
                newCall$data[] <- newData[[i]];
                testModel <- eval(newCall);
                return(coef(testModel));
            })
            # Prepare the matrix with parameters
            for(i in 1:nsim){
                coefBootstrap[i,names(coefBootstrapParallel[[i]])] <- coefBootstrapParallel[[i]];
            }
        }
    }

    # Get rid of NAs. They mean "zero"
    coefBootstrap[is.na(coefBootstrap)] <- 0;

    # Rename the variables to the originals
    colnames(coefBootstrap) <- names(coefficientsOriginal);

    # Centre the coefficients for the calculation of the vcov
    coefvcov <- coefBootstrap - matrix(colMeans(coefBootstrap), nsim, nVariables, byrow=TRUE);

    return(structure(list(vcov=(t(coefvcov) %*% coefvcov)/(nsim-1),
                          coefficients=coefBootstrap, method=method,
                          nsim=nsim, size=size, replace=replace, prob=prob, parallel=parallel,
                          model=object$call[[1]], timeElapsed=Sys.time()-startTime),
                     class="bootstrap"));
}


#' @export
print.bootstrap <- function(x, ...){
    cat(paste0("Bootstrap for the ", x$model, " model with nsim=",x$nsim," using ", x$method, " method\n"))
    cat(paste0("Time elapsed: ",round(as.numeric(x$timeElapsed,units="secs"),2)," seconds\n"));
}
