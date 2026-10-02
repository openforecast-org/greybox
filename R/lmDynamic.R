#' Combine regressions based on point information criteria
#'
#' Function combines parameters of linear regressions of the first variable
#' on all the other provided data using pAIC weights. This is an extension of the
#' \link[greybox]{lmCombine} function, which relies upon the idea that the combination
#' weights might change over time.
#'
#' The algorithm uses alm() to fit different models and then combines the models
#' based on the selected point IC. The combination weights are calculated for each
#' observation based on the point IC and then smoothed via LOWESS if the respective
#' parameter (\code{lowess}) is set to TRUE.
#'
#' Some details and examples of application are also given in the vignette
#' "Greybox": \code{vignette("greybox","greybox")}
#'
#' @template AICRef
#' @template author
#' @template keywords
#'
#' @param data Data frame containing dependent variable in the first column and
#' the others in the rest.
#' @param ic Information criterion to use.
#' @param bruteforce If \code{TRUE}, then all the possible models are generated
#' and combined. Otherwise the best model is found and then models around that
#' one are produced and then combined.
#' @param silent If \code{FALSE}, then nothing is silent, everything is printed
#' out. \code{TRUE} means that nothing is produced.
#' @param formula If provided, then the selection will be done from the listed
#' variables in the formula after all the necessary transformations.
#' @param subset an optional vector specifying a subset of observations to be
#' used in the fitting process.
#' @param distribution Distribution to pass to \code{alm()}. See \link[greybox]{alm}
#' for details.
#' @param parallel If \code{TRUE}, then the model fitting is done in parallel.
#' WARNING! Packages \code{foreach} and either \code{doMC} (Linux and Mac only)
#' or \code{doParallel} are needed in order to run the function in parallel.
#' @param lowess Logical defining, whether LOWESS should be used to smooth the
#' dynamic weights. By default it is \code{TRUE}.
#' @param f the smoother span for LOWESS. This gives the proportion of points in
#' the plot which influence the smooth at each value. Larger values give more
#' smoothness. If \code{NULL} the parameter will be optimised by minimising
#' \code{ic}.
#' @param ... Other parameters passed to \code{alm()}.
#'
#' @return Function returns \code{model} - the final model of the class
#' "greyboxD", which includes time varying parameters and dynamic importance
#' of each variable. The list of variables:
#' \itemize{
#' \item coefficients - the mean (over time) parameters of the model,
#' \item vcov - the combined covariance matrix of the model,
#' \item fitted - the fitted values,
#' \item residuals - the residuals of the model,
#' \item distribution - the distribution used in the estimation,
#' \item logLik - the mean (over time) log-likelihood of the model,
#' \item IC - dynamic values of the information criterion (pIC),
#' \item ICType - the type of information criterion used,
#' \item df.residual - mean number of degrees of freedom of the residuals of
#' the model,
#' \item df - mean number of degrees of freedom of the model,
#' \item importance - dynamic importance of the parameters,
#' \item call - call used in the function,
#' \item rank - rank of the combined model,
#' \item data - the data used in the model,
#' \item mu - the location value of the distribution,
#' \item scale - the scale parameter if alm() was used,
#' \item coefficientsDynamic - table with parameters of the model, varying over
#' the time,
#' \item df.residualDynamic - dynamic df.residual,
#' \item dfDynamic - dynamic df,
#' \item weights - the dynamic weights for each model under consideration,
#' \item timeElapsed - the time elapsed for the estimation of the model.
#' }
#'
#' @seealso \code{\link[greybox]{stepwise}, \link[greybox]{lmCombine}}
#'
#' @examples
#'
#' ### Simple example
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' inSample <- xreg[1:80,]
#' outSample <- xreg[-c(1:80),]
#' # Combine all the possible models
#' ourModel <- lmDynamic(inSample,bruteforce=TRUE)
#' predict(ourModel,outSample)
#' plot(predict(ourModel,outSample))
#'
#' @importFrom stats lowess
#' @export lmDynamic
lmDynamic <- function(data, ic=c("AICc","AIC","BIC","BICc"), bruteforce=FALSE, silent=TRUE,
                      formula=NULL, subset=NULL,
                      distribution=c("dnorm","dlaplace","ds","dgnorm","dlogis","dt","dalaplace",
                                     "dlnorm","dllaplace","dls","dlgnorm","dbcnorm","dfnorm",
                                     "dinvgauss","dgamma",
                                     "dpois","dnbinom",
                                     "dlogitnorm",
                                     "plogis","pnorm"),
                      parallel=FALSE, lowess=TRUE, f=NULL, ...){
    # Function combines linear regression models and produces the combined lm object.

    # Start measuring the time of calculations
    startTime <- Sys.time();

    cl <- match.call();
    cl$formula <- as.formula(paste0("`",colnames(data)[1],"`~ ."));

    ellipsis <- list(...);
    # Only likelihood is supported by the function
    loss <- "likelihood";
    if(is.null(ellipsis$loss)){
       ellipsis$loss <- loss;
    }

    # Use formula to form the data frame for further selection
    if(!is.null(formula) || !is.null(subset)){
        # If subset is provided, but not formula, generate one
        if(is.null(formula)){
            formula <- as.formula(paste0(colnames(data)[1],"~."));
        }

        # Do model.frame manipulations
        mf <- match.call(expand.dots = FALSE);
        mf <- mf[c(1L, match(c("formula", "data", "subset"), names(mf), 0L))];
        mf$drop.unused.levels <- TRUE;
        mf[[1L]] <- quote(stats::model.frame);

        if(!is.data.frame(data)){
            mf$data <- as.data.frame(data);
        }
        # Evaluate data frame to do transformations of variables
        data <- eval(mf, parent.frame());
        responseName <- colnames(data)[1];

        # Remove variables that have "-x" in the formula
        dataTerms <- terms(data);
        data <- data[,c(responseName, colnames(attr(dataTerms,"factors")))];
        ## We do it this way to avoid factors expansion into dummies at this stage
    }

    # Check, whether the response is numeric
    if(!is.numeric(data[[1]])){
        warning(paste0("The response variable is not numeric! ",
                       "We will make it numeric, but we cannot promise anything."),
                call.=FALSE);
        data[[1]] <- as.numeric(data[[1]]);
    }

    # If they asked for parallel, make checks and try to do that
    if(parallel){
        if(!requireNamespace("foreach", quietly = TRUE)){
            stop("In order to run the function in parallel, 'foreach' package must be installed.", call. = FALSE);
        }
        if(!requireNamespace("parallel", quietly = TRUE)){
            stop("In order to run the function in parallel, 'parallel' package must be installed.", call. = FALSE);
        }
        # Detect number of cores for parallel calculations
        nCores <- parallel::detectCores();

        # Check the system and choose the package to use
        if(Sys.info()['sysname']=="Windows"){
            if(requireNamespace("doParallel", quietly = TRUE)){
                cat(paste0("Setting up ", nCores, " clusters using 'doParallel'..."));
                cat("\n");
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
                cat(paste0("Setting up ", nCores, " clusters using 'doParallel'..."));
                cat("\n");
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need either 'doMC' (prefered) or 'doParallel' packages.",
                     call. = FALSE);
            }
        }
    }

    # The gsub is needed in order to remove accidental special characters
    # colnames(data) <- gsub("\`","",colnames(data),ignore.case=TRUE);
    colnames(data) <- make.names(colnames(data), unique=TRUE);

    # Define cases, when to use ALM
    distribution <- match.arg(distribution);
    if(distribution=="dnorm"){
        useALM <- FALSE;
    }
    else{
        useALM <- TRUE;
        if(any(distribution==c("plogis","pnorm"))){
            data[,1] <- (data[,1]!=0)*1;
            ot <- (data[,1]!=0)*1;
        }
    }

    # If the data is a vector, make it a matrix
    if(is.null(dim(data))){
        data <- as.matrix(data);
        colnames(data) <- as.character(cl$formula[[2]]);
    }

    # Check the data for NAs
    if(any(is.na(data))){
        rowsSelected <- apply(!is.na(data),1,all);
    }
    else{
        rowsSelected <- rep(TRUE,nrow(data));
    }

    # Check occurrence. If it is not "none" then use alm().
    # if(is.alm(occurrence)){
    #     useALM <- TRUE;
    #     rowsSelected <- rowsSelected & (data[,1]!=0);
    # }
    # else{
    #     occurrence <- occurrence[1];
    #     if(all(occurrence!=c("none","plogis","pnorm"))){
    #         warning(paste0("Sorry, but we don't know what to do with the occurrence '",occurrence,
    #                     "'. Switching to 'none'."), call.=FALSE);
    #         occurrence <- "none";
    #     }
    #
    #     if(any(occurrence==c("plogis","pnorm"))){
    #         useALM <- TRUE;
    #         rowsSelected <- rowsSelected | (data[,1]!=0);
    #
    #         occurrenceModel <- lmCombine(data, ic=ic, bruteforce=bruteforce, silent=silent,
    #                                      distribution=occurrence, parallel=parallel, ...);
    #         occurrenceModel$call <- cl;
    #     }
    # }

    # Define the function of IC
    ic <- match.arg(ic);
    # IC <- switch(ic,"AIC"=AIC,"BIC"=BIC,"BICc"=BICc,AICc);
    IC <- switch(ic,"AIC"=pAIC,"BIC"=pBIC,"BICc"=pBICc,pAICc);

    # Define what function to use in the estimation
    if(useALM){
        lmCall <- alm;
        listToCall <- list(distribution=distribution, fast=TRUE);
        listToCall <- c(listToCall,ellipsis);
    }
    else{
        lmCall <- function(formula, data){
            model <- list(xreg=as.matrix(cbind(1,data[,all.vars(formula)[-1]])));
            model <- c(model,.lm.fit(model$xreg, y));
            colnames(model$qr) <- c("(Intercept)",all.vars(formula)[-1]);
            return(structure(model,class=c("lmGreybox","lm")));
        }
        listToCall <- vector("list");
    }

    # Name of the response
    responseNameOriginal <- as.character(cl$formula[[2]]);
    responseName <-"y";
    y <- as.matrix(data[rowsSelected,responseNameOriginal]);
    colnames(y) <- responseName;

    # Check whether it is possible to do bruteforce
    if((ncol(data)>nrow(data)) & bruteforce){
        warning("You have more variables than observations. We have to be smart here. Switching to 'bruteforce=FALSE'.",
                call.=FALSE, immediate.=TRUE);
        bruteforce <- FALSE;
    }

    # Warning if we have a lot of models
    if((ncol(data)>14) & bruteforce){
        warning("You have more than 14 variables. The computation might take a lot of time.", call.=FALSE, immediate.=TRUE);
    }

    # If this is not bruteforce, then do stepwise first
    if(!bruteforce){
        if(!silent){
            cat("Selecting the best model...\n");
        }
        ourModel <- stepwise(data, ic=ic, distribution=distribution);
        # If the selected model does not contain variables
        if(length(coef(ourModel))==1){
            return(ourModel);
        }
    }

    # Modify the data and move to the list
    if(is.data.frame(data)){
        listToCall$data <- cbind(y,model.matrix(cl$formula, data=data[rowsSelected,])[,-1,drop=FALSE]);
    }
    else{
        listToCall$data <- cbind(y,as.data.frame(data[rowsSelected,-1,drop=FALSE]));
    }
    rm(data);

    # Other stuff for the things like alpha of ALaplace
    other <- vector("list",1);

    # Set alpha for the dalaplace. If not provided do alpha=0.5
    if(distribution=="dalaplace"){
        if(is.null(ellipsis$alpha)){
            alpha <- 0.5;
        }
        else{
            alpha <- ellipsis$alpha;
        }
        other$alpha <- listToCall$alpha <- alpha;
    }
    else if(any(distribution==c("dt","dchisq"))){
        if(is.null(ellipsis$nu)){
            nu <- NULL;
        }
        else{
            nu <- ellipsis$nu;
        }
        other$nu <- listToCall$nu <- nu;
    }
    else if(distribution=="dnbinom"){
        if(is.null(ellipsis$size)){
            size <- NULL;
        }
        else{
            size <- ellipsis$size;
        }
        other$size <- listToCall$size <- size;
    }
    else if(distribution=="dfnorm"){
        if(is.null(ellipsis$sigma)){
            sigma <- NULL;
        }
        else{
            sigma <- ellipsis$sigma;
        }
        other$sigma <- listToCall$sigma <- sigma;
    }
    else if(any(distribution==c("dgnorm","dlgnorm"))){
        if(is.null(ellipsis$shape)){
            shape <- NULL;
        }
        else{
            shape <- ellipsis$shape;
        }
        other$shape <- listToCall$shape <- shape;
    }
    else if(distribution=="dbcnorm"){
        if(is.null(ellipsis$lambdaBC)){
            lambdaBC <- NULL;
        }
        else{
            lambdaBC <- ellipsis$lambdaBC;
        }
        other$lambdaBC <- listToCall$lambdaBC <- lambdaBC;
    }

    # Observations in sample, assuming that the missing values are for the holdout
    obsInsample <- sum(!is.na(listToCall$data[,1]));
    # Names of the exogenous variables (without the intercept)
    exoNamesOriginal <- colnames(listToCall$data)[-1];
    exoNames <- paste0("x",c(1:length(exoNamesOriginal)));
    colnames(listToCall$data)[-1] <- exoNames;
    # Names of all the variables
    variablesNamesOriginal <- c("(Intercept)",exoNamesOriginal);
    variablesNames <- c("(Intercept)",exoNames);
    # Number of variables
    nVariables <- length(exoNames);

    # If nVariables is zero, return a simple model. This might be due to the stepwise returning intercept.
    if(nVariables==0){
        warning("No explanatory variables are selected / provided. Fitting the model with intercept only.",
                call.=FALSE, immediate.=TRUE);
        if(bruteforce){
            return(alm(as.formula(paste0("`",responseName,"`~1")),listToCall$data,distribution=distribution,...));
        }
        else{
            return(ourModel);
        }
    }

    # If this is a simple one, go through all the models
    if(bruteforce){
        # Number of combinations in the loop
        nCombinations <- 2^nVariables;
        # Matrix of all the combinations
        variablesBinary <- rep(1,nVariables);
        variablesCombinations <- matrix(NA,nCombinations,nVariables);
        colnames(variablesCombinations) <- exoNames;

        #Produce matrix with binaries for inclusion of variables in the loop
        variablesCombinations[,1] <- rep(c(0:1),times=prod(variablesBinary[-1]+1));

        if(nVariables>1){
            for(i in 2:nVariables){
                variablesCombinations[,i] <- rep(c(0:variablesBinary[i]),each=prod(variablesBinary[1:(i-1)]+1));
            }
        }

        # Vector of ICs
        pICs <- matrix(NA,obsInsample,nCombinations);
        # Matrix of parameters
        parameters <- matrix(0,nCombinations,nVariables+1);
        # Array of vcov of parameters
        vcovValues <- array(0,c(nCombinations,nVariables+1,nVariables+1));
        # Matrix of point likelihoods
        pointLiks <- matrix(NA,obsInsample,nCombinations);

        # Starting estimating the models with just a constant
        listToCall$formula <- as.formula(paste0(responseName,"~1"));
        ourModel <- do.call(lmCall,listToCall);
        pICs[,1] <- IC(ourModel);
        parameters[1,1] <- coef(ourModel)[1];
        vcovValues[1,1,1] <- vcov(ourModel);
        pointLiks[,1] <- pointLik(ourModel);
    }
    else{
        # Extract names of the used variables
        bestExoNamesOriginal <- names(coef(ourModel))[-1];
        bestExoNames <- exoNames[match(bestExoNamesOriginal,exoNamesOriginal)];
        variablesNames <- c("(Intercept)",exoNamesOriginal[exoNamesOriginal %in% bestExoNamesOriginal]);

        # If the number of variables is small, do bruteforce
        if(nparam(ourModel)<16){
            listToCall$data <- listToCall$data[,c(responseName,bestExoNames),drop=FALSE];
            colnames(listToCall$data) <- c(responseNameOriginal,bestExoNamesOriginal);
            ourModel <- lmDynamic(listToCall$data, ic=ic,
                                   bruteforce=TRUE, silent=silent, distribution=distribution, parallel=parallel, ...);
            ourModel$call <- cl;
            return(ourModel);
        }
        # If we have too many variables, use "stress" analysis
        else{
            nVariablesInModel <- length(bestExoNames);
            # Define number of combinations:
            # 1. Best model
            nCombinations <- 1;
            # 2. BM + {each variable not included};
            nCombinations <- nCombinations + nVariables - nVariablesInModel;
            # 3. BM - {each variable included};
            nCombinations <- nCombinations + nVariablesInModel;

            ## 4. BM + {each pair of variables not included};
            ## 5. BM - {each pair of variables included}.

            # Form combinations of variables
            variablesCombinations <- matrix(NA,nCombinations,nVariables);
            colnames(variablesCombinations) <- exoNames;
            # Fill in the first row with the variables from the best model
            for(j in 1:nVariables){
                variablesCombinations[,j] <- any(colnames(variablesCombinations)[j]==bestExoNames)*1;
            }

            # Fill in the first part with sequential inclusion
            for(i in 1:(nVariables - nVariablesInModel)){
                variablesCombinations[i+1,which(variablesCombinations[i+1,]==0)[i]] <- 1;
            }

            # Fill in the second part with sequential exclusion
            for(i in 1:nVariablesInModel){
                index <- i+(nVariables-nVariablesInModel)+1;
                variablesCombinations[index,which(variablesCombinations[index,]==1)[i]] <- 0;
            }

            # Vector of ICs
            pICs <- matrix(NA,obsInsample,nCombinations);
            # Matrix of parameters
            parameters <- matrix(0,nCombinations,nVariables+1);
            # Array of vcov of parameters
            vcovValues <- array(0,c(nCombinations,nVariables+1,nVariables+1));
            # Matrix of point likelihoods
            pointLiks <- matrix(NA,obsInsample,nCombinations);

            # Starting estimating the models with writing down the best one
            pICs[,1] <- IC(ourModel);
            bufferCoef <- coef(ourModel)[variablesNames];
            parameters[1,c(1,variablesCombinations[1,])==1] <- bufferCoef[!is.na(bufferCoef)];
            bufferCoef <- vcov(ourModel)[variablesNames,variablesNames];
            vcovValues[1,c(1,variablesCombinations[1,])==1,c(1,variablesCombinations[1,])==1] <- bufferCoef[!is.na(bufferCoef)];
            pointLiks[,1] <- pointLik(ourModel);
        }
    }

    if(any(distribution==c("dt","dchisq","dnbinom","dalaplace","dgnorm","dlgnorm","dbcnorm"))){
        otherParameters <- rep(NA, nCombinations);
        otherParameters[1] <- ourModel$other[[1]];
    }

    # Go for the loop of lm models
    if(parallel){
        if(!silent){
            cat("Estimation progress: ...");
        }
        forLoopReturns <- foreach::`%dopar%`(foreach::foreach(i=2:nCombinations),{
            listToCall$formula <- as.formula(paste0(responseName,"~",paste0(exoNames[variablesCombinations[i,]==1],collapse="+")));
            ourModel <- do.call(lmCall,listToCall);

            ICs <- IC(ourModel);
            parameters <- coef(ourModel);
            vcovValues <- vcov(ourModel);
            pointLiks <- pointLik(ourModel);

            if(any(distribution==c("dt","dchisq","dnbinom","dalaplace","dgnorm","dlgnorm","dbcnorm"))){
                otherParameters <- ourModel$other[[1]];
            }
            else{
                otherParameters <- NULL;
            }
            return(list(ICs=ICs,parameters=parameters,vcovValues=vcovValues,
                        pointLiks=pointLiks,otherParameters=otherParameters));
        });

        for(i in 2:nCombinations){
            pICs[,i] <- forLoopReturns[[i-1]]$ICs;
            parameters[i,c(1,variablesCombinations[i,])==1] <- forLoopReturns[[i-1]]$parameters;
            vcovValues[i,c(1,variablesCombinations[i,])==1,c(1,variablesCombinations[i,])==1] <- forLoopReturns[[i-1]]$vcovValues;
            pointLiks[,i] <- forLoopReturns[[i-1]]$pointLiks;

            if(any(distribution==c("dt","dchisq","dnbinom","dalaplace","dgnorm","dlgnorm","dbcnorm"))){
                otherParameters[i] <- forLoopReturns[[i-1]]$otherParameters;
            }
        }

        if(!is.null(cluster)){
            parallel::stopCluster(cluster);
        }
    }
    else{
        if(!silent){
            cat(paste0("Estimation progress: ", round(1/nCombinations,2)*100,"%"));
        }
        for(i in 2:nCombinations){
            if(!silent){
                cat(paste0(rep("\b",nchar(round((i-1)/nCombinations,2)*100)+1),collapse=""));
                cat(paste0(round(i/nCombinations,2)*100,"%"));
            }
            listToCall$formula <- as.formula(paste0(responseName,"~",paste0(exoNames[variablesCombinations[i,,drop=FALSE]==1],collapse="+")));
            ourModel <- do.call(lmCall,listToCall);

            pICs[,i] <- IC(ourModel);
            parameters[i,c(1,variablesCombinations[i,])==1] <- coef(ourModel);
            vcovValues[i,c(1,variablesCombinations[i,])==1,c(1,variablesCombinations[i,])==1] <- vcov(ourModel);
            pointLiks[,i] <- pointLik(ourModel);

            if(any(distribution==c("dt","dchisq","dnbinom","dalaplace","dgnorm","dlgnorm","dbcnorm"))){
                otherParameters[i] <- ourModel$other[[1]];
            }
        }
    }

    if(!silent){
        cat(" Done!\n");
    }

    # Calculate IC weights
    pICWeights <- pICs - apply(pICs,1,min);
    pICWeights <- exp(-0.5*pICWeights) / apply(exp(-0.5*pICWeights),1,sum)

    if(lowess){
            # Logit transform of weights
        pICWeightsLog <- log(pICWeights/(1-pICWeights));
        ICsMean <- apply(pICs,2,mean);
        # Optimis f if it is not provided
        if(is.null(f)){
            pICWeightsSmooth <- pICWeightsLog;
            fFinder <- function(f){
            # Smooth weights via LOWESS
                pICWeightsSmooth[] <- sapply(apply(pICWeightsLog, 2, lowess, f=f),"[[","y");

                # Inverse logit transform
                pICWeightsSmooth[] <- exp(pICWeightsSmooth)/(1+exp(pICWeightsSmooth));
                # Normalisation
                pICWeightsSmooth[] <- pICWeightsSmooth/apply(pICWeightsSmooth, 1, sum);

                # Dynamic weighted mean pAIC
                # This is a proxy for the true one based on logLik
                ICValue <- sum(apply(pICWeightsSmooth,2,mean) * ICsMean);
                return(ICValue);
            }

            fValue <- nloptr(0.9, fFinder, lb=0, ub=1,
                             opts=list(algorithm="NLOPT_LN_SBPLX", xtol_rel=1E-6,
                                       maxeval=100, print_level=0, xtol_abs=1E-8,
                                       ftol_rel=1E-4, ftol_abs=0));
            f <- fValue$solution;
        }

        # Smooth weights via LOWESS
        pICWeightsLogSmooth <- sapply(apply(pICWeightsLog, 2, lowess, f=f),"[[","y");

        # Inverse logit transform
        pICWeights[] <- exp(pICWeightsLogSmooth)/(1+exp(pICWeightsLogSmooth));
        # Normalisation
        pICWeights[] <- pICWeights/apply(pICWeights, 1, sum);
    }

    pICWeightsMean <- apply(pICWeights,2,mean);

    # Calculate weighted parameters
    parametersWeighted <- pICWeights %*% parameters;
    colnames(parametersWeighted) <- variablesNamesOriginal;

    parametersMean <- colMeans(parametersWeighted);
    names(parametersMean) <- variablesNamesOriginal;

    # From the matrix of exogenous variables without the response variable
    ourDataExo <- cbind(1,listToCall$data[,-1]);
    colnames(ourDataExo) <- variablesNamesOriginal;
    colnames(listToCall$data) <- variablesNamesOriginal;

    if(distribution=="dbcnorm"){
        # Function for the Box-Cox transform
        bcTransform <- function(y, lambdaBC){
            if(lambdaBC==0){
                return(log(y));
            }
            else{
                return((y^lambdaBC-1)/lambdaBC);
            }
        }

        # Function for the inverse Box-Cox transform
        bcTransformInv <- function(y, lambdaBC){
            if(lambdaBC==0){
                return(exp(y));
            }
            else{
                return((y*lambdaBC+1)^{1/lambdaBC});
            }
        }
    }

    # Calculate the mean based on the mean values of parameters
    mu <- switch(distribution,
                 "dchisq" = (as.matrix(ourDataExo) %*% parametersMean)^2,
                 "dinvgauss" =,
                 "dgamma" =,
                 "dpois" =,
                 "dnbinom" = exp(as.matrix(ourDataExo) %*% parametersMean),
                 "dnorm" =,
                 "dfnorm" =,
                 "dbcnorm"=,
                 "dlogitnorm"=,
                 "dlaplace" =,
                 "ds" =,
                 "dgnorm" =,
                 "dlogis" =,
                 "dt" =,
                 "dalaplace" =,
                 "dlnorm" =,
                 "dllaplace" =,
                 "dls" =,
                 "dlgnorm" =,
                 "pnorm" =,
                 "plogis" = as.matrix(ourDataExo) %*% parametersMean
    );

    scale <- switch(distribution,
                    "dnorm" = mean((y-mu)^2),
                    "dfnorm" = mean((y-mu)^2),
                    "dbcnorm" = mean((bcTransform(y,other)-mu)^2),
                    "dlogitnorm" = mean((log(y/(1-y))-mu)^2),
                    "dlaplace" = mean(abs(y-mu)),
                    "ds" = mean(sqrt(abs(y-mu))) / 2,
                    "dgnorm" = (otherParameters*mean(abs(y-mu)^otherParameters))^{1/otherParameters},
                    "dlogis" = sqrt(mean((y-mu)^2) * 3 / pi^2),
                    "dt" = max(2,2/(1-(mean((y-mu)^2))^{-1})),
                    "dalaplace" = mean((y-mu) * (alpha - (y<=mu)*1)),
                    "dlnorm" = mean((log(y)-mu)^2),
                    "dllaplace" = mean(abs(log(y)-mu)),
                    "dls" = mean(sqrt(abs(log(y)-mu))) / 2,
                    "dlgnorm" = (otherParameters*mean(abs(log(y)-mu)^otherParameters))^{1/otherParameters},
                    "dchisq" = pICWeightsMean %*% otherParameters,
                    "dinvgauss" = mean((y/mu-1)^2 / (y/mu)),
                    "dgamma" = mean((y/mu-1)^2),
                    "dnbinom" = pICWeightsMean %*% otherParameters,
                    "dpois" = mu,
                    "pnorm" = sqrt(mean(qnorm((y - pnorm(mu, 0, 1) + 1) / 2, 0, 1)^2)),
                    "plogis" = sqrt(mean(log((1 + y * (1 + exp(mu))) / (1 + exp(mu) * (2 - y) - y))^2)) # Here we use the proxy from Svetunkov et al. (2018)
    );

    yFitted <- switch(distribution,
                      "dfnorm" = sqrt(2/pi)*sqrt(scale)*exp(-mu^2/(2*scale))+mu*(1-2*pnorm(-mu/sqrt(scale))),
                      "dnorm" =,
                      "dlaplace" =,
                      "ds" =,
                      "dgnorm" =,
                      "dalaplace" =,
                      "dlogis" =,
                      "dt" =,
                      "dinvgauss" =,
                      "dgamma" =,
                      "dpois" =,
                      "dnbinom" = mu,
                      "dlogitnorm" = exp(mu)/(1+exp(mu)),
                      "dbcnorm" = bcTransformInv(mu,other),
                      "dchisq" = mu + df,
                      "dlnorm" =,
                      "dllaplace" =,
                      "dls" =,
                      "dlgnorm" = exp(mu),
                      "pnorm" = pnorm(mu, mean=0, sd=1),
                      "plogis" = plogis(mu, location=0, scale=1)
    );

    errors <- switch(distribution,
                     "dfnorm" =,
                     "dnorm" =,
                     "dlaplace" =,
                     "ds" =,
                     "dgnorm" =,
                     "dalaplace" =,
                     "dlogis" =,
                     "dt" =,
                     "dpois" =,
                     "dnbinom" = y - mu,
                     "dinvgauss" =,
                     "dgamma" = y / mu,
                     "dchisq" = sqrt(y) - sqrt(mu),
                     "dlnorm" =,
                     "dllaplace" =,
                     "dls" =,
                     "dlgnorm" = log(y) - mu,
                     "pnorm" = qnorm((y - pnorm(mu, 0, 1) + 1) / 2, 0, 1),
                     "plogis" = log((1 + y * (1 + exp(mu))) / (1 + exp(mu) * (2 - y) - y)) # Here we use the proxy from Svetunkov et al. (2018)
    );

    # Relative importance of variables
    importance <- cbind(1,pICWeights %*% variablesCombinations);
    colnames(importance) <- variablesNamesOriginal;

    # Some of the variables have partial inclusion, 1 stands for constant
    # This is the dynamic degrees of freedom
    df <- obsInsample - apply(importance,1,sum) - 1;

    # Calcualte logLik. This is approximate and is based on the IC weights
    logLikCombined <- logLik(ourModel);
    attr(logLikCombined,"df") <- mean(df);
    logLikCombined[1] <- colSums(pointLiks) %*% pICWeightsMean;

    # Dynamic weighted mean pAIC
    ICValue <- apply(pICWeights * pICs,1,sum);

    # Models vcov
    # Although we can make an array of vcvo over time, it is just too much...
    # So here we produce the mean vcov
    vcovCombined <- matrix(NA, nVariables+1, nVariables+1, dimnames=list(variablesNamesOriginal, variablesNamesOriginal));
    for(i in 1:(nVariables+1)){
        for(j in 1:(nVariables+1)){
            if(i<=j){
                vcovCombined[i,j] <- (pICWeightsMean^2 %*% (vcovValues[,i,j] +
                                                           (parameters[,i] - parametersMean[i]) *
                                                           (parameters[,j] - parametersMean[j])));
            }
            else{
                vcovCombined[i,j] <- vcovCombined[j,i];
            }
        }
    }
    colnames(pICWeights) <- paste0("Model",c(1:ncol(pICWeights)));

    if(any(is.nan(vcovCombined)) | any(is.infinite(vcovCombined))){
        warning("The standard errors of the parameters cannot be produced properly. It seems that we have overfitted the data.",
                call.=FALSE);
    }

    finalModel <- structure(list(coefficients=parametersMean, vcov=vcovCombined, fitted=as.vector(yFitted),
                                 residuals=as.vector(errors), distribution=distribution, logLik=logLikCombined, IC=ICValue,
                                 ICType=ic, df.residual=mean(df), df=sum(apply(importance,2,mean))+1, importance=importance,
                                 call=cl, rank=nVariables+1, data=listToCall$data, mu=mu, scale=scale,
                                 coefficientsDynamic=parametersWeighted, df.residualDynamic=df, dfDynamic=apply(importance,1,sum)+1,
                                 weights=pICWeights, other=other, f=f, loss=loss,
                                 timeElapsed=Sys.time()-startTime),
                            class=c("greyboxD","alm","greybox"));

    return(finalModel);
}
