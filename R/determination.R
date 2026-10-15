#' Coefficients of determination
#'
#' Function produces coefficients of determination for the provided data
#'
#' The function calculates coefficients of determination (aka R^2)
#' between all the provided variables. The higher the coefficient for a
#' variable is, the higher the potential multicollinearity effect in the
#' model with the variable will be. Coefficients of determination are
#' connected directly to Variance Inflation Factor (VIF): VIF = 1 / (1 -
#' determination). Arguably it is easier to interpret, because it is
#' restricted with (0, 1) bounds. The multicollinearity can be
#' considered as serious, when determination > 0.9 (which corresponds
#' to VIF > 10).
#'
#' The method \code{determ} can be applied to wide variety of classes,
#' including \code{lm}, \code{glm} and \code{alm}.
#'
#' See details in the vignette "Marketing analytics with greybox":
#' \code{vignette("maUsingGreybox","greybox")}
#'
#' @template author
#' @keywords models
#'
#' @param xreg Data frame or a matrix, containing the exogenous variables.
#' @param bruteforce If \code{TRUE}, then all the variables will be used
#' for the regression construction (sink regression). If the number of
#' observations is smaller than the number of series, the function will
#' use \link[greybox]{stepwise} function and select only meaningful
#' variables. So the reported values will be based on stepwise regressions
#' for each variable.
#' @param ... Other values passed to cor function.
#' @param object The object, for which to calculate the coefficients of
#' determination.
#'
#' @return Function returns the vector of determination coefficients.
#'
#' @seealso \link[stats]{cor}, \link[greybox]{mcor}, \link[greybox]{stepwise}
#'
#' @examples
#'
#' ### Simple example
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("x1","x2","x3","Noise")
#' determination(xreg)
#'
#' @rdname determination
#' @aliases determ
#' @export determination
determination <- function(xreg, bruteforce=TRUE, ...){

    nVariables <- ncol(xreg);
    nSeries <- nrow(xreg);
    # Form the vector to return
    vectorCorrelationsMultiple <- rep(NA,nVariables);
    names(vectorCorrelationsMultiple) <- colnames(xreg);
    if(nSeries<=nVariables & bruteforce){
        # vectorCorrelationsMultiple[] <- 1;
        warning(paste0("The number of variables is larger than the number of observations. ",
                       "Sink regression cannot be constructed. Using stepwise."),
                call.=FALSE);
        bruteforce <- FALSE;
    }

    if(!bruteforce){
        determinationCalculator <- function(residuals, actuals){
            return(1 - sum(residuals^2) / sum((actuals-mean(actuals))^2));
        }
    }

    # If it is a bloody tibble or a data.table, remove the class, treat as data.frame
    if(any(class(xreg) %in% c("tbl","tbl_df","data.table"))){
        class(xreg) <- "data.frame";
    }

    # Calculate the multiple determinations
    if(bruteforce & nVariables>1){
        # If we have a non-matrix object and there are non-numeric values in it, use mcor
        if(!is.numeric(xreg) && !all(unlist(lapply(xreg,is.numeric)))){
            for(i in 1:nVariables){
                vectorCorrelationsMultiple[i] <- suppressWarnings(mcor(xreg[,-i,drop=FALSE],xreg[,i])$value^2);
            }
        }
        else{
            # Use the complete observations, unless the user asked otherwise
            if(is.null(list(...)$use)){
                corMatrix <- cor(xreg, use="na.or.complete", ...);
            }
            else{
                corMatrix <- cor(xreg, ...);
            }
            for(i in 1:nVariables){
                vectorCorrelationsMultiple[i] <- tryCatch(corMatrix[i,-i,drop=FALSE] %*%
                                                              chol2inv(chol(corMatrix[-i,-i,drop=FALSE])) %*%
                                                              # solve(corMatrix[-i,-i,drop=FALSE], diag(nVariables-1)) %*%
                                                              corMatrix[-i,i,drop=FALSE], error=function(e) 1);
            }
        }
    }
    else if(!bruteforce & nVariables>1){
        testXreg <- xreg;
        # This fix is needed in case the names of variables contain spaces
        colnames(testXreg) <- paste0("x",c(1:nVariables));
        testModel <- suppressWarnings(stepwise(testXreg));
        vectorCorrelationsMultiple[1] <- determinationCalculator(residuals(testModel),
                                                                 actuals(testModel));
        for(i in 2:nVariables){
            testXreg[] <- xreg;
            testXreg[,1] <- xreg[,i];
            testXreg[,i] <- xreg[,1];
            testModel <- suppressWarnings(stepwise(testXreg));
            vectorCorrelationsMultiple[i] <- determinationCalculator(residuals(testModel),
                                                                     actuals(testModel));
        }
    }
    else{
        vectorCorrelationsMultiple <- 0;
    }

    return(vectorCorrelationsMultiple);
}

#' @rdname determination
#' @export determ
determ <- function(object, ...) UseMethod("determ")

#' @export
determ.default <- function(object, ...){
    return(determination(object, ...));
}

#' @export
determ.lm <- function(object, ...){
    return(determination(object$model[,-1], ...));
}

#' @export
determ.alm <- function(object, ...){
    return(determination(object$data[,-1], ...));
}
