#' Partial correlations
#'
#' Function calculates partial correlations between the provided variables
#'
#' The calculation is done based on multiple linear regressions. The function calculates
#' them for each pair of variables based on the residuals of linear models of those
#' variables from the other variables in the dataset.
#'
#' @template author
#' @keywords htest
#'
#' @param x Either data.frame or a matrix with numeric values.
#' @param y The numerical variable.
#' @param use What observations to use. See \link[stats]{cor} function for details.
#' The only option that is not available here is \code{"pairwise.complete.obs"}.
#' @param method Which method to use for the calculation of the partial correlations.
#' This can be either Pearson's, Spearman's or Kendall's coefficient. See \link[stats]{cor}
#' for details.
#'
#' @return The following list of values is returned:
#' \itemize{
#' \item{value - Matrix of the coefficients of partial correlations;}
#' \item{p.value - The p-values for the parameters;}
#' \item{method - The method used in the calculations.}
#' }
#'
#' @seealso \code{\link[greybox]{mcor}, \link[greybox]{cramer}, \link[greybox]{association}}
#'
#' @examples
#'
#' pcor(mtcars)
#'
#' @export pcor
pcor <- function(x, y=NULL, use=c("na.or.complete","complete.obs","everything","all.obs"),
                 method=c("pearson","spearman","kendall")){

    use <- match.arg(use,c("na.or.complete","complete.obs","everything","all.obs"));
    method <- match.arg(method,c("pearson","spearman","kendall"));
    # everything - returns NA if NA
    # all.obs - returns error if NA
    # complete.obs - NAs are removed, returns an error if nothing is left
    # na.or.complete - NAs are removed, returns NA if nothing is left

    # Function returns values or NAs or error
    # returner <- function(errorType=c(0,1,2)){
    #     if(errorType==0){
    #         return(structure(list(value=value,statistic=statistic,df=dfReg,df.residual=dfResid,
    #                               p.value=pf(statistic,dfReg,dfResid,lower.tail=FALSE)),class="pcor"));
    #     }
    #     else if(errorType==1){
    #         return(structure(list(value=NA,statistic=NA,df=NA,df.residual=NA,
    #                               p.value=NA),class="pcor"));
    #     }
    #     else{
    #         stop("Missing observations in pcor", call.=FALSE);
    #     }
    # }

    doWarning <- FALSE;
    if(is.matrix(x) | is.data.frame(x)){
        nVariablesX <- ncol(x);
        namesX <- colnames(x);
        if(any(sapply(x,is.factor))){
            x <- sapply(x,as.numeric);
            doWarning[] <- TRUE;
        }
        else if(is.data.frame(x)){
            x <- as.matrix(x);
        }
    }
    else if(is.factor(x)){
        nVariablesX <- 1;
        namesX <- deparse(substitute(x));
        x <- as.data.frame(x);
    }
    else{
        nVariablesX <- 1;
        namesX <- deparse(substitute(x));
        x <- as.matrix(x);
    }

    if(!is.null(y)){
        if(is.matrix(y) | is.data.frame(y)){
            nVariablesY <- ncol(y);
            namesY <- colnames(y);
            if(any(sapply(y,is.factor))){
                y <- sapply(y,as.numeric);
                doWarning[] <- TRUE;
            }
            else if(is.data.frame(y)){
                y <- as.matrix(y);
            }
        }
        else if(is.factor(y)){
            nVariablesY <- 1;
            namesY <- deparse(substitute(y));
            y <- as.data.frame(y);
        }
        else{
            nVariablesY <- 1;
            namesY <- deparse(substitute(y));
            y <- as.matrix(y);
        }

        data <- cbind(x,y);
        nVariables <- nVariablesX+nVariablesY;
        namesData <- c(namesX, namesY);
    }
    else{
        if(nVariablesX>1){
            data <- x;
            nVariables <- nVariablesX;
            namesData <- namesX;
        }
        else{
            return(structure(list(value=1, p.value=1, method=method),class="pcor"));
        }
    }

    if(doWarning){
        warning(paste0("Some of the variables are in categorical scales. ",
                       "Using partial correlations might be meaningless!"),
                call.=FALSE);
    }

    # Remove the incomplete observations, as cor() does
    obsNA <- apply(is.na(data),1,any);
    if(any(obsNA)){
        if(use=="all.obs"){
            stop("Missing observations in pcor", call.=FALSE);
        }
        else if(use!="everything"){
            data <- data[!obsNA,,drop=FALSE];
            x <- x[!obsNA,,drop=FALSE];
            if(!is.null(y)){
                y <- y[!obsNA,,drop=FALSE];
            }
        }
    }

    if(is.null(y)){
        matrixAssociation <- matrix(1,nVariables,nVariables, dimnames=list(namesData,namesData));
        matrixPValues <- matrix(0,nVariables,nVariables, dimnames=list(namesData,namesData));

        for(i in 1:nVariables){
            for(j in 1:nVariables){
                if(i>=j){
                    next;
                }

                model1 <- .lm.fit(data[,-c(i,j),drop=FALSE],data[,i,drop=FALSE]);
                model2 <- .lm.fit(data[,-c(i,j),drop=FALSE],data[,j,drop=FALSE]);
                corOutput <- suppressWarnings(cor.test(residuals(model1),residuals(model2),method=method));
                matrixAssociation[i,j] <- corOutput$estimate;
                matrixPValues[i,j] <- corOutput$p.value;
            }
        }

        matrixAssociation[lower.tri(matrixAssociation)] <- t(matrixAssociation)[lower.tri(matrixAssociation)];
        matrixPValues[lower.tri(matrixPValues)] <- t(matrixPValues)[lower.tri(matrixPValues)];
    }
    else{
        matrixAssociation <- matrix(1,nVariablesY,nVariablesX, dimnames=list(namesY,namesX));
        matrixPValues <- matrix(0,nVariablesY,nVariablesX, dimnames=list(namesY,namesX));
        for(j in 1:nVariablesY){
            for(i in 1:nVariablesX){
                model1 <- .lm.fit(cbind(x[,-i,drop=FALSE],y[,-j,drop=FALSE]),y[,j,drop=FALSE]);
                model2 <- .lm.fit(cbind(x[,-i,drop=FALSE],y[,-j,drop=FALSE]),x[,i,drop=FALSE]);
                corOutput <- suppressWarnings(cor.test(residuals(model1),residuals(model2),method=method));
                matrixAssociation[j,i] <- corOutput$estimate;
                matrixPValues[j,i] <- corOutput$p.value;
            }
        }
    }

    return(structure(list(value=matrixAssociation, p.value=matrixPValues, method=method),
                     class="pcor"));
}
