#' Multiple correlation
#'
#' Function calculates multiple correlation between y and x, constructing a linear
#' regression model
#'
#' This is based on the linear regression model with the set of variables in x. The
#' returned value is just a coefficient of multiple correlation from regression,
#' the F-statistics of the model (thus testing the null hypothesis that all the
#' parameters are equal to zero), the associated p-value and the degrees of freedom.
#'
#' See details in the vignette "Marketing analytics with greybox":
#' \code{vignette("maUsingGreybox","greybox")}
#'
#' @template author
#' @keywords htest
#'
#' @param x Either data.frame or a matrix
#' @param y The numerical variable.
#' @param use What observations to use. See \link[stats]{cor} function for details.
#' The only option that is not available here is \code{"pairwise.complete.obs"}.
#'
#' @return The following list of values is returned:
#' \itemize{
#' \item value - The value of the coefficient;
#' \item statistic - The value of F-statistics associated with the parameter;
#' \item p.value - The p-value of F-statistics associated with the parameter;
#' \item df.residual - The number of degrees of freedom for the residuals;
#' \item df - The number of degrees of freedom for the data.
#' }
#'
#' @seealso \code{\link[base]{table}, \link[greybox]{tableplot}, \link[greybox]{spread},
#' \link[greybox]{cramer}, \link[greybox]{association}}
#'
#' @examples
#'
#' mcor(mtcars$am, mtcars$mpg)
#'
#' @importFrom stats cor.test pf na.omit
#' @export mcor
mcor <- function(x, y, use=c("na.or.complete","complete.obs","everything","all.obs")){

    use <- match.arg(use);
    # everything - returns NA if NA
    # all.obs - returns error if NA
    # complete.obs - NAs are removed, returns an error if nothing is left
    # na.or.complete - NAs are removed, returns NA if nothing is left

    # Function returns values or NAs or error
    returner <- function(errorType=c(0,1,2)){
        if(errorType==0){
            return(structure(list(value=value,statistic=statistic,df=dfReg,df.residual=dfResid,
                                  p.value=pf(statistic,dfReg,dfResid,lower.tail=FALSE)),class="mcor"));
        }
        else if(errorType==1){
            return(structure(list(value=NA,statistic=NA,df=NA,df.residual=NA,
                                  p.value=NA),class="mcor"));
        }
        else{
            stop("Missing observations in mcor", call.=FALSE);
        }
    }

    # Check the presence of NAs
    obsNAy <- any(is.na(y));
    obsNAx <- any(is.na(x));
    # if(any(obsNAx) | any(obsNAy)){
    if(obsNAx || obsNAy){
        if(use=="everything"){
            return(returner(1));
        }
        else if(use=="all.obs"){
            returner(2);
        }
        else if(any(use==c("na.or.complete","complete.obs"))){
            # x <- x[!obsNAx & !obsNAy,];
            # y <- y[!obsNAx & !obsNAy,];
            dataMatrix <- na.omit(data.frame(y=y,x=x));
            y <- dataMatrix[[1]];
            # Keep the type of x: a matrix, a data.frame or a vector
            if(is.matrix(x)){
                x <- as.matrix(dataMatrix[,-1]);
            }
            else if(is.data.frame(x)){
                x <- dataMatrix[,-1,drop=FALSE];
            }
            else{
                x <- dataMatrix[[2]];
            }

            # If there are less than 2 obs left...
            if(nrow(dataMatrix)<2){
                if(use=="na.or.complete"){
                    return(returner(1));
                }
                else{
                    returner(2);
                }
            }
        }
    }

    # If it is a factor, just use it
    if(is.factor(x)){
        namesX <- colnames(x);
        # Remove the bloody ordering
        if(is.ordered(x)){
            x <- factor(x,ordered=FALSE,levels=levels(x));
        }
    }
    # If it is a matrix, transform into data.frame and create factors
    else if(is.matrix(x)){
        namesX <- colnames(x);
        # If there is no column with the intercept, then create one
        if(!all(x[,1]==1)){
            x <- cbind(1,x);
        }
    }
    # If it is a vector, create a data.frame and potentially create a factor
    else if(is.vector(x)){
        namesX <- deparse(substitute(x));
        if(length(unique(x))<=10 & length(x)>10){
            x <- factor(x);
        }
        else{
            x <- cbind(1,x);
        }
    }
    # Otherwise (is.data.frame) just record the names
    else{
        namesX <- colnames(x);
    }

    if(is.factor(x)){
        x <- model.matrix(~x);
    }
    else if(is.data.frame(x)){
        x <- model.matrix(~., data=x);
    }

    if(!is.numeric(y)){
        if(is.factor(y)){
            warning("The y variable should be numeric in order for this to work! The reported value might be meaningless",call.=FALSE);
            if(is.ordered(y)){
                y <- factor(y,ordered=FALSE,levels=levels(y));
            }
            y <- model.matrix(~y-1);
        }
        else if(!is.matrix(y)){
            y <- as.matrix(y)
        }
    }
    else if(is.data.frame(y)){
        y <- model.matrix(~y-1);
    }
    else{
        y <- as.matrix(y);
    }
    obs <- length(y);

    lmFit <- .lm.fit(x,y);

    # Use abs() in order to avoid the rounding issues in R
    value <- sqrt(abs(1 - sum(residuals(lmFit)^2) / sum((y-mean(y))^2)));
    dfResid <- length(residuals(lmFit))-lmFit$rank;
    dfReg <- lmFit$rank-1;
    statistic <- (value^2/dfReg)/((1-value^2)/dfResid);

    return(returner(0));
}
