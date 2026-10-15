#' Calculate Cramer's V for categorical variables
#'
#' Function calculates Cramer's V for two categorical variables based on the table
#' function
#'
#' The function calculates Cramer's V and also returns the associated statistics from
#' Chi-Squared test with the null hypothesis of independence of the two variables.
#'
#' See details in the vignette "Marketing analytics with greybox":
#' \code{vignette("maUsingGreybox","greybox")}
#'
#' @template author
#' @keywords htest
#'
#' @param x First categorical variable.
#' @param y Second categorical variable.
#' @param use What observations to use. See \link[stats]{cor} function for details.
#' The only option that is not available here is \code{"pairwise.complete.obs"}.
#' @param unbiased Determines whether to calculate the biased version of Cramer's V or
#' the one with the small sample correction.
#'
#' @return The following list of values is returned:
#' \itemize{
#' \item value - The value of Cramer's V;
#' \item statistic - The value of Chi squared statistic associated with the Cramer's V;
#' \item p.value - The p-value of Chi squared test associated with the Cramer's V;
#' \item df - The number of degrees of freedom from the test.
#' }
#'
#' @seealso \code{\link[base]{table}, \link[greybox]{tableplot}, \link[greybox]{spread},
#' \link[greybox]{mcor}, \link[greybox]{association}}
#'
#' @references
#' Wicher Bergsma (2013), A bias-correction for Cramér's V and Tschuprow's T. Journal
#' of the Korean Statistical Society, 42, pp. 323-328. \doi{10.1016/j.jkss.2012.10.002}.
#'
#' @examples
#'
#' cramer(mtcars$am, mtcars$gear)
#'
#' @importFrom stats chisq.test
#' @export cramer
cramer <- function(x, y, use=c("na.or.complete","complete.obs","everything","all.obs"),
                   unbiased=TRUE){

    use <- match.arg(use);

    # Function returns values or NAs or error
    returner <- function(errorType=c(0,1,2)){
        if(errorType==0){
            return(structure(list(value=cramerValue,statistic=chiTest$statistic,
                                  p.value=chiTest$p.value,df=chiTest$parameter),class="cramer"));
        }
        else if(errorType==1){
            return(structure(list(value=NA,statistic=NA,
                                  p.value=NA,df=NA),class="cramer"));
        }
        else{
            stop("Missing observations in cramer", call.=FALSE);
        }
    }

    # Check, whether x and y ar categorical or at least numerical with only 10 levels
    if(is.numeric(x)){
        if(length(unique(x))>10){
            warning("It seems that x is numeric, not categorical. Other measures of association might be more informative.",
                    call.=FALSE);
        }
    }
    if(is.numeric(y)){
        if(length(unique(y))>10){
            warning("It seems that y is numeric, not categorical. Other measures of association might be more informative.",
                    call.=FALSE);
        }
    }

    # Check the presence of NAs
    obsNAx <- is.na(x);
    obsNAy <- is.na(y);
    if(any(obsNAx) | any(obsNAy)){
        if(use=="everything"){
            return(returner(1));
        }
        else if(use=="all.obs"){
            returner(2);
        }
        else if(any(use==c("na.or.complete","complete.obs"))){
            x <- x[!obsNAx & !obsNAy];
            y <- y[!obsNAx & !obsNAy];
            if(length(x)<2){
                if(use=="na.or.complete"){
                    return(returner(1));
                }
                else{
                    returner(2);
                }
            }
        }
    }

    dataTable <- table(x,y);
    # Remove zero columns and zero rows
    dataTable <- dataTable[!apply(dataTable==0,1,all),!apply(dataTable==0,2,all)];
    chiTest <- suppressWarnings(chisq.test(dataTable));
    # Columns, rows and sample size
    n <- sum(dataTable);
    k <- ncol(dataTable);
    r <- nrow(dataTable);
    phi2 <- chiTest$statistic/n;
    # phi^2 calculation and update of columns and rows
    if(unbiased){
        phi2[] <- max(phi2 - prod(k-1,r-1)/(n-1),0);
        k[] <- k - (k-1)^2 / (n-1);
        r[] <- r - (r-1)^2 / (n-1);
    }
    cramerValue <- sqrt(phi2/(min(k-1,r-1)));

    return(returner(0));
}
