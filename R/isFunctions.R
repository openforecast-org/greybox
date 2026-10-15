#' Greybox classes checkers
#'
#' Functions to check if an object is of the specified class
#'
#' The list of functions includes:
#' \itemize{
#' \item \code{is.greybox()} tests if the object was produced by a greybox function
#' (e.g. \link[greybox]{alm} / \link[greybox]{stepwise} / \link[greybox]{lmCombine}
#' / \link[greybox]{lmDynamic});
#' \item \code{is.alm()} tests if the object was produced by \code{alm()} function;
#' \item \code{is.occurrence()} tests if an occurrence part of the model was produced;
#' \item \code{is.calm()} tests if the object was produced by \code{lmCombine()}
#' function;
#' \item \code{is.greyboxD()} tests if the object was produced by \code{lmDynamic()}
#' function;
#' \item \code{is.rmc()} tests if the object was produced by \code{rmcb()} function;
#' \item \code{is.rollingOrigin()} tests if the object was produced by \code{ro()}
#' function;
#' \item \code{is.scale()} tests if the object is of the class "scale" (produced by
#' \link[greybox]{alm} or \link[greybox]{sm} in case of heteroscedastic model);
#' }
#'
#' @param x The object to check.
#' @return  \code{TRUE} if this is the specified class and \code{FALSE} otherwise.
#'
#' @template author
#' @keywords ts univar
#' @examples
#'
#' xreg <- cbind(rlaplace(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rlaplace(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' ourModel <- alm(y~x1+x2, xreg, distribution="dnorm")
#'
#' is.alm(ourModel)
#' is.greybox(ourModel)
#' is.calm(ourModel)
#' is.greyboxD(ourModel)
#'
#' @rdname isFunctions
#' @export
is.greybox <- function(x){
    return(inherits(x,"greybox"))
}

#' @rdname isFunctions
#' @export
is.alm <- function(x){
    return(inherits(x,"alm"))
}

#' @rdname isFunctions
#' @export
is.occurrence <- function(x){
    return(inherits(x,"occurrence"))
}

#' @rdname isFunctions
#' @export
is.calm <- function(x){
    return(inherits(x,"calm"))
}

#' @rdname isFunctions
#' @export
is.greyboxD <- function(x){
    return(inherits(x,"greyboxD"))
}

#' @rdname isFunctions
#' @export
is.rollingOrigin <- function(x){
    return(inherits(x,"rollingOrigin"))
}

#' @rdname isFunctions
#' @export
is.rmc <- function(x){
    return(inherits(x,"rmcb"))
}

#' @rdname isFunctions
#' @export
is.scale <- function(x){
    return(inherits(x,"scale"))
}
