#' Exogenous variables transformer
#'
#' Function transforms each variable in the provided matrix or vector,
#' producing non-linear values, depending on the selected pool of functions.
#'
#' This function could be useful when you want to automatically select the
#' necessary transformations of the variables. This can be used together
#' with \code{xregDo="select"} in \link[smooth]{es}, \link[smooth]{ces},
#' \link[smooth]{gum} and \link[smooth]{ssarima}. However, this might be
#' dangerous, as it might lead to the overfitting the data. So be reasonable
#' when you produce the transformed variables.
#'
#' @param xreg Vector / matrix / data.frame, containing variables that need
#' to be expanded. In case of vector / matrix it is recommended to provide
#' \code{ts} object, so the frequency of the data is taken into account.
#' @param functions Vector of names for functions used.
#' @param silent If \code{silent=FALSE}, then the progress is printed out.
#' Otherwise the function won't print anything in the console.
#'
#' @return \code{ts} matrix with the transformed and the original variables
#' is returned.
#'
#' @template author
#' @template keywords
#'
#' @seealso \code{\link[smooth]{es}, \link[greybox]{stepwise},
#' \link[greybox]{xregExpander}}
#'
#' @examples
#' # Create matrix of two variables and expand it
#' x <- cbind(rnorm(100,100,1),rnorm(100,50,3))
#' xregTransformer(x)
#'
#' @export
xregTransformer <- function(xreg, functions=c("log", "exp", "inv", "sqrt", "square"), silent=TRUE){

    # Check and prepare functions
    if(any(!(functions %in% c("log", "exp", "inv", "sqrt", "square")))){
        warning("An unknown function type specified. We will drop it from the list", call.=FALSE);
        functions <- functions[(functions %in% c("log", "exp", "inv", "sqrt", "square"))];
    }

    if(length(functions)==0){
        stop("functions parameter does not contain any valid function name. Please provide something from the list.",
             call.=FALSE);
    }

    if(!silent){
        cat("Preparing matrices...    ");
    }

    functions <- unique(functions);

    nFunctions <- length(functions);

    # Check and prepare the data
    if(is.data.frame(xreg)){
        xreg <- as.matrix(xreg);
    }

    if(!is.matrix(xreg) & (is.vector(xreg) | is.ts(xreg))){
        # The names of a vector are those of its elements, not of the variable
        xregNames <- "x";
        xreg <- ts(matrix(xreg),start=start(xreg),frequency=frequency(xreg));
        colnames(xreg) <- xregNames;
    }

    if(is.matrix(xreg)){
        xregStart <- start(xreg);
        xregFrequency <- frequency(xreg);
        xregNames <- colnames(xreg);
        if(is.null(xregNames)){
            xregNames <- paste0("x",1:ncol(xreg));
        }
        obs <- nrow(xreg);
        nExovars <- ncol(xreg);
        xregNew <- matrix(NA,obs,(nFunctions+1)*nExovars);
        xregNew <- ts(xregNew,start=xregStart,frequency=xregFrequency);

        for(i in 1:nExovars){
            if(!silent){
                if(i==1){
                    cat("\b");
                }
                cat(paste0(rep("\b",nchar(round((i-1)/nExovars,2)*100)+1),collapse=""));
                cat(paste0(round(i/nExovars,2)*100,"%"));
            }
            xregNew[,(i-1)*(nFunctions+1)+1] <- xreg[,i];
            for(j in 1:nFunctions){
                if(functions[j]=="log"){
                    xregNew[,(i-1)*(nFunctions+1)+j+1] <- log(xreg[,i]);
                }
                if(functions[j]=="exp"){
                    xregNew[,(i-1)*(nFunctions+1)+j+1] <- exp(xreg[,i]);
                }
                if(functions[j]=="inv"){
                    xregNew[,(i-1)*(nFunctions+1)+j+1] <- 1/xreg[,i];
                }
                if(functions[j]=="sqrt"){
                    xregNew[,(i-1)*(nFunctions+1)+j+1] <- sqrt(xreg[,i]);
                }
                if(functions[j]=="square"){
                    xregNew[,(i-1)*(nFunctions+1)+j+1] <- xreg[,i]^2;
                }
            }
            colnames(xregNew)[((i-1)*(nFunctions+1)+1):(i*(nFunctions+1))] <- c(xregNames[i],paste(functions,xregNames[i],sep="_"));
        }
        if(!silent){
            cat(paste0(rep("\b",4),collapse=""));
            cat(" Done! \n");
        }
    }
    return(xregNew);
}
