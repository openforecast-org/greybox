#' Exogenous variables expander
#'
#' Function expands the provided matrix or vector of variables, producing
#' values with lags and leads specified by \code{lags} variable.
#'
#' This function could be handy when you want to check if lags and leads
#' of a variable influence the dependent variable. Can be used together
#' with \code{xregDo="select"} in \link[smooth]{adam}, \link[smooth]{es},
#' \link[smooth]{ces} and \link[smooth]{ssarima}. All the missing values
#' in the beginning and at the end of lagged series are substituted by
#' mean forecasts produced using \link[smooth]{adam}.
#'
#' @param xreg Vector / matrix / data.frame, containing variables that need
#' to be expanded. In case of vector / matrix it is recommended to provide
#' \code{ts} object, so the frequency of the data is taken into account.
#' @param lags Vector of lags / leads that we need to have. Negative values
#' mean lags, positive ones mean leads.
#' @param silent If \code{silent=FALSE}, then the progress is printed out.
#' Otherwise the function won't print anything in the console.
#' @param gaps Defines how to fill in the gaps in the data. \code{"NAs"} will
#' leave missing values, \code{"zero"} will substitute them by zeroes,
#' \code{"naive"} will use the last / the first actual value, while
#' \code{"extrapolate"} will use \link[smooth]{es} function from smooth package
#' (if present, otherwise - naive) in order to fill in values. Finally,
#' \code{"auto"} will let the function select between \code{"extrapolate"} and
#' \code{"naive"} depending on the length of series.
#'
#' @return \code{ts} matrix with the expanded variables is returned.
#'
#' @template author
#' @template keywords
#'
#' @seealso \code{\link[smooth]{es}, \link[greybox]{stepwise}}
#'
#' @examples
#' # Create matrix of two variables, make it ts object and expand it
#' x <- cbind(rnorm(100,100,1),rnorm(100,50,3))
#' x <- ts(x,frequency=12)
#' xregExpander(x)
#'
#' @export
xregExpander <- function(xreg, lags=c(-frequency(xreg):frequency(xreg)),
                         silent=TRUE, gaps=c("auto","NAs","zero","naive","extrapolate")){

    gaps <- match.arg(gaps);

    lagsOriginal <- lags;
    # Remove zero from lags
    lags <- lags[lags!=0]
    lagsLengthAll <- length(lags);
    if(lagsLengthAll==0){
        warning("You have not specified any leads or lags.",call.=FALSE);
        return(xreg);
    }
    # Form leads
    leads <- lags[lags>0];
    leadsLength <- length(leads);
    if(leadsLength!=0){
        maxLead <- max(leads);
    }
    else{
        maxLead <- 0;
    }

    if(!requireNamespace("smooth", quietly = TRUE)){
        message("In order to fully enjoy the power of xregExpander, please install smooth package.");
    }
    else{
        # The occurrence model of the binary variables: om(), as the Python greybox, or the older
        # oes() of the versions of smooth without it
        occurrenceModel <- if(exists("om", envir=asNamespace("smooth"))) smooth::om else smooth::oes;
    }

    # Form proper lags
    lags <- abs(lags[lags<0]);
    lagsLength <- length(lags);
    if(lagsLength!=0){
        maxLag <- max(lags);
    }
    else{
        maxLag <- 0;
    }

    if(!silent){
        cat("Preparing matrices...    ");
    }

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

        # If the auto for filling the gaps is selected
        if(gaps=="auto"){
            if(obs > 10000){
                gaps[] <- "naive";
            }
            extrapolate <- TRUE;
        }
        else if(gaps=="NAs"){
            extrapolate <- FALSE;
        }
        else{
            extrapolate <- TRUE;
        }

        nExovars <- ncol(xreg);
        xregNew <- matrix(NA,obs,(lagsLengthAll+1)*nExovars);
        xregNew <- ts(xregNew,start=xregStart,frequency=xregFrequency);

        for(i in 1:nExovars){
            if(!silent){
                if(i==1){
                    cat("\b");
                }
                cat(paste0(rep("\b",nchar(round((i-1)/nExovars,2)*100)+1),collapse=""));
                cat(paste0(round(i/nExovars,2)*100,"%"));
            }
            chosenColumn <- (lagsLengthAll+1)*(i-1);
            xregNew[,chosenColumn+1] <- xregData <- xreg[,i];
            xregCurrentName <- xregNames[i];
            colnames(xregNew)[(lagsLengthAll+1)*(i-1)+1] <- xregCurrentName;
            xregDataNew <- xregData;
            if(leadsLength!=0){
                if(extrapolate){
                    # Produce forecasts for leads
                    # If this is a binary variable, use iss function.
                    if(!requireNamespace("smooth", quietly = TRUE) | gaps=="naive"){
                        xregDataNew <- c(xregData,rep(xregData[obs],maxLead));
                    }
                    else if(gaps=="zero"){
                        xregDataNew <- c(xregData,rep(0,maxLead));
                    }
                    else{
                        if(all((xregData==0) | (xregData==1))){
                            xregModel <- suppressWarnings(occurrenceModel(xregData, model="MNN", h=maxLead, occurrence="i"));
                        }
                        else{
                            if(packageVersion("smooth")>="3.0.0"){
                                xregModel <- suppressWarnings(smooth::adam(xregData, h=maxLead, occurrence="a"));
                            }
                            else{
                                xregModel <- suppressWarnings(smooth::es(xregData, h=maxLead));
                            }
                        }
                        xregDataNew <- c(xregData,xregModel$forecast);
                    }
                }
                else{
                    xregDataNew <- c(xregData,rep(NA,maxLead));
                }
            }
            if(lagsLength!=0){
                if(extrapolate){
                    # Produce reversed forecasts for lags
                    if(!requireNamespace("smooth", quietly = TRUE) | gaps=="naive"){
                        xregDataNew <- c(rep(xregData[1],maxLag),xregDataNew);
                    }
                    else if(gaps=="zero"){
                        xregDataNew <- c(rep(0,maxLag),xregDataNew);
                    }
                    else{
                        if(leadsLength!=0){
                            # If this is a binary variable, use iss function.
                            if(all((xregData==0) | (xregData==1))){
                                xregModel <- suppressWarnings(occurrenceModel(rev(xregData), model=smooth::modelType(xregModel), occurrence="i",
                                                                          persistence=xregModel$persistence, h=maxLag));
                            }
                            else{
                                if(packageVersion("smooth")>="3.0.0"){
                                    xregModel <- suppressWarnings(smooth::adam(rev(xregData), model=smooth::modelType(xregModel),
                                                                               persistence=xregModel$persistence,
                                                                               occurrence=xregModel$occurrence,
                                                                               oesmodel=smooth::modelType(xregModel$occurrence),
                                                                               h=maxLag));
                                }
                                else{
                                    xregModel <- suppressWarnings(smooth::es(rev(xregData), model=smooth::modelType(xregModel),
                                                                             persistence=xregModel$persistence,
                                                                             occurrence=xregModel$occurrence,
                                                                             oesmodel=smooth::modelType(xregModel$occurrence),
                                                                             h=maxLag));
                                }
                            }
                            xregDataNew <- c(rev(xregModel$forecast),xregDataNew);
                        }
                        else{
                            # If this is a binary variable, use iss function.
                            if(all((xregData==0) | (xregData==1))){
                                xregModel <- suppressWarnings(occurrenceModel(rev(xregData), model="MNN", h=maxLag, occurrence="i"));
                            }
                            else{
                                if(packageVersion("smooth")>="3.0.0"){
                                    xregModel <- suppressWarnings(smooth::adam(rev(xregData), h=maxLag, occurrence="a"));
                                }
                                else{
                                    xregModel <- suppressWarnings(smooth::es(rev(xregData), h=maxLag));
                                }
                            }
                            xregDataNew <- c(rev(xregModel$forecast),xregData);
                        }
                    }
                }
                else{
                    xregDataNew <- c(rep(NA,maxLag),xregData);
                }
            }

            # Fill the lags
            if(any(lagsOriginal<0)){
                for(j in 1:lagsLength){
                    xregNew[,chosenColumn+1+j] <- xregDataNew[1:obs-lags[j]+maxLag];
                    colnames(xregNew)[chosenColumn+1+j] <- paste0(xregCurrentName,"Lag",lags[j]);
                }
            }
            # Fill the leads
            if(any(lagsOriginal>0)){
                for(j in 1:leadsLength){
                    xregNew[,chosenColumn+1+lagsLength+j] <- xregDataNew[1:obs+leads[j]+maxLag];
                    colnames(xregNew)[chosenColumn+1+lagsLength+j] <- paste0(xregCurrentName,"Lead",leads[j]);
                }
            }
        }
        if(!silent){
            cat(paste0(rep("\b",4),collapse=""));
            cat(" Done! \n");
        }
    }
    return(xregNew);
}
