#' Construct scatterplot / boxplots for the data
#'
#' Function constructs the plots depending on the types of variables in the provided
#' matrix / data frame.
#'
#' If both variables are in metric scale, then the classical scatterplot is constructed.
#' If one of them is either integer (up to 10 values) or categorical (aka 'factor'),
#' then boxplots (with grey dots corresponding to mean values) are constructed. Finally,
#' for the two categorical variables the tableplot is returned (see
#' \link[greybox]{tableplot} function for the details). All of this is packed in a matrix.
#' The colours in the plot can be changed by defining a different palette via the
#' \code{palette()} function, in which case spread() will use the first four colours
#' in the pallete.
#'
#' See details in the vignette "Marketing analytics with greybox":
#' \code{vignette("maUsingGreybox","greybox")}
#'
#' @template author
#' @keywords plots graph
#'
#' @param data Either matrix or data frame with the data.
#' @param histograms If \code{TRUE}, then the histograms and barplots are produced on
#' the diagonal of the matrix. Otherwise the names of the variables are written there.
#' @param log If \code{TRUE}, then the logarithms of all numerical variables are taken.
#' @param lowess If \code{TRUE}, then LOWESS lines are added to scatterplots and means
#' are connected with lines on boxplots, see \link[stats]{lowess} for details.
#' @param ... Other parameters passed to the plot function. Currently only "main"
#' parameter is accepted.
#'
#' @return Function does not return anything. It just plots things.
#'
#' @seealso \code{\link[graphics]{plot}, \link[base]{table}, \link[greybox]{tableplot}}
#'
#' @examples
#'
#' ### Simple example
#' spread(mtcars)
#' spread(mtcars,log=TRUE)
#'
#' @importFrom graphics barplot boxplot hist mtext text title
#' @importFrom stats formula
#' @export spread
spread <- function(data, histograms=FALSE, log=FALSE, lowess=FALSE, ...){
    ellipsis <- list(...);

    if(is.null(ellipsis$main)){
        mainTitle <- "";
        omaValues <- c(2,3,3,2);
    }
    else{
        mainTitle <- ellipsis$main;
        omaValues <- c(2,3,7,2);
    }

    if(!histograms){
        omaValues[c(2,3)] <- omaValues[c(2,3)]-1;
        omaValues[c(4)] <- omaValues[4]-1;
    }

    if(!is.data.frame(data)){
        data <- as.data.frame(data);
    }

    nVariables <- ncol(data);
    if(nVariables>20){
        stop(paste0("Too many variables! I can't work in such conditions! Too much pressure! ",
                    "Can you, please, reduce the number of variables at least to 20?"),
             call.=FALSE);
    }

    numericData <- vector(mode="logical", length=nVariables);
    for(i in 1:nVariables){
        numericData[i] <- is.numeric(data[[i]]);
        if(numericData[i]){
            if(length(unique(data[[i]]))<=10){
                numericData[i] <- FALSE;
                # If there are some data with low levels transform them into factors
                data[[i]] <- factor(data[[i]]);
            }
        }
        else{
            # If the variable is not a factor (e.g. character), make it.
            if(!is.factor(data[[i]])){
                data[[i]] <- factor(data[[i]]);
            }
        }
    }

    # Set the palette for the plots
    paletteBasic <- paletteDetector(c("black","grey41","grey","lightgrey"));

    if(log){
        if(any(data[numericData]<=0)){
            warning("Some variables have non-positive data, so logarithms cannot be produced for them.",call.=FALSE);
            nonZeroData <- numericData;
            nonZeroData[numericData] <- apply(data[numericData]>0,2,all);
        }
        else{
            nonZeroData <- numericData;
        }
        data[nonZeroData] <- log(data[nonZeroData]);
        colnames(data)[which(nonZeroData)] <- paste0("log_",colnames(data)[which(nonZeroData)]);
    }
    variablesNames <- colnames(data);

    parDefault <- par(no.readonly=TRUE);
    on.exit(par(parDefault));

    if(nVariables==1){
        if(numericData[1]){
            hist(data[[1]], main=mainTitle, col=paletteBasic[4]);
        }
        else{
            barplot(table(data[[1]]), col=paletteBasic[4]);
        }
    }
    else{
        par(mfcol=c(nVariables,nVariables), mar=rep(0,4), oma=omaValues, xaxt="s",yaxt="s",cex.main=1.5);

        for(i in 1:nVariables){
            for(j in 1:nVariables){
                # par(mar=rep(0,4), oma=omaValues, xaxt="s",yaxt="s")
                if(i==j){
                    if(histograms){
                        if(numericData[i]){
                            histPlotted <- hist(data[[i]], main="", axes=FALSE, col=paletteBasic[4]);
                            histPlottedX <- histPlotted$breaks;
                            histPlottedYRange <- range(histPlotted$counts, na.rm=TRUE);
                        }
                        else{
                            dataTable <- table(data[[i]]);
                            histPlottedX <- barplot(dataTable, main="", axes=FALSE,
                                                    axisnames=FALSE, col=paletteBasic[4]);
                            histPlottedYRange <- range(dataTable, na.rm=TRUE);
                        }
                    }
                    else{
                        if(numericData[i]){
                            midPoint <- (max(data[[i]], na.rm=TRUE)+min(data[[i]], na.rm=TRUE))/2;
                            plot(0, 0, col="white", axes=FALSE,
                                 xlim=range(data[[i]], na.rm=TRUE), ylim=range(data[[i]], na.rm=TRUE));
                            text(midPoint,midPoint,variablesNames[i],cex=1.5);
                        }
                        else{
                            uniqueValues <- levels(data[[i]]);
                            midPoint <- (1+length(uniqueValues))/2;
                            plot(0, 0, col="white", axes=FALSE,
                                 xlim=c(0.5,length(uniqueValues)+0.5), ylim=c(0.5,length(uniqueValues)+0.5));
                            text(midPoint,midPoint,variablesNames[i],cex=1.5);
                        }
                    }
                }
                else{
                    if(numericData[i] && numericData[j]){
                        plot(data[[i]],data[[j]], main="", axes=FALSE, col=paletteBasic[1]);
                        if(lowess){
                            # Get the default delta, but use na.rm to get rid of NAs
                            delta <- 0.01 * diff(range(data[[i]], na.rm=TRUE));
                            lowessLine <-lowess(data[[i]], data[[j]], delta=delta);
                            lines(lowessLine, col=paletteBasic[3], lty=2, lwd=2);
                        }
                    }
                    else if(numericData[i]){
                        boxplot(as.formula(paste0("`",variablesNames[i],"`~`",variablesNames[j],"`")),
                                data, horizontal=TRUE, main="", axes=FALSE, col=paletteBasic[4]);
                        if(lowess){
                            lines(tapply(data[[i]],data[[j]],mean,na.rm=TRUE),
                                  c(1:length(levels(data[[j]]))), col=paletteBasic[3], lty=2, lwd=2);
                        }
                        points(tapply(data[[i]],data[[j]],mean,na.rm=TRUE),
                               c(1:length(levels(data[[j]]))), pch=19, col=paletteBasic[3]);
                    }
                    else if(numericData[j]){
                        boxplot(as.formula(paste0("`",variablesNames[j],"`~`",variablesNames[i],"`")),
                                data, main="", axes=FALSE, col=paletteBasic[4]);
                        if(lowess){
                            lines(tapply(data[[j]],data[[i]],mean,na.rm=TRUE), col=paletteBasic[3], lty=2, lwd=2);
                        }
                        points(tapply(data[[j]],data[[i]],mean,na.rm=TRUE), pch=19, col=paletteBasic[3]);
                    }
                    else{
                        tableplot(data[[i]],data[[j]], labels=FALSE, legend=FALSE, main="", axes=FALSE);
                    }
                }

                # Add axis and labels if this is the first element
                if(i==1){
                    if(histograms){
                        mtext(variablesNames[nVariables-j+1], side=2,
                              at=(j-0.35)/nVariables, line=1, adj=1, outer=TRUE);
                    }
                    else{
                        if(numericData[j]){
                            axis(2);
                        }
                        else{
                            uniqueValues <- levels(data[[j]]);
                            axis(2, at=seq(1,length(uniqueValues),length.out=length(uniqueValues)),
                                 labels=uniqueValues);
                        }
                    }
                }
                #
                #                 if(j==1){
                #                     # Add axis at the top
                #                     if(!histograms){
                #                         if(numericData[i]){
                #                             axis(3);
                #                         }
                #                         else{
                #                             uniqueValues <- sort(unique(data[[i]]));
                #                             axis(3,at=seq(1,length(uniqueValues),length.out=length(uniqueValues)),
                #                                  labels=uniqueValues);
                #                         }
                #                     }
                #                 }

                # Add axis, if this is the last element in the matrix
                if(i==nVariables && histograms){
                    if(numericData[j]){
                        axis(4);
                    }
                    else{
                        uniqueValues <- levels(data[[j]]);
                        # The last element in the matrix is special in case of histogram
                        if(j==nVariables){
                            axis(4, at=seq(histPlottedYRange[1],histPlottedYRange[2]-0.5,
                                           length.out=length(uniqueValues)),
                                 labels=uniqueValues);
                        }
                        else{
                            axis(4, at=seq(1,length(uniqueValues), length.out=length(uniqueValues)),
                                 labels=uniqueValues);
                        }
                    }
                }

                box();
            }
            # Add axis at the bottom
            if(numericData[i]){
                axis(1);
            }
            else{
                if(i==nVariables && histograms){
                    uniqueValues <- levels(unique(data[[i]]));
                    axis(1,at=histPlottedX, labels=uniqueValues);
                }
                else{
                    uniqueValues <- levels(unique(data[[i]]));
                    axis(1,at=seq(1,length(uniqueValues), length.out=length(uniqueValues)),
                         labels=uniqueValues);
                }
            }
            if(histograms){
                mtext(variablesNames[i], side=3, at=(i-0.5)/nVariables, line=1, outer=TRUE);
            }
        }

        if(mainTitle!=""){
            title(main=mainTitle, outer=TRUE, line=3, cex.main=2);
        }
    }
}

