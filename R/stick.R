#' Seasonality, Trend, and Irregular Contribution Kit
#'
#' The function decomposes the variance of a time series into seasonal, trend
#' and irregular parts based on an Analysis of Variance (ANOVA) of the series
#' on the seasonal and trend factors and measures their contribution to the data.
#'
#' \code{stick()} forms internally a data frame with the response \code{y} in
#' the first column, a categorical (factor) variable for each of the provided
#' seasonal \code{lags} and a "trend" factor in the last column. For a monthly
#' series with \code{lags=12}, the seasonal factor takes values \code{1:12}
#' repeated throughout the sample (the month of the year), while the trend
#' factor takes values \code{1:ceiling(T/12)}, each repeated 12 times (the
#' year). The trend factor is constructed based on the longest of the provided
#' lags.
#'
#' \code{aov()} is then applied to the model \code{y ~ seasonal + trend} and the
#' strength of each component is measured as the share of the respective Sum of
#' Squares in the total Sum of Squares. The irregular component corresponds to
#' the share of the residual Sum of Squares. The strengths of all the components
#' sum up to one.
#'
#' The function implements the Seasonality, Trend, and Irregular (STI)
#' classification of Hans Levenbach (see the reference below).
#'
#' @param y The vector or a ts object, containing the time series to analyse.
#' @param lags The vector of seasonal lags (periodicities) in the data, e.g.
#' \code{lags=c(24,168)} for the hourly data or \code{lags=12} for the monthly
#' one. Values not greater than one are dropped. Defaults to the frequency of
#' \code{y}.
#' @param ... Other parameters passed to the \code{aov()} function.
#'
#' @return \code{stick()} returns an object of class "stick", which contains:
#' \itemize{
#' \item y - The original data;
#' \item lags - The seasonal lags used in the analysis;
#' \item model - The fitted \code{aov} model;
#' \item anova - The ANOVA table (the first element of \code{summary(model)});
#' \item strength - The named vector with the strength of each seasonal
#' component, the trend and the irregular component. The values sum up to one.
#' }
#'
#' @template author
#' @template keywords
#'
#' @references Levenbach, H. (2021). Four P's in a Pod: e-Commerce Forecasting
#' and Planning for Supply Chain Practitioners. Independently published.
#' ISBN 979-8461733575.
#'
#' @seealso \code{\link[stats]{aov}}
#'
#' @examples
#' # Generate a seasonal series and classify it
#' y <- ts(rnorm(120, 100 + 20 * sin(2 * pi * (1:120) / 12), 5), frequency=12)
#' stick(y)
#'
#' @importFrom stats aov as.formula frequency
#' @export
stick <- function(y, lags=frequency(y), ...){
    obsInSample <- length(y);

    # Only seasonal lags above one make sense; keep unique and sorted
    lags <- sort(unique(lags[lags>1]));
    if(length(lags)==0){
        stop(paste0("No seasonal lags greater than one were provided. ",
                    "stick() needs at least one seasonal periodicity."),
             call.=FALSE);
    }

    # Drop the lags that are too long to form at least two full cycles
    lagsTooLong <- lags>=obsInSample;
    if(any(lagsTooLong)){
        warning(paste0("The following lags are not smaller than the number of ",
                       "observations and are dropped: ",
                       paste(lags[lagsTooLong], collapse=", "), "."),
                call.=FALSE);
        lags <- lags[!lagsTooLong];
    }
    if(length(lags)==0){
        stop("None of the provided lags is shorter than the sample.", call.=FALSE);
    }

    # The data frame with y in the first column
    sticksData <- data.frame(y=as.vector(y));

    # A seasonal factor for each lag: the position within the cycle
    seasonalNames <- paste0("seasonal", lags);
    for(i in seq_along(lags)){
        sticksData[[seasonalNames[i]]] <-
            factor(((seq_len(obsInSample)-1) %% lags[i]) + 1);
    }

    # The trend ("year") factor, based on the longest lag
    lagMax <- max(lags);
    trendFactor <- factor(((seq_len(obsInSample)-1) %/% lagMax) + 1);
    # Only include the trend if it has at least two levels with observed y
    trendIncluded <- length(unique(trendFactor[!is.na(sticksData$y)]))>1;
    terms <- seasonalNames;
    if(trendIncluded){
        sticksData[["trend"]] <- trendFactor;
        terms <- c(terms, "trend");
    }

    # Build the formula and fit the ANOVA model
    sticksFormula <- as.formula(paste0("y ~ ", paste(terms, collapse=" + ")));
    sticksModel <- aov(sticksFormula, data=sticksData, ...);
    sticksANOVA <- summary(sticksModel)[[1]];

    # The strength of each component is its share of the total Sum of Squares
    SSQ <- sticksANOVA[,"Sum Sq"];
    strength <- SSQ / sum(SSQ);
    names(strength) <- trimws(rownames(sticksANOVA));
    # The residual Sum of Squares corresponds to the irregular component
    names(strength)[names(strength)=="Residuals"] <- "irregular";

    return(structure(list(y=y, lags=lags, model=sticksModel, anova=sticksANOVA,
                          strength=strength),
                     class="stick"));
}

#' @param x The object of class "stick" generated by the \code{stick()} function.
#' @param digits The number of digits to use when printing the strength of the
#' components.
#' @rdname stick
#' @export
print.stick <- function(x, digits=4, ...){
    cat("Seasonality, Trend, and Irregular Contribution Kit\n");
    cat("Seasonal lags: ", paste(x$lags, collapse=", "), "\n\n", sep="");
    cat("Strength of the components:\n");
    print(round(x$strength, digits));
}

#' @param which The vector of integers specifying which plots to produce: the
#' values \code{1, ..., k-1} correspond to the seasonal plots for each of the
#' seasonal \code{lags} (in the ascending order), while \code{k} corresponds to
#' the trend plot (the last one). If \code{NULL} (the default), all the plots
#' are produced.
#' @param ask Logical; if \code{TRUE}, the user is asked to hit Enter before
#' each new plot is drawn. The default only asks when the requested plots do not
#' all fit on the current device layout (see \link[graphics]{par}).
#'
#' @details
#' \code{plot.stick()} reproduces two types of plots, in the spirit of the STI
#' decomposition. For every seasonal lag it draws a "seasonal plot": the series
#' is reshaped into a matrix with one column per cycle (e.g. per year for
#' monthly data) and the value within the cycle on the x-axis; each cycle is a
#' grey line and the average seasonal profile (the mean across the cycles) is
#' overlaid as a bold dashed black line. The final plot is the "trend plot",
#' which uses the longest lag: each seasonal position (e.g. each month) is a
#' grey line drawn across the cycles, and the average level per cycle (the
#' trend) is overlaid as a bold dashed black line.
#'
#' @return \code{plot.stick()} is called for its side effect of producing the
#' plots and returns \code{NULL} invisibly.
#'
#' @examples
#' # Produce all the plots
#' par(mfcol=c(1,2))
#' plot(stick(AirPassengers))
#' # Only the trend plot
#' plot(stick(AirPassengers), which=2)
#'
#' @importFrom grDevices grey dev.interactive devAskNewPage
#' @importFrom graphics plot lines par
#' @rdname stick
#' @export
plot.stick <- function(x, which=NULL, ask, ...){
    lags <- x$lags;
    nLags <- length(lags);
    # The trend plot is only available if stick() actually fitted a trend
    hasTrend <- "trend" %in% names(x$strength);
    nPlots <- nLags + as.integer(hasTrend);

    # Resolve which plots to produce
    if(is.null(which)){
        which <- seq_len(nPlots);
    }
    else{
        whichValid <- which %in% seq_len(nPlots);
        if(any(!whichValid)){
            warning(paste0("Some values of 'which' are outside the range [1, ",
                           nPlots, "] and are dropped."), call.=FALSE);
            which <- which[whichValid];
        }
    }
    if(length(which)==0){
        return(invisible(NULL));
    }

    # Wait for "Enter" between plots only if they do not fit the layout,
    # mirroring the behaviour of plot.greybox().
    if(missing(ask)){
        ask <- prod(par("mfcol")) < length(which) && dev.interactive();
    }
    if(ask){
        oask <- devAskNewPage(TRUE);
        on.exit(devAskNewPage(oask));
    }

    yValues <- as.vector(x$y);
    obsInSample <- length(yValues);
    lagMax <- max(lags);

    # Reshape the series into a [period x cycle] matrix, padding the last
    # (incomplete) cycle with NAs. Columns are cycles, rows are the positions
    # within the cycle.
    reshapeToMatrix <- function(period){
        nCycles <- ceiling(obsInSample / period);
        yPadded <- c(yValues, rep(NA, nCycles*period - obsInSample));
        matrix(yPadded, nrow=period, ncol=nCycles);
    }

    # Seasonal plot: one grey line per cycle, the average seasonal profile
    # (mean across the cycles) overlaid as a bold dashed black line.
    plotSeasonal <- function(seasonalMatrix, lag, ...){
        ellipsis <- list(...);
        nRow <- nrow(seasonalMatrix);
        nCol <- ncol(seasonalMatrix);

        if(is.null(ellipsis$main)){
            ellipsis$main <- paste0("Seasonal plot (lag ", lag, ")");
        }
        if(is.null(ellipsis$xlab)){
            ellipsis$xlab <- "Period within the cycle";
        }
        if(is.null(ellipsis$ylab)){
            ellipsis$ylab <- "Value";
        }
        if(is.null(ellipsis$ylim)){
            ellipsis$ylim <- range(seasonalMatrix, na.rm=TRUE);
        }
        cols <- grey(0.85 * seq_len(nCol) / nCol);

        ellipsis$x <- seq_len(nRow);
        ellipsis$y <- seasonalMatrix[,1];
        ellipsis$type <- "l";
        ellipsis$col <- cols[1];
        do.call(plot, ellipsis);
        for(j in seq_len(nCol)[-1]){
            lines(seq_len(nRow), seasonalMatrix[,j], col=cols[j]);
        }
        # The average seasonal profile across the cycles
        lines(seq_len(nRow), rowMeans(seasonalMatrix, na.rm=TRUE),
              col="black", lwd=2, lty=2);
    }

    # Trend plot: one grey line per seasonal position across the cycles, the
    # average level per cycle (the trend) overlaid as a bold dashed black line.
    plotTrend <- function(seasonalMatrix, ...){
        ellipsis <- list(...);
        nRow <- nrow(seasonalMatrix);
        nCol <- ncol(seasonalMatrix);

        if(is.null(ellipsis$main)){
            ellipsis$main <- "Trend plot";
        }
        if(is.null(ellipsis$xlab)){
            ellipsis$xlab <- "Cycle";
        }
        if(is.null(ellipsis$ylab)){
            ellipsis$ylab <- "Value";
        }
        if(is.null(ellipsis$ylim)){
            ellipsis$ylim <- range(seasonalMatrix, na.rm=TRUE);
        }
        cols <- grey(0.85 * seq_len(nRow) / nRow);

        ellipsis$x <- seq_len(nCol);
        ellipsis$y <- seasonalMatrix[1,];
        ellipsis$type <- "l";
        ellipsis$col <- cols[1];
        do.call(plot, ellipsis);
        for(i in seq_len(nRow)[-1]){
            lines(seq_len(nCol), seasonalMatrix[i,], col=cols[i]);
        }
        # The average level per cycle: the trend
        lines(seq_len(nCol), colMeans(seasonalMatrix, na.rm=TRUE),
              col="black", lwd=2, lty=2);
    }

    for(i in which){
        if(i<=nLags){
            plotSeasonal(reshapeToMatrix(lags[i]), lags[i], ...);
        }
        else{
            plotTrend(reshapeToMatrix(lagMax), ...);
        }
    }

    return(invisible(NULL));
}
