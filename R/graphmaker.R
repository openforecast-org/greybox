#' Linear graph construction function
#'
#' The function makes a standard linear graph using the provided actuals and
#' forecasts.
#'
#' Function uses the provided data to construct a linear graph. It is strongly
#' advised to use \code{ts} objects to define the start of each of the vectors.
#' Otherwise the data may be plotted incorrectly. The colours can be changed by
#' defining a different palette via the \code{palette()} function. The function
#' then would use colours 1 - 6 in the palette.
#'
#' @param actuals The vector of actual values
#' @param forecast The vector of forecasts. Should be \code{ts} object that starts at
#' the end of \code{fitted} values.
#' @param fitted The vector of fitted values.
#' @param lower The vector of lower bound values of a prediction interval.
#' Should be ts object that start at the end of \code{fitted} values.
#' @param upper The vector of upper bound values of a prediction interval.
#' Should be ts object that start at the end of \code{fitted} values.
#' @param level The width of the prediction interval.
#' @param legend If \code{TRUE}, the legend is drawn.
#' @param vline Whether to draw the vertical line, splitting the in-sample
#' and the holdout sample.
#' @param parReset Whether to reset par() after plotting things or not.
#' If \code{FALSE} then you can add elements to the plot (e.g. additional lines).
#' @param ... Other parameters passed to \code{plot()} function.
#' @return Function does not return anything.
#' @author Ivan Svetunkov
#' @seealso \code{\link[stats]{ts}}
#' @keywords plots linear graph
#' @examples
#'
#' xreg <- cbind(y=rnorm(100,100,10),x=rnorm(100,10,10))
#' almModel <- alm(y~x, xreg, subset=c(1:90))
#' values <- predict(almModel, newdata=xreg[-c(1:90),], interval="prediction")
#'
#' graphmaker(xreg[,1],values$mean,fitted(values))
#' graphmaker(xreg[,1],values$mean,fitted(values),legend=FALSE)
#' graphmaker(xreg[,1],values$mean,fitted(values),legend=FALSE,lower=values$lower,upper=values$upper)
#'
#' # Produce the necessary ts objects from an arbitrary vectors
#' actuals <- ts(c(1:10), start=c(2000,1), frequency=4)
#' forecast <- ts(c(11:15),start=end(actuals)[1]+end(actuals)[2]*deltat(actuals),
#'                frequency=frequency(actuals))
#' graphmaker(actuals,forecast)
#'
#' # This should work as well
#' graphmaker(c(1:10),c(11:15))
#'
#' # This way you can add additional elements to the plot
#' graphmaker(c(1:10),c(11:15), parReset=FALSE)
#' points(c(1:15))
#' # But don't forget to do dev.off() in order to reset the plotting area afterwards
#'
#' @export graphmaker
#' @importFrom graphics rect
#' @importFrom zoo zoo
graphmaker <- function(actuals, forecast, fitted=NULL, lower=NULL, upper=NULL,
                       level=NULL, legend=TRUE, vline=TRUE,
                       parReset=TRUE, ...){
    # Function constructs the universal linear graph for any model

    ellipsis <- list(...);

    ##### Make legend change depending on the fitted values
    if(!is.null(lower) | !is.null(upper)){
        intervals <- TRUE;
        if(is.null(level)){
            if(legend){
                message("The width of prediction intervals is not provided to the function! Assuming 95%.");
            }
            level <- 0.95;
        }
    }
    else{
        lower <- NA;
        upper <- NA;
        intervals <- FALSE;
    }
    if(is.null(fitted)){
        fitted <- NA;
    }
    h <- length(forecast);

    pointForecastLabel <- "Point forecast";

    # Write down the default values of par
    parDefault <- par(no.readonly=TRUE);
    if(parReset){
        on.exit(par(parDefault));
    }

    if(legend){
        layout(matrix(c(2,1),2,1),heights=c(0.86,0.14));
        if(is.null(ellipsis$main)){
            parMar <- c(2,3,2,1);
        }
        else{
            parMar <- c(2,3,3,1);
        }
    }
    else{
        if(is.null(ellipsis$main)){
            parMar <- c(3,3,2,1);
        }
        else{
            parMar <- c(3,3,4,1);
        }
    }

    # Define palette
    paletteBasic <- paletteDetector(c("black","red","purple","blue","darkgrey","lightgrey"));
    ellipsis$col <- paletteBasic[1];

    legendCall <- list(x="bottom");
    if(length(level)>1){
        legendCall$legend <- c("Series","Fitted values",pointForecastLabel,
                               paste0(paste0(level*100,collapse=", "),"% prediction intervals"),"Forecast origin");
    }
    else{
        legendCall$legend <- c("Series","Fitted values",pointForecastLabel,
                               paste0(level*100,"% prediction interval"),"Forecast origin");
    }
    legendCall$col <- paletteBasic[c(1,3,4,5,2)];
    legendCall$lwd <- c(1,2,2,3,2);
    legendCall$lty <- c(1,2,1,2,1);
    legendCall$ncol <- 3
    legendElements <- rep(TRUE,5);

    if(all(is.na(forecast))){
        h <- 0;
        pointForecastLabel <- NULL;
        legendElements[c(3,5)] <- FALSE;
        vline <- FALSE;
    }

    if(h==1){
        legendCall$pch <- c(NA,NA,4,4,NA);
        legendCall$lty <- c(1,2,0,0,1);
    }
    else{
        legendCall$pch <- rep(NA,5);
    }

    # Estimate plot range
    if(is.null(ellipsis$ylim)){
        ellipsis$ylim <- range(c(as.vector(actuals[is.finite(actuals)]),as.vector(fitted[is.finite(fitted)]),
                                 as.vector(forecast[is.finite(forecast)]),
                                 as.vector(lower[is.finite(lower)]),as.vector(upper[is.finite(upper)])),
                               na.rm=TRUE);
    }

    # The time step of the data, used for the forecast and its origin
    if(is.ts(actuals)){
        timeStep <- deltat(actuals);
    }
    else{
        timeStep <- mean(diff(time(actuals)));
    }

    # If the forecast has no time of its own (a vector) or starts with the actuals, it follows them
    if(!is.ts(forecast) && !inherits(forecast, "zoo") || time(forecast)[1]==time(actuals)[1]){
        if(is.ts(actuals)){
            forecast <- ts(forecast, start=time(actuals)[length(actuals)]+timeStep, frequency=frequency(actuals));
        }
        else{
            forecast <- zoo(forecast, order.by=time(actuals)[length(actuals)]+timeStep*c(1:h));
        }
        if(intervals){
            if(is.ts(actuals)){
                if(length(upper)!=length(fitted) && length(lower)!=length(fitted)){
                    upper <- ts(upper, start=start(forecast), frequency=frequency(actuals));
                    lower <- ts(lower, start=start(forecast), frequency=frequency(actuals));
                }
                else{
                    upper <- ts(upper, start=start(actuals), frequency=frequency(actuals));
                    lower <- ts(lower, start=start(actuals), frequency=frequency(actuals));
                }
            }
            else{
                if(length(upper)!=length(fitted) && length(lower)!=length(fitted)){
                    upper <- zoo(upper, order.by=time(forecast));
                    lower <- zoo(lower, order.by=time(forecast));
                }
                else{
                    upper <- zoo(upper, order.by=time(forecast));
                    lower <- zoo(lower, order.by=time(forecast));
                }
            }
        }
    }

    if(is.null(ellipsis$xlim)){
        ellipsis$xlim <- range(time(actuals)[1],time(forecast)[max(h,1)]);
    }

    if(is.null(ellipsis$type)){
        ellipsis$type <- "l";
    }

    if(is.null(ellipsis$xlab)){
        ellipsis$xlab <- "";
    }
    else{
        parMar[1] <- parMar[1] + 1;
    }

    if(is.null(ellipsis$ylab)){
        ellipsis$ylab <- "";
    }
    else{
        parMar[2] <- parMar[2] + 1;
    }

    ellipsis$x <- actuals;

    if(!intervals){
        legendElements[4] <- FALSE;
        legendCall$ncol <- 2;
    }
    else if(h!=1 && is.matrix(lower) && is.matrix(upper)){
        legendCall$lwd[4] <- 2;
    }

    if(all(is.na(fitted))){
        legendElements[2] <- FALSE;
    }

    if(legend){
        legendCall$legend <- legendCall$legend[legendElements];
        legendCall$col <- legendCall$col[legendElements];
        legendCall$lwd <- legendCall$lwd[legendElements];
        legendCall$lty <- legendCall$lty[legendElements];
        legendCall$pch <- legendCall$pch[legendElements];
        legendCall$bty <- "n";

        if(parReset){
            par(cex=0.75, mar=rep(0.1,4), bty="n", xaxt="n", yaxt="n")
        }
        else{
            par(mar=rep(0.1,4), bty="n", xaxt="n", yaxt="n")
        }
        plot(0,0,col=0)
        legendDone <- do.call("legend", legendCall);
        rect(legendDone$rect$left-0.02, legendDone$rect$top-legendDone$rect$h,
             legendDone$rect$left+legendDone$rect$w+0.02, legendDone$rect$top);
    }

    if(parReset){
        par(mar=parMar, cex=1, bty="o", xaxt="s", yaxt="s");
    }
    # else{
    #     par(mar=parMar, bty="o", xaxt="s", yaxt="s");
    # }
    do.call(plot, ellipsis);

    if(vline){
        abline(v=time(forecast)[1]-timeStep,col=paletteBasic[2],lwd=2);
    }

    if(intervals){
        if(h!=1){
            # Draw the nice areas between the borders
            if(is.matrix(lower) && is.matrix(upper) && ncol(lower)==ncol(upper)){
                for(i in 1:ncol(lower)){
                    if(all(is.finite(upper[,i])) && all(is.finite(lower[,i]))){
                        polygon(c(time(upper[,i]),rev(time(lower[,i]))),
                                c(as.vector(upper[,i]), rev(as.vector(lower[,i]))),
                                col=paletteBasic[5], border=NA, density=(ncol(lower)-i+1)*10);
                    }
                }
            }
            # If the number of columns is different, then do polygon for the last ones only
            else if(is.matrix(lower) && is.matrix(upper) && ncol(lower)!=ncol(upper)){
                polygon(c(time(upper[,ncol(upper)]),rev(time(lower[,ncol(lower)]))),
                        c(as.vector(upper[,ncol(upper)]), rev(as.vector(lower[,ncol(lower)]))),
                        col=paletteBasic[5], border=NA, density=10);
            }
            # If upper is not a matrix, use it as a vector
            else if(is.matrix(lower) && !is.matrix(upper)){
                polygon(c(time(upper),rev(time(lower[,ncol(lower)]))),
                        c(as.vector(upper), rev(as.vector(lower[,ncol(lower)]))),
                        col=paletteBasic[5], border=NA, density=10);
            }
            # If lower is not a matrix, use it as a vector
            else if(!is.matrix(lower) && is.matrix(upper)){
                polygon(c(time(upper[,ncol(upper)]),rev(time(lower))),
                        c(as.vector(upper[,ncol(upper)]), rev(as.vector(lower))),
                        col=paletteBasic[5], border=NA, density=10);
            }
            # Otherwise use both as vectors
            else{
                if(all(is.finite(upper)) && all(is.finite(lower))){
                    polygon(c(time(upper),rev(time(lower))),
                            c(as.vector(upper), rev(as.vector(lower))),
                            col=paletteBasic[5], border=NA, density=10);
                }
            }


            if(is.matrix(lower) || is.matrix(upper)){
                nLevels <- max(ncol(lower), ncol(upper));
                col <- colorRampPalette(c(paletteBasic[c(5,6)]))(nLevels)[findInterval(1:nLevels,
                                                                                    seq(1, nLevels, length.out=nLevels))];
            }
            # Draw the lines
            if(is.matrix(lower)){
                for(i in 1:ncol(lower)){
                    lines(lower[,i],col=col[i],lwd=2,lty=2);
                }
            }
            else{
                lines(lower,col=paletteBasic[5],lwd=3,lty=2);
            }

            if(is.matrix(upper)){
                for(i in 1:ncol(upper)){
                    lines(upper[,i],col=col[i],lwd=2,lty=2);
                }
            }
            else{
                lines(upper,col=paletteBasic[5],lwd=3,lty=2);
            }

            lines(forecast,col=paletteBasic[4],lwd=2);
            do.call("lines", ellipsis);
        }
        # Code for the h=1
        else{
            if(length(lower)>1){
                col <- grey(1-c(1:ncol(lower))/(ncol(lower)+1));
                for(i in 1:length(lower)){
                    if(is.finite(lower[i])){
                        points(lower[,i],col=col[i],lwd=1+i,pch=4);
                    }
                }
            }
            else{
                points(lower,col=paletteBasic[5],lwd=3,pch=4);
            }
            if(length(upper)>1){
                col <- grey(1-c(1:ncol(upper))/(ncol(upper)+1));
                for(i in 1:length(upper)){
                    if(is.finite(upper[i])){
                        points(upper[,i],col=col[i],lwd=1+i,pch=4);
                    }
                }
            }
            else{
                points(upper,col=paletteBasic[5],lwd=3,pch=4);
            }
            points(forecast,col=paletteBasic[4],lwd=2,pch=4);
        }
    }
    else{
        if(h!=1){
            lines(forecast,col=paletteBasic[4],lwd=2);
        }
        else{
            points(forecast,col=paletteBasic[4],lwd=2,pch=4);
        }
    }

    if(any(!is.na(fitted))){
        lines(fitted, col=paletteBasic[3], lwd=2, lty=2);
    }

}

paletteDetector <- function(colours){
    # If the default palette is used, define the new one in the provided colours
    paletteBasic <- palette();
    palette("default");
    paletteDefault <- palette();
    if(all(paletteBasic %in% paletteDefault) &&
       all(paletteBasic==paletteDefault)){
        paletteBasic <- colours;
    }
    else{
        palette(paletteBasic);
    }
    return(paletteBasic);
}
