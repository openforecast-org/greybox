#' Construct a plot for categorical variable
#'
#' Function constructs a plot for two categorical variables based on table function
#'
#' The function produces the plot of the \code{table()} function with colour densities
#' corresponding to the respective frequencies of appearance. If the value appears more
#' often than the other (e.g. 0.5 vs 0.15), then it will be darker. By default, the
#' frequency of 0 corresponds to the white colour, the frequency of 1 corresponds to the
#' black. The colours can be changed by defining a different palette via the
#' \code{palette()} function. In that case only two first colours are used, where the
#' colour intensity changes from the first one to the second one. The function also
#' adds the number of dots (points) proportional to the number of values in each
#' category to simplify the reading of plots.
#'
#' See details in the vignette "Marketing analytics with greybox":
#' \code{vignette("maUsingGreybox","greybox")}
#'
#' @template author
#' @keywords plots graph
#'
#' @param x First categorical variable. Can be either vector, factor, matrix or a data
#' frame. If \code{y} is NULL and x is either matrix of a data frame, then the first two
#' variables of the data will be plotted against each other.
#' @param y Second categorical variable. If not provided, then only \code{x} will be
#' plotted.
#' @param labels Whether to print table labels inside the plot or not.
#' @param legend If \code{TRUE}, then the legend for the tableplot is drawn. The plot is
#' then produced on a separate canvas (new \code{par()}).
#' @param points Whether to plot points in the areas. They help in understanding how
#' many values lie in specific categories.
#' @param ... Other parameters passed to the plot function.
#'
#' @return Function does not return anything. It just plots things.
#'
#' @seealso \code{\link[graphics]{plot}, \link[base]{table}, \link[greybox]{spread}}
#'
#' @examples
#'
#' palette("Tableau")
#' tableplot(mtcars$am, mtcars$gear)
#'
#' @importFrom utils head tail
#' @importFrom grDevices colorRampPalette palette
#' @export tableplot
tableplot <- function(x, y=NULL, labels=TRUE, legend=FALSE, points=TRUE, ...){
    ellipsis <- list(...);

    if(is.null(y)){
        if(is.matrix(x)){
            if(ncol(x)>1){
                ellipsis$xlab <- colnames(x)[1];
                ellipsis$ylab <- colnames(x)[2];
                y <- x[,2];
                x <- x[,1];
                yIsProvided <- TRUE;
            }
            else{
                yIsProvided <- FALSE;
                y <- rep(0,length(x));
            }
        }
        else if(is.data.frame(x)){
            if(length(x)>1){
                ellipsis$xlab <- colnames(x)[1];
                ellipsis$ylab <- colnames(x)[2];
                y <- x[[2]];
                x <- x[[1]];
                yIsProvided <- TRUE;
            }
            else{
                if(is.null(ellipsis$xlab)){
                    ellipsis$xlab <- colnames(x)[1];
                }
                x <- x[[1]];
                yIsProvided <- FALSE;
                y <- rep(0,length(x));
            }
        }
        else{
            yIsProvided <- FALSE;
            y <- rep(0,length(x));
        }
    }
    else{
        yIsProvided <- TRUE;
    }

    tableData <- tableDataInitial <- table(x,y);
    tableData[] <- tableData / sum(tableData);
    obsInsample <- sum(tableDataInitial);

    # An option - make the same density or make it changable
    # tableDataColours <- 1 - 0.75 * tableData / max(tableData);
    tableDataColours <- 1 - tableData;

    if(is.factor(x)){
        xUnique <- levels(x);
    }
    else{
        xUnique <- sort(unique(x));
    }
    if(is.factor(y)){
        yUnique <- levels(y)
    }
    else{
        yUnique <- sort(unique(y));
    }
    xCoord <- seq(0.5,length(xUnique)+0.5,length.out=length(xUnique)+1);
    yCoord <- seq(0.5,length(yUnique)+0.5,length.out=length(yUnique)+1);
    xMid <- xCoord[-1]-0.5;
    yMid <- yCoord[-1]-0.5;

    if(is.null(ellipsis$xlab)){
        xlab <- deparse(substitute(x));
        ellipsis$xlab <- "";
    }
    else{
        xlab <- ellipsis$xlab;
    }

    if(is.null(ellipsis$ylab)){
        ylab <- deparse(substitute(y));
        ellipsis$ylab <- "";
    }
    else{
        ylab <- ellipsis$ylab;
    }

    if(!is.null(ellipsis$axes)){
        axesValue <- ellipsis$axes;
    }
    else{
        axesValue <- TRUE;
    }
    ellipsis$axes <- FALSE;

    ellipsis$xlim <- range(xCoord);
    ellipsis$ylim <- range(yCoord);
    ellipsis$col <- "white";
    ellipsis$x <- 0;
    ellipsis$y <- 0;

    if(is.null(ellipsis$main)){
        mar1 <- c(5.1,4.1,1.1,1.1);
        mar2 <- c(0,0,0.1,2.1);
        ellipsis$main <- "";
    }
    else{
        mar1 <- c(5.1,4.1,4.1,1.1);
        mar2 <- c(0,0,2.1,2.1);
    }

    # Define palette
    paletteBasic <- paletteDetector(c("black","black"));

    # Create gradients of different insensitivity
    paletteBasicCol <- colorRampPalette(c("white",paletteBasic[2]))(1000)[findInterval(t(tableData),
                                                                                       seq(0, 1, length.out=1000))];

    if(legend){
        parDefault <- par(no.readonly=TRUE);
        on.exit(par(parDefault));
        colPalette <- colorRampPalette(c("white",paletteBasic[2]));
        layout(matrix(1:2,1,2),widths=c(0.9,0.1));
        par(mar=mar1);
    }

    do.call(plot, ellipsis);
    k <- 1;
    for(i in 1:(length(xCoord)-1)){
        for(j in 1:(length(yCoord)-1)){
            polygon(c(c(xCoord[i],xCoord[i+1]),rev(c(xCoord[i],xCoord[i+1]))),
                    c(c(yCoord[j],yCoord[j]),c(yCoord[j+1],yCoord[j+1])),
                    col=paletteBasicCol[k]);
            k <- k+1;

            # Colour for text and points
            if(tableData[i,j]>0.5){
                colAdditional <- "white";
            }
            else{
                colAdditional <- paletteBasic[1];
            }
            if(labels){
                text(xMid[i],yMid[j],labels=round(tableData[i,j],5),col=colAdditional);
            }
            if(points){
                if(tableDataInitial[i,j]!=0){
                    # points(rep(xCoord[i]+0.1,tableDataInitial[i,j]),
                    #        yCoord[j]+0.1+c(1:tableDataInitial[i,j])/(tableDataInitial[i,j])*0.75)
                    points(rep(xCoord[i]+0.2,tableDataInitial[i,j]),
                           yCoord[j]+0.5+c(1:tableDataInitial[i,j]-0.5-tableDataInitial[i,j]/2)/(tableDataInitial[i,j])*0.75,
                           col=colAdditional)
                }
            }
        }
    }
    if(axesValue){
        axis(1, at=xMid, labels=xUnique);
        if(yIsProvided){
            axis(2, at=yMid, labels=yUnique);
        }
    }
    box();
    title(xlab=xlab);
    if(yIsProvided){
        title(ylab=ylab);
    }

    if(legend){
        par(mar=mar2);
        xl <- 1
        yb <- 1.25
        xr <- 1.25
        yt <- 1.75

        plot(NA,type="n",ann=FALSE,xlim=c(1,1.5),ylim=c(1,2),xaxt="n",yaxt="n",bty="n");
        rect(xl, head(seq(yb,yt,(yt-yb)/5),-1), xr, tail(seq(yb,yt,(yt-yb)/5),-1), col=colPalette(5));

        mtext(0:4/4,side=4,at=tail(seq(yb,yt,(yt-yb)/5),-1)-0.05,las=2);
    }
}
