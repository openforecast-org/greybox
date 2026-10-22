#' Dummy variables for provided seasonality type
#'
#' Function generates the matrix of dummy variables for the months / weeks / days /
#' hours / minutes / seconds of year / month / week / day / hour / minute.
#'
#' The function extracts dates from the provided object and returns a matrix with
#' dummy variables for the specified frequency type, with the number of rows equal
#' to the length of the object + the specified horizon. If a numeric vector is provided
#' then it will produce dummies based on typical values (e.g. 30 days in month). So it
#' is recommended to use proper classes with this method.
#'
#' Several notes on how the dummies are calculated in some special cases:
#' \itemize{
#' \item In case of weeks of years, the first week is defined according to ISO 8601.
#' }
#'
#' Note that not all the combinations of \code{type} and \code{of} are supported. For
#' example, there is no such thing as dummies for months of week. Also note that some
#' combinations are not very useful and would take a lot of memory (e.g. minutes of year).
#'
#' The function will return all the dummy variables. If you want to avoid the dummy
#' variables trap, you will need to exclude one of them manually.
#'
#' If you want to have a different type of dummy variables, let me know, I will
#' implement it.
#'
#' @param object Either a ts / msts / zoo / xts / tsibble object or a vector
#' of dates.
#' @param type Specifies what type of frequency to produce. For example, if
#' \code{type="month"}, then the matrix with dummies for months of the year will
#' be created.
#' @param of Specifies the frequency of what is needed. Used together with \code{type}
#' e.g. \code{type="day"} and \code{of="month"} will produce a matrix with dummies
#' for days of month (31 dummies).
#' @param h If not \code{NULL}, then the function will produce dummies for this
#' set of observations ahead as well, binding them to the original matrix.
#' @param factors If \code{TRUE}, the function will return the categorical variable
#' instead of the matrix with dummies.
#' @param ... Other parameters.
#'
#' @return One of the two is returned, depending on the value of \code{factors} variable:
#' \itemize{
#' \item \code{factors=FALSE}: A matrix with all the dummy variables, a time series
#' matrix (class "ts") in case of a "ts" object. Feel free to drop one of them, making it
#' the reference variable.
#' \item \code{factors=TRUE}: The categorical variable (factor) containing specific values
#' for each observation.
#' }
#'
#' @template author
#' @template keywords
#'
#' @seealso \code{\link[greybox]{xregExpander}, \link[greybox]{xregMultiplier},
#' \link[greybox]{outlierdummy}}
#'
#' @examples
#' # Generate matrix with dummies for a ts object
#' x <- ts(rnorm(100,100,1),frequency=12)
#' temporaldummy(x)
#'
#' # Generate matrix with monthly dummies for a zoo object
#' x <- as.Date("2003-01-01")+0:99
#' temporaldummy(x, type="month", of="year", h=10)
#'
#' @rdname temporaldummy
#' @importFrom stats cycle
#' @export
temporaldummy <- function(object, ...) UseMethod("temporaldummy")

#' @rdname temporaldummy
#' @export
temporaldummy.default <- function(object, type=c("month","quarter","week","day","hour","halfhour","minute","second"),
                            of=c("year","quarter","month","week","day","hour","minute"), factors=FALSE, h=0, ...){
    type <- match.arg(type);
    of <- match.arg(of);

    # All the observations to produce
    obsAll <- length(object)+h;

    # Define frequency (we don't know it in case of default class)
    dataFrequency <- switch(type, "quarter"=4,
                            "month"=switch(of, "year"=12, 3),
                            "week"=switch(of, "year"=52, "quarter"=13, 4),
                            "day"=switch(of, "year"=365, "quarter"=90, "month"=30, 7),
                            "hour"=switch(of, "year"=8760, "quarter"=2160, "month"=720,
                                          "week"=168, 24),
                            "halfhour"=switch(of, "year"=17520, "quarter"=4320, "month"=1440,
                                              "week"=336, 48),
                            "minute"=switch(of, "year"=525600, "quarter"=129600, "month"=43200,
                                            "week"=10080, "day"=1440, 60),
                            "second"=switch(of, "year"=31536000, "quarter"=7776000, "month"=2592000,
                                            "week"=604800, "day"=86400, "hour"=3600, 60));

    # Create factors
    factorVariable <- factor(rep(c(1:dataFrequency),ceiling(obsAll/dataFrequency))[1:obsAll],
                             levels=c(1:dataFrequency));

    if(!factors){
        # Do model matrix for sparse factors
        temporaldummy <- model.matrix(~factorVariable-1);
        colnames(temporaldummy) <- paste0(type,c(1:dataFrequency),"of",of);
        return(temporaldummy);
    }
    else{
        return(factorVariable);
    }

}

#' @rdname temporaldummy
#' @export
temporaldummy.ts <- function(object, type=c("month","quarter","week","day","hour","halfhour","minute","second"),
                             of=c("year","quarter","month","week","day","hour","minute"), factors=FALSE, h=0, ...){

    # Define frequency (we don't know it in case of default class)
    dataFrequency <- frequency(object);

    type <- switch(as.character(dataFrequency), "12"="month", "4"="quarter", "52"="week", "7"="day", "24"="hour",
                   "48"="halfhour", "60"="minute", "second");
    of <- switch(as.character(dataFrequency), "52"=, "12"=, "4"="year", "7"="week", "24"=,
                 "48"="day", "60"="hour", "minute");

    # Start the dummies from the position of the first observation in the cycle
    obsStart <- cycle(object)[1]-1;
    temporaldummy <- temporaldummy.default(c(rep(NA,obsStart),object), type=type, of=of, factors=factors, h=h);
    if(factors){
        temporaldummy <- temporaldummy[obsStart+1:(length(object)+h)];
    }
    else{
        temporaldummy <- temporaldummy[obsStart+1:(length(object)+h),,drop=FALSE];
    }

    return(ts(as.matrix(temporaldummy), start=start(object), frequency=dataFrequency));
}

#' @rdname temporaldummy
#' @export
temporaldummy.Date <- function(object, type=c("month","quarter","week","day","hour","halfhour","minute","second"),
                         of=c("year","quarter","month","week","day","hour","minute"), factors=FALSE, h=0, ...){
    type <- match.arg(type);
    of <- match.arg(of);

    # All the observations to produce
    obsAll <- length(object)+h;

    # Change the length of object if h>1
    if(h>0){
        object <- c(object,tail(object,1)+c(1:h)*diff(tail(object,2)));
    }

    # Vectors with dates / times of year / month / week etc
    if(type=="quarter"){
        dateFinal <- quarters(object);
    }
    else if(type=="month"){
        dateFinal <- dateMonthsOfYear <- months(object, abbreviate=TRUE);
        if(of=="quarter"){
            # First month of each quarter
            dateFinal[dateMonthsOfYear %in% c("Jan","Apr","Jul","Oct")] <- 1;
            dateFinal[dateMonthsOfYear %in% c("Feb","May","Aug","Nov")] <- 2;
            dateFinal[dateMonthsOfYear %in% c("Mar","Jun","Sep","Dec")] <- 3;
        }
    }
    else if(type=="week"){
        # Weeks, starting from 1 according to ISO 8601
        dateFinal <- as.numeric(strftime(object, format="%V"));

        if(of!="year"){
            warning(paste0("The option of=",of," is not yet implemented. Returning weeks of year instead."),
                    call.=FALSE);
        }
        # if(of=="quarter"){
        #     # 1 - 31 day of each Month
        #     # dateDayOfMonth <- as.numeric(strftime(object, format="%e"));
        #     for(i in 1:31){
        #         dateFinal[dateMonthsOfYear %in% c("Jan","Apr","Jul","Oct")] <- i;
        #     }
        # }
        # else if(of=="month"){
        #     dateFinal[]
        # }
        #
        # dateWeeks <- as.POSIXlt(object)$mday;
        # dateDayOfWeek <- weekdays(object, abbreviate=TRUE);
        # dateDayOfWeek <- as.numeric(strftime(object, format="%u"));
    }
    else if(type=="day"){
        if(of=="year"){
            dateFinal <- as.numeric(strftime(object, format="%j"));
        }
        else if(of=="quarter"){
            # Months of years
            dateMonthsOfYear <- months(object, abbreviate=TRUE);

            # Calculate number of days in each previous month
            dateYears <- as.numeric(strftime(object, format="%Y"));
            monthsOfYearNumber <- as.numeric(strftime(object, format="%m")) - 1;
            monthsOfYearNumber[monthsOfYearNumber==0] <- 12;
            ndaysInMonths <- as.numeric(strftime(as.Date(paste(dateYears + monthsOfYearNumber %/% 12,
                                                               monthsOfYearNumber %% 12 + 1, "01", sep="-"))-1,"%d"));

            # Extract days of months
            dateFinal <- as.numeric(strftime(object, format="%d"));
            # count number of days since the Jan / Apr / Jul / Oct
            dateFinal[dateMonthsOfYear %in% c("Feb","May","Aug","Nov")] <-
                ndaysInMonths[dateMonthsOfYear %in% c("Feb","May","Aug","Nov")] +
                dateFinal[dateMonthsOfYear %in% c("Feb","May","Aug","Nov")];

            # Amend the number of days, so that it counts both previous months
            ndaysInMonths[] <- ndaysInMonths +
                as.numeric(strftime(as.Date(paste(dateYears, monthsOfYearNumber, "01", sep="-"))-1,"%d"));
            # count number of days since the Jan / Apr / Jul / Oct
            dateFinal[dateMonthsOfYear %in% c("Mar","Jun","Sep","Dec")] <-
                ndaysInMonths[dateMonthsOfYear %in% c("Mar","Jun","Sep","Dec")] +
                dateFinal[dateMonthsOfYear %in% c("Mar","Jun","Sep","Dec")];
        }
        else if(of=="month"){
            dateFinal <- as.numeric(strftime(object, format="%d"));
        }
        else if(of=="week"){
            dateFinal <- weekdays(object, abbreviate=TRUE);
        }
    }
    else if(any(type==c("hour","halfhour","minute","second"))){
        warning("The Date class does not support hours, minutes and seconds. Use POSIXt instead.", call.=FALSE);
        return(NULL);
    }

    # Create factors
    factorVariable <- factor(dateFinal[1:obsAll]);

    if(!factors){
        # The dummies of all the observed values, even if there is only one
        temporaldummy <- diag(nlevels(factorVariable))[as.integer(factorVariable),,drop=FALSE];
        colnames(temporaldummy) <- paste0(type,levels(factorVariable),"of",of);
        return(temporaldummy);
    }
    else{
        return(factorVariable);
    }
}

#' @rdname temporaldummy
#' @export
temporaldummy.POSIXt <- function(object, type=c("month","quarter","week","day","hour","halfhour","minute","second"),
                         of=c("year","quarter","month","week","day","hour","minute"), factors=FALSE, h=0, ...){
    type <- match.arg(type);
    of <- match.arg(of);

    # All the observations to produce
    obsAll <- length(object)+h;

    # Change the length of object if h>1
    if(h>0){
        object <- c(object,tail(object,1)+c(1:h)*diff(tail(object,2)));
    }

    # Invoke the respective class for the Date
    if(any(type==c("month","quarter","week","day"))){
        # The dates of the clock time, which is already extended to the h observations
        return(temporaldummy(as.Date(format(object, format="%Y-%m-%d")), type=type, of=of, factors=factors, h=0));
    }
    else{
        dateHours <- as.numeric(format(object, format="%H"));
        dateMinutes <- as.numeric(format(object, format="%M"));
        # Days since the start of the year / quarter / month / week of each observation
        if(any(of==c("year","quarter","month"))){
            dateMonths <- as.numeric(format(object, format="%m"));
            dateMonthsStart <- switch(of, "year"=1, "quarter"=(dateMonths-1)%/%3*3+1, "month"=dateMonths);
            dateDaysOf <- as.numeric(as.Date(format(object, format="%Y-%m-%d")) -
                                         as.Date(paste(format(object, format="%Y"), dateMonthsStart, "01", sep="-")));
        }
        else{
            dateDaysOf <- switch(of, "week"=as.numeric(format(object, format="%u"))-1, 0);
        }

        if(type=="hour"){
            if(any(of==c("year","quarter","month","week"))){
                dateFinal <- dateDaysOf*24 + dateHours + 1;
            }
            else if(of=="day"){
                dateFinal <- dateHours;
            }
        }
        else if(type=="halfhour"){
            stop("Sorry, but the halfhour option is not available yet", call.=FALSE);
        }
        else if(type=="minute"){
            if(any(of==c("year","quarter","month","week","day"))){
                dateFinal <- dateDaysOf*1440 + dateHours*60 + dateMinutes + 1;
            }
            else if(of=="hour"){
                dateFinal <- dateMinutes;
            }
        }
        # Else... this is seconds
        else{
            if(of=="minute"){
                    dateFinal <- as.numeric(format(object, format="%S"));
            }
            else{
                stop(paste0("Sorry, but the seconds option is not available yet for the ",of),
                     call.=FALSE);
            }
        }
    }

    # Create factors
    factorVariable <- factor(dateFinal[1:obsAll]);

    if(!factors){
        # The dummies of all the observed values, even if there is only one
        temporaldummy <- diag(nlevels(factorVariable))[as.integer(factorVariable),,drop=FALSE];
        colnames(temporaldummy) <- paste0(type,levels(factorVariable),"of",of);
        return(temporaldummy);
    }
    else{
        return(factorVariable);
    }
}

#' @rdname temporaldummy
#' @export
temporaldummy.zoo <- function(object, type=c("month","quarter","week","day","hour","halfhour","minute","second"),
                         of=c("year","quarter","month","week","day","hour","minute"), factors=FALSE, h=0, ...){
    type <- match.arg(type);
    of <- match.arg(of);

    dates <- time(object);

    # Change the length of object if h>1
    if(h>0){
        dates <- c(dates,tail(dates,1)+c(1:h)*diff(tail(dates,2)));
    }

    # Invoke the respective class for the date / time
    return(temporaldummy(dates, type=type, of=of, factors=factors, h=0));
}
