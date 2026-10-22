context("Tests for alm() function");

# A provided occurrence model is used via its fitted values, whatever its response is called
test_that("alm() and stepwise() with an occurrence model on a differently named response", {
    set.seed(41);
    x <- rnorm(100);
    y <- rpois(100, 2) * (runif(100) > 0.4);
    occurrenceModel <- alm(y~x, data.frame(y=y, x=x), distribution="plogis");
    sizesSame <- alm(y~x, data.frame(y=y, x=x), occurrence=occurrenceModel);
    sizesOther <- alm(z~x, data.frame(z=y, x=x), occurrence=occurrenceModel);
    expect_equal(unname(coef(sizesOther)), unname(coef(sizesSame)));
    expect_equal(logLik(sizesOther), logLik(sizesSame));
    expect_s3_class(stepwise(data.frame(errors=y, x=x, x2=rnorm(100)),
                             occurrence=occurrenceModel), "alm");
})

test_that("forecast() of an ARIMA alm() without explanatory variables", {
    y <- c(2.1, 3.4, 3.0, 4.2, 5.1, 4.8, 5.5, 6.0, 5.8, 6.5, 7.1, 6.8, 7.5, 8.0, 7.8);
    testModel <- alm(y ~ 1, data.frame(y=y), distribution="dnorm", orders=c(1,0,0));
    testForecast <- suppressWarnings(forecast(testModel, h=3, interval="prediction", level=0.9));
    testPredict <- predict(testModel, data.frame(y=rep(NA, 3)), interval="prediction", level=0.9);
    expect_equal(testForecast$mean, testPredict$mean);
    expect_equal(testForecast$lower, testPredict$lower);
    expect_equal(testForecast$upper, testPredict$upper);
})
