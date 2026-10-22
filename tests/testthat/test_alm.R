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
