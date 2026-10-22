context("Tests for rmcb() function");

test_that("rmcb() returns a numeric p-value with an alm() distribution", {
    set.seed(41);
    x <- matrix(rnorm(300, 10, 1), 100, 3);
    for(distribution in c("dlaplace", "dlnorm", "dnorm", "tukey")){
        testResult <- rmcb(x, distribution=distribution, plottype="none");
        expect_false(inherits(testResult$p.value, "logLik"));
        expect_true(is.numeric(testResult$p.value) && is.null(attributes(testResult$p.value)));
    }
})
