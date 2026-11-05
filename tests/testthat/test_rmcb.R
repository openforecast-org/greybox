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

test_that("rmcb() groups contain only the methods that overlap each other", {
    # The mean ranks are 1.8, 2 and 2.2 with the critical distance of 0.33:
    # the middle method overlaps both, but the other two differ.
    x <- rbind(matrix(c(1,2,3), 60, 3, byrow=TRUE),
               matrix(c(3,2,1), 40, 3, byrow=TRUE));
    colnames(x) <- c("A","B","C");
    testResult <- rmcb(x, outplot="none");
    expect_false(testResult$methods["A","C"]);
    expect_equal(unname(testResult$vlines), rbind(c(1,2),c(2,3)));

    # Every pair within a group overlaps, and every overlapping pair shares a group
    set.seed(41);
    x <- matrix(rnorm(1200), 100, 12);
    x <- x + matrix(seq(0, 0.6, length.out=12), 100, 12, byrow=TRUE);
    for(distribution in c("tukey", "dnorm", "dlnorm")){
        testResult <- rmcb(if(distribution=="dlnorm") exp(x) else x,
                           distribution=distribution, outplot="none");
        shared <- tcrossprod(testResult$groups + 0) > 0;
        expect_equivalent(shared, testResult$methods);
    }
})
