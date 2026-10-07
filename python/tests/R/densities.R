# The reference values of test_densities.py: R's log-densities of the
# Laplace, S and Generalised Normal distributions, those of their log-variants
# as alm() takes them, and gamma(), in hex to the bit.
# Run from the repository root with this branch's greybox installed:
#   Rscript python/tests/R/densities.R
library(greybox)
set.seed(9)
q <- c(rt(40, 3)*5, 0, 1e-300, 1e3)
shapes <- c(0.07, 0.3, 0.5, 0.93, 1, 1.37, 2, 2.6, 7.3, 31)
x <- c(1/seq(0.05, 20, length.out=40), runif(40, 0, 171), seq(10, 50, by=2), 12.3, 14.99)
y <- c(exp(rt(40, 3)*1.5 + 0.37), 1, 1e-300, 1e300)
perShape <- function(prefix, shapes, values){
    do.call(rbind, lapply(shapes, function(shape){
        data.frame(name=paste0(prefix, shape), value=values(shape))
    }))
}
rows <- rbind(data.frame(name="q", value=q),
              data.frame(name="dlaplace", value=dlaplace(q, 0.37, 1.913, log=TRUE)),
              data.frame(name="ds", value=ds(q, 0.37, 1.913, log=TRUE)),
              perShape("dgnorm", shapes, function(shape) dgnorm(q, 0.37, 1.913, shape, log=TRUE)),
              data.frame(name="x", value=x),
              data.frame(name="gamma", value=gamma(x)),
              # The log-variants, as alm() computes their log-likelihoods
              data.frame(name="y", value=y),
              data.frame(name="dllaplace", value=dlaplace(log(y), 0.37, 1.913, log=TRUE)-log(y)),
              data.frame(name="dls", value=ds(log(y), 0.37, 1.913, log=TRUE)-log(y)),
              perShape("dlgnorm", shapes, function(shape) dgnorm(log(y), 0.37, 1.913, shape, log=TRUE)-log(y)))
rows$value <- sprintf("%a", rows$value)
write.csv(rows, "python/tests/densities.csv", row.names=FALSE)
