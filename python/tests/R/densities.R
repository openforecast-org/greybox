# The reference values of test_densities.py: R's densities and log-densities of
# the Laplace, S, Generalised Normal and Gamma distributions, the log-densities
# of their log-variants as alm() takes them, and gamma(), in hex to the bit.
# Run from the repository root with this branch's greybox installed:
#   Rscript python/tests/R/densities.R
library(greybox)
set.seed(9)
q <- c(rt(40, 3)*5, 0, 1e-300, 1e3)
shapes <- c(0.07, 0.3, 0.5, 0.93, 1, 1.37, 2, 2.6, 7.3, 31)
x <- c(1/seq(0.05, 20, length.out=40), runif(40, 0, 171), seq(10, 50, by=2), 12.3, 14.99)
y <- c(exp(rt(40, 3)*1.5 + 0.37), 1, 1e-300, 1e300)
gammaShapes <- c(0.07, 0.4, 0.93, 1, 1.37, 2, 12.6, 33.3, 143, 10000)
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
              perShape("dlgnorm", shapes, function(shape) dgnorm(log(y), 0.37, 1.913, shape, log=TRUE)-log(y)),
              # The densities themselves
              data.frame(name="pdf_dlaplace", value=dlaplace(q, 0.37, 1.913)),
              data.frame(name="pdf_ds", value=ds(q, 0.37, 1.913)),
              perShape("pdf_dgnorm", shapes, function(shape) dgnorm(q, 0.37, 1.913, shape)))
# The Gamma distribution, with the quantiles drawn per shape
for(shape in gammaShapes){
    z <- c(rgamma(40, shape, scale=0.8), 1e-300, 1e-8, 1e3)
    rows <- rbind(rows,
                  data.frame(name=paste0("dgamma_q", shape), value=z),
                  data.frame(name=paste0("dgamma", shape), value=dgamma(z, shape, scale=0.8, log=TRUE)),
                  data.frame(name=paste0("pdf_dgamma", shape), value=dgamma(z, shape, scale=0.8)))
}
rows$value <- sprintf("%a", rows$value)
write.csv(rows, "python/tests/densities.csv", row.names=FALSE)
