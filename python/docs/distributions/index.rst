===================
Distributions Guide
===================

This section provides detailed information about all probability distributions
available in greybox.

Overview
--------

greybox provides a comprehensive set of probability distributions for modeling
different types of data. Each distribution family includes:

* **Density function** (``d*``) - Probability density function (PDF)
* **Cumulative distribution function** (``p*``) - CDF
* **Quantile function** (``q*``) - Inverse CDF
* **Random generator** (``r*``) - Random sample generation

Common Parameters
------------------

The arguments carry the names of the R functions: those of R's ``stats`` for
the base distributions (``dnorm(x, mean, sd)``, ``dlnorm(x, meanlog, sdlog)``,
...), of ``statmod`` for the inverse Gaussian and of R's greybox for its own
families (``dlaplace(q, mu, scale)``, ``dfnorm(q, mu, sigma)``, ...). R's
dotted names become underscored and R's ``lambda`` becomes ``lambda_``. The
older Python names (``loc``, ``scale``, ``log`` of the p functions, ...) still
work for one release, with a ``DeprecationWarning``.

* ``log`` : If True, the density functions return the log of the density
* ``lower_tail`` : If True (default), probabilities are P[X ≤ x]
* ``log_p`` : If True, probabilities are given or returned as log(p)

Continuous Univariate Distributions
-----------------------------------

Normal (Gaussian)
~~~~~~~~~~~~~~~~~

.. py:function:: dnorm(x, mean=0.0, sd=1.0, log=False)

   Normal (Gaussian) distribution density.

   :param x: Value(s) at which to evaluate
   :param mean: Mean (μ)
   :param sd: Standard deviation (σ)
   :param log: If True, return log density
   :return: Density at x

   **Example**::

      from greybox import dnorm
      dnorm(0, mean=0, sd=1)  # ~0.3989

Laplace
~~~~~~~

.. py:function:: dlaplace(q, mu=0, scale=1, log=False)

   Laplace (double exponential) distribution density.

   :param q: Value(s) at which to evaluate
   :param mu: Location parameter (median)
   :param scale: Scale parameter
   :param log: If True, return log density
   :return: Density at q

S Distribution
~~~~~~~~~~~~~~

.. py:function:: ds(q, mu=0, scale=1, log=False)

   S-distribution density - a heavy-tailed distribution.

   :param q: Value(s) at which to evaluate
   :param mu: Location parameter
   :param scale: Scale parameter
   :param log: If True, return log density
   :return: Density at q

Generalized Normal
~~~~~~~~~~~~~~~~~~

.. py:function:: dgnorm(q, mu=0, scale=1, shape=1, log=False)

   Generalized Normal distribution density.

   :param q: Value(s) at which to evaluate
   :param mu: Location parameter
   :param scale: Scale parameter
   :param shape: Shape parameter (controls tail weight)
                 - shape=1: Laplace
                 - shape=2: Normal
                 - shape<2: Heavy tails
                 - shape>2: Light tails
   :param log: If True, return log density
   :return: Density at q

   **Example**::

      from greybox import dgnorm
      # Normal-like
      dgnorm(0, mu=0, scale=1, shape=2)
      # Heavy-tailed
      dgnorm(5, mu=0, scale=1, shape=1)

Logistic
~~~~~~~~

.. py:function:: dlogis(x, location=0, scale=1, log=False)

   Logistic distribution density.

   :param x: Value(s) at which to evaluate
   :param location: Location parameter
   :param scale: Scale parameter
   :param log: If True, return log density
   :return: Density at x

Student's t
~~~~~~~~~~~

.. py:function:: dt(x, df, ncp=None, log=False, *, loc=0, scale=1)

   Student's t distribution density, noncentral with ``ncp``.

   :param x: Value(s) at which to evaluate
   :param df: Degrees of freedom
   :param ncp: Non-centrality parameter (None for the central t)
   :param log: If True, return log density
   :param loc: Location parameter (not in R)
   :param scale: Scale parameter (not in R)
   :return: Density at x

Asymmetric Laplace
~~~~~~~~~~~~~~~~~~

.. py:function:: dalaplace(q, mu=0, scale=1, alpha=0.5, log=False)

   Asymmetric Laplace distribution density for quantile regression.

   :param q: Value(s) at which to evaluate
   :param mu: Location parameter
   :param scale: Scale parameter
   :param alpha: Asymmetry parameter (0 < alpha < 1)
                 - alpha < 0.5: Right-skewed
                 - alpha > 0.5: Left-skewed
   :param log: If True, return log density
   :return: Density at q

Log-Transformed Distributions
------------------------------

These distributions model the log of the response variable.

Log-Normal
~~~~~~~~~~

.. py:function:: dlnorm(x, meanlog=0, sdlog=1, log=False)

   Log-Normal distribution density.

   :param x: Value(s) - must be positive
   :param meanlog: Mean of the underlying normal distribution (on log scale)
   :param sdlog: Standard deviation of the underlying normal distribution
   :param log: If True, return log density
   :return: Density at x

   **Note**: For positive-valued data that is right-skewed.

Log-Laplace
~~~~~~~~~~~

.. py:function:: dllaplace(q, mu=0, scale=1, log=False)

   Log-Laplace distribution density.

   :param q: Value(s) - must be positive
   :param mu: Location parameter (of log values)
   :param scale: Scale parameter
   :param log: If True, return log density
   :return: Density at q

Log-S
~~~~~

.. py:function:: dls(q, mu=0, scale=1, log=False)

   Log-S distribution density.

   :param q: Value(s) - must be positive
   :param mu: Location parameter (of log values)
   :param scale: Scale parameter
   :param log: If True, return log density
   :return: Density at q

Log-Generalized Normal
~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: dlgnorm(q, mu=0, scale=1, shape=1, log=False)

   Log-Generalized Normal distribution density.

   :param q: Value(s) - must be positive
   :param mu: Location parameter (of log values)
   :param scale: Scale parameter
   :param shape: Shape parameter
   :param log: If True, return log density
   :return: Density at q

Box-Cox Normal
~~~~~~~~~~~~~~

.. py:function:: dbcnorm(q, mu=0, sigma=1, lambda_bc=0, log=False)

   Box-Cox Normal distribution density.

   :param q: Value(s) - must be positive
   :param mu: Location parameter
   :param sigma: Scale parameter
   :param lambda_bc: Box-Cox transformation parameter
   :param log: If True, return log density
   :return: Density at q

Folded and Rectified Distributions
-----------------------------------

These distributions model positive-valued data with a point mass at zero.

Folded Normal
~~~~~~~~~~~~~~

.. py:function:: dfnorm(q, mu=0, sigma=1, log=False)

   Folded Normal distribution density.

   :param q: Value(s) - must be non-negative
   :param mu: Location parameter of underlying normal
   :param sigma: Scale parameter of underlying normal
   :param log: If True, return log density
   :return: Density at q

Rectified Normal
~~~~~~~~~~~~~~~~

.. py:function:: drectnorm(q, mu=0, sigma=1, log=False)

   Rectified Normal distribution density.

   :param q: Value(s) - must be non-negative
   :param mu: Location parameter of underlying normal
   :param sigma: Scale parameter of underlying normal
   :param log: If True, return log density
   :return: Density at q

Distributions for Positive Values
---------------------------------

Inverse Gaussian
~~~~~~~~~~~~~~~~

.. py:function:: dinvgauss(x, mean=1, dispersion=1, log=False)

   Inverse Gaussian distribution density, as ``statmod::dinvgauss``. The
   dispersion is the second parameter: statmod's ``shape`` is not taken.

   :param x: Value(s) - must be positive
   :param mean: Mean parameter
   :param dispersion: Dispersion parameter (1 / shape)
   :param log: If True, return log density
   :return: Density at x

Gamma
~~~~~

.. py:function:: dgamma(x, shape=1, scale=1, log=False)

   Gamma distribution density. Unlike R's, the second argument is the scale:
   R's ``rate`` is not taken.

   :param x: Value(s) - must be positive
   :param shape: Shape parameter (often denoted alpha)
   :param scale: Scale parameter (often denoted theta)
   :param log: If True, return log density
   :return: Density at x

Exponential
~~~~~~~~~~~

.. py:function:: dexp(x, rate=1, log=False, *, loc=0)

   Exponential distribution density.

   :param x: Value(s) - must be non-negative
   :param rate: Rate parameter (1/scale)
   :param log: If True, return log density
   :param loc: Location parameter (not in R)
   :return: Density at x

Chi-Squared
~~~~~~~~~~~

.. py:function:: dchisq(x, df, ncp=0, log=False)

   Chi-squared distribution density, noncentral with a nonzero ``ncp``.
   ``dchi2`` is its deprecated name.

   :param x: Value(s) - must be non-negative
   :param df: Degrees of freedom
   :param ncp: Non-centrality parameter
   :param log: If True, return log density
   :return: Density at x

Count Distributions
--------------------

Poisson
~~~~~~~

.. py:function:: dpois(x, lambda_, log=False)

   Poisson distribution probability mass function.

   :param x: Value(s) - must be non-negative integers
   :param lambda_: Mean parameter (R's ``lambda``)
   :param log: If True, return log probability
   :return: Probability mass at x

Negative Binomial
~~~~~~~~~~~~~~~~~

.. py:function:: dnbinom(x, mu=None, size=1, log=False, *, prob=None)

   Negative Binomial distribution probability mass function. The arguments
   keep the order (mu, size), where R's is (size, prob, mu).

   :param x: Value(s) - must be non-negative integers
   :param mu: Mean parameter (1 if neither ``mu`` nor ``prob`` is given)
   :param size: Dispersion parameter
   :param log: If True, return log probability
   :param prob: Probability of success in each trial, instead of ``mu``
   :return: Probability mass at x

Geometric
~~~~~~~~~

.. py:function:: dgeom(x, prob, log=False)

   Geometric distribution probability mass function.

   :param x: Value(s) - must be non-negative integers
   :param prob: Probability of success
   :param log: If True, return log probability
   :return: Probability mass at x

Binomial
~~~~~~~~

.. py:function:: dbinom(x, size, prob, log=False)

   Binomial distribution probability mass function.

   :param x: Value(s) - must be integers between 0 and size
   :param size: Number of trials
   :param prob: Probability of success
   :param log: If True, return log probability
   :return: Probability mass at x

Binary/Bounded Distributions
-----------------------------

Beta
~~~~

.. py:function:: dbeta(x, shape1, shape2, ncp=0, log=False)

   Beta distribution density, noncentral with a nonzero ``ncp``.

   :param x: Value(s) - must be in [0, 1]
   :param shape1: First shape parameter (alpha)
   :param shape2: Second shape parameter (beta)
   :param ncp: Non-centrality parameter
   :param log: If True, return log density
   :return: Density at x

Logit-Normal
~~~~~~~~~~~~

.. py:function:: dlogitnorm(q, mu=0, sigma=1, log=False)

   Logit-Normal distribution density.

   :param q: Value(s) - must be in (0, 1)
   :param mu: Mean of underlying normal (logit scale)
   :param sigma: Standard deviation (logit scale)
   :param log: If True, return log density
   :return: Density at q

CDF-Based Distributions
-----------------------

These distributions use the CDF in the likelihood.

Logistic CDF
~~~~~~~~~~~~

.. py:function:: plogis(q, location=0, scale=1, lower_tail=True, log_p=False)

   Logistic cumulative distribution function.

   :param q: Value(s) at which to evaluate
   :param location: Location parameter
   :param scale: Scale parameter
   :param lower_tail: If True, return P[X ≤ q]
   :param log_p: If True, return log probability
   :return: CDF value at q

Probit (Normal CDF)
~~~~~~~~~~~~~~~~~~~~

.. py:function:: pnorm(q, mean=0, sd=1, lower_tail=True, log_p=False)

   Normal cumulative distribution function.

   :param q: Value(s) at which to evaluate
   :param mean: Mean
   :param sd: Standard deviation
   :param lower_tail: If True, return P[X ≤ q]
   :param log_p: If True, return log probability
   :return: CDF value at q

Summary
-------

greybox supports the following distributions. For each distribution family, use the
appropriate prefix:

* ``d*`` - Probability density/mass function (PDF/PMF)
* ``p*`` - Cumulative distribution function (CDF)
* ``q*`` - Quantile function (inverse CDF)
* ``r*`` - Random number generation

The choice of distribution depends on the nature of your data:

* **Continuous, symmetric**: Normal, Laplace, Logistic, Student's t
* **Heavy tails**: Laplace, S, Generalized Normal, Student's t
* **Positive, right-skewed**: Log-Normal, Gamma, Inverse Gaussian
* **Count data**: Poisson, Negative Binomial, Geometric, Binomial
* **Proportions**: Beta, Logit-Normal
* **Zero-inflated**: Folded Normal, Rectified Normal
