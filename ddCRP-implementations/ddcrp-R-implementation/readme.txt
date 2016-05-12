

****************************
Gibbs sampling for the DDCRP

David Blei and Peter Frazier
****************************


This is an implementation of posterior inference for the ddCRP,
described in Blei and Frazier (2010).  This code comes with no
guarantees.

The way it works is as follows.

First set up a likelihood function, e.g.,

      lhood.fn <- doc.lhood.fn(lambda)

This is a function that takes a vector as input and returns the log
likelihood.  In this example, the likelihood is an integrated
Dirichlet-multinomial.  The hyperparameter is the Dirichlet parameter.


Second, set up a decay function, e.g.,

      decay.fn <- window.decay(decay.param)
or
      decay.fn <- exp.decay(decay.param)

One special decay function sets up a ddCRP that is equivalent to a CRP,

      decay.fn <- window.decay(dim(docs)[1])


Third, set up the distance function, e.g., one of

    (a) dist.fn <- input.based.dist.fn(times, seq.dist)
    (b) dist.fn <- seq.dist
    (c) dist.fn <- link.dist.fn(links)

Example (c) is used in a network model; a function to read the links
is also provided.  In the CRP case, use seq.dist.


Finally, run the gibbs sampler, e.g.,

    ddcrp <- ddcrp.gibbs(dat=dat, alpha=alpha,
                         lhood.fn=lhood.fn,
                         dist.fn=dist.fn,
                         decay.fn=decay.fn,
                         niter=niter,
                         summary.fn=ncomp.summary,
                         log.prior.thresh = -10)

Here, "dat" is the data matrix (e.g., word counts) and alpha is a
scalar.  The summary.fn argument is run after each iteration.  It
should have the following signature

     foo <- function(dat, iter, state, lhood, alpha)

You have flexibility with how to define it.
