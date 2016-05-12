source("decay.R")
source("helper.R")

library(plyr)

# returns customers that are connected to i, directly or indirectly

connections <- function(i, links)
{
  visited <- c()
  to.visit <- c(i)
  while (length(to.visit) > 0) {
    curr <- to.visit[1]
    visited <- c(visited, curr)
    to.visit <- to.visit[-1]
    pointers <- which(links == curr)
    for (p in pointers) {
      if (!(p %in% visited))
        to.visit <- c(to.visit, p)
    }
  }
  visited
}


# example ddcrp summary functions

ncomp.summary <- function(dat, iter, state, lhood, alpha)
{
  c(iter = iter, ncomp = length(unique(state$cluster)))
}

crp.score.summary <- function(dat, iter, state, lhood, alpha)
{
  c(iter = iter,
    ncomp = length(unique(state$cluster)),
    crp.score = (crp.log.prior(alpha, state$cluster) + sum(lhood)))
}


# ddcrp sampler

ddcrp.gibbs <- function(dat, alpha, dist.fn, decay.fn, lhood.fn,
                        niter, summary.fn = ncomp.summary,
                        log.prior.thresh=-10,
                        clust.traj=FALSE, cust.traj=FALSE)
{

  ### set up summary statistics and trajectories

  ndata <- dim(dat)[1]
  msg.inc <- 10^(floor(log10(dim(dat)[1]))-1)
  if (clust.traj)
    clust.traj <- matrix(NA, nrow=niter, ncol=ndata)
  if (cust.traj)
    cust.traj <- matrix(NA, nrow=niter, ncol=ndata)
  score <- numeric(niter)
  map.score <- 0

  ### set up initial state, summaries, and cluster likelihoods

  msg("setting up the initial state")
  st <- data.frame(idx=1:ndata, cluster=1:ndata, customer=1:ndata)
  lhood <- daply(st, .(cluster), function (df) lhood.fn(dat[df$idx,]))

  summary <- summary.fn(dat, 0, st, lhood, alpha)

  ### run for niter iterations

  for (iter in 1:niter)
  {
    msg(sprintf("iter=%d", iter))

    iter.score <- 0
    for (i in seq(2,ndata)) # note: index i = 1 is correct at the outset
    {
      if ((i %% msg.inc) == 0) msg(sprintf("%04d", i))

      ### "remove" the i-th data point from the state
      ### to do this, set its cluster to i, and set its connected data to i

      old.cluster <- st$cluster[i]
      old.customer <- st$customer[i]
      conn.i <- connections(i, st$customer)
      st$cluster[conn.i] <- i
      st$customer[i] <- i

      ### if this removal splits a table update the likelihoods.
      ### note: splits only happen if c_i^{old} != i

      if (old.customer != i)
      {
        ### !!! do we need to use "idx"
        old.idx <- st[which(st$cluster==old.cluster),"idx"]
        lhood[char(old.cluster)] <- lhood.fn(dat[old.idx,])
      }
      else
      {
        lhood[char(old.cluster)] <- 0
      }

      ### compute the log prior
      ### (this should be precomputed---see opt.ddcrp.gibbs below)

      log.prior <- sapply(1:ndata,
                          function (j) safelog(decay.fn(dist.fn(i, j))))
      log.prior[i] <- log(alpha)
      log.prior <- log.prior - log.sum(log.prior)
      cand.links <- which(log.prior > log.prior.thresh)

      ### compute the likelihood of data point i (and its connectors)
      ### with all other tables (!!! do we need to use "idx"?)

      cand.clusts <- unique(st$cluster[cand.links])

      new.lhood <- daply(subset(st, cluster %in% cand.clusts), .(cluster),
                         function (df)
                         lhood.fn(dat[unique(c(df$idx,st[conn.i,"idx"])),]))

      if (length(new.lhood)==1) names(new.lhood) <- cand.clusts

      ### set up the old likelihoods

      old.lhood <- lhood[char(cand.clusts)]
      sum.old.lhood <- sum(old.lhood)

      ### compute the conditional distribution

      log.prob <-
        log.prior[cand.links] +
          sapply(cand.links,
                 function (j) {
                   c.j <- char(st$cluster[j])
                   sum.old.lhood - old.lhood[c.j] + new.lhood[c.j] })

      ### sample from the distribution

      prob <- exp(log.prob - log.sum(log.prob))
      if (length(prob)==1)
        st$customer[i] <- cand.links[1]
      else
        st$customer[i] <- sample(cand.links, 1, prob=prob)

      ### update the score with the prior and update the clusters

      iter.score <- iter.score + log.prior[st$customer[i]]
      st$cluster[conn.i] <- st$cluster[st$customer[i]]
      clust.i.idx <- subset(st, cluster == st$cluster[i])$idx
      lhood[char(st$cluster[i])] <- lhood.fn(dat[clust.i.idx,])
    }

    ### update the summary

    iter.score <- iter.score + sum(lhood)
    score[iter] <- iter.score
    if ((score[iter] > map.score) || (iter==1))
    {
      map.score <- score[iter]
      map.state <- st
    }
    summary <- rbind(summary, summary.fn(dat, iter, st, lhood, alpha))
    if (!is.null(dim(cust.traj))) cust.traj[iter,] <- st$customer
    if (!is.null(dim(clust.traj))) clust.traj[iter,] <- st$cluster
  }

  ### return everything

  list(summary=summary, cust.traj=cust.traj, clust.traj=clust.traj, score=score,
       map.score = map.score, map.state = map.state)
}


# this is an optimized version of ddcrp gibbs.  the prior is
# precomputed.

opt.ddcrp.gibbs <- function(dat, alpha, dist.fn, decay.fn, lhood.fn,
                            niter, summary.fn = ncomp.summary,
                            log.prior.thresh=-10,
                            clust.traj=FALSE, cust.traj=FALSE)
{

  ### set up summary statistics and trajectories

  ndata <- dim(dat)[1]
  msg.inc <- 10^(floor(log10(dim(dat)[1]))-1)
  if (clust.traj)
    clust.traj <- matrix(NA, nrow=niter, ncol=ndata)
  if (cust.traj)
    cust.traj <- matrix(NA, nrow=niter, ncol=ndata)
  score <- numeric(niter)
  map.score <- 0

  ### set up initial state, summaries, and cluster likelihoods

  msg("setting up the initial state")
  st <- data.frame(idx=1:ndata, cluster=1:ndata, customer=1:ndata)
  lhood <- daply(st, .(cluster), function (df) lhood.fn(dat[df$idx,]))

  summary <- summary.fn(dat, 0, st, lhood, alpha)

  ### precompute log.prior and candidate links

  msg("precomputing log prior")
  log.prior.lst <- list()
  cand.links.lst <- list()
  for (i in seq(2, ndata))
  {
    msg(sprintf("document %d", i))
    log.prior.i <- sapply(1:ndata,
                        function (j) safelog(decay.fn(dist.fn(i, j))))
    log.prior.i[i] <- log(alpha)
    log.prior.i <- log.prior.i - log.sum(log.prior.i)
    cand.links.i <- which(log.prior.i > log.prior.thresh)
    log.prior.i <- log.prior.i[cand.links.i]
    log.prior.lst[[i]] <- log.prior.i
    cand.links.lst[[i]] <- cand.links.i
  }
  msg("done")

  ### run for niter iterations

  for (iter in 1:niter)
  {
    msg(sprintf("iter=%d", iter))

    iter.score <- 0
    for (i in seq(2,ndata)) # note: index i = 1 is correct at the outset
    {
      if ((i %% msg.inc) == 0) msg(sprintf("%04d", i))

      ### "remove" the i-th data point from the state
      ### to do this, set its cluster to i, and set its connected data to i

      old.cluster <- st$cluster[i]
      old.customer <- st$customer[i]
      conn.i <- connections(i, st$customer)
      st$cluster[conn.i] <- i
      st$customer[i] <- i

      ### if this removal splits a table update the likelihoods.
      ### note: splits only happen if c_i^{old} != i

      if (old.customer != i)
      {
        ### !!! do we need to use "idx"
        old.idx <- st[which(st$cluster==old.cluster),"idx"]
        lhood[char(old.cluster)] <- lhood.fn(dat[old.idx,])
      }
      else
      {
        lhood[char(old.cluster)] <- 0
      }

      ### get log prior

      log.prior <- log.prior.lst[[i]]
      cand.links <- cand.links.lst[[i]]

      ### compute the likelihood of data point i (and its connectors)
      ### with all other tables (!!! do we need to use "idx"?)

      cand.clusts <- unique(st$cluster[cand.links])

      new.lhood <- daply(subset(st, cluster %in% cand.clusts), .(cluster),
                         function (df)
                         lhood.fn(dat[unique(c(df$idx,st[conn.i,"idx"])),]))

      if (length(new.lhood)==1) names(new.lhood) <- cand.clusts

      ### set up the old likelihoods

      old.lhood <- lhood[char(cand.clusts)]
      sum.old.lhood <- sum(old.lhood)

      ### compute the conditional distribution

      log.prob <-
        log.prior +
          sapply(cand.links,
                 function (j) {
                   c.j <- char(st$cluster[j])
                   sum.old.lhood - old.lhood[c.j] + new.lhood[c.j] })

      ### sample from the distribution

      prob <- exp(log.prob - log.sum(log.prob))
      if (length(prob)==1)
        st$customer[i] <- cand.links[1]
      else
        st$customer[i] <- sample(cand.links, 1, prob=prob)

      ### update the score with the prior and update the clusters

      iter.score <- iter.score + log.prior[which(cand.links == st$customer[i])]
      st$cluster[conn.i] <- st$cluster[st$customer[i]]
      clust.i.idx <- subset(st, cluster == st$cluster[i])$idx
      lhood[char(st$cluster[i])] <- lhood.fn(dat[clust.i.idx,])
    }

    ### update the summary

    iter.score <- iter.score + sum(lhood)
    score[iter] <- iter.score
    if ((score[iter] > map.score) || (iter==1))
    {
      map.score <- score[iter]
      map.state <- st
    }
    summary <- rbind(summary, summary.fn(dat, iter, st, lhood, alpha))
    if (!is.null(dim(cust.traj))) cust.traj[iter,] <- st$customer
    if (!is.null(dim(clust.traj))) clust.traj[iter,] <- st$cluster
  }

  ### return everything

  list(summary=summary, cust.traj=cust.traj, clust.traj=clust.traj, score=score,
       map.score = map.score, map.state = map.state)
}

