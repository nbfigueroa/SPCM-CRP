library(lda)

doc.lhood <- function(docs, lambda)
{
  if (is.null(dim(docs)))
    return(exch.dirichlet.lhood(docs, lambda))
  else
    return(exch.dirichlet.lhood(colSums(docs), lambda))
}


doc.lhood.fn <- function(lambda) function (dat) doc.lhood(dat, lambda)


heldout.doc.lhood <- function(doc, dists, alpha, eta, post.dir, decay.fn, state)
{
  # for each doc compute the log prior probability of each component
  log.prior <- safelog(decay.fn(dists))
  log.prior[length(log.prior)+1] <- log(alpha)
  log.prior <- log.prior - log.sum(log.prior)

  # for each doc compute the probability of the document under each component
  log.like <- apply(post.dir, 1, function (a) dirichlet.lhood(doc, a))
  names(log.like) <- rownames(post.dir)
  log.like <- log.like[char(state$cluster)]
  log.like[length(log.like) + 1] <- exch.dirichlet.lhood(doc, eta)

  # return the sum
  sum(log.prior + log.like)
}


compute.posterior.dirichlets <- function(dat, state, eta)
{
  comps <- sort(unique(state$cluster))
  post.dir <- laply(comps, function (k) colsum(dat[state$cluster==k,]) + eta)
  rownames(post.dir) <- comps
  colnames(post.dir) <- colnames(dat)
  post.dir
}

heldout.lhoods <- function(dat.ho, ho.idx, dat.obs, map.state,
                           dist.fn, decay.fn, alpha, lambda)
{
  one.doc <- function (doc, idx)
  {
    msg(sprintf("computing likelihood for doc %d", idx))
    dists <- laply(seq(1, dim(dat.obs)[1]), function (i) dist.fn(idx, i))
    heldout.doc.lhood(doc, dists, alpha, lambda, post.dir, decay.fn, map.state)
  }
  post.dir <- compute.posterior.dirichlets(dat.obs, map.state, lambda)
  stopifnot(dim(dat.ho)[1] == length(ho.idx))
  laply(1:length(ho.idx), function (i) one.doc(dat.ho[i,], ho.idx[i]))
}



# read links

read.links <- function (filename)
{
  links <- scan(filename, what = "", sep = "\n")
  links <- strsplit(links, " ", fixed = TRUE)
  links <- llply(links, function (x) as.integer(x) + 1)

  dist <- Matrix(0, nrow=length(links), ncol=length(links), sparse=T)
  for (i in seq(1, length(links))) {
    for (j in links[[i]]) {
      dist[i, j] <- 1
    }
  }
  dist
}


# here, corpus comes from read.documents in the lda package

corpus.to.matrix <- function(corpus, vocab, n=length(corpus))
{
  out.inc <- 10^floor(log10(n))
  m <- Matrix(0, nrow=min(length(corpus), n), ncol=length(vocab), sparse=T)
  for (i in 1:min(length(corpus), n))
  {
    if ((i %% out.inc) == 0) print(i);
    nw <- length(corpus[[i]][1,])
    if (nw > 0)
    {
      for (w in 1:nw)
      {
        m[i, corpus[[i]][1,w] + 1] <- corpus[[i]][2,w];
      }
    }
  }
  colnames(m) <- vocab
  m
}



# examples

network.example <- function ()
{
  docs <- read.documents("~/slap/data/cora/documents")
  voc <- readLines("~/slap/data/cora/lexicon")
  adj <- read.links("~/slap/data/cora/links")

  dist <- link.dist.fn(adj)
  dat <- corpus.to.matrix(docs, voc)
  lhood.fn <- doc.lhood.fn(0.5)

  # note: this has the effect of only considering docs that are linked
  decay.fn <- window.decay(2)
  summary.fn <- ncomp.summary

  res <- ddcrp.gibbs(dat=dat[1:100,], alpha=1, dist.fn=dist, decay.fn=decay.fn,
                     lhood.fn=lhood.fn, niter=5, summary.fn = ncomp.summary)

  # example: held out likelihood on 100th document

  post.dir <- compute.posterior.dirichlets(dat, res$map.state, 0.5)
  dists <- laply(1:dim(dat)[1], function (i) dist(100, i))
  heldout.doc.lhood(dat[100,], dists, 1, 0.5, post.dir, decay.fn, map.state)
}

sequential.example <- function ()
{
  library(lda)

  docs <- read.documents("~/data/science/data/sci90/sci90-mult.dat")
  voc <- readLines("~/data/science/data/sci90/sci90-vocab.dat")
  dat <- corpus.to.matrix(docs[1:500], voc)

  res <- ddcrp.gibbs(dat=dat[1:100,], dist.fn=seq.dist, alpha=1,
                     decay.fn=window.decay(100),
                     doc.lhood.fn(0.5), 5, summary.fn = ncomp.summary)
}
