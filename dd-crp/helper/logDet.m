function logdet = logDet (Sigma)
    logdet = 2*sum( log( diag( chol( Sigma ) ) ) );
end