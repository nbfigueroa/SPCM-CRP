function V = mvgammaln(d, X)
% Evaluate Multivariate Log-Gamma function
%
%   In math, the multivariate gamma function with dim d, is defined by
%
%       Gamma_d(x) = pi^(d * (d-1) / 4) 
%                  * prod_{j=1}^d Gamma(x + (1 - j)/2)
%
%       In particular, when d == 1, Gamma_d(x) = Gamma(x)
%
%       Then, its logarithm is given by
%
%       (d * (d-1) / 4) * log(pi) + sum_{j=1}^d GammaLn(x + (1 - j)/2)
%
%   V = mvgammaln(d, X);
%
%       computes the multivariate log-gamma function with dimension d
%       on X. 
%
%       X can be an array of any size, and then the output V would be
%       of the same size.
%

% Created by Dahua Lin, on Sep 2, 2011
%

%% verify input

if ~(isnumeric(d) && isscalar(d) && d == fix(d) && d >= 1)
    error('mvgammaln:invalidarg', 'd should be a positive integer scalar.');
end

if ~(isfloat(X) && isreal(X))
    error('mvgammaln:invalidarg', 'X should be a real-valued array.');
end

%% main

if d == 1
    V = gammaln(X);
    
else % d > 1

    % X --> x : row vector

    if ndims(X) == 2 && size(X,1) == 1
        x = X;
        rs = 0;
    else
        x = reshape(X, 1, numel(X));
        rs = 1;
    end
    
    % compute
    
    % this 0.2862 ... is the value of log(pi) / 4
    t0 = d * (d - 1) * 0.286182471462350043535856837838;

    Y = bsxfun(@plus, (1 - (1:d)') / 2, x);
    t1 = sum(gammaln(Y), 1);
    
    v = t0 + t1;

    % reshape back

    if rs
        V = reshape(v, size(X));
    else
        V = v;
    end
end
