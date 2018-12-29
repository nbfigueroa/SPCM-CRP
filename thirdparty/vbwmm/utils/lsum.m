function Y = lsum(X, d)
% LSUM Numerically stable computation of log(sum(exp(X),d))
maxX = max(min(max(X, [], d),realmax),-realmax);
Y = maxX + log(sum(exp(bsxfun(@minus, X, maxX)),d));
