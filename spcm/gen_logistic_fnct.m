function [y] = gen_logistic_fnct(x, a, b, c, r)
y = (a*exp((1.5-x)*r) + b*exp(r*x)) ./ (exp((1.5-x)*r) + exp(r*x));
end