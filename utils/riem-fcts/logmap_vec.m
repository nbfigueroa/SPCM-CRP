function u = logmap_vec(x,s)
% Logarithm map for the first vector form (SPD manifold)
	nbData = size(x,2);
	d = size(x,1)^.5;
	X = reshape(x, [d, d, nbData]);
	S = reshape(s, [d, d]);
	u = zeros(d^2, nbData);
	for t=1:nbData
		u(:,t) = reshape(logmap(X(:,:,t),S), [d^2, 1]);
	end
end