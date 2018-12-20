function x = expmap_vec(u,s)
% Exponential map for the first vector form (SPD manifold)
	nbData = size(u,2);
	d = size(u,1)^.5;
	U = reshape(u, [d, d, nbData]);
	S = reshape(s, [d, d]);
	x = zeros(d^2, nbData);
	for t=1:nbData
		x(:,t) = reshape(expmap(U(:,:,t),S), [d^2, 1]);
	end
end

