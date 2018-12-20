function v = symMat2Vec(S)
% Reduced vectorisation of a symmetric matrix.
[d,~,N] = size(S);

v = zeros(d+d*(d-1)/2,N);
for n = 1:N
	v(1:d,n) = diag(S(:,:,n));
	
	row = d+1;
	for i = 1:d-1
		v(row:row + d-1-i,n) = sqrt(2).*S(i+1:end,i,n);
		row = row + d-i;
	end
end
end