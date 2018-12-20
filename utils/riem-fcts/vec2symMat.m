function S = vec2symMat(v)
% Matricisation of a vector to a symmetric matrix.
[t, N] = size(v);

d = (-1 + sqrt(1+8*t))/2;
S = zeros(d,d,N);

for n= 1:N
    % Side elements
    i = d+1;
    for row = 1:d-1
        S(row,row+1:d,n) = v(i:i+d-1-row,n)./sqrt(2);
        i = i+d-row;
    end
    S(:,:,n) = S(:,:,n) + S(:,:,n)';
    % Diagonal elements
    S(:,:,n) = S(:,:,n) + diag(v(1:d,n));
end
end