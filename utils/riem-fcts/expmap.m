function S = expmap(W,S)
	S = S^.5 * expm(S^-.5 * W * S^-.5) * S^.5;
end


