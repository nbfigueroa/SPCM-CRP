function S = logmap(W,S)
	S = S^.5 * logm(S^-.5 * W * S^-.5) * S^.5;
end