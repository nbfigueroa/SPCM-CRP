function Pi = link2Pi(links)

N = length(links);

K = 0;
Pi = zeros(N,1);
for i = 1:N
  if Pi(i)==0
    K = K+1;
    Piold = Pi;
    curr = i;
    Pi(curr) = K;
    while ~isequal(Pi,Piold)
      Piold=Pi;
      curr = links(curr);
      if curr>0
        if Pi(curr)==0
          Pi(curr) = K;
        elseif Pi(curr)<K
          k = Pi(curr);
          Pi(Pi==K) = k;
          K = K-1;
          break
        end
      end
    end
  end
end
