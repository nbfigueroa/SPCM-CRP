function LP = LPddCRP_NG(X,A,cLinks,Pi,hyp,weights,seating,Klist)

if nargin<7||isempty(seating); seating = 1; end
K = max(Pi);
if nargin<8
  Klist = 1:K;
end
[T, N] = size(X);
mu0 = 0;
alpha = hyp.alpha;
kappa0 = hyp.kappa0;
a0 = hyp.a0;
b0 = hyp.b0;

LLseating = 0;
if seating
  for i = 1:length(cLinks)
    if i==cLinks(i)
      LLseating = LLseating+log(alpha./(alpha+length(A{i})));
    else
      LLseating = LLseating+log(weights{i}(A{i}==cLinks(i))./(alpha+sum(weights{i})));
    end
  end
end

LLdata = 0;
for i = 1:numel(Klist)
  k = Klist(i);
  if ~(sum(Pi==k)==1 && isempty([A{Pi==k}]))
    LLdata = LLdata+singleClustLL(X(:,Pi==k),a0,b0,mu0,kappa0);
  end
end

LP = LLseating+LLdata;
