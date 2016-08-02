function [cLinks, Pi, ClustMembers, allLLs] = ddCRP_NGgensamp(X, cLinks, A, hyp, opts)

% probabilistic model based clustering using a normal-gamma prior on
% cluster distributions

[T, N] = size(X);

% default hyperparameters
alpha = 1;
mu0 = 0;
kappa0 = 1;
a0 = T;
b0 = T*0.5;

% default options
temp = 1; % temperature
steps = 1; % number of steps
vfb = 0;

% extracting hyperparameters and weights
if nargin >3
  fld = fieldnames(hyp);
  for i = 1:length(fld)
    eval(sprintf('%s = hyp.%s;',fld{i},fld{i}))
  end
end

if nargin >4
  fld = fieldnames(opts);
  for i = 1:length(fld)
    eval(sprintf('%s = opts.%s;',fld{i},fld{i}))
  end
end

if ~exist('Pi','var')
  fprintf('Creating z(c)');
  list = 1:N;
  ClustMembers = {};
  K = 0;
  Pi = zeros(N,1);
  while ~isempty(list)
    K = K+1;
    ClustMembers{K} = memFind(cLinks,list(1));
    Pi(ClustMembers{K}) = K;
    list(ismember(list,ClustMembers{K})) = [];
  end
else
  Pi = Pi(:);
  K = max(Pi);
end

if vfb
  allPi = Pi;
end

if ~exist('allLLs','var')
  allLLs = zeros(size(Pi));
  for k = 1:K
    allLLs(k) = singleClustLL(X(:,Pi==k),a0,b0,mu0,kappa0);
  end
elseif length(allLLs)<N
  allLLs = [allLLs(:);zeros(N-length(allLLs),1)];
end

if steps<length(temp)
  steps = length(temp);
elseif numel(temp)==1&&steps>1
  temp = temp.*ones(steps,1);
elseif numel(temp)~=steps
  error('temp should match the number of steps or be scalar')
end

fprintf('; %8d clusters, %5.1f%% done',max(Pi),0)

for s = 1:steps
  t = temp(s);
  prm = randperm(N);
  for i=1:N
    tnode = prm(i); % pick an element to reassign
    if ~isempty(A{tnode})
      %         Pi_old = link2Pi(cLinks);
      oldlink = cLinks(tnode);
      oldPi = Pi;
      cLinks(tnode) = tnode;
      l = memFind(cLinks,tnode);
      if numel(l)~=numel(ClustMembers{Pi(tnode)})
        K = K+1;
        ClustMembers{K} = l;
        %             inds = builtin('_ismemberoneoutput',ClustMembers{Pi(tnode)},l);
        inds = ismember(ClustMembers{Pi(tnode)},l);
        ClustMembers{Pi(tnode)}(inds) = [];
        Pi(l) = K;
        allLLs(Pi(oldlink)) = singleClustLL(X(:,Pi==Pi(oldlink)),a0,b0,mu0,kappa0);
        allLLs(K) = singleClustLL(X(:,Pi==K),a0,b0,mu0,kappa0);
      end
      [nbs,Wind] = sort([tnode A{tnode}]);
      Plinks = [alpha weights{tnode}];Plinks = Plinks(Wind);
      Uind = [true diff(nbs)>0];
      Plinks = Plinks(Uind);
      nbs = nbs(Uind);
      Nnbs = numel(nbs);
            
      Pdata = zeros(Nnbs,1);
      mlist = false(Nnbs,1);
      currClust = Pi(tnode);
      fKlist = Pi(nbs);
      Klist = unique(fKlist);
      uKs = length(Klist);
      LLs = zeros(uKs,3);
      
      %         for j = 1:uKs
      %             k = Klist(j);
      %             LLs(j,1) = singleClustLL(X(:,Pi==k),a0,b0,mu0,kappa0);
      %         end
      LLs(:,1) = allLLs(Klist);
      
      for j = 1:uKs
        k = Klist(j);
        if currClust==k
          LLs(j,2) = sum(LLs(:,1));
        else
          others = true(size(Klist));
          others([j find(Klist==currClust)]) = false;
          LLs(j,3) = singleClustLL(X(:,Pi==k|Pi==currClust),a0,b0,mu0,kappa0);
          LLs(j,2) = sum(LLs(others,1))+LLs(j,3);%singleClustLL(X(:,Pi==k|Pi==currClust),a0,b0,mu0,kappa0);
        end
      end
      
      for n = 1:Nnbs
        if ~any(l==nbs(n))
          mlist(n) = true;
        end
        Pdata(n) = LLs(Klist==Pi(nbs(n)),2);
      end
      Pdata = exp(Pdata-max(Pdata));
      Pdata = Pdata./sum(Pdata);
      
      P = Plinks(:).*Pdata;
      P = P.^t;
      P = P./sum(P);
      P = cumsum(P);
      
      c_i = find(P>rand,1,'first');
      if mlist(c_i)
        k1 = Pi(tnode);k2 = Pi(nbs(c_i));
        c_l = min(k1,k2); c_h = max(k1,k2);
        Pi(Pi==c_h) = c_l;
        Pi(Pi>c_h) = Pi(Pi>c_h)-1;
        ClustMembers{c_l} = [ClustMembers{[k1 k2]}];
        ClustMembers(c_h) = [];
        %             allLLs(c_l) = singleClustLL(X(:,ClustMembers{c_l}),a0,b0,mu0,kappa0);
        allLLs(c_l) = LLs(Klist==fKlist(c_i),3);
        allLLs(c_h:(K-1)) = allLLs((c_h+1):K);
        K = K-1;
      end
      cLinks(tnode) = nbs(c_i);
      if vfb
        allPi = [allPi Pi];
        plot(allPi')
                    Q = zeros(N,N);
                    for k = 1:K
                        Q(ClustMembers{k},ClustMembers{k}) = 1;
                    end
                    imagesc(Q);axis square; colormap hot;
        drawnow
      end
    end
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b%8d clusters, %5.1f%% done',max(Pi),((i+(s-1)*N)./(steps*N))*100)
  end
end
fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
allLLs = allLLs(1:K);
