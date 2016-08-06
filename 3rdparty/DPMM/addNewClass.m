% Copyright (C) 2007 Jacob Eisenstein: jacobe at mit dot edu
% distributable under GPL, see README.txt

function params = addNewClass(params)
%function params = addNewClass(params)
%adds a new, empty class to the dpmm

    newclassidx = params.num_classes+1;
    params.num_classes = newclassidx;
    params.counts(newclassidx) = 0;
    params.sums(newclassidx,:) = params.kappa * params.initmean;
    
%     [~,p] = chol(params.initcov);
%     if p > 0
%         eps = 0.01;
%         [V,D] = eig(allcov);
%         d=diag(D);
%         d(d<=0)=eps;
%         A_spd = V*diag(d)*V';
%         params.initcov = A_spd;   
%     end
    
    params.cholSSE(:,:,newclassidx) = chol(params.nu * params.initcov);
    %params.SSE(:,:,newclassidx) = params.nu * params.initcov;
end
