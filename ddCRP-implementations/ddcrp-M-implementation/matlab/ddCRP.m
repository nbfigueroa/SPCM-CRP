% Main function: Fits our model, given a connectivity matrix D and spatial
%   adjacency specified by adj_list.  An initialization of the voxel links
%   init_c and a ground truth parcellation gt_z (for comparison) can optionally
%   be provided. MCMC will be run for num_passes over the dataset, with
%   hyperparameters alpha, kappa, nu, and sigsq. Diagnostic information will
%   be saved every stats_interval iterations, and will be printed to the console
%   if verbose is True. The MAP parcellation and diagnostic information is
%   returned.
function [map_z stats] = ddCRP(D, adj_list, init_c, gt_z, num_passes, ...
                          alpha, kappa, nu, sigsq, stats_interval, verbose)

hyp = ComputeCachedLikelihoodTerms(kappa, nu, sigsq);
nvox = length(adj_list);

% Generate random initialization if not specified
if (isempty(init_c))
    c = zeros(nvox, 1);
    for i = 1:nvox
        neighbors = [adj_list{i} i];
        c(i) = neighbors(randi(length(neighbors)));
    end
else
    c = init_c;
end

% Initialize spatial connection matrix
G = sparse(1:nvox,c,1,nvox,nvox);
[K, z, parcels] = ConnectedComp(G);

sym = CheckSymApprox(D);
      
curr_lp = FullProbabilityddCRP(D, c, parcels, alpha, hyp, sym);


stats = struct('times',[],'lp',[],'NMI',[],'K',[], ...
                'z', zeros(0,nvox), 'c', zeros(0,nvox));
max_lp = -Inf;
t0 = tic;
steps = 0;
for pass = 1:num_passes
    % Visit elements randomly
    order = randperm(nvox);
    
    for i = order
        if (curr_lp > max_lp)
            max_lp = curr_lp;
            map_z = z;
        end
        
        if (mod(steps, stats_interval) == 0)
            stats = UpdateStats(stats, t0, curr_lp, K, z, c, steps, gt_z, ...
                                       map_z, verbose);
        end
        
        % Compute change in log-prob when removing the edge c_i
        G(i,c(i)) = 0;
        if (c(i) == i)
            % Removing self-loop, parcellation won't change
            rem_delta_lp = -log(alpha);
            z_rem = z; parcels_rem = parcels;
        else
            [K_rem, z_rem, parcels_rem] = ConnectedComp(G);
            if (K_rem ~= K)
                % We split a cluster, compute change in likelihood
                rem_delta_lp = -LikelihoodDiff(D, ...
                                  parcels_rem, z_rem(i), z_rem(c(i)), hyp, sym);
            else
                rem_delta_lp = 0;
            end
        end
        
        % Compute change in log-prob for each possible edge c_i
        adj_list_i = adj_list{i};
        lp = zeros(length(adj_list_i)+1, 1);
        lp(end) = log(alpha);
        cached_merge = zeros(length(adj_list_i),1);
        for n_ind = 1:length(adj_list_i)
            n = adj_list_i(n_ind);
            if (z_rem(n) == z_rem(c(i)))
                % Just undoing edge removal
                lp(n_ind) = -rem_delta_lp - (c(i)==i)*log(alpha);
            elseif (z_rem(n) ~= z_rem(i)) 
                % Proposing merge
                % First check cache to see if this is already computed
                prev_lp = find(cached_merge == z_rem(n),1);
                if (~isempty(prev_lp))
                    lp(n_ind) = lp(prev_lp);
                else
                    % This is a novel merge, compute change in likelihood
                    lp(n_ind) = LikelihoodDiff(D, parcels_rem, z_rem(i), z_rem(n), hyp, sym);
                    cached_merge(n_ind) = z_rem(n);
                end
            end
        end
        
        % Pick new edge proportional to probability
        new_neighbor = ChooseFromLP(lp);
        if (new_neighbor <= length(adj_list_i))
            c(i) = adj_list_i(new_neighbor);
        else
            c(i) = i;
        end
        
        % Update likelihood and parcellation
        curr_lp = curr_lp + rem_delta_lp + lp(new_neighbor);
        G(i,c(i)) = 1;
        [K, z, parcels] = ConnectedComp(G);
        steps = steps + 1;
    end
end

stats = UpdateStats(stats, t0, curr_lp, K, z, c, steps, gt_z, map_z, verbose);

end

% Given a sparse adjacency matrix G, returns the number of (undirected)
%   connected components K, the component labels z, and a cell list of
%   arrays with the indices of elements in each component
function [K, z, parcels] = ConnectedComp(G)
    [K, z] = graphconncomp(G, 'Weak', true);
    [sorted_z, sorted_i] = sort(z);
    parcels = mat2cell(sorted_i, 1, diff(find(diff([0 sorted_z (K+1)]))));
end

% Computes the change in model likelihood for connectivity matrix D when,
%   starting with parcellation parcels_split (given as a list of arrays of
%   element indices in each parcel), we combine parcels split_i1 and split_i2.
%   The vectorized hyperparameters are specified in hyp, and the boolean
%   input sym determines whether D is symmetric (in which case only half the
%   connectivity values are considered).
function ld = LikelihoodDiff(D, parcels_split, split_i1, split_i2, hyp, sym)
    
    % Compute log-likelihood for split parcels
    K = length(parcels_split);
    s = zeros(K, K, 3);
    for split_ind = [split_i1 split_i2]
        
        % Get all connections to split_ind parcel
        for i = 1:K
            samples = D(parcels_split{i}, parcels_split{split_ind});
            if (sym && i == split_ind)
                samples = samples(logical(triu(ones(size(samples)),1)));
            else
                samples = samples(:);
            end
            s(i,split_ind,:) = SufficientStats(samples);
        end
        
        % If not symmetric, get all connections from split_ind parcel
        if (~sym)
            for i = 1:K
                samples = D(parcels_split{split_ind}, parcels_split{i});
                if (i == split_ind)
                    off_diags = true(size(samples));
                    off_diags(1:(size(samples,1)+1):end) = false;
                    samples = samples(off_diags);
                else
                    samples = samples(:);
                end
                s(split_ind,i,:) = SufficientStats(samples);
            end
        end
    end
    
    % Compute log-likelihood of all sufficient stats for split parcels
    if (sym)
        split_ll = LogLikelihood([...
            reshape(s(:,split_i1,:),[],3); ...
            reshape(s(1:K ~= split_i1, split_i2,:),[],3)], hyp);
    else
        split_ll = LogLikelihood([...
            reshape(s(:, split_i1,:),[],3); ...
            reshape(s(:, split_i2,:),[],3); ...
            reshape(s(split_i1, (1:K ~= split_i1) & (1:K ~= split_i2),:),[],3); ...
            reshape(s(split_i2, (1:K ~= split_i1) & (1:K ~= split_i2),:),[],3)], hyp);
    end
    
    
    % Compute log-likelihood for all merged parcels
    m = zeros(2, K, 3);
    
    % Compute log-likelihood for all merged parcels
    for dir = 1:2
        if (dir == 2)
            if (sym)
                break;
            else
                s = permute(s, [2 1 3]);
            end
        end
        
        for i = 1:K
            if (i ~= split_i1 && i ~= split_i2)
                s_m = reshape(s(i, [split_i1 split_i2], :),2,3);
                m(dir,i,:) = MergeSuffStats(s_m);
                
            end
        end
    end
    if (sym)
        % Compute central diagonal merge
        m_central = MergeSuffStats(...
            [MergeSuffStats(reshape(s(split_i1, [split_i1 split_i2], :),2,3)); ...
            reshape(s(split_i2, split_i2, :),1,3)]);
        % Compute log-likelihood of all sufficient stats for merged parcels
        merge_ll = LogLikelihood([reshape(m(1, :, :),[],3); m_central], hyp);
    else
        % Compute central diagonal merge
        m_central = MergeSuffStats(...
            [MergeSuffStats(reshape(s(split_i1, [split_i1 split_i2], :),2,3)); ...
             MergeSuffStats(reshape(s(split_i2, [split_i1 split_i2], :),2,3))]);
        % Compute log-likelihood of all sufficient stats for merged parcels
        merge_ll = LogLikelihood([reshape(m(1, :, :),[],3);
                                reshape(m(2, :, :),[],3); m_central], hyp);
    end
    
    ld = merge_ll - split_ll;
end

% Compute count, mean, and sum of squared deviations for vector of samples
function suffstats = SufficientStats(samples)
    suffstats = zeros(3,1);
    if (isempty(samples))
        return;
    end
    suffstats(1) = length(samples);
    suffstats(2) = sum(samples)/suffstats(1);
    suffstats(3) = sum((samples-suffstats(2)).^2);
end

% Compute sufficient statistics for merging two sets of suff stats, specified
%   as a 2x3 matrix with columns = [count, mean, sum of squared dev]
function m = MergeSuffStats(s_m)
    m = zeros(1,3);
    m(1) = s_m(1,1) + s_m(2,1);
    m(2) = (s_m(1,1)*s_m(1,2) + s_m(2,1)*s_m(2,2))/m(1);
    m(3) = s_m(1,3) + s_m(2,3) + ...
                 (s_m(1,1)*s_m(2,1))/m(1) * (s_m(1,2) - s_m(2,2))^2;
end

% Update diagnostic stats with time (reported relative to start time t0),current
%   log-probability, current number of clusters, current parcellation z, current
%   voxel links c, number of steps, ground truth (if available), best
%   parcellation so far (map_z). If verbose=True, also print to console.
function stats = UpdateStats(stats, t0, curr_lp, K, z, c, steps, gt_z, map_z, verbose)
    stats.lp = [stats.lp curr_lp];
    stats.K = [stats.K K];
    stats.z = [stats.z; z];
    elapsed = toc(t0);
    stats.times = [stats.times elapsed];
    stats.c = [stats.c; c'];
    if (~isempty(gt_z))
        stats.NMI = [stats.NMI CalcNMI(gt_z, map_z)];
    end
    if (verbose)
        if (~isempty(gt_z))
            disp(['Step: ' num2str(steps) ...
              '  Time: ' num2str(elapsed) ...
              '  LP: ' num2str(curr_lp) ...
              '  K: ' num2str(K) ...
              ' NMI: ' num2str(stats.NMI(end))]);
        else
            disp(['Step: ' num2str(steps) ...
                  '  Time: ' num2str(elapsed) ...
                  '  LP: ' num2str(curr_lp) ...
                  '  K: ' num2str(K)]);
        end
    end
end