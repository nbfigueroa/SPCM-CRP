% In order to use the Ward clustering z as an initialization to our model, we
%   need to generate voxel links "c" that are consistent with the Ward
%   clustering. There are many way to do this, but a simple one is to
%   construct a minimum spanning tree within each cluster, and set each
%   element's "c" link to point to its parent in the tree.
function c = ClusterSpanningTrees(z, adj_list)

    % Remove all adjacencies that cross clusters
    nvox = length(adj_list);
    for i = 1:nvox
        adj_list{i} = adj_list{i}(z(adj_list{i}) == z(i));
        adj_list{i} = adj_list{i}(randperm(length(adj_list{i})));
    end
    neighbor_count = cellfun(@length, adj_list);
    node_list = zeros(sum(neighbor_count), 1);
    next_edge = 1;
    for i = 1:nvox
        if (neighbor_count(i) > 0)
            node_list(next_edge:(next_edge+neighbor_count(i)-1)) = i;
            next_edge = next_edge + neighbor_count(i);
        end
    end
    G = sparse(node_list, [adj_list{:}]', 1, nvox, nvox);
    
    % Construct spanning tree in each cluster
    c = zeros(length(adj_list),1);
    for clust = unique(z)';
        clust_vox = find(z==clust);
        [~,parents] = graphminspantree(G,clust_vox(randi(length(clust_vox),1)));
        c(clust_vox) = parents(clust_vox);
    end
    roots = find(c==0);
    c(roots) = roots;

end

