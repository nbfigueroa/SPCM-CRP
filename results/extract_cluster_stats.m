function [Purity_stats, NMI_stats, ARI_stats, F2_stats, K_stats] = extract_cluster_stats(clust_stats)
E = length(clust_stats);
Purity_stats = zeros(2,4,E);
NMI_stats    = zeros(2,4,E);
ARI_stats    = zeros(2,4,E);
F2_stats     = zeros(2,4,E);
K_stats      = zeros(2,4,E);
for e=1:E
    fprintf('**************** Performance of Clustering on %s ****************\n', clust_stats{e}.embedding);
    for k=1:4
        switch k
            case 1
                clust_type = 'GMM+BIC';
            case 2
                clust_type = 'CRP-GMM';
            case 3
                clust_type = 'SPCM-CRP-GMM';
            case 4
                clust_type = 'GMM-Oracle';
        end
        Purity_stats(1,k,e) = mean(clust_stats{e}.Purity(k,:));
        Purity_stats(2,k,e) = std(clust_stats{e}.Purity(k,:));
        NMI_stats(1,k,e)    = mean(clust_stats{e}.NMI(k,:));
        NMI_stats(2,k,e)    = std(clust_stats{e}.NMI(k,:));
        ARI_stats(1,k,e)    = mean(clust_stats{e}.ARI(k,:));
        ARI_stats(2,k,e)    = std(clust_stats{e}.ARI(k,:));
        F2_stats(1,k,e)     = mean(clust_stats{e}.F2(k,:));
        F2_stats(2,k,e)     = std(clust_stats{e}.F2(k,:));
        K_stats(1,k,e)      = mean(clust_stats{e}.K(k,:));
        K_stats(2,k,e)      = std(clust_stats{e}.K(k,:));
        
        fprintf('%s: Purity [%2.4f +/- %2.2f] ', clust_type, Purity_stats(1,k,e), Purity_stats(2,k,e));              
        fprintf('NMI [%2.4f +/- %2.2f] ',  NMI_stats(1,k,e), NMI_stats(2,k,e));                
        fprintf('ARI [%2.4f +/- %2.2f] ',  ARI_stats(1,k,e), ARI_stats(2,k,e));               
        fprintf('F2 [%2.4f +/- %2.2f] ',   F2_stats(1,k,e),  F2_stats(2,k,e));
        fprintf('K [%2.4f +/- %2.2f] \n',  K_stats(1,k,e),  K_stats(2,k,e));
    end
end
end