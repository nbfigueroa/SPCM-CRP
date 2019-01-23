function [D, distance_name] = computeSDP_distances(sigmas,dist_type)

switch dist_type
    case -2        
        %%%%%%%%%%%% 'EUCL': Euclidean (Frobenius) Distance %%%%%%%%%
        D = compute_cov_sim( sigmas, 'EUCL' );
        distance_name = 'Euclidean (Frobenius) Distance';        
    case -1        
        %%%%%%%%%%%% 'CHOL': Cholesky-Euclidean (Frobenius) Distance %%%%%%%%%
        D = compute_cov_sim( sigmas, 'CHOL' );
        distance_name = 'Cholesky-Euclidean (Frobenius) Distance';        
    case 0        
        %%%%%%%%%%%% 'RIEM': Affine Invariant Riemannian Metric %%%%%%%%%
        D = compute_cov_sim( sigmas, 'RIEM' );
        distance_name = 'Affine Invariant Riemannian Distance';
    case 1
        %%%%%%%%%%%% 'LERM': Log-Euclidean Riemannian Metric %%%%%%%%%%%%
        D = compute_cov_sim( sigmas, 'LERM' );
        distance_name = 'Log-Euclidean Riemannian Distance';
    case 2 
        %%%%%%%%%%%% 'KLDM': Kullback-Liebler Divergence Metric %%%%%%%%%
        D = compute_cov_sim( sigmas, 'KLDM' );
        distance_name = 'Kullback-Liebler Divergence';
    case 3
        %%%%%%%%%%%% 'JBLD': Jensen-Bregman LogDet Divergence %%%%%%%%%%%%
        D = compute_cov_sim( sigmas, 'JBLD' );
        distance_name = 'Jensen-Bregman LogDet Divergence';
    case 4
        %%%%%%%%%%%% 'SROT': Minimum Scale-Rotation Curve Distance %%%%%%%%%%%%
        D = compute_cov_sim( sigmas, 'SROT' );
        distance_name = 'Minimum Scale Rotation Curve Distance';
end

end