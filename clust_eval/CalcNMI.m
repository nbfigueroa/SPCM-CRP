% Compute normalized mutual information between two parcellations gt_z and z
function NMI = CalcNMI(gt_z, z)

N = length(gt_z);
MI = 0;
gt_z = gt_z(:)';
z = z(:)';

gt_p = zeros(max(gt_z),1);
H_gt = 0;
for i = unique(gt_z)
    gt_p(i) = sum(gt_z == i) / N;
    H_gt = H_gt - gt_p(i) * log(gt_p(i));
end

p = zeros(max(z),1);
H = 0;
for j = unique(z)
    p(j) = sum(z == j) / N;
    H = H - p(j) * log(p(j));
end

for i = unique(gt_z)
    for j = unique(z)
        joint_p = sum(gt_z == i & z == j) / N;
        if (joint_p > 0)
            MI = MI + joint_p * log(joint_p / (gt_p(i)*p(j)));
        end
    end
end

if (MI == 0)
    NMI = 0;
else
    NMI=MI/sqrt(H*H_gt);
end

end

