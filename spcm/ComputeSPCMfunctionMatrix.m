function [spcm] = ComputeSPCMfunctionMatrix(behavs_theta, tau)

spcm = [];    
       
for i=1:length(behavs_theta)
  for j=1:length(behavs_theta)                   
        
        [b_sim s hom_fact dir] = ComputeSPCMPair(behavs_theta{i},behavs_theta{j}, tau);

        spcm(i,j,1) = s;
        spcm(i,j,2) = b_sim;
        spcm(i,j,3) = hom_fact; 
        spcm(i,j,4) = dir; 
        
   end
end

end