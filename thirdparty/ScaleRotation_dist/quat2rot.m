function R = quat2rot(q)
% QUAT2ROT converts a quaternion to 3 x 3 rotation matrix R
tol = 1e-14;


theta = 2*acos(q(1));
if abs( abs(q(1)) - 1 ) < tol
    u = [0,0,0]';
else
    u = q(2:4) / sin(theta/2);
end

    ucross = [ 0 -u(3) u(2);
        u(3) 0 -u(1);
        -u(2) u(1) 0];
    R = eye(3) + sin(theta)*ucross + (1-cos(theta)) * ucross^2;


% Ann = logm(U);
% a1 = [-Ann(2,3) Ann(1,3) -Ann(1,2)];
% theta = norm(a1); 
% w = a1 / theta ; 