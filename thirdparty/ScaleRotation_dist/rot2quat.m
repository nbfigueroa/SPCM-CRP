function quat = rot2quat(U)
% ROT2QUAT converts 3 x 3 rotation matrix U to a quaternion 
tol = 1e-14;
theta = acos((trace(U) - 1) / 2) ; 
if abs(theta) < tol
    w = [0,0,0]';
elseif abs((abs(theta) - pi)) < tol % the case theta == pi 
    % then there is no principal logarithm
    A = (U+eye(3))/2; 
    a = sqrt(diag(A)); 
    if A(1,2) < 0;
        a(2) = -a(2);
    end
    if A(1,3) < 0;
        a(3)  = -a(3);
    end
    if A(2,3) < 0 && a(2)*a(3)>0; 
        a(3) = - a(3);
    end
    w = a;
else
    w = [U(3,2) - U(2,3)
     U(1,3) - U(3,1)
     U(2,1) - U(1,2)] / ( 2 * sin(theta) );
end  
quat = [cos(theta/2)
      sin(theta/2)* w];
 
% Ann = logm(U*I{1});
% a1 = [-Ann(2,3) Ann(1,3) -Ann(1,2)];
% theta = norm(a1); 
% w = a1 / theta ; 
 