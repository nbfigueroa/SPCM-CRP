function fa = FA(d);

a = d(1);
b = d(2);
c = d(3); 

fa = sqrt(1/2) * ...
    sqrt( (a-b)^2 + (b-c)^2 + (c-a)^2) / ...
    norm(d);