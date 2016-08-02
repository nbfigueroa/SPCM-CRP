function members = memFind(cLinks,orig)

nn = 0;
members = orig;
n = 1;
inds = 1:length(cLinks);

while n>nn
    nn = n;
%     back = inds(builtin('_ismemberoneoutput',cLinks,members));
    back = inds(ismember(cLinks,members));
    members = sort([back cLinks(members) orig]);
    members = members([true diff(members)>0]);
    n = length(members);
end
