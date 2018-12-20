function [ s_norm ] = normalize_soft( s )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%# get max and min
maxVec = max(s);
minVec = min(s);

%# normalize to -1...1
s_norm = ((s-minVec)./(maxVec-minVec) - 0.5 ) *2;


%# to "de-normalize", apply the calculations in reverse
% vecD = (vecN./2+0.5) * (maxVec-minVec) + minVec;

end

