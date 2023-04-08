function cF = mycombFun2(Y,gamma)
m = size(Y,2);
n = size(Y{1},2);
cF = zeros(size(Y{1}));
% cF = zeros(n);
for p =1:m
%     Y{p} = Y{p}'*Y{p};
    cF = cF + Y{p}*gamma(p);
end