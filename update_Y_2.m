function [ Y ] = update_Y_2( X, F, beta,lambda2)
% this version is only one F
V = length(X);
% for v=1:V
%     F{v} = F{v}';
% end
F = F';
n = size(X{1}, 1);
k = size(F, 1);

loss = zeros(n, k);

for v=1:V
    loss = loss + (beta(v).^lambda2) * EuDist2(X{v}, F, 0);
end

[~, I] = min(loss, [], 2);
Y = zeros(k,n);
for j = 1 : n
    Y(I(j),j) = 1;
end
end

