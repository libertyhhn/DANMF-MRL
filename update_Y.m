function [ Y ] = update_Y( X, F, beta,lambda2)

V = length(X);
% for v=1:V
%     F{v} = F{v}';
% end
n = size(X{1}, 1);
k = size(F{1}, 1);

loss = zeros(n, k);

for v=1:V
    loss = loss + (beta(v).^lambda2) * EuDist2(X{v}, F{v}, 0);
end

[~, Y] = min(loss, [], 2);
Y = Y';
% Y = zeros(k,n);
% for j = 1 : n
%     Y(I(j),j) = 1;
% end
end

