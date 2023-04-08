function [C,IDX] = myInitializationY(avgKer,k)

% numker = size(KH,2);
% Sigma0 = ones(numker,1)/numker;
% avgKer  = mycombFun2(KH,Sigma0);
% [H_normalized1] = mykernelkmeans(avgKer, k);
% H_normalized1 = avgKer./ repmat(sqrt(sum(avgKer.^2, 2)), 1,k);
% H_normalized1 = H_normalized1./ repmat(sqrt(sum(H_normalized1.^2, 2)), 1,k);
[IDX, C] = kmeans(avgKer',k);
% [IDX, C] = kmeans(avgKer,k  , 'MaxIter',200, 'Replicates',30);
% C = orth(C);
% %returns the K cluster centroid locations in the K-by-P matrix C.