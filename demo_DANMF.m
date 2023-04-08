
addpath(genpath('.'));
clear;
%%
dataname = {'ALOI100_4_11025'};
numdata = length(dataname);

for cdata = 1:numdata
%% read data
idata = cdata;
datadir = 'data/';
dataset = char(dataname(idata));
dataf = [datadir, cell2mat(dataname(idata))];
load(dataf);
C=length(unique(label));
for v = 1:length(data)
%      data{v} = NormalizeFea(data{v}, 0);
    data{v} = mapminmax(data{v},0,1);
end
gnd = label';

%%
savePath = './results_DANMF_MRL/';

layers = [300 200 100 50] ;
graph_k = 5;
gamma = 5;   % the weight of alpha
lambda = 0.1;% the weight of graph
for i = 1:10
[ Z, H, dnorm ] = DANMF_MRL( data, layers,'gamma', gamma,'gnd',gnd, 'lambda', lambda,...
    'graph_k', graph_k, 'savePath', savePath);

[ac1(i), nmi1(i), Pri1(i),AR1(i),F1(i),P1(i),R1(i)] = printResult(H', gnd, C, 1); % kmeans clustering
end

ac1m = mean(ac1); nmi1m = mean(nmi1); Pri1m = mean(Pri1); AR1m = mean(AR1); F1m = mean(F1); P1m = mean(P1); R1m = mean(R1);
ac1s = std(ac1); nmi1s = std(nmi1); Pri1s = std(Pri1); AR1s= std(AR1);F1s= std(F1);P1s= std(P1);R1s= std(R1);
eva = [ac1m, ac1s,nmi1m,nmi1s,Pri1m,Pri1s,AR1m,AR1s,F1m,F1s,P1m,P1s, R1m,R1s]*100;
eva = roundn(eva,-2);


fprintf('10times ac: %0.2f\tnmi:%0.2f\tpur:%0.2f\tar:%0.2f\tf_sc:%0.2f\tpre:%0.2f\trec:%0.2f\n', ac1m*100, nmi1m*100, Pri1m*100,AR1m*100,F1m*100,P1m*100,R1m*100);


Tname = [savePath,dataset,num2str(layers),'.txt'];
dlmwrite(Tname,eva,'-append','delimiter','\t','newline','pc');

objectname = [savePath, dataset, '.mat' ];
save(objectname,'dnorm');
end
return