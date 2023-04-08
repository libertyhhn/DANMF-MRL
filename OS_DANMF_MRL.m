
function [ Z, Hcon, Y, derror ] = OS_DANMF_MRL( XX, layers, varargin )

pnames = { ...
    'z0' 'h0' 'bUpdateH' 'bUpdateLastH' 'maxiter' 'TolFun', ...
    'verbose', 'bUpdateZ', 'cache', 'gnd', 'gamma', 'lambda', 'graph_k', 'savePath','omega'...
    };


numOfView = numel(XX);
num_of_layers = numel(layers);
numOfSample = size(XX{1,1},2);

alpha = ones(numOfView,1).*(1/numOfView);

Q = cell(numOfView, num_of_layers+1);
Z = cell(numOfView, num_of_layers);
H = cell(numOfView, num_of_layers);

dflts  = {0, 0, 1, 1, 100, 1e-5, 1, 1, 0, 0,0};

[z0, h0, bUpdateH, bUpdateLastH, maxiter, tolfun, verbose, bUpdateZ, cache, gnd, gamma, lambda, graph_k] = ...
    internal.stats.parseArgs(pnames,dflts,varargin{:});

A_graph = cell(1,numOfView);
D_graph = cell(1,numOfView);
L_graph = cell(1,numOfView);
F = cell(numOfView,1);
options = [];
options.k = graph_k;
options.WeightMode = 'HeatKernel';
% options.WeightMode = 'Binary';
Hf = 0;

% normali
% gamma = 2;
numC = numel(unique(gnd));
Norm = 2; % normalize to have unit norm L_2; if Norm = 1, to have unit norm L_1 
NormV = 1;
% init NMF
options2 = [];
options2.maxIter = 200;
options2.error = 1e-6;
options2.nRepeat = 30;
options2.minIter = 50;
options2.meanFitRatio = 0.1;
options2.rounds = 30;
U_ = [];
V_ = [];
% init Y
rand('seed',4); % 4,8 for washington
Y = randi([1,numC],numOfSample,1);
% Y = zeros(numC,numOfSample);
% for j = 1 : numOfSample
%     Y(I(j),j) = 1;
% end
for v_ind = 1:numOfView
    X = XX{v_ind};
    nSmp = size(X, 2);
%     Hf = rand(nSmp, layers(num_of_layers))';
%     X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));

    
    
    %======== construct graph ========
    % calculate weight matrix W
    TempWt = constructW(X',options);
    if isfield(options,'NormWeight') && strcmpi(options.NormWeight,'NCW')
        D_mhalf = sum(TempWt,2).^-.5;
        D_mhalf = spdiags(D_mhalf,0,nSmp,nSmp);
        TempWt = D_mhalf*TempWt*D_mhalf;
        clear D_mhalf;
    end
    A_graph{v_ind} = TempWt;
    
    % calculate Laplance matrix L
    DCol = full(sum(A_graph{v_ind},2));
    D_graph{v_ind} = spdiags(DCol,0,speye(size(A_graph{v_ind},1)));
    TempL = D_graph{v_ind} - A_graph{v_ind};
    if isfield(options,'NormLap') && options.NormLap % note: now no options.NormLap
        D_mhalf = DCol.^-.5;
        tmpD_mhalf = repmat(D_mhalf,1,nSmp);
        TempL = (tmpD_mhalf.*TempL).*tmpD_mhalf';
        clear D_mhalf tmpD_mhalf;
        TempL = max(TempL, TempL');
    end
    L_graph{v_ind} = TempL;
    
    if  ~iscell(h0)
        for i_layer = 1:length(layers)
            if i_layer == 1
                % For the first layer we go linear from X to Z*H, so we use id
                V = X;
            else
                V = H{v_ind,i_layer-1};
            end
            
            if verbose
                display(sprintf('Initialising Layer #%d with k=%d with size(V)=%s...', i_layer, layers(i_layer), mat2str(size(V))));
            end
            if ~iscell(z0)
                %-----------the initalization from MVCC (k-means)----------
%                 if i_layer == length(layers)
%                     ilabels = zeros(nSmp,numOfView);
%                     %ilabels(:,i) = kmeans(data{i}',numC,'replicates',20);
%                     ilabels(:,v_ind) = litekmeans(X',numC,'Replicates',20);
%                     G = zeros(nSmp,numC);
%                     for j=1:numC
%                         G(:,j)=(ilabels(:,v_ind)==j*ones(nSmp,1));
%                     end 
%                     H{v_ind,i_layer}=G'+0.1*ones(nSmp,numC)';
%                     Dw=diag(sum(G,1))^-1;
%                     Z{v_ind,i_layer}=(0.1*ones(layers(i_layer-1),numC))*Dw;                    
%                 else
                    [Z{v_ind,i_layer}, H{v_ind,i_layer}, ~] = ShallowNMF(V, layers(i_layer), maxiter, tolfun);
    %                 [Z{v_ind,i_layer}, H{v_ind,i_layer}] = NMF(V, layers(i_layer), options2, U_, V_);
    %                 H{v_ind,i_layer} = H{v_ind,i_layer}';
    %                 [Z{v_ind,i_layer}, H{v_ind,i_layer}, ~] = ShallowGNMF(V, layers(i_layer), maxiter, tolfun,L_graph{v_ind});
%                 end
            else
                display('Using existing Z');
                [Z{v_ind,i_layer}, H{v_ind,i_layer}, ~] = ...
                    seminmf(V, ...
                    layers(i_layer), ...
                    'maxiter', 1, ...
                    'bUpdateH', true, 'bUpdateZ', 0, 'z0', z0{i_layer}, 'verbose', verbose, 'save', cache, 'fast', 1);
            end
%             if i_layer == 1
%                 D = Z{v_ind,1};
%             else
%                 D = D * Z{v_ind,i_layer};
%             end
        end
        
    else
        Z=z0;
        H=h0;
        
        if verbose
            display('Skipping initialization, using provided init matrices...');
        end
    end
    % initialize Hcon
     Hf = Hf + (alpha(v_ind)^gamma)*H{v_ind,numel(layers)};
    
    dnorm0(v_ind) = cost_function_graph(X, Z(v_ind,:), H(v_ind,:), alpha(v_ind)^gamma, L_graph{v_ind},Hf);
    dnorm(v_ind) = dnorm0(v_ind) + 1;
    
    if verbose
        display(sprintf('#%d error: %f', 0, sum(dnorm0)));
    end
end
% [F_temp,Y] = myInitializationF(Ker,numC);
Ker = XX{1}; % 1 for washington
[F_temp,Y] = myInitializationY(Ker,numC);
% [F_temp] = myInitializationF(H(:,numel(layers)),numC);
for v_ind = 1:numOfView
    F{v_ind} = F_temp;
end
%% Error Propagation

if verbose
    display('Finetuning...');
end
H_err = cell(numOfView, num_of_layers);
derror = [];
% Hf = sum(H{:,numel(layers)});
Hcon = Hf / sum(alpha);
tic
for iter = 1:maxiter
%     HC = zeros();
    Vum = 0; Vdm = 0;
    for v_ind = 1:numOfView
        X = XX{v_ind};
%         X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
        AAT = X * X';
        H_err{v_ind,numel(layers)} = H{v_ind,numel(layers)};
        for i_layer = numel(layers)-1:-1:1
            H_err{v_ind,i_layer} = Z{v_ind,i_layer+1} * H_err{v_ind,i_layer+1};
        end
        
        Q{v_ind,num_of_layers + 1} = eye(layers(num_of_layers));
        for i_layer = num_of_layers:-1:2 %
            Q{v_ind,i_layer} = Z{v_ind,i_layer} * Q{v_ind,i_layer + 1};
        end
        
        for i = 1:numel(layers)
            if bUpdateZ
                try
                    
                    HpHpT = H{v_ind,numel(layers)} * H{v_ind,numel(layers)}';
                    
                    if i == 1
                        R = Z{v_ind,i} * (Q{v_ind,2} * HpHpT *  Q{v_ind,2}') + AAT * (Z{v_ind,i} * (Q{v_ind,2} * Q{v_ind,2}'));
                        Ru = 2 * X * (H{v_ind,numel(layers)}' * Q{v_ind,2}');
                        Z{v_ind,i} = Z{v_ind,i}.* Ru ./ max(R, 1e-10);
                    else
                        R = D' * D * Z{v_ind,i} * Q{v_ind,i + 1} * HpHpT * Q{v_ind,i + 1}' + D' * AAT * D * Z{v_ind,i} * Q{v_ind,i + 1} * Q{v_ind,i + 1}';
                        Ru = 2 * D' * X * H{v_ind,numel(layers)}' * Q{v_ind,i + 1}';
                        Z{v_ind,i} = Z{v_ind,i}.* Ru ./ max(R, 1e-10);
                    end
                catch
                    display(sprintf('Convergance error %f. min Z{i}: %f. max %f', norm(Z{v_ind,i}, 'fro'), min(min(Z{v_ind,i})), max(max(Z{v_ind,i}))));
                end
            end
            
            if i == 1
                D = Z{v_ind,1};
            else
                D = D * Z{v_ind,i};
            end
            
            if bUpdateH && (i < numel(layers) || (i == numel(layers) && bUpdateLastH))

                 Vu = 2 * D' * X;
                 Vd = D' * D * H{v_ind,i} + H{v_ind,i};
                 H{v_ind,i} = H{v_ind,i} .* Vu ./ max(Vd, 1e-10);
                 
                 if i == numel(layers)
                     Vum = (2 * D' * X + lambda * H{v_ind,i} * A_graph{v_ind} + (alpha(v_ind)^gamma)* Hcon);
                     Vdm = ( D' * D * H{v_ind,i} + lambda * H{v_ind,i} * D_graph{v_ind} + (alpha(v_ind)^gamma+1)* H{v_ind,i});
                     H{v_ind,i} = H{v_ind,i}.* Vum ./ max(Vdm, 1e-10);
                     
                 end                
            end
        end
     % normlizing W{i}, V{i}
%         [D, H{v_ind,i}] = NormalizeWV(Ker{v_ind}, D, H{v_ind,i}, NormV, Norm, numC);
%         H{v_ind,i} = H{v_ind,i}';
        [D, H{v_ind,i}] = NormalizeUV(D, H{v_ind,i}, NormV, Norm);
        H{v_ind,i} = H{v_ind,i}';
        assert(i == numel(layers));
    end
    Hm = H(:,numel(layers));
    for v_ind = 1:numOfView
        % the following two lines are used for calculating weight
        tmpNorm = norm(H{v_ind,numel(layers)} - Hcon,'fro')^2;
        dnorm_w(v_ind) = (gamma*(tmpNorm))^(1/(1-gamma));
        Hm{v_ind} = Hm{v_ind}';
    end            
    

    I = Y;
    Y_temp = zeros(numC,numOfSample);
    for j = 1 : numOfSample
        Y_temp(I(j),j) = 1;
    end
    
    % update alpha and HC
    for v_ind = 1:numOfView     
%         alpha(v_ind) = sum(dnorm_w)/dnorm_w(v_ind);
        alpha(v_ind) = dnorm_w(v_ind)/sum(dnorm_w);
        % update Hf
%         HC = HC + (alpha(v_ind)^gamma)*H{v_ind,numel(layers)};
%         B{v_ind} = Hm{v_ind}'*Y_temp';
%         [Uh,Sh,Vh] = svd(B{v_ind},'econ');
%         F_temp = Uh * Vh';
%         F{v_ind} = F_temp;
    end
    [F] = update_F(Hm, Y, numC);
    [Y] = update_Y(Hm, F, alpha,gamma);
    
    Hcon = zeros(size(H{1,numel(layers)}));
    for v_ind = 1:numOfView 
        Hcon_temp = F{v_ind}(Y,:)';
        Hcon = Hcon + Hcon_temp;
    end
     
   
    for v_ind = 1:numOfView
        
        X = XX{v_ind};
        X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
        
%         H{v_ind,num_of_layers} = H{v_ind,num_of_layers}.* Vum ./ max(Vdm, 1e-10);
%         H{v_ind,i} = H{v_ind,i} .* Vu ./ max(Vd, 1e-10);

        % get the error for each view
        dnorm(v_ind) = cost_function_graph(X, Z(v_ind,:), H(v_ind,:), alpha(v_ind)^gamma, L_graph{v_ind},Hcon);       
    end  
    
    % finish update Z H and other variables in each view
    % disp result
    
    maxDnorm = (1./numOfView) * sum(dnorm);
    if verbose
        display(sprintf('#%d error: %f', iter, maxDnorm));
        derror(iter) = maxDnorm;
    end
    
    %     assert(dnorm <= dnorm0 + 0.01, ...
    %         sprintf('Rec. error increasing! From %f to %f. (%d)', ...
    %         dnorm0, dnorm, iter) ...
    %     );
    
%     if verbose && length(gnd) > 1
%         if mod(iter, 1) == 0|| iter ==1
% %             [acc, nmii, ~ ]= evalResults_multiview(Hcon, gnd);
%             [acc, nmii,~,~,~,~,~]= evalResults_multiview_2(Y', gnd); % Y 
%             ac = mean(acc);
%             ac_std = std(acc);
%             nmi = mean(nmii);
%             nmi_std = std(nmii);
%             
%             fprintf(1, 'Clustering accuracy is %.4f, NMI is %.4f\n', ac, nmi);
%         end
%     end
    
    %             if dnorm0-maxDnorm <= tolfun*max(1,dnorm0)
    %                 if verbose
    %                     display( ...
    %                         sprintf('Stopped at %d: dnorm: %f, dnorm0: %f', ...
    %                             iter, maxDnorm, dnorm0 ...
    %                         ) ...
    %                     );
    %                 end
    %                 break;
    %             end
    
    dnorm0 = maxDnorm;
    
end
t = toc;
end

function error = cost_function_allconsensus(HC, H,alpha,views)
tempHHc = zeros(1,views);
for i = 1:views
    HtHc = H{i} - HC;
    tempHHc(i) = tempHHc(i) + alpha(i)*norm(HtHc,'fro')^2;
end
error = sum(tempHHc);
end

function error = cost_function_graph(X, Zp, H, weight, A, Hf)
out = H{numel(H)};
error = (norm(X - reconstructionDe(Zp,H), 'fro') + norm(out - reconstructionEn(Zp , X), 'fro')+ trace(out*A*out') + weight*norm(out - Hf, 'fro'));
end

function [ out ] = reconstructionDe( Z, H )
    out = H{numel(H)};

for k = numel(H) : -1 : 1
    out =  Z{k} * out;
end

end

function [ out ] = reconstructionEn( Z, H )
out = H;
for k = 1 : 1 : numel(Z)
    out =  Z{k}' * out;
end

end