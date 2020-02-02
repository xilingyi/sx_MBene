%% Least absolute shrinkage and selection operator
% 10 times 10K-CV
% 75% training set
% tune lamda

clearvars;close all;clc;

%% step1. load dataset
addpath('Functions'); 
[~, ~, raw] = xlsread('../ML_Figure5.xlsx');
data = cell2mat(raw(2:end,2:end));
input = data(:,2:end); 
output = data(:,1); 

%% step 2: tune parameter
trialsNum = 10; % repeat 10 times
[feature_reg,~,~] = zscore(input);
output_reg = output - mean(output);
lm_max = max(abs(transpose(feature_reg)*output_reg))/length(output_reg);
lda = logspace(-3,0,100)*lm_max;
lda = lda';
dimNum = length(lda);
lda_choice = zeros(trialsNum, 1); % preallocate storage
ids = zeros(trialsNum,1); % preallocate storage
lma_best = 0;
error  = inf;
rmse_train = zeros(trialsNum, 1);

%% step 3: 10次-10折交叉验证 
for ii = 1:trialsNum 
    %% (1) 10折交叉验证 运行KRR 
    [X_train, y_train, X_test,y_test] = splitData(input, output); % split train datas and test datas
    fold10Score = zeros(dimNum, 1);% preallocate storage
    for jj = 1:dimNum
        fold10Score(jj) = kFoldScore(X_train, y_train, lda(jj), 10); % use 10-fold cross-validation to compute
    end
    [val, id] = min(fold10Score);
    ids(ii) = id;
    ldas = lda(id);
    
    %%  (2) 10 times 10K-CV
    lda_choice(ii) = ldas;
    [fit_b,fit_info]=lasso(X_train,y_train,'Alpha',1,'lambda',ldas,'Standardize',false);
    rmse_train(ii) = sqrt((fit_info.MSE));% compute MSE
    
    %%  (3)find best lamda
    if rmse_train(ii) < error
        error = rmse_train(ii);
        lma = lda_choice(ii);
    end
end

%% step3. 100 times
A=cell(1,100); %training set 
B=cell(1,100); %testing set
rmse_R2=zeros(size(A,2),4); 
rmse_train=zeros(size(A,2),1);
R2_train=zeros(size(A,2),1);
rmse_test=zeros(size(A,2),1);
R2_test=zeros(size(A,2),1);
tic;
for mmm = 1:size(A,2)
% (1) 
    [p_train, t_train, p_test,t_test] = splitData(input, output);   
% (2) run Lasso
    [fit_tr,fit_info_tr]=lasso(p_train,t_train,'Alpha',1,'lambda',lma_best,'Standardize',false);
    [fit_te,fit_info_te]=lasso(p_test,t_test,'Alpha',1,'lambda',lma_best,'Standardize',false);
    predict_1 = p_train*fit_tr+fit_info_tr.Intercept;
    predict_2 = p_test*fit_te+fit_info_te.Intercept;
    
    [rmse_train(mmm), R2_train(mmm)]=cod (t_train,predict_1);
    rmse_R2(mmm,1:2)=[rmse_train(mmm) R2_train(mmm)];
    [rmse_test(mmm), R2_test(mmm)]=cod (t_test,predict_2);
    rmse_R2(mmm,3:4)=[rmse_test(mmm) R2_test(mmm)];

% (4)
    result_train = [t_train predict_1];
    result_test = [t_test predict_2];
    A{1,mmm} = result_train;
    B{1,mmm} = result_test;   
end

% (5) calculate rmse and R2
mean_rmse_R2 = mean(rmse_R2);
[min_rmse,row_rmse]=min(rmse_R2(:,3)); 
min_train=A{1,row_rmse};
min_test=B{1,row_rmse};

%% step 4: plot
%% (1) 
figure(1)
[min_rmse_train,max_R2_train]= cod(min_train(:,1),min_train(:,2));
[min_rmse_test,max_R2_test]= cod(min_test(:,1),min_test(:,2));
plot(min_train(:,1),min_train(:,2),'LineStyle','none', 'Marker','s','MarkerSize',15,...
    'MarkerFace','y');
hold on;
plot(min_test(:,1),min_test(:,2),'LineStyle','none', 'Marker','h','MarkerSize',15,...
    'MarkerFace','b');
lg=legend('Train','Test');
set(lg,'Fontname', 'Times New Roman','FontSize',20,'location','best');
set(gca,'FontSize',20,'LineWidth',1.5);
title(['Training and Testing set of Lasso: ','rmse=',num2str(round(min_rmse_train,2)),'/',num2str(round(min_rmse_test,2)),...
    ' R^2=',num2str(round(max_R2_train,2)),'/',num2str(round(max_R2_test,2))],'Fontsize',16);
axis([-3 3 -3 3]);
hold off;

%% (2) error bar 
figure(2)
%
barmean_rmse_R2=mean_rmse_R2(:,[1 3 2 4]);%
barrmse_R2=rmse_R2(:,[1 3 2 4]);
L=zeros(1,4);
U=zeros(1,4);
for ii = 1:4
    L(:,ii) =barmean_rmse_R2(:,ii)-min(barrmse_R2(:,ii));
    U(:,ii)=max(barrmse_R2(:,ii))-barmean_rmse_R2(:,ii);
end
%
interval=1;
ngroups = 2;
nbars = 2;
groupwidth =min(interval, nbars/(nbars+1.5));
errorbar_x=zeros(1,ngroups+nbars);
counts=1;
for j = 1:ngroups
    for i = 1:nbars      
        errorbar_x(1,counts) = j - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        counts=counts+1;
    end
end
%
handle=bar([barmean_rmse_R2(1:2);barmean_rmse_R2(3:4)],interval);
set(handle(1), 'facecolor', [103 144 200]./255,'edgecolor', [0.8,0.8,0.8]);    
set(handle(2), 'facecolor', [144 200 80]./255,'edgecolor', [0.8,0.8,0.8]);
hold on;
errorbar(errorbar_x,barmean_rmse_R2,L,U,'k','Marker','none','LineStyle','none','LineWidth',1.8);
set(gca,'FontSize',20,'LineWidth',1.5,'XTickLabel',...
    {'rmse','R^2'});
lg=legend('Train','Test');
set(lg,'Fontname', 'Times New Roman','FontSize',20,...
    'location','best','Box','off');
title(('Error Bar of Lasso'),'Fontsize',16);
axis([0 3 0 1.2]);
hold off;

%% (2) error bar_II
figure(3)
%
barmean_rmse_R2=mean_rmse_R2(:,[1 3 2 4]);
barrmse_R2=rmse_R2(:,[1 3 2 4]);
Std=std(barrmse_R2,0,1);
%
interval=1;
ngroups = 2;
nbars = 2;
groupwidth =min(interval, nbars/(nbars+1.5));
errorbar_x=zeros(1,ngroups+nbars);
counts=1;
for j = 1:ngroups
    for i = 1:nbars      
        errorbar_x(1,counts) = j - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        counts=counts+1;
    end
end
%
handle=bar([barmean_rmse_R2(1:2);barmean_rmse_R2(3:4)],interval);
set(handle(1), 'facecolor', [103 144 200]./255,'edgecolor', [0.8,0.8,0.8]);    
set(handle(2), 'facecolor', [144 200 80]./255,'edgecolor', [0.8,0.8,0.8]);
hold on;
errorbar(errorbar_x,barmean_rmse_R2,Std,Std,'k','Marker','none','LineStyle','none','LineWidth',1.8);
set(gca,'FontSize',20,'LineWidth',1.5,'XTickLabel',...
    {'rmse','R^2'});
lg=legend('Train','Test');
set(lg,'Fontname', 'Times New Roman','FontSize',20,...
    'location','best','Box','off');
title(('Error Bar of Lasso'),'Fontsize',16);
axis([0 3 0 1.2]);
hold off;

save('LASSO_figure5.mat');