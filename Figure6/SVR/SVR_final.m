%% Support Vector Machine 
% 10 times 10K-CV
% 75% training set
% tune g and c

clearvars;close all;clc;

%% step1. load dataset
addpath('Functions'); 
[~, ~, raw] = xlsread('../ML_Figure6.xlsx');
data = cell2mat(raw(2:end,2:end));
input = data(:,2:end); 
input(:,[3 4 5 6 7]) = []; %rm M_bader D_bader db LM/D-M LM/D-X
output = data(:,1)'; 

%% step2. tune parameter
trialsNum = 10; % repeat 10 times
c = 2.^(0:0.05:3);
g = 2.^(-10:0.5:10);
[C, G] = meshgrid(c, g); % divide the grid
[mm, nn] = size(C);
C = C(:);G = G(:);
dimNum = length(G);
C_choice = zeros(trialsNum, 1); % preallocate storage
ids = zeros(trialsNum,1); % preallocate storage
G_choice = zeros(trialsNum, 1); % preallocate storage
bestc = 0;
bestg = 0;
error  = inf;
mse_train = zeros(trialsNum, 1);

%% step 3: 10 times 10K-CV 
for ii = 1:trialsNum 
    %% (1) 10K-CV  
    [X_train, y_train, X_test,y_test] = splitData(input, output); % split train datas and test datas
    fold10Score = zeros(dimNum, 1);% preallocate storage
    for jj = 1:dimNum
        fold10Score(jj) = kFoldScore(X_train, y_train, C(jj), G(jj), 10); % use 10-fold cross-validation to compute
    end
    [val, id] = min(fold10Score);
    ids(ii) = id;
    c = C(id);
    g = G(id);
    
    %  (2) run 10 times
    C_choice(ii) = c;
    G_choice(ii) = g;

    % trainingset
    [pn_train,inputps] = mapminmax(X_train');
    pn_train = pn_train';
    % testingset
    [tn_train,outputps] = mapminmax(y_train');
    tn_train = tn_train';
        
    cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 4 -p 0.001'];
    model = svmtrain(tn_train,pn_train,cmd);
    [Predict_1,~] = svmpredict(tn_train,pn_train,model);
    predict_1 = mapminmax('reverse',Predict_1,outputps);
    mse_train(ii) = ((predict_1-y_train)'*(predict_1-y_train))/size(predict_1,1);
    
    %%  (3)find best c and g
    if mse_train(ii) < error
        error = mse_train(ii);
        bestc = C_choice(ii);
        bestg = G_choice(ii);
    end
end

%% step3. run 100 times
A=cell(1,100); %trainingset
B=cell(1,100); %testingset
rmse_R2=zeros(size(A,2),4); 
rmse_train=zeros(size(A,2),1);
R2_train=zeros(size(A,2),1);
rmse_test=zeros(size(A,2),1);
R2_test=zeros(size(A,2),1);
tic;
for mmm = 1:size(A,2)
% (1) 
    [p_train, t_train, p_test,t_test] = splitData(input, output);
    % 
    [pn_train,inputps] = mapminmax(p_train');
    pn_train = pn_train';
    pn_test = mapminmax('apply',p_test',inputps);
    pn_test = pn_test';
    %
    [tn_train,outputps] = mapminmax(t_train');
    tn_train = tn_train';
    tn_test = mapminmax('apply',t_test',outputps);
    tn_test = tn_test';
    
% (2) run SVM
    cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 4 -p 0.0001'];
    model = svmtrain(tn_train,pn_train,cmd);
    [Predict_1,error_1] = svmpredict(tn_train,pn_train,model);
    [Predict_2,error_2] = svmpredict(tn_test,pn_test,model);

% (3). 
    predict_1 = mapminmax('reverse',Predict_1,outputps);
    predict_2 = mapminmax('reverse',Predict_2,outputps);
    
    [rmse_train(mmm), R2_train(mmm)]=cod (t_train,predict_1);
    rmse_R2(mmm,1:2)=[rmse_train(mmm) R2_train(mmm)];
    [rmse_test(mmm), R2_test(mmm)]=cod (t_test,predict_2);
    rmse_R2(mmm,3:4)=[rmse_test(mmm) R2_test(mmm)];

% (4). 
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
title(['Training and Testing set of SVM: ','rmse=',num2str(round(min_rmse_train,2)),'/',num2str(round(min_rmse_test,2)),...
    ' R^2=',num2str(round(max_R2_train,2)),'/',num2str(round(max_R2_test,2))],'Fontsize',16);
axis([-3 3 -3 3]);
hold off;

%% (2) error bar
figure(2)
%
barmean_rmse_R2=mean_rmse_R2(:,[1 3 2 4]);
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
for jj = 1:ngroups
    for i = 1:nbars      
        errorbar_x(1,counts) = jj - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
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
title(('Error Bar of SVM'),'Fontsize',16);
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
for jj = 1:ngroups
    for i = 1:nbars      
        errorbar_x(1,counts) = jj - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
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
title(('Error Bar of SVM'),'Fontsize',16);
axis([0 3 0 1.2]);
hold off;

save('SVR_figure5.mat');