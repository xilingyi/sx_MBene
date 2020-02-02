%% Support Vector Machine 支持向量机回归-郑靖楠文章1--MCMNS_HER
% 10次10折交叉验证
% 75%作为训练集
% gamma和sigma进行调参
% 运行100次

clearvars;close all;clc;

%% step1. 载入数据
addpath('D:\Program Files (x86)\matlab\bin\sunpaper_II\sunpaper_II_code\图6\LASSO\Functions\'); 
[~, ~, raw] = xlsread('../机器学习全部数据_图5.xlsx');
data = cell2mat(raw(2:end,2:end));
input = data(:,2:end); 
input(:,[3 4 5 6 7]) = []; %rm M_bader D_bader db LM/D-M LM/D-X
output = data(:,1); 

%% step 2: 调参
lmd_best = 2.184045233760190e-04;

%% step3. 运行100次
A=cell(1,100); %放训练集
B=cell(1,100); %放测试集
rmse_R2=zeros(size(A,2),4); %第一、三列是训练集和测试集的rmse；第二、四列是训练集和测试集的R2
rmse_train=zeros(size(A,2),1);
R2_train=zeros(size(A,2),1);
rmse_test=zeros(size(A,2),1);
R2_test=zeros(size(A,2),1);
tic;
for mmm = 1:size(A,2)
% (1) 载入数据
    [p_train, t_train, p_test,t_test] = splitData(input, output);   
% (2) 运行SVM
    [fit_tr,fit_info_tr]=lasso(p_train,t_train,'Alpha',1,'lambda',lmd_best,'Standardize',false);
    [fit_te,fit_info_te]=lasso(p_test,t_test,'Alpha',1,'lambda',lmd_best,'Standardize',false);
    predict_1 = p_train*fit_tr+fit_info_tr.Intercept;
    predict_2 = p_test*fit_te+fit_info_te.Intercept;
    
    [rmse_train(mmm), R2_train(mmm)]=cod (t_train,predict_1);
    rmse_R2(mmm,1:2)=[rmse_train(mmm) R2_train(mmm)];
    [rmse_test(mmm), R2_test(mmm)]=cod (t_test,predict_2);
    rmse_R2(mmm,3:4)=[rmse_test(mmm) R2_test(mmm)];

% (4). 结果对比
    result_train = [t_train predict_1];
    result_test = [t_test predict_2];
    A{1,mmm} = result_train;
    B{1,mmm} = result_test;   
end

% (5) 找性能最好的模型点以及计算平均rmse和R2
mean_rmse_R2 = mean(rmse_R2);
[min_rmse,row_rmse]=min(rmse_R2(:,3)); %定位最大的R2的点
min_train=A{1,row_rmse};
min_test=B{1,row_rmse};

%% step 4: 画图 最好性能模型预测+误差图
%% (1) 性能预测图
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
title(['Training and Testing set of LASSO: ','rmse=',num2str(round(min_rmse_train,2)),'/',num2str(round(min_rmse_test,2)),...
    ' R^2=',num2str(round(max_R2_train,2)),'/',num2str(round(max_R2_test,2))],'Fontsize',16);
axis([-3 3 -3 3]);
hold off;

%% (2) 误差图
figure(2)
%导入需要的数据
barmean_rmse_R2=mean_rmse_R2(:,[1 3 2 4]);%第一、二列是训练集和测试集的rmse；第三、四列是训练集和测试集的R2
barrmse_R2=rmse_R2(:,[1 3 2 4]);
L=zeros(1,4);
U=zeros(1,4);
for ii = 1:4
    L(:,ii) =barmean_rmse_R2(:,ii)-min(barrmse_R2(:,ii));
    U(:,ii)=max(barrmse_R2(:,ii))-barmean_rmse_R2(:,ii);
end
%数据处理
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
%画图
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

%% (2) 误差图_III
figure(3)
%导入需要的数据
barmean_rmse_R2=mean_rmse_R2(:,[1 3 2 4]);%第一、二列是训练集和测试集的rmse；第三、四列是训练集和测试集的R2
barrmse_R2=rmse_R2(:,[1 3 2 4]);
Std=std(barrmse_R2,0,1);
%数据处理
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
%画图
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