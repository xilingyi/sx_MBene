%% load dataset generated in data generation step 1
clearvars;close all;clc;
load('..\Step_1\sx_feature.mat')

%% Lasso Test

disp('Running sample LASSO analysis');

NM=[1:size(G_eV,1)];

d_f1=d_f1(NM,:);
G_eV=cell2mat(G_eV(NM));

% Normalize dataset columns
[D_Total_s,mu,sigma]=zscore(d_f1);
P_c_temp = G_eV;
P_c = P_c_temp - mean(P_c_temp);

% initial LASSO run to get max lambda
lm_max=max(abs(transpose(D_Total_s)*P_c))/length(P_c);
disp('lambda_max')
disp(lm_max)

%% Initial Lasso done
disp('Initializing LASSO run with varying lamda value');

lda = logspace(-3,0,100)*lm_max;
[fit_b,fit_info]=lasso(D_Total_s,P_c,'Alpha',1,'lambda',lda,'Standardize',false);
dm=size(fit_b);

%data_mat=zeros(100,40);
coeff=[];

% open file to write results
fileID = fopen('lasso_param.txt','w');
%%
for i=dm(2):-1: 80
    temp=find(fit_b(:,i));
    coeff=union(temp,coeff);
    fprintf(fileID,'Iteration: %d, Lambda: %f\n',i,fit_info.Lambda(i))
    for j=1:length(temp)
        fprintf(fileID,'%s\n',char(h_f1(temp(j))))
        fprintf(fileID,'%d\n',temp(j))
    end
end

fprintf(fileID, '-------End of Lasso Step------------------\n')
fprintf(fileID,'Total Number of Coeff.: %d\n', length(coeff))
fprintf(fileID,'list of Coeff. \n')
fprintf(fileID,'index\tCoeff. \n')

for j=1:length(coeff)
    fprintf(fileID,'%d\t%s\n', coeff(j), char(h_f1(coeff(j))))
end

fclose(fileID);

disp('Saving Data');
% save dataframe for next step
save('sx_lasso.mat')
