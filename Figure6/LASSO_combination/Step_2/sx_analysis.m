%% 
% Code written by 
% A S M Jonayat
% Ph.D. Candidate
% Dept. of Mechanical and Nuclear Engineering
% The Pennsylvania State University, University Park - 16802
% 
% Distributed as part of the publication - 
% Interaction trends between single metal atoms and oxide supports identified with density functional theory and statistical learning
% Nolan J. Oâ€™Connor, A S M Jonayat, Michael J. Janik*, Thomas P. Senftle*

% Department of Chemical Engineering, bDepartment of Mechanical and Nuclear Engineering, The Pennsylvania State University, University Park, PA 16802 (USA)
% Department of Chemical and Biomolecular Engineering, Rice University, Houston, TX 77005 (USA)
% *mjanik@engr.psu.edu
% *tsenftle@rice.edu

%% load dataset generated in data generation step 1. Change to appor. location.
clearvars;close all;clc;
load('D:\Program Files (x86)\matlab\bin\sunpaper_II\sunpaper_II_code\Í¼6\LASSO_combination\Step_1\sx_Í¼6ÌØÕ÷Öµ¸´ºÏ.mat')

%% Lasso Test

disp('Running sample LASSO analysis');

% choose the first 91 data points, choose random if you want to cross
% validate

% NM=[1:186];
NM=[1:length(G_eV)];

d_f1=d_f1(NM,:);
G_eV=cell2mat(G_eV(NM));

% Normalize dataset columns
[D_Total_s,mu,sigma]=zscore(d_f1);
P_c_temp = G_eV;
P_c = P_c_temp - mean(P_c_temp);

% initial LASSO run to get max lambda
%[fit_b_t,fit_info_t]=lasso(D_Total_s,P_c,'Alpha',1,'Standardize',false);
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
% fileID = fopen('zjn_lasso_param.txt','w');
fileID = fopen('sx_lasso_final.txt','w');
%%
for i=dm(2):-1: 70
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
% save('zjn_lasso_final.mat')
save('sx_lasso_final.mat')
