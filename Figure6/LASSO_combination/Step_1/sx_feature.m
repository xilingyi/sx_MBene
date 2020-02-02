
%%
clearvars;close all;clc;

%% Import the data: GH Data
[~, ~, raw] = xlsread('../ML_Figure6_rm.xlsx');
data = raw(2:end,:);
data(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),data)) = {''};
A = data(:,1);
G_eV = data(:,2);          


%% DFT calculation 
F = data(:,3);            
D = data(:,4);          
a = data(:,6);

%% elemental geometry
MM = data(:,7); 
rM = data(:,8);
MD = data(:,13);
GD = data(:,14);
rD = data(:,15);

%% elemental electron
ZM = data(:,9);   
EM = data(:,10);
AM = data(:,11);
IM = data(:,12);
ED = data(:,16);
AD = data(:,17);
ID = data(:,18);

%% ratio of layers
n = data(:,5);

%% Create D matrix
n_s=size(G_eV,1);
head_1={'F', 'D', 'a'...
    'MM', 'rM','MD', 'GD', 'rD',...
    'ZM', 'EM', 'AM', 'IM', 'ED', 'AD', 'ID',...
    'n'};
n_ap=size(head_1);
D_p1=[F D a...
    MM rM MD GD rD ...
    ZM EM AM IM ED AD ID ...
    n];
D_p1=cell2mat(D_p1);
% End of data import

%% dft : 
D_P=D_p1(:,1:3);
h_P=head_1(1:3);
unit=ones(1,size(D_P,2));
% 1st stage
list={'/absu'};
[d_f1,h_f1,u_f1]=genfeature2(D_P,h_P,unit,list,1);
% 2rd stage 
list = {'^r'; '^I'; '^2'; 'log'};
[d_d,h_d,u_d]=genfeature(d_f1,h_f1,u_f1,list,1);


%%  elemental geometry 
D_P=[D_p1(:,[4:8 16])];
h_P=[head_1([4:8 16])];
unit=ones(1,size(D_P,2));
% 1st stage
list={'/absu';'-absu'};
[d_f1,h_f1,u_f1]=genfeature2(D_P,h_P,unit,list,1);
% 2rd stage 
list={'^r';'^I';'^2';'log'};
[d_eg,h_eg,u_eg]=genfeature(d_f1,h_f1,u_f1,list,1);

%%  elemental electron
D_P=[D_p1(:,9:16)];
h_P=[head_1(9:16)];
unit=ones(1,size(D_P,2));
% 1st stage
list={'/absu';'-absu'};
[d_f1,h_f1,u_f1]=genfeature2(D_P,h_P,unit,list,1);
% 2rd stage 
list={'^r';'^I';'^2';'log'};
[d_ee,h_ee,u_ee]=genfeature(d_f1,h_f1,u_f1,list,1);


%% 
D_s= [d_d  d_eg d_ee];
h_s= [h_d h_eg h_ee];

% Final cleaning just in case
c_col=find(any(isinf(D_s)) | any(isnan(D_s)));
D_s(:,c_col)=[];
h_s(:,c_col)=[];

%% 
% special function to avoid multiplication that produces columns with 1
list={'*absI'};
unit=ones(1,size(D_s,2));
[d_f1,h_f1,u_f1]=genfeature2p(D_s,h_s,unit,list,1);

% keep unique descriptors only
[C,ia,ic]=unique(d_f1','rows','stable');

% final unscaled descriptor dataset
d_f1=d_f1(:,ia);
% final headers 
h_f1=h_f1(ia);
% final unit flags - not going to be used in future
u_f1=u_f1(ia);

% save('zjn_特征值复合数据.mat')
save('sx_图6特征值复合.mat')


