%% 

%%
clearvars;close all;clc;

%% Import the data: GH Data
[~, ~, raw] = xlsread('MCMNMB_coh.xlsx');
data = raw(2:end,:);
data([2,4,8,12,17,32,34,35,37,80,85,91,110],:) = [];
%raw = raw(2:92,:);
data(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),data)) = {''};
A = data(:,1);
G_eV = data(:,end);          % Binding energy (eV)


%% Import the information of element
M_M = data(:,2);          % Metal of atomic mass, period, group, radius, number of valence electrons 
P_M = data(:,3);            
G_M = data(:,4);     
r_M = data(:,5);             
Z_M = data(:,6);              
M_X = data(:,7);          % C/N/B of atomic mass, period, group, radius, number of valence electrons
P_X = data(:,8);           
G_X = data(:,9);               
r_X = data(:,10);              
Z_X = data(:,11); 

%% Import the electron structure of element
E_M = data(:,12);   % Metal of electronegativity, electron affinity energy, first ionization potential
A_M = data(:,13);
I_M = data(:,14);
E_X = data(:,15);    % C/N/B of electronegativity, electron affinity energy, first ionization potential
A_X = data(:,16);
I_X = data(:,17);
n = data(:,18);        % Mass ratio

%% Import the charge transfer
q_M = data(:,19);      %不加H的金属，X的电荷转移
q_X = data(:,20);
q_HM = data(:,21);   %加H的金属，X，H的电荷转移
q_HX = data(:,22);
q_H = data(:,23);

%% Import the geometry
F = data(:,24);
a = data(:,25);    % lattice parameters
L = data(:,26);    % Bond length of M-H

%% Import the DOS
d_b = data(:,27);   % d-band center and d-orbital electron number without H adsorption
d_N = data(:,28);
d_Hb = data(:,29); % d-band center and d-orbital electron number with H adsorption
d_HN = data(:,30);
% END OF IMPORT

%% Create D matrix
n_s=size(G_eV,1);
head_1={'M_M', 'P_M', 'G_M', 'r_M', 'Z_M', 'M_X', 'P_X', 'G_X', 'r_X', 'Z_X', ...
    'E_M', 'A_M', 'I_M', 'E_X', 'A_X', 'I_X','n',...
    'q_M', 'q_X','q_HM','q_HX','q_H'...
    'F','a', 'L'...
    'd_b', 'd_N', 'd_Hb', 'd_HN'};
n_ap=size(head_1);
D_p1=[M_M P_M G_M r_M Z_M M_X P_X G_X r_X Z_X ...
    E_M A_M I_M E_X A_X I_X n ...
    q_M q_X q_HM q_HX q_H ...
    F a L...
     d_b d_N d_Hb d_HN];
D_p1=cell2mat(D_p1);
% End of data import

%% information of element
D_P=D_p1(:,1:10);
h_P=head_1(1:10);
unit=ones(1,size(D_P,2));
% 1st stage
list={ '/absu'};
[d_f1,h_f1,u_f1]=genfeature2(D_P,h_P,unit,list,1);
% 2rd stage 
list={'^r'; '^I'; '^2'; 'log'};
[d_ele,h_ele,u_ele]=genfeature(d_f1,h_f1,u_f1,list,1);

%%  the electron structure of element 
D_P=[D_p1(:,11:17)];
h_P=[head_1(11:17)];
unit=ones(1,size(D_P,2));
% Pre-Process
% 1st stage
list={'/absu'};
[d_f1,h_f1,u_f1]=genfeature2(D_P,h_P,unit,list,1);
% 2rd stage 
list={'^r'; '^I'; '^2'; 'log'};
[d_tle,h_tle,u_tle]=genfeature(d_f1,h_f1,u_f1,list,1);

%% the charge transfer
D_P=D_p1(:,18:22);
h_P=head_1(18:22);
unit=ones(1,size(D_P,2));
list={ '-abs', '/absu', '+abs', '*absu'};
[d_f1,h_f1,u_f1]=genfeature2(D_P,h_P,unit,list,1);
list = {'^r'; '^I'; '^2'; 'log'};
[d_fb,h_fb,u_fb]=genfeature(d_f1,h_f1,u_f1,list);

%% the geometry
D_P=D_p1(:,23:25);
h_P=head_1(23:25);
unit=ones(1,size(D_P,2));
list={ '/absu', '*absu'};
[d_f1,h_f1,u_f1]=genfeature2(D_P,h_P,unit,list,1);
list = {'^r'; '^I'; '^2'; 'log'};
[d_fg,h_fg,u_fg]=genfeature(d_f1,h_f1,u_f1,list);

% Check for uniq columns
[C,ia,ic]=unique(d_fg','rows','stable');
d_fg=d_fg(:,ia);
h_fg=h_fg(ia);
u_fg=u_fg(ia);

%% DOS
D_P=D_p1(:,26:29);
h_P=head_1(26:29);
unit=ones(1,size(D_P,2));
list={'-abs', '/absu', '+abs', '*absu'};
[d_f1,h_f1,u_f1]=genfeature2(D_P,h_P,unit,list,1);
list = {'^r'; '^I'; '^2'; 'log'};
[d_fd,h_fd,u_fd]=genfeature(d_f1,h_f1,u_f1,list);


%% 
D_s= [d_ele d_tle d_fb  d_fg d_fd];
h_s= [h_ele h_tle h_fb  h_fg h_fd];

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

save('sx_feature.mat');