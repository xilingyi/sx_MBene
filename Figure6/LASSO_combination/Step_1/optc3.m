function [D_f2,head_f2,unit_out,n_c2] = optc3(D_p,header,unit_in,opt_flag)
    sz=size(D_p); % number of options 
    index=1:sz(2); % create index to choose which ones to operate on
    n_c2=[1:1:size(index,2)-1;repelem(size(index,2),size(index,2)-1)]'; % Number of possible combinations (ncr)
    sz2=size(n_c2); % limit for total number of combinations.
    %disp(n_c2)
    % activate this section if ratio is not after an inverse operation
    %if (strncmp(opt_flag,'/',1))    % special case for division the order matters- A/B and B/A both are considered
    %    D_f2=zeros(sz(1),2*sz2(1)); % Size of output matrix will be double for ratio case
    %    head_f2=cell(1,2*sz2(1));
    %else
        D_f2=zeros(sz(1),sz2(1)); % For all other cases the output matrix size is ncr
        head_f2=cell(1,sz2(1));
        unit_out=zeros(1,sz2(1));
    %end
    
    
            for i=1:sz2(1)
                switch opt_flag
                    case '-'
                            D_f2(:,i)=D_p(:,n_c2(i,1))-D_p(:,n_c2(i,2));
                            unit_out(i)=unit_in(n_c2(i,1));
                            head_f2(i)=strcat('(',header(n_c2(i,1)),'-',header(n_c2(i,2)),')');                       
                    case '+'
                        D_f2(:,i)=D_p(:,n_c2(i,1))+D_p(:,n_c2(i,2));
                        unit_out(i)=unit_in(n_c2(i,1));
                        head_f2(i)=strcat('(',header(n_c2(i,1)),'+',header(n_c2(i,2)),')');
                    case '/'
                        D_f2(:,i)=D_p(:,n_c2(i,1))./D_p(:,n_c2(i,2));
                        unit_out(i)=unit_in(n_c2(i,1))-unit_in(n_c2(i,2));
                        head_f2(i)=strcat('(',header(n_c2(i,1)),'/',header(n_c2(i,2)),')');
                        % activate if inverse operation is not applied
                        % first
                        %D_f2(:,i+sz2(1))=D_p(:,n_c2(i,2))./D_p(:,n_c2(i,1));
                        %head_f2(i+sz2(1))=strcat(header(n_c2(i,2)),'/',header(n_c2(i,1)));
                    case '/u'
                        D_f2(:,i)=D_p(:,n_c2(i,1))./D_p(:,n_c2(i,2));
                        unit_out(i)=unit_in(n_c2(i,1))-unit_in(n_c2(i,2));
                        head_f2(i)=strcat('(',header(n_c2(i,1)),'/',header(n_c2(i,2)),')');
                        % activate if inverse operation is not applied
                        % first
                        %D_f2(:,i+sz2(1))=D_p(:,n_c2(i,2))./D_p(:,n_c2(i,1));
                        %head_f2(i+sz2(1))=strcat(header(n_c2(i,2)),'/',header(n_c2(i,1)));
                    case '*'
                        D_f2(:,i)=D_p(:,n_c2(i,1)).*D_p(:,n_c2(i,2));
                        unit_out(i)=unit_in(n_c2(i,1))+unit_in(n_c2(i,2));
                        head_f2(i)=strcat('(',header(n_c2(i,1)),'*',header(n_c2(i,2)),')');
                    case '/abs'
                        D_f2(:,i)=abs(D_p(:,n_c2(i,1))./D_p(:,n_c2(i,2)));
                        unit_out(i)=unit_in(n_c2(i,1))-unit_in(n_c2(i,2));
                        head_f2(i)=strcat('|',header(n_c2(i,1)),'/',header(n_c2(i,2)),'|');
                        % activate if inverse operation is not applied
                        % first
                        %D_f2(:,i+sz2(1))=abs(D_p(:,n_c2(i,2))./D_p(:,n_c2(i,1)));
                        %head_f2(i+sz2(1))=strcat('|',header(n_c2(i,2)),'/',header(n_c2(i,1)),'|');                        
                    case '/absu' %ratio that considers units
                        D_f2(:,i)=abs(D_p(:,n_c2(i,1))./D_p(:,n_c2(i,2)));
                        unit_out(i)=unit_in(n_c2(i,1))-unit_in(n_c2(i,2));
                        head_f2(i)=strcat('|',header(n_c2(i,1)),'/',header(n_c2(i,2)),'|');
                    case '*abs'
                        D_f2(:,i)=abs(D_p(:,n_c2(i,1)).*D_p(:,n_c2(i,2)));
                        unit_out(i)=unit_in(n_c2(i,1))+unit_in(n_c2(i,2));
                        head_f2(i)=strcat('|',header(n_c2(i,1)),'*',header(n_c2(i,2)),'|');
                    case '*absu'
                        D_f2(:,i)=abs(D_p(:,n_c2(i,1)).*D_p(:,n_c2(i,2)));
                        unit_out(i)=unit_in(n_c2(i,1))+unit_in(n_c2(i,2));
                        head_f2(i)=strcat('|',header(n_c2(i,1)),'*',header(n_c2(i,2)),'|');
                    case '-abs'
                        D_f2(:,i)=abs(D_p(:,n_c2(i,1))-D_p(:,n_c2(i,2)));
                        unit_out(i)=unit_in(n_c2(i,1));
                        head_f2(i)=strcat('|',header(n_c2(i,1)),'-',header(n_c2(i,2)),'|');
                    case '+abs'
                        D_f2(:,i)=abs(D_p(:,n_c2(i,1))+D_p(:,n_c2(i,2)));
                        unit_out(i)=unit_in(n_c2(i,1));
                        head_f2(i)=strcat('|',header(n_c2(i,1)),'+',header(n_c2(i,2)),'|');
                    case '-^2' %dont use
                        D_f2(:,i)=(D_p(:,n_c2(i,1))-D_p(:,n_c2(i,2))).^2;
                        head_f2(i)=strcat('|',header(n_c2(i,1)),'-',header(n_c2(i,2)),'|','^2');
                    case '+^2' % dont use
                        D_f2(:,i)=(D_p(:,n_c2(i,1))+D_p(:,n_c2(i,2))).^2;
                        head_f2(i)=strcat('|',header(n_c2(i,1)),'+',header(n_c2(i,2)),'|','^2');
                    case '/^2' % dont use 
                        D_f2(:,i)=(D_p(:,n_c2(i,1))./D_p(:,n_c2(i,2))).^2;
                        head_f2(i)=strcat('|',header(n_c2(i,1)),'./',header(n_c2(i,2)),'|','^2');
                    case '-^2/3' % dont use 
                        D_f2(:,i)=((D_p(:,n_c2(i,1))-D_p(:,n_c2(i,2))).^(2)).^(1/3);
                        head_f2(i)=strcat('|',header(n_c2(i,1)),'-',header(n_c2(i,2)),'|','^(2/3)');
                    otherwise
                        disp('unknown operation')
                end
            end
        %size(D_f2);    
        unique= find(any(D_f2));   
        D_f2=(D_f2(:,unique));
        head_f2=(head_f2(unique));
        unit_out=unit_out(unique);
        
end