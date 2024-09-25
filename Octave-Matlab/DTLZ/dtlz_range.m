function lim = dtlz_range(fname, M)
%DTLZ_RANGE Returns the decision range of a DTLZ function
%   The range is simply [0,1] for all variables. What varies is the number 
%   of decision variables in each problem. The equation for that is
%     n = (M-1) + k
%   wherein k = 5 for DTLZ1, 10 for DTLZ2-6, and 20 for DTLZ7.
%   
%   Syntax:
%      lim = get_range(fname, M)
%
%   Input arguments:
%      fname: a string with the name of the function ('dtlz1', 'dtlz2' etc.)
%      M: a scalar with the number of objectives
%
%   Output argument:
%      lim: a n x 2 matrix wherein the first column is the inferior limit 
%           (0), and the second column, the superior limit of search (1)

% Checks if the string has or not the prefix 'dtlz', or if the number later
% is greater than 7
fname = lower(fname);
if strcmp(fname, 'delq')
    
    %DE决策变量上下限
    limtmp=[];
    limtmp =[limtmp;300 500];  %变量1上下限-Batch大小，至少要有...个
    limtmp =[limtmp;1 9];      %变量2上下限-Training times
    limtmp =[limtmp;1 6];      %变量3上下限-噪声数量级
    for i=1:8
        limtmp =[limtmp;1 5];  %变量4~11上下限-Q数量级
    end
    for i=1:8
        limtmp =[limtmp;3 7];  %变量12~19上下限-R数量级
    end
    limtmp =[limtmp;0.1 0.9999];%变量20上下限-折扣因子
    
else    
    
    if length(fname) < 5 || ~strcmp(fname(1:4), 'dtlz') || ...
            str2double(fname(5)) > 7
        error(['Sorry, the function ' fname ' is not implemented.'])
    end
    % If the name is o.k., defines the value of k
    if strcmp(fname, 'dtlz1')
        k = 5;
    elseif strcmp(fname, 'dtlz7')
        k = 20;
    else %any other function
        k = 10;
    end    
    n = (M-1) + k; %number of decision variables    
    limtmp = [zeros(n,1) ones(n,1)];
    
end
lim=limtmp;


