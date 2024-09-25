function [fopt, xopt] = demo_opt_decop(f, xrange, paras, options)
global rp;
global gen;
global individual_mut_record;
global prob1;
global prob2;
prob1=0.33;
prob2=0.33;
mut_mode='mutation_indmut_rand';%mutation_indmut_history // mutation_popmut_history // 
memory_horizon=5;
prob_update_gap=5;
mutator_employ_mat=zeros(memory_horizon,3);
mutator_survive_mat=zeros(memory_horizon,3);
%DEMO_OPT: Multi-objective optimization using the DEMO
%   This function uses the Differential Evolution for Multi-objective
%   Optimization (a.k.a. DEMO) to solve a multi-objective problem. The
%   result is a set of nondominated points that are (hopefully) very close
%   to the true Pareto front.
%   ===========DELQ problem=============
%   f = @(x) delq(x, 2);
%   xrange = dtlz_range('delq', 2);
%   [fopt, xopt] = demo_opt(f, xrange)
%   ========================
%   Syntax:
%      [fopt, xopt] = demo_opt(f, xrange)
%      [fopt, xopt] = demo_opt(f, xrange, options)
%
%   Input arguments:
%      f: the objective function given as a handle, a .m file, inline, or
%         anything that can be computed using the "feval" function and can
%         handle multiple inputs. The output must be, for each point, a
%         column vector of size m x 1, with m > 1 the number of objectives.
%      xrange: a matrix with the inferior and superior limits of search.
%              If n is the dimension, it will be a n x 2 matrix, such
%              that the first column contains the inferior limit, and the
%              second, the superior one;
%      options: a struct with internal parameters of the algorithm:
%         .F: the scale factor to be used in the mutation (default: 0.5);
%         .CR: the crossover factor used in the recombination (def.: 0.3);
%         .mu: the population size (number of individuals) (def.: 100);
%         .kmax: maximum number of iterations (def.: 300);
%         .display: 'on' to display the population while the algorithm is
%                   being executed, and 'off' to not (default: 'off');
%         If any of the parameters is not set, the default ones are used
%         instead.
%
%   Output arguments:
%      fopt: the m x mu_opt matrix with the mu_opt best objectives
%      xopt: the n x mu_opt matrix with the mu_opt best individuals
%
%   Example: Solving a DTLZ problem with 3 objectives
%   % First, make sure you have the DTLZ directory in your path
%   % Now, create a function handle for the DTLZ3 with 3 objectives
%   f = @(x) dtlz3(x, 3);
%   % And, the limits of the search space, which can be found using the
%   % convenient function DTLZ_RANGE:
%   xrange = dtlz_range('dtlz3', 3);
%   % Finally, execute the whole thing (go make some coffee while waiting)
%   [fopt, xopt] = demo_opt(f, xrange);
%   % If you wanna see the population advancing, you can create a 'options'
%   % struct with a 'display' field set to 'on':
%   options.display = 'on';
%   [fopt, xopt] = demo_opt(f, xrange, options);

% Check the parameters of the algorithm
if nargin < 4 %options was not provided
    options = struct();
end
options = check_input(options);
if options.mu==100
    weight=importdata('.\weights\W3D_100.txt');
end
% Initial considerations
fprintf('=============Initial===========')
n = size(xrange,1); %dimension of the problem
P.x = rand(n, options.mu); %initial decision variables
gen=0;
P.f = fobjeval(f, P.x, xrange); %evaluates the initial population
m = size(P.f, 1); %number of objectives
k = 1; %iterations counter

ispar = ndset(P.f);
NSS_pf = P.f(:,ispar);
NSS_pfvar=P.x(:,ispar);

starttime = clock;
P_min=[];

% Beginning of the main loop
while k <= options.kmax
    % Plot the current population (if desired)
    if strcmp(options.display, 'on')
        if m == 2
            plot(P.f(1,:), P.f(2,:), 'o');
            title('Objective values during the execution')
            xlabel('f_1'), ylabel('f_2')
            drawnow
        elseif m == 3
            plot3(P.f(1,:), P.f(2,:), P.f(3,:), 'o');
            title('Objective values during the execution')
            xlabel('f_1'), ylabel('f_2'), zlabel('f_3')
            drawnow
        end
    end
    fprintf('===========Episode %d========Iteration %d===========\n',rp, k)
    gen=k;
    % Perform the variation operation (mutation and recombination)
    if strcmp(mut_mode,'mutation_indmut_history')
        O.x = mutation_indmut_history(P.x,P.f, NSS_pfvar, NSS_pf,weight, options); %mutation
        memory_tmp=mutator_count(individual_mut_record);
        mutator_employ_mat=[mutator_employ_mat(2:size(mutator_employ_mat,1),:);memory_tmp];%mutator
        mutator_employ_sum=sum(mutator_employ_mat);%每一列求和
        
    elseif strcmp(mut_mode,'mutation_indmut_rand')
        O.x = mutation_indmut_rand(P.x,P.f, NSS_pfvar, NSS_pf,weight, options); %mutation
        
    elseif strcmp(mut_mode,'mutation_popmut_rand')
        O.x = mutation_popmut_rand(P.x,P.f, NSS_pfvar, NSS_pf,weight, options); %mutation
        
    elseif strcmp(mut_mode,'mutation_popmut_history')
        O.x = mutation_popmut_history(P.x,P.f, NSS_pfvar, NSS_pf,weight, options); %mutation
        memory_tmp=mutator_count(individual_mut_record);
        mutator_employ_mat=[mutator_employ_mat(2:size(mutator_employ_mat,1),:);memory_tmp];%mutator
        mutator_employ_sum=sum(mutator_employ_mat);%
    end
    
    
    
    O.x = recombination(P.x, O.x, options); %recombination    
    O.x = repair(O.x);  
    O.f = fobjeval(f, O.x, xrange); %compute objective functions
    Pop_backup=O.x;
    
    % Selection and updates    
    P = selection(P, O, options);
    
    if strcmp(mut_mode,'mutation_indmut_history')||strcmp(mut_mode,'mutation_popmut_history')%
        for i=1:size(Pop_backup,2)
            flag_survive=0;       
            for j=1:size(P.x,2)
               if isequal(Pop_backup(:,i),P.x(:,j))
                   flag_survive=1;
                   break;
               end                
            end
            if flag_survive==0
                individual_mut_record(i)=0;
            end            
        end
        memory_tmp=mutator_count(individual_mut_record);
        mutator_survive_mat=[mutator_survive_mat(2:size(mutator_survive_mat,1),:);memory_tmp];%mutator
        mutator_survive_sum=sum(mutator_survive_mat);%
        survive_rate=mutator_survive_sum./mutator_employ_sum;
        survive_rate_norm=survive_rate./sum(survive_rate);
        if mod(k,prob_update_gap)==0
            prob1=survive_rate_norm(1);
            prob2=survive_rate_norm(2);
        end        
    end
    
    p_min=P.f';
    P_min=[P_min;min(p_min)];    
    %update NSS
    comb_pf = [NSS_pf P.f];
    comb_pfvar=[NSS_pfvar P.x];
    ispar = ndset(comb_pf);
    NSS_pf = comb_pf(:,ispar);
    NSS_pfvar=comb_pfvar(:,ispar);
    k = k + 1;
end

% Return the final population
% First, unnormalize it
Xmin = repmat(xrange(:,1), 1, options.mu); %replicate inferior limit
Xmax = repmat(xrange(:,2), 1, options.mu); %replicate superior limit



span=etime(clock, starttime);
fprintf('total time used %u\n', span);
fname=['D:\DEMO-master\data-0917\MODE_time_p',num2str(p1),'p',num2str(p2),'_',num2str(rp), '.txt'];
save(fname, 'span' ,'-ascii')
fname=['D:\DEMO-master\data-0917\MODE_min_p',num2str(p1),'p',num2str(p2),'_',num2str(rp), '.txt'];
save(fname, 'P_min' ,'-ascii')

%
Xun = (Xmax - Xmin).*P.x + Xmin;
ispar = ndset(P.f);
fopttmp = P.f(:,ispar);
xopttmp = Xun(:,ispar);
fopttmp=fopttmp';
fname=['D:\DEMO-master\data-0917\MODE_PFs_p',num2str(p1),'p',num2str(p2),'_',num2str(rp), '.txt'];
save(fname, 'fopttmp' ,'-ascii')
fname=['D:\DEMO-master\data-0917\MODE_PFVaviables_p',num2str(p1),'p',num2str(p2),'_',num2str(rp), '.txt'];
save(fname, 'xopttmp' ,'-ascii')
fopt = fopttmp;
xopt = xopttmp;


%=========================== Sub-functions ================================%
%-----------------------------check---------------------------------------------%
function phi = fobjeval(f, x, xrange)
%FOBJEVAL Evaluates the objective function
%   Since the population is normalized, this function unnormalizes it and
%   computes the objective values
%
%   Syntax:
%      phi = fobjeval(f, x, options)
%
%   Input arguments:
%      f: the objective function given as a handle, a .m file, inline, or
%         anything that can be computed using the "feval" function and can
%         handle multiple inputs
%      x: a n x mu matrix with mu individuals (points) and n variables
%         (dimension size)
%      options: the struct with the parameters of the algorithm
%
%   Output argument:
%      phi: a m x mu matrix with the m objective values of the mu
%           individuals

mu = size(x, 2); %number of points
% Unnormalizes the population
Xmin = repmat(xrange(:,1), 1, mu); %replicates inferior limit
Xmax = repmat(xrange(:,2), 1, mu); %replicates superior limit
Xun = (Xmax - Xmin).*x + Xmin;

phi = feval(f, Xun);
%--------------------------------------------------------------------------%

%原始
function Xo = mutation(Xp, options)
%MUTATION Performs mutation in the individuals
%   The mutation is one of the operators responsible for random changes in
%   the individuals. Each parent x will have a new individual, called trial
%   vector u, after the mutation.
%   To do that, pick up two random individuals from the population, x2 and
%   x3, and creates a difference vector v = x2 - x3. Then, chooses another
%   point, called base vector, xb, and creates the trial vector by
%
%      u = xb + F*v = xb + F*(x2 - x3)
%
%   wherein F is an internal parameter, called scale factor.
%
%   Syntax:
%      Xo = mutation(Xp, options)
%
%   Input arguments:
%      Xp: a n x mu matrix with mu "parents" and of dimension n
%      options: the struct with the internal parameters
%
%   Output arguments:
%      Xo: a n x mu matrix with the mu mutated individuals (of dimension n)

% Creates a mu x mu matrix of 1:n elements on each row
A = repmat((1:options.mu), options.mu, 1);
% Now, as taken at the MatLab Central, one removes the diagonal of
% A, because it contains indexes that repeat the current i-th
% individual
A = A';
A(logical(eye(size(A)))) = []; %removes the diagonal
A = transpose(reshape(A, options.mu-1, options.mu)); %reshapes

% Now, creates a matrix that permutes the elements of A randomly
[~, J] = sort(rand(size(A)),2);
Ilin = bsxfun(@plus,(J-1)*options.mu,(1:options.mu)');
A(:) = A(Ilin);

% Chooses three random points (for each row)
xbase = Xp(:, A(:,1)); %base vectors
v = Xp(:,A(:,2)) - Xp(:,A(:,3)); %difference vector

% Performs the mutation
Xo = xbase + options.F*v;


function Xo = mutation_indmut_rand(Xp, Xf, Pp, Pf, weights, options)
%MUTATION Performs mutation in the individuals
%   The mutation is one of the operators responsible for random changes in
%   the individuals. Each parent x will have a new individual, called trial
%   vector u, after the mutation.
%   To do that, pick up two random individuals from the population, x2 and
%   x3, and creates a difference vector v = x2 - x3. Then, chooses another
%   point, called base vector, xb, and creates the trial vector by
%
%      u = xb + F*v = xb + F*(x2 - x3)
%
%   wherein F is an internal parameter, called scale factor.
%
%   Syntax:
%      Xo = mutation(Xp, options)
%
%   Input arguments:
%      Xp: a n x mu matrix with mu "parents" and of dimension n
%      options: the struct with the internal parameters
%
%   Output arguments:
%      Xo: a n x mu matrix with the mu mutated individuals (of dimension n)
XPOP=size(Xp,2);
Xo=zeros(size(Xp,1),size(Xp,2));
for xpop=1:XPOP                       % 对每个个体
    single_parent=Xp(:,xpop);       % 取出当前个体
    rev=randperm(XPOP);                 % 对种群Xpop的序号随机排序
    id1=rev(1,1);
    id2=rev(1,2);
    id3=rev(1,3);
    id4=rev(1,4);
    id5=rev(1,5);
    if id1==xpop
        id1=rev(1,6);
    elseif id2==xpop
        id2=rev(1,6);
    elseif id3==xpop
        id3=rev(1,6);
    elseif id4==xpop
        id4=rev(1,6);
    elseif id5==xpop
        id5=rev(1,6);
    end
    tmprd=rand();
    
    if tmprd<0.3                    % 从帕累托解集寻找最优
        PF_size=size(Pf,2);         % 当前PF中解数量
        Candidate=zeros(1,PF_size); % PF解适应度值数组
        for nds = 1:PF_size
            Candidate(1,nds)=dot(weights(xpop,:),Pf(:,nds)); % 计算所有PF解在当前个体代表权重下的适应度值
        end
        [~, b_index] = min(Candidate);                              % 选出PF里面在当前个体对应权重下表现最好的个体
        Mutant = single_parent+options.F*(single_parent-Pp(:,b_index))...
            +options.F*(Xp(:,id1)-Xp(:,id2)); % current-to-best定义
    elseif tmprd>=0.3 && tmprd <0.6 % 从当前种群寻找最优
        Candidate=zeros(1,XPOP);
        for nds = 1:XPOP
            Candidate(1,nds)=dot(weights(xpop,:),Xf(:,nds));   % 计算所有当前种群解在当前个体代表权重下的适应度值
        end
        [~, b_index] = min(Candidate);                              % 选出当前种群里面在当前个体对应权重下表现最好的个体
        Mutant= single_parent+options.F*(single_parent-Xp(:,b_index))...
            +options.F*(Xp(:,id1)-Xp(:,id2)); % current-to-best定义
        %          Mutant= single_parent+options.F*...
        %             (Xp(:,id1)-Xp(:,id2))+options.F*...
        %             (Xp(:,id3)-Xp(:,id4));
    else                            %从当前种群随机寻找
        %         Mutant= single_parent+options.F*...
        %             (Xp(:,id1)-Xp(:,id2))+options.F*...
        %             (Xp(:,id3)-Xp(:,id4));
        
        Mutant= Xp(:,id1)+options.F*...
            (Xp(:,id2)-Xp(:,id3))+options.F*...
            (Xp(:,id4)-Xp(:,id5));
        
    end
    Xo(:,xpop)=Mutant;
end


function Xo = mutation_indmut_history(Xp, Xf, Pp, Pf, weights, options)
%MUTATION Performs mutation in the individuals
global individual_mut_record;
global prob1;
global prob2;

XPOP=size(Xp,2);
Xo=zeros(size(Xp,1),size(Xp,2));


function Xo = mutation_popmut_rand(Xp, Xf, Pp, Pf, weights, options)
%MUTATION Performs mutation in the individuals
%   The mutation is one of the operators responsible for random changes in
%   the individuals. Each parent x will have a new individual, called trial
%   vector u, after the mutation.
%   To do that, pick up two random individuals from the population, x2 and
%   x3, and creates a difference vector v = x2 - x3. Then, chooses another
%   point, called base vector, xb, and creates the trial vector by
%
%      u = xb + F*v = xb + F*(x2 - x3)
%
%   wherein F is an internal parameter, called scale factor.
%
%   Syntax:
%      Xo = mutation(Xp, options)
%
%   Input arguments:
%      Xp: a n x mu matrix with mu "parents" and of dimension n
%      options: the struct with the internal parameters
%
%   Output arguments:
%      Xo: a n x mu matrix with the mu mutated individuals (of dimension n)
XPOP=size(Xp,2);
Xo=zeros(size(Xp,1),size(Xp,2));



function Xo = mutation_popmut_history(Xp, Xf, Pp, Pf, weights, options)
%MUTATION Performs mutation in the individuals
global individual_mut_record;
global prob1;
global prob2;

XPOP=size(Xp,2);
Xo=zeros(size(Xp,1),size(Xp,2));


function sum_vec = mutator_count(individual_vec)
sum_vec=zeros(1,3);
tmp=size(individual_vec,2);
for i=1:tmp
    if abs(individual_vec(i)-1)<0.001
        sum_vec(1,1)=sum_vec(1,1)+1;
    elseif abs(individual_vec(i)-2)<0.001
        sum_vec(1,2)=sum_vec(1,2)+1;
    elseif abs(individual_vec(i)-3)<0.001
        sum_vec(1,3)=sum_vec(1,3)+1;
    end
end


%--------------------------------------------------------------------------%
function Xo = recombination(Xp, Xm, options)
%RECOMBINATION Performs recombination in the individuals
%   The recombination combines the information of the parents and the
%   mutated individuals (also called "trial vectors") to create the
%   offspring. Assuming x represents the i-th parent, and u the i-th trial
%   vector (obtained from the mutation), the offspring xo will have the
%   following j-th coordinate:
%
%      xo_j = u_j if rand_j <= CR
%             x_j otherwise
%
%   wherein rand_j is a number drawn from a uniform distribution from 0 to
%   1, and CR is called the crossover factor. To prevent mere copies, at
%   least one coordinate is guaranteed to belong to the trial vector.
%
%   Syntax:
%      Xo = recombination(Xp, Xm, options)
%
%   Input arguments:
%      Xp: a n x mu matrix with the mu parents
%      Xm: a n x mu matrix with the mu mutated points
%      options: the struct with the internal parameters
%
%   Output argument:
%      Xo: a n x mu matrix with the recombinated points (offspring)

% Draws random numbers and checks whether they are smaller or
% greater than CR
n = size(Xp, 1); %dimension of the problem
aux = rand(n, options.mu) <= options.CR;
% Now assures at least one coordinate will be changed, that is,
% there is at least one 'true' in each column
auxs = sum(aux) == 0; %gets the columns with no trues
indc = find(auxs); %get the number of the columns
indr = randi(n, 1, sum(auxs)); %define random indexes of rows
if isempty(indr), indr = []; end
if isempty(indc), indc = []; end
ind = sub2ind([n, options.mu], indr, indc); %converts to indexes
aux(ind) = true;

% Finally, creates the offspring
Xo = Xp;
Xo(aux) = Xm(aux);
%--------------------------------------------------------------------------%
function Xo = repair(Xo)
%REPAIR Truncates the population to be in the feasible region

% This is easy, because the population must be inside the interval [0, 1]
Xo = max(Xo, 0); %corrects inferior limit
Xo = min(Xo, 1); %superior limit
%--------------------------------------------------------------------------%
function Pnew = selection(P, O, options)
%SELECTION Selects the next population
%   Each parent is compared to its offspring. If the parent dominates its
%   child, then it goes to the next population. If the offspring dominates
%   the parent, that new member is added. However, if they are incomparable
%   (there is no mutual domination), them both are sent to the next
%   population. After that, the new set of individuals must be truncated to
%   mu, wherein mu is the original number of points.
%   This is accomplished by the use of "non-dominated sorting", that is,
%   ranks the individual in fronts of non-domination, and within each
%   front, measures them by using crowding distance. With regard to these
%   two metrics, the best individuals are kept in the new population.
%
%   Syntax:
%      Pnew = selection(P, O, options)
%
%   Input arguments:
%      P: a struct with the parents (x and f)
%      O: a struct with the offspring
%      options: the struct with the algorithm's parameters
%
%   Output argument:
%      Pnew: the new population (a struct with x and f)

% ------ First part: checks dominance between parents and offspring
% Verifies whether parent dominates offspring
aux1 = all(P.f <= O.f, 1);
aux2 = any(P.f < O.f, 1);
auxp = and(aux1, aux2); %P dominates O
% Now, where offspring dominates parent
aux1 = all(P.f >= O.f, 1);
aux2 = any(P.f > O.f, 1);
auxo = and(aux1, aux2); %O dominates P
auxpo = and(~auxp, ~auxo); %P and O are incomparable
% New population (where P dominates O, O dominates P and where they are
% incomparable)
R.f = [P.f(:,auxp), O.f(:,auxo), P.f(:,auxpo), O.f(:,auxpo)];
R.x = [P.x(:,auxp), O.x(:,auxo), P.x(:,auxpo), O.x(:,auxpo)];

% ------- Second part: non-dominated sorting
Pnew.x = []; Pnew.f = []; %prepares the new population
while true
    ispar = ndset(R.f); %gets the non-dominated front
    % If the number of points in this front plus the current size of the new
    % population is smaller than mu, then include everything and keep going.
    % If it is greater, then stops and go to the truncation step
    if size(Pnew.f, 2) + sum(ispar) < options.mu
        Pnew.f = [Pnew.f, R.f(:,ispar)];
        Pnew.x = [Pnew.x, R.x(:,ispar)];
        R.f(:,ispar) = []; R.x(:,ispar) = []; %removes this front
    else
        % Gets the points of this front and goes to the truncation part
        Frem = R.f(:,ispar);
        Xrem = R.x(:,ispar);
        break %don't forget this to stop this infinite loop
    end
end

% ------- Third part: truncates using crowding distance
% If the remaining front has the exact number of points to fill the original
% size, then just include them. If it has too many, remove some according to
% the crowding distance (notice it cannot have too few!)
aux = (size(Pnew.f,2) + size(Frem,2)) - options.mu; %remaining points to fill
if aux == 0
    Pnew.x = [Pnew.x, Xrem]; Pnew.f = [Pnew.f, Frem];
elseif aux > 0
    for ii = 1:aux
        cdist = crowdingdistance(Frem);
        [~, imin] = min(cdist); %gets the point with smaller crowding distance
        Frem(:,imin) = []; %and remove it
        Xrem(:,imin) = [];
    end
    Pnew.x = [Pnew.x, Xrem];
    Pnew.f = [Pnew.f, Frem];
else %if there are too few points... well, we're doomed!
    error('Run to the hills! This is not supposed to happen!')
end

%--------------------------check------------------------------------------------%
function options = check_input(options,para)
%CHECK_INPUT Checks the parameters of the algorithm before
%   This sub-function checks the endogenous parameters of the algorithm. If
%   they are not set, the default ones are used

if ~isfield(options, 'F') %scale factor
    options.F = 0.5;
end

if ~isfield(options, 'CR') %crossover factor
    options.CR =0.3;
end

if ~isfield(options, 'kmax') %maximum number of iterations
    options.kmax = 100;
end

if ~isfield(options, 'mu') %population size
    options.mu = 100;
end

if ~isfield(options, 'display') %show or not the population during execution
    options.display = 'on';
end

