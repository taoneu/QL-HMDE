global rp;
if isempty(gcp('nocreate'))
    parpool;
end


for rp=1:10
    clearvars -EXCEPT rp;
    f = @(x) delq(x, 3);
    xrange = dtlz_range('delq', 3);
    [fopt, xopt] = demo_opt_decop(f, xrange);
end

%