disp ('executing gpml startup script...')
mydir = fileparts (mfilename ('fullpath'));                 % where am I located
addpath (mydir)
dirs = {'cov','doc','inf','lik','mean','prior','util'};           % core folders
for d = dirs, addpath (fullfile (mydir, d{1})), end
dirs = {{'util','minfunc'},{'util','minfunc','compiled'}};     % minfunc folders
for d = dirs, addpath (fullfile (mydir, d{1}{:})), end
% addpath([mydir,'/util/sparseinv'])

rg=1;

ns = 1000;
x = rand(ns, 1)*1;
a0 = 4; a1 = 24; a2 = 0.5; a3 = 25; a4 = 2; a5=0.3; a6 = 2; a7 =10; a8 = 0.2;
bx = x + a0;
y = a5*5*bx.^2.*sin(a1*bx)+((a2*bx).^3-0.5).*sin(3*a3*bx-0.5)+a6*cos(2*a7*bx);  % 20 noisy training targets
y = a8*y;
noise =  a4*(randn(ns, 1));
y = y + noise;

xs = 0:0.01:rg;
xs = xs';
bxs = xs + a0;
ys = a5*5*bxs.^2.*sin(a1*bxs)+((a2*bxs).^3-0.5).*sin(3*a3*bxs-0.5)+a6*cos(2*a7*bxs);  % 20 noisy training targets
ys = a8*ys;

sf2 = 1 ; ell = 1 ; sn2 = 0.1 ; 
d = size(x,2);
hyp.cov = log([ones(d,1)*ell;sf2]); hyp.lik = log(sn2); hyp.mean = [];
opts.Xnorm = 'N' ; opts.Ynorm = 'N' ;
opts.Ms = 80;
opts.ell = ell ; opts.sf2 = sf2 ; opts.sn2 = sn2 ;
opts.meanfunc = []; opts.covfunc = @covSEard; opts.likfunc = @likGauss; opts.inffunc = @infGaussLik ;

meanfunc = [];                    % empty: don't use a mean function
covfunc = opts.covfunc;              % Squared Exponental covariance function
likfunc = opts.likfunc;              % Gaussian likelihood
inffunc = opts.inffunc;

hyp2 = minimize(hyp, @gp, -70, inffunc, meanfunc, covfunc, likfunc, x, y)

[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

partitionCriterion = 'kmeans'; %, 'kmeans', 'knkmeans'
opts.numOptFC = 25 ;
opts.partitionCriterion = partitionCriterion ;
[models,t_dGP_train] = aggregation_train(x,y,opts) ;

criterion = 'TERBCM';

for q=1:0.2:1
    [tmu,ts2,t_dGP_predict] = aggregation_predict(xs,models,criterion, q);
    figure; hold on; 
    f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
    fill([xs; flipdim(xs,1)], f, [5 5 5]/8)
     plot(x, y, 'b+'); 
    plot(xs, ys, 'g','LineWidth',5);
     plot(xs, tmu, 'r','LineWidth',5);
     plot(xs, tmu+2*sqrt(ts2), 'r','LineWidth',5);
     plot(xs, tmu-2*sqrt(ts2), 'r','LineWidth',5);
    qtxt = sprintf('Predictive mean and CI (q = %.1f)', q);
    legend('Full GP confidence interval', 'data', 'ground truth', qtxt)
    drawnow
end






