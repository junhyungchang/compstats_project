clear
% Compare
% 1. KL-expansion approach
% 2. Global low rank approx via randomized eigen decomp.

k = @(x,y) exp(-(x-y).^2); % Gaussian covariance function
n = 2^20; % system size (training data)
fprintf('n = %d\n', n)
m = 25; % length of KL-expansion
t = 100; % number of test points
%% training data
rng(7)
xtr = 6*rand([n,1])-3;
% xtr = 2*rand([n,1])-1;
xtr = sort(xtr);
ytr = 2*rand([n,1]);
xtest = linspace(-3,3,t);
% xtest = linspace(-1,1,t);

%% Regression via KL-expansion
tic
phi = KLexpansion(m, k);
X = zeros(n,m);
for j = 1:n
    X(j,:) = phi(xtr(j))';
end
[U,D,~] = svd(X,'econ');
Dinv = zeros(m,m);
for i = 1:m
    Dinv(i,i) = 1/(1+D(i,i)^2);
end
% cross covariance matrix
Kstar = zeros(t, n);
for i = 1:t
    for j = 1:n
        Kstar(i,j) = k(xtest(i), xtr(j));
    end
end
mean = Kstar*(U*(Dinv*(U'*ytr)));
mean = real(mean);
toc

%% Regression via Randomized Eigen-decomposition
% tic
% % target rank
% r = 20;
% % oversampling parameter
% p = 0; 
% [Ur,Sr]=REig(k, xtr, r, p);
% Drinv = zeros(r+p,r+p);
% for i = 1:r+p
%     Drinv(i,i) = 1/(Sr(i,i)+1);
% end
% Rmean = Kstar*(Ur*(Drinv*(Ur'*ytr)));
% toc

hold on
plot(xtest, mean, 'b', 'LineWidth', 1.5);
% hold on
% plot(xtest, Rmean, 'r.', 'MarkerSize', 7)
% legend('KL-expansion', 'Rand. eig.')
title(sprintf('System size n = %d\n', n))

