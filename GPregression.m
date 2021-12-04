clear

k = @(x,y) exp(-(x-y).^2);
n = 1e+5; % system size (training data)
fprintf('n = %d\n', n)
m = 20; % length of KL-expansion
t = 100; % number of test points
rng(7)
xtr = 2*rand([n,1])-1;
xtr = sort(xtr);
ytr = 2*rand([n,1]);
xtest = linspace(-1,1,t);

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

%% For plot comparison with mldivide
%% Warning: When activating the following block of code,
%% set n less than or equal to 1e+3 to prevent memory overflow

% G = zeros(n,n);
% for i = 1:n
%     for j = 1:n
%         G(i,j) = k(xtr(i),xtr(j));
%     end
% end
% Ktr = (eye(n)+G);
% mtest = Kstar*(Ktr\ytr);
% plot(xtest, mean, 'LineWidth', 1.5);
% hold on
% plot(xtest, mtest, 'r.', 'MarkerSize', 7)
% legend('KL-expansion', 'mldivide')
% title(sprintf('System size n = %d\n', n))
% 
% % norm(G-X*X')/norm(G)


%% Regression via Randomized Eigen-decomposition
tic
% target rank
r = 10;
% oversampling parameter
p = 10; 
[Ur,Sr]=REig(k, xtr, r, p);
Drinv = zeros(r,r);
for i = 1:r
    Drinv(i,i) = 1/(Sr(i,i)+1);
end
Rmean = Kstar*(Ur*(Drinv*(Ur'*ytr)));
toc

plot(xtest, mean, 'LineWidth', 1.5);
hold on
plot(xtest, Rmean, 'r.', 'MarkerSize', 7)
legend('KL-expansion', 'Rand. eig.')
title(sprintf('System size n = %d\n', n))

