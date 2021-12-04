function [U,S] = REig(f, xtr, r, p)
% Single-pass randomized eigenvalue decomposition
% Input matrix A should be a square matrix
% Note: A need not be square in general, but is square for this purpose
% f is covariance function (function handle).
% xtr is input training data.
% k is target rank.
% p is oversapling parameter; typically p=10.
n = length(xtr);
% fprintf('n = %d\n',n)

% Randomized Eigenvalue decomposition

G = randn(n, r+p);
% A1 = zeros(n,n);
% for i = 1:n
%     for j = 1:n
%         A1(i,j) = f(x(i),x(j));
%     end
% end
% Y1 = A1*G;
Y = zeros(n,r+p);
for i = 1:n
    A = f(xtr(i), xtr(1:n))';
    Y(i, 1:r+p) = A*G;
end

[Q1,~,~] = svd(Y,'econ');
Q = Q1(:,1:r+p);


% B = zeros(k+p,n);
% for i = 1:n
%     A = f(x(1:n), x(i));
%     B(1:k+p ,i) = Q'*A;
% end
% % norm(B1-B)
% [Uhat,D,V] = svd(B,'econ');
% U = Q*Uhat;
% [Uhat1,D1,V1] = svd(B1,0);
% U1 = Q1*Uhat1;

D = G'*Q;
E = Y'*Q;
C = D\E;
C = (C+C')/2;
[Uhat, S] = eig(C);
U = Q*Uhat;
% norm(A1-U*S*U')
