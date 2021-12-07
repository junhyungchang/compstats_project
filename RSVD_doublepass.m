clear
tic
n=1e+2;
x = rand([n,1])-1;
f = @(x,y) exp(-(x-y).^2);
% Randomized SVD

% k = rank(A);
k = 10;
p = 10;
G = randn(n, k+p);
A1 = zeros(n,n);
for i = 1:n
    for j = 1:n
        A1(i,j) = f(x(i),x(j));
    end
end
% Y1 = A1*G;
Y = zeros(n,k+p);
for i = 1:n
    A = f(x(i), x(1:n))';
    Y(i, 1:k+p) = A*G;
end
[Q,~] = qr(Y,0);

% [Q1,~,~] = svd(Y,'econ');
% Q = Q1(:,1:k);


B = zeros(k+p,n);
for i = 1:n
    A = f(x(1:n), x(i));
    B(1:k+p ,i) = Q'*A;
end
[Uhat,D,V] = svd(B,'econ');
U = Q*Uhat;
% [Uhat1,D1,V1] = svd(B1,0);
% U1 = Q1*Uhat1;

% D = G'*Q;
% E = Y'*Q;
% C = D\E;
% C = (C+C')/2;
% [Uhat, S] = eig(C);
% U = Q*Uhat;
norm(A1-U*D*V')
toc