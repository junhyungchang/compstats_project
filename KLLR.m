function [U,V] = KLLR(k, x1, x2, m)
% Code for KL-expansion
% m is length of expansion (typically, m=20 works well)
% x1, x2 are subsets of training input (off-diagonal blocks)
% k is covariance function
% returns a n by m tall matrix X such that K approx XX^*

[xleg,w] = legpts(m);
xleg = xleg*3;
A = zeros(m,m);
for i = 1:m
    for j = 1:m
        A(i,j) =sqrt(w(i)*w(j))*k(xleg(i), xleg(j));
    end
end
[U, D] = eig(A);
for i = 1:m
    U(i,:) = U(i,:)/sqrt(w(i));
end

Minv = zeros(m,m);
for i = 1:m
    Minv(:,i) = legendreP(i-1, xleg/3);
end

% coefficients of Legendre poly approximation
C = Minv\U;
% find Legendre poly approx function
syms x
P = legendreP(0:m-1, x);
P = matlabFunction(P);
P = @(x) P(x/3);
ux = @(x) dot(C(:,1), P(x));
for j = 2:m
    ux = @(x) [ux(x); dot(C(:,j), P(x))];
end
L = diag(D);
phi = @(x) sqrt(L).*ux(x);

% form low rank matrix
n = length(x1);
U = zeros(n,m);
V = U;
for i = 1:n
    U(i,1:m) = phi(x1(i));
    V(i,1:m) = phi(x2(i));
end


