function phi = KLexpansion(m, k)
% Code for KL-expansion
% m is length of expansion (typically, m=20 works well)
% k is covariance function
% returns a length-m function handle containing the KL basis functions

[xleg,w] = legpts(m);
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
    Minv(:,i) = legendreP(i-1, xleg);
end

C = Minv\U;

syms x
P = legendreP(0:m-1, x);
P = matlabFunction(P);
ux = @(x) dot(C(:,1), P(x));
for j = 2:m
    ux = @(x) [ux(x); dot(C(:,j), P(x))];
end
L = diag(D);
phi = @(x) sqrt(L).*ux(x);
% o = 15; r = 15; q = 9.559790239932053e-01;
% fprintf('%.6e\n', k(q,q)- sum(phi(q).*phi(q)))


