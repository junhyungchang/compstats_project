function [U,V] = RLR(A, r, p)
% Randomized Low-Rank factors A \approx U*V
% A is a matrix
% r is the target rank
% p is the oversapling parameter
% r = 10, p = 10 yields good results
% Matrix A should have dimension less than or equal to O(10^3)
% For larger matrices, use RSVD.m or REig.m
[m,n] = size(A);
G = randn(m, r+p);
Y = A*G;
[Q1,~,~] = svd(Y,'econ');
U = Q1(:,1:r+p);
B = U'*A;
[Uhat, D, V] = svd(B,'econ');
V = Uhat*D*V';
