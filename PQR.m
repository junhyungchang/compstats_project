function [U, V] = PQR(A,r)
% Low rank approximation via rank revealing pivoted QR
% A \approx U*V'
% A is a matrix
% r is the target rank
% r = 15 yields nice results

[U1,V1,P] = qr(A);
U = U1(:, 1:r);
V = V1(1:r, :);
V = P'\V';
% norm(A-U*V')/norm(A)