clear
k = @(x,y) exp(-(x-y).^2);
n = 1e+3; % system size (number of training data)
fprintf('n = %d\n', n)
m = 20; % length of KL-expansion
t = 100; % number of test points
rng(7)
xtr = 2*rand([n,1])-1; %training data
xtr = sort(xtr);
ytr = 2*rand([n,1]);
A = zeros(n,n); % Gram matrix
for i = 1:n
    for j = 1:n
        A(i,j) = k(xtr(i), xtr(j));
    end
end
%% Test
tic
fprintf('PQR:')
[U1,V1]=PQR(A,15);
norm(A-U1*V1)
toc
tic
fprintf('RLR:')
[U2,V2]=RLR(A,15,10);
norm(A-U2*V2)
toc