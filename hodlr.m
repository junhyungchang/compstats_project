
clear
k = @(x,y) exp(-(x-y).^2); % Gaussian covariance function
n = 2^20; % system size (training data)
r = 25;
p = 0;
siz = 2^6; % size of smallest block diagonal 
fprintf('n = %d\n', n)
%% training data
t = 100; % number of test points
rng(7)
xtr = 6*rand([n,1])-3; 
% xtr = 2*rand([n,1]-3);
xtr = sort(xtr);
y = 2*rand([n,1]);
xtest = linspace(-3,3,t);
% xtest = linspace(-1,1,t);

% rank(B(n/2+1:n, 1:n/2))

%% Begin HODLR setup
tic
kappa = floor(log2(n/siz));

%% compress off-diagonal blocks
UV = zeros(n,r*kappa); % store the low rank factors
AUV = zeros(n,r*kappa); % for applying A inverse
for j = 1:6
    h = n/(2^j); % size of off diagonal blocks
    for i = 1: 2^(j-1)
        x1 = xtr((2*i-2)*h+1:(2*i-1)*h);
        x2 = xtr((2*i-1)*h+1:2*i*h);
%         [U1,V1] = RLR(k, x1, x2, r, p); % randomized scheme
        [U1,V1] = KLLR(k, x1, x2, r);   % KL-expansion approx
        UV((2*i-2)*h+1:(2*i-1)*h, (j-1)*r+1:j*r) = U1;
        UV((2*i-1)*h+1:2*i*h, (j-1)*r+1:j*r) = V1;
    end
end
for j = 7:kappa
    h = n/(2^j); % size of off diagonal blocks
    for i = 1: 2^(j-1)
        x1 = xtr((2*i-2)*h+1:(2*i-1)*h);
        x2 = xtr((2*i-1)*h+1:2*i*h);
        [U1,V1] = RLR(k, x1, x2, r, p); % randomized scheme
%         [U1,V1] = KLLR(k, x1, x2, r);   % KL-expansion approx
        UV((2*i-2)*h+1:(2*i-1)*h, (j-1)*r+1:j*r) = U1;
        UV((2*i-1)*h+1:2*i*h, (j-1)*r+1:j*r) = V1;
    end
end



toc

%% recursive formation of Hodler factors

% K_kappa (first block diagonal matrix)
% Invert and apply via mldivide
% apply inverse to y, and all U or V in the right places
tic
for i = 1:(n/siz)
    x = xtr((i-1)*siz+1:i*siz);
    A = zeros(siz,siz);
    for j = 1:siz
        for q = 1:siz
            A(j,q) = k(xtr((i-1)*siz+j),xtr((i-1)*siz+q));
        end
    end
%     norm(B((i-1)*size+1:i*size, (i-1)*size+1:i*size)-A)
    A = A + eye(siz);
    y((i-1)*siz+1:i*siz) = A\y((i-1)*siz+1:i*siz);
    
    % Apply to all U and V in the right locations
    for j = kappa:-1:1
        AUV((i-1)*siz+1:i*siz, (j-1)*r+1:j*r) = ...
            A\UV((i-1)*siz+1:i*siz, (j-1)*r+1:j*r);
    end
%     spy(AUV)
end

% begin recursion
for j = kappa:-1:2
    h = n/(2^j);
    for i = 1:2^(j-1)
        % update y
        yy = y((i-1)*2*h+1:i*2*h);
        AUu = AUV((2*i-2)*h+1:(2*i-1)*h, (j-1)*r+1:j*r);
        Uu = UV((2*i-2)*h+1:(2*i-1)*h, (j-1)*r+1:j*r);
        AVv = AUV((2*i-1)*h+1:(2*i)*h, (j-1)*r+1:j*r);
        Vv = UV((2*i-1)*h+1:(2*i)*h, (j-1)*r+1:j*r);
        yy = [Uu'*yy(1:h); Vv'*yy(h+1:2*h)];
        IVU = eye(2*r);
        IVU(1:r, r+1:2*r) = Uu'*AUu;
        IVU(r+1:2*r, 1:r) = Vv'*AVv;
        yy = IVU\yy;
        yy = [AUu*yy(r+1:2*r);AVv*yy(1:r)];
        y((i-1)*2*h+1:i*2*h) = y((i-1)*2*h+1:i*2*h)-yy;
        
        % apply A inverse to remaining U and V
        for q = j-1:-1:1
            Uur = AUV((i-1)*2*h+1:2*i*h, (q-1)*r+1:q*r);
            Uur = [Uu'*Uur(1:h, :) ; Vv'*Uur(h+1:2*h, :)];
            Uur = IVU\Uur;
            Uur = [AUu*Uur(r+1:2*r,:); AVv*Uur(1:r,:)];
            AUV((i-1)*2*h+1:2*i*h, (q-1)*r+1:q*r) = ...
                AUV((i-1)*2*h+1:2*i*h, (q-1)*r+1:q*r)- Uur;
        end
    end
end
j = 1;
h = n/(2^j);
% %     update y one last time

yy = y(1:2*h);
AUu = AUV(1:h, (j-1)*r+1:j*r);
Uu = UV(1:h, (j-1)*r+1:j*r);
AVv = AUV(h+1:2*h, (j-1)*r+1:j*r);
Vv = UV(h+1:2*h, (j-1)*r+1:j*r);
yy = [Uu'*yy(1:h); Vv'*yy(h+1:2*h)];
IVU = eye(2*r);
IVU(1:r, r+1:2*r) = Uu'*AUu;
IVU(r+1:2*r, 1:r) = Vv'*AVv;
yy = IVU\yy;
yy = [AUu*yy(r+1:2*r);AVv*yy(1:r)];
y(1:2*h) = y(1:2*h)-yy;
toc
% cross covariance matrix
Kstar = zeros(t, n);
for i = 1:t
    for j = 1:n
        Kstar(i,j) = k(xtest(i), xtr(j));
    end
end
hmean = Kstar*y;
%% Plot result
hold on
plot(xtest, hmean, 'co', 'MarkerSize', 5)


