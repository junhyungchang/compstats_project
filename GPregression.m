clear
% Compare
% 1. KL-expansion approach
% 2. Global low rank approx via randomized eigen decomp.
% 3. HODLR factorization approach
index1 = [2^8, 2^10, 2^12, 2^14, 2^16, 2^18];
index2 = [2^18, 2^20]; % suppress RSVD for index2


k = @(x,y) exp(-(x-y).^2); % Gaussian covariance function
t = 100; % number of test points
xtest = linspace(-3,3,t); % test points
Rtime = zeros(length(index1),1);
KLtime1 = zeros(length(index1),1);
Htime1 = zeros(length(index1),1);
KLtime2 = zeros(length(index2),1);
Htime2 = zeros(length(index2),1);

%% Lower n (index1)
count = 0;
for n = index1 % system size (training data)  
    fprintf('n = %d\n', n)
    count = count + 1;
    %% training/test data
    rng(7)
    xtr = 6*rand([n,1])-3;
    xtr = sort(xtr);
    ytr = 2*rand([n,1]);

    % cross covariance matrix
    Kstar = zeros(t, n);
    for i = 1:t
        for j = 1:n
            Kstar(i,j) = k(xtest(i), xtr(j));
        end
    end


    %% Regression via KL-expansion
%     tic
%     m = 25;
%     phi = KLexpansion(m, k);
%     t1 = toc;
%     tic
%     X = zeros(n,m);
%     for j = 1:n
%         X(j,:) = phi(xtr(j))';
%     end
%     [U,D,~] = svd(X,'econ');
%     Dinv = zeros(m,m);
%     for i = 1:m
%         Dinv(i,i) = 1/(1+D(i,i)^2);
%     end
%     KLmean = Kstar*(U*(Dinv*(U'*ytr)));
%     KLmean = real(KLmean);
%     t2 = toc;
%     KLtime1(count) = t1+t2;

    %% Regression via Randomized Eigen-decomposition
    % target rank
    r = 15;
    % oversampling parameter
    p = 10; 
    [Ur,Sr, t3, t4]=REig(k, xtr, r, p);
    tic
    Drinv = zeros(r+p,r+p);
    for i = 1:r+p
        Drinv(i,i) = 1/(Sr(i,i)+1);
    end
    Rmean = Kstar*(Ur*(Drinv*(Ur'*ytr)));
    t7 = toc;
    Rtime(count) = t3 + t4 + t7;

    %% Regression via HODLR factorization
%     siz = 2^6;
%     ell = 25;
%     [y, t5, t6] = hodlr(k, xtr, ytr, siz, ell);
%     Hmean = Kstar*y;   
%     Htime1(count) = t5+t6;
    %% Print times
    t1 =0; t2 = 0; t5=0; t6=0;
    fprintf('%.4f %.4f %.4f %.4f %.4f %.4f %.4f\n', t1, t2, t3, t4, t7, t5, t6)
end

%% Plot comparison of posterior mean
figure(1)
plot(xtest, KLmean, 'k', 'LineWidth', 1.5);
hold on
plot(xtest, Rmean, 'r.', 'MarkerSize', 9)
hold on
plot(xtest, Hmean, 'bo', 'MarkerSize', 7)
legend('KL-expansion', 'Rand. eig.', 'HODLR')
title(sprintf('System size n = %d\n Plot of posterior mean on [-3,3]', n))
hold off


%% Higher n (index2)
count = 0;
for n = index2 % system size (training data)  
    fprintf('n = %d\n', n)
    count = count + 1;
    %% training/test data
    rng(7)
    xtr = 6*rand([n,1])-3;
    xtr = sort(xtr);
    ytr = 2*rand([n,1]);

    % cross covariance matrix
    Kstar = zeros(t, n);
    for i = 1:t
        for j = 1:n
            Kstar(i,j) = k(xtest(i), xtr(j));
        end
    end


    %% Regression via KL-expansion
    tic
    m = 25;
    phi = KLexpansion(m, k);
    t1 = toc;
    tic
    X = zeros(n,m);
    for j = 1:n
        X(j,:) = phi(xtr(j))';
    end
    [U,D,~] = svd(X,'econ');
    Dinv = zeros(m,m);
    for i = 1:m
        Dinv(i,i) = 1/(1+D(i,i)^2);
    end
    KLmean = Kstar*(U*(Dinv*(U'*ytr)));
    KLmean = real(KLmean);
    t2 = toc;
    KLtime2(count) = t1+t2;

    %% Regression via HODLR factorization
    siz = 2^6;
    ell = 25;
    [y, t5, t6] = hodlr(k, xtr, ytr, siz, ell);
    Hmean = Kstar*y;   
        Htime2(count) = t5+t6;

    %% Print times
    fprintf('%.4f %.4f %.4f %.4f %.4f %.4f\n', t1, t2, t5, t6)
end


%% Plot comparison of posterior mean
figure(2)
plot(xtest, KLmean, 'k', 'LineWidth', 1.5);
hold on
plot(xtest, Hmean, 'bo', 'MarkerSize', 7)
legend('KL-expansion', 'HODLR')
title(sprintf('System size n = %d\n Plot of posterior mean on [-3,3]', n))
hold off

%% Plot times

figure(3)
KLtime = [KLtime1', KLtime2'];
Htime = [Htime1', Htime2'];
index22 = [index1, index2];
loglog(index1, Rtime, '-sr', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor',[0 0 0])
hold on
loglog(index22, KLtime, '-sb', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor',[0 0 0])
hold on
loglog(index22, Htime, '-sk', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor',[0 0 0])
title('Run-time comparison')
xlim([2^6, 2^21])
xticks([10^2 10^4 10^6])
xticklabels({'10^2', '10^4', '10^6'}) 
xlabel('System size')
ylabel('elapsed time (seconds)')


