index1 = [2^8, 2^10, 2^12, 2^14, 2^16, 2^18];
index2 = [2^20]; % suppress RSVD for index2
Rtime = [0.0161 0.0749 0.5736 7.2253 106.8300 1974.9187];
KLtime =[1.0790 1.2264 2.2436 5.8725 20.6885 80.3955 333.5380];
Htime = [0.0110 0.0986 0.7851 7.9894 133.3291 611.2985 2995.0448];

index22 = [index1, index2];
loglog(index1, Rtime, '-sr', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor',[1 0 0])
hold on
loglog(index22, KLtime, '-sb', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor',[0 0 1])
hold on
loglog(index22, Htime, '-sk', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor',[0 0 0])
title('Run-time comparison')
xlim([2^6, 2^21])
xticks([10^2 10^4 10^6])
xticklabels({'10^2', '10^4', '10^6'}) 
xlabel('System size')
ylabel('elapsed time (seconds)')
legend('RSVD', 'KL-expand', 'HODLR', 'Location', 'nw')
