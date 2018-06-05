subplot(5,4,1);
ksdensity(female(:,1));
xlabel('Mean Freq');ylabel('pdf');
hold on
ksdensity(male(:,1));
hold off

subplot(5,4,2);
ksdensity(female(:,2));
xlabel('Standard Deviation');ylabel('pdf');
hold on
ksdensity(male(:,2));
hold off

subplot(5,4,3);
ksdensity(female(:,3));
xlabel('Median');ylabel('pdf');
hold on
ksdensity(male(:,3));
hold off

subplot(5,4,4);
ksdensity(female(:,4));
xlabel('Quantile-25');ylabel('pdf');
hold on
ksdensity(male(:,4));
hold off

subplot(5,4,5);
ksdensity(female(:,5));
xlabel('Quantile-75');ylabel('pdf');
hold on
ksdensity(male(:,5));
hold off

subplot(5,4,6);
ksdensity(female(:,6));
xlabel('IQR');ylabel('pdf');
hold on
ksdensity(male(:,6));
hold off

subplot(5,4,7);
ksdensity(female(:,7));
xlabel('skew');ylabel('pdf');
hold on
ksdensity(male(:,7));
hold off

subplot(5,4,8);
ksdensity(female(:,8));
xlabel('kurt');ylabel('pdf');
hold on
ksdensity(male(:,8));
hold off

subplot(5,4,9);
ksdensity(female(:,9));
xlabel('Spectral Entropy');ylabel('pdf');
hold on
ksdensity(male(:,9));
hold off

subplot(5,4,10);
ksdensity(female(:,10));
xlabel('Spectral Flatness');ylabel('pdf');
hold on
ksdensity(male(:,10));
hold off

subplot(5,4,11);
ksdensity(female(:,11));
xlabel('Mode');ylabel('pdf');
hold on
ksdensity(male(:,11));
hold off

subplot(5,4,12);
ksdensity(female(:,12));
xlabel('Centroid');ylabel('pdf');
hold on
ksdensity(male(:,12));
hold off

subplot(5,4,13);
ksdensity(female(:,13));
xlabel('Mean Fun');ylabel('pdf');
hold on
ksdensity(male(:,13));
hold off

subplot(5,4,14);
ksdensity(female(:,14));
xlabel('Minimum frequency');ylabel('pdf');
hold on
ksdensity(male(:,15));
hold off

subplot(5,4,15);
ksdensity(female(:,15));
xlabel('Maximum frequency');ylabel('pdf');
hold on
ksdensity(male(:,15));
hold off

subplot(5,4,16);
ksdensity(female(:,16));
xlabel('Mean dominant freq');ylabel('pdf');
hold on
ksdensity(male(:,16));
hold off

subplot(5,4,17);
ksdensity(female(:,17));
xlabel('Minimum dominant freq');ylabel('pdf');
hold on
ksdensity(male(:,17));
hold off

subplot(5,4,18);
ksdensity(female(:,18));
xlabel('Maximum dominant freq');ylabel('pdf');
hold on
ksdensity(male(:,18));
hold off

subplot(5,4,19);
ksdensity(female(:,19));
xlabel('Range of dominant freq');ylabel('pdf');
hold on
ksdensity(male(:,19));
hold off

subplot(5,4,20);
ksdensity(female(:,20));
xlabel('Modulation Index');ylabel('pdf');
hold on
ksdensity(male(:,20));
hold off

legend('female','male')
