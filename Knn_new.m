clear all;
clc;
data_read=xlsread('voice.csv');

male=data_read(1:1585,:);%Taking the male data
female=data_read(1586:3168,:); %Taking the female data

male(:,21)=0; %male data =0
female(:,21)=1; %Female Data=1

col=21; %For both male and female

[row_m ,col_m]=size(male); %Taking the size of both male matrix
[row_f,col_f]=size(female); %Taking the size of the female matrix
%% Normalization of the Data
max_val_data_male=max(male(:,1:col-1));
min_val_data_male=min(male(:,1:col-1)); 
mean_data_male=mean(male(:,1:col-1));
std_data_male=std(male(:,1:col-1));

max_val_data_female=max(female(:,1:col-1));
min_val_data_female=min(female(:,1:col-1)); 
mean_data_female=mean(female(:,1:col-1));
std_data_female=std(female(:,1:col-1));

for i =1:col-1
   %Norm_male(:,i)=male(:,i)/max_val_data_male(i));%Max normalization
   Norm_male(:,i)=(male(:,i)-mean_data_male(i))/std_data_male(i); %z score normalization
   %Norm_male(:,i)=(male(:,i)-mean_data_male(i))/(max_val_data_male(i)-min_val_data_male(i));
   %%mean normalization
   %Norm_male(:,i)=(male(:,i)-min_val_data_male(i))/(max_val_data_male(i)-min_val_data_male(i));%Rescaling
end  

for i =1:col-1
   %Norm_female(:,i)=female(:,i)/max_val_data_female(i));%Max normalization
   Norm_female(:,i)=(female(:,i)-mean_data_female(i))/std_data_female(i); %z score normalization
   %Norm_female(:,i)=(female(:,i)-mean_data_female(i))/(max_val_data_female(i)-min_val_data_female(i));
   %%mean normalization
   %Norm_male(:,i)=(male(:,i)-min_val_data_male(i))/(max_val_data_male(i)-min_val_data_male(i));%Rescaling
end  

%% Outliers identification by more than 3 times the sigma
mu_male=mean(Norm_male);
sigma_male=std(Norm_male);
outlier_male=[];
for i=1:col-1 
    for j=1:row_m   
        if abs(Norm_male(j,i)-mu_male(i))>=3*sigma_male(i)
            outlier_male=[outlier_male,j];  %Add code here (at this position) to replace oulier with mean/median etc
        end    
    end    
end
outlier_male=unique(outlier_male);
Norm_female=horzcat(Norm_female,female(:,21));

mu_female=mean(Norm_female);
sigma_female=std(Norm_female);
outlier_female=[];
for i=1:col-1 
    for j=1:row_f   
        if abs(Norm_female(j,i)-mu_female(i))>=3*sigma_female(i)
            outlier_female=[outlier_female,j];  %Add code here (at this position) to replace oulier with mean/median etc
        end    
    end    
end
outlier_female=unique(outlier_female);
Norm_male=horzcat(Norm_male,male(:,21));

%% Outliers identification by iqr
mu_male=mean(Norm_male);
sigma_male=std(Norm_male);
outlier_male=[];

min_male=Norm_male(:,4)-(1.5*Norm_male(:,6));
max_male=Norm_male(:,5)+(1.5*Norm_male(:,6));

for i=1:col-1 
    for j=1:row_m   
        if Norm_male(j,i)>max_male || Norm_male(j,i)<min_male
            outlier_male=[outlier_male,j];  %Add code here (at this position) to replace oulier with mean/median etc
        end    
    end    
end
outlier_male=unique(outlier_male);

Norm_female=horzcat(Norm_female,female(:,21));

mu_female=mean(Norm_female);
sigma_female=std(Norm_female);
outlier_female=[];

min_female=Norm_female(:,4)-(1.5*Norm_female(:,6));
max_female=Norm_female(:,5)+(1.5*Norm_female(:,6));

for i=1:col-1 
    for j=1:row_f   
        if Norm_female(j,i)>max_female || Norm_female(j,i)<min_female
            outlier_female=[outlier_female,j];  %Add code here (at this position) to replace oulier with mean/median etc
        end    
    end    
end
outlier_female=unique(outlier_female);
Norm_male=horzcat(Norm_male,male(:,21));

%% Outliers removal by the use of Mahalanobis distance

% Y=Norm_male(1,:);
% X=Norm_male;
% d1 = mahal(Y,X);

mahabalonis_dist_male=zeros(20);
mahabalonis_dist_female=zeros(20);


for i=1:col
    mahabalonis_dist_male(i) = mahal(Norm_female(i,:),Norm_male);
	mahabalonis_dist_female(i) = mahal(Norm_female(i,:),Norm_female);
end

outlier_male=[];
for i=1:col
	if mahabalonis_dist_male(i)>100 %Some threshold
		outlier_male=[outlier_male,i];
	end
end
Norm_male=horzcat(Norm_male,male(:,21));

outlier_female=[];
for i=1:col
	if mahabalonis_dist_female(i)>100 %Some Threshold
		outlier_female=[outlier_female,i];
	end
end
Norm_female=horzcat(Norm_female,female(:,21));

%% Finding the bhattacharya distance of the male and female data 

bhattacharya_male=zeros(20,20);
bhattacharya_female=zeros(20,20);

for i=1:col-1
    for j=1:col-1
        bhattacharya_male(i,j)=bhattacharyya(female(:,i),female(:,j));
        bhattacharya_female(i,j)=bhattacharyya(female(:,i),female(:,j));
    end
end

%% Outliers removal from the data
for i=1:length(outlier_male)    %Removal of the outlier data
   Norm_male(outlier_male(i),:)=[];
   outlier_male=outlier_male-1;
end

for i=1:length(outlier_female) %Removal of the outlier data
   Norm_female(outlier_female(i),:)=[];
   outlier_female=outlier_female-1;
end

Norm_data=vertcat(Norm_male, Norm_female);

%% Plotting

% x1m=male(:,1);
% x1f=female(:,1);
% 
% pdm = fitdist(x1m,'Normal');
% pdf=fitdist(xlf,'Normal');
% 
% plot(pdm)

subplot(5,4,1);
area(ksdensity(female(:,1)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Mean Frequency');ylabel('pdf');
hold on
area(ksdensity(male(:,1)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,2);
area(ksdensity(female(:,2)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Standard Deviation');ylabel('pdf');
hold on
area(ksdensity(male(:,2)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,3);
area(ksdensity(female(:,3)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Median');ylabel('pdf');
hold on
area(ksdensity(male(:,3)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,4);
area(ksdensity(female(:,4)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Quantile-25');ylabel('pdf');
hold on
area(ksdensity(male(:,4)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,5);
area(ksdensity(female(:,5)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Quantile-75');ylabel('pdf');
hold on
area(ksdensity(male(:,5)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,6);
area(ksdensity(female(:,6)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('IQR');ylabel('pdf');
hold on
area(ksdensity(male(:,6)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,7);
area(ksdensity(female(:,7)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('skew');ylabel('pdf');
hold on
area(ksdensity(male(:,7)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,8);
area(ksdensity(female(:,8)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('kurt');ylabel('pdf');
hold on
area(ksdensity(male(:,8)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,9);
area(ksdensity(female(:,9)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Spectral Entropy');ylabel('pdf');
hold on
area(ksdensity(male(:,9)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,10);
area(ksdensity(female(:,10)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Spectral Flatness');ylabel('pdf');
hold on
area(ksdensity(male(:,10)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,11);
area(ksdensity(female(:,11)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Mode');ylabel('pdf');
hold on
area(ksdensity(male(:,11)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,12);
area(ksdensity(female(:,12)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Centroid');ylabel('pdf');
hold on
area(ksdensity(male(:,12)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,13);
area(ksdensity(female(:,13)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Mean Frequency');ylabel('pdf');
hold on
area(ksdensity(male(:,13)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,14);
area(ksdensity(female(:,14)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Minimum frequency');ylabel('pdf');
hold on
area(ksdensity(male(:,15)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,15);
area(ksdensity(female(:,15)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Maximum frequency');ylabel('pdf');
hold on
area(ksdensity(male(:,15)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,16);
area(ksdensity(female(:,16)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Mean dominant freq');ylabel('pdf');
hold on
area(ksdensity(male(:,16)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,17);
area(ksdensity(female(:,17)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Minimum dominant freq');ylabel('pdf');
hold on
area(ksdensity(male(:,17)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,18);
area(ksdensity(female(:,18)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Maximum dominant freq');ylabel('pdf');
hold on
area(ksdensity(male(:,18)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,19);
area(ksdensity(female(:,19)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Range of dominant freq');ylabel('pdf');
hold on
area(ksdensity(male(:,19)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

subplot(5,4,20);
area(ksdensity(female(:,20)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Modulation Index');ylabel('pdf');
hold on
area(ksdensity(male(:,20)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

%%
area(ksdensity(Norm_female(:,2)),'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Standard Deviation');ylabel('pdf');
hold on
area(ksdensity(Norm_male(:,2)),'FaceColor',[0 0 1],'FaceAlpha' ,0.25);
hold off

%%
[y,x]=ksdensity(female(:,13));
area(y,'FaceColor',[1 0 0],'FaceAlpha',0.25);
xlabel('Mean Frequency');ylabel('pdf');
hold on
(ksdensity(male(:,13)));
hold off
%plot(xi,f);
