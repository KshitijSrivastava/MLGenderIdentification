clear all;
clc;
data_read=xlsread('voice.csv');

male=data_read(1:1585,:);
female=data_read(1586:3168,:);

male(:,21)=0; %male data =0
female(:,21)=1; %Female Data=1

col=21; %For both male and female

[row_m ,col_m]=size(male);
[row_f,col_f]=size(female);
%%
max_val_data_male=max(male(:,1:col-1));
min_val_data_male=min(male(:,1:col-1)); 
mean_data_male=mean(male(:,1:col-1));
std_data_male=std(male(:,1:col-1));

max_val_data_female=max(female(:,1:col-1));
min_val_data_female=min(female(:,1:col-1)); 
mean_data_female=mean(female(:,1:col-1));
std_data_female=std(female(:,1:col-1));

for i =1:col-1
   %Ndata(:,i)=data(:,i)/max(data(:,i));
   Norm_male(:,i)=(male(:,i)-mean_data_male(i))/std_data_male(i); %mean normalisation
end  

for i =1:col-1
   %Ndata(:,i)=data(:,i)/max(data(:,i));
   Norm_female(:,i)=(female(:,i)-mean_data_female(i))/std_data_female(i); %mean normalisation
end  

%%
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

%%
for i=1:length(outlier_male)    %Removal of the outlier data
   Norm_male(outlier_male(i),:)=[];
   outlier_male=outlier_male-1;
end

for i=1:length(outlier_female) %Removal of the outlier data
   Norm_female(outlier_female(i),:)=[];
   outlier_female=outlier_female-1;
end

%% Plotting

% x1m=male(:,1);
% x1f=female(:,1);
% 
% pdm = fitdist(x1m,'Normal');
% pdf=fitdist(xlf,'Normal');
% 
% plot(pdm)
 figure;
 ksdensity(male(:,1));
 hold on;
 ksdensity(female(:,1));
 
  figure;
 ksdensity(male(:,2));
 hold on;
 ksdensity(female(:,2));
 
  figure;
 ksdensity(male(:,3));
 hold on;
 ksdensity(female(:,3));
 
  figure;
 ksdensity(male(:,4));
 hold on;
 ksdensity(female(:,4));
 
 figure;
 ksdensity(male(:,5));
 hold on;
 ksdensity(female(:,5));
 
 %%
 x = -3:.1:3;

 
 y2 = pdf('Normal',male(:,5),0,1);
 plot(y2);
%%
  figure;
 ksdensity(male(:,1));
 hold on;
 ksdensity(female(:,1));
 
  figure;
 ksdensity(male(:,1));
 hold on;
 ksdensity(female(:,1));
 
 
 %%
pd = makedist('Normal');
x_m = linspace(-4,4,1585);
pdf_normal = pdf(pd,male(:,3));
plot(x_m,pdf_normal,'LineWidth',2)

%%
figure;
plot(sort(male(:,1)));
