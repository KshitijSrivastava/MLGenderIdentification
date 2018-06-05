clear all;
clc;

data1=xlsread('voice.csv'); %Reading the data
data1(:,21)=0;
data1(1:1585,21)=1;

[row1,col1]=size(data1); %finding the size of the data

idx=randperm(row1);
data2(row1,col1)=0;
for i=1:row1
   data2(i,:)=data1(idx(i),:); 
end    
data=data2; %Randomly shuffling the data

[row,col]=size(data); %Find the various parameters using the data
max_val_data=max(data(:,1:col-1));
min_val_data=min(data(:,1:col-1)); 
mean_data=mean(data(:,1:col-1));
std_data=std(data(:,1:col-1));

Correlation_matrix=corr(data(:,1:col-1)); %Finding the correlation matrix of the data
for i =1:col-1 %Making the data normalized
   %Ndata(:,i)=data(:,i)/max(data(:,i)); %other normalization factor
   Ndata(:,i)=(data(:,i)-mean_data(i))/std_data(i); %z score  normalisation
end  
%for i =1:col-1
%  %Ndata(:,i)=data(:,i)/max(data(:,i));
%end
Ndata(:,col)=data(:,col); %Appending the output to the last column

%% Finding the Mahabalobis Distance
Md=[]; %
for i=1:length(Ndata) %For finding the mahabalobis distance
   Md=[Md,mahal(Ndata(i,1:col-1),Ndata(:,1:col-1))]; 
end 

%% Finding the outliers using the grub's test

outlier=isoutlier(Md,'grubbs');
num_outlier_grub=0;
outlier_grub=[];

for i=1:length(outlier)
    if outlier(i)==1
        num_outlier_grub=num_outlier_grub+1;
        outlier_grub=[outlier_grub,i]; %Storing the index found with the grubs method
    end
end

%%  For removing outlier by Grubbs

for i=1:length(outlier_grub)
    Ndata(outlier_grub(i),:)=[];%Removing the data if its an outlier
    outlier_grub=outlier_grub-1;  
end 

%% For finding outlier by Mahabalobis

out=[];
for i=1:length(Md) %Finding the outliers using the Mahabalobis distance
   if Md(i)> 49 %(Median)   15+49 =std(Md) +median(Md)
      out=[out,i];
   end    
end

%% For removing outlier by Mahabalobis

for i=1:length(out)
    Ndata(out(i),:)=[];%Removing the data if more than the threshold in the mahabalobis distance
    out=out-1;
end 

%% Calculating the Chi square value for outlier removal

mean_Ndata=mean(Ndata);
chi=zeros(1,3168);
for i=1:length(Ndata)
    for j=1:col-1
        chi(i)=chi(i)+(Ndata(i,j)-mean_Ndata(j))^2/mean_Ndata(j); %Finding the chi square value for Outlier removal
    end
end

%% Outlier removal by chi
chi_outlier=[];

for i=1:length(chi)
    if chi(i)> prctile(chi,97.5) %If the chi square value is more than the 97.5 percentile store its index
        chi_outlier=[chi_outlier,i];
    end
end

%We have used a number of methods for finding outlier in mahanobalis
%distance such as grubb's test, qq plot, 3*std method, and chi (quantile
%97.5) as threshold values.

%Methods used for outlier removal: Mahabalobis, Chi square outlier
%detection method,k means clustering for multivariate data.

%% Removing by chi square

for i=1:length(chi_outlier)
    Ndata(chi_outlier(i),:)=[];%Removing the data if more than the 97.5 quartile in chi square test
    chi_outlier=chi_outlier-1;
end

%%
qqplot(Md); %qq plot of Mahabalonis distance

%% By using IQR method

outlier=[];
min=Ndata(:,4)-(1.5*Ndata(:,6));
max=Ndata(:,5)+(1.5*Ndata(:,6));

for i=1:col-1 
    for j=1:row   
        if Ndata(j,i)>max || Ndata(j,i)<min
            outlier=[outlier,j];  %Add code here (at this position) to replace oulier with mean/median etc
        end    
    end    
end
outlier=unique(outlier);

%%
