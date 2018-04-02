clear all;
clc;
data_read=xlsread('voice.csv');

male=data_read(1:1585,:);%Taking the male data
female=data_read(1586:3168,:); %Taking the female data

male(:,21)=0; %male data =0
female(:,21)=1; %Female Data=1

col=21; %For both male and female
data_mf=vertcat(male,female);%both male and femle data combined
[row ,col]=size(data_mf); %Taking the size of both male matrix

%% Normalization of the Data

max_val_data=max(data_mf(:,1:col-1));
min_val_data=min(data_mf(:,1:col-1)); 
mean_data=mean(data_mf(:,1:col-1));
std_data=std(data_mf(:,1:col-1));

for i =1:col-1
   %Norm_data(:,i)=data_mf(:,i)/(max_val_data(i));%Max normalization
   Norm_data(:,i)=(data_mf(:,i)-mean_data(i))/std_data(i); %z score normalization
   %Norm_data(:,i)=(data_mf(:,i)-mean_data(i))/(max_val_data(i)-min_val_data(i));
   %%mean normalization
   %Norm_data(:,i)=(data_mf(:,i)-min_val_data(i))/(max_val_data(i)-min_val_data(i));%Rescaling
end

mu=mean(Norm_data);
sigma=std(Norm_data);

%% Outliers identification by more than 3 times the sigma
outlier=[];
for i=1:col-1 
    for j=1:row_m   
        if abs(Norm_data(j,i)-mu(i))>=3*sigma(i)
            outlier=[outlier,j];  %Add code here (at this position) to replace oulier with mean/median etc
        end    
    end    
end
outlier=unique(outlier);

%% Outliers identification by iqr
outlier=[];

min=Norm_data(:,4)-(1.5*Norm_data(:,6));
max=Norm_data(:,5)+(1.5*Norm_data(:,6));

for i=1:col-1 
    for j=1:row_m   
        if Norm_data(j,i)>max || Norm_data(j,i)<min
            outlier=[outlier,j];  %Add code here (at this position) to replace oulier with mean/median etc
        end    
    end    
end
outlier=unique(outlier);

%% Adding the output to the normalized table
outputs=vertcat(male(:,21),female(:,21));%Creating the output column
Norm_data=horzcat(Norm_data,outputs);%Appending the output to the normalised data 

%% Outliers removal by the use of Mahalanobis distance
mahabalonis_dist=zeros(1,row);
for i=1:row
    mahabalonis_dist(i) = mahal(Norm_data(i,:),Norm_data);
end

outlier=[];
for i=1:col
	if mahabalonis_dist(i)>100 %Some threshold
		outlier=[outlier,i];
	end
end
%% Outliers removal from the data

for i=1:length(outlier)    %Removal of the outlier data
   Norm_data(outlier(i),:)=[];
   outlier=outlier-1;
end

%%