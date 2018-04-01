clear all;
clc;
data1=xlsread('voice.csv');
data1(:,21)=0; %male data =0
data1(1:1585,21)=1; %Female Data=1
[row1,col1]=size(data1);
idx=randperm(row1);%Random shuffling of the index for making 
data2(row1,col1)=0;
for i=1:row1
   data2(i,:)=data1(idx(i),:); 
end    
data=data2;
%y(:,1)=data2(:,21);

[row,col]=size(data);

max_val_data=max(data(:,1:col-1));
min_val_data=min(data(:,1:col-1)); 
mean_data=mean(data(:,1:col-1));
std_data=std(data(:,1:col-1));

for i =1:col-1
   %Ndata(:,i)=data(:,i)/max(data(:,i));
   Ndata(:,i)=(data(:,i)-mean_data(i))/std_data(i); %mean normalisation
end  

Ndata(:,col)=data(:,col);%Appending the last column of the outputs

Correlation_matrix=corr(Ndata(:,1:col-1));

%% 
mu=mean(Ndata);
sigma=std(Ndata);
%[n,p] = size(Ndata);
outlier=[];
for i=1:col-1 
    for j=1:row   
        if abs(Ndata(j,i)-mu(i))>=3*sigma(i)
            outlier=[outlier,j];  %Add code here (at this position) to replace oulier with mean/median etc
        end    
    end    
end
outlier=unique(outlier);    

for i=1:length(outlier)
   Ndata(outlier(i),:)=[];
   outlier=outlier-1;
end    
%%
training=[Ndata(1:round(0.8*length(Ndata)),:)];
test=Ndata(round(0.8*length(Ndata))+1:length(Ndata),:);

k=5;                % k should not be even
test(:,col+1)=0;
for i=1:length(test)
    test_pt=test(i,1:col-1);
    y1(length(training),4)=0;
    y1(:,1)=training(:,col);
    for j=1:length(training)
        diff=test_pt-training(j,1:col-1);
        y1(j,2)=norm(diff);
    end   

    [y1(:,3),y1(:,4)]=sort(y1(:,2));

    for j=1:k
        y2(j)=y1((y1(j,4)),1); 
    end    
    clM=0;
    clF=0;
    for j=1:length(y2)
        if y2(j)==1
            clM=clM+1;
        else
            clF=clF+1; 
        end
    end   

    if clM>clF
        display('Male');
        test(i,col+1)=1;            
    else 
        display('Female');
    end    
    
end 

TP=0;
TN=0;
FP=0;
FN=0;
                % In test matrix: 21st col is actual and 22nd is predicted
                % M->positive
for i=1:length(test)
   if test(i,col)==1 && test(i,col+1)==1 
       TP=TP+1;
   else if test(i,col)==0 && test(i,col+1)==0 
       TN=TN+1;    
   else if  test(i,col)==0 && test(i,col+1)==1
       FP=FP+1;
   else FN=FN+1;
       end
       end
   end
end

Accuracy=((TP+TN)/(TP+TN+FP+FN))*100
Misclassification=100-Accuracy
PPV=TP/(TP+FP)
NPV=TN/(TN+FN)
tp=TP/(TP+FN)       % Sensitivity
tn=TN/(TN+FP)       % Specificity
%%
counting=0;

for i=1:3168
    for j=1:21
        if Ndata(i,j)==1
            counting=counting+1;
        end
    end
end

%%
