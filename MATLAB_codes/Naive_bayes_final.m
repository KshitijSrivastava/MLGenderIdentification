clear all
clc;

Ndata_maha_sig=Ndata;
[row,col]=size(Ndata);

%% Seperating into training and testing data

%Taking 80% of the data as training and 20% as the testing data
training=Ndata(1:round(0.8*row),:);
testing=Ndata(round(0.8*row)+1:row,:);

%% Training a model
NBModel = fitcnb(training(:,1:col-1),training(:,col));%making a naive bayes model for 

%% Predicting for the testing data
NB_prediction = predict(NBModel,testing(:,1:col-1));
true_output=testing(:,col);

%positive is Male and negative is female
TP=0;TN=0;FP=0;FN=0; 

for i=1:length(NB_prediction)
    
    if true_output(i)==1
        if NB_prediction(i)==1
            TP=TP+1;
        else
            FN=FN+1;
        end
        
    elseif true_output(i)==0
        if NB_prediction(i)==0
           TN=TN+1; 
        else
            FP=FP+1;
        end
    end
    
end

misclassification=100*(FP+FN)/(FP+TP+FN+TN);


