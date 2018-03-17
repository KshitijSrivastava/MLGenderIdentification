%%Use of Inbuilt function for use of Naive bayes Classifier
data=xlsread('voice.csv'); %Reading a CSV file
[features, outputs, alldata] = xlsread('voice.csv');% reading a csv file to seperate categories
outputs=outputs(:,21);
outputs=outputs(2:3169,1);

NBModel = fitNaiveBayes(features,outputs);%making a naive bayes model for 

cpre = predict(NBModel,features(2,:));%Predicting a value using the data

%%
%Use of Inbuilt function for use of Naive bayes Classifier for creating a
%test and train data
[features, outputs, alldata] = xlsread('voice.csv');% reading a csv file to seperate categories
outputs=outputs(:,21);
outputs=outputs(2:3169,1);

c = cvpartition(outputs,'k',2);

trIdx = c.training(1);
teIdx = c.test(1);
    
train_data=features(trIdx,:);
train_output=outputs(trIdx,:);
test_data=features(teIdx,:);
test_output=outputs(teIdx,:);

NBModel = fitNaiveBayes(train_data,train_output);%making a naive bayes model

predict=predict(NBModel,test_data);

classification_error=0;
for i=1:length(predict)
    if strcmp(predict(i,1),test_output(i,1))==0
        classification_error=classification_error+1;
    end
end
error_percent=(classification_error*100)/length(predict);
%%
data=xlsread('voice.csv'); %Reading a CSV file
[features, outputs, alldata] = xlsread('voice.csv');% reading a csv file to seperate categories
outputs=outputs(:,21);
outputs=outputs(2:3169,1);

out=zeros(3168,1);%For outputs in the form of zeros and ones
for i=1:3168
    if strcmp(outputs(i,1),'male')
        out(i,1)=1;%if male then 1
    elseif strcmp(outputs(i,1),'female')
        out(i,1)=0;%If female then 0
    end
end


% %data=csvread('voice.csv',2,1);
% table = readtable('voice.csv');
% 
% for i=1:3168
%     if table(i,21)=='male'
%         table(i,21)=1;
%     elseif table(i,21)=='female'
%         table(i,21)=0;
%     end
%    
% end
%%

load fisheriris;
y = species;
c = cvpartition(y,'k',10);
err = zeros(c.NumTestSets,1);

for i = 1:c.NumTestSets
    trIdx = c.training(i);
    teIdx = c.test(i);
    
    train_data=meas(trIdx,:);
    train_output=species(trIdx,:);
    test_data=meas(teIdx,:);
    test_output=species(teIdx,:);
    
    NBModel = fitNaiveBayes(train_data,train_output);
end

%%