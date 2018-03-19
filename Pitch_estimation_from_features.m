[features, outputs, alldata] = xlsread('voice.csv');% reading a csv file to seperate categories
outputs=outputs(:,21);
outputs=outputs(2:3169,1);

pitch=features(:,1);%Finding the pitch features from the data

classification=zeros(3168,1);%Converting the classisifcation variable to integer classifcation

for i=1:3168 %Classifiying Male as 1 and female as 0 and storing in classification array
    if strcmp(outputs(i,1),'male')==1
        classification(i,1)=1;
    else
        classification(i,1)=0;
    end
end

features(:,1)=classification; %Replacing the the pitch as a features with classification

% sub_features=features(:,1:3);

% modelfun=@(b,x)b(1) + b(2)*sub_features(:,1) + b(3)*sub_features(:,2) + b(4)*sub_features(:,3);
% 
% beta0=[10,10,10,10];

%mdl=fitnlm(sub_features,pitch,modelfun,beta0);
mdl = fitlm(features,pitch,'linear','RobustOpts','on')

pitch(:,2)=0;
for i=1:3168
    X_newdata=features(i,:);
    pitch(i,2) = predict(mdl,X_newdata);
    
end



%%
load carbig
X = [Horsepower,Weight];
y = MPG;
modelfun = @(b,x)b(1) + b(2)*x(:,1).^b(3) + ...
    b(4)*x(:,2).^b(5);
beta0 = [-50 500 -1 500 -1];
mdl = fitnlm(X,y,modelfun,beta0)

