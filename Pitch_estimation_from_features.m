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

modelfun=@(b,x)b(1) + b(2)*features(:,1) + b(3)*features(:,2) + b(4)*features(:,3);

beta0=[10,10,10,10];

mdl=fitnlm(features,pitch,modelfun,beta0);

X_newdata=[1,0.0642412677031359,0.0320269133725820,0.0150714886459209,0.0901934398654331,0.0751219512195122,12.8634618371626,274.402905502067,0.893369416700807,0.491917766397811,0,0.0597809849598081,0.0842791064403210,0.0157016683022571,0.275862068965517,0.00781250000000000,0.00781250000000000,0.00781250000000000,0,0];

pitch_new = predict(mdl,X_newdata);

%%
load carbig
X = [Horsepower,Weight];
y = MPG;
modelfun = @(b,x)b(1) + b(2)*x(:,1).^b(3) + ...
    b(4)*x(:,2).^b(5);
beta0 = [-50 500 -1 500 -1];
mdl = fitnlm(X,y,modelfun,beta0)

