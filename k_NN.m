%%Use of Inbuilt function for use of KNN Classifier

[features, outputs, alldata] = xlsread('voice.csv');% reading a csv file to seperate categories
outputs=outputs(:,21);
outputs=outputs(2:3169,1);

max_features=zeros(1,20);%Vector for finding the maximum of each features
for i=1:20
    max_features(i)=max(features(:,i));
end

normalized_features=zeros(3168,20);
for i=1:20 %matrix which has normalized features
    normalized_features(:,i)=features(:,i)/max_features(i);
end


mdl=fitcknn(features,outputs);
mdl.NumNeighbors = 3;

rloss = resubLoss(mdl); % resubstitution loss

cvmdl = crossval(mdl);%cross-validated classifier
kloss = kfoldLoss(cvmdl); %cross-validation loss

%Predict using average of all the features
mean_features = mean(features); % an average flower
mean_features_prediction = predict(mdl,mean_features);

%%
%Classification by taking only two features

[features, outputs, alldata] = xlsread('voice.csv');% reading a csv file to seperate categories
outputs=outputs(:,21);
outputs=outputs(2:3169,1);

sub_features=features(:,1);% Selecting the first two columns of the data
sub_features1=features(:,7);
gscatter(features(:,1),features(:,7),outputs);
xlabel('Pitch');% pitch features in the X axis (from 1st column of sub_features)
ylabel('Standard deviation'); % Standard deviation in y-axis (2nd column)
scatter(1:3168,features(:,9))
mdl=fitcknn(sub_features,outputs);
mdl.NumNeighbors = 3;


%%
load fisheriris
x = meas(:,3:4);
gscatter(x(:,1),x(:,2),species)
set(legend,'location','best')

newpoint = [5 1.45];
line(newpoint(1),newpoint(2),'marker','x','color','k','markersize',10,'linewidth',2);

[n,d] = knnsearch(x,newpoint,'k',3);
line(x(n,1),x(n,2),'color',[.5 .5 .5],'marker','o','linestyle','none','markersize',10)

%%
Am=zeros(20,20);
for i=1:20
    j=features(:,i);
    for k=1:20
    m=features(:,k);
   Am(i,k)=corr(j,m); 
    end    
end