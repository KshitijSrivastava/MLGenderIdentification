[row,col]=size(Ndata); %Takes in Normalized data

%% k-means clustering

[cl1,C1]=kmeans(Ndata(:,1:col-1),2); %Inputing all the features with k=2  cl1 indicates classification , C1 is cluster centroid location
[cl2,C2]=kmeans(Ndata(:,1:col-1),3); %Inputing all the features with k=3  cl2 indicates classification , C2 is cluster centroid location

%% Finding the distance from each centeroid point of the cluster using the Euclidian distance

index1=[];
index2=[];
cl1_dist_k2_1=[]; %For storing the distance of each of the points from each clusters
cl1_dist_k2_2=[]; %For storing the distance of each of the points from each clusters
for i=1:length(Ndata)
    if cl1(i)==1 %If classified into cluster 1
        a=C1(1,:)-Ndata(i,1:col-1); %Subtracting the data point from the first cluster
        cl1_dist_k2_1=[cl1_dist_k2_1,sqrt(a*a')]; %Finding distance of classified points from each cluster centroid
        index1=[index1 ,i]; %Storing the index
    elseif cl1(i)==2 %If classified into cluster 2
        b=C1(2,:)-Ndata(i,1:col-1);
        cl1_dist_k2_2=[cl1_dist_k2_2,sqrt(b*b')]; %Finding distance of classified points from each cluster centroid
        index2=[index2 ,i];
    end
end
%% Outlier removal if distance more than the 3 times the standard deviation +median 
out1=[]; %For finding the index of the outliers of cluster 1
out2=[]; %For finding the index of the outliers of cluster 2

for i=1:length(index1) %For cluster 1
    if (cl1_dist_k2_1(i)>(3*std(cl1_dist_k2_1))+median(cl1_dist_k2_1))%If K means distance more than the median+ (3*sig)
       out1=[out1,index1(i)]; %Storing the index
    end
end

for i=1:length(index2) %For cluster 2
    if (cl1_dist_k2_2(i)>(3*std(cl1_dist_k2_2))+median(cl1_dist_k2_2))%If 
       out2=[out2,index2(i)]; %Storing the index       
    end

end

out=[out1,out2]; %Combining all the outliers found from both the clusters
%%
% if on increasing 'k' the mean distance doesn't change much then stop
% incresing k. Also higher the mean value of silhouette function output
% (i.e sil1, sil2) better 'k' value that is.
%To get an idea of how well-separated the resulting clusters are, you can make a silhouette plot using the cluster indices output from kmeans. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters. This measure ranges from +1, indicating points that are very distant from neighboring clusters, through 0, indicating points that are not distinctly in one cluster or another, to -1, indicating points that are probably assigned to the wrong cluster. silhouette returns these values in its first output.
%A more quantitative way to compare the two solutions is to look at the average silhouette values for the two cases.
% check documentation of silhouette function(k-means clustering doc)
figure;[sil1,h1]=silhouette(Ndata(:,1:col-1),cl1);
figure;[sil2,h2]=silhouette(Ndata(:,1:col-1),cl2);
%%