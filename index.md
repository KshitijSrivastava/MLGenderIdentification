# Machine Learning- Gender Recognition

## INTRODUCTION
Human ear has an excellent mechanism of perceiving the voice. It distinguishes the voice based on factors such as the loudness, frequency, the pitch and the resonating frequency. We are going to test several classification algorithms mentioned above on the voice dataset provided by kaggle.com [1]. The dataset consists of 3,168 recorded voice samples, collected from male and female speakers. 20 acoustic properties of each voice are measured and included within the first 20 columns of the CSV file, whereas the genders of samples are placed at the last column. The purpose of our work is to figure out whether the voice belongs to male or female by using various statistical analysis and machine learning algorithms. We will also be analyzing and using various data preprocessing techniques to make the data more clean that can be generalized easily to apply the various Machine Learning algorithms to predict the gender of the sample based on the data samples.

## PROBLEM
A human ear can distinguish between a male and female voice easily. If we want to teach a machine to do the same then what features of a voice would the machine require to classify ?
The voice of an adult male can range between 85 to 180 Hz and that of adult female ranges between 165 to 255 Hz[2]. We can see that although the frequency ranges differ quite a lot, there is a mid range where they seem to overlay. This is why differentiating voices only on the basis of frequency is challenging and the purpose of our work is to figure out whether the voice belongs to male or female by the use of various ML techniques based on the given features in our dataset.                                                                                                                                                  

Another interesting problem we have thought from this dataset is predicting the pitch or the mean frequency  of the instance using the various regression algorithms, in this case  we have a regression problem instead of classification problem as the above one.

## FEATURES
Features in the dataset were obtained from voice samples which are pre-processed by acoustic analysis in R using the seewave and tuneR packages they are as follows:

1. Mean frequency (meanfreq)
Mean normalized frequency of the spectrum of the audio signal, measured in kHz.

2. Standard Deviation of Frequency (sd)
Standard deviation measures the amount of variation or dispersion of data values. A low standard deviation indicates that the values are more closer to the mean, whereas a high standard deviation indicates that the values are more spread out.

3. Median Frequency (median)
Median frequency is the middle value of a dataset and is measured in kHz.

4. First Quantile (Q25)
Quantiles are the points dividing range of probability distribution into contiguous intervals with equal probabilities. It is the data value when the standard distribution goes beyond the first threshold. It is measured in kHz.

5. Third Quantile (Q75)
Similar to the first quartile, the third quantile is a data point when the standard deviation reaches the one third of the highest in the range. It is measured in kHz.

6. Interquartile Range (IQR)
Interquartile Range is the difference between the first (lower) and the third (upper) quartile and is measured in KHz. It is a measure of statistical dispersion.

7. Skewness (skew)
Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. The skewness value can be positive or negative, or even undefined. Skew can be thought to refer to the direction opposite to that where curve appears to be leaning.

8. Kurtosis (kurt)
Kurtosis is a measure of the ”tailedness” of the probability distribution of a real-valued random variable.Similar to skewness, kurtosis is a descriptor of the shape of a probability distribution and just as for skewness,there are different ways of quantifying it for a theoretical distribution and corresponding ways of estimating it from a sample from a population.

9. Spectral Entropy (sp.ent)
In general, entropy is nothing more than the measure of amount of disorders in a system. Spectrum Entropy tells us how different the distribution of energy is in the system. High value of spectral entropy indicates the existence of a constant similarity of energy (small variations). Low value of spectral entropy indicates high variances and irregularity.

10. Spectral Flatness (sfm)
Spectral Flatness is a measure used in digital signal processing to characterize an audio spectrum. It is also known as Wiener entropy. It is typically measured in decibels and provides a way to quantify how noise-like a sound is, as opposed to being tone-like.

11. Mode frequency (mode)
Mode frequency is the one occurring most in the entire dataset.

12. Frequency centroid (centroid)
The frequency centroid is a measure used in digital signal processing to characterize a spectrum. It indicates where the ”center of mass” of the spectrum is. Perceptually, it
has a robust connection with the impression of ”brightness” of a sound.

13. Mean Frequency (meanfun)
Average of fundamental frequency measured across acoustic signal.

14. Minimum frequency (minfun)
Minimum fundamental frequency measured across acoustic signal.

15. Maximum frequency (maxfun)
Maximum fundamental frequency measured across acoustic signal.

16. Average dominant frequency (meandom)
Average of dominant frequency measured across acoustic signal.

17. Minimum dominant frequency (mindom)
Minimum of dominant frequency measured across acoustic signal.

18. Maximum dominant frequency (maxdom)
Maximum of dominant frequency measured across acoustic signal.

19. Range of dominant frequency (dfrange)
Range of dominant frequency measured across acoustic signal.

20. Modulation index (modindx)
Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range. It describes by how much
the modulated variable of the carrier signal varies around its unmodulated level.

## DATA & PREPROCESSING

Figure 1 will come here

From Fig 1, we can see that the number of female and male samples are the same, indicating that there is no unbalanced problem in our dataset.

Figure 2 will come here

 In Fig 2.we have plotted the PDF function of every male and female data points present in the database for a particular feature against each other. We can observe that there is more overlap region in the distribution plots of features like centroid, meanfreq, minfun, modindx and Q75.  In addition, the distribution plots of features like IQR, meanfun, mode, Q25, sd have less overlap region i.e the values are more scattered[4].The features having less overlap region are more important features since they show greater variation in data which will be of great use in predicting the gender of the given instance. We can also analyze that as the overlap region between male & female instance of a particular feature increases the redundancy of the data from that particular feature also increases,this point is also supported by the analysis done further in the project.

Figure 3 will come here

The feature importance is shown in Fig 3. is extracted using three different methods i.e Bhattacharyya distance,Entropy,t-test.

### Bhattacharyya Distance

Bhattacharyya Distance measures the similarity of two discrete or continuous probability distributions. Bhattacharyya coefficient is a measure of the amount of overlap between two statistical samples or populations.Higher the value of Bhattacharyya Distance less is the overlap region[7].It can be seen from the plot above that meanfun, IQR, Q25 are the three most significant attributes in the dataset. The lower value of the Bhattacharyya distance indicates the data quite overlap each other while the higher value of it indicates that the pdf for male and female are quite apart and is an important features for classification

### Kullback –Leibler divergence

This is a measure of how one probability distribution diverges from a second probability distribution.It is widely used in characterizing the relative (Shannon) entropy in information systems, randomness in continuous time-series, etc. A Kullback–Leibler divergence of 0 indicates that we can expect similar behavior of two different distributions, while a Kullback–Leibler divergence of 1 indicates that the two distributions behave in such a different manner[8].

### T tests

In our binary classification problem,each sample is classified either into class C1 or class C2. t-Statistics[11] helps us to evaluate that whether the values of a particular feature for class C1 is significantly different from values of same feature for class C2. If this holds, then the feature can help us to better differentiate our data.

All the three test used here gives us similar results. Based on this, the most important 5 features of our dataset are selected 

- Mean Frequency
- Inter-Quantile Range
- Q25
- Standard Deviation
- Spectral Entropy

### CORRELATION

2 Figures will be attached here

The above figures shows that some of the variables have very high correlation which is more than 0.8 (Fig 4), such as meanfreq, centroid, Q25, standard deviation, median, spectral flatness, maximum dominant frequency, range of dominant frequency and skew. Features having high correlation gives redundant data and hence can be eliminated without much loss of information. Out of two highly related features, the one having higher variance with other features is eliminated.Due to the high correlation among variables, we used PCA to reduce the dimensions.

### STATISTICAL MEASURES

Figure will be attached 

We have performed various statistical techniques on our raw dataset and observed that the range of features are quite different. Some features such as feature-7 has a very large range, This will cause problem while using distance based techniques due to non-uniformity. Therefore scaling of features is need of the hour, and also for applying PCA one has to do mean normalization and scaling. 

## NORMALIZATION

Some machine learning algorithms based on the gradient descent or on distance measures like KNN are quite sensitive to the scale of the numeric values provided. Consequently, in order for the algorithm to converge faster or to provide a more exact solution, rescaling the distribution is necessary. Rescaling mutates the range of the values of the features and can affect variance, too. Hence Normalization becomes a very important step in data preprocessing. These can be done by using various statistical standardization like z-score normalization,min-max transformation, division by max value etc.

- Z-score normalization
It is  to center the mean to zero (by subtracting the mean) and then divide the result by the standard deviation.
                                               
- The min-max transformation 
This is to remove the minimum value of the feature and then divide by the range (maximum value - minimum value). This act rescales all the values between 0 to 1. It’s preferred to standardization when the original standard deviation is too small original values are too near or when you want to preserve the zero values in a sparse matrix.

- Division by max value 
It is basically dividing the whole feature dataset by its maximum value this process also rescales all the value between 0 to 1.

After applying all the above stated normalization techniques we have used the new normalized from each of the three techniques on the ML algorithm and found that all the mentioned techniques are giving equivalent results while Z-score normalization being slightly better off than rest.

## OUTLIERS

An outlier is an observation that diverges from an overall pattern on a dataset. We know that Machine learning algorithms are very sensitive to the range and distribution of these values. Here outliers can play spoilsport and mislead the training process resulting in longer training times, less accurate models and ultimately poorer results. Hence it becomes necessary for us to remove outliers. They are of two kinds: Univariate and multivariate. Univariate outliers can be found when looking at a distribution of values in a single feature space. Multivariate outliers can be found in a n-dimensional space (of n-features). We have multivariate data here. Many univariate outlier detection methods can be extended to handle multivariate data. The central idea is to transform the multivariate outlier detection task into a univariate outlier detection problem. Here we have used Mahalanobis distance and chi-square statistic for multivariate outlier detection. The method gives a univariate variable, and thus univariate tests such as Grubb’s test,interquartile range method,3-sigma method can be applied to this measure.

### Mahalanobis distance [5][6]
It is a measure of the distance between a point P and a distribution D, It is a multi-dimensional generalization of the idea of measuring how many standard deviations away P is from the mean of D. This distance is zero if P is at the mean of D, and grows as P moves away from the mean along each principal component axis, it measures the number of standard deviations from P to the mean of D. Mahalanobis distance is unitless and scale-invariant, and takes into account the correlations of the data set.
Using this method we get a univariate in which for checking  outlier we have set the threshold  as 97.5% Quantile of chi-square distribution[12].

### chi-square test [6]
It is a statistical test of independence to determine the dependency of two variables[13]. From the definition, of chi-square we can easily deduce the application of chi-square technique in feature selection. We have a target variable from the dataset of a feature and mean variable being the mean of the data points of the feature from which we select target variable. Now, we calculate chi-square statistics between every mean variable and the target variable and observe the existence of a relationship between the mean variables and the target[14]. Larger the chi-square statistic, the target/object is an outlier in our case above 97.5%[15]

### Grubbs’ Test [6] 
It is maximum normalized residual test or extreme studentized deviate test, is a statistical test used to detect outliers in a univariate data set  of a normally distributed population. Here we used the grubbs’ test by converting the multivariate data into a univariate data by mahalanobis distance and then used it on grub’s test. Grubbs' test detects one outlier at a time. This method will continue to run until all the outliers are removed from the dataset. Grubbs' test is defined for the hypothesis:

### Interquartile range method
We process the data using the above method, here we observe that if the value of the given data is less than the value of Q25-1.5*IQR or greater than that of  Q75+1.5*IQR, then we classify it as an outlier[6]. Where Q25 is 1st quartile, Q75 is the 3rd quartile and IQR is interquartile range and the outliers are detected.

### 3 sigma method
In this we basically observe that if the data value is greater or lesser than the distance from the mean of the features to three times its standard deviation value,then it is classified as a outlier.


Box plot figure will come here

Here on the each box, the central mark indicates the median, and the bottom and top edges of the box indicate the 25th and 75th percentiles, respectively. The whiskers extend to the most extreme data points not considered outliers, and the outliers are plotted using the '+' symbol.
As shown in the Figure 5, we see that for the normalized data, some of the samples across all the features are lying in the outlier region

Box plot after outlier removal system

By comparison of number of red crosses present in the Fig 5 with the red crosses present in Fig 6 we can clearly observe that lot of outliers (in features) have been taken care of, but some outliers are still present in Fig 6,which are restricted to a small number of features and are heavily concentrated near the boundary of the outlier removal. 

Here employing univariate method for detecting outlier such as IQR(box plot) as used above is not a right practice since the data is multivariate and we may be losing some valuable information. The data point which may seem like an outlier when just looking at univariate feature may actually not be an outlier when seen in n-dimensional plane where ‘n’ is the number of features.

Outlier detection table will come here


## Clustering analysis  and outlier detection:
K-means clustering is a type of unsupervised learning which is used to find groups which have not been explicitly labeled in the data.The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity[16]. The results of the K-means clustering algorithm are:

1. The centroids of the K clusters, which can be used to label new data
2. Labels for the training data (each data point is assigned to a single cluster)


Rather than defining groups before looking at the data, clustering allows one to find and analyze the groups that have formed organically. It helps in identifying unknown groups in complex data set. Once the algorithm has been run and the groups are defined, any new data can be easily assigned to the correct group. This is a versatile algorithm that can be used for any type of grouping. To find the number of clusters in the data, the user needs to run the K-means clustering algorithm for a range of K values and compare the results. A commonly used method to compare results across different values of K is the mean distance between data points and their cluster centroid. Mean distance to the centroid as a function of K is plotted and the "elbow point," where the rate of decrease sharply shifts and then does not change much, it can be used to roughly determine K. Here K=2 seems to be the right choice.The silhouette method may also be used for validating K.To get an idea of how well-separated the resulting clusters are, one can make a silhouette plot using the cluster indices output from kmeans. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters. This measure ranges from +1, indicating points that are very distant from neighboring clusters, through 0, indicating points that are not distinctly in one cluster or another, to -1, indicating points that are probably assigned to the wrong cluster. Average silhouette value for k=2 is 0.4599 and that for k=3 is 0.2774. Therefore K=2 is the choice for the number of clusters.                                                                                                                                     This clustering method is also used for finding outliers in the dataset. The points which are very far located in the cluster may be considered as outlier. We are getting 72 outliers using this clustering method.

Silhoutte image will come here
