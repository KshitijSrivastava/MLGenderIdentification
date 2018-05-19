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
