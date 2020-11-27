# Wasserstein-Fourier

## A distance between time-series

The Wasserstein-Fourier (WF) distance is a new framework for analysing stationary time series based on optimal transport distances and spectral embeddings. 
First, we represent time series by their power spectral density (PSD), which summarises the signal energy spread across the Fourier spectrum. 
Second, we endow the space of PSDs with the Wasserstein distance, which capitalises its unique ability to preserve the geometric information of a set of distributions. 
These two steps enable us to define the Wasserstein-Fourier (WF) distance, which allows us to compare stationary time series even when they differ in sampling rate, length, magnitude and phase.


## Experiments
Basic properties and advantages of the WF distance are illustrated on synthetic data in the notebook '1_toy_data.ipynb'. 
The remaining 4 notebooks illustrate how the WF distance can be used on real world datasets. It focuses mainly on classification problems, we propose versions of Softmax and KNN classification algorithms equipped with the WF distance.  




