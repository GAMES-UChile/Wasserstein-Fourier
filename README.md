# Wasserstein-Fourier

Python code to reproduce the results of the paper "The Wasserstein-Fourier Distance for Stationary Time Series", Elsa Cazelles, Arnaud Robert, Felipe Tobar".
ArXiv : https://arxiv.org/abs/1912.05509

## A distance between time-series

The Wasserstein-Fourier (WF) distance is a new framework for analysing stationary time series based on optimal transport distances and spectral embeddings. 
First, we represent time series by their power spectral density (PSD), which summarises the signal energy spread across the Fourier spectrum. 
Second, we endow the space of PSDs with the Wasserstein distance, which capitalises its unique ability to preserve the geometric information of a set of distributions. 
These two steps enable us to define the Wasserstein-Fourier (WF) distance, which allows us to compare stationary time series even when they differ in sampling rate, length, magnitude and phase.


## Content

* toy_data.ipynb              : main script with basic properties, advantages and applications of the WF distance illustrated on synthetic data
* classification_exemples/    : includes 4 notebooks illustrating how the WF distance can be used for classification on real world datasets. We also propose versions of Softmax and KNN classification algorithms equipped with the WF distance.  
* PGA_fungi.ipynb             : principal geodesic analysis for the fungi dataset, as well as GPCA on Gaussian distributions for informative purpose
* interpolant_C_elegans.ipynb : geodesic path between two signals from the C. elegans database using the proposed WF distance
* toolbox/                    : various functions
* figures                     : some of the figures produce by the code
* data                        : data used in various applications