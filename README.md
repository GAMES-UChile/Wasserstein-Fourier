# Wasserstein-Fourier

Python code to reproduce the results of the paper "The Wasserstein-Fourier Distance for Stationary Time Series", Elsa Cazelles, Arnaud Robert, Felipe Tobar".
ArXiv : https://arxiv.org/abs/1912.05509

## A distance between time-series

The Wasserstein-Fourier (WF) distance is a new framework for analysing stationary time series based on optimal transport distances and spectral embeddings. 
First, we represent time series by their power spectral density (PSD), which summarises the signal energy spread across the Fourier spectrum. 
Second, we endow the space of PSDs with the Wasserstein distance, which capitalises its unique ability to preserve the geometric information of a set of distributions. 
These two steps enable us to define the Wasserstein-Fourier (WF) distance, which allows us to compare stationary time series even when they differ in sampling rate, length, magnitude and phase.


## Content

* toy_data.ipynb              : highlights basic properties, advantages and applications of the WF distance on synthetic data
* classification_exemples/    : includes 4 notebooks illustrating how the WF distance can be used for classification on real world datasets. Each notebook is dedicated to a different dataset, yet the same algorithms are used, namely softmax and KNN equipped with various distances such as KL, L2 and Wasserstein. 
* PGA_fungi.ipynb             : principal geodesic analysis for the fungi dataset, as well as GPCA on Gaussian distributions for informative purpose
* interpolant_C_elegans.ipynb : geodesic path between two signals from the C. elegans database using the proposed WF distance
* toolbox/                    : contains the code for all the algorithms presented in the paper such as principal geodesic analysis, knn and softmax.
* figures                     : some of the figures produce by the code
* data                        : small dataset and/or preprocessed dataset. Larger dataset could be freely downloaded online: [Urbansound8k](https://urbansounddataset.weebly.com/urbansound8k.html).
