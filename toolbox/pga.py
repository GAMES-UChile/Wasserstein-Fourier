import numpy as np
import ot

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import scipy.linalg as la

from toolbox.exact_barycenter import *

def logMap(h, Fb, S, method='slinear'):
    """
    Compute the logarithmic map of an one-dimensional histogram at a reference measure

    Parameters
    ----------
    
    h      : an one-dimensional histogram
    Fb     : histogram of the reference measure
    S      : support of the histogram
    method : smoothing method used among 'pchip', 'linear' and 'spline'

    Returns
    -------
    
    Finv   : smoothed quantile function of the input histogram barycenter
 
    """
    
    epsilon = 1e-15
    Fb[0] = 0
    A = (h > epsilon)
    A[-1] = 0
    Sf = S[A]
    Sf = np.append(S[0]-1, Sf)
    Sf = np.append(Sf, S[-1])

    H = np.cumsum(h);
    Hf = H[A]
    if (Hf[-1] == 1):
        Hf[-1] = Hf[-1] - epsilon
    Hf = np.append(0, Hf)
    Hf = np.append(Hf, 1)

    
    interpolation = interp1d(Hf, Sf, kind = method)
    FFinv = interpolation(Fb)

    V = FFinv - S
    
    return V


def consecutive(vect, stepsize=1):  
    '''  
    Partition an input vector V into smaller series of subvectors of consecutive increasing elements
    '''
    
    return np.split(vect, np.cumsum( np.where(vect[1:] - vect[:-1] > 1) )+1)



def pushforward_density(T, bar, Omega):  
    """
    
    Compute the pushforward density of a histogram by a map non-necessary increasing

    Parameters
    ----------

    T        : map used to push the histogram
    f        : histogram to transport
    OmegaExt : support of the histogram f to transport

    Returns
    -------
    
    hfinal   : the histogram transported by the map T from the measure f

    """
    
    epsilon = 1e-5
    inter = Omega[1]-Omega[0]

    cst = np.where(np.abs(T[0:-1]-T[1:])<epsilon)[0]
    aa = consecutive(cst)

    noncst = np.arange(len(Omega))
    noncst = np.delete(noncst,cst)
    bb = consecutive(noncst)

    dR = -savgol_filter(T[noncst],15,1, deriv = 1)/inter

    interpolation = interp1d(T[noncst], bar[noncst]/np.abs(dR),fill_value='extrapolate', kind = 'linear')
    g = interpolation(Omega[noncst])

    h = np.zeros(len(Omega))
    h[noncst] = g

    if aa:
        pass
    else:
        if aa[0][0] == 0:
            h[aa[0]] = 0
        else:
            for i in np.arange(aa):
                h[aa[i]] = h[a[i][0]-1]

    return h



def perform_GPCA(mu, Omega, n):
    """
    
    Compute the Geodesic Principal component Analysis for one dimensional probability distributions

    Parameters
    ----------

    mu      : matrix of histograms of size n*len(Omega)
    Omega   : support of the histograms
    n       : number of histograms

    Returns
    -------
    
    bar     : Wasserstein barycenter of the histograms
    tV      : Values of the projection locations
    eigV    : Eigenvectors of the covariance of the log-map matrix
    Vc      : Euclidean mean of the log-map matrix

    """

    ## Barycenter
    n_inv = 100000
    bar, Finv_mu, Finv_bar, cdf_bar = get_barycenter(mu, Omega, n_inv, method = 'linear')

    ## Compute log maps
    V = np.zeros((n,len(Omega)))
    for i in np.arange(n):
        V[i,:] = logMap(mu[i,:],cdf_bar,Omega,'linear')

    ## Compute PCA on logmap (log-PCA)
    Vc = np.mean(V,0) # log-map matrix
    Vp = np.matmul(V-Vc,np.diag(np.sqrt(bar)))
    C = np.matmul(Vp.T,Vp)/np.size(Vp,0) # covariance matrix

    eigValsV, eigV = la.eig(C) # Perform PCA

    # Normalisation
    nonzero_ind = (bar > 0)
    eigV[nonzero_ind,:] = np.matmul( np.diag(1/np.sqrt(bar[nonzero_ind])) , eigV[nonzero_ind,:] )

    eigValsV = np.diag(eigValsV.real)
    eigV = eigV.real
    tV = np.matmul(V-Vc,np.matmul(np.diag(bar),eigV)) # inner product of the dataset with the eigenvectors

    
    return bar, tV, eigV, Vc






