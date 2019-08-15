import numpy as np
from scipy.interpolate import interp1d


def inverse_histograms(mu, S, Sinv, method='linear'):
    """

    Given a distribution mu compute its inverse quantile function

    Parameters
    ----------

    mu     : histogram
    S      : support of the histogram
    Sinv   : support of the quantile function
    method : name of the interpolation method (linear, quadratic, ...)

    Returns
    -------

    cdfa   : the cumulative distribution function and
    q_Sinv : the inverse quantile function of the distribution mu

    """

    epsilon = 1e-14
    A = mu>epsilon
    A[-1] = 0
    Sa = S[A]

    cdf = np.cumsum(mu)
    cdfa = cdf[A]
    if (cdfa[-1] == 1):
        cdfa[-1] = cdfa[-1] - epsilon

    cdfa = np.append(0, cdfa)
    cdfa = np.append(cdfa, 1)

    if S[0] < 0:
        print('weird for a psd!')
        Sa = np.append(S[0]-1, Sa)
    else:
        # set it to zero in case of PSDs
        Sa = np.append(0, Sa)
    Sa = np.append(Sa, S[-1])

    q = interp1d(cdfa, Sa, kind=method)
    q_Sinv = q(Sinv)
    return cdfa, q_Sinv


def get_barycenter(mus, S, n, method='linear'):
    """

    Compute the Wasserstein barycenter of 1d-distributions

    Parameters
    ----------

    mus    : NxD matrix that contains the D samples of N distributions
    S      : the D support points
    n      : the number of points used for the support of the quantile function


    Returns
    -------

    res      : the Wasserstein barycenter of distributions mus.
    Finv     : the inverse quantile fuction of distributions mus.
    Finv_bar : the inverse quantile function of the barycenter.

    """

    N, D = mus.shape
    Finv = np.zeros((N,n))
    Sinv = np.linspace(0, 1, n)
    for i in range(N):
        _, Finv[i,:] = inverse_histograms(mus[i,:], S, Sinv, method)

    Finv_bar = np.mean(Finv, axis=0)
    Sd = np.append(S[0]-1, S)
    cdf = interp1d(Finv_bar, Sinv, bounds_error=False, fill_value=(0,1),kind=method)
    cdf_S = cdf(S)
    res = cdf_S.copy()
    res[1:] = res[1:] - res[:-1]
    return res, Finv, Finv_bar


def get_distance(quantiles, quantile_bar):
    """
    Compute the square Wasserstein distance between a set of distributions and a barycenter

    Parameters
    ----------

    quantiles     :
    quantiles_bar :

    Returns
    -------

    distances :

    """

    distances = np.sum((quantiles-quantile_bar)**2, axis=1)
    return distances
