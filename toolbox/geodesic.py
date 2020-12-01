import numpy as np
import ot


def get_geodesic(h1, h2, support, n_step):
    
    """ 
     
    Compute the geodesic in the Wasserstein space between two distributions

    Parameters
    ----------

    h1         : one-dimensional histogram
    h2         : one-dimensional histogram
    support    : support of the histograms
    n_step     : number of interpolant

    Returns
    -------

    supports   : supports of the n_step interpolants
    values     : n_step interpolants (histograms)

    """
        
    u, v = np.meshgrid(support, support)
    cost = (u-v)**2
    cost = np.ascontiguousarray(cost, dtype='float64')
    
    plan = ot.emd(np.ascontiguousarray(h1), np.ascontiguousarray(h2), cost)
    taus = np.linspace(0, 1, n_step)
    supports = []
    values = []
    for tau in taus:
        bar_support = dict()
        for idx_i, i in enumerate(support):
            for idx_j, j in enumerate(support):
                if plan[idx_i, idx_j] > 0: 
                    if (tau*i + (1-tau)*j) in bar_support:
                        bar_support[tau*i + (1-tau)*j] +=  plan[idx_i, idx_j]
                    else:
                        bar_support[tau*i + (1-tau)*j] =  plan[idx_i, idx_j]
    

        s = list(bar_support.keys())
        v = list(bar_support.values())
        supports.append(s)
        values.append(v)
        
    return supports, values


def get_geodesic_entropy(h1, h2, support, n_step, reg = 1e-5):
    """

    Compute the geodesic between two distributions from entropy regularized from the Wasserstein space

    Parameters
    ----------

    h1         : one-dimensional histogram
    h2         : one-dimensional histogram
    support    : support of the histograms
    n_step     : number of interpolant
    reg        : parameter for entropy regularization of the optimal transport plan
    
    Returns
    -------

    supports   : supports of the n_step interpolants
    values     : n_step interpolants (histograms)

    """
    
    # compute cost
    u, v = np.meshgrid(support, support)
    cost = (u-v)**2
    cost = np.ascontiguousarray(cost, dtype='float64')

    A = np.transpose((h1,h2))
    taus = np.linspace(0, 1, n_step)

    supports = []
    values = []
    for tau in taus:
        w = np.array([1 - tau, tau])
        values.append(ot.bregman.barycenter(A, cost, reg, w))     
        supports.append(support)
    
    return supports, values
        
        
        
