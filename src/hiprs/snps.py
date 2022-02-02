import numpy as np
import pandas as pd

def xgenerate(n, p, seed = 0):
    """Generates n obervations of p SNPs.
    
    Input:
         n        (int)     number of observations.
         p        (int)     number of SNPs.
         seed     (int)     fixes a random seed for reproducibility.
         
    Output:
         (numpy.ndarray)     n x p array taking values in {0,1,2}.
         """
    np.random.seed(seed)
    R = np.round(3*np.random.rand(n, p)) % 3
    R[:, 2] = 0
    R[:, 1] = R[:,0]
    return np.ndarray.astype(R, 'int')

def AND(*args):
    """Logical AND operator acting over boolean vectors."""
    res = np.ones(len(args[0]))
    for x in args:
        res = res * (x + 0)
    return res
    
def OR(*args):
    """Logical OR operator acting over boolean vectors."""
    return np.max(np.stack(args), axis = 0)

def response(x, noise):
    """Computes the response given the SNPs values (with possible mislabels).
    
    Input:
         x         (numpy.ndarray)     Observed SNPs values (each SNP must refer to a column).
         noise     (float)             probability by which each single observation may be mislabeled.
    
    Output:
         (numpy.ndarray)     Observed outcomes.
    """
    y = OR(AND(x[:, 3]==2,  x[:, 4]==2,  x[:, 5]==2, x[:,6]==2), 
           AND(x[:, 3]==1,  x[:, 4]==1,  x[:, 5]!=2, x[:,6]!=2),
           AND(x[:, 7]==x[:,8], x[:,8]==x[:,9]))   
    
    y = np.ndarray.astype(y, 'int')
    prob = 1.0-noise
    mislabel = np.random.rand(len(y)) >= prob
    y[mislabel] = 1-y[mislabel]
    return y

def generate(n, p, noise, seed = 0):
    """Generates a random dataset consisting of n observations of p SNPs and a response variable ('Outcome').
    Data is randomly perturbed, via label flipping, accordingly to the probability 'noise'.
    
    Input:
        n         (int)       number of observations.
        p         (int)       number of SNPs.
        noise     (float)     probability that each single observation has of being mislabeled.
        seed      (int)       fixes a random seed for reproducibility of the results.
        
    Output:
        (pandas.DataFrame)     Simulated data.
    """
    X = xgenerate(n, p, seed)    
    Y = response(X, noise)
    
    data = {'SNP%d' % i:X[:,i-1] for i in range(1, p+1)}
    data.update({'Outcome': Y})
    return pd.DataFrame(data)