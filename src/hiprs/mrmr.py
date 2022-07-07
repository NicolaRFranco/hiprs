import warnings
import numpy as np
    
def contingency(x, y):
    """Computes the contingency tabels of multiple covariates x and a single outcome y.
    
    Input:
         x     (pandas.DataFrame)     Observed values in the x variables.
         y     (numpy.ndarray)        Observed values in the y variable.
         
    Output:
         (tuple)     Four numpy.ndarrays containg the four entries in the contigency tables of x vs y.
                     Each array is a vector of length p, where p is the number of covariates.
    """
    ztable, otable = x[y==0].values, x[y==1].values
    b = np.sum(ztable, axis = 0)
    a = np.sum(otable, axis = 0)
    d = np.sum(y==0) - b
    c = np.sum(y==1) - a    
    return a, b, c, d

def mutual_info(x, y):
    """Computes the (empirical) mutual information of x and y, namely I(x, y).
    
    Input:
         x     (pandas.DataFrame)     Observed values in the x variables.
         y     (numpy.ndarray)        Observed values in the y variable.
         
    Output:
         (numpy.ndarray)     Mutual information of each covariate with respect to y. Results in a vector
                             of lenght p, where p is number of covariates (columns of x).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b, c, d = contingency(x, y)
        n = len(y)
        res = ((np.log(n*a/((a+c)*(a+b)))*a +
                np.log(n*b/((b+d)*(a+b)))*b +
                np.log(n*c/((a+c)*(c+d)))*c +
                np.log(n*d/((c+d)*(b+d)))*d)/n)
        res[np.isnan(res)] = 0
        return res
    
def oddsratio(x, y):
    """Computes the oddsratio of multiple covariates x (independently) and a single outcome y.
    NOTE: the anscombe correction is always applied to ensure nonzero denominators.
    
    Input:
         x     (pandas.DataFrame)     Observed values in the x variables.
         y     (numpy.ndarray)        Observed values in the y variable.
         
    Output:
         (numpy.ndarray)     Odds-ratio of each predictor when associated to y. Results in a single vector
                             of length p, where p is the number of covariates (columns of x).
    """
    a, b, c, d = contingency(x, y)
    return ((a+0.5)*(d+0.5))/((b+0.5)*(c+0.5))
    
def mRMR(variables, relevance, redundancies, m, correction = False, quantile = None):    
    """Implementation of the mRMR selection algorithm (quotient variant).
    
    Input:
         variables     (numpy.ndarray)     Array listing all candidate variables.
         relevance     (numpy.ndarray)     Relevance measures of the given variables.
         redundancies  (numpy.ndarray)     2D array with all pairwise redundancy measures
                                           of the given variables.
         m             (numpy.ndarray)     Number of variables to be selected.
         correction    (bool)              Whether the relevance/redundancy ratio should be
                                           substituted or not with relevance/(redundancy+1)
                                           to avoid dividing by zero. Default value is False,
                                           in which case possible divisions by zero are
                                           intended as evaluating to infinity (if several
                                           infinite ratios are encountered, higher relevances
                                           are preferred).
    Output:
         (numpy.ndarray)     Sublist of the input variables consisting of the selected ones only.
    """
    if (len(variables)<m):
        raise RuntimeError("Cannot select %d terms as there are only %d candidates." % (m, len(variables)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        delta = int(correction)
        picked = np.array([False]*len(relevance))
        
        q = -1.0 if(quantile == None) else np.quantile(relevance, quantile)
        while(np.sum(picked)<m):
            redund = np.mean(redundancies[picked], axis = 0)
            redund[np.isnan(redund)] = 1
            V = relevance / (delta+redund)
            V[picked] = -np.inf
            V[np.isnan(V)] = -np.inf
            V[relevance < q] = -np.inf
            if(np.max(V) == np.inf):
                relev = relevance + 0
                relev[V != np.inf] = -np.inf
                new = np.argmax(relev)
            else:
                new = np.argmax(V)
            picked[new] = True
            
    return variables[picked]
