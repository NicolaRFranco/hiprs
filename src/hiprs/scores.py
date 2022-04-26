import numpy as np
import pandas as pd
from time import perf_counter
from mlxtend.frequent_patterns import apriori
from scipy.stats import norm
from IPython.display import clear_output
from sklearn import linear_model as LM
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from hiprs.interactions import DummiesDictionary, Pattern, Interaction
from hiprs.mrmr import mRMR, mutual_info

class Classifier(object):
    """Abstract class representing a generic classification model. All objects of this class should have the following attributes:
    
       Attributes:
           model    (Object)         An object provided with the method 'predict_proba'. The latter should take
                                     as input the observed covariates X and should return an Nx2 array of predicted
                                     probabilities, where N is the number of observations and the two columns
                                     refer to the estimates of P(Y = 0 | X = x) and P(Y = 1 | X = x), respectively.
           clock    (hiprs.Clock)    An object used to measure the fitting time.
           covs     (list)           A list of strings that label the model covariates.
    """
    
    def auc(self, xtest, ytest):
        """Computes the Area Under the ROC Curve using the provided test data.
        Input:
            xtest    (pandas.DataFrame)    observed covariates in the test data
            ytest    (numpy.ndarray)       observed outcomes in the test data
        
        Output: 
            (float) AUC score.
        
        Note: the method assumes that the model has been fitted already."""
        yscores = self.predict(xtest)
        return roc_auc_score(ytest, yscores)
    
    def ap(self, xtest, ytest):
        """Computes the Average Precision over the provided test data.
        Input:
            xtest    (pandas.DataFrame)    observed covariates in the test data
            ytest    (numpy.ndarray)       observed outcomes in the test data
        
        Output: 
            (float) AP score.
        
        Note: the method assumes that the model has been fitted already."""
        yscores = self.predict(xtest)
        return average_precision_score(ytest, yscores)
    
    def predict(self, x):
        """Returns the predicted probability that Y = 1 given X.
        Input:
            x    (pandas.DataFrame)    observed value of the covariates X.
        
        Output:
            (numpy.ndarray) Estimates for P(Y = 1 | X = x)."""
        return self.model.predict_proba(x)[:,1]
    
    def score(self, x):
        """Returns the model score at a given X. The score is computed as logit( P(Y = 1 | X = x)) - m,
        where m is the model intercept.
        Input:
            x    (pandas.DataFrame)    observed value of the covariates X.
        
        Output:
            (numpy.ndarray) Estimates for logit(P(Y = 1 | X = x)) - m."""
        
        p = self.predict(x)
        return np.log(p) - np.log(1-p) - self.model.intercept_
    
    def fittingtime(self):
        """Returns the amount of time (seconds) elapsed while fitting the classifier."""
        return self.clock.elapsed()
    
    def betas(self):
        """Returns predictors and corresponding coefficients in the fitted ADDITIVE model.
        Output:
            (dict) A dictionary where each predictor is associated to the corresponding effect size.
        """
        mask = self.model.coef_[0] != 0
        return {cov:val for cov, val in zip(self.covs[mask], self.model.coef_[0][mask])}

class PRS(Classifier):
    """Implementation of an additive PRS based on Logistic Regression model. 
    Objects of this class have the following additional attributes:
    
    Attributes:
        penalty    (string)    string that specifies whether a penalty term is added during the loglikelihood optimization.
                               Accepted values are 'none', 'l1' (Lasso regression), 'elasticnet' and 'l2' (Ridge regression).
        l1_ratio   (float)     Proportion of l1 penalty vs l2 penalty. Automatically deduced from self.penalty.
        solver     (string)    Solver to be used for fitting the model. Automatically deduced from self.penalty.
                               """
    
    def __init__(self):
        """Initialize the PRS."""
        self.penalty = None
        self.l1_ratio = None
        self.solver = None
        self.clock = Clock()
            
    def fit(self, x, y, penalty = 'none'):
        """Fits the Logistic Regression model over the given training data.
        Input:
            x       (pandas.DataFrame)    observed values of the covariates in the training data.
            y       (numpy.ndarray)       observed outcomes in the training data.
            penalty (string)              type of penalty to be added during the optimization.
        """
        xdummies = pd.get_dummies(x.astype('category'))
        self.penalty = penalty
        self.l1_ratio = None
        if(penalty == 'l1'):
            self.solver = 'liblinear'
        elif(penalty == 'elasticnet'):
            self.solver = 'saga'
            self.l1_ratio = 0.5
        elif(penalty == 'l2' or penalty == 'none'):
            self.solver = 'lbfgs'
        else:
            raise RuntimeError("Unrecognized penalty type.")
        self.clock.start()
        self.model = LM.LogisticRegression(penalty = self.penalty, max_iter = 500,
                                           solver = self.solver, l1_ratio = self.l1_ratio).fit(xdummies, y)
        self.clock.stop()
        self.covs = np.array([str(s) for s in xdummies.columns])
        
    def transform(self, x):
        """Given observed values of the covariates x, returns the corresponding dummified dataset 
        (ignoring possibly unseen levels during training).
        Input:
            x    (pandas.DataFrame)    observed values of the covariates.
       
        Output:
            (pandas.DataFrame) dummified version of x.
            """
        res = {str(k):None for k in self.covs}
        for k in self.covs:
            snp, val = str(k).split("_")
            res[str(k)] = (x[snp] == int(val))+0
        return pd.DataFrame(res)
    
    def predict(self, x):
        return super(PRS, self).predict(self.transform(x))


class hiPRS(Classifier):
    """Implementation of hiPRS as detailed in https://www.biorxiv.org/content/10.1101/2022.04.22.489134v1. It consists of an interaction-aware classifier
    with categorical covariates. Aside from those coming from the superclass, hiPRS objects have the following
    additional attributes:

    dummies_dict    (interactions.DummiesDictionary)    An object for switching between patterns and interactions.
    levels_dict     (dict)                              A dictionary that to each (encoded) variable associates
                                                        the corresponding number of levels.
    interactions    (list)                              List of interactions.Interaction objects, containing all
                                                        candidate interaction terms.
    relevances      (numpy.ndarray)                     Array of all relevance measures I(X, Y), where X varies
                                                        in self.interactions.
    redundancies    (numpy.ndarray)                     Matrix with i,j entries S(X_i, X_j), where S is the similarity measure
                                                        while X_i, X_j vary in self.interactions.
    selected        (tuple)                             Tuple either containing a single list or a pair of lists.
                                                        In the first case, the list is comprised of interactions.Interaction
                                                        objects and contains the filtered list of candidates, that is
                                                        the one actually used in the classification model. If self.selected
                                                        is a pair, then the two lists refer to the Risk and Protection
                                                        interactions, as in https://doi.org/10.1016/j.radonc.2021.03.024.
    """

    def __init__(self):
        """Initializes a new hiPRS model."""
        self.clock = Clock()
        self.dummies_dict = None
        self.levels_dict = None
        self.selected = None

    def establish_encoding(self, covariates_dataset):
        """Builds the dictionaries self.dummies_dict and self.levels_dict using the given data.

        Input:
            covariates_dataset    (pandas.DataFrame)    Dataset containing only the variables that should
                                                        be later considered as building blocks for the interactions
                                                        (in the most simple setting, these are the SNPs).
                                                        NOTE: variables should be CATEGORICAL."""
        self.dummies_dict = DummiesDictionary(covariates_dataset)

        self.levels_dict = dict()
        covariates = list(covariates_dataset.columns)
        N = len(covariates)
        for i in range(N):
            aux, levels = covariates_dataset[covariates[i]].factorize()
            self.levels_dict.update({(i+1):len(levels)})

    def encrypt(self, interaction_list):
        """Given a list of interactions, returns the corresponding list of (encoded) patterns.

        Input:
            interaction_list    (list) List of interactions.Interaction objects.

        Output:
            (list) List with the corresponding interactions.Pattern objects."""
        return [x.to_pattern(self.dummies_dict) for x in interaction_list]

    def decrypt(self, patterns_list):
        """Decodes a list of patterns by returning the corresponding interactions (cfr. hiPRS.encrypt).

        Input:
            patterns_list    (list)    List of interactions.Pattern objects.

        Output:
            (list) List with the corresponding interactions.Interaction objects."""
        return [Interaction(x, self.dummies_dict) for x in patterns_list]


    def train(self, covariates_data, target_data, threshold = 0.1, maxlength = None, verbose = False):
        """Trains hiPRS on the provided data, finding interaction candidates and computing all measures of
        relevance and redundancy. Interactions are searched in the class of observations having
        Y = 1, which is assumed to be the MINORITY CLASS.

        Input:
            covariates_data    (pandas.DataFrame)    Observed values of the covariates (SNPs).
            target_data        (pandas.DataFrame)    Observed outcomes.
            threshold          (float)               Minimal empirical frequency required by an
                                                     interaction to be selected. Should be a value in [0,1].
                                                     Default value is 0.1.
            maxlength          (int)                 Maximal length admitted for the interactions.
                                                     None value corresponds to no upper bound on the
                                                     interactions length. Default value is None.
            verbose            (bool)                Whether the steps performed during training should
                                                     be reported explicitely on the console. Default is False.
        """
        clock = Clock()
        clock.start()

        display = Display()
        display.mute = not verbose
        display.cleanup = False

        self.establish_encoding(covariates_data)
        cov_ds = self.dummies_dict.encrypt(covariates_data)
        cov_ds = cov_ds.astype('category')
        dummy_cov = pd.get_dummies(cov_ds, dtype = bool)

        ones = (target_data == 1).values
        zeros = (target_data == 0).values

        ones_dummies = dummy_cov[ones]
        n_ones = np.shape(ones_dummies)[0]
        zeros_dummies = dummy_cov[zeros]
        n_zeros = np.shape(zeros_dummies)[0]

        if(maxlength == None):
            maxlength = np.shape(cov_ds)[1]

        frequent_set = apriori(ones_dummies, min_support=threshold, use_colnames=True, max_len=maxlength)
        frequent_set['length'] = frequent_set['itemsets'].apply(lambda x: len(x))
        frequent_set.reset_index(drop=True, inplace=True)

        Nfreq = len(frequent_set.index)
        display.printf("Interaction search completed. %d candidate interactions have been found.\nEncoding interactions as patterns..."
                       % Nfreq)

        def encodeDummie(dummie):
            return ( str(dummie).replace('_', '.') )

        def encodePattern(patt):
            L = len(patt)
            s = [0]*L
            for j in range(L):
                s[j] = encodeDummie(patt[j])
            s.sort()
            code = s[0]
            for j in range(1, L):
                code = code+'-'+s[j]
            return code

        codes = []
        patterns = []
        for rule in range(Nfreq):
            codes.append(encodePattern(list(frequent_set.itemsets[rule])))
            patterns.append(Pattern.parse(codes[-1]))
        self.interactions = np.array(self.decrypt(patterns))

        display.printf("Computing incompatibility matrix for faster computations...")
        pdummies = [Pattern.parse(d.replace("_",".")) for d in dummy_cov.columns]
        ndummies = len(pdummies)
        incompatibility_matrix = np.zeros(shape = (ndummies, Nfreq), dtype = bool)
        for i in range(ndummies):
            incompatibility_matrix[i,:] += np.array([(not pattern.compatible_with(pdummies[i])) for pattern in patterns])

        display.printf("Counting frequencies for each individual...")
        tab = np.zeros((n_zeros+n_ones, Nfreq+1))
        tab[target_data == 1,:-1] = 1-np.dot(ones_dummies.values,  incompatibility_matrix)
        tab[target_data == 0,:-1] = 1-np.dot(zeros_dummies.values, incompatibility_matrix)
        tab[:,-1] = target_data
        self.table = pd.DataFrame(np.ndarray.astype(tab, 'int'), columns = list(self.interactions)+['Target'])

        display.printf("Computing relevance and redundancy measures...")
        self.relevances = mutual_info(self.table.iloc[:,:-1], self.table.iloc[:,-1])
        variables = self.dummies_dict.lab_map.keys()
        res = pd.DataFrame({var:[""]*len(self.interactions) for var in variables})
        for k, i in enumerate(self.interactions):
            for vr, vs in zip(i.vars, i.values):
                res[vr][k] = vs
        dumint = pd.get_dummies(res)
        for d in dumint.columns:
            if d[-1]=="_":
                dumint.drop(d, axis = 1, inplace = True)
        dumint = np.ndarray.astype(dumint.values, 'float64')
        self.redundancies = np.dot(dumint, dumint.T)
        clock.stop()
        display.printf("... done. Training completed. Elapsed time: %s." % clock.elapsedTime())

    def select(self, *ninteractions):
        """Apply the mRMR algorithm to select a given number of interactions from self.interactions.

        Input:
            *ninteractions    (tuple)    one or two integers specifying the number of interactions to be selected.
                                         If one number is passed, the selection proceeds in the usual way. If two
                                         numbers (k1, k2) are passed, then the algorithm performs two separate
                                         selections to find k1 risk interactions and k2 protective interactions
                                         (cf. https://doi.org/10.1016/j.radonc.2021.03.024).
        Output:
            (tuple) If *ninteractions contains a single number, then a tuple with a single list of K interactions
            is returned, the latter being stored as interactions.Interaction objects. If *ninteractions is a
            a pair (k1, k2), then two lists of lenght k1 and k2 are returned (resp. risk and protective terms).
        """
        nlists = len(ninteractions)
        if(nlists == 1):
            selected =  mRMR(self.interactions,
                             relevance = self.relevances, redundancies = self.redundancies, m = ninteractions[0])
            return tuple([selected])
        elif(nlists == 2):
            ORs = oddsratio(self.table.iloc[:,:-1], self.table.iloc[:,-1])
            risk = (ORs >= 1.0)
            protection = (ORs < 1.0)
            rselected = mRMR(self.interactions[risk],
                             relevance = self.relevances[risk], redundancies = self.redundancies[risk][:,risk],
                             m = ninteractions[0])
            pselected = mRMR(self.interactions[protection],
                             relevance = self.relevances[protection], redundancies = self.redundancies[protection][:,protection],
                             m = ninteractions[1])
            return rselected, pselected
        else:
            raise RuntimeError("Invalid value tuple lenght for ninteractions. Accepted lenghts are 1 and 2.")

    def picked(self):
        """Returns the selected interactions self.selected in string format."""
        return tuple([[str(s) for s in l] for l in self.selected])

    def transform(self, x):
        """Given a dataset with the observed covariates (e.g. SNPs), returns the corresponding observed values
        attained by the model regressors. The latter can be either interactions (as in the classical case),
        or just the Risk and Protection scores, as in the splitted variant.

        Input:
            x    (pandas.DataFrame)    Observed covariates values.

        Output:
            (pandas.DataFrame) Observed value for the regressors."""
        if(len(self.selected)==1):
            return Interaction.make_dataset(x, self.selected[0])
        if(len(self.selected)==2):
            rx = Interaction.make_dataset(x, self.selected[0]).values
            px = Interaction.make_dataset(x, self.selected[1]).values
            return pd.DataFrame({'Risk score':np.sum(rx, axis = 1), 'Protection score':np.sum(px, axis = 1)})

    def fit(self, x, y, ninteractions, threshold = 0.1, maxlength = None, verbose = False):
        """Fits the data, i.e. (1) trains the hiPRS model by finding the interaction candidates, (2) filters the list
        of candidates to provide a simpler model, (3) estimates the beta coefficient in the regression model.
        After this call, the hiPRS object can be used for prediction.

        Input:
            x                (pandas.DataFrame)    Observed covariates values (training data).
            y                (pandas.DataFrame)    Observed outcomes (training data).
            ninteractions    (int or tuple)        Number of interactions to be selected for the
                                                   regression model. If an integer K is passed, then K
                                                   interactions are selected, as in the usual implementation.
                                                   If a pair of integers (k1, k2) is passed, then two
                                                   separate lists of risk and protection interactions are
                                                   selected, leading to a regression model with only two
                                                   regressors (resp. Risk and Protection score).
           threshold         (float)               Minimal empirical frequency required by an
                                                   interaction to be selected. Should be a value in [0,1].
                                                   Default value is 0.1.
            maxlength        (int)                 Maximal length admitted for the interactions.
                                                   None value corresponds to no upper bound on the
                                                   interactions length. Default value is None.
            verbose          (bool)                Whether the steps performed during training should
                                                   be reported explicitely on the console. Default is False."""
        self.clock.start()
        self.train(x, y, threshold, maxlength, verbose)
        if(isinstance(ninteractions, int)):
            self.selected = self.select(ninteractions)
            self.covs = np.array([str(s) for s in self.selected[0]])
        else:
            self.selected = self.select(*ninteractions)
            self.covs = np.array(['Risk score', 'Protection score'])
        self.model = LM.LogisticRegression(penalty = 'none', max_iter = 500).fit(self.transform(x), y)
        self.redundancies = None
        self.clock.stop()


    def predict(self, x):
        """Returns the predicted probability P(Y = 1 | X = x).

        Input:
            x    (pandas.DataFrame)    Observed values of the covariates (e.g. SNPs).

        Output:
            (numpy.ndarray) Predicted probabilities. Polygenic scores can be obtained by applying the logit transform
            to such values.
        """
        return self.model.predict_proba(self.transform(x))[:,1]

    def predict_proba(self, x):
        """Returns the predicted probabilities [P(Y = 0 | X = x), P(Y = 1 | X = x)].

        Input:
            x    (pandas.DataFrame)    Observed values of the covariates (e.g. SNPs).

        Output:
            (numpy.ndarray) Predicted probabilities for the two classes {0,1}, stored in an Nx2 array.
        """
        prob = self.predict(x)
        return np.stack((1.0-prob, prob), axis = 1)


class Display(object):
    """Auxiliary class for displaying messages on the output console. Objects of this class have the following attributes:

       Attributes:
           mute       (bool)    Whether next calls should actually print outputs or not.
           cleanup    (bool)    If True, clears the output console before printing a new message."""

    def __init__(self):
        """Initializes a Display with default values mute = False and cleanup = True."""
        self.mute = False
        self.cleanup = True

    def printf(self, string):
        """Prints the given string (if the Display is not muted).
        Input:
            string    (str)    string to be displayed on the console."""
        if(not self.mute):
            if(self.cleanup):
                clear_output(wait=True)
            print(string)


class Clock(object):
    """Auxiliary class for measuring time intervals. Objects of this type have the following attributes:

       Attributes:
           tstart    (float)    the time (in seconds) at which the clock was started.
           tstop     (float)    the time (in seconds) at which the clock was stopped."""

    def __init__(self):
        """Initializes a clock with tstart = tstop = 0."""
        self.tstart = 0
        self.tstop = 0

    def start(self):
        """Starts the clock."""
        self.tstart = perf_counter()

    def stop(self):
        """Stops the clock."""
        self.tstop = perf_counter()

    def elapsed(self):
        """Returns the amount of time passed tstop-tstart. If the clock has not been stopped
        yet, an error is returned.

        Output:
            (int) Seconds passed between the calls .start() and .stop()."""
        dt = self.tstop-self.tstart

        if(dt<0):
            raise RuntimeError("Clock still running, cannot access the elapsed time.")
        else:
            return dt

    def elapsedTime(self):
        """As Clock.elapsed but yields an hour-minutes-seconds representation.

        Output:
            (str) Elapsed time between the calls .start() and .stop()."""
        dt = self.elapsed()
        h = dt//3600
        m = (dt-3600*h)//60
        s = dt-3600*h-60*m

        if(h>0):
            return ("%d hours %d minutes %.2f seconds" % (h,m,s))
        elif(m>0):
            return ("%d minutes %.2f seconds" % (m,s))
        else:
            return ("%.2f seconds" % s)
