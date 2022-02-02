import numpy as np
import pandas as pd


class Dummie(object):
    """Class for representing dummie variables. Objects of Dummie type have the following attributes.
    
         Attributes:
              var     (str or int)     Label indicating to which variable the dummie corresponds.
              val     (int)            Value / level to which the dummie refers."""
    
    def __init__(self, variable, value):
        """Creates a dummie variable.
        
        Input:
             variable     (str or int)     Label for the original variable.
             value        (int)            Value / level to which the dummie refers.
            
        NOTE: Labels should NOT contain the character '.' ."""
        self.var = variable
        self.val = value
        
    def __eq__(self, other):
        """Checks whether two dummies are the same (that is, if they share both the same label and value).
        
        Input:
             other     (interactions.Dummie)     Dummie to be compared with the current one.
             
        Output:
             (bool)."""
        return (self.var == other.var and self.val == other.val)
    
    def __str__(self):
        """String representation of the dummie variable, where a '.' separates label and value."""
        return str(self.var)+"."+str(self.val)
    
    def __hash__(self):
        """Returns a hash that uniquely determines the dummie (as long as the labels do NOT contain the '.' character!!)."""
        return hash(str(self))
    

class Pattern(object):
    """Class for representing patterns, i.e. sequences or sets, of dummie variables. 
    Objects of type Pattern have the following attributes:
    
        Attributes:
            elements    (frozenset)    Collection of dummies that form the pattern.
        
    The Pattern class is a poset with the convention that 'p <= q iff the dummies of p are contained in q'."""
    
    def __init__(self, dummies):
        """Creates a Pattern from a list (or a set) of dummie variables.
        
        Input:
            dummies    (list or set)    Dummies that form the pattern.
            
        NOTE: Pattern object should not contain two or more dummies that refer to the same variable but different levels."""
        self.elements = frozenset(dummies)
        
        
    def toset(self):
        """Returns the collection of dummies within the pattern in the form of a frozenset."""
        return self.elements
    
    def __hash__(self):
        """Returns a hash that uniquely determines the pattern (assuming the dummies in the pattern admit a unique hash)."""
        return hash(self.elements)
    
    def __len__(self):
        """Returns the number of dummies (int) within the pattern."""
        return len(self.elements)
    
    def __str__(self):
        """Returns a string representation of the pattern. Dummies are listed in lexicographic order, 
        e.g. if a = Dummie('G', 0), b = Dummie('AB', 1) and p = Pattern([a,b]), then str(p) = 'AB.1-G.0'."""
        s = [str(e) for e in self.elements]
        s.sort()
        return "-".join(s)
        
    def __le__(self, other):
        """Returns whether the current pattern is contained in the other one.
        
        Input:
            other    (interactions.Pattern).
            
        Output:
            (bool)."""
        return ((self.elements).issubset(other.elements))
                
    def __ge__(self, other):
        """Returns whether the current pattern contains the other pattern (cf. interactions.Pattern.__le__)."""
        return ((self.elements).issuperset(other.elements))
    
    def __lt__(self, other):
        """Returns whether the current pattern is strictly contained in the other pattern (cf. interactions.Pattern.__le__)."""
        if(len(self) >= len(other)):
            return False
        else:
            return (self <= other)
                
    def __gt__(self, other):
        """Returns whether the current pattern strictly contains the other pattern (cf. interactions.Pattern.__le__)."""
        if(len(self) <= len(other)):
            return False
        else:
            return (self >= other)
    
    def __eq__(self, other):
        """Returns whether the two patterns share the same dummies."""
        return (self <= other and self >= other)
    
    
    def __pow__(self, other):
        """Returns the 'weak' intersection of two patterns.
        
        Input:
            other    (interactions.Pattern)    The pattern to be intersected with the current one.
            
        Output:
            (interactions.Pattern)    Pattern object corresponding to the biggest pattern contained both in self and other.
        """
        return Pattern(self.elements.intersection(other.elements))
    
    def __mul__(self, other):
        """Returns the 'strong' intersection of two patterns.
        
        Input:
            other    (interactions.Pattern)    The pattern to be intersected with the current one.
            
        Output:
            (interactions.Pattern)    If the two patterns are compatible, the weak intersection is returned (cf.
                                      interactions.Pattern.__pow__); otherwise, the operation results in an 
                                      empty pattern, i.e. Pattern([]).
        Example: 
            Let a, b, c, be patterns with string representation 'A.1-B.0', 'A.1-C.0' and 'A.0-B.0' 
            respectively. Then a*b = 'A.1' while a*c = '' (on contrary, notice that a**c = 'B.0')."""
        if(self.compatible_with(other)):
            return self**other
        else:
            return Pattern([])
        
    def __add__(self, other):
        """Union of two patterns, when possible. Note: this is not a commutative operation!
        
        Input: 
            other    (interactions.Pattern)    The Pattern to be added to the current one.
            
        Output: 
            (interactions.Pattern)    If the two patterns are compatible, their union is returned; 
                                      otherwise the original pattern (self) is returned.
            
        Example: 
            Let a, b, c, be patterns with string representation 'A.1-B.0', 'A.1-C.0' and 'A.0-B.0' 
            respectively. Then a+b = 'A.1-B.0-C.0' while a+c = a = 'A.1-B.0' and c+a = c = 'A.0-B.0'.
            The idea is that second pattern is added to the first one IF that is an allowed operation
            (patterns should never contain two (or more) dummies that refer to the same variable but
            different values!)."""
        if(self.compatible_with(other)):
            return Pattern(self.elements.union(other.elements))
        else:
            return self
        
    
    def __sub__(self, other):
        """Subtraction of two patterns. Given two patterns p and q, the result p-q is defined to
        be the pattern having the dummies in p that are not in q.
        
        Input: 
            other    (interactions.Pattern)    The Pattern to be subtracted from the current one.
            
        Output:
            (interactions.Pattern)    Pattern object consisting of the dummies in self that are not in other."""
        return Pattern(self.elements.difference(other.elements))
    
    
    def isempty(self):
        """Boolean valued function. Returns whether a pattern has length 0."""
        return len(self)==0
    
    def split(self):
        """Returns a list of atomic patterns (i.e. Pattern objects of length one) that form
        the current pattern.
        
            Output: 
                (list)    List of Pattern objects, each corresponding to one of the dummies in self."""
        return [Pattern.parse(str(e)) for e in self.elements]
            
                
    def compatible_with(self, other):
        """Boolean valued method. Checks whether the current pattern is compatible with another one.
        Two patterns are said to be incompatible if they contain dummies that refer to the same variable but different values.
        
        Output:
            other    (interactions.Pattern)    Pattern object whose compatibility will be checked with respect to the current one.
            
        Example: 
            Let a, b, c, be patterns with string representation 'A.1-B.0', 'A.1-C.0' and 'A.0-B.0' 
            respectively. Then a and b are compatible, a.compatible_with(b) returns True, whereas a and c are not."""
        res = True        
        for a in self.elements:
            for b in other.elements:
                res = res and (a.var != b.var or a.val == b.val)
                if(not res):                
                    break
            if(not res):                
                    break                    
        return res
    
    
    def subpatterns(self):
        """Returns a list of all the patterns that are contained in the current one.
        
        Output:
            (list) List of interactions.Pattern objects."""
        N = len(self)
        lista = list(self.elements)
        
        if(N==0):
            return [self]       
        else:
            res = [Pattern([])]
            for i in range(N):
                basic = frozenset([lista[i]])
                temp = Pattern(lista[(i+1):]).subpatterns()
                for x in temp:
                    x.elements = x.elements.union(basic)
                res += temp
            return res
    
    
    @classmethod
    def parse(cls, string):        
        """Given a string representation of a pattern, returns the corresponding interactions.Pattern object.
        
        Input:
            string    (str)    string representing the pattern to be created.
            
        Output:
            (interactions.Pattern)     Pattern deduced from the input 'string'. The string should follow the
                                       convention by which dummies are separated with the character '-'. 
                                       It is furhter assumed that each dummie is written in the
                                       format label.value, e.g. 'B.10', 'zeta.3', 'H0.72').
        """
        s = string.split("-")      
        return Pattern([Dummie(*tuple([int(y) for y in x.split(".")]) ) for x in s])
    
    @classmethod
    def parse_all(cls, List_of_strings):
        """Given a list of representations, returns the corresponding list of patterns (cf. interactions.Pattern.parse).
        
        Input:
            List_of_strings    (list)    String representations of multiple patterns.
            
        Output:
            (list)    A list of interactions.Pattern objects."""
        return [Pattern.parse(s) for s in List_of_strings]
    
    @classmethod
    def tostring(cls, List_of_patterns):
        """Given a list of patterns, returns a list of the corresponding string representations.
        
        Input:
            List_of_patterns    (list)    List of interactions.Pattern objects.
            
        Output:
            (list)    List of strings."""
        return [str(p) for p in List_of_patterns]
    
    
    @classmethod
    def same_variables(cls, p, q):
        """Returns whether two patterns p and q are defined through dummies that correspond to the same variables.
        
        Input:
            p   (interactions.Pattern).
            q   (interactions.Pattern).
            
        Output:
            A boolean variable that equals True iff the two patterns refer to dummies based on exactly the same variables (but
            possibly different levels)."""
        if(len(p)!=len(q)):
            return False
        else:
            varp = [x.var for x in p.elements]
            varq = [x.var for x in q.elements]
            varp.sort()
            varq.sort()
        return (varp == varq)
    
    

class DummiesDictionary(object):
    """Class for encoding (and decoding) dummy variables with acronyms. 
    The convention is to introduce a number for each categorical variable and a number for the 
    corresponding level held by the dummie. Objects of this class have the following attributes:
    
    Attributes:
        lab_map        (dict)    Dictionary that to each variable name assigns a number
        inv_lab_map    (dict)    The inverse dictionary of 'lab_map' (used for decoding)
        var_maps       (dict)    A dictionary of dictionaries. Returns the levels encoding of each
                                 encoded variable.
        inv_var_maps   (dict)    A dictionary of dictionaries that inverts 'var_maps'. Returns the levels
                                 decoding of each variable."""
    
    def __init__(self, dataset):        
        """Given a dataset, it assigns a short label (a capital letter) to each variable, then computes
        the number of levels for each variable and assigns a corresponding number. The dataset is not
        changed in anyway: all the information are stored within the DummiesDictionary to allow for
        later encoding\decoding.
        
        Input:
            dataset    (pandas.DataFrame)    dataset of CATEGORICAL variables.
        """
        names = dataset.columns
        nvars = len(names)
        labels = [(i+1) for i in range(nvars)]
        self.lab_map = {names[i]:labels[i] for i in range(nvars)}
        self.inv_lab_map = {labels[i]:names[i] for i in range(nvars)}
        self.var_maps = {labels[i]:dict() for i in range(nvars)}
        self.inv_var_maps = {labels[i]:dict() for i in range(nvars)}
        
        for i in range(nvars):
            temp = list(set(dataset.iloc[:,i]))
            aux = dict(list(enumerate(temp)))
            self.inv_var_maps[labels[i]] = aux
            self.var_maps[labels[i]] = {v: k for k, v in aux.items()}                
    
    def encrypt(self, dataset):
        """Encodes a given dataset using the conventions in the DummiesDictionary.
        Input:
            dataset    (pandas.DataFrame)    Dataset to be encoded.
        Output:
            (pandas.DataFrame) Encoded version of the original dataset."""
        newcolumns = [self.lab_map[c] for c in dataset.columns]
        newdata = [[self.var_maps[self.lab_map[c]][x] for x in dataset[c]] for c in dataset.columns]        
        return pd.DataFrame(np.array(newdata).T, columns = newcolumns)    
        
    def decrypt(self, dataset):
        """Decodes a given dataset using the conventions in the DummiesDictionary.
        Input:
            dataset    (pandas.DataFrame)    Dataset to be decoded.
        Output:
            (pandas.DataFrame) Decoded version of the encoded dataset.
        
        NOTE: if X is a pandas.DataFrame of categorical variables and D = DummiesDictionary(X) then
        D.decrypt(D.encrypt(X)) = X. If Y is another dataframe with the same variables as X, where
        levels are indicated using the same labels, then D.decrypt(D.encrypt(Y)) = Y too, assuming
        all levels in Y where observed in X."""
        newcolumns = [self.inv_lab_map[c] for c in dataset.columns]
        newdata = [[self.inv_var_maps[c][x] for x in dataset[c]] for c in dataset.columns]        
        return pd.DataFrame(np.array(newdata).T, columns = newcolumns)
        None
       
    

class Interaction(object):
    """Class for dealing with interactions terms of dummie variables. Objects this type
    have the following attributes:
    
    Attributes:
        vars      (list)    Labels of the variables involved in the interaction
        values    (list)    Labels of the levels attained by each variable in the interaction
        
    Example: if I is an interaction corresponding to the dummie variables "age = 10", 
    "hair = blonde", then I.vars = ["age", "hair"] while I.values = ["10", "blonde"].
    
    The construction of interaction terms can be done in multiple ways: exploiting the
    interaction-pattern duality (see Interaction.__init__) or by parsing a string
    representation of the interaction (see Interaction.parse)."""
    
    def __init__(self, pattern, dummies_dict = None):
        """Given a pattern of dummie variables, costructs the corresponding interaction.
        It is assumed that patterns are using an encoded format, reason for which a DummiesDictionary
        for decoding is needed. Not passing the dummies_dict is allowed if and only if the pattern is empty (null interaction).
        
        Input:
            pattern         (interactions.Pattern)    Pattern describing the interaction.
            dummies_dict    (DummiesDictionary)       Dictionary to be use for decoding the pattern. 
                                                      Default value is 'None'."""
        self.vars, self.values = [], []
        dummies = list(pattern.elements)
        N = len(dummies)
        for i in range(N):
            self.vars.append(dummies_dict.inv_lab_map[dummies[i].var])
            self.values.append(dummies_dict.inv_var_maps[dummies[i].var][dummies[i].val])
        self.adjust_order()
        
    def adjust_order(self):
        """Sorts the vectors self.vars and self.values so that variables are listed in lexicographic order."""
        argsort = np.argsort(self.vars)
        self.vars = [self.vars[i] for i in argsort]
        self.values = [self.values[i] for i in argsort]
                
    def __str__(self):      
        """Yields a string representation of the interaction term. For an interactions consisting of k dummies,
        the convention is 'DUMMIE1-DUMMIE2-..-DUMMIEK' where each dummie is printed as: '(variable, value)'."""
        s = ""
        for x,y in zip(self.vars, self.values):
            s += "("+x+", "+str(y)+")-"            
        return s[:-1]
    
    def to_shortstring(self):        
        """Yields a shortformat string representation of the the interaction term. 
        For interactions consisting of more than 5 dummies, the middle part of the string is replaced with 3 dots '...'.
        
        Output:
            (str)."""
        def f(variables, values):
            aux = ""
            for x,y in zip(variables, values):
                aux += "("+x+", "+str(y)+")-"            
            return aux[:-1]        
        s = str(self)
        if( len(self.vars)>5 ):
            s = f(self.vars[:2], self.values[:2])+"-...-"+f(self.vars[-2:], self.values[-2:])
        return s
    
    def __len__(self):
        """Returns the number of variables involved in the interaction.
        
        Output:
            (int)."""
        return len(self.vars)
        
    def get_dummies(self):
        """Returns a list of the dummies present in the interaction.
        
        Output:
            (list)    List of interactions.Dummie objects.
            
        NOTE: Dummies are created without preliminar encoding of the variables. 
              This should be considered a private method and the public use is discouraged."""
        N = len(self)
        return [Dummie(self.vars[i], self.values[i]) for i in range(N)]
        
    def compatible_with(self, other):
        """Checks if the given interaction term is compatible with another one.
        Two interaction terms are incompatible if they present two different dummies for the same variable. 
        Equivalently, if the product of the two results in the null interaction.
        
        Input:
            other    (interactions.Interaction) Object to be compared with the current one.
            
        Output:
            (bool)    Equals True iff self and other are compatible."""
        p1 = Pattern(self.get_dummies())
        p2 = Pattern(other.get_dummies())
        return p1.compatible_with(p2)
    
    def __mul__(self, other):
        """Algebraic multiplication of two interactions.
        
        Input:
            other    (interactions.Interaction) Interaction to be multiplied with self.
        
        Output:
            (interactions.Interaction)    Algebraic multiplication of the two interactions
                                          intended as polynomials of dummie variables. If 
                                          the two interactions are incompatible, then the 
                                          output results in the null Interaction."""
        if(self.compatible_with(other)):
            p1 = Pattern(self.get_dummies())
            p2 = Pattern(other.get_dummies())
            p = p1+p2
            res = Interaction.empty()
            for e in p.elements:
                res.vars.append(e.var)
                res.values.append(e.val)
            res.adjust_order()
            return res
        else:
            return Interaction.empty()
     
    def count(self, dataset):
        """Observed values of the interaction term on a given dataset.
        
        Input:
            dataset    (pandas.DataFrame)    Dataset whos columns contain the interaction variables.
                                             For each observation in the dataset, the value for the 
                                             interaction will be computed.
            
        Output:
            (numpy.ndarray)    Interaction values along the dataset."""
        N = np.shape(dataset)[0]
        values = np.array([True]*N)
        
        for var, val in zip(self.vars, self.values):
            values *= (dataset[var] == val).values
            
        return values.astype(int)
        
    def get_datacount(self, dataset, label = None):
        """As Interaction.count but creates a whole separate dataset, the interaction being
        labelled as indicated (if label = None, the string representation is used).
        
        Input:
            dataset    (pandas.DataFrame)    Dataset whos columns contain the interaction variables.
                                             For each observation in the dataset, the value for the 
                                             interaction will be computed.
            label      (str)    Label for the interaction term. Default value is None.
            
        Output:
            (pandas.DataFrame)    Observed values of the interaction."""
        if(label == None):
            label = str(self)
        return pd.DataFrame(pd.Series( self.count(dataset) , name = label  ))
    
    def to_pattern(self, dummies_dictionary):
        """Encodes the interaction as a Pattern, using the format provided by a DummiesDictionary.
        
        Input:
            dummies_dictionary    (DummiesDictionary)    Object used for encoding the interaction.
            
        Output:
            (interactions.Pattern)."""
        dummies = []
        for x, y in zip(self.vars, self.values):
            variable = dummies_dictionary.lab_map[x]
            value = dummies_dictionary.var_maps[variable][y]
            dummies.append(Dummie(variable, value))            
        return Pattern(dummies)    
    
    @classmethod
    def empty(cls):
        """Returns the null interaction."""
        return Interaction(Pattern([]))
        
    @classmethod
    def parse(cls, string):
        """Given a string representation of an interaction (..,..)-...-(..,..), yields the corresponding
        object of Interaction type. Can be seen as the inverse of Interaction.__str__.
        
        Input:
            string    (str)    Representing the interaction.
            
        Output:
            (interactions.Interaction)."""
        s = string.split("-")
        variables, vals = [], []
        for t in s:
            aux = t.split(",")
            variables.append(aux[0][1:])
            try:
                vals.append(int(aux[1][:-1]))      
            except:
                vals.append(aux[1][:-1])  
        x = Interaction.empty()
        x.vars = variables
        x.values = vals
        return x
    
    @classmethod
    def make_dataset(cls, dataset, List_of_interactions):
        """Given a list of interactions and a dataset, creates a new dataset containing the corresponding values 
        of all interactions in the list (cf. Interaction.get_datacount).
        
        Input:
            dataset                 (pandas.DataFrame)    Observed values for the categorical variables.
            List_of_interactions    (list)                List of interactions.Interaction objects.
        Output:
            (pandas.DataFrame)    Observed values for the listed interactions.
            """
        L = List_of_interactions
        data =  L[0].get_datacount(dataset)

        for i in range(1, len(L)):
            data = data.join(L[i].get_datacount(dataset))
    
        return data