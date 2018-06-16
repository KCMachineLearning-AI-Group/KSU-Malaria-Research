from numpy import concatenate, nan, ndarray, append, array, std
from sklearn.feature_selection import f_regression
from multiprocessing import Pool
from pandas import DataFrame
import os


class InteractionChecker:
    def __init__(self, alpha=0.05):
        """
        Instantiates InteractionChecker
        :param alpha: Float value of the significance level to test each interaction
        :return: InteractionChecker
        """
        self.alpha        = alpha
        self.x_data       = None
        self.n            = 0
        self.y_data       = None
        self.prange       = None
        self.mat          = None
        self.int_list     = None
        self.interactions = None
        self.interaction_terms = None


    def fit(self, x_data, y_data, mp=True):
        """
        Loops through all potential interactions and adds to self.interactions if the p-value of the interaction coefficient is less than alpha.
        :param x_data: Pandas DataFrame of x values
        :param y_data: Pandas DataFrame of y values
        :param mp: set to False to use a single process for lower-dimensional data sets.
        :return: None
        """
        self.x_data    = x_data
        self.y_data    = y_data
        self.y         = self.y_data.values
        self.mat       = self.x_data.values.T
        self.n         = len(self.mat[0])
        self.prange    = range(0,len(self.x_data.columns)-1)
        self.int_list  = [(a, b) for a in self.prange for b in self.prange if a < b]
        # Test all potential interactions
        if mp:
            with Pool(processes=os.cpu_count() - 1) as pool:
                self.interactions =  pool.map(self.test_interaction,self.int_list)
        else:
            self.interactions =  [self.test_interaction(i) for i in self.int_list]

    def test_interaction(self, arg):
        """
        Tests a single interaction term with a linear regression model containing both terms and the ineraction term
        :param arg: Tuple of two predictor columns
        :return: False if interaction is rejected, an array of both predictors and the interaction values if interaction is not rejected
        """
        p1, p2 = arg
        inter = self.mat[p1]*self.mat[p2]
        # Reject if the standard devation of the interaction term is 0
        if std(inter) == 0:
            return False
        # Ndarray of both terms, plus the interaction term
        x = concatenate((self.mat[p1], self.mat[p2], inter)).reshape(self.n, 3)
        fr = f_regression(x,self.y_data)
        # Return both column indices and the interaction if the p-value of the interaction coefficient is less than alpha
        if fr[1][2] < self.alpha:
            return (self.x_data.columns[p1], self.x_data.columns[p2])
        # Reject if the p-value of the interaction coefficient is greater than alpha
        return False

    def transform(self, data):
        """
        Reshapes all interaction terms not rejected into a dataframe containing all interactions not rejected
        :return: Pandas dataframe object of all the interactions not rejected by test_interaction
        """
        # Column indices and values for non-rejected interactions
        sign_interacts = [i for i in self.interactions if i]
        columns=["{}*{}".format(i,j) for i, j in sign_interacts]
        m = [(data[i].values*data[j].values) for i, j in sign_interacts]
        return DataFrame(array(m).T, index=data.index, columns=columns)
