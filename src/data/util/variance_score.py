import numpy as np

class VarianceScorer:

    def score(x, y):
        """
        Get the variance for each column of X.

        Because principal components have decreasing variance
        (i.e. PC4 has less variance than PC3 which has less variance
        than PC2 etc.), we can use this function in SelectKBest to select
        only the top X number of principal components.

        """
        scores = [np.var(column) for column in x.T]
        return scores, np.array([np.NaN]*len(scores))
