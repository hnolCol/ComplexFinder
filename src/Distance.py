
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
class DistanceCalculator(object):

    def __init__(self,X):
        ""
        print(self._euclidean(X))
        print(self._pearson(X))


    def _pearson(self,X, rowvar=True):
        ""
        return 1-np.corrcoef(X,rowvar=rowvar)

    def _euclidean(self,X):
        ""
        return squareform(pdist(X,"euclidean"))



    def _ajaxScore(self):
        ""



if __name__ == "__main__":
    X = np.array([
        [1,4,5,6,2],
        [3,4,5,6,4],
        [4,4,5,2,1]
    ])
    DistanceCalculator(X)






