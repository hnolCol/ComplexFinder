
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
class DistanceCalculator(object):

    def __init__(self,X):
        ""
        print(self.euclidean(X))
        print(self.pearson(X))


    def pearson(self,X, rowvar=True):
        ""
        return 1-np.corrcoef(X,rowvar=rowvar)

    def euclidean(self,X):
        ""
        return squareform(pdist(X,"euclidean"))


    def apex(self,X):

        if X.size[1] != 2:
            raise ValueError("X must be of size (,2) containg mu and sigma")

        return squareform(pdist(X,lambda e1,e2: self.apexScore(e1,e2)))

        
    def apexScore(self, entry1, entry2):
        ""
        mu1,sigma1 = entry1
        mu2, sigma2 = entry2
        return np.sqrt((mu1-mu2)**2 + (sigma1-sigma2)**2)


    def getDistanceMatrix(self,X, distance="pearson"):
        ""
        if hasattr(self,distance):
            return getattr(self, distance)(X)
        
        
        raise ValueError("Distance unknown.")



if __name__ == "__main__":
    X = np.array([
        [1,4],
        [3,4],
        [4,3]
    ])



    print(DistanceCalculator(X).apex(X))






