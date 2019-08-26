
import numpy as np
import pandas as pd
import itertools
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
class DistanceCalculator(object):

    def __init__(self):
        ""
        #print(self.euclidean(X))
        #print(self.pearson(X))


    def pearson(self,X, rowvar=True):
        ""
        print("pearson is equal to 1-r therefore the range of r is [0,2]")
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

    def _transformToLong(self,ids,distM,distance):
        ""
        df = pd.DataFrame(columns = ["idx","distance"])
        distMatrix = pd.DataFrame(distM,index=ids,columns=ids)
        for idPair in itertools.combinations(ids, 2):
            e1,e2 = idPair
            df = df.append({"idx":"{};{}".format(e1,e2),
                       "distance":distMatrix.loc[e1,e2]},
                ignore_index=True)
        return df


    def getDistanceMatrix(self,X, distance="pearson", longFormOutput=False):
        ""
        if hasattr(self,distance):
            distM = getattr(self, distance)(X)
            if longFormOutput:
                return self._transformToLong(X.index,distM,distance)
            else:
                return distM
                
        raise ValueError("Distance unknown.")



if __name__ == "__main__":
    X = np.array([
        [1,4],
        [3,4],
        [4,3]
    ])



    print(DistanceCalculator(X).apex(X))






