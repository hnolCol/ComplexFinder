
import numpy as np
import pandas as pd
import itertools
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from joblib import Parallel, delayed

import gc 

class DistanceCalculator(object):

    def __init__(self):
        ""

    def p_pears(self,u,v):
        "returns p value for pearson correlation"
        _, p = pearsonr(u,v)

        return p

    def p_pearson(self,X):
        ""
        return squareform(pdist(X,self.p_pears))

    def pearson(self,X, rowvar=True):
        ""
        print("pearson is equal to 1-r therefore the range of r is [0,2]")
        return 1-np.corrcoef(X,rowvar=rowvar)

    def euclidean(self,X):
        ""
        print("Euclidean distance calculation")
        return squareform(pdist(X,"euclidean"))


    def apex(self,X):
        
        otherSignals = [(Signal.ID,Signal._collectPeakResults()) for Signal in X]
        distM = []#Parallel(n_jobs=n_jobs)(delayed(Signal.calculateApexDistance)(otherSignals) for Signal in X)
        signalIds = [x[0] for x in otherSignals]
        df = pd.DataFrame(columns=["E1;E2","metric"])
        for n,Signal in enumerate(X):
            
            df = df.append(Signal.calculateApexDistance(otherSignals,n), ignore_index = True)
            gc.collect()



        return df

        
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
        if hasattr(self,distance) and distance != "apex":
            distM = getattr(self, distance)(X)
            if longFormOutput:
                return self._transformToLong(X.index,distM,distance)
            else:
                return distM

        elif distance == "apex":

            if not isinstance(X,list):
                raise ValueError("X must be list for apex distance")
            print("Apex calculation started ..")
            distM = self.apex(X)

            return distM

        
                
        raise ValueError("Distance unknown.")



if __name__ == "__main__":
    X = np.array([
        [1,4],
        [3,4],
        [4,3]
    ])



    print(DistanceCalculator(X).apex(X))






