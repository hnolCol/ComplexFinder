
import numpy as np
import pandas as pd
import itertools
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import os
import gc 

class DistanceCalculator(object):

    def __init__(self, Y, E2, ID, otherSignalPeaks, ownPeaks, metrices = ["apex","euclidean","pearson","p_pearson"] ,pathToTmp = '', chunkName = ''):
        ""
        self.Y = Y
      #  self.Ys = Ys
        self.ID = ID
        self.E2s = E2
        self.ownPeaks = ownPeaks
        self.metrices = metrices
        self.otherSignalPeaks = otherSignalPeaks
        self.pathToTmp = pathToTmp

        Ys = np.load(os.path.join(pathToTmp,"source.npy"),allow_pickle=True)
        boolIdx = np.isin(Ys[:,0],E2)
        Ys = Ys[boolIdx]
        self.Ys = Ys[:,[n for n in range(Ys.shape[1]) if n != 0]]

    def _apex(self,p1,p2):
        ""

        return np.sqrt( (p1['mu'] - p2['mu']) ** 2  + (p1['sigma'] - p2['sigma']) ** 2 )


    def p_pears(self,u,v):
        "returns p value for pearson correlation"
        r, p = pearsonr(u,v)

        return 1-r, p

    def euclideanDistance(self):
        ""
        return [np.linalg.norm(self.Y - Y) for Y in self.Ys]

    def pearson(self):
        ""
        return [self.p_pears(self.Y,Y) for Y in self.Ys]
       
    def apex(self,otherSignalPeaks):
        "Calculates Apex Distance"
        apexDist = []     
        for otherPeaks in otherSignalPeaks:

            apexDist.append([self._apex(p1,p2) for p1 in self.ownPeaks for p2 in otherPeaks])
        minArgs = [np.argmin(x) for x in apexDist]
        return [np.min(x) for x in apexDist]

    def calculateMetrices(self):

        collectedDf = pd.DataFrame()

        collectedDf["E1"] = [self.ID] * len(self.E2s)
        collectedDf["E2"] = self.E2s

        collectedDf["E1E2"] = [''.join(sorted([self.ID,E2])) for E2 in self.E2s]
        
        for metric in self.metrices:

            if metric == "pearson":
                collectedDf["pearson"], collectedDf["p_pearson"] = zip(*self.pearson())

            elif metric == "euclidean":

                collectedDf["euclidean"] = self.euclideanDistance()
                    
            elif metric == "apex":
            
                collectedDf["apex"] = self.apex(self.otherSignalPeaks)

            elif metric == "max_location":

                maxOwnY = np.argmax(self.Y)
                collectedDf["max_location"] = [np.argmax(Y)-maxOwnY for Y in self.Ys]

        gc.collect()
        return collectedDf.values
            #collectedDf.to_csv(pathToFile, index=False)


if __name__ == "__main__":
    X = np.array([
        [1,4],
        [3,4],
        [4,3]
    ])



    print(DistanceCalculator(X).apex(X))






