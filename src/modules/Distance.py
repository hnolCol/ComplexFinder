
import numpy as np
import pandas as pd
import itertools
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed
import os
import gc 

class DistanceCalculator(object):

    def __init__(self, 
                    Y, 
                    E2, 
                    ID, 
                    otherSignalPeaks, 
                    ownPeaks, 
                    metrices = ["apex","euclidean","pearson","p_pearson"] ,
                    pathToTmp = '', 
                    chunkName = '', 
                    embedding = [], 
                    otherSignalEmbeddings = []):
        """Signal-centric Distance Calculator.

        Calculates given distance metrices of signals.

        Note
        ----
        
        Parameters
        ----------
        Y : numpy array
            Signal profile of E1

        ID : string
            Identifier of E1
        
        E2 : obj:`list`of obj `np.array`
            Signal intensity of E2s. Disntances
            betwenn ID and E2 are calculated. 
            The intensitiy profiles of E2s are uploaded from source.npy.

        ownPeaks : obj:`list`of obj `dict`
            List of modelled peaks for Y. Required to calculate apex distance, 
            which is equal to the euclidean dinstance of the closest peaks. 
        
        metrices : obj:`list` of obj:`str` or obj`list` of obj`dict`
            List of strings or dictionories of metrices used to calculate distance. 
            If dict is provided, two keys namely `fn`and `name`must be provided. 
            The name must be unique (if more than one dict is provided.)

        pathToTmp : string
            Path to the temporary folder for the current anaylsis. Required to load
            Signals (called Ys)

        chunkName : string
            Name of the current chunk. 
       
        """

        self.Y = Y
        self.ID = ID
        self.E2s = E2
        self.ownPeaks = ownPeaks
        self.metrices = metrices
        self.otherSignalPeaks = otherSignalPeaks
        #print(self.otherSignalPeaks)
        self.pathToTmp = pathToTmp
        self.embedding = embedding
        self.otherSignalEmbeddings = otherSignalEmbeddings

        Ys = np.load(os.path.join(pathToTmp,"source.npy"),allow_pickle=True)
        boolIdx = np.isin(Ys[:,0],E2)
        Ys = Ys[boolIdx]
        self.Ys = Ys[:,[n for n in range(Ys.shape[1]) if n != 0]]

    def _apex(self,p1,p2):
        """
        Apex between two peaks. Basically euclidean distance between two peaks.

        Parameters
        ----------
        p1 : dict
            Peak parameters, must contain keywords: mu (center) and sigma

        p2 : dict
            Peak parameters, must contain keywords: mu (center) and sigma
        

        Returns
        -------
        Apex score between two peaks.

        """

        return np.sqrt( (p1['mu'] - p2['mu']) ** 2  + (p1['sigma'] - p2['sigma']) ** 2 )


    def _pearson(self,u,v):
        """
        Calculates pearson correlation between two arrays.

        Parameters
        ----------
        u : numpy array, array-like
            

        v : numpy array
            

        Returns
        -------
        Tuple of 1- pearson correlation and the p value 

        """
        r, p = pearsonr(u,v)

        return 1-r, p

    def euclideanDistance(self):
        ""
        return [np.linalg.norm(self.Y - Y) for Y in self.Ys]

    def pearson(self):
        ""
        return [self._pearson(self.Y,Y) for Y in self.Ys]
       
    def apex(self,otherSignalPeaks):
        "Calculates Apex Distance"
        apexDist = []    
        apexMinArg = [] 
        for otherPeaks in otherSignalPeaks:
            
            apexDistCalc, minPeaks = map(list,zip(*[(self._apex(p1,p2),("{}_{}".format(p1["ID"],p2["ID"]))) for p1 in self.ownPeaks for p2 in otherPeaks]))
            apexDist.append(apexDistCalc)
            apexMinArg.append(minPeaks)

        return [(np.min(x),apexMinArg[n][np.argmin(x)]) for n,x in enumerate(apexDist)]

    def spearman(self):
        """
        Calculates 1 - spearman correlation for Y versus other signals.

        Parameters
        ----------
        

        Returns
        -------
        List of 1 - rho for signals.

        """
        return [1-spearmanr(Y,self.Y)[0] for Y in self.Ys]

    def calculateMetrices(self):
        """
        Calculates metrices between the signal Y and other signals Ys.

        E1 and E2 denote entries. To enable parallel calculations
        the distance calculator is entry-centric (E1) and calculates
        the distance metrices versus all other signals (E2 and Ys)

        Parameters
        ----------
        

        Returns
        -------
        two dimensional numpy.array

        """
        collectedDf = pd.DataFrame()

        collectedDf["E1"] = [self.ID] * len(self.E2s)
        collectedDf["E2"] = self.E2s

        collectedDf["E1E2"] = [''.join(sorted([self.ID,E2])) for E2 in self.E2s]
        
        for metric in self.metrices:
            
            if isinstance(metric,dict) and callable(metric["fn"]):
                collectedDf[metric["name"]] = [metric["fn"](self.Y,Y) for Y in self.Ys]

            elif (metric == "pearson" or metric == "p_pearson") and "pearson" not in collectedDf.columns:
                collectedDf["pearson"], collectedDf["p_pearson"] = zip(*self.pearson())

            elif metric == "spearman":

                collectedDf["spearman"] = self.spearman()

            elif metric == "euclidean":

                collectedDf["euclidean"] = self.euclideanDistance()
                    
            elif metric == "apex":
            
                collectedDf["apex"], collectedDf["apex_peakId"] = zip(*self.apex(self.otherSignalPeaks))

            elif metric == "max_location":

                maxOwnY = np.argmax(self.Y)
                collectedDf["max_location"] = [np.abs(np.argmax(Y)-maxOwnY) for Y in self.Ys]

            elif metric == "umap-dist" and len(self.embedding) == 2:
                xa, ya = self.embedding
                collectedDf["umap-dist"] = [np.sqrt((xa - xb)**2 + (ya-yb)**2) for xb,yb in self.otherSignalEmbeddings] 
                
        if "apex_peakId" in collectedDf.columns:
            
            firstCols = ["E1","E2","E1E2","apex_peakId"]
            columnsResorted = self.metrices
        else:
            firstCols = ["E1","E2","E1E2"] 
            columnsResorted = self.metrices

        collectedDf = collectedDf[firstCols + columnsResorted]

        return collectedDf.values







