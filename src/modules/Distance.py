
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import itertools

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from numpy.lib.stride_tricks import sliding_window_view

from joblib import Parallel, delayed
from numba import jit
import os
import gc 


def minMaxNorm(X,axis=0):
    "Normalize array betweem 0 and 1"
    Xmin = np.nanmin(X,axis=axis, keepdims=True)
    Xmax = np.nanmax(X,axis=axis,keepdims=True)
    X_transformed = (X - Xmin) / (Xmax-Xmin)
    return X_transformed

@jit(nopython=True, fastmath=True)
def init_w(w, n):
    """
    :purpose:
    Initialize a weight array consistent of 1s if none is given
    This is called at the start of each function containing a w param
    :params:
    w      : a weight vector, if one was given to the initial function, else None
             NOTE: w MUST be an array of np.float64. so, even if you want a boolean w,
             convert it to np.float64 (using w.astype(np.float64)) before passing it to
             any function
    n      : the desired length of the vector of 1s (often set to len(u))
    :returns:
    w      : an array of 1s with shape (n,) if w is None, else return w un-changed
    """
    if w is None:
        return np.ones(n)
    else:
        return w

@jit(fastmath=True)
def _apexDistance(mu1,mu2,s1,s2):
    
    return np.sqrt( (mu1 - mu2) ** 2  + (s1 - s2) ** 2 )


@jit()
def _apexScore(ownPeaks,otherSignalPeaks):

    apex = np.inf
    apexOut = np.zeros(shape = (ownPeaks.shape[0] *  otherSignalPeaks.shape[0],3)) #array of id1, id2 and apex
    ii = 0
    for n in range(ownPeaks.shape[0]):
        for m in range(otherSignalPeaks.shape[0]):
            mu1, s1 = ownPeaks[n,0:2]
            mu2, s2 = otherSignalPeaks[m,0:2]
           
            a = _apexDistance(mu1,mu2,s1,s2)
            apexOut[ii,:] = [n,m,a]
            
            if a < apex:
                apex = a
                id1 = n
                id2 = m
            ii += 1

    return apex, id1, id2, apexOut

@jit()
def signalDifference(nY,Ys):
    "Calculates the absolute difference between two arrays."
    r = np.empty(shape=Ys.shape)
    Y1 = nY.reshape(1,Ys.shape[1])
    for n in range(Ys.shape[0]):
        for m in range(Ys.shape[1]):
            if Y1[0,m] > 0 and Ys[n,m] > 0:
                r[m,n] = abs(Y1[0,m] - Ys[n,m])
            else:
                r[m,n] = 1
    return r 
    

@jit()
def umapDistance(xa,ya,otherSignalEmbeddings):
    return [np.sqrt((xa - xb)**2 + (ya-yb)**2) for xb,yb in otherSignalEmbeddings] 

@jit()
def euclideanDistance(nY,Ys):
    ""
    return [np.linalg.norm(nY - Y) for Y in Ys]

@jit(nopython=True, parallel=False, error_model='numpy')
def _pearson(u,v):
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
    return 1 - np.corrcoef(u, v)[0,1]



@jit()
def pearson(nY,Ys):
    "Calcualtes pearson correlation."
    return [_pearson(nY,Y) for Y in Ys]


@jit(nopython=True, fastmath=True)
def _cosine(u, v, w=None):
    """
    Copied from the fast.dist package! 
    https://github.com/talboger/fastdist/blob/master/fastdist/fastdist.py
    Not modified.

    :purpose:
    Computes the cosine similarity between two 1D arrays
    Unlike scipy's cosine distance, this returns similarity, which is 1 - distance
    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    cosine  : float, the cosine similarity between u and v
    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.cosine(u, v, w)
    0.7495065944399267
    """

    n = len(u)
    w = init_w(w, n)
    num = 0
    u_norm, v_norm = 0, 0
    for i in range(n):
        num += u[i] * v[i] * w[i]
        u_norm += abs(u[i]) ** 2 * w[i]
        v_norm += abs(v[i]) ** 2 * w[i]

    denom = (u_norm * v_norm) ** (1 / 2)
    return 1 - num / denom

@jit()
def cosineDistance(nY,Ys):
    ""
    return [_cosine(nY,Y) for Y in Ys]

@jit(fastmath=True)
def slidingPearson(slidingWindow,fixIdx = 0):
    
    results = np.empty(shape=(slidingWindow.shape[0],1))
    #Yfixed = slidingWindow

    for n in range(slidingWindow.shape[0]):
        if n != fixIdx:
            r = 1
            for nw in range(slidingWindow.shape[1]):
                Y1 = slidingWindow[fixIdx][nw]
                Y2 = slidingWindow[n][nw]
                if np.sum(Y1) > 0 and np.sum(Y2) > 0 and np.count_nonzero(Y1) >= 4 and np.count_nonzero(Y2) >= 4:
                    rw = _pearson(Y1,Y2)
                    if rw < r:
                        r = rw
                if r < 1e-5: #early stop 
                    results[n] = r
                    continue
                results[n] = r
    return results

    

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
                    Ys = None,
                    correlationWindowSize = 4,
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

        self.Y = Y.astype(np.float32)
        self.ID = ID
        self.E2s = E2
        self.ownPeaks = ownPeaks
        self.metrices = metrices
        self.otherSignalPeaks = otherSignalPeaks
        self.pathToTmp = pathToTmp
        self.embedding = embedding
        if len(self.embedding) > 0:#o_csv(os.path.join(self.params["pathToTmp"][analysisName],"chunks","embeddings.txt"),sep="\t")
            pathToE = os.path.join(pathToTmp,"chunks","embeddings.txt")
            if os.path.exists(pathToE):
                e = pd.read_csv(pathToE,sep="\t", index_col=0)
                self.otherSignalEmbeddings = e.loc[self.E2s,:].values.astype(np.float32)
       # self.otherSignalEmbeddings = otherSignalEmbeddings
        self.correlationWindowSize = correlationWindowSize
        Ys = np.load(os.path.join(pathToTmp,"source.npy"),allow_pickle=True)
        boolIdx = np.isin(Ys[:,0],E2)
        Ys = Ys[boolIdx]
        self.Ys = Ys[:,[n for n in range(Ys.shape[1]) if n != 0]]
        self.Ys = self.Ys.astype(np.float32)

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
        if "gamma" in p1 and "gamma" in p2:
            return np.sqrt( (p1['mu'] - p2['mu']) ** 2  + (p1['sigma'] - p2['sigma']) ** 2 + (p1['gamma'] - p2['gamma']) ** 2)
        else:
            return np.sqrt( (p1['mu'] - p2['mu']) ** 2  + (p1['sigma'] - p2['sigma']) ** 2 )


    def euclideanDistance(self):
        ""
        
        return euclideanDistance(self.Y,self.Ys)


    def pearson(self):
        ""
        return pearson(self.Y,self.Ys)
    

    def apex(self,otherSignalPeaks):
        "Calculates Apex Distance"
        out = []
        
        E2s = []
        E1 = []
        XX = pd.DataFrame(columns=["E1","E2","id1","id2","apex"])
        r = []
        
        for n,otherPeaks in enumerate(otherSignalPeaks):

            apex, ownIdx, otherIdx, apexScores = _apexScore(self.ownPeaks["peaks"],otherPeaks["peaks"])
            out.append((apex,"{}_{}".format(ownIdx,otherIdx)))
            
            r.append(apexScores)
            E2s.extend([self.E2s[n]]*apexScores.shape[0])
            E1.extend([self.ID]*apexScores.shape[0])

        rr = np.concatenate(r)

        XX.loc[:,"E1"] = E1 
        XX.loc[:,"E2"] = E2s 
        XX.loc[:,["id1","id2","apex"]] = rr
 
        return out, XX
        
    def cosine(self):
        """
        Calculates 1-cosine distance
        
        Returns 
        ---------
        List of 1-cosine distances
        """
        cosineDist =  cosineDistance(self.Y,self.Ys)
        
        return cosineDist

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

    def rollingCorrelation(self):
        ""
        #to do, takes the longest by far
        #use numba to calculate this
        # print(Ys)
        # print(Ys.shape)
        #Ys = np.concatenate([self.Y.reshape(1,self.Ys.shape[1]),self.Ys],axis=0)
        
        slides =  sliding_window_view(self.Ys,self.correlationWindowSize,axis=1)
        return  slidingPearson(slides)
        # YSignal = pd.Series(self.Y).replace(0,np.nan)
        # Ys = pd.DataFrame(self.Ys).replace(0,np.nan)
        # rollingPearson =  pd.Series([1-YSignal.rolling(self.correlationWindowSize,min_periods=5,center=True).corr(Ys.iloc[idx]).replace([np.inf, -np.inf, np.nan], 0).max() for idx in Ys.index])
        
        # return rollingPearson
    
    def calculateMetrices(self):
        """
        Calculates metrices between the signal Y and other signals Ys.

        E1 and E2 denote entries. To enable parallel calculations
        the distance calculator is entry-centric (E1) and calculates
        the distance metrices versus all other signals (E2 and Ys)

        TO IMPROVE
        -iteration through all signals for every metric -> should do this only once.
        -run more metrics calculations using numba jit

        Parameters
        ----------
        

        Returns
        -------
        two dimensional numpy.array

        """
        collectedDf = pd.DataFrame()
        detailedApexResults = pd.DataFrame() 
        collectedDf["E1"] = [self.ID] * len(self.E2s)
        collectedDf["E2"] = self.E2s

        collectedDf["E1E2"] = [''.join(sorted([self.ID,E2])) for E2 in self.E2s]
        
        for metric in self.metrices:
            
            if isinstance(metric,dict) and callable(metric["fn"]):
                collectedDf[metric["name"]] = [metric["fn"](self.Y,Y) for Y in self.Ys]

            elif (metric == "pearson" or metric == "p_pearson") and "pearson" not in collectedDf.columns:
               # collectedDf["pearson"], collectedDf["p_pearson"] = zip(*self.pearson())
                collectedDf["pearson"] = self.pearson()

            elif metric == "spearman":

                collectedDf["spearman"] = self.spearman()

            elif metric == "euclidean":

                collectedDf["euclidean"] = self.euclideanDistance()
            
            elif metric == "rollingCorrelation":

                collectedDf["rollingCorrelation"] = self.rollingCorrelation()
                collectedDf["rollingCorrelation"] = collectedDf["rollingCorrelation"].replace([np.inf, -np.inf, np.nan], 2)

            elif metric == "cosine":

                collectedDf["cosine"] = self.cosine()    
            
            elif metric == "apex":
                idScore, detailedApexResults = self.apex(self.otherSignalPeaks)
                collectedDf["apex"], collectedDf["apex_peakId"] = zip(*idScore)

            elif metric == "max_location":

                maxOwnY = np.argmax(self.Y)
                collectedDf["max_location"] = [np.abs(np.argmax(Y)-maxOwnY) for Y in self.Ys]

            elif metric == "umap-dist" and len(self.embedding) == 2:
                xa, ya = self.embedding
                collectedDf["umap-dist"] = umapDistance(xa,ya,self.otherSignalEmbeddings)

            elif metric == "signalDiff":
                signalDiffColumnNames = ["{}-diff".format(x) for x in np.arange(self.Ys.shape[1])]
                Y = self.Y / np.max(self.Y)
                Ys = minMaxNorm(self.Ys,axis=1)
                collectedDf[signalDiffColumnNames] = np.subtract(Y,Ys)
                #collectedDf[signalDiffColumnNames] = collectedDf[signalDiffColumnNames].astype(np.float32)
                
        
        columnsResorted = [metricName if not isinstance(metricName,dict) else metricName["name"] for metricName in self.metrices if metricName != "signalDiff"]
        if "signalDiff" in self.metrices:
            columnsResorted.extend(signalDiffColumnNames)
        if "apex_peakId" in collectedDf.columns:
            
            firstCols = ["E1","E2","E1E2","apex_peakId"]
            
        else:
            firstCols = ["E1","E2","E1E2"] 

        collectedDf = collectedDf[firstCols + columnsResorted]

        return collectedDf.values, detailedApexResults







