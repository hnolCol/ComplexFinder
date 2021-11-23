import os 
import shutil 
import numpy as np
from .Distance import DistanceCalculator
import pickle 
from numba import jit
import pandas as pd
#@jit()
def extractMetricByShiftBounds(NPeakModels,peakBounds,quantData,shift,nFractions):
    out = np.zeros(shape=(NPeakModels,shift))
    idxFull = np.arange(0,quantData.shape[1])
    for shiftN in range(shift):
        for n in range(NPeakModels):
           
            lowerB = int((peakBounds[n,0]-1) * shift) + shiftN
            upperB = int((peakBounds[n,1]-1) * shift) + shiftN
        
            if upperB == lowerB:
                out[n,shiftN] = quantData[n,lowerB]
                #out[n,1] = np.nan
            else:
                
                idx = idxFull[lowerB:upperB:shift]
            
                X = np.empty(shape=idx.size)
                for ii in range(idx.size):
                    X[ii] = quantData[n,idx[ii]]
            
                #X = quantData[n,idx]
                out[n,shiftN] = np.nansum(X)
                #out[n,1] = np.nanstd(X)
            #  print(B)
    return out
#@jit()
def extractMetricByShiftBounds2(NPeakModels,peakBounds,quantData,shift,nFractions):
    out = np.zeros(shape=(NPeakModels,2))
    idxFull = np.arange(0,quantData.shape[1])
    
    for n in range(NPeakModels):
        lowerB = peakBounds[n,0]
        upperB = peakBounds[n,1]
       
        if upperB == lowerB:
            out[n,0] = quantData[n,lowerB]
            out[n,1] = np.nan
        else:
            
            
           # print(idxFull)
            upperIdx = int(upperB+((shift-1)*nFractions))
           # print(upperIdx)
           # print(lowerB,upperIdx,shift)
            #print(idxFull[lowerB:upperIdx:shift])
           # print(A)
            idx = idxFull[lowerB:upperIdx:shift]
          
            X = np.empty(shape=idx.size)
            for ii in range(idx.size):
                X[ii] = quantData[n,idx[ii]]
           
            #X = quantData[n,idx]
            out[n,0] = np.nanmean(X)
            out[n,1] = np.nanstd(X)
            if n == 1:
                print(X)
                print(idx)
                print(b)
          #  print(B)
    return out

@jit()
def extractMeanByBounds(NPeakModels,peakBounds,quantData):
    "Calculate data from given bounds"
    out = np.zeros(shape=(NPeakModels,2))
    for n in range(NPeakModels):
        lowerB = peakBounds[n,0] 
        upperB = peakBounds[n,1]
        if upperB == lowerB:
            out[n,0] = quantData[n,lowerB]
            out[n,1] = np.nan
            
        else:
            X = quantData[n,lowerB:upperB]
            out[n,0] = np.nanmean(X)
            out[n,1] = np.nanstd(X)
    return out 

def calculateDistanceP(pathToFile):
    """
    Calculates the distance metrices per chunk
    """
    with open(pathToFile,"rb") as f:
        chunkItems = pickle.load(f)
    exampleItem = chunkItems[0] #used to get specfici chunk name to save under same name
    if "chunkName" in exampleItem:
        XX = [DistanceCalculator(**c).calculateMetrices() for c in chunkItems]
        data = np.concatenate([X[0] for X in XX],axis=0)
        np.save(os.path.join(exampleItem["pathToTmp"],"chunks",exampleItem["chunkName"]),data)  
        if not XX[0][1].empty:
            pd.concat([X[1] for X in XX],ignore_index=True).to_csv(os.path.join(exampleItem["pathToTmp"],"ApexDetails_{}.txt".format(exampleItem["chunkName"])),sep="\t",index=None)
        return (exampleItem["chunkName"],[''.join(sorted(row.tolist())) for row in data[:,[0,1]]])
        

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def cleanPath(pathToFolder):
    for root, dirs, files in os.walk(pathToFolder):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))



def minMaxNorm(X,axis=0):
        ""
        #transformedColumnNames = ["0-1({}):{}".format("row" if axis else "column",col) for col in columnNames.values]
        Xmin = np.nanmin(X,axis=axis, keepdims=True)
        Xmax = np.nanmax(X,axis=axis,keepdims=True)
        X_transformed = (X - Xmin) / (Xmax-Xmin)
        return X_transformed
