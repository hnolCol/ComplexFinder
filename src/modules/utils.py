import os 
import shutil 
import numpy as np
from .Database import DistanceCalculator

def calculateDistanceP(pathToFile):
    """
    Calculates the distance metrices per chunk
    """
    with open(pathToFile,"rb") as f:
        chunkItems = pickle.load(f)
    exampleItem = chunkItems[0] #used to get specfici chunk name to save under same name
    if "chunkName" in exampleItem:
        data = np.concatenate([DistanceCalculator(**c).calculateMetrices() for c in chunkItems],axis=0)
        np.save(os.path.join(exampleItem["pathToTmp"],"chunks",exampleItem["chunkName"]),data)  
      
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
