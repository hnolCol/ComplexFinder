


import os 
import pandas as pd
import numpy as np 
from collections import OrderedDict

from modules.signal import Signal

class ComplexFinder(object):

    def __init__(self,
                indexIsID = True,
                idColumn = "Uniprot ID",
                databaseName="CORUM",
                peakModel = "Lorentzian",
                imputeNaN = True,
                metrices = ["apex","euclidean","pearson"]):
        ""

        self.params = {
            "indexIsID" : indexIsID,
            "idColumns" : idColumn,
            "databaseName" : databaseName,
            "imputeNaN" : imputeNaN,
            "metrices" : metrices,
            "peakModel" : peakModel
            }
    

    def _load(self, X):
        "Load data"
        
        if isinstance(X, pd.DataFrame):
            
            self.X = X

            if not self.params["indexIsID"]:

                self.X = self.X.set_index(self.params["idColumn"])

        else:

            raise ValueError("X must be a pandas data frame")


    def _clean(self,X):
        ""


    def _findPeaks(self):
        ""
        self.Signals = OrderedDict()
        peakModel = self.params['peakModel']
        for entryID, signal in self.X.iterrows():
            print(entryID)
            s = Signal(signal.values,peakModel)
            s.fitModel()
            s.modeledPeaks


    def run(self,X):
        ""

        self._load(X)
        self._findPeaks()


if __name__ == "__main__":
    X = pd.DataFrame(np.array([
        [0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.60,0.5,0.3,0.9,0.3,0.2,0.05,0,0,0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.8,0.9,0.3,0.2,0.05,0],
        [0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.60,0.5,0.3,0.9,0.3,0.2,0.05,0,0,0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.8,0.9,0.3,0.2,0.05,0]
    ]))
    ComplexFinder().run(X)





    