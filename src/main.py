


import os 
import pandas as pd
import numpy as np 
from collections import OrderedDict
from modules.Distance import DistanceCalculator
from modules.signal import Signal
from modules.Database import Database

class ComplexFinder(object):

    def __init__(self,
                indexIsID = True,
                idColumn = "Uniprot ID",
                databaseName="CORUM",
                peakModel = "Lorentzian",
                imputeNaN = True,
                metrices = ["euclidean","pearson"]):
        ""

        self.params = {
            "indexIsID" : indexIsID,
            "idColumn" : idColumn,
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
                self.X = self.X.astype(np.float)

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
            print(signal.values)
            s = Signal(signal.values,ID=entryID,peakModel=peakModel)
            s.fitModel()
            s.modeledPeaks


    def _calculateDistance(self):
        ""
        self.distM = dict()
        for metric in self.params['metrices']:
            if metric not in self.distM:

                self.distM[metric] = DistanceCalculator().getDistanceMatrix(
                                        X = self.X,
                                        distance=metric,
                                        longFormOutput=True)

        
    def _loadReferenceDB(self):
        ""
       # completeDf = pd.DataFrame()
        self.DB = Database()
        self.DB.pariwiseProteinInteractions("subunits(UniProt IDs)")

        

    def run(self,X):
        ""
        self._loadReferenceDB()
        self._load(X)
        self._calculateDistance()
        print(self.distM)
        self._findPeaks()


if __name__ == "__main__":
    X = pd.DataFrame(np.array([
        [0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.60,0.5,0.3,0.9,0.3,0.2,0.05,0,0,0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.8,0.9,0.3,0.2,0.05,0],
        [0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.60,0.5,0.3,0.9,0.3,0.2,0.05,0,0,0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.8,0.9,0.3,0.2,0.05,0]
    ]))

    X = pd.read_csv("../example-data/HeuselEtAlAebersoldLab.txt", sep="\t", nrows=50)

 #X = X.set_index("Uniprot ID")
  #  X

    ComplexFinder(indexIsID=False).run(X)





    