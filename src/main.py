


import os 
import pandas as pd
import numpy as np 
from collections import OrderedDict
from modules.Distance import DistanceCalculator
from modules.signal import Signal
from modules.Database import Database
from modules.Predictor import Classifier
from joblib import Parallel, delayed

class ComplexFinder(object):

    def __init__(self,
                indexIsID = True,
                maxPeaksPerSignal = 15,
                n_jobs = 4,
                idColumn = "Uniprot ID",
                databaseName="CORUM",
                peakModel = "Lorentzian",
                imputeNaN = True,
                metrices = ["euclidean","pearson"]):
        ""

        self.params = {
            "indexIsID" : indexIsID,
            "idColumn" : idColumn,
            "n_jobs" : n_jobs,
            "databaseName" : databaseName,
            "imputeNaN" : imputeNaN,
            "metrices" : metrices,
            "peakModel" : peakModel,
            "maxPeaksPerSignal" : maxPeaksPerSignal
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


    def _findPeaks(self, n_jobs=3):
        ""
        self.Signals = OrderedDict()
        peakModel = self.params['peakModel']
        for entryID, signal in self.X.iterrows():

            self.Signals[entryID] = Signal(signal.values,
                                            ID=entryID, 
                                            peakModel=peakModel, 
                                            maxPeaks=self.params["maxPeaksPerSignal"]) 

        print("starting parallel Signal modelling .. (n_jobs = {})".format(n_jobs))
        Parallel(n_jobs=n_jobs)(delayed(Signal.fitModel)() for Signal in self.Signals.values())
        
       # for entryID, signal in self.X.iterrows():
       #     print(entryID)
        #    print(signal.values)
       #     s = Signal(signal.values,ID=entryID,peakModel=peakModel)
        #    s.fitModel()
          #  s.modeledPeaks


    def _calculateDistance(self):
        ""
        self.distM = dict()
        for metric in self.params['metrices']:
            if metric not in self.distM:

                self.distM[metric] = pd.DataFrame(DistanceCalculator().getDistanceMatrix(
                                        X = self.X,
                                        distance=metric,
                                        longFormOutput=False), index = self.X.index, columns = self.X.index)
                

        
    def _loadReferenceDB(self):
        ""
       # completeDf = pd.DataFrame()
        self.DB = Database()
        self.DB.pariwiseProteinInteractions("subunits(UniProt IDs)")


    def _addMetricesToDB(self):

        for metric,df in self.distM.items():
            self.DB.matchInteractions(metric,df)
    


    def _trainPredictor(self):
        ""
        metrices = list(self.distM.keys())
        data = self.DB.df[metrices + ['Class','E1;E2']].dropna(subset=metrices)
        self.Y = data['Class'].values
        X = data.loc[:,metrices].values
        print(self.Y.size)
        print(X)
        print(self.Y)

        self.classifier = Classifier(n_jobs=self.params['n_jobs']).fit(X,self.Y)


        

    def run(self,X):
        ""
        
        self._load(X)
        self._calculateDistance()
        self._loadReferenceDB()
        self._addMetricesToDB()
        #self._trainPredictor()
        self._findPeaks(self.params["n_jobs"])


if __name__ == "__main__":
    X = pd.DataFrame(np.array([
        [0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.60,0.5,0.3,0.9,0.3,0.2,0.05,0,0,0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.8,0.9,0.3,0.2,0.05,0],
        [0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.60,0.5,0.3,0.9,0.3,0.2,0.05,0,0,0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.8,0.9,0.3,0.2,0.05,0]
    ]))

    X = pd.read_csv("../example-data/HeuselEtAlAebersoldLab.txt", sep="\t", nrows=1000)

 #X = X.set_index("Uniprot ID")
  #  X

    ComplexFinder(indexIsID=False).run(X)





    