


import os 
import pandas as pd
import numpy as np 
from collections import OrderedDict
from modules.Distance import DistanceCalculator
from modules.signal import Signal
from modules.Database import Database
from modules.Predictor import Classifier
from joblib import Parallel, delayed, dump, load
import gc 
import string
import random
import time
from multiprocessing import Pool

filePath = os.path.dirname(os.path.realpath(__file__)) 
pathToTmp = os.path.join(filePath,"tmp")

RF_GRID_SEARCH = {#'bootstrap': [True, False],
 'max_depth': [50, 60, 70,80,  None],
 'max_features': ['sqrt','auto'],
 'min_samples_leaf': [2, 3,5],
 'min_samples_split': [2, 3,5],
 'n_estimators': [300]}


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class SignalHelper(object):

    ""
    def __init__(self,id,signals,funcName):


        data = np.empty(shape=(1,7))
        for signal in signals:
            data = np.append(data,getattr(signal,funcName)(id))

        self.saveChunks(str(id),data)
    
    def saveChunks(self,fileName,data):

        print("savve")
        print(data)
        np.save(fileName,data)


class ComplexFinder(object):

    def __init__(self,
                indexIsID = True,
                maxPeaksPerSignal = 15,
                n_jobs = 4,
                analysisName = None,
                idColumn = "Uniprot ID",
                databaseName="CORUM",
                peakModel = "Lorentzian",
                imputeNaN = True,
                metrices = ["apex","euclidean","pearson","p_pearson","max_location"],
                classiferGridSearch = RF_GRID_SEARCH):
        ""

        self.params = {
            "indexIsID" : indexIsID,
            "idColumn" : idColumn,
            "n_jobs" : n_jobs,
            "analysisName" : analysisName,
            "databaseName" : databaseName,
            "imputeNaN" : imputeNaN,
            "metrices" : metrices,
            "peakModel" : peakModel,
            "maxPeaksPerSignal" : maxPeaksPerSignal,
            "classiferGridSearch" : classiferGridSearch
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


    def _addModelToSignals(self,signalModels):

        for fitModel in signalModels:
            modelID = fitModel["id"]
            if modelID in self.Signals:
                for k,v in fitModel.items():
                    if k != 'id':
                        setattr(self.Signals[modelID],k,v)
            

    def _findPeaks(self, n_jobs=3):
        ""
        pathToSignal = os.path.join(self.params["pathToTmp"],"signals.lzma")
        if os.path.exists(pathToSignal):
            self.Signals = load(pathToSignal)
            print("loading pickled signal intensity")
        else:
            self.Signals = OrderedDict()
            peakModel = self.params['peakModel']
            for entryID, signal in self.X.iterrows():

                self.Signals[entryID] = Signal(signal.values,
                                                ID=entryID, 
                                                peakModel=peakModel, 
                                                maxPeaks=self.params["maxPeaksPerSignal"],
                                                metrices=self.params["metrices"],
                                                pathToTmp = self.params["pathToTmp"]) 

            
            t1 = time.time()
            print("starting parallel Signal modelling .. (n_jobs = {})".format(n_jobs))
            fittedModels = Parallel(n_jobs=n_jobs)(delayed(Signal.fitModel)() for Signal in self.Signals.values())
            self._addModelToSignals(fittedModels)
            print("Peak fitting done time : {} minutes".format(round((time.time()-t1)/60)))
        
        dump(self.Signals,pathToSignal)


        
       # for entryID, signal in self.X.iterrows():
       #     print(entryID)
        #    print(signal.values)
       #     s = Signal(signal.values,ID=entryID,peakModel=peakModel)
        #    s.fitModel()
          #  s.modeledPeaks


    def _calculateDistance(self):
        ""
        X = list(self.Signals.values())
        print("Starting Distance Calculation ..")
        t1 = time.time()
       # t1 = time.time()
       # Parallel(n_jobs=6, prefer="processes")(delayed(Signal.calculateMetrices)(otherSignals = X[n:], 
         #                                                    metrices=self.params["metrices"],
        #                                                     pathToTmp = self.params["pathToTmp"]) for n,Signal in enumerate(self.Signals.values()))
        #print("Paralllel job",time.time()-t1)
        for n,Signal in enumerate(self.Signals.values()):
            setattr(Signal,"otherSignals", X[n:])
        nSignals = len(self.Signals)


      #  Parallel(n_jobs=self.params["n_jobs"], prefer="threads",verbose=20)(delayed(SignalHelper)(id=n,signals=chunk,funcName="calculateMetrices") for n,chunk in enumerate(chunks(list(self.Signals.values()),4)))
        for n, chunk in enumerate(chunks(list(self.Signals.values()),4)):
                SignalHelper(id=n,signals=chunk,funcName="calculateMetrices")
        #pool = Pool(5) # run 10 task at most in parallel
        # pool.map(compute_cluster, range(10))

        print("parallel computing: {} secs".format(round(time.time()-t1))
      #  t1 = time.time()
      #  nSignals = len(self.Signals)
      #  for n,Signal in enumerate(self.Signals.values()):
       #     Signal.calculateMetrices(otherSignals = X[n:], 
                                    )
        #    if n % int(nSignals*0.15) == 0:
         #       print("{} % done".format(round(n/nSignals*100,2)))
        
       # print("Time to calculate / load distances {} minutes".format(round(time.time()-t1)/60))


        
    def _loadReferenceDB(self):
        ""
       # completeDf = pd.DataFrame()
        print("load data base")
        self.DB = Database()
        self.DB.pariwiseProteinInteractions("subunits(UniProt IDs)")
        self.DB.filterDBByEntryList(self.X.index)
        self.DB.addDecoy()


    def _addMetricesToDB(self):
        self.DB.matchMetrices(self.params["pathToTmp"])


    def _trainPredictor(self):
        ""
        metricColumns = [col for col in self.DB.df.columns if any(x in col for x in self.params["metrices"])]
        totalColumns = metricColumns + ['Class']
        data = self.DB.df[totalColumns].dropna(subset=metricColumns)
        self.Y = data['Class'].values
        X = data.loc[:,metricColumns].values
        print(self.Y.size)
        print(X)
        print("YYY")
        print(self.Y)

        self.classifier = Classifier(
            n_jobs=self.params['n_jobs'], 
            gridSearch = self.params["classiferGridSearch"]).fit(X,self.Y)


    def _randomStr(self,n):

        letters = string.ascii_lowercase + string.ascii_uppercase
        return "".join(random.choice(letters) for i in range(n))
        

    def _makeTmpFolder(self):

        if self.params["analysisName"] is None:

            analysisName = self._randomStr(50)

        else:

            analysisName = str(self.params["analysisName"])

        pathToTmpFolder = os.path.join(pathToTmp,analysisName)

        if os.path.exists(pathToTmpFolder):
            print("Path to tmp folder exsists")
            print("Will take files from there, if they exist")
            return pathToTmpFolder
        else:
            try:
                os.mkdir(pathToTmpFolder)
                print("Tmp folder created -- ",analysisName)
                return pathToTmpFolder
            except OSError:
                print("Could not create tmp folder")

    def run(self,X):
        ""  

        pathToTmpFolder = self._makeTmpFolder()
        self.params["pathToTmp"] = pathToTmpFolder

        if pathToTmpFolder is not None:
        
            self._load(X)
            self._findPeaks(self.params["n_jobs"])
            self._calculateDistance()
            self._loadReferenceDB()
            self._addMetricesToDB()
            self._trainPredictor()
        


if __name__ == "__main__":
    X = pd.DataFrame(np.array([
        [0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.60,0.5,0.3,0.9,0.3,0.2,0.05,0,0,0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.8,0.9,0.3,0.2,0.05,0],
        [0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.60,0.5,0.3,0.9,0.3,0.2,0.05,0,0,0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.8,0.9,0.3,0.2,0.05,0]
    ]))

    X = pd.read_csv("../example-data/HeuselEtAlAebersoldLab.txt", 
                    sep="\t", nrows=50)

 #X = X.set_index("Uniprot ID")
  #  X

    ComplexFinder(indexIsID=False).run(X)





    
