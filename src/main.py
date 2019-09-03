


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
import pickle
from multiprocessing import Pool
from joblib import wrap_non_picklable_objects

filePath = os.path.dirname(os.path.realpath(__file__)) 
pathToTmp = os.path.join(filePath,"tmp")

RF_GRID_SEARCH = {#'bootstrap': [True, False],
 'max_depth': [70,80,120, None],
 'max_features': ['sqrt','auto'],
 'min_samples_leaf': [2, 3,4],
 'min_samples_split': [2, 3,5],
 'n_estimators': [300]}


entriesInChunks = dict() 


def _calculateDistanceP(pathToFile):
    
    with open(pathToFile,"rb") as f:
        chunkItems = pickle.load(f)
    exampleItem = chunkItems[0]
    data = np.concatenate([DistanceCalculator(**c).calculateMetrices() for c in chunkItems],axis=0)
    np.save(os.path.join(exampleItem["pathToTmp"],exampleItem["chunkName"]),data)        
    return (exampleItem["chunkName"],[''.join(sorted(row.tolist())) for row in data[:,[0,1]]])
        

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]



class ComplexFinder(object):

    def __init__(self,
                indexIsID = True,
                maxPeaksPerSignal = 5,
                n_jobs = 4,
                analysisName = None,
                idColumn = "Uniprot ID",
                databaseName="CORUM",
                peakModel = "LorentzianModel",
                imputeNaN = True,
                interactionProbabCutoff = 0.85,
                metrices = ["apex","euclidean","pearson","p_pearson"],
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
            "interactionProbabCutoff":interactionProbabCutoff,
            "maxPeaksPerSignal" : maxPeaksPerSignal,
            "classiferGridSearch" : classiferGridSearch
            }
    

    def _load(self, X):
        "Load data"
        
        if isinstance(X, pd.DataFrame):
            
            self.X = X

            if not self.params["indexIsID"]:
                
                np.save(os.path.join(self.params["pathToTmp"],"source"),self.X.values)
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
            print("\nLoading pickled signal intensity")
        else:
            self.Signals = OrderedDict()
            peakModel = self.params['peakModel']
            for entryID, signal in self.X.iterrows():

                self.Signals[entryID] = Signal(signal.values,
                                                ID=entryID, 
                                                peakModel=peakModel, 
                                                savePlots = False,
                                                maxPeaks=self.params["maxPeaksPerSignal"],
                                                metrices=self.params["metrices"],
                                                pathToTmp = self.params["pathToTmp"]) 

            
            t1 = time.time()
            print("\n\nStarting Signal modelling .. (n_jobs = {})".format(n_jobs))
            fittedModels = Parallel(n_jobs=n_jobs, verbose=1)(delayed(Signal.fitModel)() for Signal in self.Signals.values())
            self._addModelToSignals(fittedModels)
            
            print("Peak fitting done time : {} secs".format(round((time.time()-t1))))
        
        dump(self.Signals,pathToSignal)



    def _calculateDistance(self):
        ""
        X = list(self.Signals.values())
        print("\n\nStarting Distance Calculation ...")
        t1 = time.time()

        for n,Signal in enumerate(self.Signals.values()):
            setattr(Signal,"otherSignals", X[n:])

        chunks = self._createSignalChunks()

        chunkItems = Parallel(n_jobs=self.params["n_jobs"], verbose=10)(delayed(_calculateDistanceP)(c) for c in chunks)
        

        for k,v in chunkItems:
            entriesInChunks[k] = v 

        print("Distance computing: {} secs\n".format(round(time.time()-t1)))

    def _createSignalChunks(self):

        nSignals = len(self.Signals)
        signals = list(self.Signals.values())
        chunkSize = 30#int(float(nSignals*0.10))

        c = []

        for n,chunk in enumerate(chunks(signals,chunkSize)):
            chunkItems = []
            for signal in chunk:
                chunkItems.append({"ID":str(signal.ID),
                                    "chunkName":str(n),
                                    "Y":np.array(signal.Y),
                                    "ownPeaks" : signal._collectPeakResults(),
                                    "otherSignalPeaks" : [s._collectPeakResults() for s in signal.otherSignals],
                                    "E2":[str(s.ID) for s in signal.otherSignals],
                                    #"Ys":[np.array(s.Y) for s in signal.otherSignals],
                                    "pathToTmp":self.params["pathToTmp"]})

            with open(os.path.join(self.params["pathToTmp"], str(n)+".pkl"),"wb") as f:
                pickle.dump(chunkItems,f)
            gc.collect()
            c.append(os.path.join(self.params["pathToTmp"], str(n)+".pkl"))
        del chunkItems
        return c


#class SignalHelper(object):

 #   ""
  #  def __init__(self,**kwargs):
   #     
  #      data = np.array([])
    #    for signal in kwargs["signals"]:
     #       data = np.append(data,getattr(signal,kwargs["funcName"])(kwargs["id"]))
#
 #       self.saveChunks(str(kwargs["id"]),data,kwargs["pathToTmp"])
    
   # def saveChunks(self,fileName,data,pathToTmp):

    #    data = data.reshape(-1,7)
     #   entriesInChunks[fileName] = np.unique(data[:,[0,1]])
#
 #       np.save(os.path.join(pathToTmp,fileName),data)




    def _loadReferenceDB(self):
        ""

        print("Load positive data base")
        self.DB = Database()
        self.DB.pariwiseProteinInteractions("subunits(UniProt IDs)")
        self.DB.filterDBByEntryList(self.X.index)
        self.DB.addDecoy()


    def _addMetricesToDB(self):
        self.DB.matchMetrices(self.params["pathToTmp"],entriesInChunks,self.params["metrices"])


    def _trainPredictor(self):
        ""
        metricColumns = [col for col in self.DB.df.columns if any(x in col for x in self.params["metrices"])]
        totalColumns = metricColumns + ['Class']
        data = self.DB.df[totalColumns].dropna(subset=metricColumns)
        self.Y = data['Class'].values
        X = data.loc[:,metricColumns].values
        print(X)

        self.classifier = Classifier(
            n_jobs=self.params['n_jobs'], 
            gridSearch = self.params["classiferGridSearch"])

        self.classifier.fit(X,self.Y)

    def _loadPairs(self):

        chunks = [f for f in os.listdir(self.params["pathToTmp"]) if f.endswith(".npy") and f != "source.npy"]
        pathToPredict = os.path.join(self.params["pathToTmp"],"predictions")
        if not os.path.exists(pathToPredict):
            os.mkdir(pathToPredict)
        print("\nPrediction started...")
        for chunk in chunks:

            X = np.load(os.path.join(self.params["pathToTmp"],chunk),allow_pickle=True).reshape(-1,3+len(self.params["metrices"]))
            
            yield (X, os.path.join(pathToPredict,chunk))
            
           

    def _predictInteractions(self):
        ""
        predInteractions = None
        for X,pathToChunk in self._loadPairs():
            boolSelfIntIdx = X[:,0] == X[:,1] 

            X = X[boolSelfIntIdx == False]
            #first two rows E1 and E2, remove before predict
            classProba = self.classifier.predict(X[:,[n+3 for n in range(len(self.params["metrices"]))]])
            
            predX = np.append(X,classProba,axis=1)
            print(predX)
            boolPredIdx = classProba[:,1] >= self.params["interactionProbabCutoff"]
            if predInteractions is None:
                predInteractions = predX[boolPredIdx]
            else:
                predInteractions = np.append(predInteractions,predX[boolPredIdx], axis=0)




            print(predInteractions.size)
            np.save(
                file = pathToChunk,
                arr = predX)


        self.predInteractions = predInteractions
        boolDbMatch = np.isin(self.predInteractions[:,2],self.DB.df["E1E2"])
        
        self.predInteractions = np.append(self.predInteractions,boolDbMatch,axis=1)
        print(self.predInteractions)
        print(self.predInteractions.size)

        pd.DataFrame(self.predInteractions).to_csv("OUTPUT.txt", sep="\t")

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
            self._predictInteractions()
        


if __name__ == "__main__":

    X = pd.read_csv("C:/Users/age/Documents/GitHub/ComplexFinder/example-data/HeuselEtAlAebersoldLab.txt", 
                    sep="\t", nrows=150)

 #X = X.set_index("Uniprot ID")
  #  X

    ComplexFinder(indexIsID=False).run(X)





    
