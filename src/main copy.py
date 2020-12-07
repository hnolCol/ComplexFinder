
#python imports
import os 
import gc 
import string
import random
import time
import pickle

#internal imports 

from modules.Distance import DistanceCalculator
from modules.Signal import Signal
from modules.Database import Database
from modules.Predictor import Classifier, ComplexBuilder

import joblib
from joblib import Parallel, delayed, dump, load


import pandas as pd
import numpy as np 
from collections import OrderedDict

from itertools import combinations
from multiprocessing import Pool
from joblib import wrap_non_picklable_objects
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
#sklearn imports
from sklearn.metrics import classification_report, homogeneity_score, v_measure_score
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale, minmax_scale, robust_scale
from scipy.stats import ttest_ind
import umap
import shutil 

filePath = os.path.dirname(os.path.realpath(__file__)) 
pathToTmp = os.path.join(filePath,"tmp")

alignModels = { "LinearRegression": LinearRegression, "RadiusNeighborsRegressor" : RadiusNeighborsRegressor, "KNeighborsRegressor":KNeighborsRegressor}
alignModelsParams = { "LinearRegression": {}, "RadiusNeighborsRegressor" : {"weights":"distance","radius":30} , "KNeighborsRegressor":{"weights":"distance","n_neighbors":10}}
RF_GRID_SEARCH = {#'bootstrap': [True, False],
 'max_depth': [70,None,30],#30,50,
 'max_features': ['sqrt','auto'],
 'min_samples_leaf': [2, 3, 4],
 'min_samples_split': [2, 3, 4],
 'n_estimators': [400]}

# RF_GRID_SEARCH = {#'bootstrap': [True, False],
#  'max_depth': [None],#30,50,
#  'max_features': ['auto'],
#  'min_samples_leaf': [2],
#  'min_samples_split': [4,5],
#  'n_estimators': [200]}
OPTICS_PARAM_GRID = {"min_samples":[2,3,5,8], "max_eps": [np.inf,2,1,0.9,0.8], "xi": np.linspace(0,0.3,num=30), "cluster_method" : ["xi"]}
AGGLO_PARAM_GRID = {"n_clusters":[None,115,110,105,100,90,95],"distance_threshold":[None,0.5,0.4,0.2,0.1,0.05,0.01], "linkage":["complete","single","average"]}
AFF_PRO_PARAM = {"damping":np.linspace(0.5,1,num=50)}
HDBSCAN_PROPS = {"min_cluster_size":[3,4,5,8,12],"min_samples":[1,8,10,15,20]}
CLUSTER_PARAMS = {"OPTICS":OPTICS_PARAM_GRID,"AGGLOMERATIVE_CLUSTERING":AGGLO_PARAM_GRID,"AFFINITY_PROPAGATION":AFF_PRO_PARAM,"HDBSCAN":HDBSCAN_PROPS}

param_grid = {'C': [1, 10, 100, 1000], 'kernel': ['linear','rbf','poly'], 'gamma': [0.01,0.1,1,2,3,4,5]}

entriesInChunks = dict() 


def _calculateDistanceP(pathToFile):
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


class ComplexFinder(object):

    def __init__(self,
                indexIsID = True,
                plotSignalProfiles = False,
                plotComplexProfiles = False,
                removeSingleDataPointPeaks = True,
                restartAnalysis = False,
                maxPeaksPerSignal = 15,
                maxPeakCenterDifference = 1.8,
                n_jobs = 12,
                classifierTestSize = 0.2,
                binaryDatabase = False,
                noDatabaseForPredictions = False,
                useRawDataForDimensionalReduction = False,
                scaleRawDataBeforeDimensionalReduction = True,
                kFold = 3,
                alignRuns = False,
                alignMethod = "RadiusNeighborsRegressor",#"RadiusNeighborsRegressor",#"KNeighborsRegressor",#"LinearRegression", # RadiusNeighborsRegressor
                alignWindow = 3,
                minDistanceBetweenTwoPeaks = 3,
                analysisName = None,
                idColumn = "Uniprot ID",
                databaseFileName = "20190823_CORUM.txt",#s"humap2.txt",#
                peakModel = "GaussianModel",#"SkewedGaussianModel",#"LorentzianModel",
                imputeNaN = True,
                smoothSignal = True,
                classifierClass = "GaussianNB",# "random_forest",
                grouping = {"KO": ["D3_KO_01.txt","D3_KO_01.txt"], "WT" : ["D3_WT_01.txt","D3_WT_02.txt","D3_WT_03.txt","D3_WT_04.txt"]},
                analysisMode = "label-free", #replicate #[label-free,SILAC,SILAC-TMT]
                retrainClassifier = False,
                interactionProbabCutoff = 0.7,
                databaseFilter = {'Organism': ["Human"]}, #{"Confidence" : [1,2,3]},#
                databaseIDColumn = "subunits(UniProt IDs)",#"Uniprot_ACCs",#
                databaseHasComplexAnnotations = True,
                normValueDict = {},
                r2Thresh = 0.85,
                metrices = ["apex","pearson","euclidean","p_pearson","max_location","umap-dist"],#"euclidean","p_pearson","max_location"], #"euclidean",,"spearman"##,
                metricesForPrediction = None,#["pearson","euclidean","apex"],
                classiferGridSearch = RF_GRID_SEARCH,
                hdbscanDefaultKwargs = {"min_cluster_size":4,"min_samples":1},
                umapDefaultKwargs = {"min_dist":0.0000001,"n_neighbors":3,"n_components":2},
                runName = None,
                metricQuantileCutoff = 0.90,
                recalculateDistance = False,
                decoySizeFactor = 2.5):
        """
        Init ComplexFinder Class

        

        Parameters
        ----------

       
    
        Returns
        -------
        None

        """

        self.params = {
            "indexIsID" : indexIsID,
            "idColumn" : idColumn,
            "n_jobs" : n_jobs,
            "kFold" : kFold,
            "analysisName" : analysisName,
            "restartAnalysis" : restartAnalysis,
            "imputeNaN" : imputeNaN,
            "metrices" : metrices,
            "peakModel" : peakModel,
            "classifierClass" : classifierClass,
            "retrainClassifier" : retrainClassifier,
            "interactionProbabCutoff":interactionProbabCutoff,
            "maxPeaksPerSignal" : maxPeaksPerSignal,
            "maxPeakCenterDifference" : maxPeakCenterDifference,
            "classiferGridSearch" : classiferGridSearch,
            "plotSignalProfiles" : plotSignalProfiles,
            "savePeakModels" : True, #must be true to process pipeline, depracted, remove from class arguments.
            "removeSingleDataPointPeaks" : removeSingleDataPointPeaks,
            "grouping" : grouping,
            "analysisMode" : analysisMode,
            "normValueDict" : normValueDict,
            "databaseFilter" : databaseFilter,
            "databaseIDColumn" : databaseIDColumn,
            "databaseFileName" : databaseFileName,
            "databaseHasComplexAnnotations" : databaseHasComplexAnnotations,
            "r2Thresh" : r2Thresh,
            "smoothSignal" : smoothSignal,
            "umapDefaultKwargs" : umapDefaultKwargs,
            "hdbscanDefaultKwargs" : hdbscanDefaultKwargs,
            "noDatabaseForPredictions" : noDatabaseForPredictions,
            "alignRuns" : alignRuns,
            "alignMethod" : alignMethod,
            "runName" : runName,
            "useRawDataForDimensionalReduction" : useRawDataForDimensionalReduction,
            "scaleRawDataBeforeDimensionalReduction" : scaleRawDataBeforeDimensionalReduction,
            "metricQuantileCutoff": metricQuantileCutoff,
            "recalculateDistance" : recalculateDistance,
            "metricesForPrediction" : metricesForPrediction,
            "minDistanceBetweenTwoPeaks" : minDistanceBetweenTwoPeaks,
            "plotComplexProfiles" : plotComplexProfiles,
            "decoySizeFactor" : decoySizeFactor,
            "classifierTestSize" : classifierTestSize
            }
        print("\n" + str(self.params))
        self._checkParameterInput()

    def _checkParameterInput(self):
        ""

        #check anaylsis mode
        validModes = ["label-free","SILAC","TMT-SILAC"]
        if self.params["analysisMode"] not in validModes:
            raise ValueError("Parmaeter analysis mode is not valid. Must be one of: {}".format(validModes))

        if not isinstance(self.params["maxPeaksPerSignal"],int):
            raise ValueError("maxPeaksPerSignal must be an integer. Current setting: {}".forma(self.params["maxPeaksPerSignal"]))

        elif self.params["maxPeaksPerSignal"] < 1:
            raise ValueError("maxPeaksPerSignal must be greater than or equal 1")

        elif self.params["maxPeaksPerSignal"] > 20:
            print("Warning :: maxPeaksPerSignal is set to above 20, this may take quite long to model.")

        #r2 validation
        if not isinstance(self.params["r2Thresh"],float):
            raise ValueError("Parameter r2Trehsh mus be a floating number.")
        elif self.params["r2Thresh"] < 0.6:
            print("Warning :: threshold for r2 is set below 0.6. This might result in fits of poor quality")
        elif self.params["r2Thresh"] > 0.95:
            print("Warning :: threshold for r2 is above 0.95. Relatively few features might pass this limit.")
        elif self.params["r2Thresh"] > 0.99:
            raise ValueError("Threshold for r2 was above 0.99. Please set a lower value.")

        #k-fold
        if not isinstance(self.params["kFold"],int):
            raise ValueError("Parameter kFold mus be an integer.")
        elif self.params["kFold"] < 2:
            raise ValueError("Parameter kFold must be at least 2.")

        if self.params["alignMethod"] not in alignModels:
            raise ValueError("Parameter alignMethod must be in {}".format(alignModels.values()))

        if not isinstance(self.params["metricQuantileCutoff"],float) or self.params["metricQuantileCutoff"] <= 0 or self.params["metricQuantileCutoff"] >= 1:
            raise ValueError("Parameter metricQuantileCutoff must be a float greater than 0 and smaller than 1.")
        #add database checks


        if self.params["metricesForPrediction"] is not None:
            if not isinstance(self.params["metricesForPrediction"],list):
                raise TypeError("metricesForPrediction must be a list.")
            else:
                if not all(x in self.params["metrices"] for x in self.params["metricesForPrediction"]):
                    raise ValueError("All metrices given in 'metricesForPrediction' must be present in 'metrices'.")
        else:
            self.params["metricesForPrediction"] = self.params["metrices"]

    def _load(self, X):
        "Load data"
        if isinstance(X, pd.DataFrame):
            
            self.X = X

            if not self.params["indexIsID"]:
                print("checking for duplicates")
                dupRemoved = self.X.drop_duplicates(subset=[self.params["idColumn"]])
                if dupRemoved.index.size < self.X.index.size:
                    print("Warning :: Duplicates detected.")
                    print("File contained duplicate ids which will be removed: {}".format(self.X.index.size-dupRemoved.index.size))
                    self.X = dupRemoved
                np.save(os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"source"),self.X.values)

                self.X = self.X.set_index(self.params["idColumn"])
                self.X = self.X.astype(np.float32)
            else:
                self.X = self.X.loc[self.X.index.drop_duplicates()] #remove duplicaates
                self.X = self.X.astype(np.float32) #set dtype to 32 to save memory

        else:

            raise ValueError("X must be a pandas data frame")


    def _addModelToSignals(self,signalModels):
        "Asigns the model atributes to the signal."
        for fitModel in signalModels:
            modelID = fitModel["id"]
            if len(fitModel) == 1:
                del self.Signals[self.currentAnalysisName][modelID]
            if modelID in self.Signals[self.currentAnalysisName]:
                for k,v in fitModel.items():
                    if k != 'id':
                       # print(self.Signals[self.currentAnalysisName][modelID])
                        setattr(self.Signals[self.currentAnalysisName][modelID],k,v)
                self.Signals[self.currentAnalysisName][modelID].saveResults()
                
    def _checkGroups(self):
        "Checks grouping. For comparision of multiple co-elution data sets."
        
        if isinstance(self.params["grouping"],dict):
            if len(self.params["grouping"]) == 0:
                raise ValueError("Example for grouping : {'KO':['KO_01.txt','KO_02.txt'], 'WT':['WT_01.txt','WT_02.txt'] } Aborting.. ")
            else:
                combinedSamples = sum(self.params["grouping"].values(), [])
                if all(x in combinedSamples for x in self.params["analysisName"]):
                    print("Grouping checked..\nAll columnSuffixes found in grouping.")
                else:
                    raise ValueError("Could not find all grouping names in loaded dataframe.. Aborting ..")
                

    def _findPeaks(self, n_jobs=3):
        ""
        if self.allSamplesFound:
            print("Info :: Signals loaded and found. Proceeding ...")
            return
        pathToSignal = os.path.join(self.params["pathToComb"],"signals.lzma")
        if os.path.exists(pathToSignal):
            self.Signals = load(pathToSignal)
            print("\nLoading pickled signal intensity")
            if all(analysisName in self.Signals for analysisName in self.params["analysisName"]):
                print("Info :: All samples found in loaded Signals..")
                self.allSamplesFound = True
                return
        
            
        if not hasattr(self , "Signals"):
            self.Signals = OrderedDict()

        self.Signals[self.analysisName] = dict()
        peakModel = self.params['peakModel']

        for entryID, signal in self.X.iterrows():

            self.Signals[self.analysisName][entryID] = Signal(signal.values,
                                            ID=entryID, 
                                            peakModel=peakModel, 
                                            smoothSignal = self.params["smoothSignal"],
                                            savePlots = self.params["plotSignalProfiles"],
                                            savePeakModels = self.params["savePeakModels"],
                                            maxPeaks=self.params["maxPeaksPerSignal"],
                                            metrices=self.params["metrices"],
                                            pathToTmp = self.params["pathToTmp"][self.currentAnalysisName],
                                            normalizationValue = self.params["normValueDict"][entryID] if entryID in self.params["normValueDict"] else None,
                                            removeSingleDataPointPeaks = self.params["removeSingleDataPointPeaks"],
                                            analysisName = self.currentAnalysisName,
                                            r2Thresh = self.params["r2Thresh"],
                                            minDistanceBetweenTwoPeaks = self.params["minDistanceBetweenTwoPeaks"])

        
        t1 = time.time()
        print("\n\nStarting Signal modelling .. (n_jobs = {})".format(n_jobs))
        
        fittedModels = Parallel(n_jobs=n_jobs, verbose=1)(delayed(Signal.fitModel)() for Signal in self.Signals[self.currentAnalysisName].values())
        
        self._addModelToSignals(fittedModels)
        
        print("Peak fitting done time : {} secs".format(round((time.time()-t1))))
        print("Each feature's fitted models is stored as pdf and txt is stored in model plots (if savePeakModels and plotSignalProfiles was set to true)")
        
        #dump(self.Signals,pathToSignal)

    def _saveSignals(self):
        ""
        if hasattr(self,"Signals") :
            pathToSignal = os.path.join(self.params["pathToComb"],"signals.lzma")
            dump(self.Signals.copy(),pathToSignal)
        
        self.Xs = {}
        for analysisName in self.params["analysisName"]:
            signals = self.Signals[analysisName]
            data = dict([(k,v.Y) for k,v in signals.items() if v.valid and v.validModel])
            df = pd.DataFrame().from_dict(data,orient="index")
            self.Xs[analysisName] = df
            

    def _calculateDistance(self):
        """
        Calculates distances between Signals by creating chunks of signals. Each chunk is analysed separately.
        ToDo : 
        delete all npy files in chunks? To be on safe side, if df is reduced.
        parameters

        returns
            None
        """
        global entriesInChunks
        
        print("\nStarting Distance Calculation ...")
        t1 = time.time()
        
        chunks = self.signalChunks[self.currentAnalysisName]
        #return
        entrieChunkPath = os.path.join(self.params["pathToComb"], "entriesInChunk.pkl")
        if not self.params["recalculateDistance"] and all(os.path.exists(x.replace(".pkl",".npy")) for x in chunks) and os.path.exists(entrieChunkPath):
            print("All chunks found for distance calculation.")

            with open(os.path.join(self.params["pathToComb"], "entriesInChunk.pkl"),"rb") as f:
                       entriesInChunks = pickle.load(f)

        else:

            chunkItems = Parallel(n_jobs=self.params["n_jobs"], verbose=10)(delayed(_calculateDistanceP)(c) for c in chunks)
            entriesInChunks[self.currentAnalysisName] = {}
            for k,v in chunkItems:
                for E1E2 in v:
                    entriesInChunks[self.currentAnalysisName][E1E2] = k 
            
            with open(os.path.join(self.params["pathToComb"], "entriesInChunk.pkl"),"wb") as f:
                        pickle.dump(entriesInChunks,f)

        print("Distance computing: {} secs\n".format(round(time.time()-t1)))

    def _createSignalChunks(self,chunkSize = 30):
        """
        Creates signal chunks at given chunk size. 
        
        parameter
            chunkSize - int. default 30. Nuber of signals in a single chunk.

        returns
            list of paths to the saved chunks.
        """
        pathToSignalChunk = os.path.join(self.params["pathToComb"],"signalChunkNames.lzma")
        if os.path.exists(pathToSignalChunk) and not self.params["recalculateDistance"]:
            self.signalChunks = load(pathToSignalChunk)
            print("Signal chunks loaded and found.")
            if all(analysisName in self.signalChunks for analysisName in self.params["analysisName"]):
                print("Checked... all samples found.")
                return
            else:
                print("Info :: Not all samples found. Creating new signal chunks..")
        
        if not hasattr(self,"signalChunks"):
            self.signalChunks = dict() 
        else:
            self.signalChunks.clear()
        print("Info :: Creating signal chunks for distance calculation.")
        #print(self.params["metrices"])
        for analysisName in self.params["analysisName"]:
            print("Info :: {} signal chunk creation started." .format(analysisName))
            if "umap-dist" in self.params["metrices"]:
                embed = umap.UMAP(min_dist=0.0000000000001, n_neighbors=5, metric = "correlation").fit_transform(minMaxNorm(self.Xs[analysisName].values,axis=1))
                embed = pd.DataFrame(embed,index=self.Xs[analysisName].index)

            signals = list(self.Signals[analysisName].values())

            for n,Signal in enumerate(self.Signals[analysisName].values()):
                setattr(Signal,"otherSignals", signals[n:])

            c = []
            

            for n,chunk in enumerate(chunks(signals,chunkSize)):
                pathToChunk = os.path.join(self.params["pathToTmp"][analysisName],"chunks",str(n)+".pkl")
                #if not os.path.exists(pathToChunk) and not self.params["recalculateDistance"]:
                chunkItems =  [
                    {"ID" : str(signal.ID),
                    "chunkName" : str(n),
                    "Y" : np.array(signal.Y),
                    "ownPeaks" : signal._collectPeakResults(),
                    "otherSignalPeaks" : [s._collectPeakResults() for s in signal.otherSignals],
                    "E2" : [str(s.ID) for s in signal.otherSignals],
                    "metrices" : self.params["metrices"],
                    "pathToTmp" : self.params["pathToTmp"][analysisName],
                    "embedding" : embed.loc[signal.ID].values if "umap-dist" in self.params["metrices"] else [],
                    "otherSignalEmbeddings" : [embed.loc[s.ID].values for s in signal.otherSignals] if "umap-dist" in self.params["metrices"] else []} for signal in chunk]
                
                with open(pathToChunk,"wb") as f:
                    pickle.dump(chunkItems,f)
                
                gc.collect()
                c.append(pathToChunk)
            
            self.signalChunks[analysisName] = c
        
        dump(self.signalChunks,pathToSignalChunk)



    def _collectRSquaredAndFitDetails(self):
        """
        Data are collected from txt files in the modelPlots folder. 
            
        """
        if not self.params["savePeakModels"]:
            print("!! Warning !! This parameter is depracted and from now on always true.")

        rSqured = [] 
        entryList = []
        pathToPlotFolder = os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"result","modelPlots")
        resultFolder = os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"result")

        fittedPeaksPath = os.path.join(resultFolder,"fittedPeaks.txt")
        nPeaksPath = os.path.join(resultFolder,"nPeaks.txt")
        if os.path.exists(fittedPeaksPath) and os.path.exists(nPeaksPath):
            print("Warning :: FittedPeaks detected. If you changed the data, you have to set the paramter 'restartAnalysis' True to include changes..")
            return
        if not os.path.exists(resultFolder):
            os.mkdir(resultFolder)
        #ugly solution to read file names
        #find squared R
        for file in os.listdir(pathToPlotFolder):
            if file.endswith(".txt"):
                try:
                    r = float(file.split("_")[-1][:-4])
                    entryList.append(file.split("_r2")[0])
                    rSqured.append({"ID":file.split("_")[0],"r2":r})
                except:
                    continue
        df = pd.DataFrame(rSqured, columns = ["r2"])
        df["EntryID"] = entryList
         
        df.to_csv(os.path.join(resultFolder,"rSquared.txt"),sep="\t")

        #number of peaks
        collectNumbPeaks = []

        # find peak properties..
        df = pd.DataFrame(columns=["Key","ID","Amplitude","Center","Sigma","fwhm","height","auc"])
        for file in os.listdir(pathToPlotFolder):
            if file.endswith(".txt"):
                try:
                    dfApp = pd.read_csv(os.path.join(pathToPlotFolder,file), sep="\t")
                    df = df.append(dfApp)
                    collectNumbPeaks.append({"Key":dfApp["Key"].iloc[0],"N":len(dfApp.index)})
                except:
                    continue

        df.index = np.arange(df.index.size)
        df.to_csv(fittedPeaksPath,sep="\t", index = None)

        pd.DataFrame(collectNumbPeaks).to_csv(nPeaksPath,sep="\t", index = None)


    def _loadReferenceDB(self):
        """
        Load reference database.

        filterDB (dict) is passed to the pandas pd.DataFrame.isin function.

        Parameters
        ----------
    
        Returns
        -------
        None

        """
        if self.params["noDatabaseForPredictions"]:
            print("Info ::  Parameter noDatabaseForPredictions was set to True. No database laoded.")
            return

        print("Info :: Load positive set from data base")
        self.DB = Database(nJobs = self.params["n_jobs"])
        pathToDatabase = os.path.join(self.params["pathToTmp"][self.currentAnalysisName], "InteractionDatabase.txt")
        if os.path.exists(pathToDatabase):

            dbSize = self.DB.loadDatabaseFromFile(pathToDatabase)
            print("Info :: Database found and loaded. Contains {} positive interactions.".format(dbSize))
            self._addMetricToStats("nPositiveInteractions",dbSize)
        else:
            
            self.DB.pariwiseProteinInteractions(
                            self.params["databaseIDColumn"],
                            dbID = self.params["databaseFileName"],
                            filterDb=self.params["databaseFilter"])
            entryList = [entryID for entryID,Signal in self.Signals[self.currentAnalysisName].items() if Signal.valid]
            print("Info :: Entries used for filtering: {}".format(len(entryList)))
            dbSize = self.DB.filterDBByEntryList(entryList)
            self._addMetricToStats("nPositiveInteractions",dbSize)
            #add decoy to db
            if dbSize == 0:
                raise ValueError("Warning :: No hits found in database. Check dabaseFilter keyword.")
            elif dbSize < 150:
                raise ValueError("Warining :: Less than 150 pairwise interactions found.")
            elif dbSize < 200:
                #raise ValueError("Filtered positive database contains less than 200 interactions..")
                print("Warning :: Filtered positive database contains less than 200 interactions.. {}".format(dbSize))
                print("Warning :: Please check carefully, if the classifier has enough predictive power.")
            self.DB.addDecoy(sizeFraction=self.params["decoySizeFactor"])
            self.DB.df.to_csv(pathToDatabase,sep="\t")
            print("Info :: Database saved to {}".format(pathToDatabase))

    def _addMetricesToDB(self):
        """
        Adds distance metrices to the database entries
        that were found in the co-elution profiles.

        Parameters
        ----------
    
        Returns
        -------
        None

        """
        metricColumns = self.params["metrices"] 
        if not self.params["noDatabaseForPredictions"]:
            self.DB.matchMetrices(self.params["pathToTmp"][self.currentAnalysisName],entriesInChunks[self.currentAnalysisName],metricColumns,forceRematch=False)


    def _trainPredictor(self):
        """
        Trains the predictor based on positive interactions
        in the database.

        Parameters
        ----------
    
        Returns
        -------
        None

        """
        #metricColumns = [col for col in self.DB.df.columns if any(x in col for x in self.params["metrices"])]
        
        if self.params["noDatabaseForPredictions"]:
            print("Predictor training skipped (noDatabaseForPredictions = True). Distance metrices are used for dimensional reduction.")
            return 

        folderToResults = os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"result")
        classifierFileName = os.path.join(self.params["pathToTmp"][self.currentAnalysisName],'trainedClassifier_{}.sav'.format(self.params["classifierClass"]))

        if not self.params["retrainClassifier"] and os.path.exists(classifierFileName): #enumerate(
            print("Info :: Prediction was done already... loading file")
            self.classifier = joblib.load(classifierFileName)
            return

        if not os.path.exists(folderToResults):
            os.mkdir(folderToResults)
        
       # if self.params["metricesForPrediction"] is None:
        metricColumnsForPrediction = self.params["metrices"]
       # else:
          #  metricColumnsForPrediction = self.params["metricesForPrediction"]

        totalColumns = metricColumnsForPrediction + ['Class']
        data = self.DB.dfMetrices[totalColumns].dropna(subset=metricColumnsForPrediction)
        Y = data['Class'].values
        X = data.loc[:,metricColumnsForPrediction].values

        self.classifier = Classifier(
            classifierClass = self.params["classifierClass"],
            n_jobs=self.params['n_jobs'], 
            gridSearch = self.params["classiferGridSearch"],
            testSize = self.params["classifierTestSize"])

        probabilites, meanAuc, stdAuc, oobScore, optParams = self.classifier.fit(X,Y,kFold=self.params["kFold"],pathToResults=folderToResults, metricColumns = metricColumnsForPrediction)
       
        boolClass = probabilites > self.params["interactionProbabCutoff"]

        data["PredictionClass"] = probabilites
        pathToFImport = os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"result","PredictorSummary(p>{}){}.txt".format(self.params["interactionProbabCutoff"],self.params["metrices"]))
        #create and save classification report
        classReport = classification_report(
                            Y,
                            boolClass,
                            digits=3,
                            output_dict=True)
        classReport = OrderedDict([(k,v) for k,v in classReport.items() if k != 'accuracy'])

        pd.DataFrame().from_dict(classReport, orient="index").to_csv(pathToFImport, sep="\t", index=True)

        #save database prediction
        data.to_csv(os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"DBpred.txt"),sep="\t", index=False)
        
        self._plotFeatureImportance(folderToResults)

        joblib.dump(self.classifier, classifierFileName)
        self._addMetricToStats("Metrices",str(metricColumnsForPrediction))
        self._addMetricToStats("OOB_Score",oobScore)
        self._addMetricToStats("ROC_Curve_AUC","{}+-{}".format(meanAuc,stdAuc))
        self._addMetricToStats("ClassifierParams",optParams)
        
        print("DB prediction saved - DBpred.txt :: Classifier pickled and saved 'trainedClassifier.sav'")

    def _addMetricToStats(self,metricName, value):

        if metricName in self.stats.columns:
            self.stats.loc[self.currentAnalysisName,metricName] = value

    def _loadPairsForPrediction(self):
        ""
        chunks = [f for f in os.listdir(os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"chunks")) if f.endswith(".npy") and f != "source.npy"]
       
        print("\nInfo :: Prediction/Dimensional reduction started...")
        for chunk in chunks:

            X = np.load(os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"chunks",chunk),allow_pickle=True)
            
            yield (X,len(chunks))

            
        
    def _chunkPrediction(self,pathToChunk,classifier,nMetrices,probCutoff):
        """
        Predicts for each chunk the proability for positive interactions.

        Parameters
        ----------
            pathToChunk : str

            nMetrices : int
                Number if metrices used. (since chunks are simple numpy arrays, no column headers are loaded)

            probCutoff : float
                Probability cutoff.
        Returns
        -------
        Chunks with appended probability.
        """

        X =  np.load(pathToChunk,allow_pickle=True)
        boolSelfIntIdx = X[:,0] != X[:,1] 
        X = X[boolSelfIntIdx]
        classProba = classifier.predict(X[:,[n+3 for n in range(nMetrices)]])

        #boolPredIdx = classProba >= probCutoff
        #boolIdx = np.sum(boolPredIdx,axis=1) > 0
        predX = np.append(X[:,2],classProba.reshape(X.shape[0],-1),axis=1)
        np.save(
                file = pathToChunk,
                arr = predX)

        return predX



    def _predictInteractions(self):
        ""
        if self.params["noDatabaseForPredictions"]:
            print("Info :: Skipping predictions. (noDatabaseForPredictions = True)")
            return
        paramDict = {"nInteracotrs" : 0, "positiveInteractors" : 0, "decoyInteractors" : 0, "novelInteractions" : 0, "interComplexInteractions" : 0}
        probCutoffs = dict([(cutoff,paramDict.copy()) for cutoff in np.linspace(0.0,0.99,num=25)])

        print("Info :: Starting prediction ..")
        folderToOutput = os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"result")
        pathToPrediction = os.path.join(folderToOutput,"predictedInteractions{}_{}.txt".format(self.params["metricesForPrediction"],self.params["classifierClass"]))
        if not self.params["retrainClassifier"] and os.path.exists(pathToPrediction):
            predInts = pd.read_csv(pathToPrediction, sep="\t")
            self.stats.loc[self.currentAnalysisName,"nInteractions ({})".format(self.params["interactionProbabCutoff"])] = predInts.index.size
            return predInts
       # del self.Signals
        gc.collect()   
        #create prob columns of k fold 
        pColumns = ["Prob_{}".format(n) for n in range(len(self.classifier.predictors))]
        dfColumns = ["E1","E2","E1E2","apexPeakDist"] + [x if not isinstance(x,dict) else x["name"] for x in self.params["metrices"]] + pColumns + ["In DB"] 

        if not os.path.exists(folderToOutput):
            os.mkdir(folderToOutput)

        predInteractions = None                                     
         
        metricIdx = [n + 4 if "apex" in self.params["metrices"] else n + 3 for n in range(len(self.params["metrices"]))] #in order to extract from dinstances
        #print("metric idx", metricIdx)
        #print(self.params["metrices"])
        

        #classProba = Parallel(n_jobs=self.params["n_jobs"], verbose=10)(delayed(self.classifier.predict)(X[:,metricIdx].astype(np.float32)) for X,pathToChunk,nChunks in self._loadPairsForPrediction())
        for n,(X,nChunks) in enumerate(self._loadPairsForPrediction()):
            boolSelfIntIdx = X[:,0] == X[:,1] 
            print(round(n/nChunks*100,2),r"% done.")
            X = X[boolSelfIntIdx == False]
            #first two rows E1 E2, and E1E2, apexPeakDist remove before predict
            
            classProba = self.classifier.predict(X[:,metricIdx])
            
            if classProba is None:
                continue
            predX = np.append(X,classProba.reshape(X.shape[0],-1),axis=1)
            interactionClass = self.DB.getInteractionClassByE1E2(X[:,2],X[:,0],X[:,1])
            #print(interactionClass.loc[boolPredIdx].value_counts())
            

            for cutoff in probCutoffs.keys():
                
                boolPredIdx = classProba >= cutoff
                
                if len(boolPredIdx.shape) > 1:
                    boolIdx = np.sum(boolPredIdx,axis=1) == self.params["kFold"]
                else:
                    boolIdx = boolPredIdx

                counts = interactionClass.loc[boolIdx].value_counts()
                
                n = np.sum(boolIdx)
                probCutoffs[cutoff]["nInteracotrs"] += n
                probCutoffs[cutoff]["positiveInteractors"] += counts["pos"] if "pos" in counts.index else 0
                probCutoffs[cutoff]["decoyInteractors"] += counts["decoy"] if "decoy" in counts.index else 0 
                probCutoffs[cutoff]["novelInteractions"] += counts["unknown/novel"] if  "unknown/novel" in counts.index else 0 
                probCutoffs[cutoff]["interComplexInteractions"] += counts["inter"] if "inter" in counts.index else 0

            boolPredIdx = classProba >= self.params["interactionProbabCutoff"]
            if len(boolPredIdx.shape) > 1:
                boolIdx = np.sum(boolPredIdx,axis=1) == self.params["kFold"]
            else:
                boolIdx = boolPredIdx
            
            predX = np.append(predX,interactionClass.values.reshape(predX.shape[0],1),axis=1)
            if predInteractions is None:
                predInteractions = predX[boolIdx,:]
            else:
                predInteractions = np.append(predInteractions,predX[boolIdx], axis=0)

            

            #del predX
            #gc.collect()
        probData = pd.DataFrame().from_dict(probCutoffs, orient="index")
        probData.to_csv(os.path.join(folderToOutput,"classiferProbsMetric{}.txt".format(self.params["classifierClass"])),sep="\t")

        # print("Interactions > cutoff :", predInteractions.shape[0])
        # print("Info :: Finding interactions in DB")
        # boolDbMatch = np.isin(predInteractions[:,2],self.DB.df["E1E2"].values, assume_unique=True)
        # print("Info :: Appending matches.")
        # predInteractions = np.append(predInteractions,boolDbMatch.reshape(predInteractions.shape[0],1),axis=1)
       
        d = pd.DataFrame(predInteractions, columns = dfColumns)
        boolDbMatch = d["In DB"] == "pos"
        print("Info :: Annotate complexes to pred. interactions.")
        d["ComplexID"], d["ComplexName"] = zip(*[self._attachComplexID(_bool,E1E2) for E1E2, _bool in zip(predInteractions[:,2], boolDbMatch)])

        d = self._attachPeakIDtoEntries(d)
        d.to_csv(pathToPrediction, sep="\t")

        

        self.stats.loc[self.currentAnalysisName,"nInteractions ({})".format(self.params["interactionProbabCutoff"])] = d.index.size
        self.stats.loc[self.currentAnalysisName,"Classifier"] = self.params["classifierClass"]
        
        
        
        return d

    def _attachComplexID(self,_bool,E1E2):
        ""
        if not _bool:
            return ("","")
        else:
            df = self.DB.df[self.DB.df["E1E2"] == E1E2]
            return (';'.join([str(x) for x in df["ComplexID"].tolist()]),
                    ';'.join([str(x) for x in df["complexName"].tolist()]))


    def _plotChunkSummary(self, data, fileName, folderToOutput):
        # scale features and plot decision function? 
        # most important featues in SVM?
        ""
        data[self.params["metrices"]] = self.classifier._scaleFeatures(data[self.params["metrices"]].values)
        fig, ax = plt.subplots()

        
        XX = data.melt(id_vars =  [x for x in data.columns if x not in self.params["metrices"]],value_vars=self.params["metrices"])
        sns.boxplot(data = XX, ax=ax, y = "value", x = "variable", hue = "Class")

        plt.savefig(os.path.join(folderToOutput,"{}.pdf".format(fileName)))
        plt.close()
        
        
    def _plotFeatureImportance(self,folderToOutput,*args,**kwargs):
        """
        Creates a bar chart showing the estimated feature importances

        Parameters
        ----------
        folderToOutput : string
            Path to folder to save the pdf. Will be created if it does not exist.
        *args
            Variable length argument list passed to matplotlib.bar.
        **kwargs
            Arbitrary keyword arguments passed to matplotlib.bar.

        Returns
        -------
        None

        """
        fImp = self.classifier.getFeatureImportance()
        
        
        self._makeFolder(folderToOutput)
        if fImp is not None:
            #save as txt file
            pd.DataFrame(fImp, columns= self.params["metrices"]).to_csv(os.path.join(folderToOutput,"featureImportance{}.txt".format(self.params["metrices"])), sep="\t")
            #plot feature importance
            fig, ax = plt.subplots()
            xPos = np.arange(len(self.params["metrices"]))
            ax.bar(x = xPos, height = np.mean(fImp,axis=0), *args,**kwargs)
            ax.errorbar(x = xPos, y = np.mean(fImp,axis=0), yerr = np.std(fImp,axis=0))
            ax.set_xticks(xPos)
            ax.set_xticklabels(self.params["metrices"], rotation = 45)
            plt.savefig(os.path.join(folderToOutput,"featureImportance.pdf"))
            plt.close()
        

    def _randomStr(self,n):
        """
        Returns a random string (lower and upper case) of size n

        Parameters
        ----------
        n : int
            Length of string

        Returns
        -------
        random string of length n

        """

        letters = string.ascii_lowercase + string.ascii_uppercase
        return "".join(random.choice(letters) for i in range(n))
    
    def _scoreComplexes(self, complexDf, complexMemberIds = "subunits(UniProt IDs)", beta=2.5):
        ""

        entryPositiveComplex = [] 
        #entries = [e.split("_",1)[0] for e in complexDf.index]
        #print(complexDf.index)
        for e in complexDf.index:
           # e = e.split("_")[0]

            posComplex = self.DB.assignComplexToProtein(str(e),complexMemberIds,"ComplexID")
            entryPositiveComplex.append(posComplex)

        complexDf.loc[:,"ComplexID"] = entryPositiveComplex
        identifiableCs = self.DB.identifiableComplexes(complexMemberIds) 

        scores = []
        for c,d in self.DB.indentifiedComplexes.items():
            clearedEntries = pd.Series([x.split("_")[0] for x in complexDf.index], index=complexDf.index)
            boolMatch = clearedEntries.isin(d["members"])
            clusters = complexDf.loc[boolMatch,"Cluster Labels"]
            clusterCounts = clusters.value_counts() 
            #clusterCountsEntropy = clusterCounts * np.log(clusterCounts)
            #print(clusterCounts )
            #print(clusterCountsEntropy)
            nMatches = np.sum(boolMatch)
            groundTruth = [1]*nMatches
            if nMatches > 1:
                if clusterCounts.index[0] == -1:
                    if clusterCounts.index.size == 1:
                        continue
                    else:
                        s =  (nMatches - clusterCounts.iloc[1] ) / nMatches
                else:
                #s = v_measure_score(groundTruth,complexDf.loc[boolMatch,"Cluster Labels"], beta = beta) 
                    s =  (nMatches - clusterCounts.iloc[0] ) / nMatches
                scores.append(s)
      
        
        return complexDf , np.nanmean(scores)  #np.unique(complexDf["Cluster Labels"]).size - 


    def _clusterInteractions(self, predInts, clusterMethod = "HDBSCAN", plotEmbedding = True):
        """
        Performs dimensional reduction and clustering of prediction distancen matrix over a defined parameter grid.
        Parameter
            predInts -  ndarray. 
            clusterMethod   -   string. Any string of ["HDBSCAN",]
            plotEmbedding   -   bool. If true, embedding is plotted and save to pdf and txt file. 


        returns
            None
        """
        embedd = None
        bestDf = None
        splitLabels = False
        recordScore = OrderedDict()
        maxScore = np.inf 
        metricColumns = [x if not isinstance(x,dict) else x["name"] for x in self.params["metricesForPrediction"]] 
        cb = ComplexBuilder(method=clusterMethod)

        print("\nPredict complexes")
        if predInts is None:
            print("No database provided. UMAP and clustering will be performed using defaultKwargs. (noDatabaseForPredictions = True)")

        pathToFolder = self._makeFolder(self.params["pathToTmp"][self.currentAnalysisName],"result","complexParamGrid")

        if not self.params["databaseHasComplexAnnotations"] and not self.params["noDatabaseForPredictions"] and predInts is not None:
            print("Database does not contain complex annotations. Therefore standard UMAP settings are HDBSCAN settings are used for complex identification.")
            cb.set_params(self.params["hdbscanDefaultKwargs"])
            clusterLabels, intLabels, matrix , reachability, core_distances, embedd = cb.fit(predInts, 
                                                                                            metricColumns = metricColumns, 
                                                                                            scaler = self.classifier._scaleFeatures,
                                                                                            umapKwargs=  self.params["umapDefaultKwargs"])
        
        elif self.params["noDatabaseForPredictions"]:
            print("Info :: No database given for complex scoring. UMAP and HDBSCAN are performed to identify complexes.")

            if self.params["useRawDataForDimensionalReduction"]:
                if self.params["scaleRawDataBeforeDimensionalReduction"]:
                    X = self.Xs[self.currentAnalysisName]
                    predInts = pd.DataFrame(minMaxNorm(X.values,axis=1), index=X.index, columns = ["scaled({})".format(colName) for colName in X.columns]).dropna()
                else:
                    predInts = self.Xs[self.currentAnalysisName]

                cb.set_params(self.params["hdbscanDefaultKwargs"])
                clusterLabels, intLabels, matrix , reachability, core_distances, embedd, pooledDistances = cb.fit(predInts, 
                                                                                                metricColumns = self.X.columns, 
                                                                                                scaler = None,
                                                                                                umapKwargs =  self.params["umapDefaultKwargs"],
                                                                                                generateSquareMatrix = False,
                                                                                                )
            else:
                predInts = self._loadAndFilterDistanceMatrix()
                predInts[metricColumns] = minMaxNorm(predInts[metricColumns].values,axis=0)
                cb.set_params(self.params["hdbscanDefaultKwargs"])
                clusterLabels, intLabels, matrix , reachability, core_distances, embedd, pooledDistances = cb.fit(predInts, 
                                                                                                metricColumns = metricColumns, 
                                                                                                scaler = None,
                                                                                                poolMethod= "min",
                                                                                                umapKwargs =  self.params["umapDefaultKwargs"],
                                                                                                generateSquareMatrix = True,
                                                                                                )
                df = pd.DataFrame().from_dict({"Entry":intLabels,"Cluster Labels":clusterLabels,"reachability":reachability,"core_distances":core_distances})
                df = df.sort_values(by="Cluster Labels")
                df = df.set_index("Entry")
                if pooledDistances is not None:
                    pooledDistances.to_csv(os.path.join(pathToFolder,"PooledDistance_{}.txt".format(self.currentAnalysisName)),sep="\t")
                
                squaredDf = pd.DataFrame(matrix,columns=intLabels,index=intLabels).loc[df.index,df.index]
                squaredDf.to_csv(os.path.join(pathToFolder,"SquaredSorted_{}.txt".format(self.currentAnalysisName)),sep="\t")

                noNoiseIndex = df.index[df["Cluster Labels"] > 0]

                squaredDf.loc[noNoiseIndex,noNoiseIndex].to_csv(os.path.join(pathToFolder,"NoNoiseSquaredSorted_{}.txt".format(self.currentAnalysisName)),sep="\t")
                splitLabels = True
            if embedd is not None and plotEmbedding:

                #save embedding
                dfEmbed = pd.DataFrame(embedd, columns = ["UMAP_0{}".format(n) for n in range(embedd.shape[1])])
                dfEmbed["clusterLabels"] = clusterLabels
                dfEmbed["labels"] = intLabels
                if splitLabels:
                    dfEmbed["sLabels"] = dfEmbed["labels"].str.split("_",expand=True).values[:,0]
                    dfEmbed = dfEmbed.set_index("sLabels")
                else:
                    dfEmbed = dfEmbed.set_index("labels")

                if self.params["scaleRawDataBeforeDimensionalReduction"] and self.params["useRawDataForDimensionalReduction"]:
                    dfEmbed = dfEmbed.join([self.Xs[self.currentAnalysisName],predInts],lsuffix="_",rsuffix="__")
                else:
                    dfEmbed = dfEmbed.join(self.Xs[self.currentAnalysisName])
                dfEmbed.to_csv(os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"result","UMAP_Embedding.txt"),sep="\t")
                #plot embedding.
                fig, ax = plt.subplots()
                ax.scatter(embedd[:,0],embedd[:,1],s=50, c=clusterLabels, cmap='Spectral')
                plt.savefig(os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"result","E.pdf"))
                plt.close()

        else:
            embedd = None
            for n, params in enumerate(list(ParameterGrid(CLUSTER_PARAMS[clusterMethod]))):
                try:
                    cb.set_params(params)
                    if clusterMethod == "HDBSCAN":
                        clusterLabels, intLabels, matrix , reachability, core_distances, embedd, pooledDistances = cb.fit(predInts, 
                                                                                            metricColumns = [colName for colName in predInts.columns if "Prob_" in colName],#,#[colName for colName in predInts.columns if "Prob_" in colName], 
                                                                                            scaler = None,#self.classifier._scaleFeatures, #
                                                                                            inv = True,
                                                                                            poolMethod="min",
                                                                                            preCompEmbedding = None,
                                                                                            )
                    else:
                        clusterLabels, intLabels, matrix , reachability, core_distances = cb.fit(predInts, 
                                                                                            metricColumns = [colName for colName in predInts.columns if "Prob_" in colName], 
                                                                                            scaler = self.classifier._scaleFeatures)
                # clusterLabels, intLabels, matrix , reachability, core_distances = cb.fit(predInts, metricColumns = probColumn, scaler = None, inv=True, poolMethod="mean")
                except Exception as e:
                    print(e)
                    print("\nWarning :: There was an error performing clustering and dimensional reduction, using the params:\n" + str(params))
                    continue
                df = pd.DataFrame().from_dict({"Entry":intLabels,"Cluster Labels":clusterLabels,"reachability":reachability,"core_distances":core_distances})
                df = df.sort_values(by="Cluster Labels")
                df = df.set_index("Entry")


            # clusteredComplexes = df[df["Cluster Labels"] != -1]
                df, score = self._scoreComplexes(df)
                
            # df = df.join(assignedIDs[["ComplexID"]])
                if True:#maxScore > score:
                    df.to_csv(os.path.join( pathToFolder,"Complexes_{}_{}.txt".format(n,score)),sep="\t") 
                    print("Info :: Current best params ... ")
                    print(params)
                    squaredDf = pd.DataFrame(matrix,columns=intLabels,index=intLabels).loc[df.index,df.index]
                    squaredDf.to_csv(os.path.join(pathToFolder,"SquaredSorted_{}.txt".format(n)),sep="\t")

                    noNoiseIndex = df.index[df["Cluster Labels"] > 0]
                    squaredDf.loc[noNoiseIndex,noNoiseIndex].to_csv(os.path.join(pathToFolder,"NoNoiseSquaredSorted_{}_{}.txt".format(self.currentAnalysisName,n)),sep="\t")
                    maxScore = score
                    bestDf = df
                    self._plotComplexProfiles(bestDf, pathToFolder, str(n))

                if embedd is not None and plotEmbedding:
                    #save embedding
                    umapColumnNames = ["UMAP_{}".format(n) for n in range(embedd.shape[1])]
                    dfEmbed = pd.DataFrame(embedd, columns = umapColumnNames)
                    embedd = dfEmbed[umapColumnNames]
                    dfEmbed["clusterLabels"] = clusterLabels
                    dfEmbed["Entry"] = intLabels
                    dfEmbed = dfEmbed.set_index("Entry")
                    dfEmbed.loc[dfEmbed.index,"ComplexID"] = df["ComplexID"].loc[dfEmbed.index]
                    dfEmbed = dfEmbed.join(self.Xs[self.currentAnalysisName])
                    dfEmbed.to_csv(os.path.join(pathToFolder,"UMAP_Embeding_{}_{}.txt".format(n,params)),sep="\t")
                    
                    #plot embedding.
                    fig, ax = plt.subplots()
                    ax.scatter(embedd["UMAP_0"].values, embedd["UMAP_1"].values,s=50, c=clusterLabels, cmap='Spectral')
                    plt.savefig(os.path.join(pathToFolder,"E_n{}.pdf".format(n)))
                    plt.close()

                recordScore[n] = {"score":score,"params":params}
        

    def _loadAndFilterDistanceMatrix(self):
        """

        Parameters
        ----------

        Returns
        -------
        None

        """
        #analysisName = self.currentAnalysisName
        metricColumns = [x if not isinstance(x,dict) else x["name"] for x in self.params["metrices"]] 
        dfColumns = ["E1","E2","E1E2","apexPeakDist"] + metricColumns
        print(dfColumns)
        q = None
        df = pd.DataFrame(columns = dfColumns)
        filteredExisting = False
        pathToFile = os.path.join(self.params["pathToComb"],"highQualityInteractions({}).txt".format(self.currentAnalysisName))
        #if os.path.exists(pathToFile):
         #   df = pd.read_csv(pathToTmp,sep="\t")
          #  return df
        for X,nChunks in self._loadPairsForPrediction():
            
            boolSelfIntIdx = X[:,0] == X[:,1] 
            X = X[boolSelfIntIdx == False]
            if q is None:
                df = df.append(pd.DataFrame(X, columns = dfColumns), ignore_index=True)
            else:
                if not filteredExisting:
                    #first reduce existing df
                    mask = df[metricColumns] < q#X[:,[n+4 for n in range(len(self.params["metrices"]))]] < q
                    df = df.loc[np.any(mask,axis=1)] #filtered
                    filteredExisting = True
                toAttach = pd.DataFrame(X, columns = dfColumns)
                mask = toAttach[metricColumns] < q 
                toAttach = toAttach.loc[np.any(mask,axis=1)]
                df = df.append(toAttach, ignore_index=True)

            if df.index.size > 50000 and q is None:
                q = np.quantile(df[metricColumns].astype(float).values, q = 1-self.params["metricQuantileCutoff"], axis = 0)
                


            
           # print("Current Chunk: ",pathToChunk)
            
            #first two rows E1 E2, and E1E2, apexPeakDist remove before predict
            #classProba = self.classifier.predict(X[:,[n+4 for n in range(len(self.params["metrices"]))]])
        
        print("Info :: {} total pairwise protein-protein pairs at any distance below 10% quantile.".format(df.index.size))
        df = self._attachPeakIDtoEntries(df)
        
        df.to_csv(pathToFile, sep="\t")
        print("Info :: Saving low distance interactions in result folder.")
        return df

    def _plotComplexProfiles(self,complexDf,outputFolder,name):
        """
        Creates line charts as pdf for each profile.

        Parameters
        ----------
        complexDf : pd.DataFrame
            asd
        
        outputFolder : string
            Path to folder, will be created if it does not exist.
        
        name : string
            Name of complex.

        Returns
        -------
        None

        """
        if self.params["plotComplexProfiles"]:
            toProfiles = self._makeFolder(outputFolder,"complexProfiles")
            pathToFolder = self._makeFolder(toProfiles,str(name))
            
            x = np.arange(0,len(self.X.columns))
            for c in complexDf["Cluster Labels"].unique():
            
                if c != -1:
                    fig, ax = plt.subplots(nrows=2,ncols=1)
                    entries = complexDf.loc[complexDf["Cluster Labels"] == c,:].index
                    lineColors = sns.color_palette("Blues",desat=0.8,n_colors=entries.size)
                    for n,e in enumerate(entries):
                        uniprotID = e.split("_")[0]
                        if uniprotID in self.Signals[self.analysisName]:
                            y = self.Signals[self.analysisName][uniprotID].Y
                            normY = y / np.nanmax(y)
                            ax[0].plot(x,y,linestyle="-",linewidth=1, label=e, color = lineColors[n])
                            ax[1].plot(x,normY,linestyle="-",linewidth=1, label=e, color = lineColors[n])
                    plt.legend(prop={'size': 5}) 

                    plt.savefig(os.path.join(pathToFolder,"{}_n{}.pdf".format(c,len(entries))))
                    plt.close()
            
            
    def _attachPeakIDtoEntries(self,predInts):
        ""
        if not "apexPeakDist" in predInts.columns:
            return predInts
        peakIds = [peakID.split("_") for peakID in predInts["apexPeakDist"]]
        predInts["E1p"], predInts["E2p"] = zip(*[("{}_{}".format(E1,peakIds[n][0]),"{}_{}".format(E2,peakIds[n][1])) for n,(E1,E2) in enumerate(zip(predInts["E1"],predInts["E2"]))])
        return predInts

    def _makeFolder(self,*args):
        ""
        pathToFolder = os.path.join(*args)
        if not os.path.exists(pathToFolder):
            os.mkdir(pathToFolder)
        return pathToFolder

    def _createTxtFile(self,pathToFile,headers):
        ""
        
        with open(pathToFile,"w+") as f:
            f.write("\t".join(headers))

    def _makeTmpFolder(self, n = 0):
        """
        Creates temporary fodler.


        Parameters
        ----------
        n : int

        Returns
        -------
        pathToTmp : str
            ansolute path to tmp/anlysis name folder.

        """
        if not hasattr(self,"analysisNames"):
            self.analysisNames = []
        if self.params["analysisName"] is None:
            analysisName = self._randomStr(50)
        elif isinstance(self.params["analysisName"],list) and n < len(self.params["analysisName"]):
            analysisName = self.params["analysisName"][n]
        else:
            analysisName = str(self.params["analysisName"])

        pathToTmp = os.path.join(".","tmp")
        
        if not os.path.exists(pathToTmp):
            os.mkdir(pathToTmp)

        
        self.currentAnalysisName = analysisName
        #store analysis name
        if analysisName not in self.analysisNames:
            self.analysisNames.append(analysisName)

        date = datetime.today().strftime('%Y-%m-%d')
        runName = self.params["runName"] if self.params["runName"] is not None else self._randomStr(3)
        self.params["pathToComb"] = self._makeFolder(pathToTmp,"{}_{}_{}runs".format(date,runName,len(self.params["analysisName"])))
        print("Folder create in which combined results will be saved: " + self.params["pathToComb"])
        pathToTmpFolder = os.path.join(self.params["pathToComb"],analysisName)
        if os.path.exists(pathToTmpFolder):
            print("Info :: Path to tmp folder exsists")
            if self.params["restartAnalysis"]:
                print("Warning :: Argument restartAnalysis was set to True .. cleaning folder.")
                #to do - shift to extra fn
                for root, dirs, files in os.walk(pathToTmpFolder):
                    for f in files:
                        os.unlink(os.path.join(root, f))
                    for d in dirs:
                        shutil.rmtree(os.path.join(root, d))
            else:
            
                print("Info :: Will take files from there, if they exist")
            
                return pathToTmpFolder
        
        try:
            self._makeFolder(pathToTmpFolder)
            print("Info :: Tmp folder created -- ",analysisName)
            self._makeFolder(pathToTmpFolder,"chunks")
            print("Info :: Chunks folder created/checked")
            self._makeFolder(pathToTmpFolder,"result")
            print("Info :: Result folder created/checked")
            self._makeFolder(pathToTmpFolder,"result","alignments")
            print("Info :: Alignment folder created/checked")
            self._makeFolder(pathToTmpFolder,"result","modelPlots")
            print("Info :: Result/modelPlots folder created/checked. In this folder, all model plots will be saved.")
            self._createTxtFile(pathToFile = os.path.join(pathToTmpFolder,"runTimes.txt"),headers = ["Date","Time","Step","Comment"])

            return pathToTmpFolder
        except OSError as e:
            print(e)
            raise OSError("Could not create tmp folder due to OS Error")
            

    def run(self,X, maxValueToOne = False):
        """
        Runs the ComplexFinder Script.

        Parameters
        ----------
        X : str, list, pd.DataFrame 

        Returns
        -------
        pathToTmp : str
            ansolute path to tmp/anlysis name folder.

        """
        self.allSamplesFound = False
        
        global entriesInChunks

        if isinstance(X,list) and all(isinstance(x,pd.DataFrame) for x in X):

            print("Multiple dataset detected - each one will be analysed separetely")
            if self.params["analysisName"] is None or not isinstance(self.params["analysisName"],list) or len(self.params["analysisName"]) != len(X):
                self.params["analysisName"] = [self._randomStr(10) for n in range(len(X))] #create random analysisNames
                print("Info :: 'anylsisName' did not match X shape. Created random strings per dataframe.")

        elif isinstance(X,str) and os.path.exists(X):

            loadFiles = [f for f in os.listdir(X) if f.endswith(".txt")]
            self.params["analysisName"] = loadFiles
            X = [pd.read_csv(os.path.join(X,fileName), sep="\t") for fileName in loadFiles]#nrows=300, 
            
            if maxValueToOne:
                maxValues = pd.concat([x.max(axis=1) for x in X], axis=1).max(axis=1)
                normValueDict = dict([(X[0][self.params["idColumn"]].values[n],maxValue) for n,maxValue in enumerate(maxValues.values)])
                self.params["normValueDict"] = normValueDict

        elif isinstance(X,pd.DataFrame):
            X = [X]
            self.params["analysisName"] = [self._randomStr(10)]
        else:
            ValueError("X must be either a string, a list of pandas data frames or pandas data frame itself.")
        
        self.params["pathToTmp"] = {}
        statColumns = ["nInteractions ({})".format(self.params["interactionProbabCutoff"]),"nPositiveInteractions","OOB_Score","ROC_Curve_AUC","Metrices","Classifier","ClassifierParams"]
        self.stats = pd.DataFrame(index = self.params["analysisName"],columns = statColumns)
        
        
        for n,X in enumerate(X):
            
           # entriesInChunks.clear()

            pathToTmpFolder = self._makeTmpFolder(n)
            self.params["pathToTmp"][self.currentAnalysisName] = pathToTmpFolder
            print("------------------------")
            print("--"+self.currentAnalysisName+"--")
            print("--------Started---------")
            print("--Signal Processing  &--")
            print("------Peak Fitting------")
            print("------------------------")
            #set current analysis Name
            self.analysisName = self.analysisNames[n]

            if pathToTmpFolder is not None:
                
                #loading data
                self._load(X)
                #self._checkGroups()
                self._findPeaks(self.params["n_jobs"])
                self._collectRSquaredAndFitDetails()
                self._saveSignalFitStatistics()
       
        

        self._combinePeakResults()
        self._saveSignals()
        self._createSignalChunks()
        
        
        print("Peak modeling done. Starting with distance calculations and predictions (if enabled)..")
        for n,X in enumerate(X):
            if n < len(self.params["analysisName"]):
                self.currentAnalysisName = self.params["analysisName"][n]
                print(self.currentAnalysisName," :: Starting distance calculations.")

                self._calculateDistance()
                self._loadReferenceDB()
                self._addMetricesToDB()
                self._trainPredictor()
                predInts = self._predictInteractions()
            #predInts = None
                self._clusterInteractions(predInts)

        
        self.stats.to_csv(os.path.join(self.params["pathToComb"],"statistics.txt"),sep="\t")
        self._combineInteractionsAndClusters()
    def _combinePredictedInteractions(self, pathToComb):
        ""
        print("Info :: Combining interactions of runs.")
        preditctedInteractions = []
        for analysisName in self.params["analysisName"]:

            pathToResults = os.path.join(self.params["pathToTmp"][analysisName],"result")
            pathToPrediction = os.path.join(pathToResults,"predictedInteractions{}_{}.txt".format(self.params["metricesForPrediction"],self.params["classifierClass"]))

            if os.path.exists(pathToPrediction):
                preditctedInteractions.append(pd.read_csv(pathToPrediction,sep="\t").set_index(["E1E2","E1","E2"],drop=False))
            else:
                raise ValueError("Warning :: PredictedInteractions not found. " + str(pathToPrediction))

        for n,df in enumerate(preditctedInteractions):

            analysisName = self.params["analysisName"][n]
            if n == 0:
                combResults = df 
                combResults.columns = ["{}_({})".format(colName,analysisName) for colName in df.columns]
                combResults[analysisName]  = pd.Series(["+"]*df.index.size, index = df.index)
                
            else:

                df.columns = ["{}_({})".format(colName,analysisName) for colName in df.columns] 
                #columnNames = [colName for colName in df.columns if colName] # we have them already from n = 0
                df[analysisName]  = pd.Series(["+"]*df.index.size, index = df.index)
               # combResults["validSignal({})".format(analysisName)]  = df[["E1_({})".format(analysisName),"E2_({})".format(analysisName)]].apply(lambda x: all(e in self.Signals[analysisName] and self.Signals[analysisName][e].valid for e in x.values),axis=1)
                combResults = combResults.join(df, how="outer")
        combResults = combResults.reset_index()
        for analysisName in self.params["analysisName"]:
            

            combResults["validSignal({})".format(analysisName)]  = combResults[["E1","E2"]].apply(lambda x: all(e in self.Signals[analysisName] and self.Signals[analysisName][e].valid for e in x.values),axis=1)
        combResults["Valid in"]  = combResults[["validSignal({})".format(analysisName) for analysisName in self.params["analysisName"]]].sum(axis=1)
        
        detectedColumn = [analysisName for analysisName in self.params["analysisName"]]
        boolIdx = combResults[detectedColumn] == "+"
        combResults["Detected in"] = np.sum(boolIdx,axis=1)
        combResults.sort_values(by="Detected in", ascending = False, inplace = True)
        combResults.to_csv(os.path.join(pathToComb,"combinedInteractions.txt"),sep="\t",index=True)
            


    def _combineInteractionsAndClusters(self):
        ""
        pathToComb = self.params["pathToComb"]
        self._combinePredictedInteractions(pathToComb)

        # for analysisName in self.params["analysisName"]:

        #     pathToTmp = self.params["pathToTmp"][analysisName]
        #     pathToResult = os.path.join(pathToTmp,"result")
        #     embeddingPath = os.path.join(pathToResult,"UMAP_Embedding.txt")
        #     if os.path.exists(embeddingPath):
        #         embedd = pd.read_csv(embeddingPath,sep="\t")
                

    def _saveSignalFitStatistics(self):
        ""
        pathToTxt = os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"result","fitStatistic.txt")
        with open(pathToTxt , "w") as f:
            f.write("\t".join(["Invalid signals",str(np.sum([signal.valid for signal in self.Signals[self.currentAnalysisName].values()]))])+"\n")
            f.write("\t".join(["Valid signal models",str(np.sum([not signal.validModel for signal in self.Signals[self.currentAnalysisName].values()]))])+"\n")
            f.write("\t".join(["Max signal peaks",str(np.sum([signal.maxNumbPeaksUsed for signal in self.Signals[self.currentAnalysisName].values()]))])+"\n")

    def _checkAlignment(self,data):
        ""

        data = data.dropna() 

        centerColumns = [colName for colName in data.columns if colName.startswith("auc")]
        data[centerColumns].corr()
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.scatter(data[centerColumns[0]],data[centerColumns[1]])
        plt.show()

    def _alignProfiles(self,fittedPeaksData):
        ""
        alignMethod = self.params["alignMethod"]
        
        
        if len(fittedPeaksData) > 1 and alignMethod in alignModels and os.path.exists(self.params["pathToComb"]):
            
            alignResults = OrderedDict([(analysisName,[]) for analysisName in self.analysisNames])
            fittedModels = dict()
            removedDuplicates = [X.loc[~X.duplicated(subset=["Key"],keep=False)] for X in fittedPeaksData]
            preparedData = []
            for n,dataFrame in enumerate(removedDuplicates):
                dataFrame.columns = ["{}_{}".format(colName,self.analysisNames[n]) if colName != "Key" else colName for colName in dataFrame.columns ]
                dataFrame = dataFrame.set_index("Key")
                preparedData.append(dataFrame)
            #join data frames
            joinedDataFrame = preparedData[0].join(preparedData[1:],how="outer")
            if joinedDataFrame .index.size < 30:
                print("Less than 30 data profiles with single peaks found. Aborting alignment")
                return fittedPeaksData
            #use linear regression or lowess
           
            for comb in combinations(self.analysisNames,2):
                c1, c2  = comb
                columnHeaders = ["Center_{}".format(c1),"Center_{}".format(c2)]
                data = joinedDataFrame.dropna(subset=columnHeaders)[columnHeaders]
                absDiff = np.abs(data[columnHeaders[0]] - data[columnHeaders[1]])
                pd.DataFrame(data).to_csv("alignedPeaks.txt",sep="\t")
                boolIdx = absDiff > 5 #remove everything with a higher difference of 5.
                data = data.loc[~boolIdx]
                
                nRows = data.index.size
                X, Y = data[[columnHeaders[0]]].values, data[[columnHeaders[1]]].values
                
                model = alignModels["LinearRegression"](**alignModelsParams["LinearRegression"]).fit(X,Y)
                lnSpace = np.linspace(np.min(data.values),np.max(data.values)).reshape(-1,1) #get min / max values
                Yplot = model.predict(lnSpace)
                #store R2
                R2 = model.score(X,Y) 
                alignResults[c1].append(R2)
                alignResults[c2].append(R2)

                #save model
                fittedModels[comb] = {"model":model,"R2":R2}
                #plot alignment
                f = plt.figure()
                ax = f.add_subplot(111)
                ax.scatter(joinedDataFrame["Center_{}".format(c1)],joinedDataFrame["Center_{}".format(c2)])
                ax.plot(lnSpace,Yplot)
                plt.savefig(os.path.join(self.params["pathToComb"],"{}.pdf".format(comb)))
                #ax.plot()
                

                #save alignment 
                o = pd.DataFrame(lnSpace)
                o["y"] = Yplot
                o.to_csv("curve_{}.txt".format(alignMethod),sep="\t")
                


            #find run with highest R2s - this run will be used to align all other
            maxR2SumRun = max(alignResults, key=lambda key: sum(alignResults[key]))
            print("The run that will be used as a reference (highest sum of R2 for all fits) is {}".format(maxR2SumRun))
            diffs = pd.DataFrame()
            #calculate difference to reference run
            for analysisName in self.analysisNames:
                if analysisName != maxR2SumRun:
                    columnHeaders = ["Center_{}".format(maxR2SumRun),"Center_{}".format(analysisName)]
                    data = joinedDataFrame.dropna(subset=columnHeaders)[columnHeaders]
                    absDiff = data[columnHeaders[0]] - data[columnHeaders[1]]
                    diffs[analysisName] = absDiff
                    diffs["c({})".format(analysisName)] = data[columnHeaders[0]]

            fig, ax = plt.subplots(len(self.analysisNames))
            for n,analysisName in enumerate(self.analysisNames):
                if analysisName in diffs.columns:
                    data = joinedDataFrame.dropna(subset=columnHeaders)[columnHeaders]
                    

                    diffs = diffs.sort_values(by="c({})".format(analysisName))
                    boolIdx = np.abs(diffs[analysisName]) < 3
                    X = diffs.loc[boolIdx,["c({})".format(analysisName)]].values
                    Y = diffs.loc[boolIdx,[analysisName]].values
                    ax[n].plot(X,Y,color="darkgrey")

                    model = alignModels[alignMethod](**alignModelsParams[alignMethod]).fit(X,Y)
                    lnSpace = np.linspace(np.min(data.values),np.max(data.values)).reshape(-1,1) #get min / max values
                    Yplot = model.predict(lnSpace)
                    ax[n].plot(lnSpace,Yplot,color="red")

            plt.savefig(os.path.join(self.params["pathToComb"],"centerDiff.pdf"))
        return fittedPeaksData


    def _combinePeakResults(self):
        ""
        alignRuns = self.params["alignRuns"]
        if self.params["peakModel"] == "SkewedGaussianModel":
            columnsForPeakFit = ["Amplitude","Center", "Sigma", "Gamma", "fwhm", "height","auc","ID", "relAUC"]
        else:
            columnsForPeakFit = ["Amplitude","Center", "Sigma", "fwhm", "height","auc","ID", "relAUC"]
        
        print("Info :: Combining peak results.")
        print(" : ".join(self.params["analysisName"]))
        if len(self.params["analysisName"]) == 1:
            print("Info :: Single run analysed. Will continue to create peak-centric output. No alignment performed.")
            alignRuns = False
        fittedPeaksData = []
        suffixedColumns = []
        for analysisName in self.analysisNames:
            suffixedColumns.extend(["{}_{}".format(colName,analysisName) for colName in columnsForPeakFit])
            tmpFolder = os.path.join(".","tmp",analysisName)
            resultsFolder = os.path.join(tmpFolder,"result")
            fittedPeaks = os.path.join(resultsFolder,"fittedPeaks.txt")
            if os.path.exists(fittedPeaks):
                data = pd.read_csv(fittedPeaks,sep="\t")
                fittedPeaksData.append(data)
        if alignRuns:
            print("Info :: Aligning runs started.")
            fittedPeaksData = self._alignProfiles(fittedPeaksData)

        uniqueKeys = np.unique(np.concatenate([x["Key"].unique().flatten() for x in fittedPeaksData]))
        print("{} unique keys detected".format(uniqueKeys.size))
        print("Combining peaks using max peak center diff of {}".format(self.params["maxPeakCenterDifference"]))
        #combinedData = pd.DataFrame(columns=["Key","ID","PeakCenter"])
        txtOutput = os.path.join(self.params["pathToComb"],"CombinedModelPeakResults.txt")
        if not os.path.exists(txtOutput):
            concatDataFrames = []
            for uniqueKey in uniqueKeys:
                boolIdxs = [x["Key"] == uniqueKey for x in fittedPeaksData]

                filteredData = [fittedPeaksData[n].loc[boolIdx] for n,boolIdx in enumerate(boolIdxs)]
                d = pd.DataFrame(columns=["Key"])
                for n,df in enumerate(filteredData):
                    
                    if df.empty:
                        
                        continue
                    if n == 0:
                        df.columns = [colName if colName == "Key" else "{}_{}".format(colName,self.analysisNames[n]) for colName in df.columns.values.tolist()]
                        d = d.append(df)
                    else:
                        meanCenters = d[[colName for colName in d.columns if "Center" in colName]].mean(axis=1)
                        idx = meanCenters.index
                        if idx.size == 0:
                            df.columns = [colName if colName == "Key" else "{}_{}".format(colName,self.analysisNames[n]) for colName in df.columns.values.tolist()]
                            d = d.append(df)
                            continue
                        newIdx = []
                        for m,peakCenter in enumerate(df["Center"]):
                            peaksInRange = meanCenters.between(peakCenter-self.params["maxPeakCenterDifference"],peakCenter+self.params["maxPeakCenterDifference"])
                            if not np.any(peaksInRange):
                                newIdx.append(np.max(idx.values+1+m))
                            else:
                                newIdx.append(idx[peaksInRange].values[0])
                        
                        df.index = newIdx
                        df.columns = [colName if colName == "Key" else "{}_{}".format(colName,self.analysisNames[n]) for colName in df.columns.values.tolist()]
                        targetColumns = [colName for colName in df.columns if self.analysisNames[n] in colName]
                        d = pd.merge(d,df[targetColumns], how="outer",left_index=True,right_index=True)
                       
                        d["Key"] = [uniqueKey] * d.index.size

           

                concatDataFrames.append(d)
            if len(concatDataFrames) == 0 or all(X.empty for X in concatDataFrames):
                print("Warning - no matching keys found. Aborting.")
                return
            data = pd.concat(concatDataFrames,axis=0, ignore_index=True)
            #peakPropColumns = [col for col in data.columns if col.split("_")[0] in columnsForPeakFit]
            
            data.sort_index(axis=1, inplace=True)

            #perform t-test
            grouping = self.params["grouping"]
            groups = list(self.params["grouping"].keys())            
            groupComps =  combinations(groups,2)
            print(groupComps)
            resultColumnNames = ["t({})_({})".format(group1,group0) for group0,group1 in groupComps]
            for n,(group0, group1) in enumerate(groupComps):
                sampleNames1 = grouping[group0]
                sampleNames2 = grouping[group1]
                columnNames1 = ["auc_{}".format(sanmpleName) for sanmpleName in sampleNames1]
                columnNames2 = ["auc_{}".format(sanmpleName) for sanmpleName in sampleNames2]
                X = data[columnNames1].values
                Y = data[columnNames2].values
               
                t, p = ttest_ind(X,Y,axis=1,nan_policy="omit",equal_var = True)
                
                data["-log10-p-value:({})".format(resultColumnNames[n])] = np.log10(p) * (-1)
                data["T-stat:({})".format(resultColumnNames[n])] = t
                data["diff:({})".format(resultColumnNames[n])] = np.nanmean(Y,axis=1) - np.nanmean(X,axis=1) 


            data.to_csv(txtOutput,sep="\t")
        else:
            data = pd.read_csv(txtOutput,sep="\t")

       # self._checkAlignment(data)
           # print(filteredData)
            
        


if __name__ == "__main__":

   # X = pd.read_csv("../example-data/HeuselEtAlAebersoldLab.txt", 
    #                sep="\t")
    # X = pd.read_csv("../example-data/TMEM70/WT_01.txt", 
    #                 #nrows = 500,
    #                 sep="\t")
    # boolIdx = X["Uniprot ID"].str.contains("NDUF")
    # X = X.loc[boolIdx]

    # Y = pd.read_csv("../example-data/TMEM70/WT_02.txt", 
    #                 #nrows = 500,
    #                 sep="\t")
    # boolIdx = Y["Uniprot ID"].str.contains("NDUF")
    # Y = Y.loc[boolIdx]

 #X = X.set_index("Uniprot ID")
  #  X

   # ComplexFinder(indexIsID=False,analysisName="500restoreTry",classiferGridSearch=param_grid, classifierClass="SVM").run(X)""
    ComplexFinder(restartAnalysis = False,recalculateDistance  = False, runName = "D3_5xDecoy", indexIsID=False,classifierClass="random forest",retrainClassifier=False,interactionProbabCutoff = 0.66,removeSingleDataPointPeaks=True, smoothSignal=True).run("../example-data/TMEM70/")#[X,Y])





    
