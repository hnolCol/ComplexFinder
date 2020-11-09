


import os 
import pandas as pd
import numpy as np 
from collections import OrderedDict
from modules.Distance import DistanceCalculator
from modules.Signal import Signal
from modules.Database import Database
from modules.Predictor import Classifier, ComplexBuilder
from joblib import Parallel, delayed, dump, load
import gc 
import string
import random
import time
import pickle
from multiprocessing import Pool
from joblib import wrap_non_picklable_objects
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, homogeneity_score, v_measure_score
from sklearn.model_selection import ParameterGrid
import joblib

filePath = os.path.dirname(os.path.realpath(__file__)) 
pathToTmp = os.path.join(filePath,"tmp")

RF_GRID_SEARCH = {#'bootstrap': [True, False],
 'max_depth': [70],#30,50,
 'max_features': ['sqrt','auto'],
 'min_samples_leaf': [2, 3, 4],
 'min_samples_split': [2, 3, 4],
 'n_estimators': [600]}

OPTICS_PARAM_GRID = {"min_samples":[2,3,5,8], "max_eps": [np.inf,2,1,0.9,0.8], "xi": np.linspace(0,0.3,num=30), "cluster_method" : ["xi"]}
AGGLO_PARAM_GRID = {"n_clusters":[None,115,110,105,100,90,95],"distance_threshold":[None,0.5,0.4,0.2,0.1,0.05,0.01], "linkage":["complete","single","average"]}
AFF_PRO_PARAM = {"damping":np.linspace(0.5,1,num=50)}
HDBSCAN_PROPS = {"min_cluster_size":[2,3,4],"min_samples":[1,2,3,4]}
CLUSTER_PARAMS = {"OPTICS":OPTICS_PARAM_GRID,"AGGLOMERATIVE_CLUSTERING":AGGLO_PARAM_GRID,"AFFINITY_PROPAGATION":AFF_PRO_PARAM,"HDBSCAN":HDBSCAN_PROPS}

param_grid = {'C': [1, 10, 100, 1000], 'kernel': ['linear','rbf','poly'], 'gamma': [0.01,0.1,1,2,3,4,5]}

entriesInChunks = dict() 


def _calculateDistanceP(pathToFile):
    
    with open(pathToFile,"rb") as f:
        chunkItems = pickle.load(f)
    exampleItem = chunkItems[0]
    data = np.concatenate([DistanceCalculator(**c).calculateMetrices() for c in chunkItems],axis=0)
    np.save(os.path.join(exampleItem["pathToTmp"],"chunks",exampleItem["chunkName"]),data)        
    return (exampleItem["chunkName"],[''.join(sorted(row.tolist())) for row in data[:,[0,1]]])
        

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]



class ComplexFinder(object):

    def __init__(self,
                indexIsID = True,
                plotSignalProfiles = True,
                removeSingleDataPointPeaks = True,
                #savePeakModels = True,
                maxPeaksPerSignal = 15,
                n_jobs = 6,
                kFold = 5,
                analysisName = None,
                idColumn = "Uniprot ID",
                databaseName="CORUM",
                peakModel = "GaussianModel",#"LorentzianModel",
                imputeNaN = True,
                classifierClass = "random_forest",
                columnSuffixes = ["KO_01","WT_01"],
                grouping = {"KO":["KO_01"],"WT":["WT_01"]},
                analysisMode = "separate", #replicate
                retrainClassifier = False,
                interactionProbabCutoff = 0.7,
                normValueDict = {},
                metrices = ["apex","euclidean","pearson","p_pearson","spearman","max_location"],
                classiferGridSearch = RF_GRID_SEARCH):
        ""

        self.params = {
            "indexIsID" : indexIsID,
            "idColumn" : idColumn,
            "n_jobs" : n_jobs,
            "kFold" : kFold,
            "analysisName" : analysisName,
            "databaseName" : databaseName,
            "imputeNaN" : imputeNaN,
            "metrices" : metrices,
            "peakModel" : peakModel,
            "classifierClass" : classifierClass,
            "retrainClassifier" : retrainClassifier,
            "interactionProbabCutoff":interactionProbabCutoff,
            "maxPeaksPerSignal" : maxPeaksPerSignal,
            "classiferGridSearch" : classiferGridSearch,
            "plotSignalProfiles" : plotSignalProfiles,
            "savePeakModels" : True, #must be true to process pipeline, remove from class arguments.
            "removeSingleDataPointPeaks" : removeSingleDataPointPeaks,
            "columnSuffixes" : columnSuffixes,
            "grouping" : grouping,
            "analysisMode" : analysisMode,
            "normValueDict" : normValueDict,
            }
        print("\n" + str(self.params))
    

    def _load(self, X):
        "Load data"
        if isinstance(X, pd.DataFrame):
            
            self.X = X

            if not self.params["indexIsID"]:
                print("checking for duplicates")
                dupRemoved = self.X.drop_duplicates(subset=[self.params["idColumn"]])
                if dupRemoved.index.size < self.X.index.size:
                    print("#### Duplicates detected.")
                    print("file contained duplicate ids... removed: {}".format(self.X.index.size-dupRemoved.index.size))
                    self.X = dupRemoved
                np.save(os.path.join(self.params["pathToTmp"],"source"),self.X.values)

                self.X = self.X.set_index(self.params["idColumn"])
                self.X = self.X.astype(np.float32)

        else:

            raise ValueError("X must be a pandas data frame")


    def _addModelToSignals(self,signalModels):
        "Asigns the model atributes to the signal."
        for fitModel in signalModels:
            modelID = fitModel["id"]
            if len(fitModel) == 1:
                del self.Signals[modelID]
            if modelID in self.Signals:
                for k,v in fitModel.items():
                    if k != 'id':
                        setattr(self.Signals[modelID],k,v)
                
    def _checkGroups(self):
        "Checks grouping. For comparision of multiple co-elution data sets."
        if self.params["columnSuffixes"] is None:
            print("No grouping found, assuming the whole dataset is a single co-elution profile..")
            return
        elif isinstance(self.params["columnSuffixes"],list):
            if len(self.params["columnSuffixes"]) == 1:
                print("single suffix found, assuming whole dataset is a single co-elution profile ..")
                return
            else:
                print("{} different column suffixes found".format(len(self.params["columnSuffixes"])))
                print("Checking grouping now...")

        if isinstance(self.params["grouping"],dict):
            if len(self.params["grouping"]) == 0:
                raise ValueError("Parmaeter 'columnSuffixes' was provided, but grouping was not defined. Example for grouping : {'KO':['KO_01','KO_02'], 'WT':['WT_01','WT_02'] } Aborting.. ")
            else:
                combinedValues = sum(self.params["grouping"].values(), [])
                if all(x in combinedValues for x in self.params["columnSuffixes"]):
                    print("Grouping checked..\nAll columnSuffixes found in grouping.")
                else:
                    raise ValueError("Could not find all column suffixes.. Aborting ..")
                

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
                                                savePlots = self.params["plotSignalProfiles"],
                                                savePeakModels = self.params["savePeakModels"],
                                                maxPeaks=self.params["maxPeaksPerSignal"],
                                                metrices=self.params["metrices"],
                                                pathToTmp = self.params["pathToTmp"],
                                                normalizationValue = self.params["normValueDict"][entryID] if entryID in self.params["normValueDict"] else None,
                                                removeSingleDataPointPeaks = self.params["removeSingleDataPointPeaks"],
                                                analysisName = self.currentAnalysisName) 

            
            t1 = time.time()
            print("\n\nStarting Signal modelling .. (n_jobs = {})".format(n_jobs))
            
            fittedModels = Parallel(n_jobs=n_jobs, verbose=1)(delayed(Signal.fitModel)() for Signal in self.Signals.values())
            
            self._addModelToSignals(fittedModels)
            
            print("Peak fitting done time : {} secs".format(round((time.time()-t1))))
            print("Each feature's fitting as pdf and txt is stored in model plots (if savePeakModels and plotSignalProfiles was set to true)")
        
        dump(self.Signals,pathToSignal)

    def _calculateDistance(self):
        """
        Calculates distances between Signals by creating chunks of signals. Each chunk is analysed separately.

        parameters

        returns
            None
        """
        global entriesInChunks
        X = list(self.Signals.values())
        print("\nStarting Distance Calculation ...")
        t1 = time.time()

        for n,Signal in enumerate(self.Signals.values()):
            setattr(Signal,"otherSignals", X[n:])

        chunks = self._createSignalChunks()

        
        if all(os.path.exists(x.replace(".pkl",".npy")) for x in chunks):
            print("all chunks found for distance calculation")

            with open(os.path.join(self.params["pathToTmp"], "entriesInChunk.pkl"),"rb") as f:
                       entriesInChunks = pickle.load(f)

        else:

            chunkItems = Parallel(n_jobs=self.params["n_jobs"], verbose=10)(delayed(_calculateDistanceP)(c) for c in chunks)
            
            for k,v in chunkItems:
                entriesInChunks[k] = v 
            
            with open(os.path.join(self.params["pathToTmp"], "entriesInChunk.pkl"),"wb") as f:
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
        
        nSignals = len(self.Signals)
        signals = list(self.Signals.values())

        c = []

        for n,chunk in enumerate(chunks(signals,chunkSize)):
            pathToChunk = os.path.join(self.params["pathToTmp"],"chunks",str(n)+".pkl")
            if not os.path.exists(pathToChunk):
                chunkItems =  [
                    {"ID":str(signal.ID),
                    "chunkName":str(n),
                    "Y":np.array(signal.Y),
                    "ownPeaks" : signal._collectPeakResults(),
                    "otherSignalPeaks" : [s._collectPeakResults() for s in signal.otherSignals],
                    "E2":[str(s.ID) for s in signal.otherSignals],
                    "metrices":self.params["metrices"],
                    "pathToTmp":self.params["pathToTmp"]} for signal in chunk]

                with open(pathToChunk,"wb") as f:
                    pickle.dump(chunkItems,f)

            gc.collect()
            c.append(pathToChunk)
        
        return c



    def _collectRSquaredAndFitDetails(self):
        """
        Data are collected from txt files in the modelPlots folder. 
            
        """
        if not self.params["savePeakModels"]:
            print("!! Warning !!")

        rSqured = [] 
        entryList = []
        pathToPlotFolder = os.path.join(self.params["pathToTmp"],"modelPlots")
        resultFolder = os.path.join(self.params["pathToTmp"],"result")
        if not os.path.exists(resultFolder):
            os.mkdir(resultFolder)
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
        df.to_csv(os.path.join(resultFolder,"fittedPeaks.txt"),sep="\t")
        pd.DataFrame(collectNumbPeaks).to_csv(os.path.join(resultFolder,"nPeaks.txt"),sep="\t")


    def _loadReferenceDB(self):
        ""
        print("Load positive set from data base")
        self.DB = Database()
        self.DB.pariwiseProteinInteractions("subunits(UniProt IDs)")
        entryList = []
        for x in self.X.index:
            entryList.extend(x.split(";"))
        self.DB.filterDBByEntryList(entryList)
        #add decoy to db
        self.DB.addDecoy()


    def _addMetricesToDB(self):
        self.DB.matchMetrices(self.params["pathToTmp"],entriesInChunks,self.params["metrices"])


    def _trainPredictor(self):
        ""
        #metricColumns = [col for col in self.DB.df.columns if any(x in col for x in self.params["metrices"])]

        folderToResults = os.path.join(self.params["pathToTmp"],"result")
        classifierFileName = os.path.join(self.params["pathToTmp"],'trainedClassifier.sav')

        pathToPrediction = os.path.join(folderToResults,"predictedInteractions.txt")
        if not self.params["retrainClassifier"] and os.path.exists(pathToPrediction) and os.path.exists(classifierFileName):
            print("prediction was done already... loading file")
            self.classifier = joblib.load(classifierFileName)
            return

        if not os.path.exists(folderToResults):
            os.mkdir(folderToResults)
        
        totalColumns = self.params["metrices"] + ['Class']
        data = self.DB.dfMetrices[totalColumns].dropna(subset=self.params["metrices"])
        self.Y = data['Class'].values
        X = data.loc[:,self.params["metrices"]].values

        self.classifier = Classifier(
            classifierClass = self.params["classifierClass"],
            n_jobs=self.params['n_jobs'], 
            gridSearch = self.params["classiferGridSearch"])

        probabilites = self.classifier.fit(X,self.Y,kFold=self.params["kFold"],pathToResults=folderToResults)
        
        boolClass = probabilites > 0.5

        data["PredictionClass"] = probabilites

        pd.DataFrame(
            classification_report(self.Y,boolClass,digits=3,output_dict=True)).to_csv(os.path.join(self.params["pathToTmp"],"PredictorSummary.txt"),sep="\t", index=False)

        data.to_csv(os.path.join(self.params["pathToTmp"],"DBpred.txt"),sep="\t", index=False)
        
        self._plotFeatureImportance(folderToResults)

        joblib.dump(self.classifier, classifierFileName)
        
        print("DB prediction saved - DBpred.txt :: Classifier pickled and save 'trainedClassifier.sav'")


    def _loadPairs(self):
        ""
        chunks = [f for f in os.listdir(os.path.join(self.params["pathToTmp"],"chunks")) if f.endswith(".npy") and f != "source.npy"]
        pathToPredict = os.path.join(self.params["pathToTmp"],"predictions")
        if not os.path.exists(pathToPredict):
            os.mkdir(pathToPredict)
        print("\nPrediction started...")
        for chunk in chunks:

            X = np.load(os.path.join(self.params["pathToTmp"],"chunks",chunk),allow_pickle=True)
            
            yield (X,os.path.join(pathToPredict,chunk))

            
        
    def _chunkPrediction(self,pathToChunk,classifier,nMetrices,probCutoff):
        """
        Predicts for each chunk the proability for positive interactions.

        Parameter
            pathToChunk -   string. 
            classifier  -   classifier object.
            nMetrices   -   int. number if metrices used. (since chunks are simple numpy arrays, no column headers are loaded)
            probCutoff  -   float. Probability cutoff.

        returns 
            chunks with appended probability.
        """
        X =  np.load(pathToChunk,allow_pickle=True)
        boolSelfIntIdx = X[:,0] != X[:,1] 
        X = X[boolSelfIntIdx]
        classProba = classifier.predict(X[:,[n+3 for n in range(nMetrices)]])

        boolPredIdx = classProba >= probCutoff
        boolIdx = np.sum(boolPredIdx,axis=1) > 0
        predX = np.append(X[:,2],classProba.reshape(X.shape[0],-1),axis=1)
        np.save(
                file = pathToChunk,
                arr = predX)

        return predX



    def _predictInteractions(self):
        ""
        
        folderToOutput = os.path.join(self.params["pathToTmp"],"result")
        pathToPrediction = os.path.join(folderToOutput,"predictedInteractions.txt")
        if not self.params["retrainClassifier"] and os.path.exists(pathToPrediction):
            return pd.read_csv(pathToPrediction, sep="\t")
       # del self.Signals
        gc.collect()   
        #create prob columns of k fold 
        pColumns = ["Prob_{}".format(n) for n in range(self.params["kFold"])]
        dfColumns = ["E1","E2","E1E2","apexPeakDist"] + self.params["metrices"] + pColumns + ["In DB"] 

        if not os.path.exists(folderToOutput):
            os.mkdir(folderToOutput)

        predInteractions = None
       # predInteractions = Parallel(n_jobs=self.params["n_jobs"], verbose=5)(delayed(self._chunkPrediction)(
        #                                                                **c) for c in self._loadPairs())
                                                                    
       
        for X,pathToChunk in self._loadPairs():
            boolSelfIntIdx = X[:,0] == X[:,1] 
            print("Current Chunk: ",pathToChunk)
            X = X[boolSelfIntIdx == False]
            #first two rows E1 E2, and E1E2, remove before predict
            classProba = self.classifier.predict(X[:,[n+4 for n in range(len(self.params["metrices"]))]])
 
            if classProba is None:
                continue
            predX = np.append(X,classProba.reshape(X.shape[0],-1),axis=1)
            boolPredIdx = classProba >= self.params["interactionProbabCutoff"]
            boolIdx = np.sum(boolPredIdx,axis=1) == self.params["kFold"]
            

            if predInteractions is None:
                predInteractions = predX[boolIdx,:]
            else:
                predInteractions = np.append(predInteractions,predX[boolIdx], axis=0)


            del predX
            gc.collect()


        boolDbMatch = np.isin(predInteractions[:,2],self.DB.df["E1E2"])
        predInteractions = np.append(predInteractions,boolDbMatch.reshape(predInteractions.shape[0],1),axis=1)


        d = pd.DataFrame(predInteractions, columns = dfColumns)
        d["ComplexID"], d["ComplexName"] = zip(*[self._attachComplexID(_bool,E1E2) for E1E2, _bool in zip(predInteractions[:,2], boolDbMatch)])


        d = self._attachPeakIDtoEntries(d)

        d.to_csv(os.path.join(folderToOutput,"predictedInteractions.txt"), sep="\t")
        
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
        fImp = self.classifier.featureImportance()
        
        if not os.path.exists(folderToOutput):
            os.mkdir(folderToOutput)
        if fImp is not None:
            fig, ax = plt.subplots()
            xPos = list(range(len(self.params["metrices"])))
            ax.bar(x = xPos, height = fImp, *args,**kwargs)
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
        print(complexDf.index)
        for e in complexDf.index:
            e = e.split("_")[0]
            posComplex = self.DB.assignComplexToProtein(e,complexMemberIds,"ComplexID")
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
      
        
        return complexDf , np.mean(scores)  #np.unique(complexDf["Cluster Labels"]).size - 


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
        print("\nPredict complexes")
        probColumn = ["Prob_{}".format(n) for n in range(self.params["kFold"])]
        pathToFolder = self._makeFolder(self.params["pathToTmp"],"result","complexParamGrid")
        recordScore = OrderedDict()
        bestDf = None
        maxScore = np.inf 
        cb = ComplexBuilder(method=clusterMethod)

        for n, params in enumerate(list(ParameterGrid(CLUSTER_PARAMS[clusterMethod]))):
            try:
                cb.set_params(params)
                if clusterMethod == "HDBSCAN":
                    clusterLabels, intLabels, matrix , reachability, core_distances, embedd = cb.fit(predInts, 
                                                                                         metricColumns = self.params["metrices"], 
                                                                                         scaler = self.classifier._scaleFeatures)
                else:
                    clusterLabels, intLabels, matrix , reachability, core_distances = cb.fit(predInts, 
                                                                                         metricColumns = self.params["metrices"], 
                                                                                         scaler = self.classifier._scaleFeatures)
               # clusterLabels, intLabels, matrix , reachability, core_distances = cb.fit(predInts, metricColumns = probColumn, scaler = None, inv=True, poolMethod="mean")
            except Exception as e:
                print(e)
                print("There was an error performing clustering and dimensional reduction, using the params" + str(params))
                continue
            df = pd.DataFrame().from_dict({"Entry":intLabels,"Cluster Labels":clusterLabels,"reachability":reachability,"core_distances":core_distances})
            df = df.sort_values(by="Cluster Labels")
            df = df.set_index("Entry")

           # clusteredComplexes = df[df["Cluster Labels"] != -1]
            df, score = self._scoreComplexes(df)
            
           # df = df.join(assignedIDs[["ComplexID"]])
            if True:#maxScore > score:
                df.to_csv(os.path.join( pathToFolder,"Complexes_{}_{}.txt".format(n,score)),sep="\t") 
                print("current best params ... ")
                print(params)
                pd.DataFrame(matrix,columns=intLabels,index=intLabels).loc[df.index,df.index].to_csv(os.path.join(pathToFolder,"SquaredSorted_{}.txt".format(n)),sep="\t")
                maxScore = score
                bestDf = df
                self._plotComplexProfiles(bestDf, pathToFolder, str(n))

            if embedd is not None and plotEmbedding:
                #save embedding
                dfEmbed = pd.DataFrame(embedd)
                dfEmbed["clusterLabels"] = clusterLabels
                dfEmbed["labels"] = intLabels
                dfEmbed.to_csv(os.path.join(pathToFolder,"UMAP_Embeding_{}.txt".format(n)),sep="\t")
                #plot embedding.
                fig, ax = plt.subplots()
                ax.scatter(embedd[:,0],embedd[:,1],s=50, c=clusterLabels, cmap='Spectral')
                plt.savefig(os.path.join(pathToFolder,"E_n{}.pdf".format(n)))
                plt.close()

            recordScore[n] = {"score":score,"params":params}
        

            
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
        toProfiles = self._makeFolder(outputFolder,"complexProfiles")
        pathToFolder = self._makeFolder(toProfiles,str(name))
        
        x = np.arange(0,len(self.X.columns))
        for c in complexDf["Cluster Labels"].unique():
           
            if c != -1:
                fig, ax = plt.subplots()
                entries = complexDf.loc[complexDf["Cluster Labels"] == c,:].index
                lineColors = sns.color_palette("Blues",desat=0.8,n_colors=entries.size)
                for n,e in enumerate(entries):
                    uniprotID = e.split("_")[0]
                    if uniprotID in self.Signals:
                        y = self.Signals[uniprotID].Y
                        ax.plot(x,y,linestyle="-",linewidth=1, label=e, color = lineColors[n])
            
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
    def _createTxtFile(self,name,headers):
        ""
        with open("{}.txt".format(name),"w+") as f:
            f.write("\t".join(headers))

    def _makeTmpFolder(self, n = 0):

        if self.params["analysisName"] is None:

            analysisName = self._randomStr(50)
        elif isinstance(self.params["analysisName"],list) and n < len(self.params["analysisName"]):
            analysisName = self.params["analysisName"][n]
            print(analysisName)
        else:
            analysisName = str(self.params["analysisName"])

        pathToTmpFolder = os.path.join(pathToTmp,analysisName)
        self.currentAnalysisName = analysisName
        if os.path.exists(pathToTmpFolder):
            print("Path to tmp folder exsists")
            print("Will take files from there, if they exist")
            return pathToTmpFolder
        else:
            try:
                os.mkdir(pathToTmpFolder)
                print("Tmp folder created -- ",analysisName)
                self._makeFolder(pathToTmpFolder,"chunks")
                self._createTxtFile("runTimes",["Date","Time","Step","Comment"])

                return pathToTmpFolder
            except OSError:
                print("Could not create tmp folder")

    def run(self,X, maxValueToOne = False):
        ""  
        if isinstance(X,list):
            print("multiple dataset given - will be analysed separetely")
        elif isinstance(X,str) and os.path.exists(X):
            loadFiles = [f for f in os.listdir(X) if f.endswith(".txt")]
            self.params["analysisName"] = loadFiles
            X = [pd.read_csv(os.path.join(X,fileName), sep="\t") for fileName in loadFiles]

           # boolidx = pd.read_csv("../example-data/filter.txt")["filter"]
           # X = [x.loc[boolidx.values] for x in X]
            
            if maxValueToOne:
                maxValues = pd.concat([x.max(axis=1) for x in X], axis=1).max(axis=1)
                normValueDict = dict([(X[0][self.params["idColumn"]].values[n],maxValue) for n,maxValue in enumerate(maxValues.values)])
                self.params["normValueDict"] = normValueDict
        else:
            X = [X]
        for n,X in enumerate(X):
            pathToTmpFolder = self._makeTmpFolder(n)
            self.params["pathToTmp"] = pathToTmpFolder

            if pathToTmpFolder is not None:
                
                self._load(X)
                self._checkGroups()
                self._findPeaks(self.params["n_jobs"])
                self._collectRSquaredAndFitDetails()
                self._calculateDistance()
                self._loadReferenceDB()
                self._addMetricesToDB()
                self._trainPredictor()
                predInts = self._predictInteractions()
                self._clusterInteractions(predInts)



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
    ComplexFinder(indexIsID=False,analysisName=["TMEM70_WT_01","TMEM70_WT_02"],classifierClass="random forest",retrainClassifier=False,interactionProbabCutoff = 0.6,removeSingleDataPointPeaks=True).run("../example-data/TMEM70/")#[X,Y])





    
