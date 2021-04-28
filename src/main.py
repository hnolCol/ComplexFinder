
#python imports
import os 
import gc 
import string
import random
import time
import pickle
import shutil 
from datetime import datetime

#internal imports 
from modules.Signal import Signal
from modules.Database import Database
from modules.Predictor import Classifier, ComplexBuilder
from modules.utils import calculateDistanceP, chunks, cleanPath, minMaxNorm

import joblib
from joblib import Parallel, delayed, dump, load

import pandas as pd
import numpy as np 

from collections import OrderedDict
from itertools import combinations

from multiprocessing import Pool
from joblib import wrap_non_picklable_objects

#plotting
import matplotlib.pyplot as plt
import seaborn as sns

#sklearn imports
from sklearn.metrics import classification_report, homogeneity_score, v_measure_score, completeness_score
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale, minmax_scale, robust_scale
from scipy.stats import ttest_ind, f_oneway

#dimensional reduction 
import umap



__VERSION__ = "0.4.48"

filePath = os.path.dirname(os.path.realpath(__file__)) 
pathToTmp = os.path.join(filePath,"tmp")

alignModels = { "LinearRegression": LinearRegression, 
                "RadiusNeighborsRegressor" : RadiusNeighborsRegressor, 
                "KNeighborsRegressor":KNeighborsRegressor
            }
alignModelsParams = { 

        "LinearRegression": {}, 
        "RadiusNeighborsRegressor" : {"weights":"distance","radius":30} , 
        "KNeighborsRegressor":{"weights":"distance","n_neighbors":10}
                    }


STACKING_CLASSIFIER_GRID = {
                            'rf__max_depth': [70,None,30],#30,,
                            #'rf__max_features': ['auto'],
                           # 'rf__min_samples_leaf': [2, 3, 5],
                            'rf__min_samples_split': [2,4],#[2, 3, 4],
                            #'rf__n_estimators': [200],
                            "SVM__C" : [1, 10,1000],
                            "SVM__kernel": ['rbf','poly'],
                            'SVM__gamma': [0.01,10,100]
                            }

OPTICS_PARAM_GRID = {
                "min_samples":[2,3,5,8], 
                "max_eps": [np.inf,2,1,0.9,0.8], 
                "xi": np.linspace(0,0.3,num=30), 
                "cluster_method" : ["xi"]
                }
AGGLO_PARAM_GRID = {
                "n_clusters":[None,115,110,105,100,90,95],
                "distance_threshold":[None,0.5,0.4,0.2,0.1,0.05,0.01], 
                "linkage":["complete","single","average"]
                }
AFF_PRO_PARAM = {"damping":np.linspace(0.5,1,num=50)}
HDBSCAN_PROPS = {
                "min_cluster_size":[2,3,4,6],
                "min_samples":[2,3,4,5]
                }
                #{"min_cluster_size":[2,3,4,6],"min_samples":[2,3,4,5,8,10]}
CLUSTER_PARAMS = {
                "OPTICS":OPTICS_PARAM_GRID,
                "AGGLOMERATIVE_CLUSTERING":AGGLO_PARAM_GRID,
                "AFFINITY_PROPAGATION":AFF_PRO_PARAM,
                "HDBSCAN":HDBSCAN_PROPS
                }

svm_param_grid = {
        'C': [1, 10, 100, 1000], 
        'kernel': ['linear','rbf','poly'], 
        'gamma': [0.01,0.1,1,2,3,4,5]
        }

RF_GRID_SEARCH = {
                'max_depth': [70,None,30,50,10],#30,,,50,5
                'max_features': ['auto'],
                'min_samples_leaf': [2,5,3,15], # 5, 15
                'min_samples_split': [2 ,3,10],
                'n_estimators': [300, 500, 600]
                }
entriesInChunks = dict() 


class ComplexFinder(object):

    def __init__(self,
                addImpurity = 0.0,
                alignMethod = "RadiusNeighborsRegressor",#"RadiusNeighborsRegressor",#"KNeighborsRegressor",#"LinearRegression", # RadiusNeighborsRegressor
                alignRuns = False,
                alignWindow = 3,
                analysisMode = "label-free", #[label-free,SILAC,SILAC-TMT]
                analysisName = None,
                binaryDatabase = False,
                classifierClass = "random_forest",
                classifierTestSize = 0.25,
                classiferGridSearch = RF_GRID_SEARCH,#STACKING_CLASSIFIER_GRID,#
                compTabFormat = False,
                considerOnlyInteractionsPresentInAllRuns = 2,
                correlationWindowSize = 5,
                databaseFilter = {'Organism': ["Human"]},#{'Organism': ["Human"]},#{"Confidence" : [1,2,3,4]} - for hu.map2.0,# {} for HUMAN_COMPLEX_PORTAL
                databaseIDColumn = "subunits(UniProt IDs)",
                databaseFileName = "20190823_CORUM.txt",#"humap2.txt
                databaseHasComplexAnnotations = True,
                databaseEntrySplitString = ";",
                decoySizeFactor = 1.2,
                grouping = {"WT": ["D3_WT_03.txt"]},
                hdbscanDefaultKwargs = {"min_cluster_size":4,"min_samples":1},
                indexIsID = False,
                idColumn = "Uniprot ID",
                interactionProbabCutoff = 0.7,
                justFitAndMatchPeaks = False,
                keepOnlySignalsValidInAllConditions = False,
                kFold = 3,
                maxPeaksPerSignal = 15,
                maxPeakCenterDifference = 1.8,
                metrices = ["apex","pearson","euclidean","umap-dist","cosine","max_location","rollingCorrelation","signalDiff"],
                metricesForPrediction = None,#["pearson","euclidean","apex"],
                metricQuantileCutoff = 0.001,
                minDistanceBetweenTwoPeaks = 3,
                minimumPPsPerFeature = 6,
                minPeakHeightOfMax = 0.05,
                n_jobs = 12,
                noDatabaseForPredictions = False,
                normValueDict = {},
                noDistanceCalculationAndPrediction = False,
                peakModel = "GaussianModel",#"SkewedGaussianModel",#"LorentzianModel",
                plotSignalProfiles = False,
                plotComplexProfiles = False,
                precision = 0.5,
                r2Thresh = 0.85,
                removeSingleDataPointPeaks = True,
                restartAnalysis = False,
                retrainClassifier = False,
                recalculateDistance = False,
                rollingWinType = None,
                runName = None,
                scaleRawDataBeforeDimensionalReduction = True,
                smoothSignal = True,
                smoothWindow = 2,
                takeRondomSampleFromData =False,
                topNCorrFeaturesForUMAPAlignment = 200,
                useRawDataForDimensionalReduction = False,
                umapDefaultKwargs = {"min_dist":0.001,"n_neighbors":5,"n_components":2,"random_state":120},
                quantFiles = [],
                usePeakCentricFeatures = False
                ):
        """
        Init ComplexFinder Class
        

        Parameters
        ----------
        
        * alignMethod = "RadiusNeighborsRegressor",
        
        * alignRuns = False, 
                    Alignment of runs is based on signal profiles that were found to have 
                    a single modelled peak. A refrence run is assign by correlation anaylsis 
                    and choosen based on a maximum R2 value. Then fraction-shifts per signal 
                    profile is calculated (must be in the window given by *alignWindow*). 
                    The fraction residuals are then modelled using the method provided in 
                    *alignMethod*. Model peak centers are then adjusted based on the regression results. 
                    Of note, the alignment is performed after peak-modelling and before distance calculations. 
        
        * alignWindow = 3, 
                    Number of fraction +/- single-peal profile are accepted for the run alignment. 
        
        * analysisMode = "label-free", #[label-free,SILAC,SILAC-TMT]
        
        * analysisName = None,
        
        * binaryDatabase = False,
        
        * classifierClass = "random_forest",
        
        * classifierTestSize = 0.25, 
                    Fraction of the created database containing positive and negative protein-protein 
                    interactions that will be used for testing (for example ROC curve analysis) and classification report.
        
        * classiferGridSearch = RF_GRID_SEARCH. 
                    Dict with keywords matching parameters/settings of estimator (SVM, random forest) 
                    and list of values forming the grid used to find the best estimator settings (evaluated 
                    by k-fold cross validation). Runtime is effected by number of parameter settings as well as k-fold. 

        * compTabFormat = False
                    True indicates that the data are in the CompBat data format which was recently introduced. 
                    In contrast to standard txt files generated by for example MaxQuant. It contains multiple
                    headers. More information can be found here https://www3.cmbi.umcn.nl/cedar/browse/comptab
                    ComplexFinder will try to identifiy the samples and fractions and create separeted txt files. 


        * considerOnlyInteractionsPresentInAllRuns = 2, 
                    Can be either bool to filter for protein - protein interactions that are present 
                    in all runs. If an integer is provided. the pp interactions are filtered based on 
                    the number of runs in which they were quantified. A value of 4 would indicate that 
                    the pp interaction must have been predicted in all runs. 

        * correlationWindowSize = 5,
                    Number of fractions used for rolling pearson correlation

        * databaseFilter = {'Organism': ["Human"]}, 
                    Filter dict used to find relevant complexes from database. By default, 
                    the corum database is filtered based on the column 'Organism' using 'Mouse' as a search string. 
                    If no filtering is required, pass an empty dict {}. 
        * databaseIDColumn = "subunits(UniProt IDs)",
        
        * databaseFileName = "20190823_CORUM.txt",
        
        * databaseHasComplexAnnotations = True, 
                    Indicates if the provided database does contain complex annotations. If you have a database with 
                    only pairwise interactions, this setting should be *False*. Clusters are identified by dimensional 
                    reduction and density based clustering (HDBSCAN). In order to alter UMAP and HDBSCAN settings use the 
                    kewywords *hdbscanDefaultKwargs* and *umapDefaultKwargs*.

        * databaseEntrySplitString = ";",
                String by which complex members are separated in the provided database. CORUM = ";", Embl ComplexMap = "|"

        * decoySizeFactor = 1.2,
                Size factor for creating the decoy database from the positive proterin connectivity database such as CORUM.
        
        * grouping = {"WT": ["D3_WT_04.txt","D3_WT_02.txt"],"KO":["D3_KO_01.txt","D3_KO_02.txt"]}, 
                None or dict. Indicates which samples (file) belong to one group. Let's assume 4 files with the name 
                'KO_01.txt', 'KO_02.txt', 'WT_01.txt' and 'WT_02.txt' are being analysed. 
                The grouping dict should like this : {"KO":[KO_01.txt','KO_02.txt'],"WT":['WT_01.txt','WT_02.txt']} 
                in order to combine them for statistical testing (e.g. t-test of log2 transformed peak-AUCs). 
                Note that when analysis multiple runs (e.g. grouping present) then calling ComplexFinder().run(X) - X must be a 
                path to a folder containing the files.
                When using compTabFormat = True. Provide the sample name as <compTabFileName>:<SampleName>. 

        
        * hdbscanDefaultKwargs = {"min_cluster_size":4,"min_samples":1},
        
        * indexIsID = False,
        
        * idColumn = "Uniprot ID",
        
        * interactionProbabCutoff = 0.7
            Cutoff for estimator probability. Interactions with probabilities below threshold will be removed.

        * keepOnlySignalsValidInAllConditions = False
            If True, removes all Signals that were not found to be valid in all files (experiments). 

        * kFold = 3
            Cross validation of classifier optimization.
        
        * justFitAndMatchPeaks = False
            If true, the pipeline stops after peak detection/model fitting and matching of peaks (if more than one file is supplied.)
        
        * maxPeaksPerSignal = 15
            Number of peaks allowed for on signal profile.
        
        * maxPeakCenterDifference = 1.8
        
        * metrices = ["apex","pearson","euclidean","p_pearson","max_location","umap-dist","rollingCorrelation"], Metrices to access distance between two profiles. Can be either a list of strings and/or dict. In case of a list of dicts, each dict must contain the keywords: 'fn' and 'name' providing a callable function with 'fn' that returns a single floating number and takes two arrays as an input.
        
        * metricesForPrediction = None
        
        * metricQuantileCutoff = 0.90
        
        * minDistanceBetweenTwoPeaks = 3 
                Distance in fractions (int) between two peaks. Setting this to a smaller number results in more peaks.

        * n_jobs = 12, 
                Number of workers to model peaks, to calculate distance pairs and to train and use the classifer.

        * noDatabaseForPredictions = False, 
                If you want to use ComplexFinder without any database. Set this to *True*.

        * normValueDict = {},

        * noDistanceCalculationAndPrediction = False,
                Set to *True* to use ComplexFinder without distance calculation and database prediction.

        * peakModel = "GaussianModel", 
                Indicates which model should be used to model signal profiles. In principle all models from lmfit can be used. 
                However, the initial parameters are only optimized for GaussianModel and LaurentzianModel. 
                This might effect runtimes dramatically. 

        * plotSignalProfiles = False, 
            If True, each profile is plotted against the fractio along with the fitted models. 
            If you are concerned about time, you might set this to False at the cost of losing visible asessment of the fit quality.
        
        * plotComplexProfiles = False,
        
        * precision = 0.5
            Precision to use to filter protein-protein interactions. 
            If None, the filtering will be performed based on the parameter *interactionProbabCutoff*.
        
        * r2Thresh = 0.85 
            R2 threshold to accept a model fit. Models below the threshold will be ignored.
        
        * removeSingleDataPointPeaks = True,
        
        * restartAnalysis = False, bool. 
            Set True if you want to restart the anaylsis from scratch. If the tmp folder exsists, items and dirs will be deleted first.
        
        * retrainClassifier = False, 
            Even if the trainedClassifier.sav file is found, the classifier is loaded and the training is skipped. 
            If you change the classifierGridSearch, you should set this to True. 
            This will ensure that the classifier training is never skipped.
        
        * recalculateDistance = False,

        * rollingWinType = None,
            If None, all points are evenly weighted. Can be any string of scipy.signal window function.
            (https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows)
        
        * runName = None,
        
        * <del>savePeakModels = True</del> *depracted. always True and will be removed in the next version*.
        
        * scaleRawDataBeforeDimensionalReduction = True, 
            If raw data should be used (*useRawDataForDimensionalReduction*) 
            enable this if you want to scale them. Scaling will be performed that values of each row are scaled between zero and one.
        
        * smoothSignal = True
            Enable/disable smoothing. Defaults to True. A moving average of at least 3 adjacent datapoints is calculated using 
            pandas rolling function. Effects the analysis time as well as the nmaximal number of peaks detected.
        
        * smoothWindow = 2,

        * topNCorrFeaturesForUMAPAlignment = 200,
            Number of profiles used for UMAP Alignment. Only used if useRawDataForDimensionalReduction = True or noDistanceCalculationAndPrediction = True. The Features
            will be identified by calculating the Pearson correlation coefficient. 
        
        * useRawDataForDimensionalReduction = False, Setting this to true, will force the pipeline to use the raw values for dimensional reduction. Distance calculations are not automatically turned off and the output is generated but they are not used.
        
        * umapDefaultKwargs = {"min_dist":0.0000001,"n_neighbors":3,"n_components":2},
            If you want to perform an aligned UMPA consider altering the parameter alignment_window_size and alignment_regularisation. Find more information here
            (https://umap-learn.readthedocs.io/en/latest/aligned_umap_basic_usage.html#aligning-varying-parameters)
        
        * quantFiles = [] list of str.
            
        Returns
        -------
        None

        """

        self.params = {
            "addImpurity" : addImpurity,
            "indexIsID" : indexIsID,
            "idColumn" : idColumn,
            "n_jobs" : n_jobs,
            "kFold" : kFold,
            "analysisName" : analysisName,
            "restartAnalysis" : restartAnalysis,
            "metrices" : metrices,
            "peakModel" : peakModel,
            "smoothWindow" : smoothWindow,
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
            "minimumPPsPerFeature" : minimumPPsPerFeature,
            "plotComplexProfiles" : plotComplexProfiles,
            "decoySizeFactor" : decoySizeFactor,
            "classifierTestSize" : classifierTestSize,
            "considerOnlyInteractionsPresentInAllRuns" : considerOnlyInteractionsPresentInAllRuns,
            "precision" : precision,
            "quantFiles" : quantFiles,
            "compTabFormat" : compTabFormat,
            "correlationWindowSize" : correlationWindowSize,
            "takeRondomSampleFromData" : takeRondomSampleFromData,
            "minPeakHeightOfMax" : minPeakHeightOfMax,
            "justFitAndMatchPeaks" : justFitAndMatchPeaks,
            "keepOnlySignalsValidInAllConditions" : keepOnlySignalsValidInAllConditions,
            "noDistanceCalculationAndPrediction" : noDistanceCalculationAndPrediction,
            "topNCorrFeaturesForUMAPAlignment" : topNCorrFeaturesForUMAPAlignment,
            "databaseEntrySplitString": databaseEntrySplitString,
            "version" : __VERSION__,
            "usePeakCentricFeatures" : usePeakCentricFeatures
            }
        print("\n" + str(self.params))
        self._checkParameterInput()
    
    def _addMetricesToDB(self,analysisName):
        """
        Adds distance metrices to the database entries
        that were found in the co-elution profiles.

        Parameters
        ----------
    
        Returns
        -------
        None

        """
        if self.params["noDistanceCalculationAndPrediction"]:
            print("Info :: Skipping matching metrices to DB.")
            return 

        if "signalDiff" in self.params["metrices"]:
            
            self.params["metrices"] = [x for x in self.params["metrices"] if x != "signalDiff"] + ["{}-diff".format(x) for x in np.arange(self.Xs[analysisName].columns.size)]
        metricColumns = self.params["metrices"] 
        if not self.params["noDatabaseForPredictions"]:
            self.DB.matchMetrices(self.params["pathToTmp"][analysisName],entriesInChunks[analysisName],metricColumns,analysisName,forceRematch=self.params["recalculateDistance"])


    def _addMetricToStats(self,metricName, value):
        """
        Adds a metric to the stats data frame.
        Does not check if metric is represent, if present,
        it will just overwrite.

        Parameters
        ----------

        metricName str
            Name of metric to add

        value str
            Value of metric
    
        Returns
        -------
        None

        """
        if metricName in self.stats.columns:
            self.stats.loc[self.currentAnalysisName,metricName] = value

    def _addModelToSignals(self,signalModels):
        """
        Adds fitted models to Signals. If not a valid
        model was found, then the signal profile is removed.

        Parameters
        ----------
        signalModels - list
            List of modelfits (dict)

        Returns
        -------
        None

        """
        for fitModel in signalModels:
            modelID = fitModel["id"]
            if len(fitModel) == 1:
                del self.Signals[self.currentAnalysisName][modelID]
            if modelID in self.Signals[self.currentAnalysisName]:
                for k,v in fitModel.items():
                    if k != 'id':
                        setattr(self.Signals[self.currentAnalysisName][modelID],k,v)
                self.Signals[self.currentAnalysisName][modelID].saveResults()

    def _attachQuantificationDetails(self):
        """
        """


    def _checkParameterInput(self):
        """
        Checks the input.

        Parameters
        ----------
    
        Returns
        -------
        None

        Raises
        -------
        ValueErrors if datatype if given parameters do not match. 

        """

        #check anaylsis mode
        validModes = ["label-free","SILAC","TMT-SILAC"]
        if self.params["analysisMode"] not in validModes:
            raise ValueError("Parmaeter analysis mode is not valid. Must be one of: {}".format(validModes))
        elif self.params["analysisMode"] != "label-free" and len(self.params["quantFiles"]) == 0:
            raise ValueError("Length 'quantFiles must be at least 1.")

        if not isinstance(self.params["maxPeaksPerSignal"],int):
            raise ValueError("maxPeaksPerSignal must be an integer. Current setting: {}".forma(self.params["maxPeaksPerSignal"]))

        elif self.params["maxPeaksPerSignal"] <= 2:
            raise ValueError("maxPeaksPerSignal must be greater than or equal 2")

        elif self.params["maxPeaksPerSignal"] > 20:
            print("Warning :: maxPeaksPerSignal is set to above 20, this may take quite long to model.")

        #r2 validation
        if not isinstance(self.params["r2Thresh"],float):
            raise ValueError("Parameter r2Trehsh mus be a floating number.")
        elif self.params["r2Thresh"] < 0.5:
            print("Warning :: threshold for r2 is set below 0.5. This might result in fits of poor quality")
        elif self.params["r2Thresh"] > 0.95:
            print("Warning :: threshold for r2 is above 0.95. Relatively few features might pass this limit.")
        elif self.params["r2Thresh"] > 0.99:
            raise ValueError("Threshold for r2 was above 0.99. Please set a lower value.")

        #minPeakHeightOfMax
        if not isinstance(self.params["minPeakHeightOfMax"],float) and self.params["minPeakHeightOfMax"] < 1 and self.params["minPeakHeightOfMax"] >= 0:
            raise ValueError("Parameter 'minPeakHeightOfMax' must be a float smaller than 1.0 and greather/equal 0.0.")

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
        
        
    
    def _chunkPrediction(self,pathToChunk,classifier,nMetrices,probCutoff):
        """
        Predicts for each chunk the proability for positive interactions.

        Parameters
        ----------
            pathToChunk : str

            classifier : classfierClass 
                Trained classifier. 

            nMetrices : int
                Number if metrices used. (since chunks are simple numpy arrays, no column headers are loaded)

            probCutoff : float
                Probability cutoff.
        Returns
        -------
            Numpy array. Chunks with appended probability.
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


    def _load(self, X):
        """
        Intitiates data.

        Parameters
        ----------

        X  pd.DataFrame 
    
        Returns
        -------
        None

        Raises
        -------
        ValueError if X is not a pandas data frame.

        """
        if isinstance(X, pd.DataFrame):
            
            self.X = X
            
            if not self.params["indexIsID"]:
                print("Info :: Checking for duplicates")
                dupRemoved = self.X.drop_duplicates(subset=[self.params["idColumn"]])
                if dupRemoved.index.size < self.X.index.size:
                    print("Warning :: Duplicates detected.")
                    print("File contained duplicate ids which will be removed: {}".format(self.X.index.size-dupRemoved.index.size))
                    self.X = dupRemoved
                

                self.X = self.X.set_index(self.params["idColumn"])
                self.X = self.X.astype(np.float32)
            else:
                self.X = self.X.loc[self.X.index.drop_duplicates()] #remove duplicaates
                self.X = self.X.astype(np.float32) #set dtype to 32 to save memory
            if self.params["takeRondomSampleFromData"] != False and self.params["takeRondomSampleFromData"] > 50:
                self.X = self.X.sample(self.params["takeRondomSampleFromData"])
                print("Random samples taken from data. New data size {}".format(self.X.index.size))
            self.params["rawData"][self.currentAnalysisName] = self.X.copy()
        else:

            raise ValueError("X must be a pandas data frame")

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
        if self.params["noDistanceCalculationAndPrediction"]:
            print("noDistanceCalculationAndPrediction was enabled. No database laoded.")
            return 

        if self.params["noDatabaseForPredictions"]:
            print("Info ::  Parameter noDatabaseForPredictions was set to True. No database laoded.")
            return

        print("Info :: Load positive set from data base")
        if not hasattr(self,"DB"):
            self.DB = Database(nJobs = self.params["n_jobs"], splitString=self.params["databaseEntrySplitString"])

        pathToDatabase = os.path.join(self.params["pathToComb"], "InteractionDatabase.txt")
        if os.path.exists(pathToDatabase):

            dbSize = self.DB.loadDatabaseFromFile(pathToDatabase)
            print("Info :: Database found and loaded. Contains {} positive interactions.".format(dbSize))
           # self._addMetricToStats("nPositiveInteractions",dbSize)
        else:
            
            self.DB.pariwiseProteinInteractions(
                            self.params["databaseIDColumn"],
                            dbID = self.params["databaseFileName"],
                            filterDb=self.params["databaseFilter"])

            entryList = []
            for analysisName in self.params["analysisName"]:
                entryList.extend([entryID for entryID,Signal in self.Signals[analysisName].items() if Signal.valid])
            entryList = np.unique(np.array(entryList).flatten())
            print("Info :: Features used for filtering: {}".format(len(entryList)))
            dbSize = self.DB.filterDBByEntryList(entryList)

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
    
                
    def _checkGroups(self):
        "Checks grouping. For comparision of multiple co-elution data sets."
        
        if isinstance(self.params["grouping"],dict):
            if len(self.params["grouping"]) == 0:
                raise ValueError("Example for grouping : {'KO':['KO_01.txt','KO_02.txt'], 'WT':['WT_01.txt','WT_02.txt'] } Aborting.. ")
            else:
                combinedSamples = sum(self.params["grouping"].values(), [])
                if all(x in combinedSamples for x in self.params["analysisName"]):
                    print("Grouping checked..\nAll columnSuffixes found in grouping.")
                    print("If you are using the combat format, the grouping has to be named as '<combatFileName><sample name>")
                else:
                    raise ValueError("Could not find all grouping names in loaded dataframe.. Aborting ..")
                

    def _findPeaks(self, n_jobs=3):
        """
        Initiates for each feature in the data a Signal instance. 
        Peak detection and modelling is then performed.
        Results are saved to hard drive for each run. 

        Numerous parameters effect signal modelling (smoothing, maxPeaks, r2Thresh, ...)

        Create self.Signals (OrderedDict) which is a dict. Key = analysisName, which
        contains another dict with entries as keys and values are of type Signal class.

        Parameters
        ----------

        n_jobs  int. Number of worker processes.
    
        Returns
        -------
        None

        """
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

        self.Signals[self.currentAnalysisName] = dict()
        peakModel = self.params['peakModel']

        for entryID, signal in self.X.iterrows():

            self.Signals[self.currentAnalysisName][entryID] = Signal(signal.values,
                                            ID = entryID, 
                                            peakModel = peakModel, 
                                            smoothSignal = self.params["smoothSignal"],
                                            savePlots = self.params["plotSignalProfiles"],
                                            savePeakModels = self.params["savePeakModels"],
                                            maxPeaks = self.params["maxPeaksPerSignal"],
                                            metrices = self.params["metrices"],
                                            pathToTmp = self.params["pathToTmp"][self.currentAnalysisName],
                                            normalizationValue = self.params["normValueDict"][entryID] if entryID in self.params["normValueDict"] else None,
                                            removeSingleDataPointPeaks = self.params["removeSingleDataPointPeaks"],
                                            analysisName = self.currentAnalysisName,
                                            r2Thresh = self.params["r2Thresh"],
                                            smoothRollingWindow = self.params["smoothWindow"],
                                            minDistanceBetweenTwoPeaks = self.params["minDistanceBetweenTwoPeaks"],
                                            minPeakHeightOfMax = self.params["minPeakHeightOfMax"])

        
        t1 = time.time()
        print("\n\nStarting Signal modelling .. (n_jobs = {})".format(n_jobs))
        
        fittedModels = Parallel(n_jobs=n_jobs, verbose=1)(delayed(Signal.fitModel)() for Signal in self.Signals[self.currentAnalysisName].values())
        
        self._addModelToSignals(fittedModels)
        self._saveSignalFitStatistics()
        
        print("Peak fitting done time : {} secs".format(round((time.time()-t1))))
        print("Each feature's fitted models is stored as pdf and txt is stored in model plots (if savePeakModels and plotSignalProfiles was set to true)")
        

    def _saveSignals(self):
        ""
        if hasattr(self,"Signals") :
            pathToSignal = os.path.join(self.params["pathToComb"],"signals.lzma")
            dump(self.Signals.copy(),pathToSignal)
        
        self.Xs = OrderedDict()
        for analysisName in self.params["analysisName"]:
            pathToFile = os.path.join(self.params["pathToTmp"][analysisName],"validProcessedSignals({}).txt".format(analysisName))
            signals = self.Signals[analysisName]
            validSignalData = dict([(k,v.Y) for k,v in signals.items() if v.valid and v.validModel])
            fitDataSignal = dict([(k,v.fitSignal.flatten()) for k,v in signals.items() if v.valid and v.validModel and v.fitSignal is not None])
            
            dfProcessedSignal = pd.DataFrame().from_dict(validSignalData,orient="index")
            dfFit = pd.DataFrame().from_dict(fitDataSignal, orient="index")
            
            #print(self.params["rawData"][analysisName].index)
            df = dfProcessedSignal.join(self.params["rawData"][analysisName],rsuffix="_raw",lsuffix="_processed")
            df = df.join(dfFit,rsuffix = "_fit")
            
            df.to_csv(pathToFile,sep="\t")
            self.Xs[analysisName] = dfProcessedSignal
            X = self.Xs[analysisName].reset_index()
        
            np.save(os.path.join(self.params["pathToTmp"][analysisName],"source.npy"),X.values)

        for analysisName in self.params["analysisName"]:
            #clean invalid signals
            if self.params["keepOnlySignalsValidInAllConditions"]:
                toDelete = [k for k,v in self.Signals[analysisName].items() if not all(k in self.Signals[analysisName] and self.Signals[analysisName][k].valid for analysisName in self.params["analysisName"])]
            else:
                toDelete = [k for k,v in self.Signals[analysisName].items() if not v.valid]
            #delete Signals that do match criteria
            for k in toDelete:
                del self.Signals[analysisName][k]

    def _calculateDistance(self):
        """
        Calculates Distance between protein protein pairs based
        on their signal profile.

        Parameters
        ----------
        signalModels - list
            List of modelfits (dict)

        Returns
        -------
        None

        """
        if self.params["noDistanceCalculationAndPrediction"]:
            print("noDistanceCalculationAndPrediction was enabled. Skipping Distance Calculations.")
            return 
        global entriesInChunks
       
        
        print("\nStarting Distance Calculation ...")
        t1 = time.time()
        
        chunks = self.signalChunks[self.currentAnalysisName]
        #return
        entrieChunkPath = os.path.join(self.params["pathToComb"], "entriesInChunk.pkl")
        if not self.params["recalculateDistance"] and all(os.path.exists(x.replace(".pkl",".npy")) for x in chunks) and os.path.exists(entrieChunkPath):
            print("All chunks found for distance calculation.")
            if not self.entriesInChunkLoaded:
                with open(os.path.join(self.params["pathToComb"], "entriesInChunk.pkl"),"rb") as f:
                       entriesInChunks = pickle.load(f)
                self.entriesInChunkLoaded = True

        else:

            chunkItems = Parallel(n_jobs=self.params["n_jobs"], verbose=10)(delayed(calculateDistanceP)(c) for c in chunks)
            entriesInChunks[self.currentAnalysisName] = {}
            for k,v in chunkItems:
                for E1E2 in v:
                    entriesInChunks[self.currentAnalysisName][E1E2] = k 
            
            with open(os.path.join(self.params["pathToComb"], "entriesInChunk.pkl"),"wb") as f:
                        pickle.dump(entriesInChunks,f)
    
        
        

        print("Distance computing/checking: {} secs\n".format(round(time.time()-t1)))

    def _createSignalChunks(self,chunkSize = 30):
        """
        Creates signal chunks at given chunk size. 
        
        Parameter
        ---------
            chunkSize - int. default 30. Nuber of signals in a single chunk.

        Returns
        -------
            list of paths to the saved chunks.
        """
        pathToSignalChunk = os.path.join(self.params["pathToComb"],"signalChunkNames.lzma")
        if os.path.exists(pathToSignalChunk) and not self.params["recalculateDistance"]:
            self.signalChunks = load(pathToSignalChunk)
            print("Info :: Signal chunks loaded and found. Checking if all runs are present.")
            if all(analysisName in self.signalChunks for analysisName in self.params["analysisName"]):
                print("Info :: Checked... all samples found.")
                return
            else:
                print("Info :: Not all samples found. Creating new signal chunks..")
        
        if not hasattr(self,"signalChunks"):
            self.signalChunks = dict() 
        else:
            self.signalChunks.clear()

        for analysisName in self.params["analysisName"]:
            print("Info :: {} signal chunk creation started.\nThis may take some minutes.." .format(analysisName))
            if "umap-dist" in self.params["metrices"]:
                #umap dist calculations
                print("Info :: Calculation UMAP.")
                embed = umap.UMAP(min_dist=0.0000000000001, n_neighbors=5, metric = "correlation", random_state=56).fit_transform(minMaxNorm(self.Xs[analysisName].values,axis=1))
                embed = pd.DataFrame(embed,index=self.Xs[analysisName].index)
                #save embedding
                embed.to_csv(os.path.join(self.params["pathToTmp"][analysisName],"chunks","embeddings.txt"),sep="\t")
            signals = list(self.Signals[analysisName].values())

            for n,Signal in enumerate(self.Signals[analysisName].values()):
                setattr(Signal,"otherSignals", signals[n:])

            c = []
            

            for n,chunk in enumerate(chunks(signals,chunkSize)):
                pathToChunk = os.path.join(self.params["pathToTmp"][analysisName],"chunks",str(n)+".pkl")
                #if not os.path.exists(pathToChunk) and not self.params["recalculateDistance"]:

                chunkItems =  [
                    {
                    "ID"                : str(signal.ID),
                    "chunkName"         : str(n),
                    "Y"                 : np.array(signal.Y),
                    "ownPeaks"          : signal.getPeaksAndsIDs(),
                    "otherSignalPeaks"  : [s.getPeaksAndsIDs() for s in signal.otherSignals],
                    "E2"                : [str(s.ID) for s in signal.otherSignals],
                    "metrices"          : self.params["metrices"],
                    "pathToTmp"         : self.params["pathToTmp"][analysisName],
                    "correlationWindowSize" : self.params["correlationWindowSize"],
                    "embedding"         : embed.loc[signal.ID].values if "umap-dist" in self.params["metrices"] else [],
                            } for signal in chunk]
                
                with open(pathToChunk,"wb") as f:
                    pickle.dump(chunkItems,f)
                
                c.append(pathToChunk)
            
            self.signalChunks[analysisName] = [p for p in c if os.path.exists(p)] #

        #saves signal chunls.
        dump(self.signalChunks,pathToSignalChunk)



    def _collectRSquaredAndFitDetails(self):
        """
        Data are collected from txt files in the modelPlots folder. 
            
        """
        if not self.params["savePeakModels"]:
            print("!! Warning !! This parameter is depracted and from now on always true.")
            self.params["savePeakModels"] = True

        pathToPlotFolder = os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"result","modelPlots")
        resultFolder = os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"result")

        fittedPeaksPath = os.path.join(resultFolder,"fittedPeaks.txt")
        nPeaksPath = os.path.join(resultFolder,"nPeaks.txt")
        if os.path.exists(fittedPeaksPath) and os.path.exists(nPeaksPath):
            print("Warning :: FittedPeaks detected. If you changed the data, you have to set the paramter 'restartAnalysis' True to include changes..")
            return
        if not os.path.exists(resultFolder):
            os.mkdir(resultFolder)

        #number of peaks
        collectNumbPeaks = []


        data = [{"Key":signal.ID,
                "ID" : n,
                "R2":signal.Rsquared,
                "#Peaks":len(signal.modelledPeaks),
                "Center":peakParam["mu"],
                "Amplitude":peakParam["A"],
                "Sigma":peakParam["sigma"],
                "fwhm":peakParam["fwhm"],
                "height" : peakParam["height"],
                "AUC" : peakParam["AUC"],
                "relAUC" : peakParam["relAUC"],
                "validModel":signal.validModel,
                "validData":signal.validData,
                "Y": ",".join([str(round(x,3)) for x in peakParam["Y"]])} for signal in self.Signals[self.currentAnalysisName].values() if signal.valid for n,peakParam in enumerate(signal.modelledPeaks)]

        df = pd.DataFrame().from_dict(data)
        df.to_csv(fittedPeaksPath,sep="\t",index=None)
        # # find peak properties..
        # df = pd.DataFrame(columns=["Key","ID","Amplitude","Center","Sigma","fwhm","height","auc"])

        # for file in os.listdir(pathToPlotFolder):
        #     if file.endswith(".txt"):
        #         try:
        #             dfApp = pd.read_csv(os.path.join(pathToPlotFolder,file), sep="\t")
        #             df = df.append(dfApp)
        #             collectNumbPeaks.append({"Key":dfApp["Key"].iloc[0],"N":len(dfApp.index)})
        #         except:
        #             continue

        
        

        #pd.DataFrame(collectNumbPeaks).to_csv(nPeaksPath,sep="\t", index = None)


    def _trainPredictor(self, addImpurity = 0.3, apexTraining = False):
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
   
        if self.params["noDatabaseForPredictions"] or self.params["noDistanceCalculationAndPrediction"]:
            print("Predictor training skipped (noDatabaseForPredictions = True or noDistanceCalculationAndPrediction = True). Distance metrices/Raw signals are used for dimensional reduction.")
            return 

        folderToResults = [os.path.join(self.params["pathToTmp"][analysisName],"result") for analysisName in self.params["analysisName"]]
        classifierFileName = os.path.join(self.params["pathToComb"],'trainedClassifier_{}.sav'.format(self.params["classifierClass"]))

        if not self.params["retrainClassifier"] and os.path.exists(classifierFileName): #enumerate(
            print("Info :: Prediction was done already... loading file")
            self.classifier = joblib.load(classifierFileName)
            return

        
        metricColumnsForPrediction = self.params["metrices"]

        totalColumns = metricColumnsForPrediction + ['Class',"E1E2"] 
        
        data = [self.DB.dfMetrices[analysisName][totalColumns].dropna(subset=metricColumnsForPrediction) for analysisName in self.params["analysisName"]]
        data = pd.concat(data, ignore_index=True)
        dataForTraining = data[["E1E2","Class"] + metricColumnsForPrediction]
        dataForTraining["Class"] = dataForTraining["Class"].astype(np.float64)
        print("Info :: Merging database metrices.")
        print("Test size for classifier: {}".format(self.params["classifierTestSize"]))
        if apexTraining and "apex" in totalColumns:
            print("Info :: Performing apex based pooling.")
            dataForTraining = dataForTraining.sort_values("apex").drop_duplicates("E1E2")
            
        else:
            dataForTraining = dataForTraining.groupby(dataForTraining['E1E2']).aggregate("min")
        dataForTraining['Class'] = dataForTraining['Class'].astype(np.int64)
        dataForTraining = dataForTraining.reset_index()
        
        print("Info :: Using a total of {} features for classifier training.".format(dataForTraining.index.size))

        if addImpurity > 0:
            nRows = dataForTraining.index.size
            rowIdx = np.random.choice(nRows,int(nRows * addImpurity),replace=False)#np.random.randint(0,nRows,size=int(nRows * addImpurity))
            print(dataForTraining.loc[rowIdx,'Class'] ^ 1)
            dataForTraining.loc[rowIdx,'Class'] = dataForTraining.loc[rowIdx,'Class'] ^ 1
            print("Warning :: Stop! Using impurity for the training data is not advisable other than for testing. You should probably not do this?")


        Y = dataForTraining['Class'].values
        X = dataForTraining.loc[:,metricColumnsForPrediction].values
        

        self.classifier = Classifier(
            classifierClass = self.params["classifierClass"],
            n_jobs=self.params['n_jobs'], 
            gridSearch = self.params["classiferGridSearch"],
            testSize = self.params["classifierTestSize"])

        probabilites, meanAuc, stdAuc, oobScore, optParams, Y_test, Y_pred = self.classifier.fit(X,Y,kFold=self.params["kFold"],pathToResults=self.params["pathToComb"], metricColumns = metricColumnsForPrediction)
    
        dataForTraining["PredictionClass"] = probabilites

        #save prediction summary
        pathToFImport = os.path.join(self.params["pathToComb"],"PredictorSummary{}_{}.txt".format(self.params["metrices"],self.params["addImpurity"]))
        #create and save classification report
        classReport = classification_report(
                            Y_test,
                            Y_pred,
                            digits=3,
                            output_dict=True)
        classReport = OrderedDict([(k,v) for k,v in classReport.items() if k != 'accuracy'])

        pd.DataFrame().from_dict(classReport, orient="index").to_csv(pathToFImport, sep="\t", index=True)

        #save database prediction
        dataForTraining.to_csv(os.path.join(self.params["pathToComb"],"DBpred({}).txt".format(self.params["addImpurity"])),sep="\t", index=False)
        
        self._plotFeatureImportance(self.params["pathToComb"])

        joblib.dump(self.classifier, classifierFileName)
        self._addMetricToStats("Metrices",str(metricColumnsForPrediction))
        self._addMetricToStats("OOB_Score",oobScore)
        self._addMetricToStats("ROC_Curve_AUC","{}+-{}".format(meanAuc,stdAuc))
        self._addMetricToStats("ClassifierParams",optParams)
        
        print("DB prediction saved - DBpred.txt :: Classifier pickled and saved 'trainedClassifier.sav'")


    def _loadPairsForPrediction(self):
        ""
        #load chunks that were saved
        chunks = [f for f in os.listdir(os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"chunks")) if f.endswith(".npy") and f != "source.npy"]
        print("\nInfo :: Prediction/Dimensional reduction started...")
        for chunk in chunks:
            X = np.load(os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"chunks",chunk),allow_pickle=True)
            yield (X,len(chunks))

    def _predictInteractions(self):
        ""
        if self.params["noDatabaseForPredictions"] or self.params["noDistanceCalculationAndPrediction"]:
            print("Info :: Skipping predictions. (noDatabaseForPredictions = True or noDistanceCalculationAndPrediction = True)")
            return
        paramDict = {"NumberInteractions" : 0, "positiveInteractors" : 0, "decoyInteractors" : 0, "novelInteractions" : 0, "interComplexInteractions" : 0}
        probCutoffs = dict([(cutoff,paramDict.copy()) for cutoff in np.linspace(0.0,0.99,num=30)])

        print("Info :: Starting prediction ..")
        folderToOutput = os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"result")
        pathToPrediction = os.path.join(folderToOutput,"predictedInteractions{}_{}_{}.txt".format(self.params["metricesForPrediction"],self.params["classifierClass"],self.params["addImpurity"]))
        if False and not self.params["retrainClassifier"] and os.path.exists(pathToPrediction):
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
         
        metricIdx = [n + 4 if "apex" in self.params["metrices"] else n + 3 for n in range(len(self.params["metrices"]))] #in order to extract from dinstances, apex creates an extra column (apex_dist)
        

        for n,(X,nChunks) in enumerate(self._loadPairsForPrediction()):
            boolSelfIntIdx = X[:,0] == X[:,1] 
            if n % 5 == 0:
                percDone = round(n/nChunks*100,1)
                print(percDone,r"%")
            X = X[boolSelfIntIdx == False]
            #first two rows E1 E2, and E1E2, apexPeakDist remove before predict
            if X.shape[0] > 0:
                classProba = self.classifier.predict(X[:,metricIdx])
            else:
                continue
            if classProba is None:
                continue
            predX = np.append(X,classProba.reshape(X.shape[0],-1),axis=1)
            interactionClass = self.DB.getInteractionClassByE1E2(X[:,2],X[:,0],X[:,1])
            
            for cutoff in probCutoffs.keys():
                
                boolPredIdx = classProba >= cutoff
                
                if len(boolPredIdx.shape) > 1:
                    boolIdx = np.sum(boolPredIdx,axis=1) == self.params["kFold"]
                else:
                    boolIdx = boolPredIdx

                counts = interactionClass.loc[boolIdx].value_counts()
                n = np.sum(boolIdx)
                probCutoffs[cutoff]["NumberInteractions"] += n
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
                predInteractions = np.append(predInteractions,predX[boolIdx,:], axis=0)

        probData = pd.DataFrame().from_dict(probCutoffs, orient="index")
        probData["FalseNegatives"] = probData["positiveInteractors"].iloc[0] - probData["positiveInteractors"]
        probData["precision"] = (probData["positiveInteractors"]) / (probData["positiveInteractors"] + probData["interComplexInteractions"] + probData["decoyInteractors"])
        probData["recall"] = (probData["positiveInteractors"]) / (probData["positiveInteractors"] + probData["FalseNegatives"])
        probData["F1-measure"] = 2 * ((probData["precision"] * probData["recall"]) / (probData["precision"] + probData["recall"]))
        probData["F-measure(b=2)"] = (1+2**2) * ((probData["precision"] * probData["recall"]) / (((2**2) * probData["precision"]) + probData["recall"]))
        probData["F-measure(b=0.5)"] = (1+0.5**2)* ((probData["precision"] * probData["recall"]) / (((0.5**2) * probData["precision"]) + probData["recall"]))
        
        self.params["interactionProbabCutoff"] = float(probData.idxmax().loc["F1-measure"])
        print("Info :: Interaction probability was set to: {} based on the F-metric using beta = 1.".format(self.params["interactionProbabCutoff"] ))
        # boolPredIdx = classProba >= self.params["interactionProbabCutoff"]
        # if len(boolPredIdx.shape) > 1:
        #     boolIdx = np.sum(boolPredIdx,axis=1) == self.params["kFold"]
        # else:
        #     boolIdx = boolPredIdx

        probData.to_csv(os.path.join(folderToOutput,"classiferPerformanceMetrics_{}_addImp{}.txt".format(self.params["classifierClass"],self.params["addImpurity"])),sep="\t")

        # print("Interactions > cutoff :", predInteractions.shape[0])
        # print("Info :: Finding interactions in DB")
        # boolDbMatch = np.isin(predInteractions[:,2],self.DB.df["E1E2"].values, assume_unique=True)
        # print("Info :: Appending matches.")
        # predInteractions = np.append(predInteractions,boolDbMatch.reshape(predInteractions.shape[0],1),axis=1)
       
        d = pd.DataFrame(predInteractions, columns = dfColumns)
        
        print("Info :: Number of interactions detected: {} at cut-off {}".format(d.index.size,self.params["interactionProbabCutoff"]))

        boolDbMatch = d["In DB"] == "pos"
        print("Info :: Annotate complexes to pred. interactions.")
        d["ComplexID"], d["ComplexName"] = zip(*[self._attachComplexID(_bool,E1E2) for E1E2, _bool in zip(predInteractions[:,2], boolDbMatch)])

        d = self._attachPeakIDtoEntries(d)
       # boolIdx = d[pColumns[0]] > self.params["interactionProbabCutoff"]
       # d = d.loc[boolIdx]
        origSize = d.index.size
        print("Info : Filter for at least {} times in predicted interactions".format(self.params["minimumPPsPerFeature"]))
        if self.params["usePeakCentricFeatures"]:
            eColumns = ["E1p","E2p"]
        else:
            eColumns = ["E1","E2"]

        Es = pd.Series(d[eColumns].values.flatten())
        EsCounted = Es.value_counts()
        boolIdx = EsCounted >= self.params["minimumPPsPerFeature"]
        duplicatedPPs = EsCounted.index[boolIdx]
        
        d = d.loc[d[eColumns].isin(duplicatedPPs).all(axis="columns")]
        print("Removed interactions {}".format(origSize-d.index.size))


        d.to_csv(pathToPrediction, sep="\t", index=False)

        

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
        "util fn"
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
        
        entryPositiveComplex = [self.DB.assignComplexToProtein(str(e),complexMemberIds,"ComplexID") for e in complexDf.index]
           
        complexDf.loc[:,"ComplexID"] = entryPositiveComplex

        matchingResults = pd.DataFrame(columns = ["Entry","Cluster Labels","Complex ID", "NumberOfInteractionsInDB"])
        clearedEntries = pd.Series([x.split("_")[0] for x in complexDf.index], index=complexDf.index)
        for c,d in self.DB.indentifiedComplexes.items():

            boolMatch = clearedEntries.isin(d["members"])
            clusters = complexDf.loc[boolMatch,"Cluster Labels"].values.flatten()
            nEntriesMatch = np.sum(boolMatch)
            if nEntriesMatch > 1:
                groundTruth = [c] * nEntriesMatch

                matchingResults = matchingResults.append(pd.DataFrame().from_dict({"Entry":complexDf.index[boolMatch].values,
                                "Cluster Labels" : clusters,
                                "Complex ID": groundTruth,
                                "NumberOfInteractionsInDB" : [d["n"]] * nEntriesMatch}) ,ignore_index=True)
        if not matchingResults.empty:
            
            score = v_measure_score(matchingResults["Complex ID"],matchingResults["Cluster Labels"],beta = beta)
        else:
            score = np.nan
        
        return complexDf , score, matchingResults


    def _clusterInteractions(self, predInts, clusterMethod = "HDBSCAN", plotEmbedding = True, groupFiles = [], combineProbs = True, groupName = ""):
        """
        Performs dimensional reduction and clustering of prediction distance matrix over a defined parameter grid.
        Parameter
            predInts -  ndarray. 
            clusterMethod   -   string. Any string of ["HDBSCAN",]
            plotEmbedding   -   bool. If true, embedding is plotted and save to pdf and txt file. 


        returns
            None
        """
        
        embedd = None
        bestDf = None
        topCorrFeatures = None
        splitLabels = False
        recordScore = OrderedDict()
        saveEmbeddings = []
        maxScore = np.inf 
        metricColumns = [x if not isinstance(x,dict) else x["name"] for x in self.params["metricesForPrediction"]] 
        cb = ComplexBuilder(method=clusterMethod)

        print("\nPredict complexes")
        if predInts is None:
            print("No database provided. UMAP and clustering will be performed using defaultKwargs. (noDatabaseForPredictions = True)")

        pathToFolder = self._makeFolder(self.params["pathToComb"],"complexIdentification_{}".format(self.params["addImpurity"]))

        if not self.params["databaseHasComplexAnnotations"] and not self.params["noDatabaseForPredictions"] and predInts is not None:
            print("Database does not contain complex annotations. Therefore standard UMAP settings are HDBSCAN settings are used for complex identification.")
            cb.set_params(self.params["hdbscanDefaultKwargs"])
            clusterLabels, intLabels, matrix , reachability, core_distances, embedd = cb.fit(predInts, 
                                                                                            metricColumns = metricColumns, 
                                                                                            scaler = self.classifier._scaleFeatures,
                                                                                            umapKwargs=  self.params["umapDefaultKwargs"])
        
        elif self.params["noDistanceCalculationAndPrediction"] or self.params["noDatabaseForPredictions"]:
            print("Info :: No database given for complex scoring. UMAP and HDBSCAN are performed to identify complexes.")
            alignedEmbeddings = OrderedDict() 
            if len(self.Xs) > 1:
                #correlate with each other
                firstKey = list(self.Xs.keys())[0]
                corrDfs = [self.Xs[firstKey].corrwith(df,axis=1,drop=True) for k,df in self.Xs.items() if k != firstKey]
                
                mergedDf = pd.concat(corrDfs,join="inner",axis=1).mean(axis=1).sort_values(ascending=False)
                topCorrFeatures = mergedDf.head(self.params["topNCorrFeaturesForUMAPAlignment"]).index
                
                dataSets = [minMaxNorm(X.values,axis=1) for X in self.Xs.values()]
                relations = []
                for k,v in self.Xs.items():
                    if k != firstKey:
                        relationDict = dict([(self.Xs[prevKey].index.get_loc(idx),v.index.get_loc(idx)) for idx in topCorrFeatures])
                        relations.append(relationDict)
                    prevKey = k 
                print("Info :: Computing aligned UMAP using top correlated features.")
                aligned_mapper = umap.aligned_umap.AlignedUMAP(**self.params["umapDefaultKwargs"]).fit(dataSets, relations=relations)
                
                for n,umapE in enumerate(aligned_mapper.embeddings_):
                    key = list(self.Xs.keys())[n]
                    df = pd.DataFrame(umapE, index=self.Xs[key].index, columns = ["E({})_0".format(key),"E({})_0".format(key)])
                    alignedEmbeddings[key] = df.copy()
                    

            for analysisName in self.params["analysisName"]:
                if self.params["useRawDataForDimensionalReduction"]:
                    print("Info :: Using raw intensity data for dimensional reduction. Not calculated distances")
                    if self.params["scaleRawDataBeforeDimensionalReduction"]:
                        X = self.Xs[analysisName]
                        
                        predInts = pd.DataFrame(minMaxNorm(X.values,axis=1), index=X.index, columns = ["scaled_({})_{}".format(analysisName,colName) for colName in X.columns]).dropna()
                        
                    else:
                        predInts = self.Xs[analysisName]
                    
                    
                    cb.set_params(self.params["hdbscanDefaultKwargs"])
                    clusterLabels, intLabels, matrix , reachability, core_distances, embedd, pooledDistances = cb.fit(predInts, 
                                                                                                    preCompEmbedding = alignedEmbeddings[analysisName] if analysisName in alignedEmbeddings else None,
                                                                                                    metricColumns = self.X.columns, 
                                                                                                    scaler = None,
                                                                                                    umapKwargs =  self.params["umapDefaultKwargs"],
                                                                                                    generateSquareMatrix = False,
                                                                                                    )

                    df = pd.DataFrame().from_dict({"Entry":intLabels,"Cluster Labels":clusterLabels,"reachability":reachability,"core_distances":core_distances})
                    df = df.sort_values(by="Cluster Labels")
                    df = df.set_index("Entry")

                    predInts.to_csv(os.path.join(pathToFolder,"predInts_{}.txt".format(analysisName)))

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
                    df = pd.DataFrame().from_dict({"Entry":intLabels,"Cluster Labels({})".format(analysisName):clusterLabels,"reachability":reachability,"core_distances":core_distances})
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
                    dfEmbed = pd.DataFrame(embedd, columns = ["UMAP_{}_0{}".format(analysisName,n) for n in range(embedd.shape[1])])
                    dfEmbed["clusterLabels({})".format(analysisName)] = clusterLabels
                    dfEmbed["labels({})".format(analysisName)] = intLabels
                    if splitLabels:
                        dfEmbed["sLabels"] = dfEmbed["labels"].str.split("_",expand=True).values[:,0]
                        dfEmbed = dfEmbed.set_index("sLabels")
                    else:
                        dfEmbed = dfEmbed.set_index("labels({})".format(analysisName))

                    if self.params["scaleRawDataBeforeDimensionalReduction"] and self.params["useRawDataForDimensionalReduction"]:
                        dfEmbed = dfEmbed.join([self.Xs[self.currentAnalysisName],predInts],lsuffix="_",rsuffix="__")
                    else:
                        dfEmbed = dfEmbed.join(self.Xs[self.currentAnalysisName])
                    if topCorrFeatures is not None:
                        dfEmbed["FeatureForUMAPAlign"] = dfEmbed.index.isin(topCorrFeatures)
                    saveEmbeddings.append(dfEmbed)
                    dfEmbed.to_csv(os.path.join(pathToFolder,"UMAP_Embedding_{}.txt".format(analysisName)),sep="\t")
                    #plot embedding.
                    fig, ax = plt.subplots()
                    ax.scatter(embedd[:,0],embedd[:,1],s=12, c=clusterLabels, cmap='Spectral')
                    plt.savefig(os.path.join(pathToFolder,"E({}).pdf".format(analysisName)))
                    plt.close()

            pd.concat(saveEmbeddings,axis=1).to_csv(os.path.join(pathToFolder,"concatEmbeddings.txt"),sep="\t")



        else:
            embedd = None
            if len(groupFiles) > 0:
                groupMetricColumns = ["Prob_0_({})".format(analysisName) for analysisName in groupFiles] 
               # print(groupMetricColumns)
            usePeaks = self.params["usePeakCentricFeatures"]
            print("Using peaks for clustering.")
            print(groupMetricColumns)
            if usePeaks:
               # if len(groupFiles) > 0:
                eColumns = ["E1p_({})".format(groupFiles[0]),"E2p_({})".format(groupFiles[0])] 
                predInts = predInts[groupMetricColumns + eColumns + ["E1E2"]]
                
            else:
                predInts = predInts[groupMetricColumns + ["E1","E2","E1E2"]]
                eColumns = ["E1","E2"]
            #
            predInts.dropna(subset=groupMetricColumns,inplace=True,thresh=1)
            for n, params in enumerate(list(ParameterGrid(CLUSTER_PARAMS[clusterMethod]))):
                try:
                    cb.set_params(params)
                    if clusterMethod == "HDBSCAN":
                        clusterLabels, intLabels, matrix , reachability, core_distances, embedd, pooledDistances = cb.fit(predInts, 
                                                                                            metricColumns = groupMetricColumns,#,#[colName for colName in predInts.columns if "Prob_" in colName], 
                                                                                            scaler = None,#self.classifier._scaleFeatures, #
                                                                                            inv = True, # after pooling by poolMethod, invert (1-X)
                                                                                            poolMethod="max",
                                                                                            preCompEmbedding = None,
                                                                                            entryColumns = eColumns
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
                df = df.sort_values(by=["Cluster Labels"])
                if usePeaks:
                    df["E"] = df["Entry"].str.split("_",expand=True)[0]
                    df = df.set_index("E")
                else:
                    df = df.set_index("Entry")


            # clusteredComplexes = df[df["Cluster Labels"] != -1]
                df, score, matchingResults = self._scoreComplexes(df)
                
            # df = df.join(assignedIDs[["ComplexID"]])
                if True:#maxScore > score:
                    df.to_csv(os.path.join( pathToFolder,"Complexes:{}_{}_{}.txt".format(groupName,n,score)),sep="\t")
                    matchingResults.to_csv(os.path.join( pathToFolder,"ComplexPerEntry(ScoreCalc):{}_{}_{}.txt".format(groupName,n,score)),sep="\t")
                    print("Info :: Current best params ... ")
                   
                    # squaredDf = pd.DataFrame(matrix,columns=df.index,index=df.index).loc[df.index,df.index]
                    # squaredDf.to_csv(os.path.join(pathToFolder,"SquaredSorted{}_{}.txt".format(groupName,n)),sep="\t")
                    # if usePeaks:
                    #     noNoiseIndex = df["Entry"].loc[df["Cluster Labels"] > 0]
                    # else:
                    #     noNoiseIndex = df.index[df["Cluster Labels"] > 0]
                    # squaredDf.loc[noNoiseIndex,noNoiseIndex].to_csv(os.path.join(pathToFolder,"NoNoiseSquaredSorted_{}_{}.txt".format(groupName,n)),sep="\t")
                    maxScore = score
                    bestDf = df
                    self._plotComplexProfiles(bestDf, pathToFolder, str(n))

                if embedd is not None and plotEmbedding:
                    #save embedding
                    umapColumnNames = ["UMAP_{}".format(n) for n in range(embedd.shape[1])]
                    dfEmbed = pd.DataFrame(embedd, columns = umapColumnNames)
                    embedd = dfEmbed[umapColumnNames]
                    dfEmbed["clusterLabels"] = clusterLabels
                    if usePeaks:
                        dfEmbed["Ep"] = intLabels
                        dfEmbed["Entry"] = [x.split("_")[0] for x in intLabels]
                    else:
                        dfEmbed["Entry"] = intLabels
                    dfEmbed = dfEmbed.set_index("Entry")
                    dfEmbed.loc[dfEmbed.index,"ComplexID"] = df["ComplexID"].loc[dfEmbed.index]
                    rawDataMerge = [self.Xs[analysisName] for analysisName in groupFiles]
                    
                    if n == 0:
                        for sampleN,fileName in enumerate(groupFiles):
                            rawDataMerge[sampleN].columns = ["{}_({}):F{}".format(colName,fileName,sampleN) for colName in rawDataMerge[sampleN].columns]
                    dfEmbed = dfEmbed.join(other = rawDataMerge)
                    dfEmbed.to_csv(os.path.join(pathToFolder,"UMAP_Embeding_{}_{}_{}.txt".format(n,params,groupName)),sep="\t")
                    
                    #plot embedding.
                    fig, ax = plt.subplots()
                    ax.scatter(embedd["UMAP_0"].values, embedd["UMAP_1"].values,s=50, c=clusterLabels, cmap='Spectral')
                    plt.savefig(os.path.join(pathToFolder,"UMAP_Embedding_{}_n{}.pdf".format(groupName,n)))
                    plt.close()

                recordScore[n] = {"score":score,"params":params}
        

    def _loadAndFilterDistanceMatrix(self):
        """
        
        Output to disk: 'highQualityInteractions(..).txt
        However they are just the ones that show the lowest distance metrices.

        Parameters
        ----------

        Returns
        -------
        None

        """
        metricColumns = [x if not isinstance(x,dict) else x["name"] for x in self.params["metrices"]] 
        dfColumns = ["E1","E2","E1E2","apexPeakDist"] + metricColumns
        q = None
        df = pd.DataFrame(columns = dfColumns)
        filteredExisting = False
        pathToFile = os.path.join(self.params["pathToComb"],"highQualityInteractions({}).txt".format(self.currentAnalysisName))
        
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
                
        
        print("Info :: {} total pairwise protein-protein pairs at any distance below 10% quantile.".format(df.index.size))
        df = self._attachPeakIDtoEntries(df)
        
        df.to_csv(pathToFile, sep="\t")
        print("Info :: Saving low distance interactions in result folder.")
        return df

    def _plotComplexProfiles(self,complexDf,outputFolder,name):
        """
        Creates line charts as pdf for each profile.
        Chart has two axes, one shows realy values and the bottom one 
        is scaled by normalizing the highest value to one and the lowest to zero.
        Enabled/Disabled by the parameter "plotComplexProfiles". 

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
                        if uniprotID in self.Signals[self.currentAnalysisName]:
                            y = self.Signals[self.currentAnalysisName][uniprotID].Y
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
       
        if self.params["analysisName"] is None:
            analysisName = self._randomStr(50)
        elif isinstance(self.params["analysisName"],list) and n < len(self.params["analysisName"]):
            analysisName = self.params["analysisName"][n]
        else:
            analysisName = str(self.params["analysisName"])

        pathToTmp = os.path.join(".","results")
        
        if not os.path.exists(pathToTmp):
            os.mkdir(pathToTmp)
        self.currentAnalysisName = analysisName

        date = datetime.today().strftime('%Y-%m-%d')
        self.params["Date of anaylsis"] = date
        runName = self.params["runName"] if self.params["runName"] is not None else self._randomStr(3)
        self.params["pathToComb"] = self._makeFolder(pathToTmp,"{}_n({})runs".format(runName,len(self.params["analysisName"])))
        print("Info :: Folder created in which combined results will be saved: " + self.params["pathToComb"])
        pathToTmpFolder = os.path.join(self.params["pathToComb"],analysisName)
        if os.path.exists(pathToTmpFolder):
            print("Info :: Path to results folder exsists")
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
            print("Info :: Result folder created -- ",analysisName)
            self._makeFolder(pathToTmpFolder,"chunks")
            print("Info :: Chunks folder created/checked")
            self._makeFolder(pathToTmpFolder,"result")
            print("Info :: Result folder created/checked")
            self._makeFolder(pathToTmpFolder,"result","alignments")
            print("Info :: Alignment folder created/checked")
            self._makeFolder(pathToTmpFolder,"result","modelPlots")
            print("Info :: Result/modelPlots folder created/checked. In this folder, all model plots will be saved here, if savePlots equals true, otherwise empty.")
           # self._createTxtFile(pathToFile = os.path.join(pathToTmpFolder,"runTimes.txt"),headers = ["Date","Time","Step","Comment"])

            return pathToTmpFolder
        except OSError as e:
            print(e)
            raise OSError("Could not create result folder due to OS Error")
            

    def _handleComptabFormat(self,X,filesToLoad):
        """
        Extracts different samples from comptab format.

        Parameters
        ----------
        X : str
            Path to folder where comptab files is located

        filesToLoad:
            list of txt/tsv files present in the folder

        Returns
        -------
        detectedDataFrames : list of pd.DataFrame
            list of identified data farmes from compbat file

        fileNames : list of str
            Internal names <comptabfileName>:<sampleName>

        """
        detectedDataFrames = []
        fileNames = []
        for fileName in filesToLoad:
            comptFile = pd.read_csv(os.path.join(X,fileName), sep="\t", header=[0,1], index_col=0)
            columnsToKeep = [colNameTuple for colNameTuple in comptFile.columns if "unique peptides" not in colNameTuple and "coverage" not in colNameTuple and "protein length" not in colNameTuple]
            comptFile = comptFile[columnsToKeep]
            #find unique sample names given in the first header
            samples = np.unique([colNameTuple[0] for colNameTuple in comptFile.columns])
       
            for sampleName in samples:
                sampleColumns = [colNameTuple for colNameTuple in comptFile.columns if colNameTuple[0] == sampleName]
                dataFrame = pd.DataFrame(comptFile[sampleColumns].values, 
                        columns = [colNameTuple[1] for colNameTuple in comptFile.columns if colNameTuple[0] == sampleName])
                dataFrame["Uniprot ID"] = comptFile.index
                detectedDataFrames.append(dataFrame)
                fileNames.append("{}:{}".format(fileName,sampleName))
 
        return detectedDataFrames, fileNames


    def _mergeDistancesForGroups(self):
        ""






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
        self.entriesInChunkLoaded = False

        global entriesInChunks

        if isinstance(X,list) and all(isinstance(x,pd.DataFrame) for x in X):
            if self.params["compTabFormat"]:
                raise TypeError("If 'compTabFormat' is True. X must be a path to a folder. Either set compTabFormat to False or provide a path.")
            print("Multiple dataset detected - each one will be analysed separetely")
            if self.params["analysisName"] is None or not isinstance(self.params["analysisName"],list) or len(self.params["analysisName"]) != len(X):
                self.params["analysisName"] = [self._randomStr(10) for n in range(len(X))] #create random analysisNames
                print("Info :: 'anylsisName' did not match X shape. Created random strings per dataframe.")

        elif isinstance(X,str):
            if os.path.exists(X):
                loadFiles = [f for f in os.listdir(X) if f.endswith(".txt") or f.endswith(".tsv")]
                
                if self.params["compTabFormat"]:
                    Xs, loadFiles = self._handleComptabFormat(X,loadFiles)
                else:
                    Xs = [pd.read_csv(os.path.join(X,fileName), sep="\t") for fileName in loadFiles]
                   # filterId = pd.read_csv(os.path.join("filter","Mito_d3.txt"),index_col=None)
                   # Xs = [X.loc[X["Uniprot ID"].isin(filterId["Uniprot ID"].values)] for X in Xs]

            
                self.params["analysisName"] = loadFiles
                if maxValueToOne:
                    maxValues = pd.concat([x.max(axis=1) for x in X], axis=1).max(axis=1)
                    normValueDict = dict([(X[0][self.params["idColumn"]].values[n],maxValue) for n,maxValue in enumerate(maxValues.values)])
                    self.params["normValueDict"] = normValueDict
            else:
                raise ValueError("Provided path {} does not exist.".format(X))

        elif isinstance(X,pd.DataFrame):
            Xs = [X]
            self.params["analysisName"] = [self._randomStr(10)]
        else:
            ValueError("X must be either a string, a list of pandas data frames or pandas data frame itself.")
        
        self.params["pathToTmp"] = {}
        statColumns = ["nInteractions ({})".format(self.params["interactionProbabCutoff"]),"nPositiveInteractions","OOB_Score","ROC_Curve_AUC","Metrices","Classifier","ClassifierParams"]
        self.stats = pd.DataFrame(index = self.params["analysisName"],columns = statColumns)
        
        self.params["rawData"] = {}
        self.params["runTimes"] = {}
        
        self.params["runTimes"]["StartTime"] = time.time() 
        

        for n,X in enumerate(Xs):

            pathToTmpFolder = self._makeTmpFolder(n)
            self.params["pathToTmp"][self.currentAnalysisName] = pathToTmpFolder
            
            if n == 0:
                pathToParams = os.path.join(self.params["pathToComb"],"params.json")
                pd.DataFrame().from_dict(self.params,orient="index").sort_index().to_json(pathToParams,indent = 4, orient="columns")
                print("Info :: Parameters saved to output folder.")
            
            if os.path.exists(os.path.join(self.params["pathToComb"],"runTimes.txt")):
                if not self.params["restartAnalysis"] and not self.params["recalculateDistance"] and not self.params["retrainClassifier"]:
                    print("Warning :: Analysis done. Aborting (detected by finding the file 'runTimes.txt'")
                    return 

            print("------------------------")
            print("--"+self.currentAnalysisName+"--")
            print("--------Started---------")
            print("--Signal Processing  &--")
            print("------Peak Fitting------")
            print("------------------------")
            

            if pathToTmpFolder is not None:
                
                #loading data
                self._load(X)
                #self._checkGroups()
                self._findPeaks(self.params["n_jobs"])
                self._collectRSquaredAndFitDetails()
                
        self._saveSignals()
        self._combinePeakResults()
        self._attachQuantificationDetails()
        
        endSignalTime = time.time()

        self.params["runTimes"]["SignalFitting&Comparision"] = time.time() - self.params["runTimes"]["StartTime"]

        if not self.params["justFitAndMatchPeaks"]:
            
            print("Info :: Peak modeling done. Starting with distance calculations and predictions (if enabled)..")
            
            
            self._createSignalChunks()
            for n,X in enumerate(X):
                if n < len(self.params["analysisName"]): #happnes if others than txt file are present
                    self.currentAnalysisName = self.params["analysisName"][n]
                    print(self.currentAnalysisName," :: Starting distance calculations.")
                    
                    self._calculateDistance()
    

            self._mergeDistancesForGroups()

            self.params["runTimes"]["Distance Calculation"] = time.time() - endSignalTime
            distEndTime = time.time()
            self._loadReferenceDB()

            for analysisName in self.params["analysisName"]:
                self._addMetricesToDB(analysisName)
            dataPrepEndTime = time.time()
            self.params["runTimes"]["Database Preparation"] = dataPrepEndTime - distEndTime
            
            self._trainPredictor(self.params["addImpurity"])

            for analysisName in self.params["analysisName"]:
                
                self.currentAnalysisName = analysisName
                self._predictInteractions()
                #
            
            #save statistics
            self.stats.to_csv(os.path.join(self.params["pathToComb"],"statistics.txt"),sep="\t")

            #combine interactions
            if not self.params["noDistanceCalculationAndPrediction"]:
                if not self.params["noDatabaseForPredictions"]:
                    combinedInteractions = self._combineInteractionsAndClusters()
            else:
                print("Warning/Info :: noDistancenCalculationAndPrediction is True, skipping combineInteraction step.")

            endTrainingTime = time.time()
            self.params["runTimes"]["Classifier Training & Prediction"] = endTrainingTime - dataPrepEndTime


            if not self.params["noDistanceCalculationAndPrediction"] and len(self.params["grouping"]) > 0 and not self.params["noDatabaseForPredictions"]:   
                for groupName,groupFileNames in self.params["grouping"].items():
                    if isinstance(groupFileNames,str):
                        groupFileNames = [groupFileNames]
                    self._clusterInteractions(combinedInteractions,groupFiles = groupFileNames,groupName = groupName)
            else:
                print("Info :: Cluster Interactions")
                self._clusterInteractions(None)


            self.params["runTimes"]["Interaction Clustering and Embedding"] = time.time() - endTrainingTime
        print("Info :: Run Times :: ")
        print(self.params["runTimes"])
        pd.DataFrame().from_dict(self.params["runTimes"],orient="index").to_csv(os.path.join(self.params["pathToComb"],"runTimes.txt"),sep="\t")
        print("Info :: Analysis done.")

    def _combinePredictedInteractions(self, pathToComb):
        """
        Combines predicted Interactions based on the output
        files : predictedInteractions[..].txt of each run. 

        Parameters
        ----------
        pathToComb : str, path to combined result folder. 

        Returns
        -------
        combResults : pd.DataFrame
            combined data frame for each run. All metrices and predictions are provided. 

        """
        pathToInteractions = os.path.join(pathToComb,"combinedInteractions.txt")
        if False and os.path.exists(pathToInteractions) and not self.params["retrainClassifier"]:
            combResults = pd.read_csv(pathToInteractions,sep="\t")
            combResults = self._filterCombinedInteractions(combResults)
            print("Info :: Combined interactions found and loaded.")
            return combResults
        print("Info :: Combining interactions of runs.")
        preditctedInteractions = []
        for analysisName in self.params["analysisName"]:

            pathToResults = os.path.join(self.params["pathToTmp"][analysisName],"result")
            pathToPrediction = os.path.join(pathToResults,"predictedInteractions{}_{}_{}.txt".format(self.params["metricesForPrediction"],self.params["classifierClass"],self.params["addImpurity"]))

            if os.path.exists(pathToPrediction):
                df = pd.read_csv(pathToPrediction,sep="\t", low_memory=False).set_index(["E1E2","E1","E2"])
                df = df.loc[df["Prob_0"] > self.params["interactionProbabCutoff"]]
                preditctedInteractions.append(df)
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
        
            combResults["validSignalFit({})".format(analysisName)]  = combResults[["E1","E2"]].apply(lambda x: all(e in self.Signals[analysisName] and self.Signals[analysisName][e].valid for e in x.values),axis=1)
        
        combResults["#Valid Signal Fit"]  = combResults[["validSignalFit({})".format(analysisName) for analysisName in self.params["analysisName"]]].sum(axis=1)
        
        detectedColumn = [analysisName for analysisName in self.params["analysisName"]]
        #detected in grouping
        for groupName,groupItems in self.params["grouping"].items():
            if all(groupItem in combResults.columns for groupItem in groupItems):
                boolIdx = combResults[groupItems] == "+"
            
                if isinstance(boolIdx,pd.Series):
                    #grouping equals 1 (groupItems, nonsenese (always ture), but repoted due to conisitency)
                    combResults["Complete in {}".format(groupName)] = boolIdx
                else:
                    combResults["Complete in {}".format(groupName)] = np.sum(boolIdx,axis=1) == len(groupItems)

        boolIdx = combResults[detectedColumn] == "+"
        combResults["# Detected in"] = np.sum(boolIdx,axis=1)
        combResults.sort_values(by="# Detected in", ascending = False, inplace = True)
       # combResults.loc[combResults["E1E2"].str.contains("A0A087WU95")].to_csv("BiasedSelection.txt",sep="\t")
        combResults.to_csv(pathToInteractions,sep="\t",index=True)
        combResults = self._filterCombinedInteractions(combResults)
    
        return combResults

    def _filterCombinedInteractions(self,combResults):
        """
        Filters combined interactions.

        Parameters
        ----------
        combResults : pd.DataFrame. Combined interactions. 

        Returns
        -------
        combResults : pd.DataFrame
            filteredCombResults

        """
        interactionsInAllSamples = self.params["considerOnlyInteractionsPresentInAllRuns"]
        if isinstance(interactionsInAllSamples,bool) and interactionsInAllSamples:
            filteredCombResults = combResults.loc[combResults["# Detected in"] == len(self.params["analysisName"])]
        elif isinstance(interactionsInAllSamples,int):
            if interactionsInAllSamples > len(self.params["analysisName"]):
                interactionsInAllSamples = len(self.params["analysisName"])
            filteredCombResults = combResults.loc[combResults["# Detected in"] >= interactionsInAllSamples]
        else:
            #if no filtering is applied.
            filteredCombResults = combResults
        return filteredCombResults

    def _combineInteractionsAndClusters(self):
        ""
        pathToComb = self.params["pathToComb"]
        combinedInteractions = self._combinePredictedInteractions(pathToComb)
        return combinedInteractions
        

    def _saveSignalFitStatistics(self):
        "Save Fit Statistic to disk"
        pathToTxt = os.path.join(self.params["pathToTmp"][self.currentAnalysisName],"result","fitStatistic({}).txt".format(self.currentAnalysisName))
        data = [{
                "id":signal.ID,
                "R2":signal.Rsquared,
                "#Peaks":len(signal.modelledPeaks) if hasattr(signal,"modelledPeaks") else 0,
                "valid":signal.valid,
                "validModel":signal.validModel,
                "validData":signal.validData
                    } for signal in self.Signals[self.currentAnalysisName].values()]

        pd.DataFrame().from_dict(data).to_csv(pathToTxt,sep="\t")
       
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
            
            alignResults = OrderedDict([(analysisName,[]) for analysisName in self.params["analysisName"]])
            fittedModels = dict()
            removedDuplicates = [X.loc[~X.duplicated(subset=["Key"],keep=False)] for X in fittedPeaksData]
            preparedData = []
            for n,dataFrame in enumerate(removedDuplicates):
                dataFrame.columns = ["{}_{}".format(colName,self.params["analysisName"][n]) if colName != "Key" else colName for colName in dataFrame.columns ]
                dataFrame = dataFrame.set_index("Key")
                preparedData.append(dataFrame)
            #join data frames
            joinedDataFrame = preparedData[0].join(preparedData[1:],how="outer")
            if joinedDataFrame .index.size < 30:
                print("Less than 30 data profiles with single peaks found. Aborting alignment")
                return fittedPeaksData
            #use linear regression or lowess
           
            for comb in combinations(self.params["analysisName"],2):
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
            for analysisName in self.params["analysisName"]:
                if analysisName != maxR2SumRun:
                    columnHeaders = ["Center_{}".format(maxR2SumRun),"Center_{}".format(analysisName)]
                    data = joinedDataFrame.dropna(subset=columnHeaders)[columnHeaders]
                    absDiff = data[columnHeaders[0]] - data[columnHeaders[1]]
                    diffs[analysisName] = absDiff
                    diffs["c({})".format(analysisName)] = data[columnHeaders[0]]

            fig, ax = plt.subplots(len(self.params["analysisName"]))
            for n,analysisName in enumerate(self.params["analysisName"]):
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
        """
        Combine Peak results. For each run, each signal profile per feature
        is represented by an ensemble of peaks. This function matches
        the peaks using a maximimal distance of 1.8 by default defined 
        by the parameter 'maxPeakCenterDifference'. 

        Peak height or area under curve are compared using a t-test and or an ANOVA.

        Parameters
        ----------
        

        Returns
        -------
        combResults : pd.DataFrame
            filteredCombResults

        """
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
        for analysisName in self.params["analysisName"]:
            suffixedColumns.extend(["{}_{}".format(colName,analysisName) for colName in columnsForPeakFit])
            tmpFolder = self.params["pathToTmp"][analysisName]#os.path.join(".","tmp",analysisName)
            resultsFolder = os.path.join(tmpFolder,"result")
            fittedPeaks = os.path.join(resultsFolder,"fittedPeaks.txt")
            if os.path.exists(fittedPeaks):
                data = pd.read_csv(fittedPeaks,sep="\t")
                if self.params["keepOnlySignalsValidInAllConditions"]:
                    validSignals = list(self.Signals[analysisName].keys())
                    boolIdx = data["Key"].isin(validSignals)
                    fittedPeaksData.append(data.loc[boolIdx])
                else:
                    fittedPeaksData.append(data)
        if alignRuns:
            print("Info :: Aligning runs started.")
            fittedPeaksData = self._alignProfiles(fittedPeaksData)

        uniqueKeys = np.unique(np.concatenate([x["Key"].unique().flatten() for x in fittedPeaksData]))
        print("Info :: {} unique keys detected".format(uniqueKeys.size))

        print("Info :: Combining peaks using max peak center diff of {}".format(self.params["maxPeakCenterDifference"]))
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
                        df.columns = [colName if colName == "Key" else "{}_{}".format(colName,self.params["analysisName"][n]) for colName in df.columns.values.tolist()]
                        d = d.append(df)
                    else:
                        meanCenters = d[[colName for colName in d.columns if "Center" in colName]].mean(axis=1)
                        idx = meanCenters.index
                        if idx.size == 0:
                            df.columns = [colName if colName == "Key" else "{}_{}".format(colName,self.params["analysisName"][n]) for colName in df.columns.values.tolist()]
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
                        df.columns = [colName if colName == "Key" else "{}_{}".format(colName,self.params["analysisName"][n]) for colName in df.columns.values.tolist()]
                        targetColumns = [colName for colName in df.columns if self.params["analysisName"][n] in colName]
                        d = pd.merge(d,df[targetColumns], how="outer",left_index=True,right_index=True)
                       
                        d["Key"] = [uniqueKey] * d.index.size

                concatDataFrames.append(d)

            if len(concatDataFrames) == 0 or all(X.empty for X in concatDataFrames):
                print("Warning - no matching keys found. Aborting.")
                return
            data = pd.concat(concatDataFrames,axis=0, ignore_index=True)
            #peakPropColumns = [col for col in data.columns if col.split("_")[0] in columnsForPeakFit]
            
            data.sort_index(axis=1, inplace=True)
           # print(data)
            #perform t-test
            grouping = self.params["grouping"]
            if len(grouping) > 1:
                groups = list(self.params["grouping"].keys())  
               # print(groups)          
                groupComps =  list(combinations(groups,2))
               # print(groupComps)
                resultColumnNames = ["t({})_({})".format(group1,group0) for group0,group1 in groupComps]
                for n,(group0, group1) in enumerate(groupComps):
                    print("Comparing {} vs {}".format(group0,group1))
                    sampleNames1 = grouping[group0] #get file names for group
                    sampleNames2 = grouping[group1] #get file names for group
                    columnNames1 = ["auc_{}".format(sampleName) for sampleName in sampleNames1 if "auc_{}".format(sampleName) in data.columns]
                    columnNames2 = ["auc_{}".format(sampleName) for sampleName in sampleNames2 if "auc_{}".format(sampleName) in data.columns]
                    if any(len(x) < 2 for x in [columnNames1,columnNames2]):
                        print("Grouping not found in data or less than 2, correct suffixes? Skipping t-test..")
                        continue
                    X = np.log2(data[columnNames1].replace(0,np.nan).values)
                    Y = np.log2(data[columnNames2].replace(0,np.nan).values)
                    data[["log2({})".format(colName) for colName in columnNames1]] = X
                    data[["log2({})".format(colName) for colName in columnNames2]] = Y
                    if len(columnNames1) > 1 and len(columnNames2) > 1:
                        t, p = ttest_ind(X,Y,axis=1,nan_policy="omit",equal_var = True)
                    
                        data["-log10-p-value:({})".format(resultColumnNames[n])] = np.log10(p) * (-1)
                        data["T-stat:({})".format(resultColumnNames[n])] = t
                        data["diff:({})".format(resultColumnNames[n])] = np.nanmean(Y,axis=1) - np.nanmean(X,axis=1) 
                    elif len(columnNames1) == 1 or len(columnNames2) == 1:
                        data["diff:({})".format(resultColumnNames[n])] = np.nanmean(Y,axis=1) - np.nanmean(X,axis=1) 
            if len(grouping) > 2 and all(len(groupNames)>1 for groupNames in grouping.values()):
                testGroupData = [np.log2(data[groupNames].replace(0,np.nan).values) for groupNames in grouping.values()]
                F,p = f_oneway(*testGroupData,axis=1)
                data["-log10-p-value:(1W-ANOVA)"] = np.log10(p) * (-1)
                data["Fvalue-1W-ANOVA)"] = F

            data.to_csv(txtOutput,sep="\t",index=False)
        else:
            data = pd.read_csv(txtOutput,sep="\t")

      
        


if __name__ == "__main__":
            #
    ComplexFinder(
        addImpurity=0.0,
        considerOnlyInteractionsPresentInAllRuns = 1,
        compTabFormat = False,
        restartAnalysis = False,
        recalculateDistance  = True,
        retrainClassifier = False,
        minPeakHeightOfMax= minH,
        takeRondomSampleFromData = False,
        justFitAndMatchPeaks = False,
        noDistanceCalculationAndPrediction = False,
        runName = "myFristRun",
        noDatabaseForPredictions=False, 
        rollingWinType = "triang",
        #  databaseFileName="HUMAN_COMPLEX_PORTAL.txt",
        #  databaseIDColumn= "Expanded participant list",
        # databaseEntrySplitString = "|",
        grouping = {"D0":["D0_aebersold.txt"]},
        n_jobs = 12,
        databaseFilter = {'Organism': ["Mouse"]},
        peakModel = peakModel,
        indexIsID =False,
        decoySizeFactor= 1.1,
        classifierClass="random forest",
        minDistanceBetweenTwoPeaks = minDistPeaks,
        smoothWindow = 3,
        classifierTestSize = 0.20,
        plotSignalProfiles = False,
        correlationWindowSize = 8,
        interactionProbabCutoff = 0.8,
        usePeakCentricFeatures = True, ## careful!
        removeSingleDataPointPeaks=True, 
        keepOnlySignalsValidInAllConditions = True,
        useRawDataForDimensionalReduction = False).run("../example-data/D0")
                    
                #  n = 1
               
                    





    
