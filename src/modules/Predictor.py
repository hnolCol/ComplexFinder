
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier



from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, AgglomerativeClustering, AffinityPropagation
import os 
from scipy.spatial.distance import squareform
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
from joblib import Parallel, delayed, dump, load
import umap
import hdbscan 



def chunks(l, n):
    """
    Iterator for chunks of numpy array (row wise). 

    Parameters
    ----------
    l : two-dimensional array
        Array which should be separated into chunks
    n : int
        Number of chunks
    
    Returns
    -------
    Numpy array chunk 

    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Classifier(object):
    
    def __init__(self, classifierClass = "random forest", n_jobs = 4, gridSearch = None, testSize = 0.25):
        """Classifier module for prediction of positive / negative feature interaction

        Note
        ----

        
        Parameters
        ----------
        classifierClass : str
            Name of classifier matching a sklearn classifier name
        n_jobs : int
            Number of workers used by the LokyBackend
        gridSearch : dict or None
            Parameter gridsearch to be used for estimator optimization.

        """

        self.gridSerach = gridSearch if classifierClass != "GaussianNB" else None
        self.n_jobs = n_jobs
        self.classifierClass = classifierClass
        self.testSize = testSize
        
        self.classifier = self._initClassifier()


    def _initClassifier(self):
        """
        Initiate Classifer

        Parameters
        ----------
    
    
        Returns
        -------
        Init Classifier


        Raises
        ------
        ValueError if class argument `classifierClass` unknown.

        """
        if self.classifierClass in ["random_forest","random forest","ensemble tree"]:
            return RandomForestClassifier(n_estimators=200,
                                        oob_score=True,
                                        min_samples_split=2,
                                        n_jobs=self.n_jobs,
                                        random_state=42)
                                                        
        elif self.classifierClass == "SVM":
            
            return SVC(gamma=2, C=1, probability=True)

        elif self.classifierClass == "GradientBoost":

            return GradientBoostingClassifier(n_estimators=200, random_state=42)

        elif self.classifierClass == "GaussianNB":

            return GaussianNB()
        
        elif self.classifierClass == "StackedClassifiers":
            estimators = [("rf",RandomForestClassifier(n_estimators=100, random_state=42)),
                          ("NB",GaussianNB()),
                          ("SVM",SVC(gamma=2, C=1, probability=True))]

            return StackingClassifier(estimators)

        else:
            raise ValueError("Argument `classifierClass` is not known.")

    def _scaleFeatures(self,X):
        """
        Feature scaling. Data are scaled by StandardScaler (0-1)

        Importantly, the scaler is not retrained once it was initiated
        to ensure that the scaling remains similiar for predictors.

        Parameters
        ----------
        X : two dimensional numpy array (feature paris in rows)
            Distance matrix for feature pairs
    
    
        Returns
        -------
        Scaled data of same dimension as X.

        """
        if not hasattr(self,"Scaler"):
            self.Scaler = StandardScaler()
            return self.Scaler.fit_transform(X)
        else:
            return self.Scaler.transform(X)
        

    def _gridOptimization(self,X,Y):
        """
        Classifier optimization based on grid search.

        Parameters
        ----------
        X : two dimensional numpy array (feature paris in rows)
            Distance matrix for feature pairs
        Y : numpy array 
            Array containing class labels of X (0,1)
    
        Returns
        -------
        Best estimator found by the grid search.

        """
        ftwo_scorer = make_scorer(fbeta_score, beta=2)
        gridSearch = GridSearchCV(self.classifier, 
                                    scoring = "f1", 
                                    param_grid = self.gridSerach, 
                                    n_jobs = self.n_jobs, 
                                    cv = 4, 
                                    verbose=1,
                                    refit = True)

        gridSearch.fit(X,Y)

        print("Info :: The maximal F(beta=2) Score was found to be: {}".format(gridSearch.best_score_))

        return gridSearch.best_estimator_, gridSearch.best_params_


    def getFeatureImportance(self):
        """
        Returns estimatore feature imporantance, if estimator allows for this.

        Parameters
        ----------
    

        Returns
        -------
        Array of feature importances (sum = 1)

        """
        

        if hasattr(self.predictors[0],"feature_importances_"):
            return np.array([pred.feature_importances_ for pred in self.predictors])


    def predict(self,X,scale=True):
        """
        Predict class of interaction using predictors. 

        Parameters
        ----------
        X : two dimensional numpy array
            Distance matrix for feature pairs
       scale : bool. Defaults to True.
            Scales data if true. Importantly,
            the scaler is not retrained using the X. 
            The scaler fit is performed when classifier
            is trained. 

        Returns
        -------
        Two dimensional array (n feature pairs x predictors) 
        containing the class proability 
        if predictors (default: 3 - see fit function)

        """
        probas_ = None
        if hasattr(self,"predictors"):
            if scale:
                try:
                    X = self._scaleFeatures(X)
                except Exception as e:
                    raise ValueError("There was a problem in scaling the data. {}".format(e))
            for p in self.predictors:
                if probas_ is None:
                    probas_ = p.predict_proba(X)
                else:
                    probas_ = np.append(probas_,p.predict_proba(X), axis=1)
            if probas_.shape[1] == 2:
                resultClass = probas_[:,1] 
            else:
                resultClass = probas_[:,1::2]
           
            return resultClass

    def fit(self, X, Y, kFold = 3, optimizedParams=None, pathToResults = '', plotROCCurve = True, metricColumns = []):
        """
        Runs grid search estimator optimization

        Parameters
        ----------
        X : two dimensional numpy array
            Distance matrix for feature pairs
        Y : np.array
            Class labels (1 - 0) for postive 
            and negative interaction
        kFold : int
            Number of cross validations. Equals the number of predictors.
        optimizedParams: dict or None
            Already optimized parameters set to the classifier.
        pathToResults : str
            Path to the folder in which the results should be stored.
        plotROCCurve : bool.
            If True a pdf will be created showing the ROC curve of 
            trained classifier with individual k-fold line

        Returns
        -------
        Array of feature importances (sum = 1)

        """
        print("Info :: Predictor training started")
        X = self._scaleFeatures(X)

        X_train, X_test, y_train, y_test = train_test_split(X,Y,stratify=Y,test_size=self.testSize)

        rocCurveData = OrderedDict()
       # xTrain, xTest, yTrain, yTest = train_test_split(X,Y,test_size=0.2)

        if self.gridSerach is not None and optimizedParams is None:
            optimizedClassifier, optimizedParams = self._gridOptimization(X_train,y_train)
        else:
            print("Info :: Grid serach skipped. Automatically skipped when using Guassian NB or parameter 'classiferGridSearch' is None.")
            optimizedClassifier = self.classifier
        #cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
        if optimizedParams is not None:
            print("Info : Optimized parameters")
            print(optimizedParams)

        self.predictors = [optimizedClassifier]
        probasOut = optimizedClassifier.predict_proba(X) 
        #predict probabiliteis for complete data set to create a classfier report.
        tprs = []
        aucs = []
        oobScore = np.nan
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()
        i=0
        
        probas_ = optimizedClassifier.predict_proba(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
        rocCurveData["FPR_{}".format(i)] = fpr
        rocCurveData["TPR_{}".format(i)] = tpr
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i+=1

        if all(hasattr(est,"oob_score_") for est in self.predictors):
            if len(self.predictors) == 1:
                oobScore = np.mean([optimizedClassifier.oob_score_ for classifier in self.predictors])
            else:
                oobScore = optimizedClassifier.oob_score_

        print("Info :: Predictor {} training done.".format(self.classifierClass))
        if plotROCCurve:
            #plotting change line
            ax.plot(    [0, 1], 
                        [0, 1], 
                        linestyle='--', 
                        lw=2, color='r',
                        label='Chance', 
                        alpha=.8)
    
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color='b',
                    label=r'ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                    lw=2, alpha=.8)

            # std_tpr = np.std(tprs, axis=0)
            # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
            #                 label=r'$\pm$ 1 std. dev.')
            plt.legend()
            if  pathToResults != '':
                aucFile = os.path.join(pathToResults,"ROC curve (testSize : {}).pdf".format(self.testSize))
            else:
                aucFile = "ROC curve (testSize : {}).pdf".format(self.testSize)

            plt.savefig(aucFile)
        else:
            mean_auc = np.nan
            std_auc = np.nan
        
        # save ROC cure data
        self.saveToROCCurveData(rocCurveData, pathToResults, metricColumns)

        return probasOut[:,1], mean_auc, std_auc, oobScore, str(optimizedClassifier.get_params()), y_test, optimizedClassifier.predict(X_test)

    def saveToROCCurveData(self,rocCurveData, pathToResults,metricColumns):
        """
        Saves ROC curve data (FPR, TPR) in txt file.

        Parameters
        ----------
        rocCurveData : dict.
            k : `FPR_i` and `TPR_i` 
            v : FPR and TPR values, array
            
        pathToResults : str
            path to which the txt file should be saved.
        
        Returns
        -------
        None. 

        """
        maxValues = np.max([x.size for x in rocCurveData.values()])
        rocData = pd.DataFrame()
        for k,v in rocCurveData.items():
            diff = maxValues - v.size
            if diff > 0:
                fill = np.full(diff, np.nan)
                v = np.concatenate([v,fill])
            rocData[k] = v

        rocData.to_csv(os.path.join(pathToResults,"rocCurveData{}_{}_{}.txt".format(str(metricColumns),self.classifierClass,self.testSize)),sep="\t")

        


class ComplexBuilder(object):


    def __init__(self,method="HDBSCAN"):
        ""
        if method == "OPTICS":
            self.clustering = OPTICS(min_samples=2,metric="precomputed", n_jobs=4)
        elif method == "AGGLOMERATIVE_CLUSTERING":
            self.clustering = AgglomerativeClustering(affinity="precomputed")
        elif method == "AFFINITY_PROPAGATION":
            self.clustering = AffinityPropagation(affinity="precomputed")
        elif method == "HDBSCAN":
            self.clustering = hdbscan.HDBSCAN(min_cluster_size=2)
        self.method = method

    def set_params(self, params):

        self.clustering.set_params(**params) 


    def fit(self, 
                X, 
                metricColumns, 
                scaler = None, 
                inv = False, 
                poolMethod="min", 
                umapKwargs = {"min_dist":1e-7,"n_neighbors":4,"random_state":350}, 
                generateSquareMatrix = True, 
                preCompEmbedding = None, 
                useSquareMatrixForCluster = False,
                entryColumns = ["E1","E2"]):
        """
        Fits predicted interactions to potential macromolecular complexes.


        """
        pooledDistances = None
        if X is not None and generateSquareMatrix and preCompEmbedding is None:
        #  print("Generate Square Matrix ..")
           # print(scaler)
            X, labels, pooledDistances = self._makeSquareMatrix(X, metricColumns, scaler, inv,  poolMethod, entryColumns)
           # print(X)
            print("Info :: Umap calculations started.")
            umapKwargs["metric"] = "precomputed"
            embed = umap.UMAP(**umapKwargs).fit_transform(X)
        elif preCompEmbedding is not None:
            embed = preCompEmbedding.values
            labels = preCompEmbedding.index.values
            pooledDistances = None
            print("Info :: Aligned UMAP was precomputed. ")
        elif not generateSquareMatrix:
            labels = X.index.values
            umapKwargs["metric"] = "correlation"
            embed = umap.UMAP(**umapKwargs).fit_transform(X)
        else:
            raise ValueError("X and preCompEmbedding are both None. No data for UMAP.")

      #  print("done .. - starting clustering")
        if self.method == "OPTICS":
            clusterLabels = self.clustering.fit_predict(X)
            return clusterLabels, labels, X, self.clustering.reachability_[self.clustering.ordering_], self.clustering.core_distances_[self.clustering.ordering_]
        elif self.method in ["AGGLOMERATIVE_CLUSTERING","AFFINITY_PROPAGATION"]:
            clusterResult = self.clustering.fit_predict(X)
            return clusterResult, labels, X, ["None"] * labels.size, ["None"] * labels.size
        elif self.method == "HDBSCAN":
            if useSquareMatrixForCluster:
                self.set_params({"metric":"precomputed"})
                clusterResult = self.clustering.fit(X)
            else:
                clusterResult = self.clustering.fit(embed)
           # self.clustering.condensed_tree_.to_pandas()
            return clusterResult.labels_ , labels, X, clusterResult.probabilities_, ["None"] * labels.size, embed, pooledDistances
        
    def _makeSquareMatrix(self, X, metricColumns, scaler, inv,  poolMethod, entryColumns):
        
        if scaler is None:
            if poolMethod == "mean":
                X["meanDistance"] = X[metricColumns].mean(axis=1)
            elif poolMethod == "max":
                X["meanDistance"] = X[metricColumns].max(axis=1)
            elif poolMethod == "min":
                X["meanDistance"] = X[metricColumns].min(axis=1)
        else:
            if poolMethod == "mean":
                X["meanDistance"] = scaler(X[metricColumns]).mean(axis=1)
            elif poolMethod == "max":
                X["meanDistance"] = scaler(X[metricColumns]).max(axis=1)
            elif poolMethod == "min":
                X["meanDistance"] = scaler(X[metricColumns]).min(axis=1)
            
        if inv:
            X['meanDistance'] = 1 - X['meanDistance']
           
        
        X = X.dropna(subset=["meanDistance"])

        uniqueValues = np.unique(X[entryColumns])
        uniqueVDict = dict([(value,n) for n,value in enumerate(uniqueValues)])
        nCols = nRows = uniqueValues.size 
        print("Info :: Creating {} x {} distance matrix".format(nCols,nCols))
        matrix = np.full(shape=(nRows,nCols), fill_value = 2.0 if scaler is not None else 1.0)
        columnNames = entryColumns+["meanDistance"]
        for row in X[columnNames].values:
            
            nRow = uniqueVDict[row[0]]
            nCol = uniqueVDict[row[1]]
            
            matrix[[nRow,nCol],[nCol,nRow]] = row[2]
        if scaler is not None:
            matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
        np.fill_diagonal(matrix,0)
        
        
        return matrix, uniqueValues, X