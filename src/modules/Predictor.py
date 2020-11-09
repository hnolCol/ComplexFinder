


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
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
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class Classifier(object):
    
    def __init__(self, classifierClass = "random forest", n_jobs = 4, gridSearch = None):
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

        self.gridSerach = gridSearch
        self.n_jobs = n_jobs
        self.classifierClass = classifierClass
        self.classifier = self._initClassifier()


    def _initClassifier(self):
        ""
        if self.classifierClass in ["random_forest","random forest","ensemble tree"]:
            return RandomForestClassifier(n_estimators=200,
                                            oob_score=True,
                                            min_samples_split=2,
                                            n_jobs=self.n_jobs)
        elif self.classifierClass == "SVM":
            
            return SVC(gamma=2, C=1, probability=True)

    def _scaleFeatures(self,X):
        if not hasattr(self,"Scaler"):
            self.Scaler = StandardScaler()
            return self.Scaler.fit_transform(X)
        else:
            return self.Scaler.transform(X)
        

    def _gridOptimization(self,X,Y):

        gridSearch = GridSearchCV(self.classifier, 
                                    scoring = "f1", 
                                    param_grid = self.gridSerach, 
                                    n_jobs = self.n_jobs, 
                                    cv = 8, 
                                    verbose=4)

        gridSearch.fit(X,Y)

        self.bestClassifier = gridSearch.best_estimator_

        return gridSearch.best_params_


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
            return self.predictors[0].feature_importances_


    def predict(self,X,scale=True):
        ""
        probas_ = None
        if hasattr(self,"predictors"):
            if scale:
                try:
                    X = self._scaleFeatures(X)
                except Exception as e:
                    print("scale failed")
                    print(e)
                    return
            for p in self.predictors:
                if probas_ is None:
                    probas_ = p.predict_proba(X)
                else:
                    probas_ = np.append(probas_,p.predict_proba(X), axis=1)

           # resultClass = np.mean(probas_[:,1::2],axis=1)
            resultClass = probas_[:,1::2]
            return resultClass

    def fit(self, X, Y, kFold = 3, optimizedParams=None, pathToResults = '', plotROCCurve = True):
        """
        Runs grid search estimator optimization

        Parameters
        ----------
        X : two dimensional numpy array
            Distance matrix for feature pairs
        Y : np.array
            Class labels (1 - 0) for postive 
            and negative interaction
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
        print("predictor training started")
        X = self._scaleFeatures(X)

        rocCurveData = OrderedDict()
       # xTrain, xTest, yTrain, yTest = train_test_split(X,Y,test_size=0.2)

        if self.gridSerach is not None and optimizedParams is None:
            optimizedParams = self._gridOptimization(X,Y)

        #cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
        if optimizedParams is not None:
            print("Optimized parameters")
            print(optimizedParams)
        cv = StratifiedKFold(n_splits=kFold, shuffle=True)
        self.predictors = []
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()
        i=0
        probasOut = None
        for train, test in cv.split(X, Y):
            classifier_ = self._initClassifier()
            if optimizedParams is not None:
                classifier_.set_params(**optimizedParams)

            classifier_.fit(X[train], Y[train])
            probas_ = classifier_.predict_proba(X[test])
            self.predictors.append(classifier_)

            if hasattr(classifier_,"feature_importances_"):
                print("Feature Importance")
                print(classifier_.feature_importances_)

            fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
            rocCurveData["FPR_{}".format(i)] = fpr
            rocCurveData["TPR_{}".format(i)] = tpr
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, lw=1, alpha=0.3,
                    label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            i+=1

            if probasOut is None: 
                    probasOut = classifier_.predict_proba(X)
            else:
                    probasOut = np.append(probasOut_,classifier_.predict_proba(X), axis=1)

        print("predictor training done")
        if plotROCCurve:
            #plotting
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
                    label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                    lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                            label=r'$\pm$ 1 std. dev.')
            plt.legend()
            if  pathToResults != '':
                aucFile = os.path.join(pathToResults,"ROC curve.pdf")
            else:
                aucFile = "ROC curve.pdf"

            plt.savefig(aucFile)
        
        # save ROC cure data
        meanProbsOut = self.saveToROCCurveData(rocCurveData,probasOut, pathToResults)
        return meanProbsOut

    def saveToROCCurveData(self,rocCurveData, probasOut, pathToResults):
        ""
        maxValues = np.max([x.size for x in rocCurveData.values()])
        rocData = pd.DataFrame()
        for k,v in rocCurveData.items():
            diff = maxValues - v.size
            if diff > 0:
                fill = np.full(diff, np.nan)
                v = np.concatenate([v,fill])
            rocData[k] = v
            

        rocData.to_csv(os.path.join(pathToResults,"rocCurveData.txt"),sep="\t")

        return np.mean(probasOut[:,1::2],axis=1)


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


    def fit(self, X, metricColumns, scaler = None, inv = False, poolMethod="min"):
        ""
      #  print("Generate Square Matrix ..")
        X, labels = self._makeSquareMatrix(X, metricColumns, scaler, inv,  poolMethod)
        print(X)
        print("SQUAR MATRIX")
        

        embed = umap.UMAP(metric="precomputed", min_dist=0.00001, n_neighbors=4).fit_transform(X)

      #  print("done .. - starting clustering")
        if self.method == "OPTICS":
            clusterLabels = self.clustering.fit_predict(X)
            return clusterLabels, labels, X, self.clustering.reachability_[self.clustering.ordering_], self.clustering.core_distances_[self.clustering.ordering_]
        elif self.method in ["AGGLOMERATIVE_CLUSTERING","AFFINITY_PROPAGATION"]:
            clusterResult = self.clustering.fit_predict(X)
            return clusterResult, labels, X, ["None"] * labels.size, ["None"] * labels.size
        elif self.method == "HDBSCAN":
            clusterResult = self.clustering.fit(embed)
            return clusterResult.labels_ , labels, X, clusterResult.probabilities_, ["None"] * labels.size, embed
        
    def _makeSquareMatrix(self, X, metricColumns, scaler, inv,  poolMethod):
        import time
        if scaler is None:
            if poolMethod == "mean":
                X["meanDistance"] = X[metricColumns].mean(axis=1)
            elif poolMethod == "min":
                X["meanDistance"] = X[metricColumns].min(axis=1)
        else:
            if poolMethod == "mean":
                X["meanDistance"] = scaler(X[metricColumns]).mean(axis=1)
            elif poolMethod == "min":
                X["meanDistance"] = scaler(X[metricColumns]).min(axis=1)

        if inv:
            X['meanDistance'] = 1 - X['meanDistance']

        uniqueValues = np.unique(X[["E1p","E2p"]])
        nCols = nRows = uniqueValues.size 
        print("Creating {} x {} distance matrix".format(nCols,nCols))
        matrix = np.full(shape=(nRows,nCols), fill_value = 3.0)

       # print(nCols, "is the number of columns in square matrix\nwhich correpsonds to the number of proteins for which a protein-protein interaction was found.")
        
        t1 = time.time()

        for row in X[["E1p","E2p","meanDistance"]].values:
            
            nRow = np.where(uniqueValues == row[0])
            nCol = np.where(uniqueValues == row[1])
            matrix[[nRow,nCol],[nCol,nRow]] = row[2]
       
        matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
        np.fill_diagonal(matrix,0)
        
        return matrix, uniqueValues

    def evaluateClusters(self):
        "checks for false positives?"


    def optimizeClustering(self):
        ""
    

if __name__ == "__main__":
    print("PREDCITOR TEST")
