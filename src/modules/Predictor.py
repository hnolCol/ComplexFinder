


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS

class Classifier(object):
    ""

    def __init__(self, classifierClass = "random forest", n_jobs = 4, gridSearch = None):
        ""
        self.gridSerach = gridSearch
        self.n_jobs = n_jobs
        self.classifier = self._initClassifier(classifierClass,n_jobs)


    def _initClassifier(self,classifierClass,n_jobs):
        ""
        if classifierClass in ["random forest","ensemble tree"]:
            return RandomForestClassifier(n_estimators=200,
                                            oob_score=True,
                                            min_samples_split=2,
                                            n_jobs=n_jobs)

    def _scaleFeatures(self,X):

        return StandardScaler().fit_transform(X)
        

    def _gridOptimization(self,X,Y):

        gridSearch = GridSearchCV(self.classifier, scoring = "f1", param_grid = self.gridSerach, n_jobs = self.n_jobs, cv = 10, verbose=5)
        gridSearch.fit(X,Y)

        print(gridSearch.best_estimator_.oob_score_)

        return gridSearch.best_params_


    def fit(self, X, Y, optimizedParams=None):
        ""
        print("predictor training started")
        X = self._scaleFeatures(X)

        xTrain, xTest, yTrain, yTest = train_test_split(X,Y,test_size=0.25)


        if self.gridSerach is not None:
            optimizedParams = self._gridOptimization(X,Y)

        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2)

        #cv = StratifiedKFold(n_splits=3)
        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()
        i=0

        for train, test in cv.split(X, Y):
            if optimizedParams is not None:
                self.classifier.set_params(**optimizedParams)
            probas_ = self.classifier.fit(X[train], Y[train]).predict_proba(X[test])
            print(self.classifier.feature_importances_)
            print(self.classifier.oob_score_)
            print(self.classifier.classes_)
            print(probas_)
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, lw=1, alpha=0.3,
                    label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            #self.classifier.fit(X,Y)
            i+=1
        print("predictor training done")
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

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
        plt.savefig("FDR.pdf")


class ComplexBuilder(object):


    def __init__(self):
        ""


    def fit():
        ""
        OPTICS.fit(self.X)
        

    def evaluateClusters(self):
        "checks for false positives?"


    def optimizeClustering(self):
        ""
        





if __name__ == "__main__":
    print("PREDCITOR TEST")
