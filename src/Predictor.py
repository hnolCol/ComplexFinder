


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS

class Classifier(object):
    ""

    def __init__(self, classifierClass = "random forest"):
        ""
        self.classifier = self._initClassifier(classifierClass)


    def _initClassifier(self,classifierClass):
        ""
        if classifierClass in ["random forest","ensemble tree"]:
            return RandomForestClassifier(n_estimators=500,oob_score=True)

    def _scaleFeatures(self,X):

        X = StandardScaler.fit_transform(X)


    def fit(self, X, Y):
        ""
        X = self._scaleFeatures(X)
        self.classifier.fit(X,Y)



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
