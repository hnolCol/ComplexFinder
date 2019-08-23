


from sklearn.ensemble import RandomForestClassifier


class Classifier(object):
    ""

    def __init__(self, classifierClass = "random forest"):
        ""
        self.classifier = self._initClassifier(classifierClass)


    def _initClassifier(self,classifierClass):
        ""
        return RandomForestClassifier(n_estimators=500,oob_score=True)

    def _scaleFeatures(self,data):


    def fit(self, X, Y):
        ""
        self.classifier.fit(X,Y)


if __name__ == "__main__":
    print("PREDCITOR TEST")
