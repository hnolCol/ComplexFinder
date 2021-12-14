
import os


class Plotter(object):
    ""
    def __init__(self,complexFolder,*args,**kwargs):
        ""
        self.complexFolder = complexFolder
    

    def _checkFolder(self):
        ""
        os.path.exists(self.complexFolder)


    def plotFeature(self):
        ""

    def plotSignalProfilesByIds(self,ids=[],signalType = "fit"):
        """
        Apex between two peaks. Basically euclidean distance between two peaks.

        Parameters
        ----------
        ids : array-like
            Identifiers that should be plotted.

        signalType : str
            Type of signal profile intensity that should be used for plotting.
            Must be in ["raw","fit","processed"] or a combination of those: "raw,fit"
        

        Returns
        -------
        None

        """

    def plotComplexProfileByClusterLabel(self):
        ""

    
    def plotFeatureDistanceMetrics(self):
        ""



