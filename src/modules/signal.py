import pandas as pd 
import numpy as np 
import os
import gc
import scipy.signal as sciSignal 
import matplotlib.pyplot as plt
from lmfit import models
import numpy.random as random
import lmfit.models as lmModels
from lmfit.parameter import Parameter
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import pearsonr
import time


class Signal(object):

    def __init__(self, 
            Y, 
            ID= "", 
            peakModel = 'LorentzianModel', 
            nonNan = 4, 
            maxPeaks = 12, 
            savePlots = True, 
            savePeakModels = True, 
            setMaxToOne = False,
            smoothSignal = True,
            metrices = [], 
            pathToTmp = "", 
            removeSingleDataPointPeaks = True,
            normalizationValue = None,
            otherSignals = [],
            analysisName = ""):
        
        """Signal module for pre-processing and modeling


        The Signal module allows to do severl pre-processing/modelling
        steps such as
            a) smoothing (rolling average)
            b) filtering by number of nonNaN values
            c) removal of single data points (surrounded by zeros or nans)
            b) Peak detection (finds peaks) - required for further anaylsis

        The peak modelling allows for usage of `LorentzianModel` or `GaussianModel`



        Note
        ----
        
        Parameters
        ----------
        
       
        """
        self.Y = Y 
        self.ID = ID
        self.peakModel = peakModel
        self.maxPeaks = maxPeaks
        self.savePlots = savePlots 
        self.metrices = metrices
        self.pathToTmp = pathToTmp
        self.savePeakModels = savePeakModels
        self.analysisName = analysisName
        self.setMaxToOne = setMaxToOne
        self.normalizationValue = normalizationValue
        self.otherSignals = otherSignals
        if removeSingleDataPointPeaks:
            self.Y = self._removeSingleDataPointPeaks()
        if smoothSignal:
            self.Y = self.smoothSignal()
        if normalizationValue is not None:
            self.Y = self.Y / normalizationValue
        elif setMaxToOne:
            self.Y = self._scaleToHighestToOne()

    def _findNaNs(self,Y):
        "Finds nans in array"
        return np.isna(Y)

    def _getN(self,Y):
        ""
        N = int(float(Y.size * 0.025))
        if N < 3:
            N  = 3  
        return N

    def _scaleToHighestToOne(self):
        ""
        return self.Y / np.max(self.Y)


    def _removeSingleDataPointPeaks(self):
        "Removes signal data where a peak would only be made it by one data point"
        flilteredY = []

        for i,x in enumerate(self.Y):
            if i == 0: #first item is different
                if self.Y[i+1] == 0:
                    flilteredY.append(0)
                else:
                    flilteredY.append(x)

            elif i == self.Y.size - 1: #last item also
                if self.Y[-1] != 0 and self.Y[-1]:
                    flilteredY.append(0)
                else:
                    flilteredY.append(x)

            else:
                if self.Y[i-1] == 0 and self.Y[i+1] == 0:
                    flilteredY.append(0)
                else:
                    flilteredY.append(x)

        return np.array(flilteredY)

    def isValid(self, nonZero = 3):
        ""
        valid = np.sum(self.Y > 0) > nonZero
        if not valid:
            print("Signal {} is not valid.".format(self.ID))
        return valid

    def _movingAverage(self,Y,N):
        ""

        if N == "auto":
            N = self._getN(Y)
        return pd.Series(Y).rolling(window=N,center=True,win_type = "boxcar").mean().fillna(0).values

    def smoothSignal(self,method='moving average',N="auto"):
        ""

        if method == "moving average":
            return self._movingAverage(self.Y,N)
        
    
    def findPeaks(self,cwt=False,widths=[2,3,4]):

        #
        if cwt:
            peakDetection = sciSignal.find_peaks_cwt(self.Y,widths)
        else:
            peakDetection, _ = sciSignal.find_peaks(self.Y, distance = 2)

        return peakDetection


    def _generateSpec(self,x,y, N = 7):

        spec = {
            'x': x,
            'y': y,
            'model': [{'type': self.peakModel}] * N
            }
        return spec


    def _lorentzianModel(self,x,A,sigma,mu):
        ""

        return A/np.pi * (sigma / ((x-mu)**2 + sigma**2))


    def _addParam(self,modelParams,name,value,min=-np.inf,max = np.inf):

        modelParams.add(Parameter(name=name, value=value,min=min,max=max))


    def _addParams(self,modelParams,prefix,peakIdx,i):
           
        self._addParam(modelParams,
                            name=prefix+'amplitude',
                            max = self.Y[peakIdx[i]] * 9,
                            value = self.Y[peakIdx[i]] * 2,
                            min = self.Y[peakIdx[i]] * 1.2)

        self._addParam(modelParams,
                name=prefix+'sigma', 
                value = 0.255,
                min = 0.01, 
                max = 2.5)

        self._addParam(modelParams,
                name=prefix+'center', 
                value = peakIdx[i],
                min = peakIdx[i] - 0.2, 
                max = peakIdx[i] + 0.2)

        # self._addParam(modelParams,
        #         name=prefix+'height', 
        #         value = self.Y[peakIdx[i]],
        #         min = self.Y[peakIdx[i]] - self.Y[peakIdx[i]] * 0.01, 
        #         max = self.Y[peakIdx[i]] + self.Y[peakIdx[i]] * 0.01)


    def _findParametersForModels(self,spec,peakIdx):
        modelComposite = None
        params = None
        for i, basis_func in enumerate(spec['model']):
            prefix = f'm{i}_'
                     
            model = getattr(models, basis_func['type'])(prefix=prefix)
            modelParams = model.make_params()
            self._addParams(modelParams,prefix,peakIdx,i)
    
           # Parameter
           # modelParams = model.make_params(**defaultParams, **basis_func.get('params', {}))
            if modelComposite is None:
                modelComposite = model
            else:
                modelComposite = modelComposite + model
            if params is None:
                params = modelParams
            else:
                params.update(modelParams)
        return modelComposite,params
        
    def _checkPeakIdx(self,peakIdx, maxPeaks = 15):
        "Retrieves highest peaks"
        if peakIdx.size <=  maxPeaks:
            return peakIdx
        else:
            signal = self.Y[peakIdx]
            indices = np.argpartition(signal, -maxPeaks)[-maxPeaks:]
            return peakIdx[indices]


    def fitModel(self):
        ""
        if not self.isValid():
            return {"id":self.ID}
        peakIdx = self.findPeaks()
        peakIdx = self._checkPeakIdx(peakIdx,self.maxPeaks)
        spec = self._generateSpec(np.arange(self.Y.size) , self.Y, N = peakIdx.size)
        modelComposite, params = self._findParametersForModels(spec,peakIdx)
        if modelComposite is None:
            return {"id":self.ID}
        fitOutput = modelComposite.fit(self.Y, params, x=spec['x'], method="powell")
        r2 = self._calculateSquredR(fitOutput,spec)

        if self.savePlots or self.savePeakModels:
        
            self.plotSummary(fitOutput,spec,r2,peakIdx)
        
        return {"id":self.ID,"fitOutput":fitOutput,'spec':spec,'peakIdx':peakIdx}

    @property
    def modeledPeaks(self):
        ""
        if hasattr(self,"fitOutput"):
            detectedPeaks = self._collectPeakResults()
            squaredR = self._calculateSquredR()

    def _apex(self,p1,p2):
        ""
        
        return np.sqrt( (p1['mu'] - p2['mu']) ** 2  + (p1['sigma'] - p2['sigma']) ** 2 )


    def p_pears(self,u,v):
        "returns p value for pearson correlation"
        r, p = pearsonr(u,v)

        return r, p

    def euclideanDistance(self,Ys):
        ""
        return [np.linalg.norm(self.Y - Y) for Y in Ys]

    def pearson(self,Ys):
        ""
        return [self.p_pears(self.Y,Y) for Y in Ys]
       
    def apex(self,otherSignalPeaks):
        "Calculates Apex Distance"
        ownPeaks = self._collectPeakResults()
        apexDist = []     
        for otherPeaks in otherSignalPeaks:

            apexDist.append([self._apex(p1,p2) for p1 in ownPeaks for p2 in otherPeaks])
        minArgs = [np.argmin(x) for x in apexDist]
        return [np.min(x) for x in apexDist]

    def calculateMetrices(self,ID):

        metrices = self.metrices
        pathToTmp = self.pathToTmp 
        otherSignals = self.otherSignals

        pathToFile = os.path.join(pathToTmp,"{}".format(self.ID))

        if os.path.exists(pathToFile):
            pass

        else:

            collectedDf = pd.DataFrame()

            collectedDf["E1"] = [self.ID] * len(otherSignals)
            collectedDf["E2"] = [Signal.ID for Signal in otherSignals]
            
            for metric in metrices:

                Ys = np.array([Signal.Y for Signal in otherSignals])
                
                if metric == "pearson":
                    collectedDf["pearson"], collectedDf["p_pearson"] = zip(*self.pearson(Ys))
                    collectedDf["pearson"] = 1 - collectedDf["pearson"].values 

                elif metric == "euclidean":

                    collectedDf["euclidean"] = self.euclideanDistance(Ys)
                        
                elif metric == "apex":
                    otherSignalPeaks = [Signal._collectPeakResults() for Signal in otherSignals]

                    collectedDf["apex"] = self.apex(otherSignalPeaks)

                elif metric == "max_location":

                    maxOwnY = np.argmax(self.Y)
                    collectedDf["max_location"] = [np.argmax(Signal.Y)-maxOwnY for Signal in otherSignals]
            gc.collect()
            return collectedDf.values
            #collectedDf.to_csv(pathToFile, index=False)


                
    def calculateApexDistance(self,otherSignals,n):
        "Calculates Apex distance compared to all other Signals."
        ownPeaks = self._collectPeakResults()
        df = pd.DataFrame()
        
        indices = []
        apexDist = []

        for ID, otherPeaks in otherSignals[n:]:

            #distToOtherPeaks = Parallel(n_jobs=4)(delayed(self._apex)(p1,p2) for p1 in ownPeaks for p2 in otherPeaks)
            distToOtherPeaks = [self._apex(p1,p2) for p1 in ownPeaks for p2 in otherPeaks]
            #append results
            indices.append(ID)
            if len(distToOtherPeaks) == 0:
                apexDist.append(np.nan)
            else:
                apexDist.append(np.min(distToOtherPeaks))
        df["E1;E2"] = ["{};{}".format(self.ID,idx) for idx in indices]
        df["metric"] = apexDist

        return df


    def saveResults(self):

        self.Rsquared = self._calculateSquredR()
        self._collectPeakResults()
        
        

    def _collectPeakResults(self):
        "Put results of peaks in a list"

        if hasattr(self,"modelledPeaks"):

            return self.modelledPeaks

        if not hasattr(self , "fitOutput"):

            return []

        self.modelledPeaks = []
        best_values = self.fitOutput.best_values
        for i, model in enumerate(self.spec['model']):
            params = {}
            prefix = f'm{i}_'
            params['ID'] = i
            params["mu"] = best_values[prefix+"center"]
            params["sigma"] = best_values[prefix+"sigma"]
            params["height"] = self._getHeight(best_values,prefix) 
            params["fwhm"] = self._getFWHM(best_values,prefix) 
            params["E"] = self.ID
            self.modelledPeaks.append(params)

        return self.modelledPeaks

    def _calculateSquredR(self,model = None, spec= None):
        ""
        if model is None:
            model = self.fitOutput
        if spec is None:
            spec = self.spec
        return 1 - model.residual.var() / np.var(spec['y'])

    def _getFWHM(self,params,prefix):

        if self.peakModel == "LorentzianModel":
            FWHM =  2 * params[prefix+"sigma"]  
        else:
            FWHM =  2.3548 * params[prefix+"sigma"]
        return FWHM

    def _getHeight(self,params,prefix):
        if self.peakModel == "LorentzianModel":
            height = params[prefix+"amplitude"] / (params[prefix+"sigma"] * np.pi )
        else:
            height =  params[prefix+"amplitude"] / (params[prefix+"sigma"] * np.sqrt(np.pi*2)) 
        
        return height

    def plotSummary(self, fitOutput, spec, R, peakIdx):
        ""
        if self.savePlots:
            fileName = "{}_{}.pdf".format(self.analysisName,self.ID)
            pathToPlotFolder = os.path.join(self.pathToTmp,"modelPlots")
            if not os.path.exists(pathToPlotFolder):
                os.mkdir(pathToPlotFolder)
            pathToSaveFigure = os.path.join(pathToPlotFolder,fileName)
            
            fig, ax = plt.subplots()
            components = fitOutput.eval_components(x=spec['x'])
            best_values = fitOutput.best_values
            for i, model in enumerate(spec['model']):
                prefix = f'm{i}_'
                ax.plot(spec['x'], components[prefix], 
                    linestyle="-", 
                    linewidth=0.5, 
                    label="s:{}, A:{}, c:{}, fwhm:{} maxH:{}".format(round(best_values[prefix+"sigma"],3),
                                                            round(best_values[prefix+"amplitude"],3),
                                                            round(best_values[prefix+"center"],2),
                                                            round(self._getFWHM(best_values,prefix) ,3),
                                                            round(self._getHeight(best_values,prefix),3)
                                                            )                             
                        )                                            

            ax.plot(spec['x'],self.Y , color="black" , linestyle="--", linewidth=0.5)
            for peak in peakIdx:
                ax.axvline(peak, color="darkgrey",linestyle="--",linewidth=0.1)
            ax.set_title("R^2:{}".format(round(R,3)))
                                                     
            ax.legend(prop={'size': 5})
            plt.savefig(pathToSaveFigure)
            plt.close()

        if self.savePeakModels:
            pathToTxt = os.path.join(pathToPlotFolder,"{}_{}_r2_{}.txt".format(self.analysisName,self.ID,round(R,3)))
            with open(pathToTxt , "w") as f:
                f.write("\t".join(["Key","ID","Amplitude","Center","Sigma","fwhm","height","auc"])+"\n")
                for i, model in enumerate(spec['model']):
                    prefix = f'm{i}_'
                    f.write("\t".join([str(x) for x in [self.ID,
                                    i,
                                    round(best_values[prefix+"amplitude"],3),
                                    round(best_values[prefix+"center"],2),
                                    round(best_values[prefix+"sigma"],3),
                                    round(self._getFWHM(best_values,prefix) ,3),
                                    round(self._getHeight(best_values,prefix),3),
                                    round(np.trapz(components[prefix],dx = 0.2),3)]])+"\n")


    def __getstate__(self):
        state = self.__dict__.copy()
        if not hasattr(self,"modelledPeaks"):
            self._collectPeakResults()
        if "fitOutput" in state:
            del state["fitOutput"]
        return state
            

        
    #def _lorentzianModel(self,x,A,sigma,mu):
if __name__ == "__main__":

    Y = np.array([0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.60,0.5,0.3,0.9,0.3,0.2,0.05,0,0,0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.8,0.9,0.3,0.2,0.05,0])
    s = Signal(Y)
    
    
   # print(s.findPeaks())
   # print(s.smoothSignal())
    s.fitModel()

    f1 = plt.figure()

    s.plotSummary(f1)





    



