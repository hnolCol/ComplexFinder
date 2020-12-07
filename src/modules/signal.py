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
            analysisName = "",
            r2Thresh = 0.0,
            minDistanceBetweenTwoPeaks = 3):
        
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
        self.r2Thresh = r2Thresh
        self.minDistanceBetweenTwoPeaks = minDistanceBetweenTwoPeaks

        self.valid = True
        self.validModel = True
        self.maxNumbPeaksUsed = False
    
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
        """Scales highest signal intensity to 1

        Parameters
        ----------
        

        Returns
        -------
        Scaled Y

        """
        return self.Y / np.max(self.Y)


    def _removeSingleDataPointPeaks(self):
        """Removes signal data where a peak would only be made it by one data point

        Parameters
        ----------
        

        Returns
        -------
        Filtered Y

        """
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

    def isValid(self, nonZero = 4):
        """Returns true if signal contains more than 
        argument nonZero (int) value that are higher than 0.

        Parameters
        ----------

        nonZero : int
            Number of non zero intensities.
        

        Returns
        -------
        boolean, True if vald

        """
        valid = np.sum(self.Y > 0) > nonZero
        if not valid:
            print("Signal {} is not valid.".format(self.ID))
            self.valid = False
        return valid

    def _movingAverage(self,Y,N):
        ""

        if N == "auto":
            N = self._getN(Y)
        return pd.Series(Y).rolling(window=N,center=False, win_type = None).mean().fillna(0).values

    def smoothSignal(self,method='moving average',N="auto"):
        ""

        if method == "moving average":
            return self._movingAverage(self.Y,N)
        
    
    def findPeaks(self,cwt=False,widths=[2,3,4]):

        if cwt:
            peakDetection = sciSignal.find_peaks_cwt(self.Y,widths)
        else:
            peakDetection, _ = sciSignal.find_peaks(self.Y, distance = self.minDistanceBetweenTwoPeaks)

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
        """
        Adds parameter to the modelParam object

        Parameters
        ----------

        mdeolParams : 
            modelParam object. Returned by model.make_params() (lmfit package) 
            Documentation: https://lmfit.github.io/lmfit-py/model.html

        prefix : str
            Prefix for the model (e.g. peak), defaults to f'm{i}_'.format(i)

        peakIdx : int
            Arary index at which the peak was detected in the Signal arary self.Y 

        i : int 
            index of detected models
    
        Returns
        -------
        None

        """
        self._addParam(modelParams,
                            name=prefix+'amplitude',
                            max = self.Y[peakIdx[i]] * 9,
                            value = self.Y[peakIdx[i]] * 2,
                            min = self.Y[peakIdx[i]] * 1.2)

        self._addParam(modelParams,
                name=prefix+'sigma', 
                value = 0.255,
                min = 0.05, 
                max = 3.0)

        self._addParam(modelParams,
                name=prefix+'center', 
                value = peakIdx[i],
                min = peakIdx[i] - 0.2, 
                max = peakIdx[i] + 0.2)

        if self.peakModel == "SkewedGaussianModel":
            
            self._addParam(modelParams,
                name=prefix+'gamma', 
                value = 0,
                min = -1, 
                max = 1)

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
        """
        Checks if number of peaks exceed the max number of 
        allwed peaks. (paramater: maxPeaks)

        If the number exceeds maxPeaks, the peaks with the 
        highest value are taken. Others are removed

        Parameters
        ----------

        peakIdx : np.array
            Array with idx matching self.Y at which peaks were detected.


        maxPeaks : int
            Number of max peaks allowed.

    
        Returns
        -------
        np.array of peak indices.

        """
        if peakIdx.size <=  maxPeaks:
            return peakIdx
        else:
            self.maxNumbPeaksUsed  = True
            signal = self.Y[peakIdx]
            indices = np.argpartition(signal, -maxPeaks)[-maxPeaks:]
            return peakIdx[indices]


    def fitModel(self):
        """
        Fits the model (ensemble of several peaks).
        The number of models equals the number of 
        detected peaks. Please not that that the maximum
        number of peaks is limited by the parameter: 

            maxPeaks (defaults to 12)

        Depending on the user settings: 

            - peak models + signal profile are plotted and saved as pdf (folder modelPlots)

            - if squaredR for the model fit is below threshold (r2Tresh - deufault 0.85), the
                signal profile is ignored. A message is printed if this happens.

        Parameters
        ----------

    
        Returns
        -------
        dict with keys ("id","fitOutput","spec" and "peakIdx")

        """
        if not self.isValid():
            return {"id":self.ID}
        peakIdx = self.findPeaks()
        peakIdx = self._checkPeakIdx(peakIdx,self.maxPeaks)
        spec = self._generateSpec(np.arange(self.Y.size) , self.Y, N = peakIdx.size)
        modelComposite, params = self._findParametersForModels(spec,peakIdx)
        if modelComposite is None:
            self.validModel = False
            self.valid = False
            return {"id":self.ID}
        fitOutput = modelComposite.fit(self.Y, params, x=spec['x'], method="powell")
        r2 = self._calculateSquredR(fitOutput,spec)
        if r2 < self.r2Thresh:
            print("Model optimization did yield r2 below threshold ({}) for Signal {}".format(self.r2Thresh, self.ID))
            self.validModel = False
            self.valid = False
            return {"id":self.ID}
        if self.savePlots or self.savePeakModels:
        
            self.plotSummary(fitOutput,spec,r2,peakIdx)
        #save r2.
        self.Rsquared = r2 
        self.fitOutput = fitOutput
        
        return {"id":self.ID,"fitOutput":fitOutput,'spec':spec,'peakIdx':peakIdx}


    def saveResults(self):

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

        elif self.peakModel in ["SkewedGaussianModel","GaussianModel"]:
            FWHM =  2.3548 * params[prefix+"sigma"]
        else:
            FWHM = np.nan
        return FWHM

    def _getHeight(self,params,prefix):
        if self.peakModel == "LorentzianModel":
            height = params[prefix+"amplitude"] / (params[prefix+"sigma"] * np.pi )
        
        elif self.peakModel in ["SkewedGaussianModel","GaussianModel"]:
            
            height =  params[prefix+"amplitude"] / (params[prefix+"sigma"] * np.sqrt(np.pi*2)) 
        else:

            height = np.nan
        return height

    def plotSummary(self, fitOutput, spec, R, peakIdx):
        ""
        components = fitOutput.eval_components(x=spec['x'])
        best_values = fitOutput.best_values
        pathToPlotFolder = os.path.join(self.pathToTmp,"result","modelPlots")
        if not os.path.exists(pathToPlotFolder):
            os.mkdir(pathToPlotFolder)
            
        if self.savePlots:
            fileName = "{}_{}.pdf".format(self.analysisName,self.ID)
            pathToSaveFigure = os.path.join(pathToPlotFolder,fileName)
            fig, ax = plt.subplots()
            for i, model in enumerate(spec['model']):
                prefix = f'm{i}_'
                if self.peakModel == "SkewedGaussianModel":
                    ax.plot(spec['x'], components[prefix], 
                        linestyle="-", 
                        linewidth=0.5, 
                        label="s:{}, A:{}, c:{},gamma:{} fwhm:{} maxH:{} ".format(round(best_values[prefix+"sigma"],3),
                                                                round(best_values[prefix+"amplitude"],3),
                                                                round(best_values[prefix+"center"],2),
                                                                round(best_values[prefix+"gamma"],2),
                                                                round(self._getFWHM(best_values,prefix) ,3),
                                                                round(self._getHeight(best_values,prefix),3)
                                                                )                             
                        ) 
                else:
                    ax.plot(spec['x'], components[prefix], 
                        linestyle="-", 
                        linewidth=0.5, 
                        label="s:{}, A:{}, c:{} fwhm:{} maxH:{} ".format(round(best_values[prefix+"sigma"],3),
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
            AUCs = [np.trapz(components[f'm{i}_'],dx = 0.15) for i,_ in enumerate(spec['model'])]
            sumAUC = np.sum(AUCs)
            reltiveAUC = [x/sumAUC for x in AUCs]
            with open(pathToTxt , "w") as f:
                if self.peakModel == "SkewedGaussianModel":
                    f.write("\t".join(["Key","ID","Amplitude","Center","Sigma","Gamma","fwhm","height","auc","relAUC"] + [str(x) for x in spec["x"]])+"\n")
                else:
                    f.write("\t".join(["Key","ID","Amplitude","Center","Sigma","fwhm","height","auc","relAUC"] + [str(x) for x in spec["x"]])+"\n")
                for i, model in enumerate(spec['model']):
                    prefix = f'm{i}_'
                    if self.peakModel == "SkewedGaussianModel":
                        f.write("\t".join([str(x) for x in [self.ID,
                                    i,
                                    round(best_values[prefix+"amplitude"],3),
                                    round(best_values[prefix+"center"],2),
                                    round(best_values[prefix+"sigma"],3),
                                    round(best_values[prefix+"gamma"],3),
                                    round(self._getFWHM(best_values,prefix) ,3),
                                    round(self._getHeight(best_values,prefix),3),
                                    round(AUCs[i],3),
                                    round(reltiveAUC[i],3)] + components[prefix].tolist()])+"\n")
                    else:
                        f.write("\t".join([str(x) for x in [self.ID,
                                    i,
                                    round(best_values[prefix+"amplitude"],3),
                                    round(best_values[prefix+"center"],2),
                                    round(best_values[prefix+"sigma"],3),
                                    round(self._getFWHM(best_values,prefix) ,3),
                                    round(self._getHeight(best_values,prefix),3),
                                    round(AUCs[i],3),
                                    round(reltiveAUC[i],3)] + components[prefix].tolist()])+"\n")


    def __getstate__(self):
        state = self.__dict__.copy()
        if not hasattr(self,"modelledPeaks"):
            
            self._collectPeakResults()
        if "fitOutput" in state:
            del state["fitOutput"]
        return state
            




    



