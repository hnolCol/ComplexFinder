import pandas as pd 
import numpy as np 

import scipy.signal as sciSignal 
import matplotlib.pyplot as plt
from lmfit import models
import numpy.random as random
import lmfit.models as lmModels
from lmfit.parameter import Parameter
import matplotlib.pyplot as plt


class Signal(object):

    def __init__(self, Y, ID= "", peakModel = "", nonNan = 4):

        self.Y = Y 
        self.ID = ID

    def _findNaNs(self,Y):
        "Finds nans in array"
        return np.isna(Y)

    def _getN(self,Y):
        ""
        N = int(float(Y.size * 0.025))
        if N < 1:
            N  = 2  
        return N

    def isValid(self, nonNan = 4):
        ""
        
        return self._findNaNs(Y) - self.Y.size <= 4


    def _movingAverage(self,Y,N):
        ""

        if N == "auto":
            N = self._getN(Y)
           
        return pd.Series(Y).rolling(window=N).mean().values

    def smoothSignal(self,method='moving average',N="auto"):
        ""

        if method == "moving average":
            return self._movingAverage(self.Y,N)
        
    
    def findPeaks(self,cwt=False,widths=[2,3,4]):

        #
        if cwt:
            peakDetection = sciSignal.find_peaks_cwt(self.Y,widths)
        else:
            peakDetection, _ = sciSignal.find_peaks(self.Y)
        print(peakDetection)

        return peakDetection


    def _generateSpec(self,x,y, N = 7):

        spec = {
            'x': x,
            'y': y,
            'model': [{'type': 'LorentzianModel'}] * N
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
                            value = 1,
                            min = 1e-6)

        self._addParam(modelParams,
                name=prefix+'sigma', 
                value = 3,
                min = 0.01, 
                max = 6)

        self._addParam(modelParams,
                name=prefix+'center', 
                value = peakIdx[i],
                min = peakIdx[i] - 1, 
                max = peakIdx[i] + 1)

        self._addParam(modelParams,
                name=prefix+'height', 
                value = self.Y[peakIdx[i]],
                min = self.Y[peakIdx[i]] - self.Y[peakIdx[i]] * 0.05, 
                max = self.Y[peakIdx[i]] + self.Y[peakIdx[i]] * 0.05)


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
        

    def fitModel(self):
        ""
        self.peakIdx = self.findPeaks()
        if self.peakIdx.size < 22:
            self.spec = self._generateSpec(np.arange(self.Y.size) , self.Y, N = self.peakIdx.size)
            modelComposite, params = self._findParametersForModels(self.spec,self.peakIdx)
            self.fitOutput = modelComposite.fit(self.Y, params, x=self.spec['x'])
            self.plotSummary()
        else:
            print("too many peaks")

    @property
    def modeledPeaks(self):
        ""
        if hasattr(self,"fitOutput"):
            print(self._collectPeakResults())
            print(self._calculateSquredR())


    def _collectPeakResults(self):
        "Put results of peaks in a list"
        peaks = []
        best_values = self.fitOutput.best_values
        for i, model in enumerate(self.spec['model']):
            params = {}
            prefix = f'm{i}_'
            params['ID'] = i
            params["mu"] = best_values[prefix+"center"]
            params["sigma"] = best_values[prefix+"sigma"]
            peaks.append(params)

        return peaks

    def _calculateSquredR(self):
        ""
        return 1 - self.fitOutput.residual.var() / np.var(self.spec['y'])


       # fig, gridspec = output.plot(data_kws={'markersize': 1})

       # self.print_best_values(spec,output)
       # plt.show()
       # fig, ax = plt.subplots()
       # ax.plot(spec['x'],self.Y , color="black" , linestyle="--")
       
       # components = output.eval_components(x=spec['x'])
      
      
       # for i, model in enumerate(spec['model']):
       #     ax.plot(spec['x'], components[f'm{i}_'])
      #  plt.show()




        #composite_model, params = self._generateModels(spec)
        #peak_indicies, params = self.update_spec_from_peaks(composite_model,spec,[0,1,2],peak_widths=[1,4])
        #output = model.fit(spec['y'], params, x=spec['x'])
        #fig, gridspec = output.plot(data_kws={'markersize': 1})
        #plt.show()




    def plotSummary(self, figure = None):
        ""

        if figure is None:
            fig, ax = plt.subplots()
            components = self.fitOutput.eval_components(x=self.spec['x'])
            for i, model in enumerate(self.spec['model']):
                ax.plot(self.spec['x'], components[f'm{i}_'], linestyle="-", linewidth=0.5)
                ax.plot(self.spec['x'],self.Y , color="black" , linestyle="--", linewidth=0.5)
            for peak in self.peakIdx:
                ax.axvline(peak, color="darkgrey",linestyle="--")
            plt.savefig("{}.pdf".format(self.ID))
            plt.close()

            

        
    #def _lorentzianModel(self,x,A,sigma,mu):
if __name__ == "__main__":

    Y = np.array([0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.60,0.5,0.3,0.9,0.3,0.2,0.05,0,0,0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.8,0.9,0.3,0.2,0.05,0])
    s = Signal(Y)
    
    
   # print(s.findPeaks())
   # print(s.smoothSignal())
    s.fitModel()

    f1 = plt.figure()

    s.plotSummary(f1)





    



