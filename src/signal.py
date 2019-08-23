import pandas as pd 
import numpy as np 

import scipy.signal as sciSignal 
import matplotlib.pyplot as plt
from lmfit import models
import numpy.random as random
import lmfit.models as lmModels
from lmfit.parameter import Parameter


class Signal(object):

    def __init__(self, Y, nonNan = 4):

        self.Y = Y 

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
        
    
    def findPeaks(self,widths=[2,3]):

        peakDetection = sciSignal.find_peaks_cwt(self.Y,widths)
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

    def _findParametersForModels(self,spec,peakIdx):
        modelComposite = None
        params = None
        print(peakIdx)
        for i, basis_func in enumerate(spec['model']):
            prefix = f'm{i}_'
                     
            model = getattr(models, basis_func['type'])(prefix=prefix)
            modelParams = model.make_params()
            modelParams.add(Parameter(name=prefix+'amplitude',
                                        value = 1,
                                        min = 1e-6))
                                         
            modelParams.add(Parameter(name=prefix+'height', value = self.Y[peakIdx[i]], 
                                        min = 0.01, 
                                        max = self.Y[peakIdx[i]] + self.Y[peakIdx[i]] * 0.5))                                        
                                    
            modelParams.add(Parameter(name=prefix+'sigma', value = 3, min=1, max = 4))
            modelParams.add(Parameter(name=prefix+'center', value =  peakIdx[i], min =  1, max = peakIdx[i] + 5))
            
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
        

    def _generateModels(self,spec):

        composite_model = None
        params = None

        x = spec['x']
        y = spec['y']
        x_min = np.min(x)
        x_max = np.max(x)
        x_range = x_max - x_min
        y_max = np.max(y)
        for i, basis_func in enumerate(spec['model']):
            prefix = f'm{i}_'
            print(prefix)
            model = getattr(models, basis_func['type'])(prefix=prefix)
            if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']: # for now VoigtModel has gamma constrained to sigma
                model.set_param_hint('sigma', min=1e-6, max=x_range)
                model.set_param_hint('center', min=x_min, max=x_max)
                model.set_param_hint('height', min=1e-6, max=1.1*y_max)
                model.set_param_hint('amplitude', min=1e-6)
                # default guess is horrible!! do not use guess()
                default_params = {
                    prefix+'center': x_min + x_range * random.random(),
                    prefix+'height': y_max * random.random(),
                    prefix+'sigma': x_range * random.random()
                }
            else:
                raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
            if 'help' in basis_func:  # allow override of settings in parameter
                for param, options in basis_func['help'].items():
                    model.set_param_hint(param, **options)
            model_params = model.make_params(**default_params, **basis_func.get('params', {}))
            print(model_params)
            if params is None:
                params = model_params
            else:
                params.update(model_params)
            if composite_model is None:
                composite_model = model
            else:
                composite_model = composite_model + model
        return composite_model, params

    def print_best_values(self,spec, output):
        model_params = {
            'GaussianModel':   ['amplitude', 'sigma'],
            'LorentzianModel': ['amplitude', 'sigma'],
            'VoigtModel':      ['amplitude', 'sigma', 'gamma']
        }
        best_values = output.best_values
        print('center    model   amplitude     sigma      gamma')
        for i, model in enumerate(spec['model']):
            prefix = f'm{i}_'
            values = ', '.join(f'{best_values[prefix+param]:8.3f}' for param in model_params[model["type"]])
            print(f'[{best_values[prefix+"center"]:3.3f}] {model["type"]:16}: {values}')

    def fitModel(self):
        ""
        peakIdx = self.findPeaks()
        spec = self._generateSpec(np.arange(self.Y.size) , self.Y, N = peakIdx.size)
        modelComposite, params = self._findParametersForModels(spec,peakIdx)
        print(params)
        output = modelComposite.fit(self.Y, params, x=spec['x'])
        fig, gridspec = output.plot(data_kws={'markersize': 1})
        self.print_best_values(spec,output)
        plt.show()
        fig, ax = plt.subplots()
        ax.plot(spec['x'],self.Y , color="black" , linestyle="--")
        components = output.eval_components(x=spec['x'])
        for i, model in enumerate(spec['model']):
            ax.plot(spec['x'], components[f'm{i}_'])
        plt.show()




        #composite_model, params = self._generateModels(spec)
        #peak_indicies, params = self.update_spec_from_peaks(composite_model,spec,[0,1,2],peak_widths=[1,4])
        #output = model.fit(spec['y'], params, x=spec['x'])
        #fig, gridspec = output.plot(data_kws={'markersize': 1})
        #plt.show()



    def plotSummary(self, figure = None):
        ""

        if figure is not None:
            ax=plt.subplot(2, 2, 1)
            for sigma in [1,2,5,6,10]:
                x = np.arange(100)
                #print(x)
                y = self._lorentzianModel(x,5,sigma,20)
                ax.plot(x,y)
            plt.show()

        
    #def _lorentzianModel(self,x,A,sigma,mu):
if __name__ == "__main__":

    Y = np.array([0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.60,0.5,0.3,0.9,0.3,0.2,0.05,0,0,0,0.3,0.5,0.8,0.9,0.5,0.3,0.15,0,0,0,0.3,0.7,0.8,0.9,0.3,0.2,0.05,0])
    s = Signal(Y)
    
    
   # print(s.findPeaks())
   # print(s.smoothSignal())
    s.fitModel()

    f1 = plt.figure()

    s.plotSummary(f1)





    



