# ComplexFinder
 Finds complexes from Blue-Native and SEC Fractionation analyzed by Liquid Chromatogrpahy coupled to Mass Spectrometry. In 
 principal it works with any separation technique resulting in co-elution profiles are available. 

 ## Workflow

For thousdands of features (peptides/protein) a signal was measured over different fractions. The applied technique separates protein clusters from each other. This package aims for different things:

* signal processing including filtering, smoothing.
* if more than one replicate is analysed, the profiles over fractions will be aligned. 
* identification of protein-protein interactions.
* identification of protein cluster using diminesional reduction and density based clustering. 

 ![Signal processing and protein-protein interaction prediction](/img/workflow.png)


As a next step, we want to identify clusters. To this end, we are using the interaction probabiliy matrix obtained by the
random forest classifier. To this end, we are calculating the UMAP embedding and apply HDBSCAN clustering. Again, we 
are using the CORUM database to quantify the the clustering result. Both techniques, UMAP and HDBSCAN are performed 
using a paramter grid to cycle through different options and find the best clustering. 
The rating of the clustering takes into account, the number of correct protein-protein interactions (defined by CORUM database), the total number of protein-protein interactions (basically the amount of noise found by HDBSCAN). 

 ![Quantification](/img/quantDetails.png)


## Installation

Download the zip file containing the source code from github.
Navigate to the folder in terminal/command line tool.
On Mac / Linux:
```
#create virt env
python3 -m venv env
#activate
source env/bin/activate
#install packages from req file
pip install -r requirements.txt
```
For windows user:
```
#create virt env 
py -m venv env
#actve
.\env\Scripts\activate
#install packages from req file
pip3 install -r requirements.txt
````

## Usage Example

Upon downlaod and extraction of the package. You can find example data in the example-data folder. 
To run the anaylsis, you can enter:
```python
from .src.main import ComplexFinder
X = pd.read_table("./example-data/SILAC_01.txt", sep = "\t") #loading tab delimited txt file. 
ComplexFinder(analysisName = "ExampleRun_01").run(X)
```
You can also pass a folder path to run. This will yield in the anaylsis of each txt file in the folder.

```python
import os
from main import ComplexFinder
folderPath = os.path(".","<my folder>")
ComplexFinder().run(folderPath)
```
Additionally, you can pass a list of datasets. However, we  recommend to copy required datasets in a separate folder.



## Parameters
Find below parameters to set. The default is given in brackets after the parameter name.

* indexIsID [False] bool.
* restartAnalysis [False] bool. Set True if you want to restart the anaylsis from scratch. If the tmp folder exsists, items and dirs will be deleted first.
* plotSignalProfiles [True] bool. If True, each profile is plotted against the fractio along with the fitted models. If you are concerned about time, you might set this to False at the cost of losing visible asessment of the fit quality.
* removeSingleDataPointPeaks [True] bool.
* <del>savePeakModels [True]</del> *depracted. always True and will be removed in the next version*.
* maxPeaksPerSignal [9] Number of peaks allowed for on signal profile.
* minDistBetweenPeaks [3] Distance in fractions (int) between two peaks. Setting this to a smaller number results in more peaks.
* n_jobs [12] Number of workers to model peaks, to calculate distance pairs and to train and use the classifer.
* kFold [5] Cross validation of classifier.
* analysisName [None]
* idColumn ["Uniprot ID"]
* databaseName ["20190823_CORUM.txt"]
* peakModel ["GaussianModel"] - which model should be used to model signal profiles. In principle all models from lmfit can be used. However, the initial parameters are only optimized for GaussianModel and LaurentzianModel. This might effect runtimes dramatically. 
* classifierClass ["random_forest"] string. Must be string 
* retrainClassifier [False] False, if the trainedClassifier.sav file is found, the classifier is loaded and the training is skipped. If you change the classifierGridSearch, you should set this to True. This will ensure that the classifier training is never skipped.
* interactionProbabCutoff [0.7] Cutoff for estimator probability. Interactions with probabilities below threshold will be removed.
* metrices [["apex","euclidean","pearson","p_pearson","spearman","max_location"]] Metrices to access distance between two profiles. Can be either a list of strings and/or dict. In case of a list of dicts, each dict must contain the keywords: 'fn' and 'name' providing a callable function with 'fn' that returns a single floating number and takes two arrays as an input.
* classiferGridSearch = RF_GRID_SEARCH (see below). dict with keywords matching parameters/settings of estimator (SVM, random forest) and list of values forming the grid used to find the best estimator settings (evaluated by k-fold cross validation). Runtime is effected by number of parameter settings as well as k-fold. 
* smoothSignal [True]) Enable/disable smoothing. Defaults to True. A moving average of at least 3 adjacent datapoints is calculated using pandas rolling function. Effects the analysis time as well as the nmaximal number of peaks detected.
* r2Thresh [0.85] R2 threshold to accept a model fit. Models below the threshold will be ignored.
* databaseFilter  [{'Organism': ["Mouse"]}] Filter dict used to find relevant complexes from database. By default, the corum database is filtered based on the column 'Organism' using 'Mouse' as a search string. If no filtering is required, pass an empty dict {}. 
* databaseIDColumn ["subunits(UniProt IDs)"]
* databaseHasComplexAnnotations [True] Indicates if the provided database does contain complex annotations. If you have a database with only pairwise interactions, this setting should be *False*. Clusters are identified by dimensional reduction and density based clustering (HDBSCAN). In order to alter UMAP and HDBSCAN settings use the kewywords *hdbscanDefaultKwargs* and *umapDefaultKwargs*.
* hdbscanDefaultKwargs [{}]
* umapDefaultKwargs [{"min_dist":0.0001,"n_neighbors":4}]
* noDatabaseForPredictions [False]. If you want to use ComplexFinder without any database. Set this to *True*.
* alignRuns [True] Alignment of runs is based on signal profiles that were found to have a single modelled peak. A refrence run is assign by correlation anaylsis and choosen based on a maximum R2 value. Then fraction-shifts per signal profile is calculated (must be in the window given by *alignWindow*). The fraction residuals are then modelled using the method provided in *alignMethod*. Model peak centers are then adjusted based on the regression results. Of note, the alignment is performed after peak-modelling and before distance calculations. 
* alignMethod ["RadiusNeighborsRegressor"]  
* alignWindow [3] Number of fraction +/- single-peal profile are accepted for the run alignment. 
* runName [None] Name of the analysis. A folder with date + runName + number of data frames will be generated. In this folder, combined results will be stored such as matched peaks and graphs.   
* useRawDataForDimensionalReduction [False] Setting this to true, will force the pipeline to use the raw values for dimensional reduction. Distance calculations are not automatically turned off and the output is generated but they are not used.
* scaleRawDataBeforeDimensionalReduction [True] If raw data should be used (*useRawDataForDimensionalReduction*) enable this if you want to scale them. Scaling will be performed that values of each row are scaled between zero and one.
* grouping [None] None or dict. Indicates which samples (file) belong to one group. Let's assume 4 files with the name 'KO_01.txt', 'KO_02.txt', 'WT_01.txt' and 'WT_02.txt' are being analysed. The grouping dict should like this : {"KO":[KO_01.txt','KO_02.txt'],"WT":['WT_01.txt','WT_02.txt']} in order to combine them for statistical testing (e.g. t-test of log2 transformed peak-AUCs). Note that when analysis multiple runs (e.g. grouping present) then calling ComplexFinder().run(X) - X must be a path to a folder containing the files.
* decoySizeFactor [1.2] float. Fraction of decoy pairwise interactions compared to positive interactions found in the database. If > 1, the decoy database will be bigger than the positive data which reflects more the true situation (less proteins interact with each other compared to the ones that form a complex.)
* classifierTestSize  [0.25] float. Fraction of the created database containing positive and negative protein-protein interactions that will be used for testing (for example ROC curve analysis) and classification report.
* precision [None] Precision to use to filter protein-protein interactions. If None, the filtering will be performed based on the parameter *interactionProbabCutoff*.
* considerOnlyInteractionsPresentInAllRuns [False] Can be either bool to filter for protein - protein interactions that are present in all runs. If an integer is provided. the pp interactions are filtered based on the number of runs in which they were quantified. A value of 4 would indicate that the pp interaction must have been predicted in all runs. 

```python
#random forest grid search
RF_GRID_SEARCH = {
                'max_depth':            [70],
                'max_features':         ['sqrt','auto'],
                'min_samples_leaf':     [2, 3, 4],
                'min_samples_split':    [2, 3, 4],
                'n_estimators':         [600]
                }
```

Sklearn library is used for predictions. Please check the comprehensive [documention](https://scikit-learn.org/stable/user_guide.html) for more details and for construction of a grid search dict. 

 # Frequently asked questions

 * *I get the Error message: no positive hits found in database. What does it mean?*

 Please check the class argument databaseFilter of type dict. For example the default is 
 ```python
 databaseFilter = {'Organism': ["Human"]}
```
This means that the database is filtered on column 'Organism' using "Human" as the search string.  

* *How can I change the positive database?*

The required format for the database is a tab-delimited txt file. The file must contain the columns: ComplexID and ComplexName. 
Additionally, the pipeline requires a column with the feature IDs (same ID as in the provided co-elution/migration data) of a complex divided by a ";". 
Easiest might be to check the provided databases in the folder *reference-data*. 
If you want to use ComplexFinder without a database, check out the FAQ (*How to run the pipeline without a database?*) below.

* *How can I change the peak model?*

The peak built in models are from the package [lmfit](https://lmfit.github.io/lmfit-py/builtin_models.html). 
```python
#in the Signal module
#import from lmfit
from lmfit import models

#line 208, gets the model based on the string
model = getattr(models, basis_func['type'])(prefix=prefix)
```

Therefore, you can provide any string that matches a model name in the lmfit package. Please note that, only peak parameters and constraints 
are implemented and tested for Gaussian, Lorentzian and Skewed Gaussian. So if your fit does not work, you may want to check the following
function of the *Signal.py* class module.

```python
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
                        min = 0.01, 
                        max = 2.5)

        self._addParam(modelParams,
                        name=prefix+'center', 
                        value = peakIdx[i],
                        min = peakIdx[i] - 0.2, 
                        max = peakIdx[i] + 0.2)

        ## enter other model params here, you may have to change the min and max
        ## for the other parameters as well to get a nice fit. 
```

Please not that you also have to alter the functions *_getHeight* and *_getFWHM* for your peak models. 
You can check the equations [here](http://openafox.com/science/peak-function-derivations.html).


 # Future Directions

In the future, we would like to implement the following features:

* Web application with an easy uster interface to proide easy access to the pipeline
* Implement more classifiers. 
* Test various peak models for better performance. 

 # Requirements

The following python packages are required to run the scripts (from the requirements.txt file.)

* asteval==0.9.19
* certifi==2020.11.8
* cycler==0.10.0
* Cython==0.29.21
* future==0.18.2
* hdbscan==0.8.26
* joblib==0.17.0
* kiwisolver==1.3.1
* llvmlite==0.34.0
* lmfit==1.0.1
* matplotlib==3.3.2
* numba==0.51.2
* numpy==1.19.4
* pandas==1.1.4
* Pillow==8.0.1
* pyparsing==2.4.7
* python-dateutil==2.8.1
* pytz==2020.4
* scikit-learn==0.23.2
* scipy==1.5.4
* seaborn==0.11.0
* six==1.15.0
* threadpoolctl==2.1.0
* umap-learn==0.4.6
* uncertainties==3.1.4

# Citation/Publication

If you found usage of this piepline helpful, please consider citation of the following paper. We highly appreciate any acknowledgement. 

Info on publication status: Paper submitted.

# Contact 

Of note, please use the Issue GitHub functionality of this repository to report bugs.
Nevertheless, you can contact us if you have any question or requests for a feature functions of the pipeline via [e-mail](mailto:h.nolte@age.mpg.de?subject=ComplexFinder%20Request). 


 


