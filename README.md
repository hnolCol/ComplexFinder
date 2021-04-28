# ComplexFinder

 Finds complexes from Blue-Native and SEC Fractionation analyzed by Liquid Chromatogrpahy coupled to Mass Spectrometry. In 
 principal it works with any separation technique resulting in co-elution signal profiles. To avoid licence issues and accumulation of old database files, please first download the database of choce (see below *Download Protein-Protein Interaction Data*). 

## Next Feature (Testing) Implementations (05.2021)

- [ ] Extend plotting capabibilties to extract profiles of features and complex
```python

#plotting selected feature's profile
ComplexFinder(analysisName="<Created folder in results folder>").plotFeature()

#plotting selected feature's distance metrics compared to the complete population (all features)
#due to scaling of features prior training of the predictor, boxplot should display scaled and raw features.
ComplexFinder(analysisName="<Created folder in results folder>").plotFeatureDistanceMetircs()

#plotting features of complex found by ComplexFinder (clusterLabels == ID)
ComplexFinder(analysisName="<Created folder in results folder>").plotComplexProfileByClusterLabel()

#plotting features of known complex in database (correspondng to ComplexID column in the database - see below)
ComplexFinder(analysisName="<Created folder in results folder>").plotComplexProfileInDatabaseByID() 

```
- [ ] Test a DeepLearning Implementation


## Workflow

For thousdands of features (peptides/protein) a signal was measured over different fractions. The applied technique separates protein clusters from each other. This package aims for different things:

* signal processing including filtering, smoothing.
* if more than one replicate is analysed, the profiles over fractions will be aligned. 
* identification of protein-protein interactions.
* identification of protein cluster using diminesional reduction and density based clustering. 

![Signal processing and protein-protein interaction prediction](/img/workflow.png)

Importantly, ComplexFinder can also be utized to analyse the data without prior knowledge of protein connectivitiy (e.g. positive database). In this case, there are two options: 

* using raw profile signal intensities
* distance between profile pairs 

which are then subjected for dimensional reduction and HDBSCAN clusering. Importantly, when using the raw profile intensities, the derived UMAP representation is aligned using the top N correlated features between samples (e.g. same protein across all samples). 

As a next step, we want to identify clusters of proteins with predicted interaction. To this end, we are using the interaction probabiliy matrix obtained by the
random forest classifier. We then apply the UMAP embedding calculton and apply HDBSCAN clustering. Again, we 
are using the CORUM database to quantify the the clustering result using the v-measure. Both techniques, UMAP and HDBSCAN are performed 
using a paramter grid to cycle through different options and find the best clustering. 

 ![Quantification](/img/quantDetails.png)

 In cases of uisng the raw signal intensity or the distance metrics, those data are subjected to dimensional reduction (UMAP) and clustering (HDBSCAN). Noteworthy, other clusering algorithmns are available and can be utilized. HDBSCAN is however the default. 

## Depositing Data analyzed using ComplexFinder

If you analyzed your data using ComplexFinder, we highly recommend to upload the data along the raw fiiles deposition at mass spectrometry repisatories such as PRIDE / ProteomeXChange or similiar. Especially, the params.json file which is written to the results folder is of particular interest in order to reproduce the data analysis. Of note, if you upload the complete result folder, other users will be able to analyse these data using the plotting utilities of ComplexFinder.

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
from .src.main import ComplexFinder
folderPath = os.path(".","<my folder>")
ComplexFinder().run(folderPath)
```
Additionally, you can pass a list of datasets. However, we  recommend to copy required datasets in a separate folder.
Results are saved by default in the results folder the ComplexFinder folder.

## Download Protein-Protein Interaction Data

To run the ComplexFinder pipeline, you have to provide a positive set protein-protein interactions.
Below we provide examples and specific settings for frequently used databases of protein-protein interactions.

### CORUM

Download the dataset from the [Website link](https://mips.helmholtz-muenchen.de/corum/) and save it to reference-data folder in ComplexFinder.
If not present, add a column with the header ComplexID providing a unique ID for each complex.
The CORUM database contains complexes for mammalian systems therefore we need to pass a filterDictionary as shown below (databaseFilter). 
You can pass any column of the database as the key, and the target value for which we want to filter as a list. 
The parameter databaseEntrySplitString gives the splitstring by which the Uniprot identifiers (or any other feature ID matching your input data) of complexes are separated.

```python
ComplexFinder(
    ...
    databaseFilter = {'Organism': ["Human"]},
    databaseIDColumn = "subunits(UniProt IDs)",
    databaseEntrySplitString = ";", 
    databaseFileName = "CORUM.txt" #depends on how you save the COURM database
    ).run(...)
```

### Complex Portal 

Go the [Complex Portal Website](https://www.ebi.ac.uk/complexportal/home) and download the database (save it as HUMAN_COMPLEX_PORTAL.txt) for the utilized organismn. 


```python
ComplexFinder(
    databaseFileName="HUMAN_COMPLEX_PORTAL.txt", #depends on how you save the Complex Portal database
    databaseIDColumn= "Expanded participant list",
    databaseEntrySplitString = "|",              
    databaseFilter = {}
    ).run(...)

```


### hu.Map 2.0 

The hu.MAP 2.0 has recently beend published and is available at this [link](http://humap2.proteincomplexes.org).

```python
ComplexFinder(
    databaseFileName="humap2.txt", #depends on how you save the Complex Portal database
    databaseIDColumn= "subunits(UniProt IDs)", #requires renaming
    databaseEntrySplitString = ";",              
    databaseFilter = {"Confidence":[1,2,3,4]} #example to filter for a spcific complex confidence
    ).run(...)

```

## Grouping of Replicates

The grouping parameter in ComplexFinder is used to group files, which is used to group replicates together. 
Assume, that we have 4 files, 2 KO and 2 WT files which we put together in the folder "./data". 
The grouping will be used to calculate pariwise statistics between fitted peaks. Moreover, complex prediction and protein-protein prediction summary.
```python
pathToFiles = os.path.join(".","data")
ComplexFinder(
    grouping = {
        "WT" : ["D3_WT_01.txt","D3_WT_02.txt"],
        "KO" : ["D3_KO_01.txt","D3_KO_02.txt"]
    }
            ).run(pathToFiles)
```


## Using ComplexFinder without protein connectivity

### Using raw signal profiles


### Using distances metrics


#### Using just the apex distance 



Please respect the respective liscence for the different databases.

## Parameters

- [ ] requires updating for new parameters and additional documentation.

Find below parameters to set. The default is given in brackets after the parameter name.
* alignMethod = "RadiusNeighborsRegressor",
* alignRuns = False, Alignment of runs is based on signal profiles that were found to have a single modelled peak. A refrence run is assign by correlation anaylsis and choosen based on a maximum R2 value. Then fraction-shifts per signal profile is calculated (must be in the window given by *alignWindow*). The fraction residuals are then modelled using the method provided in *alignMethod*. Model peak centers are then adjusted based on the regression results. Of note, the alignment is performed after peak-modelling and before distance calculations. 
* alignWindow = 3, Number of fraction +/- single-peal profile are accepted for the run alignment. 
* analysisMode = "label-free", #[label-free,SILAC,SILAC-TMT]
* analysisName = None,
* binaryDatabase = False,
* classifierClass = "random_forest",
* classifierTestSize = 0.25, Fraction of the created database containing positive and negative protein-protein interactions that will be used for testing (for example ROC curve analysis) and classification report.
* classiferGridSearch = RF_GRID_SEARCH, (see below). dict with keywords matching parameters/settings of estimator (SVM, random forest) and list of values forming the grid used to find the best estimator settings (evaluated by k-fold cross validation). Runtime is effected by number of parameter settings as well as k-fold. 
* considerOnlyInteractionsPresentInAllRuns = 2, Can be either bool to filter for protein - protein interactions that are present in all runs. If an integer is provided. the pp interactions are filtered based on the number of runs in which they were quantified. A value of 4 would indicate that the pp interaction must have been predicted in all runs. 
* databaseFilter = {'Organism': ["Human"]}, Filter dict used to find relevant complexes from database. By default, the corum database is filtered based on the column 'Organism' using 'Mouse' as a search string. If no filtering is required, pass an empty dict {}. 
* databaseIDColumn = "subunits(UniProt IDs)",
* databaseFileName = "20190823_CORUM.txt",
* databaseHasComplexAnnotations = True, Indicates if the provided database does contain complex annotations. If you have a database with only pairwise interactions, this setting should be *False*. Clusters are identified by dimensional reduction and density based clustering (HDBSCAN). In order to alter UMAP and HDBSCAN settings use the kewywords *hdbscanDefaultKwargs* and *umapDefaultKwargs*.
* decoySizeFactor = 1.2,
* grouping = {"WT": ["D3_WT_04.txt","D3_WT_02.txt"],"KO":["D3_KO_01.txt","D3_KO_02.txt"]}, None or dict. Indicates which samples (file) belong to one group. Let's assume 4 files with the name 'KO_01.txt', 'KO_02.txt', 'WT_01.txt' and 'WT_02.txt' are being analysed. The grouping dict should like this : {"KO":[KO_01.txt','KO_02.txt'],"WT":['WT_01.txt','WT_02.txt']} in order to combine them for statistical testing (e.g. t-test of log2 transformed peak-AUCs). Note that when analysis multiple runs (e.g. grouping present) then calling ComplexFinder().run(X) - X must be a path to a folder containing the files.
* hdbscanDefaultKwargs = {"min_cluster_size":4,"min_samples":1},
* indexIsID = False,
* idColumn = "Uniprot ID",
* interactionProbabCutoff = 0.7, Cutoff for estimator probability. Interactions with probabilities below threshold will be removed.
* kFold = 3, Cross validation of classifier optimiation.
* maxPeaksPerSignal = 15, Number of peaks allowed for on signal profile.
* maxPeakCenterDifference = 1.8,
* metrices = ["apex","pearson","euclidean","p_pearson","max_location","umap-dist"], Metrices to access distance between two profiles. Can be either a list of strings and/or dict. In case of a list of dicts, each dict must contain the keywords: 'fn' and 'name' providing a callable function with 'fn' that returns a single floating number and takes two arrays as an input.
* metricesForPrediction = None,#["pearson","euclidean","apex"],
* metricQuantileCutoff = 0.90,
* minDistanceBetweenTwoPeaks = 3, distance in fractions (int) between two peaks. Setting this to a smaller number results in more peaks.
* n_jobs = 12, number of workers to model peaks, to calculate distance pairs and to train and use the classifer.
* noDatabaseForPredictions = False, If you want to use ComplexFinder without any database. Set this to *True*.
* normValueDict = {},
* peakModel = "GaussianModel", which model should be used to model signal profiles. In principle all models from lmfit can be used. However, the initial parameters are only optimized for GaussianModel and LaurentzianModel. This might effect runtimes dramatically. 
* plotSignalProfiles = False, if True, each profile is plotted against the fractio along with the fitted models. If you are concerned about time, you might set this to False at the cost of losing visible asessment of the fit quality.
* plotComplexProfiles = False,
* precision = 0.5, Precision to use to filter protein-protein interactions. If None, the filtering will be performed based on the parameter *interactionProbabCutoff*.
* r2Thresh = 0.85, R2 threshold to accept a model fit. Models below the threshold will be ignored.
* removeSingleDataPointPeaks = True,
* restartAnalysis = False, bool. Set True if you want to restart the anaylsis from scratch. If the tmp folder exsists, items and dirs will be deleted first.
* retrainClassifier = False, if the trainedClassifier.sav file is found, the classifier is loaded and the training is skipped. If you change the classifierGridSearch, you should set this to True. This will ensure that the classifier training is never skipped.
* recalculateDistance = False,
* runName = None,
* rollingWinType = "triang", the win type used for calculating the rolling metric. If None, all points are evenly weighted. Can be any string of scipy.signal window function.
            (https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows)
* <del>savePeakModels = True</del> *depracted. always True and will be removed in the next version*.
* scaleRawDataBeforeDimensionalReduction = True, If raw data should be used (*useRawDataForDimensionalReduction*) enable this if you want to scale them. Scaling will be performed that values of each row are scaled between zero and one.
* smoothSignal = True, Enable/disable smoothing. Defaults to True. A moving average of at least 3 adjacent datapoints is calculated using pandas rolling function. Effects the analysis time as well as the nmaximal number of peaks detected.
* smoothWindow = 2,
* topNCorrFeaturesForUMAPAlignment = 200, Using top N features to to align UMAP Embeddings. The features are ranked by using Pearson correlation coefficient,
* useRawDataForDimensionalReduction = False, Setting this to true, will force the pipeline to use the raw values for dimensional reduction. Distance calculations are not automatically turned off and the output is generated but they are not used.
* umapDefaultKwargs = {"min_dist":0.0000001,"n_neighbors":3,"n_components":2},
* quantFiles = [] list of str.
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

# Database Quality

For the prediction of protein-protein interactions the quality and size of the database is of importance. 

As a quick test, we performed predictions using 2000 randomly selected features of dataset D1 and siwtched the class labels (interactor vs non-interactor) of the database to train the classifier. We observed that the number of predicted protein-protein interaction was strongly reduced in after label switch of more than 5% of the features. We have used the CORUM human database for interactions. This highlights that the complexes in the database need to describe the complexome in the measured dataset accurately. The gold-standard is therefore the usage of a complex database that were experimentally validated, which is sadly often not possible due to the workload.


# Usin SILAC - TMT peak centric quantifiaction

*in preparation* 

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
Easiest might be to check the default parameter which can be used to upload the CORUM database.
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
* Pillow==8.1.1
* pyparsing==2.4.7
* python-dateutil==2.8.1
* pytz==2020.4
* scikit-learn==0.23.2
* scipy==1.5.4
* seaborn==0.11.0
* six==1.15.0
* sklearn==0.0
* threadpoolctl==2.1.0
* umap-learn==0.4.6
* uncertainties==3.1.4

# Citation/Publication

If you found usage of this piepline helpful, please consider citation of the following paper. We highly appreciate any acknowledgement. 

Info on publication status: Paper submitted.

# Contact 

Of note, please use the Issue GitHub functionality of this repository to report bugs.
Nevertheless, you can contact us if you have any question or requests for a feature functions of the pipeline via [e-mail](mailto:h.nolte@age.mpg.de?subject=ComplexFinder%20Request). 


 


