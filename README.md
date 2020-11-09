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
The rating of the clustering takes into account, the number of correct protein-protein interactions (defined by CORUM database), the total number of protein-protein interactions (basically the amount of noise found by HDBSCAN). The size of a cluster is not considered in the clustering score calculation. 

 ![Cluster identifiation](/img/workflow.png)

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

Additionally, you can pass a list of datasets. 
```python
x01 = pd.read_table("./example-data/SILAC_01.txt", sep = "\t") #loading tab delimited txt file. 
x02 = pd.read_table("./example-data/SILAC_02.txt", sep = "\t") #loading tab delimited txt file. 
ComplexFinder(analysisName = ["SILAC_01","SILAC_02"]).run([x01,x02]) #the result folders will be called as the anaylsisName
```


## Parameters
Find below parameters to set. The default is given in brackets after the parameter name.

* indexIsID [True] bool.
* plotSignalProfiles [True] bool.
* removeSingleDataPointPeaks [True] bool.
* <del>savePeakModels [True]</del> *depracted. always True and will be removed in the next version*.
* maxPeaksPerSignal [9]
* minDistBetweenPeaks [3] Distance in fractions (int) between two peaks. Setting this to a smaller number results in more peaks.
* n_jobs [4] Number of workers to model peaks, to calculate distance pairs and to train and use the classifer.
* kFold [5] Cross validation of classifier.
* analysisName [None]
* idColumn ["Uniprot ID"]
* databaseName ["CORUM"]
* peakModel ["LorentzianModel"],
* imputeNaN [True],
* classifierClass ["random_forest"] string. Must be string 
* retrainClassifier [False]
* interactionProbabCutoff [0.7]
* metrices [["apex","euclidean","pearson","p_pearson","spearman","max_location"]]
* classiferGridSearch = RF_GRID_SEARCH (see below). dict with keywords matching parameters/settings of estimator (SVM, random forest) and list of values forming the grid used to find the best estimator settings (evaluated by k-fold cross validation). Runtime is effected by number of parameter settings as well as k-fold. 
* smoothSignal [True]):

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



# Contact 

Of note, please use the Issue functionality in GitHub to report bugs.
Nevertheless, you can contact us if you have any question or requests for a feature via [e-mail](mailto:h.nolte@age.mpg.de?subject=ComplexFinder%20Request). 


 


