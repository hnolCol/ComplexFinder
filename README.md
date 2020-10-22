# ComplexFinder
 Finds complexes from Blue-Native and SEC Fractionation analyzed by Liquid Chromatogrpahy coupled to Mass Spectrometry. In 
 principal it works with any separation technique. 

 ## Workflow


 ![Workflow](/img/workflow.png)



 # Usage Example

Upon downlaod and extraction of the package. You can find example data in the example-data folder. 
To run the anaylsis, you can enter:
```python
X = pd.read_table("./example-data/SILAC_01.txt", sep = "\t) #loading tab delimited txt file. 
ComplexFinder(analysisName = "ExampleRun_01").run(X)
```

## Parameters
Find below parameters to set. The default is given in brackets after the parameter name.
* indexIsID [True]
* plotSignalProfiles [True]
* removeSingleDataPointPeaks = True,
* savePeakModels = True,
* maxPeaksPerSignal = 9,
* n_jobs = 4,
* kFold = 5,
* analysisName = None,
* idColumn = "Uniprot ID",
* databaseName="CORUM",
* peakModel = "LorentzianModel",
* imputeNaN = True,
* classifierClass = "random_forest",
* retrainClassifier = False,
* interactionProbabCutoff = 0.7,
* metrices = ["apex","euclidean","pearson","p_pearson","spearman","max_location"],
* classiferGridSearch = RF_GRID_SEARCH):

 # Requirements

The following python packages are required to run the scripts. 
 * lmfit
 * matplotlib
 * numpy
 * pandas
 * scipy
 * seaborn
 * sklearn



 


