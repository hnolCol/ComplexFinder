# ComplexFinder
 Finds complexes from Blue-Native and SEC Fractionation analyzed by Liquid Chromatogrpahy coupled to Mass Spectrometry. In 
 principal it works with any separation technique. 

 # Tool Features



 # Usage Example

Upon downlaod and extraction of the package. You can find example data in the example-data folder. 
To run the anaylsis, you can enter:

X = pd.read_table("./example-data/SILAC_01.txt", sep = "\t) #loading tab delimited txt file. 
ComplexFinder(analysisName = "ExampleRun_01").run(X)

 # Requirements

The following python packages are required to run the scripts. 
 . lmfit
 . matplotlib
 . numpy
 . pandas
 . scipy
 . seaborn
 . sklearn



 


