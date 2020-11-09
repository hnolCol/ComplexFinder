



import pandas as pd 
import os





df = pd.read_csv("nPeaks.txt",sep="\t")
df = df.set_index("Key")

dfPeaks = pd.read_csv("fittedPeaks.txt",sep="\t")
dfPeaks = dfPeaks.set_index("Key")

singlePeakBool = df["N"] == 1

singlePeakFitsBool = dfPeaks.index.isin(df[singlePeakBool].index)


dfPeaksF = dfPeaks[singlePeakFitsBool]

dfPeaksF.to_csv("singlePeakFit.txt",sep="\t")
