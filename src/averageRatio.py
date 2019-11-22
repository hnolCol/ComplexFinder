import pandas as pd 
import numpy as np 
import os



X = pd.read_csv("data.tx",sep="\t")
X = X.set_index("Uniprot ID")

peaks = pd.read_csv("singlePeakFit.txt",sep="\t")

meanRatios = []

for proteinID, row in peaks.iterrows():
    c = row["center"]
    idx1 = c - row["fwhm"] 
    idx2 = c + row["fwhm"] 
    peakColumns = X.columns[idx1:idx2]
    r = np.mean(X.loc[proteinID,peakColumns])
    meanRatios.append(r)


peaks["meanSILAC"] = meanRatios


peaks.to_csv("addedRToFittedPeaks.txt",sep="\t")
