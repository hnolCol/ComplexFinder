


import pandas as pd 
import numpy as np 
df = pd.read_csv("fittedPeaks.txt", sep="\t")



countDataFrame = pd.DataFrame(columns = ["Key","nOverlaps"])
for groupName, groupData in df.groupby("Key"):
    print(groupName)
    for idx,row in groupData.iterrows():
        fwhmHalf = row["fwhm"]/2
        low, high = row["center"] - fwhmHalf,row["center"] + fwhmHalf
        boolIdx = groupData["center"].between(low,high)
        peaksInRange = np.sum(boolIdx)
        countDataFrame = countDataFrame.append({"Key":row["key"],"nOverlaps":peaksInRage-1}, ignore_index = True)


countDataFrame.to_csv("OverlappingPeaks.txt",sep="\t")
