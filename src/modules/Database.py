
import os
import pandas as pd
import numpy as np
import itertools
import random 
import gc 
from joblib import Parallel, delayed

import time

class Database(object):


    def __init__(self):
        ""
        self.dbs = dict() 
        self._load()

    def _load(self):
        ""
        folderPath = self._getPathToReferenceFiles()
        self._loadFiles(folderPath)
    
    def _loadFiles(self, folderPath):
        ""
        for f in self._getFiles(folderPath):
            self._loadFileToPandas(f,folderPath)
        
    
    def _filterDb(self,ID, filterDict,complexIDsColumn,complexNameColumn,complexNameFilterString):
        ""
        if ID in self.dbs:
            columnNames = list(filterDict.keys())
            boolIdx = self.dbs[ID].isin(filterDict)[columnNames].sum(axis=1) == len(columnNames)
            filteredDf = self.dbs[ID].loc[boolIdx,:]
            if complexNameFilterString is None:
                return filteredDf.loc[:,complexIDsColumn]
            else:
                if complexNameColumn in self.dbs[ID].columns:
                    boolIdx = filteredDf[complexNameColumn].str.contains(complexNameFilterString)
                    return filteredDf.loc[boolIdx,complexIDsColumn]
                else:
                    raise ValueError("complexNameColumn not in database")


    def pariwiseProteinInteractions(self, 
                                    complexIDsColumn,
                                    dbID = "20190823_CORUM.txt", 
                                    filterDb = {'Organism': ["Human"]}, 
                                    complexNameColumn = "ComplexName",
                                    complexNameFilterString = None):
   
        ""
        if self._checkIfFilteredFileExists(dbID,filterDb):

            self.df = self._loadFile()
            print("File was saved already and is loaded")

        else:

            df = pd.DataFrame( columns=["InteractionID", "ComplexName", "Entry1","Entry2","Class"])
            filteredDB = self._filterDb(dbID,filterDb,complexIDsColumn,complexNameColumn,complexNameFilterString)
            self.df = self._findPositiveInteractions(filteredDB,df,dbID,complexNameColumn)
            print("{} interactions found using the given filter criteria.".format(self.df.index.size))
            self._saveFilteredDf(dbID)


    def addDecoy(self):
        ""
        complexIdx = np.unique(self.df["ComplexID"])
        complexMembers = self.df["ComplexID"].value_counts()
        nData = len(self.df.index)
        randCombinations = np.random.randint(low=0,high=complexIdx.size, size = (nData,2))
        decoyData = []

        print("\nCreating decoy db for {} interactions".format(nData))
        for n,(x1,x2) in enumerate(randCombinations):
            e1Idx = np.random.randint(0,complexMembers.loc[complexIdx[x1]]) 
            e2Idx = np.random.randint(0,complexMembers.loc[complexIdx[x2]]) 
            e1 =  self.df[self.df["ComplexID"] == complexIdx[x1]].iloc[e1Idx]["E1"]
            e2 =  self.df[self.df["ComplexID"] == complexIdx[x2]].iloc[e2Idx]["E2"]

            decoyData.append({"ComplexID":"F({})".format(n),"E1":e1,"E2":e2,"complexName":"Fake({})".format(n),"Class":0})

            if n % 30 == 0:
                print(round(n/nData*100,2), "% done")
        
        df = pd.concat([self.df,pd.DataFrame(decoyData)],ignore_index=True)
        df.index = np.arange(0,df.index.size)
        boolSelfInt = df["E1"] == df["E2"]
        self.df = df.loc[boolSelfInt == False,:]
        print("\nCreating decoy is done..")


    def filterDBByEntryList(self,entryList):


        dbEntries = self.df[['E1', 'E2']].values
        print("entries in DB:", str(self.df.index.size))

        boolIdx = [e1 in entryList and e2 in entryList for e1,e2 in dbEntries]
        self.df = self.df.loc[boolIdx,:]
        print("filtered database, new size: ",str(self.df.index.size))


    def _loadFile(self):
        ""
        return pd.read_csv(self.pathToFile,sep="\t",index_col=False)

    def _checkIfFilteredFileExists(self,dbID,filterDb):
        ""
        fileName = self._generateFileName(dbID,filterDb)
        sourcePath = self._getPathToReferenceFiles()
        self.pathToFile = os.path.join(sourcePath,fileName)
        return os.path.exists(self.pathToFile)

    def _generateFileName(self,dbName,filterDb):
        ""
        fileName = dbName.replace('.txt','')
        for k,v in filterDb.items():
            fileName = fileName + "_{}_".format(k) + '_'.join(v)
        return fileName + '.txt'

    def _saveFilteredDf(self,fileName):
        ""
        self.df.to_csv(self.pathToFile,
                        sep="\t",
                        index=False)

    def _generateFalseInterctions(self,df,filteredDB,matchSize):

        if isinstance(matchSize,bool) and matchSize:

            numOfFalseComplex = df.index.size
        elif isinstance(matchSize,int):
            numOfFalseComplex = matchSize
        else:
            raise ValueError("matchSize must be boolean or int")

        nInteractions = df.index.size
        
        print("Generating {} false protein protein interactions".format(nInteractions))
        
        maxInt = filteredDB.size-1

        allInteractors = pd.unique(df[['E1', 'E2']].values.ravel('K'))
       # randomParis =  Parallel(n_jobs=4)(delayed(self._generateRandomIndexPairs)(max=maxInt) for n in range(numOfFalseComplex))


        falsePairs = Parallel(n_jobs=4)(delayed(self.randomPairs)(n, max=maxInt, allInteractors = allInteractors, nInts = allInteractors.size-1) for n in range(numOfFalseComplex))

        return pd.concat([df,pd.DataFrame(falsePairs)])


    def _getRandomEntries(self,allInteractors,idxPair):
        idx1,idx2 = idxPair
        return (self._getRandomEntry(filteredDB,idx1),
                self._getRandomEntry(filteredDB,idx2))

    def randomPairs(self,n,max,allInteractors,nInts):

        x1, x2 = random.randint(0,nInts), random.randint(0,nInts)
        e1, e2 = allInteractors[x1], allInteractors[x2]
        
        return {"ComplexID":nInts+n,"E1":e1,"E2":e2,"complexName":"Fake({})".format(n),"Class":0}


    def _getRandomEntry(self,filteredDB,idx):
        ""
        entries = filteredDB.iloc[idx].split(";")
        nEntries = len(entries)
        if nEntries == 1:
            return entries[0]
        else:
            idx = random.randint(0,len(entries)-1)
            return entries[idx]

        
    def _generateRandomIndexPairs(self,min=0,max=100):
        ""
        return (random.randint(min,max),random.randint(min,max))
        

    def collectPairwiseInt(self,i,interactors,complexName,predictClass):

        collectedResult = []
        for interaction in self._getPariwiseInteractions(interactors.split(";")):
               collectedResult.append({"ComplexID":i,"E1":interaction[0],"E2":interaction[1],"complexName":complexName,"Class":predictClass})
        return collectedResult



    def _findPositiveInteractions(self,filteredDB, df, dbID, complexNameColumn):
        ""

        pairWise =  Parallel(n_jobs=4)(delayed(self.collectPairwiseInt)(i,interactors,self.dbs[dbID].loc[i,complexNameColumn],1) for i, interactors in filteredDB.iteritems())
        
        df = pd.DataFrame([item for sublist in pairWise for item in sublist])
        return df

    def _appendToDF(self,df,ID,complexName,entry1,entry2,predictClass):
                        
        return df.append({"InteractionID": ID,
                           "ComplexName" : complexName,
                           "E1":entry1,
                           "E2":entry2,
                           "Class":predictClass}, ignore_index=True)


    def _getPariwiseInteractions(self,entryList):
        ""
        return itertools.combinations(entryList, 2)



    def _loadFileToPandas(self,fileName, path,sep="\t"):
        ""
        self.dbs[fileName] = pd.read_csv(
                                        os.path.join(path,fileName),
                                        sep=sep)

    def _getFiles(self, folderPath, extn = 'txt'):
        ""
        return [f for f in os.listdir(folderPath) if f.endswith(extn)]


    def _getPathToReferenceFiles(self):
        ""
        filePath = os.path.dirname(os.path.realpath(__file__))
        mainPath = os.path.abspath(os.path.join(filePath ,"../.."))
        pathToReferenceFolder = os.path.join(
                mainPath,
                'reference-data'
            )
        return pathToReferenceFolder


    def findComplex(self,pair,df):
        e1,e2 = pair.split(";")
        if e1 in df.index and e2 in df.index:
            return df.loc[e1,e2].value

    @property
    def dbInteractions(self):
        ""
        return self.df

    def matchRowsToMatrix(self,row,distM):
        ""
        e1, e2 = str(row).split(";")

        if e1 in distM.index and e2 in  distM.index:
            return distM.loc[e1,e2] if distM.loc[e1,e2] != np.nan else distM.loc[e2,e1]

        else:
            return np.nan 

    def findMatch(self,x,metricDf, mCols):

        search = str(x[0]) + str(x[1])
        if search in metricDf["E1E2"].values:
            return metricDf.loc[metricDf["E1E2"] == search,mCols]
        elif search in metricDf["E2E1"].values:
            return metricDf.loc[metricDf["E2E1"] == search,mCols]


    def matchMetrices(self,pathToTmp):#metricDf):
        ""
        distanceFile = os.path.join(pathToTmp,"DBdistances.txt")
        if os.path.exists(distanceFile):
            print("File found and loaded")
            self.df = pd.read_csv(distanceFile, sep="\t", index_col=False)

        else:

            txtFiles = [f for f in os.listdir(pathToTmp) if f.endswith(".npy")]
            metricColumns = ["apex","euclidean","pearson","p_pearson","max_location"]
            t1 = time.time()
            newDBData = []

            newDBData = Parallel(n_jobs=4,verbose=10,prefer="threads")(delayed(self.findInteraction)(self.df.loc[idx,"E1"],
                        self.df.loc[idx,"E2"],
                        self.df.loc[idx,"Class"],   
                        txtFiles,pathToTmp,metricColumns) for idx in self.df.index)

        #    for idx in self.df.index:
         #       dataDir = self.findInteraction(idx,txtFiles,pathToTmp,metricColumns)
          #      if dataDir is not None:
           #         newDBData.append(dataDir)
#
   #             if idx % 100 == 0:
 #                   print(idx, " matches of metrices to DB done.")
  #              

            print("Time to macht Interactions for: ",time.time()-t1)


            self.df = pd.DataFrame([x for x in newDBData if x is not None])
            print(self.df)
            self.df.to_csv(os.path.join(pathToTmp,"DBdistances.txt"),sep="\t", index=False)




    def findInteraction(self,E1,E2,className,txtFiles,pathToTmp,metricColumns):
            for f in txtFiles:

                data = np.load(os.path.join(pathToTmp,f),allow_pickle=True).reshape(-1,7)

                boolIdx = np.logical_and(data[:,0] == E1, data[:,1] == E2)
            

                if any(boolIdx):
                    dataDir = {metric:data[boolIdx,n+2][0] for n,metric in enumerate(metricColumns)}

                else:
                   # boolIdx = data[:,2] == E1 + data[:,1] == E2
                    boolIdx = np.logical_and(data[:,1] == E1, data[:,0] == E2)
                    if any(boolIdx):
                        dataDir = {metric:data[boolIdx,n+2][0] for n,metric in enumerate(metricColumns)}

                    else:
                        continue

                dataDir["E1"] = E1
                dataDir["E2"] = E2 
                dataDir["Class"] = className  
                del data
                gc.collect()  
                return dataDir




      #      dataDir = {}

            

       #     fileE1 = E1 + ".npy"
       #     fileE2 = E2 + ".npy"#
#
#            if fileE1 in txtFiles:
#
 #               df = np.load(os.path.join(pathToTmp,fileE1)) # read_csv(os.path.join(pathToTmp,fileE1), index_col=False)
#
 #               if E2 in df[:,1]:
#
 #                   boolIdx = df[:,1] == E2
  ##                  dataDir = {metric:df[boolIdx,n+2][0] for n,metric in enumerate(metricColumns)}
#
 #           if fileE2 in txtFiles:
                
 #               df = np.load(os.path.join(pathToTmp,fileE2))
#
  #              if E1 in df[:,1]:

   #                 boolIdx = df[:,1] == E1
    #                dataDir = {metric:df[boolIdx,n + 2][0] for n,metric in enumerate(metricColumns)}
     #       
#
 ##           if len(dataDir) > 0:
#
      #          dataDir["E1"] = E1
 #               dataDir["E2"] = E2 
  #              dataDir["Class"] = self.df.loc[idx,"Class"]   
   #             del df
    #            gc.collect()  
     #           return dataDir

            

        
        

#        boolIdx = self.df[["E1","E2"]].apply(lambda x, df = metricDf, mCols = metricColumn :self.findMatch(x,df, mCols) , axis=1)
 #       print(boolIdx)

  #      print("HWWWW")
   #     print(metricDf)
    #    print(self.df)
     #  # print(self.df.columns)
        #self.df = self.df.merge(metricDf, how="inner", on = ["E1","E2"])
      #  print(self.df)
       # #df = self.df.merge(metricDf, how="inner", on = ["E2","E2"])
       
       # boolIdx = metricDF[["E1","E2"]].apply(axis=1)
        

    def matchInteractions(self,columnLabel, distanceMatrix):
        ""

        self.df[columnLabel] = self.df["E1;E2"].apply(lambda row, distM = distanceMatrix: self.matchRowsToMatrix(row,distM))


    def fillComplexMatrixFromData(self, X):
        ""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas data frame with index and columns containg ID")
        
        return X.merge(self.df,how="left",left_index=True,right_on="E1;E2")


if __name__ == "__main__":
    Database().pariwiseProteinInteractions("subunits(UniProt IDs)")

    