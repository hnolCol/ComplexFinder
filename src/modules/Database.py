
import os
import pandas as pd
import numpy as np
import itertools
import random 
import gc 
from joblib import Parallel, delayed
import itertools
import time
import pickle
from collections import OrderedDict

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Database(object):


    def __init__(self):
        ""
        self.dbs = dict() 
        self.params = {"n_jobs":4}
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

            df = pd.DataFrame( columns=["InteractionID", "ComplexName", "E1","E2","E1E2","Class"])
            filteredDB = self._filterDb(dbID,filterDb,complexIDsColumn,complexNameColumn,complexNameFilterString)
            self.df = self._findPositiveInteractions(filteredDB,df,dbID,complexNameColumn)
            print("{} interactions found using the given filter criteria.".format(self.df.index.size))
            self._saveFilteredDf(dbID)


    def addDecoy(self):
        ""
        complexIdx = np.unique(self.df["ComplexID"])
        if complexIdx.size == 0:
            raise ValueError("No positive hits found in complex.")
        complexMembers = self.df["ComplexID"].value_counts()
        nData = len(self.df.index)
        randCombinations = np.random.randint(low=0,high=complexIdx.size, size = (nData,2))
        decoyData = []

        print("\nCreating decoy db for {} interactions".format(nData))
        for n,(x1,x2) in enumerate(randCombinations):
            e1Idx = np.random.randint(0,complexMembers.loc[complexIdx[x1]]) 
            e2Idx = np.random.randint(0,complexMembers.loc[complexIdx[x2]]) 
            if x1 != x2:
                e1 =  self.df[self.df["ComplexID"] == complexIdx[x1]].iloc[e1Idx]["E1"]
                e2 =  self.df[self.df["ComplexID"] == complexIdx[x2]].iloc[e2Idx]["E2"]
                decoyData.append({"ComplexID":"F({})".format(n),"E1":e1,"E2":e2,"E1E2":''.join(sorted([e1,e2])),"complexName":"Fake({})".format(n),"Class":0})

            if n % int(nData*0.15) == 0:
                print(round(n/nData*100,2), "%")
        
        df = pd.concat([self.df,pd.DataFrame(decoyData)],ignore_index=True)
        df.index = np.arange(0,df.index.size)
        boolSelfInt = df["E1"] == df["E2"]
        self.df = df.loc[boolSelfInt == False,:]
        print("\nCreating decoy is done..")


    def filterDBByEntryList(self,entryList, maxSize = 10000):


        dbEntries = self.df[['E1', 'E2']].values
        print("entries in DB:", str(self.df.index.size))

        boolIdx = [e1 in entryList and e2 in entryList for e1,e2 in dbEntries]
        self.df = self.df.loc[boolIdx,:]
        if self.df.index.size > maxSize:
            self.df = self.df.sample(n=maxSize)
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
               collectedResult.append({"ComplexID":i,"E1":interaction[0],"E2":interaction[1],"E1E2":''.join(sorted(interaction)),"complexName":complexName,"Class":predictClass})
        return collectedResult



    def _findPositiveInteractions(self,filteredDB, df, dbID, complexNameColumn):
        ""

        pairWise =  Parallel(n_jobs=4)(delayed(self.collectPairwiseInt)(i,interactors,self.dbs[dbID].loc[i,complexNameColumn],1) for i, interactors in filteredDB.iteritems())
        
        df = pd.DataFrame([item for sublist in pairWise for item in sublist])
        return df


    def _getPariwiseInteractions(self,entryList):
        ""
        return itertools.combinations(entryList, 2)



    def _loadFileToPandas(self,fileName, path,sep="\t"):
        ""
        self.dbs[fileName] = pd.read_csv(
                                        os.path.join(path,fileName),
                                        index_col = "ComplexID",
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

    @property
    def indentifiedComplexes(self):
        if hasattr(self,'uniqueComplexesIdentified'):
            return self.uniqueComplexesIdentified
   
    def identifiableComplexes(self,complexMemberIds, ID = "20190823_CORUM.txt"):
        ""
        identifiableMebmers = OrderedDict()
        if hasattr(self,'uniqueComplexesIdentified'):
            for k in self.uniqueComplexesIdentified.keys():
                identifiableMebmers[k] = {}
                boolIdx = self.dbs[ID].index == k 
                complexData = self.dbs[ID][boolIdx]
                cMembers = complexData[complexMemberIds].tolist()[0].split(";")
                identifiableMebmers[k]["n"] = len(cMembers)
                identifiableMebmers[k]["members"] = cMembers
        
        return identifiableMebmers


    def assignComplexToProtein(self, e, complexMemberIds, complexIDColumn, ID = "20190823_CORUM.txt", filterDict = {'Organism': ["Human"]}):
        
        if hasattr(self,'uniqueComplexesIdentified') == False:
            self.uniqueComplexesIdentified = OrderedDict() 

        if ID in self.dbs:
            if hasattr(self,"filteredDfToMatch") == False:
               
                columnNames = list(filterDict.keys())
                boolIdx = self.dbs[ID].isin(filterDict)[columnNames].sum(axis=1) == len(columnNames)
                self.filteredDfToMatch = self.dbs[ID].loc[boolIdx,:]
            
            boolIdxC = [] 
            splitIDs = e.split(";")
            
            for cMembers in self.filteredDfToMatch[complexMemberIds].tolist():
                cMSplit = cMembers.split(";")
                boolIdxC.append(any(x in cMSplit for x in splitIDs))
     
            eDf = self.filteredDfToMatch[boolIdxC]

            complexesForE = eDf.index.tolist()

            for c in  complexesForE:

                if c not in self.uniqueComplexesIdentified:
                    self.uniqueComplexesIdentified[c] = {"n":1,"members":[e]} 
                else:
                    self.uniqueComplexesIdentified[c]["n"] += 1
                    self.uniqueComplexesIdentified[c]["members"].append(e)

            return ';'.join([str(x) for x in  complexesForE])



    def matchMetrices(self,pathToTmp,entriesInChunks,metricColumns):#metricDf):
        ""
        print("matching metrices to DB and decoy .. ")
        distanceFile = os.path.join(pathToTmp,"DBdistances.txt")
        if os.path.exists(distanceFile):

            print("File found and loaded")
            self.dfMetrices = pd.read_csv(distanceFile, sep="\t", index_col=False)

        else:
            
            t1 = time.time()
            newDBData = []

            chunks = self._createChunks(pathToTmp,entriesInChunks,metricColumns)
            newDBData = Parallel(n_jobs=self.params["n_jobs"],verbose=15)(delayed(self._chunkInteraction)(c) for c in chunks)
            newDBData = list(itertools.chain(*newDBData))
    
            print("\n\nTime to macht {} interactions: {} secs".format(len(newDBData),time.time()-t1))

            
            self.dfMetrices  = pd.DataFrame([x for x in newDBData if x is not None])
            print(self.dfMetrices )
            self.dfMetrices.to_csv(os.path.join(pathToTmp,"DBdistances.txt"),sep="\t", index=False)


    def _createChunks(self,pathToTmp,entriesInChunks,metricColumns):
        ""
        print("prepare chunks ...")
        c = []
        folderPath = os.path.join(pathToTmp,"dbMatches")
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
        #n = int(self.df.index.size/self.params["n_jobs"])
        for n,chunk in enumerate(chunks(self.df.index,250)):
            print("chunk : {}".format(n))
            chunkItems = [self._createSignleChunk(idx,entriesInChunks,pathToTmp,metricColumns) for idx in chunk]
            with open(os.path.join(folderPath, str(n)+".pkl"),"wb") as f:
                pickle.dump(chunkItems,f)
            
            c.append(os.path.join(folderPath, str(n)+".pkl"))
           
        del chunkItems
        gc.collect()
        
        return c

    def _createSignleChunk(self,idx, entriesInChunks,pathToTmp,metricColumns):
        E1 = self.df.loc[idx,"E1"]
        E2 = self.df.loc[idx,"E2"] 
        E1E2 = ''.join(sorted([E1,E2]))
        className = self.df.loc[idx,"Class"]
        requiredFiles = ["{}.npy".format(k) for k,v in  entriesInChunks.items() if E1E2 in v]
        return {"E1":E1,"E2":E2,"E1E2":E1E2,"className":className,"requiredFiles":requiredFiles,"metricColumns":metricColumns,"pathToTmp":pathToTmp}

    def _chunkInteraction(self,pathToChunk):
        with open(pathToChunk,"rb") as f:
            chunkItems = pickle.load(f)
        return [self.findInteraction(**c) for c in chunkItems]



    def findInteraction(self,E1,E2,E1E2,className,requiredFiles,pathToTmp,metricColumns):
            
            
            for f in requiredFiles:

                data = np.load(os.path.join(pathToTmp,f),allow_pickle=True)

                boolIdx = data[:,2] == E1E2

               # boolIdx = np.logical_and(data[:,0] == E1, data[:,1] == E2)

                if any(boolIdx):
                    dataDir = OrderedDict([(metric,data[boolIdx,n+4][0]) for n,metric in enumerate(metricColumns)])
                else:
                    continue

                dataDir["E1"] = E1
                dataDir["E2"] = E2 
                dataDir["E1E2"] = E1E2
                dataDir["Class"] = className  
                del data
                gc.collect()  
                return dataDir

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

    