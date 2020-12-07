
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

def createSingleChunk(self,idx, entriesInChunks,pathToTmp,metricColumns,df):
    """
    Create required arguments for chunk. 

    Parameters
    ----------

    idx : pd.Index

    entriesInChunks : 

    pathToTmp : str


    metricColumns : obj`list` of obj`str``


    Returns
    -------
    dict 

    """
    E1 = df.loc[idx,"E1"]
    E2 = df.loc[idx,"E2"] 
    E1E2 = ''.join(sorted([E1,E2]))
    className = df.loc[idx,"Class"]
    requiredFiles = ["{}.npy".format(k) for k,v in  entriesInChunks.items() if E1E2 in v]
    return [{"E1":E1,"E2":E2,"E1E2":E1E2,"className":className,"requiredFiles":requiredFiles,"metricColumns":metricColumns,"pathToTmp":pathToTmp}]


class Database(object):


    def __init__(self, nJobs = 4):
        """Database Module. 

        The pipeline requires a database containing positve feature interactions.
        This module find interactions present in the dataset to be analysed,
        creates decoy interactions and matches metrices to databases. 


        Note
        ----
        
        Parameters
        ----------
        
       
        """
        self.dbs = dict() 
        self.params = {"n_jobs":nJobs}
        self._load()

    def _load(self):
        ""
        folderPath = self._getPathToReferenceFiles()
        self._loadFiles(folderPath)
    
    def _loadFiles(self, folderPath):
        """
        Load all txt files in a folder.

        Parameters
        ----------

        folderPath : str
    
        Returns
        -------
        None

        """
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
   
        """
        Pairwise protein interactions.

        Parameters
        ----------

        complexIDsColumn : str

        dbID : str

        filterDb : dict
        
        complexNameColumn : str

        complexNameFilterString : str or None

    
        Returns
        -------
        None

        """
        if self._checkIfFilteredFileExists(dbID,filterDb):

            self.df = self._loadFile()
            print("File was saved already and is loaded")

        else:

            df = pd.DataFrame( columns=["InteractionID", "ComplexName", "E1","E2","E1E2","Class"])
            filteredDB = self._filterDb(dbID,filterDb,complexIDsColumn,complexNameColumn,complexNameFilterString)
            self.df = self._findPositiveInteractions(filteredDB,df,dbID,complexNameColumn)
            print("Info :: {} interactions found using the given filter criteria.".format(self.df.index.size))
            self._saveFilteredDf(dbID)


    def addDecoy(self, sizeFraction = 1.2):
        """
        Adds a decoy database to the module.

        Random entries from positive data are taken and Fake
        complexes are build. Self-ineractions (x1 == x2) are
        not allowed and ignored. Duplicated interactions are
        also ignored as well as positive Interactions that is
        reported in a different positive complex.

        Parameters
        ----------

        sizeFraction : float
            Fraction of combinations (nData * sizeFraction). 
            If this is below 1, decoy db will be smaller than
            the positive interactions. It defaults to 1.2.
            

    
        Returns
        -------
        None

        """
        complexIdx = np.unique(self.df["ComplexID"])
        if complexIdx.size == 0:
            raise ValueError("Warning :: Aborting .. No positive hits found in complex.")
        complexMembers = self.df["ComplexID"].value_counts()
        nData = np.int64(len(self.df.index) * sizeFraction)
        #random combinations
        randCombinations = np.random.randint(low=0,high=complexIdx.size, size = (nData*3,2))
        decoyData = []
        E1E2sInDecoy = []
        prevLength = -1

        print("\nCreating decoy db for {} interactions".format(nData))
        for n,(x1,x2) in enumerate(randCombinations):
            e1Idx = np.random.randint(0,complexMembers.loc[complexIdx[x1]]) 
            e2Idx = np.random.randint(0,complexMembers.loc[complexIdx[x2]]) 
            if x1 != x2:
                e1 =  self.df[self.df["ComplexID"] == complexIdx[x1]].iloc[e1Idx]["E1"]
                e2 =  self.df[self.df["ComplexID"] == complexIdx[x2]].iloc[e2Idx]["E2"]
                E1E2 = ''.join(sorted([e1,e2]))
                if E1E2 not in self.df["E1E2"].values and E1E2 not in E1E2sInDecoy: #check if this is also reported as a positive interaction, some protein are in multiple complexes
                    decoyData.append({"ComplexID":"F({})".format(n),"E1":e1,"E2":e2,"E1E2":E1E2,"complexName":"Fake({})".format(n),"Class":0})
                    E1E2sInDecoy.append(E1E2)
            decoyLength = len(decoyData)
            if nData == decoyLength:
                break

            if decoyLength % int(nData*0.15) == 0 and decoyLength != prevLength:
                print(round(decoyLength /nData*100,2), "%")
                prevLength = decoyLength #avoid double per printing

        df = pd.concat([self.df,pd.DataFrame(decoyData)],ignore_index=True)
        df.index = np.arange(0,df.index.size)
        boolSelfInt = df["E1"] == df["E2"]
        self.df = df.loc[boolSelfInt == False,:]
        print("\nInfo :: Creating decoy database is done. Total size of db: {}".format(self.df.index.size))

    def loadDatabaseFromFile(self,pathToDatabase):
        ""
        self.df = pd.read_csv(pathToDatabase, sep="\t")
        boolClassOne = self.df["Class"] == 1
        return np.sum(boolClassOne)

    def filterDBByEntryList(self,entryList, maxSize = np.inf):


        dbEntries = self.df[['E1', 'E2']].values
        print("Info :: Entries in DB:", str(self.df.index.size))

        boolIdx = [e1 in entryList and e2 in entryList for e1,e2 in dbEntries] #insufficient, change!
        self.df = self.df.loc[boolIdx,:] #overwrite df
        sizeBefore = self.df.index.size
        self.df = self.df.drop_duplicates("E1E2")
        print("Info :: {} duplicates removed.".format(sizeBefore-self.df.index.size))
        if self.df.index.size > maxSize:
            self.df = self.df.sample(n=maxSize)
        newDBSize = self.df.index.size
        print("Info :: Filtered database, new size: ",str(newDBSize))
        return newDBSize

    def getInteractionClassByE1E2(self,E1E2s,E1s,E2s):
        ""
        # inter = 0
        # notInDB = 0
        # decoy = 0
        # pos = 0 

        E1E2Type = []

        #db = self.df.drop_duplicates("E1E2")
        db = self.df.set_index("E1E2")
        
        boolDBFilter = db.index.isin(E1E2s) #find indices that are present as positive interactors (e.g. in one complex)
        e1e2Unique = np.unique(self.df.loc[~boolDBFilter][["E1","E2"]].values) #remove positive interactions

        for n,E1E2 in enumerate(E1E2s):
            #boolIdx = self.df["E1E2"] == E1E2
            if E1E2 in db.index:
                E1E2Class = db.loc[E1E2,["Class"]].values[0]
                if E1E2Class == 1:
                    E1E2Type.append("pos")
                else:
                    E1E2Type.append("decoy")
            else:
                #if we get here, those itneractions cannot be positive
                e1 = E1s[n]
                e2 = E2s[n]
                if e1 in e1e2Unique and e2 in e1e2Unique:
                    E1E2Type.append("inter")
                else:
                    E1E2Type.append("unknown/novel")

        return pd.Series(E1E2Type)
        


    def _loadFile(self, *args, **kwargs):
        """
        Reads file

        Parameters
        ----------

        args
            Passed to pandas read_csv fn

        kwargs 
            Passed to pandas read_csv fn
    
        Returns
        -------
        pd.DataFrame

        """
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
            fileName = fileName + "_{}_".format(k) + '_'.join(str(v))
        return fileName + '.txt'

    def _saveFilteredDf(self,fileName):
        ""
        self.df.to_csv(self.pathToFile,
                        sep="\t",
                        index=False)
        

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
        """
        Assigns complex to protein.

        Parameters
        ----------

        e : str

        complexMemberIds 

        complexIDColumn : str.

        ID : str.

        filterDict : dict.
    
        Returns
        -------
        String of form ComplexID1;ComplexID2 or None

        """
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



    def matchMetrices(self,pathToTmp,entriesInChunks,metricColumns,analysisName,forceRematch = False):#metricDf):
        """
        Matches metrices to database. 

        Parameters
        ----------

        pathToTmp : string
            path to temp. folder

        entriesInChunks : obj `list` of obj`str`
            Entries in chunks.

        metricColumns : obj `list` of obj`str`
            List of metrices (strings)
    
        Returns
        -------
        None

        """
        if not hasattr(self,"dfMetrices"):
            self.dfMetrices = dict() 
        print("Info :: Matching metrices to DB and decoy .. ")
        distanceFile = os.path.join(pathToTmp,"result","DBdistances{}.txt".format(metricColumns))
        print("Info :: Distance File : {}".format(distanceFile))
        if not forceRematch and os.path.exists(distanceFile):

            self.dfMetrices[analysisName] = pd.read_csv(distanceFile, sep="\t", index_col=False)
            print("Info :: Database distances found.File loaded")

        else:
            
            t1 = time.time()
            newDBData = []

            chunks = self._createChunks(pathToTmp,entriesInChunks,metricColumns)
           # print(chunks)
            newDBData = Parallel(n_jobs=self.params["n_jobs"],verbose=15)(delayed(self._chunkInteraction)(c) for c in chunks)
            newDBData = list(itertools.chain(*newDBData))
    
            print("\n\n Info :: Time to match {} interactions: {} secs".format(len(newDBData),time.time()-t1))

            self.dfMetrices[analysisName]  = pd.DataFrame([x for x in newDBData if x is not None])
            #save data to csv
            self.dfMetrices[analysisName].to_csv(distanceFile, sep="\t", index=False)


    def _createChunks(self,pathToTmp,entriesInChunks,metricColumns):
        """
        Craetes chunks


        To do:

        Parellelerize.

        Parameters
        ----------

        pathToTmp : string
            path to temp. folder

        entriesInChunks : obj `list` of obj`str`
            Entries in chunks.

        metricColumns : obj `list` of obj`str`
            List of metrices (strings)
    
        Returns
        -------
        None

        """
        print("Info :: Prepare database chunks ...")
        c = []
        folderPath = os.path.join(pathToTmp,"dbMatches")
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
        #n = int(self.df.index.size/self.params["n_jobs"])
        for n,chunk in enumerate(chunks(self.df.index,400)):
            
            chunkPath = os.path.join(folderPath, str(n)+".pkl")
           
            chunkItems = [self._createSingleChunk(idx,entriesInChunks,pathToTmp,metricColumns,self.df) for idx in chunk]
            #chunkItems = Parallel(n_jobs=self.params["n_jobs"],verbose=15)(delayed(self._createSingleChunk)(idx,entriesInChunks,pathToTmp,metricColumns,df = self.df) for idx in chunk)
            #chunkItems = list(itertools.chain(*chunkItems))
            with open(chunkPath,"wb") as f:
                pickle.dump(chunkItems,f)
            
            c.append(chunkPath)
           
        
        return c

    def _createSingleChunk(self,idx, entriesInChunks,pathToTmp,metricColumns,df):
        """
        Create required arguments for chunk. 

        Parameters
        ----------

        idx : pd.Index

        entriesInChunks : 

        pathToTmp : str


        metricColumns : obj`list` of obj`str``


        Returns
        -------
        dict 

        """
        E1 = df.loc[idx,"E1"]
        E2 = df.loc[idx,"E2"] 
        E1E2 = ''.join(sorted([E1,E2]))
        className = df.loc[idx,"Class"]
        requiredFiles = []
        if E1E2 in entriesInChunks:
            requiredFiles.append("{}.npy".format(entriesInChunks[E1E2]))
          
        #requiredFiles = ["{}.npy".format(k) for k,v in  entriesInChunks.items() if E1E2 in v]
        return {"E1":E1,"E2":E2,"E1E2":E1E2,"className":className,"requiredFiles":requiredFiles,"metricColumns":metricColumns,"pathToTmp":pathToTmp}

    def _chunkInteraction(self,pathToChunk):
        ""
        if not os.path.exists(pathToChunk):
            return []
        with open(pathToChunk,"rb") as f:
            chunkItems = pickle.load(f)
        return [self.findInteraction(**c) for c in chunkItems]



    def findInteraction(self,E1,E2,E1E2,className,requiredFiles,pathToTmp,metricColumns):
        """
        Finds interactions in chunks. 

        Parameters
        ----------

        E1 : 

        E2 : 
        
        E1E2 : 

        className : 

        requiredFiles : 

        pathToTmp : str

        metricColumns : obj`list` of obj`str``
            List of distance metrices.


        Returns
        -------
        OrderedDict that contains detected Interactions.

        """   
            
        for f in requiredFiles:
            
            data = np.load(os.path.join(pathToTmp,"chunks",f),allow_pickle=True)

            boolIdx = data[:,2] == E1E2

            # boolIdx = np.logical_and(data[:,0] == E1, data[:,1] == E2)

            if np.any(boolIdx):
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

    