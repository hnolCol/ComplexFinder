
import os
import pandas as pd
import itertools
import random 

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
                                    filterDb = {'Organism': ["Mouse"]}, 
                                    complexNameColumn = "ComplexName",
                                    complexNameFilterString = None, 
                                    falsePositives=True, 
                                    matchSize=True):
        ""
        if self._checkIfFilteredFileExists(dbID,filterDb):

            self.df = self._loadFile()
            print("File was saved already and is loaded")
            return self.df

        else:

            df = pd.DataFrame( columns=["InteractionID", "ComplexName", "Entry1","Entry2","Class"])
            filteredDB = self._filterDb(dbID,filterDb,complexIDsColumn,complexNameColumn,complexNameFilterString)
            self.df = self._findPositiveInteractions(filteredDB,df,dbID,complexNameColumn)
            print("{} interactions found using the given filter criteria.".format(self.df.index.size))
            if falsePositives:
                self.df = self._generateFalseInterctions(self.df,filteredDB, matchSize)
            self._saveFilteredDf(dbID)
            return self.df 

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
        
        for n in range(numOfFalseComplex):
            for idx1,idx2 in self._generateRandomIndexPairs(max=filteredDB.size-1):
                df = self._appendToDF(
                    df,
                    nInteractions+n,
                    "Fake Complex {}".format(n),
                    self._getRandomEntry(filteredDB,idx1),
                    self._getRandomEntry(filteredDB,idx2),
                    "Non interactor"
                    )
        return df

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
        yield (random.randint(min,max),random.randint(min,max))
        

    def _findPositiveInteractions(self,filteredDB, df, dbID, complexNameColumn):
        ""
        for i, interactors in filteredDB.iteritems():

            for interaction in self._getPariwiseInteractions(interactors.split(";")):
                df = self._appendToDF(df,
                                    df.index.size,
                                    self.dbs[dbID].loc[i,complexNameColumn],
                                    interaction[0],
                                    interaction[1],
                                    "Interactors")
        
        return df

    def _appendToDF(self,df,ID,complexName,entry1,entry2,predictClass):
                        
        return df.append({"InteractionID": ID,
                           "ComplexName" : complexName,
                           "Entry1":entry1,
                           "Entry2":entry2,
                           "E1;E2":"{};{}".format(entry1,entry2),
                           "E2;E1":"{};{}".format(entry2,entry1),
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
        return self.df.loc[:,['InteractionID','E1;E2','E2;E1',"Class"]]


    def matchInteractions(self,):
        ""
        

    def fillComplexMatrixFromData(self, X):
        ""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas data frame with index and columns containg ID")
        
        return X.merge(self.df,how="left",left_index=True,right_on="E1;E2")


        



if __name__ == "__main__":
    Database().pariwiseProteinInteractions("subunits(UniProt IDs)")

    