
## Download Protein-Protein Interaction Data

To run the ComplexFinder pipeline, you have to provide a positive set protein-protein interactions.
Below we provide examples and specific settings for frequently used databases of protein-protein interactions.

### CORUM

Download the dataset from the [Website link](https://mips.helmholtz-muenchen.de/corum/) and save it to reference-data folder in ComplexFinder.
If not present, add a column with the header ComplexID providing a unique ID for each complex.
The CORUM database contains complexes for mammalian systems therefore we need to pass a filterDictionary as shown below (databaseFilter). 
You can pass any column of the database as the key, and the target value for which we want to filter as a list. 
The parameter databaseEntrySplitString gives the splitstring by which the Uniprot identifiers (or any other feature ID matching your input data) of complexes are separated.

```python
ComplexFinder(
    ...
    databaseFilter = {'Organism': ["Human"]},
    databaseIDColumn = "subunits(UniProt IDs)",
    databaseEntrySplitString = ";", 
    databaseFileName = "CORUM.txt" #depends on how you save the COURM database
    ).run(...)
```

### Complex Portal 

Go the [Complex Portal Website](https://www.ebi.ac.uk/complexportal/home) and download the database (save it as HUMAN_COMPLEX_PORTAL.txt) for the utilized organismn. 


```python
ComplexFinder(
    databaseFileName="HUMAN_COMPLEX_PORTAL.txt", #depends on how you save the Complex Portal database
    databaseIDColumn= "Expanded participant list",
    databaseEntrySplitString = "|",              
    databaseFilter = {}
    ).run(...)

```


### hu.Map 2.0 

The hu.MAP 2.0 has recently beend published and is available at this [link](http://humap2.proteincomplexes.org).

```python
ComplexFinder(
    databaseFileName="humap2.txt", #depends on how you save the Complex Portal database
    databaseIDColumn= "subunits(UniProt IDs)", #requires renaming
    databaseEntrySplitString = ";",              
    databaseFilter = {"Confidence":[1,2,3,4]} #example to filter for a spcific complex confidence
    ).run(...)

```