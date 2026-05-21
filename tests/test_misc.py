import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory, tempdir

import pandas as pd

from complexfinder import ComplexFinder
from complexfinder.Database import Database

REPO_ROOT = Path(__file__).parents[1]
SAMPLE_DATA_DIR = REPO_ROOT / "example-data"


def test_workflow_completes():
    """A basic test to check the workflow finishes without error."""
    with TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        # subset sample data for faster test run
        for f in (SAMPLE_DATA_DIR / "D1").glob("*.txt"):
            (
                pd.read_csv(f, sep="\t")
                .iloc[:500]
                .to_csv(Path(tmpdir) / f.name, sep="\t", index=False)
            )

        # use all cores when running under GitHub actions, use half otherwise
        n_jobs = (
            os.cpu_count() if "GITHUB_ACTIONS" in os.environ
            else min(1, os.cpu_count() // 2)
        )

        ComplexFinder(
            analysisMode="label-free",
            considerOnlyInteractionsPresentInAllRuns=1,
            compTabFormat=False,
            restartAnalysis=False,
            recalculateDistance=False,
            retrainClassifier=True,
            minPeakHeightOfMax=0.01,
            takeRondomSampleFromData=False,
            justFitAndMatchPeaks=False,
            noDistanceCalculationAndPrediction=False,
            runName="D1_exampleTest",
            noDatabaseForPredictions=False,
            rollingWinType="triang",
            idColumn="Uniprot ID",
            grouping={"interphase": ["D1_interphase.txt"],
                      "mitosis": ["D1_mitosis.txt"]},
            n_jobs=n_jobs,
            databaseFilter={'Organism': ["Human"]},
            databaseFileName="CORUM.txt",
            indexIsID=False,
            decoySizeFactor=1.1,
            classifierClass="random forest",
            minDistanceBetweenTwoPeaks=1,
            smoothWindow=3,
            classifierTestSize=0.20,
            maxPeaksPerSignal=20,
            smoothSignal=True,
            plotSignalProfiles=True,
            r2Thresh=0.75,
            correlationWindowSize=5,
            interactionProbabCutoff=0.7,
            minimumPPsPerFeature=2,
            removeSingleDataPointPeaks=True,
            keepOnlySignalsValidInAllConditions=False,
            quantFiles={},
            useRawDataForDimensionalReduction=False).run(
            tmpdir
        )


def test_custom_database_dir():
    """Test that a custom database directory can be used."""
    db_file = REPO_ROOT / "src" / "complexfinder" / "reference-data" / "CORUM.txt"
    with TemporaryDirectory() as tmpdir:
        shutil.copyfile(db_file, Path(tmpdir) / db_file.name)
        Database(databaseDir=tmpdir).pariwiseProteinInteractions(
            dbID=db_file.name,
            complexIDsColumn="subunits(UniProt IDs)",
            filterDb={'Organism': ["Human"]}
        )
        assert Path(tmpdir, f"{db_file.stem}_Organism__Human.txt").is_file()
