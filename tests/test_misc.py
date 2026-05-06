import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

# update sys.path until there is a proper package installation
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main import ComplexFinder

SAMPLE_DATA_DIR = Path(__file__).parents[1] / "example-data"


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
            takeRandomSampleFromData=False,
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
