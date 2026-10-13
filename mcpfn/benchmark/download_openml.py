"""
Download the numeric OpenML datasets used in the imputation benchmark.

The benchmark uses a fixed, hard-coded list of OpenML dataset IDs so that anyone
can re-run the exact same benchmark. These are the 41 datasets listed in the
OpenML datasets table (Table 9) of the TabImpute paper. They were originally
selected from OpenML with these criteria: no missing values, 50-250 rows,
5-60 features, all-numeric features, and excluding the synthetic "fri_*" datasets.

Usage (from mcpfn/benchmark):
    python download_openml.py

Each dataset is saved to datasets/openml/<name>/dataset.pkl, which is what
create_openml_missingness_datasets.py reads.

The categorical benchmark datasets live in download_openml_categorical.py.
"""
import os

import numpy as np
import openml
import pandas as pd

# name -> OpenML dataset ID (shape is rows x columns)
OPENML_DATASETS = {
    "alcohol-qcm-sensor": 42930,  # 125 x 15
    "analcatdata_election2000": 527,  # 67 x 15
    "analcatdata_gviolence": 518,  # 74 x 9
    "analcatdata_olympic2000": 530,  # 66 x 12
    "appendicitis_test": 45589,  # 106 x 8
    "appendicitis_test_edsa": 45591,  # 106 x 8
    "autoPrice": 207,  # 159 x 16
    "baskball": 214,  # 96 x 5
    "benzo32": 434,  # 195 x 33
    "breast-cancer-coimbra": 42900,  # 116 x 10
    "chatfield_4": 695,  # 235 x 13
    "chscase_vine1": 691,  # 52 x 10
    "dataset_analcatdata_creditscore": 46534,  # 100 x 7
    "edm": 41552,  # 154 x 18
    "EgyptianSkulls": 1099,  # 150 x 5
    "FacultySalaries": 1096,  # 50 x 5
    "Glass-Classification": 43750,  # 214 x 10
    "humans_numeric": 1058,  # 75 x 15
    "ICU": 1097,  # 200 x 20
    "machine_cpu": 230,  # 209 x 7
    "MercuryinBass": 1090,  # 53 x 10
    "metafeatures": 46307,  # 75 x 32
    "pollution": 542,  # 60 x 16
    "pwLinear": 229,  # 200 x 11
    "pyrim": 217,  # 74 x 28
    "rabe_131": 692,  # 50 x 6
    "rabe_148": 710,  # 66 x 6
    "rabe_265": 660,  # 51 x 7
    "Rainfall-in-Kerala-1901-2017": 43806,  # 117 x 18
    "sleuth_case1201": 707,  # 50 x 7
    "sleuth_ex1605": 687,  # 62 x 6
    "slump": 41558,  # 103 x 10
    "SMSA": 1091,  # 59 x 16
    "SolarPower": 42328,  # 204 x 5
    "Student-Scores": 45592,  # 56 x 13
    "Swiss-banknote-conterfeit-detection": 43477,  # 200 x 7
    "treepipit": 40505,  # 86 x 10
    "visualizing_hamster": 708,  # 73 x 6
    "WineDataset": 43571,  # 178 x 14
    "wisconsin": 191,  # 194 x 33
    "witmer_census_1980": 668,  # 50 x 5
}


def fetch_openml_datasets(
    datasets: dict[str, int] = OPENML_DATASETS,
    verbose: bool = True
):
    """
    Fetch the benchmark's numeric OpenML datasets.

    Returns a list of (dataframe, name, did) tuples with float32 columns.
    """
    collected = []
    for name, did in datasets.items():
        dataset = openml.datasets.get_dataset(did, download_data=True)
        if dataset.name != name:
            raise ValueError(f"OpenML dataset {did} is named {dataset.name!r}, expected {name!r}")
        df, _, _, _ = dataset.get_data(dataset_format="dataframe")

        non_numeric = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            raise ValueError(f"{name} (ID: {did}) has non-numeric columns: {non_numeric}")

        df = df.astype(np.float32)
        collected.append((df, name, did))
        if verbose:
            print(f"✅ Loaded: {name} (ID: {did}, shape: {df.shape})")

    return collected


if __name__ == "__main__":
    datasets = fetch_openml_datasets(verbose=True)

    for df, name, did in datasets:
        os.makedirs(f"datasets/openml/{name}", exist_ok=True)
        df.to_pickle(f"datasets/openml/{name}/dataset.pkl")
