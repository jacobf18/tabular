"""
Download the categorical (mixed-type) OpenML datasets used in the imputation benchmark.

The benchmark uses a fixed, hard-coded list of OpenML dataset IDs so that anyone
can re-run the exact same benchmark. The list was originally selected from OpenML
with these criteria: no missing values, 50-250 rows, 5-60 features, more than one
symbolic feature, and excluding the synthetic "fri_*" datasets.

Usage (from mcpfn/benchmark):
    python download_openml_categorical.py

Each dataset is saved to datasets/openml_categorical/<name>/dataframe.pkl, which is
what create_openml_categorical_missingness_datasets.py reads.

The numeric benchmark datasets live in download_openml.py.
"""
import os

import numpy as np
import openml

# name -> OpenML dataset ID (shape is rows x columns after dropping object columns)
OPENML_CATEGORICAL_DATASETS = {
    "acute-inflammations": 1556,  # 120 x 7
    "aids": 346,  # 50 x 5
    "analcatdata_creditscore": 46379,  # 100 x 5
    "analcatdata_cyyoung8092": 465,  # 97 x 11
    "analcatdata_cyyoung9302": 479,  # 92 x 10
    "analcatdata_impeach": 501,  # 100 x 10
    "analcatdata_ncaa": 521,  # 120 x 20
    "analcatdata_wildcat": 520,  # 163 x 6
    "auto_price": 745,  # 159 x 16
    "backache": 463,  # 180 x 32
    "blogger": 1463,  # 100 x 6
    "caesarian-section": 42901,  # 80 x 6
    "ChonicKidneyDisease": 43106,  # 250 x 28
    "cloud": 890,  # 108 x 8
    "cm1_req": 1055,  # 89 x 9
    "cocomo_numeric": 1051,  # 60 x 17
    "conference_attendance": 41538,  # 246 x 7
    "corral": 40669,  # 160 x 7
    "cpu": 796,  # 209 x 8
    "Dog_Breeds_Ranked": 46081,  # 87 x 16
    "fl2000": 477,  # 67 x 16
    "flags": 1012,  # 194 x 29
    "fruitfly": 714,  # 125 x 5
    "grub-damage": 1026,  # 155 x 9
    "HappinessRank_2015": 40916,  # 158 x 11
    "hutsof99_logis": 681,  # 70 x 8
    "iris": 42871,  # 150 x 7
    "kidney": 945,  # 76 x 7
    "lowbwt": 941,  # 189 x 10
    "lung": 46162,  # 167 x 10
    "lungcancer_GSE31210": 1412,  # 226 x 24
    "lymph": 1006,  # 148 x 19
    "molecular-biology_promoters": 164,  # 106 x 58
    "mux6": 40681,  # 128 x 7
    "MY_DB": 42689,  # 120 x 7
    "nadeem": 42337,  # 100 x 11
    "nasa_numeric": 1076,  # 93 x 24
    "Online_Sales": 46079,  # 240 x 6
    "Parkinson_Dataset": 46094,  # 195 x 23
    "postoperative-patient-data": 40683,  # 88 x 9
    "prnn_crabs": 446,  # 200 x 8
    "prnn_viruses": 480,  # 61 x 19
    "qualitative-bankruptcy": 1495,  # 250 x 7
    "servo": 46292,  # 167 x 5
    "sleuth_case1202": 891,  # 93 x 7
    "sleuth_case2002": 665,  # 147 x 7
    "sleuth_ex2015": 864,  # 60 x 8
    "sleuth_ex2016": 682,  # 87 x 11
    "tae": 48,  # 151 x 6
    "teachingAssistant": 45688,  # 151 x 6
    "veteran": 46166,  # 137 x 8
    "white-clover": 1009,  # 63 x 32
    "zoo": 62,  # 101 x 17
}


def fetch_openml_categorical_datasets(
    datasets: dict[str, int] = OPENML_CATEGORICAL_DATASETS,
    verbose: bool = True
):
    """
    Fetch the benchmark's categorical OpenML datasets.

    Object columns are dropped and numeric columns are converted to float32;
    categorical columns are kept as pandas categoricals.

    Returns a list of (dataframe, name, did) tuples.
    """
    collected = []
    for name, did in datasets.items():
        dataset = openml.datasets.get_dataset(did, download_data=True)
        if dataset.name != name:
            raise ValueError(f"OpenML dataset {did} is named {dataset.name!r}, expected {name!r}")
        df, _, _, _ = dataset.get_data(dataset_format="dataframe")

        # remove object columns
        df = df.select_dtypes(exclude=['object'])

        # convert numeric columns to float32
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)

        collected.append((df, name, did))
        if verbose:
            print(f"✅ Loaded: {name} (ID: {did}, shape: {df.shape})")

    return collected


if __name__ == "__main__":
    datasets = fetch_openml_categorical_datasets(verbose=True)

    for df, name, did in datasets:
        os.makedirs(f"datasets/openml_categorical/{name}", exist_ok=True)
        df.to_pickle(f"datasets/openml_categorical/{name}/dataframe.pkl")
