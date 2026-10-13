"""
Benchmark datasets.

OpenML (MissBench): a fixed, hard-coded list of OpenML dataset IDs so that anyone
can re-run the exact same benchmark. These are the 42 datasets of the TabImpute paper:
the 41 listed in its OpenML datasets table (Table 9) plus divorce_prediction, which was
benchmarked but left out of that table. They were originally selected from
OpenML with these criteria: no missing values, 50-250 rows, 5-60 features,
all-numeric features, and excluding the synthetic "fri_*" datasets.

UCI: the datasets used in the HyperImpute paper.

Every dataset is saved as <data_dir>/<suite>/<name>/dataset.pkl (a float32 DataFrame).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib import request

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OpenMLDataset:
    did: int
    rows: int
    cols: int
    domain: str
    description: str


# name -> OpenML dataset, in the order of paper Table 9 (divorce_prediction last)
OPENML_DATASETS: dict[str, OpenMLDataset] = {
    "EgyptianSkulls": OpenMLDataset(1099, 150, 5, "Anthropology", "Cranial measurements over time in Egypt"),
    "humans_numeric": OpenMLDataset(1058, 75, 15, "Biology", "Human body measurements"),
    "FacultySalaries": OpenMLDataset(1096, 50, 5, "Education/Economics", "University faculty salary data"),
    "SMSA": OpenMLDataset(1091, 59, 16, "Demographics/Economics", "U.S. metropolitan statistical area data"),
    "Student-Scores": OpenMLDataset(45592, 56, 13, "Education", "Student exam scores"),
    "analcatdata_election2000": OpenMLDataset(527, 67, 15, "Political science", "2000 U.S. presidential election results"),
    "analcatdata_gviolence": OpenMLDataset(518, 74, 9, "Criminology", "Gun violence statistics"),
    "analcatdata_olympic2000": OpenMLDataset(530, 66, 12, "Sports/Economics", "Olympic results and country stats"),
    "baskball": OpenMLDataset(214, 96, 5, "Sports analytics", "Basketball performance data"),
    "visualizing_hamster": OpenMLDataset(708, 73, 6, "Education/Toy", "Example dataset for teaching"),
    "witmer_census_1980": OpenMLDataset(668, 50, 5, "Demographics", "U.S. census microdata (1980)"),
    "MercuryinBass": OpenMLDataset(1090, 53, 10, "Environmental chemistry", "Mercury concentrations in fish"),
    "SolarPower": OpenMLDataset(42328, 204, 5, "Energy/Engineering", "Solar power output records"),
    "WineDataset": OpenMLDataset(43571, 178, 14, "Chemistry/Oenology", "Wine physicochemical properties"),
    "alcohol-qcm-sensor": OpenMLDataset(42930, 125, 15, "Analytical chemistry", "Alcohol detection sensor readings"),
    "benzo32": OpenMLDataset(434, 195, 33, "Chemistry/Toxicology", "Benzodiazepine compound data"),
    "machine_cpu": OpenMLDataset(230, 209, 7, "Computer systems", "Predicting CPU performance"),
    "pwLinear": OpenMLDataset(229, 200, 11, "Mathematics/Engineering", "Piecewise linear regression benchmark"),
    "pyrim": OpenMLDataset(217, 74, 28, "Chemistry/Pharmacology", "Pyrimethamine bioassay compounds"),
    "slump": OpenMLDataset(41558, 103, 10, "Civil engineering", "Concrete slump test properties"),
    "ICU": OpenMLDataset(1097, 200, 20, "Medicine", "Intensive care patient data"),
    "appendicitis_test": OpenMLDataset(45589, 106, 8, "Medicine", "Appendicitis diagnosis"),
    "appendicitis_test_edsa": OpenMLDataset(45591, 106, 8, "Medicine", "Educational appendicitis dataset"),
    "breast-cancer-coimbra": OpenMLDataset(42900, 116, 10, "Medicine", "Breast cancer diagnosis data"),
    "Rainfall-in-Kerala-1901-2017": OpenMLDataset(43806, 117, 18, "Climate science", "Rainfall time series in Kerala"),
    "pollution": OpenMLDataset(542, 60, 16, "Environmental science", "Air pollution measurements"),
    "treepipit": OpenMLDataset(40505, 86, 10, "Ecology", "Bird habitat distribution"),
    "autoPrice": OpenMLDataset(207, 159, 16, "Business/Economics", "Automobile pricing dataset"),
    "dataset_analcatdata_creditscore": OpenMLDataset(46534, 100, 7, "Finance", "Credit scoring dataset"),
    "Swiss-banknote-conterfeit-detection": OpenMLDataset(43477, 200, 7, "Finance/Fraud", "Banknote authenticity classification"),
    "Glass-Classification": OpenMLDataset(43750, 214, 10, "Forensics/Materials", "Glass chemical composition (forensics)"),
    "chatfield_4": OpenMLDataset(695, 235, 13, "Statistics/Time series", "Textbook time series data (Chatfield)"),
    "chscase_vine1": OpenMLDataset(691, 52, 10, "Agriculture/Statistics", "Vine growth study"),
    "edm": OpenMLDataset(41552, 154, 18, "Education", "Student learning performance"),
    "metafeatures": OpenMLDataset(46307, 75, 32, "Meta-learning", "Dataset-level features"),
    "rabe_131": OpenMLDataset(692, 50, 6, "Chemistry/Benchmark", "Spectroscopy regression dataset"),
    "rabe_148": OpenMLDataset(710, 66, 6, "Chemistry/Benchmark", "Spectroscopy regression dataset"),
    "rabe_265": OpenMLDataset(660, 51, 7, "Chemistry/Benchmark", "Spectroscopy regression dataset"),
    "sleuth_case1201": OpenMLDataset(707, 50, 7, "Statistics/Education", "Applied regression textbook data"),
    "sleuth_ex1605": OpenMLDataset(687, 62, 6, "Statistics/Education", "Applied regression textbook data"),
    "wisconsin": OpenMLDataset(191, 194, 33, "Medicine", "Wisconsin breast cancer dataset"),
    "divorce_prediction": OpenMLDataset(46756, 170, 55, "Psychology", "Divorce predictors scale responses"),
}


# UCI datasets used in the HyperImpute paper
UCI_DATASETS: dict[str, dict] = {
    "airfoil_self_noise": {"type": "ucimlrepo", "id": 291},
    "blood_transfusion": {"type": "ucimlrepo", "id": 146},
    "california_housing": {"type": "sklearn", "loader": "fetch_california_housing"},
    "concrete_compression": {"type": "ucimlrepo", "id": 165},
    "diabetes": {"type": "sklearn", "loader": "load_diabetes"},
    "ionosphere": {"type": "ucimlrepo", "id": 52},
    "iris": {"type": "sklearn", "loader": "load_iris"},
    "letter_recognition": {"type": "ucimlrepo", "id": 59},
    "libras_movement": {
        "type": "url",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data",
    },
    "spam_base": {"type": "ucimlrepo", "id": 94},
    # Wine Quality (ID 186) has red and white wines together; split by the "color" column
    "wine_quality_red": {"type": "ucimlrepo", "id": 186, "color": "red"},
    "wine_quality_white": {"type": "ucimlrepo", "id": 186, "color": "white"},
}

SUITE_DATASETS = {"openml": OPENML_DATASETS, "uci": UCI_DATASETS}


def dataset_path(data_dir: Path, suite: str, name: str) -> Path:
    return Path(data_dir) / suite / name / "dataset.pkl"


def load_dataset(data_dir: Path, suite: str, name: str) -> np.ndarray:
    return pd.read_pickle(dataset_path(data_dir, suite, name)).to_numpy().astype(np.float32)


def fetch_openml_dataset(name: str, info: OpenMLDataset) -> pd.DataFrame:
    import openml

    dataset = openml.datasets.get_dataset(info.did, download_data=True)
    if dataset.name != name:
        raise ValueError(f"OpenML dataset {info.did} is named {dataset.name!r}, expected {name!r}")
    df, _, _, _ = dataset.get_data(dataset_format="dataframe")
    return df.astype(np.float32)


def fetch_uci_dataset(name: str, info: dict) -> pd.DataFrame:
    """Fetch a UCI dataset's feature columns (targets are dropped, except for the
    sklearn datasets, which keep their target column as in the original benchmark)."""
    if info["type"] == "sklearn":
        import sklearn.datasets

        data = getattr(sklearn.datasets, info["loader"])(as_frame=True)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
    elif info["type"] == "url":
        with request.urlopen(info["url"]) as f:
            df = pd.read_csv(f, header=None)
        # The last column is the class label
        df = df.iloc[:, :-1]
    else:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=info["id"])
        features = dataset.data.features.columns.tolist()
        if "color" in info:
            original = dataset.data.original
            df = original.loc[original["color"] == info["color"], features]
        else:
            df = dataset.data.features
        df = df.apply(pd.to_numeric, errors="coerce")
    return df.reset_index(drop=True).astype(np.float32)


def download(data_dir: Path, suite: str, names: list[str], overwrite: bool = False) -> None:
    fetch = fetch_openml_dataset if suite == "openml" else fetch_uci_dataset
    datasets = SUITE_DATASETS[suite]
    for name in names:
        path = dataset_path(data_dir, suite, name)
        if path.exists() and not overwrite:
            print(f"  {name}: already downloaded")
            continue
        df = fetch(name, datasets[name])
        if df.isna().any().any():
            raise ValueError(f"{name} has missing or non-numeric values after download")
        os.makedirs(path.parent, exist_ok=True)
        df.to_pickle(path)
        print(f"  {name}: saved {df.shape[0]} x {df.shape[1]} to {path}")
