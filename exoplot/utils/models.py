"""
Mass-Radius Theoretical Models Loader
-------------------------------------
Provides a unified way to load and retrieve theoretical mass–radius relationship curves.

Author: S.WITTMANN
Repository: https://github.com/SimonWtmn/Exoplot
"""

import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "theoretical_models"

# Model key map: key => filename
MODEL_CATALOG = {
    # --- Core Composition Models ---
    "zeng_rocky"              : ("zeng_2019_pure_rock",        "Zeng+2019: Pure Rock"),
    "zeng_iron"               : ("zeng_2019_pure_iron",        "Zeng+2019: Pure Iron"),
    "zeng_earth"              : ("zeng_2019_earth_like",       "Zeng+2019: Earth-like"),
    "zeng_2016_20fe"          : ("zeng_2016_20_Fe",            "Zeng+2016: 20% Iron"),

    # --- 50% H₂O Planets ---
    "zeng_50h2o_300K"         : ("zeng_2019_50_H2O_300K",       "Zeng+2019: 50% H₂O @ 300K"),
    "zeng_50h2o_500K"         : ("zeng_2019_50_H2O_500K",       "Zeng+2019: 50% H₂O @ 500K"),
    "zeng_50h2o_700K"         : ("zeng_2019_50_H2O_700K",       "Zeng+2019: 50% H₂O @ 700K"),
    "zeng_50h2o_1000K"        : ("zeng_2019_50_H2O_1000K",      "Zeng+2019: 50% H₂O @ 1000K"),

    # --- 100% H₂O Planets ---
    "zeng_100h2o_300K"        : ("zeng_2019_100_H2O_300K",      "Zeng+2019: 100% H₂O @ 300K"),
    "zeng_100h2o_500K"        : ("zeng_2019_100_H2O_500K",      "Zeng+2019: 100% H₂O @ 500K"),
    "zeng_100h2o_700K"        : ("zeng_2019_100_H2O_700K",      "Zeng+2019: 100% H₂O @ 700K"),
    "zeng_100h2o_1000K"       : ("zeng_2019_100_H2O_1000K",     "Zeng+2019: 100% H₂O @ 1000K"),

    # --- 0.1% H₂ on Earth-like Core ---
    "zeng_01h2_300K"          : ("zeng_2019_01_H2_onto_earth_like_300K",  "Zeng+2019: 0.1% H₂ @ 300K"),
    "zeng_01h2_500K"          : ("zeng_2019_01_H2_onto_earth_like_500K",  "Zeng+2019: 0.1% H₂ @ 500K"),
    "zeng_01h2_700K"          : ("zeng_2019_01_H2_onto_earth_like_700K",  "Zeng+2019: 0.1% H₂ @ 700K"),
    "zeng_01h2_1000K"         : ("zeng_2019_01_H2_onto_earth_like_1000K", "Zeng+2019: 0.1% H₂ @ 1000K"),
    "zeng_01h2_2000K"         : ("zeng_2019_01_H2_onto_earth_like_2000K", "Zeng+2019: 0.1% H₂ @ 2000K"),

    # --- 0.3% H₂ on Earth-like Core ---
    "zeng_03h2_300K"          : ("zeng_2019_03_H2_onto_earth_like_300K",  "Zeng+2019: 0.3% H₂ @ 300K"),
    "zeng_03h2_500K"          : ("zeng_2019_03_H2_onto_earth_like_500K",  "Zeng+2019: 0.3% H₂ @ 500K"),
    "zeng_03h2_700K"          : ("zeng_2019_03_H2_onto_earth_like_700K",  "Zeng+2019: 0.3% H₂ @ 700K"),
    "zeng_03h2_1000K"         : ("zeng_2019_03_H2_onto_earth_like_1000K", "Zeng+2019: 0.3% H₂ @ 1000K"),
    "zeng_03h2_2000K"         : ("zeng_2019_03_H2_onto_earth_like_2000K", "Zeng+2019: 0.3% H₂ @ 2000K"),

    # --- 1% H₂ on Earth-like Core ---
    "zeng_1h2_300K"           : ("zeng_2019_1_H2_onto_earth_like_300K",    "Zeng+2019: 1% H₂ @ 300K"),
    "zeng_1h2_500K"           : ("zeng_2019_1_H2_onto_earth_like_500K",    "Zeng+2019: 1% H₂ @ 500K"),
    "zeng_1h2_700K"           : ("zeng_2019_1_H2_onto_earth_like_700K",    "Zeng+2019: 1% H₂ @ 700K"),
    "zeng_1h2_1000K"          : ("zeng_2019_1_H2_onto_earth_like_1000K",   "Zeng+2019: 1% H₂ @ 1000K"),
    "zeng_1h2_2000K"          : ("zeng_2019_1_H2_onto_earth_like_2000K",   "Zeng+2019: 1% H₂ @ 2000K"),

    # --- 2% H₂ on Earth-like Core ---
    "zeng_2h2_300K"           : ("zeng_2019_2_H2_onto_earth_like_300K",    "Zeng+2019: 2% H₂ @ 300K"),
    "zeng_2h2_500K"           : ("zeng_2019_2_H2_onto_earth_like_500K",    "Zeng+2019: 2% H₂ @ 500K"),
    "zeng_2h2_700K"           : ("zeng_2019_2_H2_onto_earth_like_700K",    "Zeng+2019: 2% H₂ @ 700K"),
    "zeng_2h2_1000K"          : ("zeng_2019_2_H2_onto_earth_like_1000K",   "Zeng+2019: 2% H₂ @ 1000K"),
    "zeng_2h2_2000K"          : ("zeng_2019_2_H2_onto_earth_like_2000K",   "Zeng+2019: 2% H₂ @ 2000K"),

    # --- 5% H₂ on Earth-like Core ---
    "zeng_5h2_300K"           : ("zeng_2019_5_H2_onto_earth_like_300K",    "Zeng+2019: 5% H₂ @ 300K"),
    "zeng_5h2_500K"           : ("zeng_2019_5_H2_onto_earth_like_500K",    "Zeng+2019: 5% H₂ @ 500K"),
    "zeng_5h2_700K"           : ("zeng_2019_5_H2_onto_earth_like_700K",    "Zeng+2019: 5% H₂ @ 700K"),
    "zeng_5h2_1000K"          : ("zeng_2019_5_H2_onto_earth_like_1000K",   "Zeng+2019: 5% H₂ @ 1000K"),
    "zeng_5h2_2000K"          : ("zeng_2019_5_H2_onto_earth_like_2000K",   "Zeng+2019: 5% H₂ @ 2000K"),
}



# ------------------------------------- HELPER -------------------------------------

# ------------ Trace ------------  -
def get_model_curve(key: str) -> pd.DataFrame:
    if key not in MODEL_CATALOG:
        raise KeyError(f"Invalid model key '{key}'. Use list_models() to view available options.")

    filename = MODEL_CATALOG[key][0]
    filepath = MODELS_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    df = pd.read_csv(filepath, sep=r'\s+|\t+', header=None, engine='python')
    if df.shape[1] != 2:
        raise ValueError(f"Model file '{filename}' should have exactly 2 columns.")

    df.columns = ['mass', 'radius']
    return df.dropna()


# ----------- Get List ------------
def list_models() -> dict:
    return MODEL_CATALOG


# ----------- Get Label ----------- 
def get_model_label(key: str) -> str:
    return MODEL_CATALOG[key][1] if key in MODEL_CATALOG else key