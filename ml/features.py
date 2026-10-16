"""Shared feature engineering used by BOTH training and serving (no train/serve skew)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MEDS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
        "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
        "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
DROP_DISPOSITION = [11, 13, 14, 19, 20, 21]  # died / hospice
AGE_MID = {f"[{a}-{a + 10})": a + 5 for a in range(0, 100, 10)}
RAW_NUM = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
NUM = RAW_NUM + ["age_num", "prior_visits", "n_meds_active", "n_med_changes", "lab_intensity",
                 "meds_per_day", "emergency_admit", "high_utilizer"]
CAT = ["race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id",
       "diag_1_grp", "diag_2_grp", "diag_3_grp", "max_glu_serum", "A1Cresult", "change",
       "diabetesMed", "medical_specialty"]
# Counterfactuals / what-if may only change these (care-team controllable).
MODIFIABLE = ["num_medications", "time_in_hospital", "num_lab_procedures", "discharge_disposition_id"]


def icd_group(code) -> str:
    """Map an ICD-9 code to a coarse diagnosis category."""
    if code is None or pd.isna(code):
        return "Unknown"
    c = str(code)
    if c.startswith("250"):
        return "Diabetes"
    try:
        v = float(c)
    except ValueError:
        return "Other"
    if 390 <= v < 460 or v == 785: return "Circulatory"
    if 460 <= v < 520 or v == 786: return "Respiratory"
    if 520 <= v < 580 or v == 787: return "Digestive"
    if 580 <= v < 630 or v == 788: return "Genitourinary"
    if 140 <= v < 240: return "Neoplasms"
    if 710 <= v < 740: return "Musculoskeletal"
    if 800 <= v < 1000: return "Injury"
    return "Other"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Training-time cleaning: '?'->NaN, drop sparse cols, drop died/hospice, add target."""
    df = df.replace("?", np.nan).drop(columns=["weight", "payer_code"], errors="ignore")
    df = df[~df["discharge_disposition_id"].isin(DROP_DISPOSITION)].copy()
    df["target"] = (df["readmitted"] == "<30").astype(int)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Raw encounter rows -> model-ready feature frame (NUM + CAT columns)."""
    d = df.copy().replace("?", np.nan)
    for c in MEDS:
        if c not in d: d[c] = "No"
    for c in RAW_NUM:
        if c not in d: d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["age", "diag_1", "diag_2", "diag_3"] + [c for c in CAT if not c.endswith("_grp")]:
        if c not in d: d[c] = np.nan
    d["age_num"] = d["age"].map(AGE_MID)
    d["prior_visits"] = d["number_outpatient"].fillna(0) + d["number_emergency"].fillna(0) + d["number_inpatient"].fillna(0)
    d["n_meds_active"] = (d[MEDS] != "No").sum(axis=1)
    d["n_med_changes"] = d[MEDS].isin(["Up", "Down"]).sum(axis=1)
    d["lab_intensity"] = d["num_lab_procedures"] / d["time_in_hospital"].clip(lower=1)
    d["meds_per_day"] = d["num_medications"] / d["time_in_hospital"].clip(lower=1)
    d["emergency_admit"] = (pd.to_numeric(d["admission_type_id"], errors="coerce") == 1).astype(int)
    d["high_utilizer"] = (d["prior_visits"] >= 3).astype(int)
    for k in (1, 2, 3):
        d[f"diag_{k}_grp"] = d[f"diag_{k}"].map(icd_group)
    for c in CAT:
        d[c] = d[c].astype(str).replace({"nan": "Unknown", "None": "Unknown"})
    return d[NUM + CAT]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False), CAT),
    ])


def original_feature(name: str) -> str:
    """Map a transformed column name back to its source feature."""
    s = name.split("__", 1)[1]
    for c in sorted(CAT, key=len, reverse=True):
        if s.startswith(c + "_"): return c
    return s
