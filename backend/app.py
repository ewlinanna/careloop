"""FastAPI service: /predict, /whatif, /counterfactual + static frontend."""
import pathlib
import joblib, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from ml.features import MODIFIABLE, engineer, original_feature

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "models/bundle.joblib"
if not BUNDLE_PATH.exists():
    raise RuntimeError("Model not trained yet. Run: python -m ml.train")
B = joblib.load(BUNDLE_PATH)
NAMES = B["pre"].get_feature_names_out()
DISCLAIMER = "Decision support only. Not a diagnostic tool; never replaces clinical judgement."

app = FastAPI(title="CareLoop API", version="1.0")


class Patient(BaseModel):
    model_config = ConfigDict(extra="allow")  # extra fields (e.g. individual meds) pass through
    age: str = "[60-70)"
    gender: str = "Female"
    race: str = "Caucasian"
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 7
    time_in_hospital: int = 5
    num_lab_procedures: int = 42
    num_procedures: int = 1
    num_medications: int = 16
    number_outpatient: int = 0
    number_emergency: int = 0
    number_inpatient: int = 1
    number_diagnoses: int = 8
    diag_1: str = "428"
    diag_2: str = "250.01"
    diag_3: str = "401"
    max_glu_serum: str = "None"
    A1Cresult: str = "None"
    change: str = "No"
    diabetesMed: str = "Yes"
    medical_specialty: str = "InternalMedicine"
    insulin: str = "No"


class WhatIf(BaseModel):
    patient: Patient
    changes: dict[str, float]


def score(rec: dict) -> dict:
    X = engineer(pd.DataFrame([rec]))
    Xt = B["pre"].transform(X)
    probs = {k: float(m.predict_proba(Xt)[0, 1]) for k, m in B["models"].items()}
    p = float(B["iso"].predict([np.mean(list(probs.values()))])[0])
    t = B["tiers"]
    tier = "HIGH" if p >= t["high"] else "MEDIUM" if p >= t["medium"] else "LOW"
    contrib = B["models"]["lightgbm"].predict(Xt, pred_contrib=True)[0][:-1]
    agg: dict[str, float] = {}
    for n, v in zip(NAMES, contrib):
        agg[original_feature(n)] = agg.get(original_feature(n), 0.0) + float(v)
    top = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:6]
    return {"risk": p, "tier": tier, "model_probs": probs, "base_rate": B["base_rate"],
            "factors": [{"feature": f, "effect": v, "direction": "raises risk" if v > 0 else "lowers risk"} for f, v in top],
            "disclaimer": DISCLAIMER}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: Patient):
    return score(p.model_dump())


@app.post("/whatif")
def whatif(w: WhatIf):
    bad = [k for k in w.changes if k not in MODIFIABLE]
    if bad:
        raise HTTPException(400, f"Only modifiable features can change: {MODIFIABLE}. Got {bad}")
    before, rec = score(w.patient.model_dump())["risk"], w.patient.model_dump()
    rec.update(w.changes)
    after = score(rec)["risk"]
    return {"before": before, "after": after, "delta": after - before, "note": "Associational, not causal."}


@app.post("/counterfactual")
def counterfactual(p: Patient):
    rec, base = p.model_dump(), score(p.model_dump())["risk"]
    out = []
    for f in ["num_medications", "time_in_hospital", "num_lab_procedures"]:
        best = None
        for step in range(1, 11):
            v = rec[f] - step * (3 if f == "num_lab_procedures" else 1)
            if v < 1: break
            r = score({**rec, f: v})["risk"]
            if r < base - 0.002 and (best is None or r < best["new_risk"]):
                best = {"feature": f, "from": rec[f], "to": v, "old_risk": base, "new_risk": r}
        if best: out.append(best)
    return {"suggestions": sorted(out, key=lambda s: s["new_risk"]), "note": "Associational, not causal."}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="web")
