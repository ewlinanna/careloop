# CareLoop – 30-day readmission risk (3 ML models)

Models: Logistic Regression + LightGBM + CatBoost → averaged → isotonic calibration.
Patient-grouped 60/20/20 split (no leakage). SHAP-style factors come from LightGBM `pred_contrib`.
Shared feature code (`ml/features.py`) is used by training and serving.

## Run in VS Code (Python 3.10+)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py        # or put diabetic_data.csv in data/raw/
python -m ml.train                     # trains, prints metrics, writes reports/evaluation.md
python -m uvicorn backend.app:app --reload
```
Open http://localhost:8000 (UI) or http://localhost:8000/docs (API).
In VS Code you can also use Run and Debug → "Train models" / "CareLoop API".

## Endpoints
`POST /predict`, `POST /whatif` (modifiable features only), `POST /counterfactual`, `GET /health`

## Honest limits
UCI data is 1999–2008, diabetic patients only. Expect ROC-AUC ≈ 0.65–0.70. Counterfactuals are associational, not causal.
Decision support only; never replaces clinicians.

## Next (from the full spec)
Optuna tuning, conformal intervals, survival proxy, fairness (Fairlearn), drift (Evidently), auth/DB, React UI.
