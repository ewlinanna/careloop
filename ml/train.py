"""Train LogisticRegression + LightGBM + CatBoost, average, isotonic-calibrate, save bundle + report.
Patient-grouped 60/20/20 split (train / calibration / test) prevents leakage across encounters."""
import os, pathlib
import joblib, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from ml.features import build_preprocessor, clean, engineer

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(os.getenv("DATA_PATH", ROOT / "data/raw/diabetic_data.csv"))


def metrics(y, p) -> dict:
    k = max(1, len(y) // 10)
    top = np.argsort(-p)[:k]
    pk = float(y[top].mean())
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "precision@top10%": pk, "lift@top10%": pk / float(y.mean())}


def main() -> None:
    df = clean(pd.read_csv(DATA))
    X, y, g = engineer(df), df["target"].to_numpy(), df["patient_nbr"].to_numpy()
    a, rest = next(GroupShuffleSplit(1, train_size=0.6, random_state=42).split(X, y, g))
    b, c = next(GroupShuffleSplit(1, train_size=0.5, random_state=42).split(X.iloc[rest], y[rest], g[rest]))
    cal, te = rest[b], rest[c]
    pre = build_preprocessor().fit(X.iloc[a])
    T = {n: pre.transform(X.iloc[i]) for n, i in [("tr", a), ("cal", cal), ("te", te)]}
    pos = (1 - y[a].mean()) / y[a].mean()
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"),
        "lightgbm": LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, scale_pos_weight=pos ** 0.5, verbose=-1, random_state=42),
        "catboost": CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, verbose=0, random_seed=42,
                                       auto_class_weights="SqrtBalanced"),
    }
    for m in models.values(): m.fit(T["tr"], y[a])
    raw = lambda t: np.mean([m.predict_proba(t)[:, 1] for m in models.values()], axis=0)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw(T["cal"]), y[cal])
    yt, rows = y[te], {}
    for n, m in models.items(): rows[n] = metrics(yt, m.predict_proba(T["te"])[:, 1])
    rows["ensemble_uncalibrated"] = metrics(yt, raw(T["te"]))
    pt = iso.predict(raw(T["te"]))
    rows["ensemble_calibrated"] = metrics(yt, pt)
    tiers = {"medium": float(np.quantile(pt, 0.70)), "high": float(np.quantile(pt, 0.90))}
    (ROOT / "models").mkdir(exist_ok=True); (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump({"pre": pre, "models": models, "iso": iso, "tiers": tiers, "base_rate": float(y.mean())}, ROOT / "models/bundle.joblib")
    lines = ["# CareLoop evaluation (patient-grouped test split)", f"Test encounters: {len(te)}, positive rate {yt.mean():.3f}", "",
             "| model | ROC-AUC | PR-AUC | Brier | P@top10% | Lift@top10% |", "|---|---|---|---|---|---|"]
    lines += [f"| {n} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['brier']:.4f} | {r['precision@top10%']:.3f} | {r['lift@top10%']:.2f} |" for n, r in rows.items()]
    lines += ["", "ROC-AUC around 0.65-0.70 is realistic for this dataset. Accuracy is not reported (imbalanced target)."]
    (ROOT / "reports/evaluation.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
