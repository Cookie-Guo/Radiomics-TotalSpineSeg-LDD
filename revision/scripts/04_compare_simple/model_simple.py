#!/usr/bin/env python3
"""
CatBoost on the simple-measurement table, same split and hyperparameters as primary.
Writes revision/results/04_compare_simple/performance_vs_radiomics.csv.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
B1 = ROOT / "results" / "04_compare_simple"
PRIMARY = ROOT / "results" / "02_primary"
SPLITS = ROOT / "splits" / "assignments.csv"
FEAT_SIMPLE = B1 / "features.csv"
EXTRACTED = ROOT / "data" / "extracted_data.xlsx"
LABELS = ROOT / "data" / "labels.csv"
SEL3D = PRIMARY / "selected_features_3d.csv"

SEED = 4321
N_BOOT = 1000
CB_PARAMS = dict(
    depth=2,
    learning_rate=0.05,
    l2_leaf_reg=1,
    iterations=223,
    loss_function="MultiClass",
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
)

SIMPLE_CORE = [
    "dhi",
    "delta_peak_si_norm",
    "disc_csf_mean_ratio",
    "area_mm2",
    "sphericity_2d",
]
SIMPLE_PLUS = SIMPLE_CORE + ["disc_height_mm", "peak_si_disc", "delta_peak_si"]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def macro_auc_ovr(y_true: np.ndarray, proba: np.ndarray) -> float:
    yb = label_binarize(y_true, classes=[1, 2, 3, 4, 5])
    aucs = []
    for i in range(5):
        if yb[:, i].sum() in (0, len(yb)):
            continue
        aucs.append(roc_auc_score(yb[:, i], proba[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def metrics_block(y_true, y_pred, proba) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    recalls, specs = [], []
    for i in range(5):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
        specs.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return {
        "macro_AUC": macro_auc_ovr(y_true, proba),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_sensitivity": float(np.mean(recalls)),
        "macro_specificity": float(np.mean(specs)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def bootstrap_ci(y_true, y_pred, proba, n_boot=N_BOOT, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    keys = list(metrics_block(y_true, y_pred, proba).keys())
    store = {k: [] for k in keys}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = metrics_block(y_true[idx], y_pred[idx], proba[idx])
        for k in keys:
            if np.isfinite(m[k]):
                store[k].append(m[k])
    out = {}
    for k in keys:
        arr = np.asarray(store[k], dtype=float)
        out[f"{k}_ci_low"] = float(np.percentile(arr, 2.5)) if len(arr) else float("nan")
        out[f"{k}_ci_high"] = float(np.percentile(arr, 97.5)) if len(arr) else float("nan")
    return out


def class_auc_rows(y_true, proba, config: str) -> list[dict]:
    yb = label_binarize(y_true, classes=[1, 2, 3, 4, 5])
    rows = []
    for i, g in enumerate([1, 2, 3, 4, 5]):
        n_pos = int(yb[:, i].sum())
        auc = float("nan")
        if 0 < n_pos < len(y_true):
            auc = float(roc_auc_score(yb[:, i], proba[:, i]))
        rows.append({
            "config": config, "class": g, "AUC": auc,
            "n_test_pos": n_pos, "n_test_total": len(y_true),
        })
    return rows


def fit_eval(df: pd.DataFrame, feat_cols: list[str], config: str) -> tuple[dict, list[dict], pd.DataFrame]:
    use = [c for c in feat_cols if c in df.columns]
    train = df[df["holdout"] == "train"].copy()
    test = df[df["holdout"] == "test"].copy()
    Xtr = train[use].apply(pd.to_numeric, errors="coerce")
    Xte = test[use].apply(pd.to_numeric, errors="coerce")
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    # drop all-nan cols
    keep = [c for c in use if Xtr[c].notna().any() and float(Xtr[c].std() or 0) > 1e-12]
    Xtr, Xte = Xtr[keep], Xte[keep]
    ytr = train["pfirrmann"].astype(int).values
    yte = test["pfirrmann"].astype(int).values

    bc = np.bincount(ytr, minlength=6)[1:6].astype(float)
    w = (bc.sum() / (5.0 * np.maximum(bc, 1))).tolist()
    class_weights = {i + 1: w[i] for i in range(5)}

    clf = CatBoostClassifier(**CB_PARAMS, class_weights=class_weights)
    clf.fit(Xtr, ytr)
    proba_raw = clf.predict_proba(Xte)
    proba = np.zeros((len(yte), 5))
    for j, c in enumerate(clf.classes_):
        proba[:, int(c) - 1] = proba_raw[:, j]
    pred = proba.argmax(axis=1) + 1

    m = metrics_block(yte, pred, proba)
    m.update(bootstrap_ci(yte, pred, proba))
    m.update({
        "config": config,
        "n_features": len(keep),
        "features": ";".join(keep),
        "n_train": len(train),
        "n_test": len(test),
    })
    # feature importance
    imp = pd.DataFrame({
        "feature": keep,
        "importance": clf.get_feature_importance(),
        "config": config,
    }).sort_values("importance", ascending=False)
    cauc = class_auc_rows(yte, proba, config)
    pred_df = pd.DataFrame({
        "disc_id": test["disc_id"].values,
        "patient_id": test["patient_id"].values,
        "y_true": yte,
        "y_pred": pred,
        "config": config,
        **{f"prob_{g}": proba[:, g - 1] for g in range(1, 6)},
    })
    return m, cauc, imp, pred_df


def load_simple(assign: pd.DataFrame) -> pd.DataFrame:
    s = pd.read_csv(FEAT_SIMPLE)
    if "error" in s.columns:
        s = s[s["error"].fillna("").astype(str).eq("")].copy()
    drop = [c for c in ("patient_id", "level", "pfirrmann") if c in s.columns]
    m = assign.merge(s.drop(columns=drop, errors="ignore"), on="disc_id", how="inner")
    return m


def load_radiomics_plus_simple(assign: pd.DataFrame, simple: pd.DataFrame) -> pd.DataFrame:
    ex = pd.read_excel(EXTRACTED)
    labels = pd.read_csv(LABELS)
    ex = ex.copy()
    ex["disc_id"] = labels["disc_id"].values
    sel = pd.read_csv(SEL3D)["Feature"].tolist()
    # keep only existing
    sel = [c for c in sel if c in ex.columns]
    cols = ["disc_id"] + sel
    r = ex[cols]
    # simple core cols
    scols = ["disc_id"] + [c for c in SIMPLE_CORE if c in simple.columns]
    sm = simple[scols].drop_duplicates("disc_id")
    m = assign.merge(r, on="disc_id", how="inner").merge(sm, on="disc_id", how="inner")
    return m, sel


def main() -> None:
    B1.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    assign = pd.read_csv(SPLITS)
    simple = load_simple(assign)
    print(f"simple rows after merge: {len(simple)} (expect 630)")

    results = []
    caucs = []
    imps = []
    preds = []

    print("simple_core …")
    m, c, imp, pr = fit_eval(simple, SIMPLE_CORE, "simple_core")
    results.append(m); caucs.extend(c); imps.append(imp); preds.append(pr)
    print(f"  macro AUC {m['macro_AUC']:.3f}  n_feat={m['n_features']}")

    print("simple_plus …")
    m2, c2, imp2, pr2 = fit_eval(simple, SIMPLE_PLUS, "simple_plus")
    results.append(m2); caucs.extend(c2); imps.append(imp2); preds.append(pr2)
    print(f"  macro AUC {m2['macro_AUC']:.3f}  n_feat={m2['n_features']}")

    # joint with primary 517 features
    if SEL3D.exists() and EXTRACTED.exists():
        print("radiomics3d_plus_simple …")
        joint, sel = load_radiomics_plus_simple(assign, simple)
        print(f"  joint rows {len(joint)}, rad feats {len(sel)}")
        m3, c3, imp3, pr3 = fit_eval(
            joint, sel + [c for c in SIMPLE_CORE if c in joint.columns],
            "radiomics3d_plus_simple",
        )
        results.append(m3); caucs.extend(c3); imps.append(imp3); preds.append(pr3)
        print(f"  macro AUC {m3['macro_AUC']:.3f}  n_feat={m3['n_features']}")

    # attach primary reference row from file (not re-run)
    pref = PRIMARY / "primary_performance.csv"
    if pref.exists():
        p = pd.read_csv(pref)
        row = p[p["config"] == "3D_primary"].iloc[0].to_dict()
        results.append({
            "config": "3D_primary_ref",
            "macro_AUC": row.get("macro_AUC"),
            "macro_AUC_ci_low": row.get("macro_AUC_ci_low"),
            "macro_AUC_ci_high": row.get("macro_AUC_ci_high"),
            "accuracy": row.get("accuracy"),
            "macro_sensitivity": row.get("macro_sensitivity"),
            "macro_specificity": row.get("macro_specificity"),
            "kappa": row.get("kappa"),
            "kappa_quadratic": row.get("kappa_quadratic"),
            "n_features": row.get("n_features_selected"),
            "n_train": row.get("n_train"),
            "n_test": row.get("n_test"),
            "features": "02_primary_517",
        })

    perf = pd.DataFrame(results)
    perf.to_csv(B1 / "performance_vs_radiomics.csv", index=False)
    pd.DataFrame(caucs).to_csv(B1 / "class_auc.csv", index=False)
    pd.concat(imps, ignore_index=True).to_csv(B1 / "feature_importance.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(B1 / "test_predictions.csv", index=False)

    # univariate Spearman with grade (train+test descriptive)
    from scipy import stats
    uni = []
    for c in SIMPLE_PLUS:
        if c not in simple.columns:
            continue
        x = pd.to_numeric(simple[c], errors="coerce")
        y = simple["pfirrmann"].astype(int)
        msk = x.notna()
        if msk.sum() < 30:
            continue
        rho, p = stats.spearmanr(x[msk], y[msk])
        uni.append({"feature": c, "spearman_rho_vs_grade": float(rho), "p": float(p), "n": int(msk.sum())})
    pd.DataFrame(uni).to_csv(B1 / "univariate_spearman.csv", index=False)

    meta = {
        "script": "model_simple.py",
        "seed": SEED,
        "cb_params": CB_PARAMS,
        "simple_core": SIMPLE_CORE,
        "n_simple_rows": len(simple),
        "elapsed_sec": round(time.time() - t0, 2),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "inputs_md5": {
            "assignments.csv": md5(SPLITS),
            "features.csv": md5(FEAT_SIMPLE) if FEAT_SIMPLE.exists() else None,
        },
        "performance_summary": {
            r["config"]: r.get("macro_AUC") for r in results
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (B1 / "performance_vs_radiomics.meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    print(perf[["config", "macro_AUC", "accuracy", "kappa_quadratic", "n_features"]].to_string(index=False))
    print(f"Done in {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
