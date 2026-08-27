#!/usr/bin/env python3
"""
One-feature CatBoost models for each simple measurement on the official split.
Writes revision/results/04_compare_simple/single_feature_performance.csv.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[2]
B1 = ROOT / "results" / "04_compare_simple"
SPLITS = ROOT / "splits" / "assignments.csv"
FEAT = B1 / "features.csv"

SEED = 4321
N_BOOT = 1000
CB = dict(
    depth=2,
    learning_rate=0.05,
    l2_leaf_reg=1,
    iterations=223,
    loss_function="MultiClass",
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
)

CORE = [
    ("dhi", "DHI (disc height index)"),
    ("delta_peak_si_norm", "Normalized peak SI difference (Waldenberg-style)"),
    ("disc_csf_mean_ratio", "Disc/CSF mean SI ratio"),
    ("area_mm2", "Mid-sagittal disc area (mm2)"),
    ("sphericity_2d", "Mid-sagittal 2D sphericity/circularity"),
]


def macro_auc(y, proba):
    yb = label_binarize(y, classes=[1, 2, 3, 4, 5])
    aucs = []
    for i in range(5):
        if yb[:, i].sum() in (0, len(yb)):
            continue
        aucs.append(roc_auc_score(yb[:, i], proba[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def metrics(y, pred, proba):
    cm = confusion_matrix(y, pred, labels=[1, 2, 3, 4, 5])
    rec, spe = [], []
    for i in range(5):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        rec.append(tp / (tp + fn) if tp + fn else 0.0)
        spe.append(tn / (tn + fp) if tn + fp else 0.0)
    return dict(
        macro_AUC=macro_auc(y, proba),
        accuracy=float(accuracy_score(y, pred)),
        macro_sensitivity=float(np.mean(rec)),
        macro_specificity=float(np.mean(spe)),
        kappa=float(cohen_kappa_score(y, pred)),
        kappa_quadratic=float(cohen_kappa_score(y, pred, weights="quadratic")),
    )


def boot_ci(y, pred, proba):
    rng = np.random.default_rng(SEED)
    keys = list(metrics(y, pred, proba).keys())
    store = {k: [] for k in keys}
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = metrics(y[idx], pred[idx], proba[idx])
        for k in keys:
            if np.isfinite(m[k]):
                store[k].append(m[k])
    out = {}
    for k in keys:
        a = np.asarray(store[k], float)
        out[f"{k}_ci_low"] = float(np.percentile(a, 2.5)) if len(a) else float("nan")
        out[f"{k}_ci_high"] = float(np.percentile(a, 97.5)) if len(a) else float("nan")
    return out


def main() -> None:
    t0 = time.time()
    assign = pd.read_csv(SPLITS)
    s = pd.read_csv(FEAT)
    if "error" in s.columns:
        s = s[s["error"].fillna("").astype(str).eq("")].copy()
    drop = [c for c in ("patient_id", "level", "pfirrmann") if c in s.columns]
    df = assign.merge(s.drop(columns=drop, errors="ignore"), on="disc_id", how="inner")

    rows = []
    for col, desc in CORE:
        train = df[df["holdout"] == "train"]
        test = df[df["holdout"] == "test"]
        Xtr = pd.to_numeric(train[col], errors="coerce")
        Xte = pd.to_numeric(test[col], errors="coerce")
        med = Xtr.median()
        Xtr = Xtr.fillna(med).to_numpy().reshape(-1, 1)
        Xte = Xte.fillna(med).to_numpy().reshape(-1, 1)
        ytr = train["pfirrmann"].astype(int).to_numpy()
        yte = test["pfirrmann"].astype(int).to_numpy()

        bc = np.bincount(ytr, minlength=6)[1:6].astype(float)
        w = (bc.sum() / (5.0 * np.maximum(bc, 1))).tolist()
        cw = {i + 1: w[i] for i in range(5)}
        clf = CatBoostClassifier(**CB, class_weights=cw)
        clf.fit(Xtr, ytr)
        pr = clf.predict_proba(Xte)
        proba = np.zeros((len(yte), 5))
        for j, c in enumerate(clf.classes_):
            proba[:, int(c) - 1] = pr[:, j]
        pred = proba.argmax(1) + 1

        m = metrics(yte, pred, proba)
        m.update(boot_ci(yte, pred, proba))
        xall = pd.to_numeric(df[col], errors="coerce")
        msk = xall.notna()
        rho, p = stats.spearmanr(xall[msk], df.loc[msk, "pfirrmann"].astype(int))
        m.update(
            dict(
                config=f"single_{col}",
                feature=col,
                feature_label=desc,
                spearman_rho_vs_grade=float(rho),
                spearman_p=float(p),
                n_features=1,
                n_train=len(train),
                n_test=len(test),
            )
        )
        rows.append(m)
        print(
            f"{col:25s} AUC={m['macro_AUC']:.3f} "
            f"({m['macro_AUC_ci_low']:.3f}-{m['macro_AUC_ci_high']:.3f}) "
            f"Acc={m['accuracy']:.3f} kq={m['kappa_quadratic']:.3f} rho={rho:.3f}"
        )

    out = pd.DataFrame(rows)
    front = [
        "feature", "feature_label", "macro_AUC", "macro_AUC_ci_low", "macro_AUC_ci_high",
        "accuracy", "macro_sensitivity", "macro_specificity", "kappa", "kappa_quadratic",
        "spearman_rho_vs_grade", "spearman_p", "n_features", "n_train", "n_test", "config",
    ]
    cols = [c for c in front if c in out.columns] + [c for c in out.columns if c not in front]
    out = out[cols]
    out.to_csv(B1 / "single_feature_performance.csv", index=False)

    # attach combo + primary ref for one comparison table
    combo_path = B1 / "performance_vs_radiomics.csv"
    ref_rows = []
    if combo_path.exists():
        combo = pd.read_csv(combo_path)
        for cfg in ("simple_core", "3D_primary_ref"):
            sub = combo[combo["config"] == cfg]
            if len(sub):
                r = sub.iloc[0].to_dict()
                ref_rows.append({
                    "feature": cfg,
                    "feature_label": "All 5 simple combined" if cfg == "simple_core" else "3D radiomics primary",
                    "macro_AUC": r.get("macro_AUC"),
                    "macro_AUC_ci_low": r.get("macro_AUC_ci_low"),
                    "macro_AUC_ci_high": r.get("macro_AUC_ci_high"),
                    "accuracy": r.get("accuracy"),
                    "kappa_quadratic": r.get("kappa_quadratic"),
                    "n_features": r.get("n_features"),
                    "config": cfg,
                })
    summary = pd.concat([out, pd.DataFrame(ref_rows)], ignore_index=True, sort=False)
    summary.to_csv(B1 / "head_to_head.csv", index=False)

    meta = {
        "script": "single_feature_models.py",
        "seed": SEED,
        "n_boot": N_BOOT,
        "elapsed_sec": round(time.time() - t0, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": "Each of 5 simple features modeled alone (CatBoost); not only the 5-feature joint model.",
    }
    (B1 / "single_feature_performance.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(out[["feature", "macro_AUC", "accuracy", "kappa_quadratic", "spearman_rho_vs_grade"]].to_string(index=False))
    print(f"Done {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
