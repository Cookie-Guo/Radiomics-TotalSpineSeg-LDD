#!/usr/bin/env python3
"""
Patient-level CatBoost primary for 3D and 2D radiomics on the same split.
Train-only reduction (Kruskal–Wallis + Bonferroni, NZV, |r|>0.9) then CatBoost.
Writes revision/results/02_primary/primary_performance.csv (3D macro AUC 0.936).
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
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
from sklearn.utils import resample

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "02_primary"
B1 = ROOT / "results" / "03_compare_2d"
DATA = ROOT / "data"
SPLITS = ROOT / "splits" / "assignments.csv"
EXTRACTED = DATA / "extracted_data.xlsx"
FEAT_2D = B1 / "features_2d.csv"
LABELS = DATA / "labels.csv"

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

META_COLS = {
    "disc_id", "patient_id", "level", "pfirrmann", "holdout",
    "MASK", "disc_degree", "quality", "id",
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in META_COLS or str(c).startswith("cv_rep"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def select_features_with_log(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple[list[str], dict]:
    """Train-only feature reduction; return selected names and stage counts."""
    cols = list(X_train.columns)
    log = {"n_input": len(cols)}

    # 1) Kruskal-Wallis + Bonferroni
    pvals = []
    for c in cols:
        groups = [X_train.loc[y_train == g, c].dropna().values for g in range(1, 6)]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            pvals.append(1.0)
            continue
        try:
            _, p = stats.kruskal(*groups)
            pvals.append(float(p) if np.isfinite(p) else 1.0)
        except Exception:  # noqa: BLE001
            pvals.append(1.0)
    pvals = np.asarray(pvals)
    thr = 0.05 / max(len(cols), 1)
    keep = [c for c, p in zip(cols, pvals) if p < thr]
    if len(keep) < 10:
        keep = [cols[i] for i in np.argsort(pvals)[: min(100, len(cols))]]
    log["after_kruskal_bonferroni"] = len(keep)
    X = X_train[keep].copy()

    # 2) near-zero variance
    keep = [
        c for c in keep
        if X[c].nunique(dropna=True) > 1 and float(X[c].std(skipna=True) or 0) > 1e-12
    ]
    log["after_nzv"] = len(keep)
    X = X[keep]

    if len(keep) > 1:
        order = X.var().sort_values(ascending=False).index.tolist()
        corr = X[order].corr().abs()
        selected: list[str] = []
        for c in order:
            if not selected:
                selected.append(c)
                continue
            if all(corr.loc[c, s] <= 0.9 or c == s for s in selected):
                selected.append(c)
        keep = selected
    log["after_corr"] = len(keep)
    log["final"] = len(keep)
    return keep, log


def macro_auc_ovr(y_true: np.ndarray, proba: np.ndarray) -> float:
    yb = label_binarize(y_true, classes=[1, 2, 3, 4, 5])
    aucs = []
    for i in range(5):
        if yb[:, i].sum() in (0, len(yb)):
            continue
        aucs.append(roc_auc_score(yb[:, i], proba[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
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
    keys = ["macro_AUC", "accuracy", "macro_sensitivity", "macro_specificity", "kappa", "kappa_quadratic"]
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


def load_3d(assign: pd.DataFrame) -> pd.DataFrame:
    ex = pd.read_excel(EXTRACTED)
    labels = pd.read_csv(LABELS)
    assert len(ex) == len(labels)
    ex = ex.copy()
    ex["disc_id"] = labels["disc_id"].values
    ex["patient_id"] = labels["patient_id"].values
    ex["level"] = labels["level"].values
    ex["pfirrmann"] = labels["pfirrmann"].values
    m = assign[["disc_id", "holdout"]].merge(ex, on="disc_id", how="inner")
    assert len(m) == 630
    return m


def load_2d(assign: pd.DataFrame) -> pd.DataFrame:
    f2 = pd.read_csv(FEAT_2D)
    drop = [c for c in ("patient_id", "level", "pfirrmann") if c in f2.columns]
    return assign[["disc_id", "holdout", "patient_id", "level", "pfirrmann"]].merge(
        f2.drop(columns=drop, errors="ignore"), on="disc_id", how="inner"
    )


def run_arm(df: pd.DataFrame, config: str) -> tuple[dict, list[dict], list[str], dict, pd.DataFrame]:
    train = df[df["holdout"] == "train"].copy()
    test = df[df["holdout"] == "test"].copy()
    fcols = feature_columns(train)
    fcols = [c for c in fcols if train[c].notna().sum() > 10]
    Xtr = train[fcols].apply(pd.to_numeric, errors="coerce")
    Xte = test[fcols].apply(pd.to_numeric, errors="coerce")
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    ytr = train["pfirrmann"].astype(int).values
    yte = test["pfirrmann"].astype(int).values

    selected, flog = select_features_with_log(Xtr, ytr)
    Xtr_s, Xte_s = Xtr[selected], Xte[selected]

    bc = np.bincount(ytr, minlength=6)[1:6].astype(float)
    w = (bc.sum() / (5.0 * np.maximum(bc, 1))).tolist()
    class_weights = {i + 1: w[i] for i in range(5)}

    clf = CatBoostClassifier(**CB_PARAMS, class_weights=class_weights)
    clf.fit(Xtr_s, ytr)
    proba_raw = clf.predict_proba(Xte_s)
    proba = np.zeros((len(yte), 5))
    for j, c in enumerate(clf.classes_):
        proba[:, int(c) - 1] = proba_raw[:, j]
    pred = proba.argmax(axis=1) + 1

    m = metrics_block(yte, pred, proba)
    m.update(bootstrap_ci(yte, pred, proba))
    m.update({
        "config": config,
        "n_features_input": len(fcols),
        "n_features_selected": len(selected),
        "n_train": len(train),
        "n_test": len(test),
        **{f"sel_{k}": v for k, v in flog.items()},
    })
    cauc = class_auc_rows(yte, proba, config)

    pred_df = pd.DataFrame({
        "disc_id": test["disc_id"].values,
        "patient_id": test["patient_id"].values if "patient_id" in test.columns else "",
        "y_true": yte,
        "y_pred": pred,
        **{f"prob_{g}": proba[:, g - 1] for g in range(1, 6)},
    })
    return m, cauc, selected, flog, pred_df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    assign = pd.read_csv(SPLITS)

    print("3D …")
    d3 = load_3d(assign)
    m3, c3, sel3, log3, pred3 = run_arm(d3, "3D_primary")
    print("2D …")
    d2 = load_2d(assign)
    assert len(d2) == 630, f"2D rows {len(d2)}"
    m2, c2, sel2, log2, pred2 = run_arm(d2, "2D_midsagittal")

    comp = pd.DataFrame([m3, m2])
    front = [
        "config", "macro_AUC", "macro_AUC_ci_low", "macro_AUC_ci_high",
        "accuracy", "accuracy_ci_low", "accuracy_ci_high",
        "macro_sensitivity", "macro_sensitivity_ci_low", "macro_sensitivity_ci_high",
        "macro_specificity", "macro_specificity_ci_low", "macro_specificity_ci_high",
        "kappa", "kappa_ci_low", "kappa_ci_high",
        "kappa_quadratic", "kappa_quadratic_ci_low", "kappa_quadratic_ci_high",
        "n_features_input", "n_features_selected", "n_train", "n_test",
        "sel_n_input", "sel_after_kruskal_bonferroni", "sel_after_nzv", "sel_after_corr",
    ]
    cols = [c for c in front if c in comp.columns] + [c for c in comp.columns if c not in front]
    comp = comp[cols]
    comp.to_csv(OUT / "primary_performance.csv", index=False)
    B1.mkdir(parents=True, exist_ok=True)
    comp.to_csv(B1 / "performance_2d_vs_3d.csv", index=False)

    pd.concat([pd.DataFrame(c3), pd.DataFrame(c2)], ignore_index=True).to_csv(
        OUT / "class_auc.csv", index=False
    )
    pd.concat([pd.DataFrame(c3), pd.DataFrame(c2)], ignore_index=True).to_csv(
        B1 / "class_auc.csv", index=False
    )

    pd.Series(sel3, name="Feature").to_csv(OUT / "selected_features_3d.csv", index=False)
    pd.Series(sel2, name="Feature").to_csv(OUT / "selected_features_2d.csv", index=False)
    pd.Series(sel3, name="Feature").to_csv(B1 / "selected_features_3d.csv", index=False)
    pd.Series(sel2, name="Feature").to_csv(B1 / "selected_features_2d.csv", index=False)

    pd.DataFrame([
        {"Stage": "input", "N_Features": log3["n_input"]},
        {"Stage": "after_kruskal_bonferroni", "N_Features": log3["after_kruskal_bonferroni"]},
        {"Stage": "after_nzv", "N_Features": log3["after_nzv"]},
        {"Stage": "after_corr_abs_r_gt_0.9", "N_Features": log3["after_corr"]},
    ]).to_csv(OUT / "feature_reduction_log_3d.csv", index=False)
    pd.DataFrame([
        {"Stage": "input", "N_Features": log2["n_input"]},
        {"Stage": "after_kruskal_bonferroni", "N_Features": log2["after_kruskal_bonferroni"]},
        {"Stage": "after_nzv", "N_Features": log2["after_nzv"]},
        {"Stage": "after_corr_abs_r_gt_0.9", "N_Features": log2["after_corr"]},
    ]).to_csv(OUT / "feature_reduction_log_2d.csv", index=False)

    pred3.to_csv(OUT / "test_predictions_3d.csv", index=False)
    pred2.to_csv(OUT / "test_predictions_2d.csv", index=False)

    deprecation = {
        "superseded_primary": {
            "source": "results/patient_level/",
            "catboost_macro_AUC": 0.914320105820106,
            "n_features": 313,
            "reason": "R/caret correlation filter differed from the unified Python pipeline; not used as primary",
        },
        "new_primary": {
            "source": "results/02_primary/",
            "config": "3D_primary",
            "catboost_macro_AUC": m3["macro_AUC"],
            "macro_AUC_CI": [m3["macro_AUC_ci_low"], m3["macro_AUC_ci_high"]],
            "n_features": m3["n_features_selected"],
            "feature_reduction": log3,
        },
        "dimensionality_comparison": {
            "3D": m3["macro_AUC"],
            "2D": m2["macro_AUC"],
            "delta_3d_minus_2d": m3["macro_AUC"] - m2["macro_AUC"],
            "note": "Same reduction and CatBoost as primary; no second 3D AUC",
        },
    }
    (OUT / "supersedes_a1_catboost.json").write_text(
        json.dumps(deprecation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "catboost_params": CB_PARAMS,
        "split": "revision/splits/assignments.csv",
        "seconds": round(time.time() - t0, 1),
        "performance": comp.to_dict(orient="records"),
        "inputs_md5": {
            "assignments.csv": md5(SPLITS),
            "extracted_data.xlsx": md5(EXTRACTED),
            "features_2d.csv": md5(FEAT_2D),
            "labels.csv": md5(LABELS),
        },
        "policy": "Single CatBoost primary = 3D_primary from this script; patient_level 0.914 deprecated for manuscript point estimates",
    }
    (OUT / "primary.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(comp[["config", "macro_AUC", "macro_AUC_ci_low", "macro_AUC_ci_high",
                "n_features_selected", "accuracy", "kappa_quadratic"]].to_string(index=False))
    print(f"\n3D reduction: {log3}")
    print(f"2D reduction: {log2}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
