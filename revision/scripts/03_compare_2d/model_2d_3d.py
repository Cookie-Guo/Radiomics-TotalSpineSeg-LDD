#!/usr/bin/env python3
"""
Fit the 2D CatBoost comparator with the same split and hyperparameters as 3D.
Writes revision/results/03_compare_2d/performance_2d_vs_3d.csv.
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

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "03_compare_2d"
A1 = ROOT / "results" / "patient_level"
DATA = ROOT / "data"
SPLITS = ROOT / "splits" / "assignments.csv"
EXTRACTED = DATA / "extracted_data.xlsx"
FEAT_2D = RESULTS / "features_2d.csv"
LABELS = DATA / "labels.csv"

SEED = 4321
# A1 C_corrected
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


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


META_COLS = {
    "disc_id", "patient_id", "level", "pfirrmann", "holdout",
    "MASK", "disc_degree", "quality", "id", "cv_rep01",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in META_COLS or c.startswith("cv_rep"):
            continue
        if df[c].dtype.kind in "biufc":
            cols.append(c)
    return cols


def select_features(X_train: pd.DataFrame, y_train: np.ndarray) -> list[str]:
    """KW+Bonferroni then NZV then |r|>0.9 (train only)."""
    cols = list(X_train.columns)
    # Kruskal-Wallis + Bonferroni
    keep = []
    pvals = []
    for c in cols:
        groups = [X_train.loc[y_train == g, c].dropna().values for g in range(1, 6)]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            pvals.append(1.0)
            continue
        try:
            _, p = stats.kruskal(*groups)
        except Exception:  # noqa: BLE001
            p = 1.0
        pvals.append(float(p) if np.isfinite(p) else 1.0)
    pvals = np.array(pvals)
    thr = 0.05 / max(len(cols), 1)
    keep = [c for c, p in zip(cols, pvals) if p < thr]
    if not keep:
        order = np.argsort(pvals)
        keep = [cols[i] for i in order[: min(100, len(cols))]]
    X = X_train[keep].copy()

    # near zero variance
    nunique = X.nunique(dropna=True)
    keep = [c for c in keep if nunique[c] > 1 and X[c].std(skipna=True) > 1e-12]
    X = X[keep]

    # correlation |r|>0.9
    if len(keep) > 1:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if any(upper[c] > 0.9)]
        keep = [c for c in keep if c not in drop]

    return keep


def macro_auc_ovr(y_true: np.ndarray, proba: np.ndarray, classes=None) -> float:
    if classes is None:
        classes = np.arange(1, 6)
    yb = label_binarize(y_true, classes=classes)
    # proba columns must match classes order 1..5
    aucs = []
    for i, _ in enumerate(classes):
        if yb[:, i].sum() == 0 or yb[:, i].sum() == len(yb):
            continue
        aucs.append(roc_auc_score(yb[:, i], proba[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    # per-class recall
    recalls = []
    specs = []
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


def class_auc_table(y_true: np.ndarray, proba: np.ndarray, config: str) -> pd.DataFrame:
    rows = []
    yb = label_binarize(y_true, classes=[1, 2, 3, 4, 5])
    for i, g in enumerate([1, 2, 3, 4, 5]):
        n_pos = int(yb[:, i].sum())
        if n_pos == 0 or n_pos == len(y_true):
            auc = float("nan")
        else:
            auc = float(roc_auc_score(yb[:, i], proba[:, i]))
        rows.append({
            "config": config,
            "class": g,
            "AUC": auc,
            "n_test_pos": n_pos,
            "n_test_total": len(y_true),
        })
    return pd.DataFrame(rows)


def load_3d_matrix(assign: pd.DataFrame) -> pd.DataFrame:
    """Map extracted_data MASK onto disc_id via labels.csv."""
    ex = pd.read_excel(EXTRACTED)
    labels = pd.read_csv(LABELS)
    if len(ex) != len(labels):
        raise AssertionError(f"extracted {len(ex)} vs labels {len(labels)}")
    ex = ex.copy()
    ex["disc_id"] = labels["disc_id"].values
    ex["patient_id"] = labels["patient_id"].values
    ex["level"] = labels["level"].values
    ex["pfirrmann"] = labels["pfirrmann"].values
    m = assign[["disc_id", "holdout"]].merge(ex, on="disc_id", how="inner")
    if len(m) != 630:
        raise AssertionError(f"3D merge got {len(m)}")
    return m


def load_2d_matrix(assign: pd.DataFrame) -> pd.DataFrame:
    f2 = pd.read_csv(FEAT_2D)
    m = assign[["disc_id", "holdout", "patient_id", "level", "pfirrmann"]].merge(
        f2.drop(columns=[c for c in ("patient_id", "level", "pfirrmann") if c in f2.columns], errors="ignore"),
        on="disc_id",
        how="inner",
    )
    return m


def run_arm(df: pd.DataFrame, config: str) -> tuple[dict, pd.DataFrame, list[str]]:
    train = df[df["holdout"] == "train"].copy()
    test = df[df["holdout"] == "test"].copy()
    fcols = feature_columns(train)
    fcols = [c for c in fcols if train[c].notna().sum() > 10]
    Xtr = train[fcols].apply(pd.to_numeric, errors="coerce")
    Xte = test[fcols].apply(pd.to_numeric, errors="coerce")
    med = Xtr.median()
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)
    ytr = train["pfirrmann"].astype(int).values
    yte = test["pfirrmann"].astype(int).values

    selected = select_features(Xtr, ytr)
    Xtr_s, Xte_s = Xtr[selected], Xte[selected]

    bc = np.bincount(ytr, minlength=6)[1:6].astype(float)
    w = (bc.sum() / (5.0 * bc)).tolist()
    class_weights = {i + 1: w[i] for i in range(5)}

    clf = CatBoostClassifier(**CB_PARAMS, class_weights=class_weights)
    clf.fit(Xtr_s, ytr)
    proba = clf.predict_proba(Xte_s)
    order = list(clf.classes_)
    proba_ord = np.zeros((len(yte), 5))
    for j, c in enumerate(order):
        proba_ord[:, int(c) - 1] = proba[:, j]
    pred = proba_ord.argmax(axis=1) + 1

    m = metrics_block(yte, pred, proba_ord)
    m.update({
        "config": config,
        "n_features_input": len(fcols),
        "n_features_selected": len(selected),
        "n_train": len(train),
        "n_test": len(test),
    })
    cauc = class_auc_table(yte, proba_ord, config)
    return m, cauc, selected


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    assign = pd.read_csv(SPLITS)

    print("loading 3D ...")
    d3 = load_3d_matrix(assign)
    print("loading 2D ...")
    if not FEAT_2D.exists():
        raise SystemExit(f"missing {FEAT_2D}; run extract_2d_features.py first")
    d2 = load_2d_matrix(assign)
    print(f"  2D rows={len(d2)} / 630")
    if len(d2) < 600:
        print(f"  [warn] 2D incomplete ({len(d2)}); continuing on the intersection")

    if len(d2) < 630:
        common = set(d2["disc_id"])
        d3 = d3[d3["disc_id"].isin(common)].copy()
        print(f"  aligned 3D/2D rows={len(d3)}")

    m3, c3, sel3 = run_arm(d3, "3D_3mm_existing")
    m2, c2, sel2 = run_arm(d2, "2D_midsagittal")

    a1_row = {
        "config": "3D_A1_official_CatBoost",
        "macro_AUC": 0.914320105820106,
        "accuracy": 0.658730158730159,
        "macro_sensitivity": 0.600952380952381,
        "macro_specificity": 0.907729439106251,
        "kappa": 0.529442417926003,
        "kappa_quadratic": 0.833830275229358,
        "n_features_input": 1762,
        "n_features_selected": 313,
        "n_train": 504,
        "n_test": 126,
        "note": "from patient_level; not re-fit in this script",
    }

    comp = pd.DataFrame([m3, m2, a1_row])
    front = ["config", "macro_AUC", "accuracy", "macro_sensitivity", "macro_specificity",
             "kappa", "kappa_quadratic", "n_features_input", "n_features_selected", "n_train", "n_test"]
    cols = front + [c for c in comp.columns if c not in front]
    comp = comp[cols]
    comp.to_csv(RESULTS / "performance_2d_vs_3d.csv", index=False)

    cauc = pd.concat([c3, c2], ignore_index=True)
    cauc.to_csv(RESULTS / "class_auc.csv", index=False)

    pd.Series(sel3, name="Feature").to_csv(RESULTS / "selected_features_3d.csv", index=False)
    pd.Series(sel2, name="Feature").to_csv(RESULTS / "selected_features_2d.csv", index=False)

    a1_feats = set(pd.read_csv(A1 / "final_selected_features.csv")["Feature"])
    overlap = len(a1_feats & set(sel3))

    meta = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "catboost_params": CB_PARAMS,
        "seed": SEED,
        "seconds": round(time.time() - t0, 1),
        "n_2d_discs": int(len(d2)),
        "n_3d_discs": int(len(d3)),
        "selected_3d_n": len(sel3),
        "selected_2d_n": len(sel2),
        "overlap_3d_selected_with_A1_313": overlap,
        "inputs_md5": {
            "assignments.csv": md5(SPLITS),
            "extracted_data.xlsx": md5(EXTRACTED),
            "features_2d.csv": md5(FEAT_2D) if FEAT_2D.exists() else None,
        },
        "comparison": comp.to_dict(orient="records"),
    }
    (RESULTS / "performance_2d_vs_3d.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(comp.to_string(index=False))
    print(f"\nwrote {RESULTS / 'performance_2d_vs_3d.csv'}")
    print(f"3D selected ∩ A1 313 = {overlap}/{len(sel3)}")


if __name__ == "__main__":
    main()
