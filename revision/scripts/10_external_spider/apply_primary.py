#!/usr/bin/env python3
"""
Apply the frozen 517-feature CatBoost to SPIDER expert-mask features.
Does not re-select features on SPIDER; writes results/10_external_spider/performance.csv.
"""

from __future__ import annotations

import json
import platform
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
OUT = ROOT / "results" / "10_external_spider"
DATA = ROOT / "data"
SPLITS = ROOT / "splits" / "assignments.csv"
EXTRACTED = DATA / "extracted_data.xlsx"
LABELS = DATA / "labels.csv"
PRIMARY = ROOT / "results" / "02_primary"
SEL3D = PRIMARY / "selected_features_3d.csv"
PRIMARY_PERF = PRIMARY / "primary_performance.csv"
FEAT = OUT / "features.csv"
INV = OUT / "inventory.csv"

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
META = {
    "disc_id", "patient_id", "mapped_level", "pfirrmann", "manufacturer",
    "field_T", "mask_label_used", "n_voxels", "holdout", "level",
}


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


def class_weights_from_y(ytr: np.ndarray) -> dict:
    bc = np.bincount(ytr, minlength=6)[1:6].astype(float)
    w = (bc.sum() / (5.0 * np.maximum(bc, 1))).tolist()
    return {i + 1: w[i] for i in range(5)}


def align_proba(clf, pr: np.ndarray, n: int) -> np.ndarray:
    proba = np.zeros((n, 5))
    for j, c in enumerate(clf.classes_):
        proba[:, int(c) - 1] = pr[:, j]
    return proba


def fit_apply(Xtr, ytr, Xte, yte, class_weights):
    clf = CatBoostClassifier(**CB_PARAMS, class_weights=class_weights)
    clf.fit(Xtr, ytr)
    pr = clf.predict_proba(Xte)
    proba = align_proba(clf, pr, len(yte))
    pred = proba.argmax(1) + 1
    m = metrics_block(yte, pred, proba)
    m.update(bootstrap_ci(yte, pred, proba))
    return m, pred, proba, clf


def feature_class(name: str) -> str:
    n = str(name)
    if "_shape_" in n or n.startswith("original_shape_"):
        return "shape"
    for t in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        if f"_{t}_" in n:
            return t
    return "other"


def load_internal():
    assign = pd.read_csv(SPLITS)
    labels = pd.read_csv(LABELS)
    ex = pd.read_excel(EXTRACTED)
    ex = ex.copy()
    ex["disc_id"] = labels["disc_id"].values
    ex["patient_id"] = labels["patient_id"].values
    ex["pfirrmann"] = labels["pfirrmann"].values
    m = assign[["disc_id", "holdout"]].merge(ex, on="disc_id", how="inner")
    assert len(m) == 630
    return m


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SEL3D)["Feature"].tolist()
    internal = load_internal()
    train = internal[internal["holdout"] == "train"].copy()
    test = internal[internal["holdout"] == "test"].copy()
    ytr = train["pfirrmann"].astype(int).to_numpy()
    yte = test["pfirrmann"].astype(int).to_numpy()
    cw = class_weights_from_y(ytr)

    use = [c for c in selected if c in train.columns]
    Xtr = train[use].apply(pd.to_numeric, errors="coerce")
    Xte = test[use].apply(pd.to_numeric, errors="coerce")
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    keep = [c for c in use if float(Xtr[c].std() or 0) > 1e-12]
    Xtr, Xte = Xtr[keep], Xte[keep]

    print("Sanity refit 517 on original holdout …")
    m_int, _, _, clf = fit_apply(Xtr, ytr, Xte, yte, cw)
    print(f"  internal refit macro AUC={m_int['macro_AUC']:.3f}")

    official = None
    if PRIMARY_PERF.exists():
        pref = pd.read_csv(PRIMARY_PERF)
        row = pref[pref["config"] == "3D_primary"]
        if len(row):
            official = {
                "config": "3D_primary_official",
                "macro_AUC": float(row.iloc[0]["macro_AUC"]),
                "macro_AUC_ci_low": float(row.iloc[0]["macro_AUC_ci_low"]),
                "macro_AUC_ci_high": float(row.iloc[0]["macro_AUC_ci_high"]),
                "n_features": 517,
                "n_test": 126,
            }

    spider = pd.read_csv(FEAT)
    ys = spider["pfirrmann"].astype(int).to_numpy()
    present = [c for c in keep if c in spider.columns]
    missing = [c for c in keep if c not in spider.columns]
    Xs = spider.reindex(columns=keep).apply(pd.to_numeric, errors="coerce")
    Xs = Xs.fillna(med.reindex(keep))

    print(f"SPIDER n={len(spider)} features present {len(present)}/{len(keep)} missing {len(missing)}")
    pr = clf.predict_proba(Xs)
    proba = align_proba(clf, pr, len(ys))
    pred = proba.argmax(1) + 1
    m_ext = metrics_block(ys, pred, proba)
    m_ext.update(bootstrap_ci(ys, pred, proba))
    m_ext.update(
        dict(
            config="SPIDER_expert_mask_frozen517",
            n_features=len(keep),
            n_features_present=len(present),
            n_features_missing=len(missing),
            n_test=len(spider),
            n_patients=int(spider["patient_id"].nunique()),
        )
    )
    print(
        f"  SPIDER macro AUC={m_ext['macro_AUC']:.3f} "
        f"({m_ext['macro_AUC_ci_low']:.3f}–{m_ext['macro_AUC_ci_high']:.3f})"
    )

    pred_df = spider[["disc_id", "patient_id", "mapped_level", "pfirrmann", "manufacturer", "field_T"]].copy()
    pred_df["y_pred"] = pred
    for g in range(1, 6):
        pred_df[f"p_{g}"] = proba[:, g - 1]
    pred_df.to_csv(OUT / "test_predictions.csv", index=False)

    # class AUC
    yb = label_binarize(ys, classes=[1, 2, 3, 4, 5])
    class_rows = []
    for i, g in enumerate([1, 2, 3, 4, 5]):
        n_pos = int(yb[:, i].sum())
        auc = float("nan")
        if 0 < n_pos < len(ys):
            auc = float(roc_auc_score(yb[:, i], proba[:, i]))
        class_rows.append({"config": "SPIDER", "class": g, "AUC": auc, "n_pos": n_pos, "n_total": len(ys)})
    pd.DataFrame(class_rows).to_csv(OUT / "class_auc.csv", index=False)

    # strata
    strata = []
    for col, name in (("manufacturer", "mfr"), ("field_T", "field")):
        for val, gdf in pred_df.groupby(col):
            if len(gdf) < 20:
                continue
            idx = gdf.index.to_numpy()
            # align to spider row order
            sl = spider.index.get_indexer(gdf.index)
            # pred_df shares index with spider if we copied from spider
            ys_g = gdf["pfirrmann"].astype(int).to_numpy()
            pr_g = gdf[[f"p_{k}" for k in range(1, 6)]].to_numpy()
            pd_g = gdf["y_pred"].to_numpy()
            mm = metrics_block(ys_g, pd_g, pr_g)
            mm.update(dict(config=f"SPIDER_{name}={val}", n_test=len(gdf), n_patients=int(gdf["patient_id"].nunique())))
            strata.append(mm)

    # family transfer: firstorder / shape subsets of 517
    family_rows = []
    for fam in ("firstorder", "shape"):
        cols = [c for c in keep if feature_class(c) == fam]
        if len(cols) < 3:
            continue
        mf, _, _, _ = fit_apply(Xtr[cols], ytr, Xs[cols], ys, cw)
        mf.update(dict(config=f"SPIDER_{fam}_only", n_features=len(cols), n_test=len(spider)))
        family_rows.append(mf)
        # also internal family for context
        mi, _, _, _ = fit_apply(Xtr[cols], ytr, Xte[cols], yte, cw)
        mi.update(dict(config=f"internal_{fam}_only", n_features=len(cols), n_test=len(test)))
        family_rows.append(mi)

    perf_rows = [
        dict(config="internal_refit_517_sanity", n_features=len(keep), n_test=len(test), **m_int),
        m_ext,
    ]
    if official:
        perf_rows.insert(0, official)
    perf_rows.extend(strata)
    perf_rows.extend(family_rows)
    perf = pd.DataFrame(perf_rows)
    front = ["config", "n_features", "n_test", "macro_AUC", "macro_AUC_ci_low", "macro_AUC_ci_high",
             "accuracy", "macro_sensitivity", "macro_specificity", "kappa", "kappa_quadratic"]
    rest = [c for c in perf.columns if c not in front]
    perf = perf[front + rest]
    perf.to_csv(OUT / "performance.csv", index=False)

    vs = pd.DataFrame([
        {k: official[k] for k in official} if official else {},
        {k: m_ext.get(k) for k in (
            "config", "macro_AUC", "macro_AUC_ci_low", "macro_AUC_ci_high",
            "accuracy", "kappa_quadratic", "n_test", "n_patients",
        )},
    ])
    if official:
        vs.loc[vs["config"] == "SPIDER_expert_mask_frozen517", "delta_vs_primary"] = (
            m_ext["macro_AUC"] - official["macro_AUC"]
        )
    vs.to_csv(OUT / "vs_primary.csv", index=False)

    # feature shift: median |z| of spider vs internal train
    shift_rows = []
    for c in keep:
        a = pd.to_numeric(train[c], errors="coerce")
        b = pd.to_numeric(spider[c], errors="coerce") if c in spider.columns else pd.Series(dtype=float)
        mu, sd = float(a.mean()), float(a.std() or 0)
        if sd < 1e-12 or b.notna().sum() < 10:
            continue
        z = (b.dropna() - mu) / sd
        shift_rows.append({
            "feature": c,
            "feature_class": feature_class(c),
            "train_mean": mu,
            "spider_mean": float(b.mean()),
            "median_abs_z": float(z.abs().median()),
            "mean_z": float(z.mean()),
        })
    shift = pd.DataFrame(shift_rows).sort_values("median_abs_z", ascending=False)
    shift.to_csv(OUT / "feature_shift.csv", index=False)
    by_cls = (
        shift.groupby("feature_class")["median_abs_z"]
        .agg(["median", "mean", "count"])
        .reset_index()
        .sort_values("median", ascending=False)
    )
    by_cls.to_csv(OUT / "feature_shift_by_class.csv", index=False)

    summary = {
        "internal_refit_macro_AUC": m_int["macro_AUC"],
        "official_primary_macro_AUC": None if not official else official["macro_AUC"],
        "spider": {
            "macro_AUC": m_ext["macro_AUC"],
            "macro_AUC_ci": [m_ext["macro_AUC_ci_low"], m_ext["macro_AUC_ci_high"]],
            "accuracy": m_ext["accuracy"],
            "kappa_quadratic": m_ext["kappa_quadratic"],
            "n_discs": int(len(spider)),
            "n_patients": int(spider["patient_id"].nunique()),
            "grade_n": spider["pfirrmann"].value_counts().sort_index().to_dict(),
            "n_features_present": len(present),
            "n_features_missing": len(missing),
            "missing_features_head": missing[:15],
        },
        "delta_spider_minus_primary": (
            None if not official else float(m_ext["macro_AUC"] - official["macro_AUC"])
        ),
        "shift_by_class_median_abs_z": by_cls.to_dict("records"),
        "strata": [
            {"config": s.get("config"), "macro_AUC": s.get("macro_AUC"), "n_test": s.get("n_test")}
            for s in strata
        ],
        "what_this_is": "frozen 517 CatBoost applied to SPIDER expert-mask T2 L3-S1 discs",
        "what_this_is_not": [
            "TotalSpineSeg end-to-end external validation",
            "new primary AUC",
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    meta = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "seed": SEED,
        "seconds": round(time.time() - t0, 1),
        "catboost_params": CB_PARAMS,
    }
    (OUT / "apply_primary.meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    print("D3 done", json.dumps(summary["spider"], indent=2, default=str))


if __name__ == "__main__":
    main()
