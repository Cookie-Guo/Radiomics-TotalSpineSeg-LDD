#!/usr/bin/env python3
"""
CatBoost head on frozen encoder embeddings (ImageNet or RadImageNet).
Same split and hyperparameters as the 3D primary.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
B1 = ROOT / "results" / "05_compare_cnn"
PRIMARY = ROOT / "results" / "02_primary"
SPLITS = ROOT / "splits" / "assignments.csv"
FEAT = B1 / "imagenet_features.csv"

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


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def macro_auc_ovr(y_true, proba):
    yb = label_binarize(y_true, classes=[1, 2, 3, 4, 5])
    aucs = []
    for i in range(5):
        if yb[:, i].sum() in (0, len(yb)):
            continue
        aucs.append(roc_auc_score(yb[:, i], proba[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def metrics_block(y_true, y_pred, proba):
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


def bootstrap_ci(y_true, y_pred, proba):
    rng = np.random.default_rng(SEED)
    keys = list(metrics_block(y_true, y_pred, proba).keys())
    store = {k: [] for k in keys}
    n = len(y_true)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = metrics_block(y_true[idx], y_pred[idx], proba[idx])
        for k in keys:
            if np.isfinite(m[k]):
                store[k].append(m[k])
    out = {}
    for k in keys:
        a = np.asarray(store[k], float)
        out[f"{k}_ci_low"] = float(np.percentile(a, 2.5)) if len(a) else float("nan")
        out[f"{k}_ci_high"] = float(np.percentile(a, 97.5)) if len(a) else float("nan")
    return out


def class_auc_rows(y_true, proba, config):
    yb = label_binarize(y_true, classes=[1, 2, 3, 4, 5])
    rows = []
    for i, g in enumerate([1, 2, 3, 4, 5]):
        n_pos = int(yb[:, i].sum())
        auc = float("nan")
        if 0 < n_pos < len(y_true):
            auc = float(roc_auc_score(yb[:, i], proba[:, i]))
        rows.append({"config": config, "class": g, "AUC": auc, "n_test_pos": n_pos, "n_test_total": len(y_true)})
    return rows


def reduce_features(Xtr: pd.DataFrame, corr_thr: float = 0.95, max_keep: int = 256) -> list[str]:
    cols = list(Xtr.columns)
    # NZV
    keep = [c for c in cols if Xtr[c].nunique(dropna=True) > 1 and float(Xtr[c].std(skipna=True) or 0) > 1e-12]
    X = Xtr[keep]
    if len(keep) <= 1:
        return keep
    order = X.var().sort_values(ascending=False).index.tolist()
    # greedy corr on a sample of columns if huge — 2048 is OK
    # compute corr in chunks if needed; full 2048x2048 is fine (~30MB)
    corr = X[order].corr().abs()
    selected: list[str] = []
    for c in order:
        if len(selected) >= max_keep:
            break
        if not selected:
            selected.append(c)
            continue
        if all(corr.loc[c, s] <= corr_thr or c == s for s in selected):
            selected.append(c)
    return selected


def fit_eval(df: pd.DataFrame, feat_cols: list[str], config: str, do_reduce: bool) -> tuple:
    train = df[df["holdout"] == "train"].copy()
    test = df[df["holdout"] == "test"].copy()
    Xtr = train[feat_cols].apply(pd.to_numeric, errors="coerce")
    Xte = test[feat_cols].apply(pd.to_numeric, errors="coerce")
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)

    if do_reduce:
        selected = reduce_features(Xtr, corr_thr=0.95, max_keep=256)
    else:
        selected = [c for c in feat_cols if float(Xtr[c].std() or 0) > 1e-12]
        # cap at 512 by variance for tractability if needed
        if len(selected) > 512:
            selected = Xtr[selected].var().sort_values(ascending=False).head(512).index.tolist()

    Xtr_s, Xte_s = Xtr[selected], Xte[selected]
    scaler = StandardScaler()
    Xtr_n = scaler.fit_transform(Xtr_s)
    Xte_n = scaler.transform(Xte_s)

    ytr = train["pfirrmann"].astype(int).values
    yte = test["pfirrmann"].astype(int).values
    bc = np.bincount(ytr, minlength=6)[1:6].astype(float)
    w = (bc.sum() / (5.0 * np.maximum(bc, 1))).tolist()
    cw = {i + 1: w[i] for i in range(5)}

    clf = CatBoostClassifier(**CB_PARAMS, class_weights=cw)
    clf.fit(Xtr_n, ytr)
    proba_raw = clf.predict_proba(Xte_n)
    proba = np.zeros((len(yte), 5))
    for j, c in enumerate(clf.classes_):
        proba[:, int(c) - 1] = proba_raw[:, j]
    pred = proba.argmax(1) + 1

    m = metrics_block(yte, pred, proba)
    m.update(bootstrap_ci(yte, pred, proba))
    m.update({
        "config": config,
        "n_features_input": len(feat_cols),
        "n_features_used": len(selected),
        "n_train": len(train),
        "n_test": len(test),
        "reduced": do_reduce,
    })
    cauc = class_auc_rows(yte, proba, config)
    pred_df = pd.DataFrame({
        "disc_id": test["disc_id"].values,
        "patient_id": test["patient_id"].values,
        "y_true": yte,
        "y_pred": pred,
        "config": config,
        **{f"prob_{g}": proba[:, g - 1] for g in range(1, 6)},
    })
    return m, cauc, selected, pred_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(FEAT))
    ap.add_argument("--tag", default="resnet50", help="output prefix: B3_{tag}_vs_radiomics.csv")
    ap.add_argument("--outdir", default="", help="output directory (default: results/05_compare_cnn)")
    args = ap.parse_args()
    feat_path = Path(args.features)
    tag = args.tag.strip() or "resnet50"
    outdir = Path(args.outdir) if args.outdir else B1
    outdir.mkdir(parents=True, exist_ok=True)
    reduced_name = f"vision_{tag}_reduced"
    topvar_name = f"vision_{tag}_topvar512"

    t0 = time.time()
    assign = pd.read_csv(SPLITS)
    feat = pd.read_csv(feat_path)
    if "error" in feat.columns:
        feat = feat[feat["error"].fillna("").astype(str).eq("")].copy()
    fcols = [c for c in feat.columns if c.startswith("resnet50_")]
    assert len(fcols) == 2048, f"expected 2048 feats, got {len(fcols)}"

    drop = [c for c in ("patient_id", "level", "pfirrmann") if c in feat.columns]
    df = assign.merge(feat.drop(columns=drop, errors="ignore"), on="disc_id", how="inner")
    print(f"merged rows={len(df)} features={len(fcols)}")

    results, caucs, preds = [], [], []

    print(f"{reduced_name} …")
    m1, c1, sel1, p1 = fit_eval(df, fcols, reduced_name, do_reduce=True)
    results.append(m1); caucs.extend(c1); preds.append(p1)
    print(f"  AUC={m1['macro_AUC']:.3f} n_used={m1['n_features_used']}")

    print(f"{topvar_name} …")
    m2, c2, sel2, p2 = fit_eval(df, fcols, topvar_name, do_reduce=False)
    results.append(m2); caucs.extend(c2); preds.append(p2)
    print(f"  AUC={m2['macro_AUC']:.3f} n_used={m2['n_features_used']}")

    # primary ref
    pref = PRIMARY / "primary_performance.csv"
    if pref.exists():
        p = pd.read_csv(pref)
        for cfg, label in [("3D_primary", "3D_primary_ref"), ("2D_midsagittal", "2D_midsagittal_ref")]:
            sub = p[p["config"] == cfg]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            results.append({
                "config": label,
                "macro_AUC": r["macro_AUC"],
                "macro_AUC_ci_low": r.get("macro_AUC_ci_low"),
                "macro_AUC_ci_high": r.get("macro_AUC_ci_high"),
                "accuracy": r["accuracy"],
                "macro_sensitivity": r.get("macro_sensitivity"),
                "macro_specificity": r.get("macro_specificity"),
                "kappa": r.get("kappa"),
                "kappa_quadratic": r.get("kappa_quadratic"),
                "n_features_used": r.get("n_features_selected"),
                "n_train": r.get("n_train"),
                "n_test": r.get("n_test"),
            })

    # simple best single ref
    sp = ROOT / "results" / "04_compare_simple" / "single_feature_performance.csv"
    if sp.exists():
        s = pd.read_csv(sp).sort_values("macro_AUC", ascending=False).iloc[0]
        results.append({
            "config": "simple_best_single_ref",
            "macro_AUC": s["macro_AUC"],
            "macro_AUC_ci_low": s.get("macro_AUC_ci_low"),
            "macro_AUC_ci_high": s.get("macro_AUC_ci_high"),
            "accuracy": s["accuracy"],
            "kappa_quadratic": s["kappa_quadratic"],
            "n_features_used": 1,
            "note": s.get("feature"),
        })

    if tag == "resnet50":
        perf_path = B1 / "imagenet_vs_radiomics.csv"
        cauc_path = B1 / "imagenet_class_auc.csv"
        pred_path = B1 / "imagenet_test_predictions.csv"
        sel_path = B1 / "imagenet_selected_features.csv"
        meta_path = B1 / "imagenet_vs_radiomics.meta.json"
    else:
        perf_path = outdir / f"B3_{tag}_vs_radiomics.csv"
        cauc_path = outdir / f"B3_{tag}_class_auc.csv"
        pred_path = outdir / f"B3_{tag}_test_predictions.csv"
        sel_path = outdir / f"B3_{tag}_selected_features_reduced.csv"
        meta_path = outdir / f"B3_{tag}_vs_radiomics.meta.json"

    perf = pd.DataFrame(results)
    perf.to_csv(perf_path, index=False)
    pd.DataFrame(caucs).to_csv(cauc_path, index=False)
    pd.concat(preds, ignore_index=True).to_csv(pred_path, index=False)
    pd.Series(sel1, name="feature").to_csv(sel_path, index=False)

    meta = {
        "script": "model_vision.py",
        "tag": tag,
        "seed": SEED,
        "cb_params": CB_PARAMS,
        "n_rows": len(df),
        "n_resnet_features": len(fcols),
        "elapsed_sec": round(time.time() - t0, 2),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "inputs_md5": {
            "assignments.csv": md5(SPLITS),
            feat_path.name: md5(feat_path),
        },
        "performance_summary": {r["config"]: r.get("macro_AUC") for r in results},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "primary_config_note": f"{reduced_name} is the pre-specified report arm",
        "outputs": {
            "performance": str(perf_path),
            "predictions": str(pred_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    cols = [c for c in ["config", "macro_AUC", "accuracy", "kappa_quadratic", "n_features_used"] if c in perf.columns]
    print(perf[cols].to_string(index=False))
    print(f"Done {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
