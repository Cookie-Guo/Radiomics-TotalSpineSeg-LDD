#!/usr/bin/env python3
"""
Paired bootstrap of comparator models versus 3D radiomics on the test set.
Writes revision/results/06_pairwise/pairwise_auc.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "06_pairwise"
PRIMARY = ROOT / "results" / "02_primary"
SPLITS = ROOT / "splits" / "assignments.csv"
EXTRACTED = ROOT / "data" / "extracted_data.xlsx"
LABELS = ROOT / "data" / "labels.csv"
SIMPLE = ROOT / "results" / "04_compare_simple" / "features.csv"
PRED_3D = PRIMARY / "test_predictions_3d.csv"
PRED_2D = PRIMARY / "test_predictions_2d.csv"
PRED_CNN = ROOT / "results" / "05_compare_cnn" / "imagenet_test_predictions.csv"
PRED_RADI = ROOT / "results" / "05_compare_cnn" / "radimagenet_test_predictions.csv"
OUTDIR = OUT

SEED = 4321
N_BOOT = 1000
CLASSES = [1, 2, 3, 4, 5]
VOLUME_COL = "original_shape_MeshVolume"
DISC_CSF = "disc_csf_mean_ratio"

REF_AUC = {
    "3D_radiomics": 0.936,
    "2D_radiomics": 0.916,
    "CNN_ImageNet": 0.864,
    "disc_csf_ratio": 0.760,
    "MeshVolume": 0.552,
}

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

MODELS = [
    "3D_radiomics",
    "2D_radiomics",
    "CNN_RadImageNet",
    "CNN_ImageNet",
    "disc_csf_ratio",
    "MeshVolume",
]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def macro_auc_ovr(y_true, proba) -> float:
    yb = label_binarize(y_true, classes=CLASSES)
    aucs = []
    for i in range(5):
        if yb[:, i].sum() in (0, len(yb)):
            continue
        aucs.append(roc_auc_score(yb[:, i], proba[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def stars(p: float) -> str:
    if not np.isfinite(p):
        return "na"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def compare_metric(boot_a, boot_b, obs_a, obs_b) -> dict:
    diffs = boot_a - boot_b
    obs_d = float(obs_a - obs_b)
    se = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_tail = 2.0 * min(float(np.mean(diffs <= 0.0)), float(np.mean(diffs >= 0.0)))
    p_tail = min(p_tail, 1.0)
    if np.isfinite(se) and se > 0:
        p_wald = float(2.0 * (1.0 - norm.cdf(abs(obs_d) / se)))
    else:
        p_wald = 1.0 if abs(obs_d) < 1e-15 else 0.0
    return {
        "Value_Model1": float(obs_a),
        "Value_Model2": float(obs_b),
        "Difference": obs_d,
        "CI_Lower": float(ci_lo),
        "CI_Upper": float(ci_hi),
        "SE": se,
        "P_Value": p_wald,
        "P_bootstrap": p_tail,
        "Significance": stars(p_wald),
        "Significance_bootstrap": stars(p_tail),
        "CI_excludes_0": bool(ci_hi < 0 or ci_lo > 0),
    }


def load_saved(path: Path, config: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if config is not None:
        df = df[df["config"] == config].copy()
    need = ["disc_id", "y_true", "y_pred"] + [f"prob_{g}" for g in CLASSES]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} missing {missing}")
    return df[need].drop_duplicates("disc_id").reset_index(drop=True)


def class_weights(ytr: np.ndarray) -> dict:
    bc = np.bincount(ytr, minlength=6)[1:6].astype(float)
    w = (bc.sum() / (5.0 * np.maximum(bc, 1))).tolist()
    return {i + 1: w[i] for i in range(5)}


def pack_from_df(df: pd.DataFrame, order: pd.Index):
    m = df.set_index("disc_id").loc[order]
    y = m["y_true"].astype(int).to_numpy()
    pred = m["y_pred"].astype(int).to_numpy()
    proba = m[[f"prob_{g}" for g in CLASSES]].to_numpy(float)
    return y, pred, proba


def main() -> None:
    global PRED_RADI, OUTDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--radi_pred", default="", help="RadImageNet test-prediction csv (default: legacy path)")
    ap.add_argument("--radi_config", default="vision_radimagenet_reduced", help="config row to select")
    ap.add_argument("--outdir", default="", help="output directory (default: results/06_pairwise)")
    args = ap.parse_args()
    if args.radi_pred:
        PRED_RADI = Path(args.radi_pred)
    if args.outdir:
        OUTDIR = Path(args.outdir)
        OUTDIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    p3 = load_saved(PRED_3D)
    p2 = load_saved(PRED_2D)
    cnn = load_saved(PRED_CNN, "vision_resnet50_reduced")
    if not PRED_RADI.is_file():
        raise SystemExit(f"missing {PRED_RADI}; run model_vision.py --tag radimagenet* first")
    radi = load_saved(PRED_RADI, args.radi_config)
    order = p3["disc_id"]
    for name, d in [("2D", p2), ("CNN_ImageNet", cnn), ("CNN_RadImageNet", radi)]:
        if set(d["disc_id"]) != set(order):
            raise SystemExit(f"{name} disc_id set != 3D test set")

    assign = pd.read_csv(SPLITS)
    labels = pd.read_csv(LABELS)
    simple = pd.read_csv(SIMPLE)
    if "error" in simple.columns:
        simple = simple[simple["error"].fillna("").astype(str).eq("")].copy()
    ex = pd.read_excel(EXTRACTED)
    if VOLUME_COL not in ex.columns:
        raise SystemExit(f"missing {VOLUME_COL}")
    lab = labels.merge(assign[["disc_id", "holdout"]], on="disc_id", how="left")
    lab[VOLUME_COL] = pd.to_numeric(ex[VOLUME_COL].values, errors="coerce")
    drop = [c for c in ("patient_id", "level", "pfirrmann") if c in simple.columns]
    sm = assign.merge(simple.drop(columns=drop, errors="ignore"), on="disc_id", how="inner")
    if "pfirrmann" not in sm.columns:
        sm = sm.merge(labels[["disc_id", "pfirrmann"]], on="disc_id", how="left")

    # disc/CSF
    tr = sm["holdout"] == "train"
    te = sm["holdout"] == "test"
    ytr = sm.loc[tr, "pfirrmann"].astype(int).to_numpy()
    yte_s = sm.loc[te, "pfirrmann"].astype(int).to_numpy()
    xtr = pd.to_numeric(sm.loc[tr, DISC_CSF], errors="coerce")
    xte = pd.to_numeric(sm.loc[te, DISC_CSF], errors="coerce")
    med = xtr.median()
    clf = CatBoostClassifier(**CB, class_weights=class_weights(ytr))
    clf.fit(xtr.fillna(med).to_numpy().reshape(-1, 1), ytr)
    raw = clf.predict_proba(xte.fillna(med).to_numpy().reshape(-1, 1))
    proba_csf = np.zeros((len(yte_s), 5))
    for j, c in enumerate(clf.classes_):
        proba_csf[:, int(c) - 1] = raw[:, j]
    pred_csf = proba_csf.argmax(1) + 1
    csf_df = pd.DataFrame({
        "disc_id": sm.loc[te, "disc_id"].to_numpy(),
        "y_true": yte_s,
        "y_pred": pred_csf,
    })
    for i, g in enumerate(CLASSES):
        csf_df[f"prob_{g}"] = proba_csf[:, i]

    # volume
    ytr_v = lab.loc[lab["holdout"] == "train", "pfirrmann"].astype(int).to_numpy()
    te_v = lab["holdout"] == "test"
    xtr_v = lab.loc[lab["holdout"] == "train", VOLUME_COL]
    xte_v = lab.loc[te_v, VOLUME_COL]
    med_v = xtr_v.median()
    clf_v = CatBoostClassifier(**CB, class_weights=class_weights(ytr_v))
    clf_v.fit(xtr_v.fillna(med_v).to_numpy().reshape(-1, 1), ytr_v)
    raw_v = clf_v.predict_proba(xte_v.fillna(med_v).to_numpy().reshape(-1, 1))
    yte_v = lab.loc[te_v, "pfirrmann"].astype(int).to_numpy()
    proba_v = np.zeros((len(yte_v), 5))
    for j, c in enumerate(clf_v.classes_):
        proba_v[:, int(c) - 1] = raw_v[:, j]
    pred_v = proba_v.argmax(1) + 1
    vol_df = pd.DataFrame({
        "disc_id": lab.loc[te_v, "disc_id"].to_numpy(),
        "y_true": yte_v,
        "y_pred": pred_v,
    })
    for i, g in enumerate(CLASSES):
        vol_df[f"prob_{g}"] = proba_v[:, i]

    store_df = {
        "3D_radiomics": p3,
        "2D_radiomics": p2,
        "CNN_RadImageNet": radi,
        "CNN_ImageNet": cnn,
        "disc_csf_ratio": csf_df,
        "MeshVolume": vol_df,
    }
    y_ref, _, _ = pack_from_df(p3, order)
    store = {}
    for name, df in store_df.items():
        y, pred, proba = pack_from_df(df, order)
        if not np.array_equal(y, y_ref):
            raise SystemExit(f"{name} y_true mismatch vs 3D")
        auc = macro_auc_ovr(y, proba)
        if name in REF_AUC:
            ref = REF_AUC[name]
            if abs(auc - ref) > 5e-4:
                raise SystemExit(f"{name} AUC {auc:.6f} != published ~{ref}")
            print(f"{name:16s} AUC={auc:.6f} (ref {ref})", flush=True)
        else:
            print(f"{name:16s} AUC={auc:.6f}", flush=True)
        store[name] = {"y": y, "pred": pred, "proba": proba, "auc": auc}

    n = len(y_ref)
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    boot = {name: np.empty(N_BOOT) for name in MODELS}
    for b, idx in enumerate(boot_idx):
        yb = y_ref[idx]
        for name in MODELS:
            boot[name][b] = macro_auc_ovr(yb, store[name]["proba"][idx])

    rows = []
    mat = pd.DataFrame(1.0, index=MODELS, columns=MODELS)
    mat_b = pd.DataFrame(1.0, index=MODELS, columns=MODELS)
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1 :]:
            rec = compare_metric(boot[a], boot[b], store[a]["auc"], store[b]["auc"])
            rec["Model1"] = a
            rec["Model2"] = b
            rec["Metric"] = "macro_AUC"
            rows.append(rec)
            mat.loc[a, b] = rec["P_Value"]
            mat.loc[b, a] = rec["P_Value"]
            mat_b.loc[a, b] = rec["P_bootstrap"]
            mat_b.loc[b, a] = rec["P_bootstrap"]
    out = pd.DataFrame(rows).sort_values("P_Value")
    cols = [
        "Model1", "Model2", "Metric", "Value_Model1", "Value_Model2",
        "Difference", "CI_Lower", "CI_Upper", "SE",
        "P_Value", "Significance", "P_bootstrap", "Significance_bootstrap",
        "CI_excludes_0",
    ]
    out[cols].to_csv(OUTDIR / "pairwise_auc.csv", index=False)
    mat.to_csv(OUTDIR / "pairwise_p_matrix.csv")
    mat_b.to_csv(OUTDIR / "pairwise_p_bootstrap_matrix.csv")

    vs_rows = []
    for name in MODELS:
        if name == "3D_radiomics":
            continue
        rec = compare_metric(
            boot[name], boot["3D_radiomics"],
            store[name]["auc"], store["3D_radiomics"]["auc"],
        )
        vs_rows.append({
            "Model": name,
            "AUC": rec["Value_Model1"],
            "AUC_3D": rec["Value_Model2"],
            "AUC_delta_vs_3D": rec["Difference"],
            "AUC_CI_low": rec["CI_Lower"],
            "AUC_CI_high": rec["CI_Upper"],
            "AUC_P_wald": rec["P_Value"],
            "AUC_P_bootstrap": rec["P_bootstrap"],
            "AUC_sig": rec["Significance"],
        })
    vs = pd.DataFrame(vs_rows).sort_values("AUC", ascending=False)
    vs.to_csv(OUTDIR / "pairwise_vs_3d.csv", index=False)

    pred_dump = []
    for name, df in store_df.items():
        tmp = df.set_index("disc_id").loc[order].reset_index()
        tmp.insert(0, "Model", name)
        pred_dump.append(tmp)
    pd.concat(pred_dump, ignore_index=True).to_csv(
        OUTDIR / "test_predictions.csv", index=False
    )

    meta = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "bootstrap": "paired unstratified disc-level; same indices across models",
        "p_primary": "Wald 2*(1-Phi(|d|/SE)) matching ten_models_pairwise / original Bootstrap_*_Comparison",
        "p_secondary": "two-sided bootstrap tail 2*min(P(d<=0), P(d>=0))",
        "n_test": int(n),
        "seconds": round(time.time() - t0, 1),
        "observed_auc": {k: float(store[k]["auc"]) for k in MODELS},
        "inputs_md5": {
            "assignments.csv": md5(SPLITS),
            "test_predictions_3d.csv": md5(PRED_3D),
            "test_predictions_2d.csv": md5(PRED_2D),
            "imagenet_test_predictions.csv": md5(PRED_CNN),
            PRED_RADI.name: md5(PRED_RADI),
            "features.csv": md5(SIMPLE),
            "extracted_data.xlsx": md5(EXTRACTED),
        },
    }
    (OUTDIR / "pairwise.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out[cols].to_string(index=False))
    print(vs.to_string(index=False))
    print(f"Wrote comparator pairwise tables in {OUTDIR} ({meta['seconds']}s)")


if __name__ == "__main__":
    main()
