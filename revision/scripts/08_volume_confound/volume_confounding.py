#!/usr/bin/env python3
"""
Spearman correlation of radiomics features versus MeshVolume, plus a volume-only CatBoost.
Same split and hyperparameters as the 3D primary; writes results/08_volume_confound/.
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
OUT = ROOT / "results" / "08_volume_confound"
DATA = ROOT / "data"
SPLITS = ROOT / "splits" / "assignments.csv"
EXTRACTED = DATA / "extracted_data.xlsx"
LABELS = DATA / "labels.csv"
PRIMARY = ROOT / "results" / "02_primary"
SEL3D = PRIMARY / "selected_features_3d.csv"
PRIMARY_PERF = PRIMARY / "primary_performance.csv"

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

# ROI volume definition (PyRadiomics mesh volume, mm³)
VOLUME_COL = "original_shape_MeshVolume"
VOLUME_ALT = "original_shape_VoxelVolume"

# Always report these (manuscript / SHAP narrative)
ANCHOR_FEATURES = [
    "original_shape_Sphericity",
    "original_shape_Flatness",
    "original_shape_SurfaceArea",
    "original_shape_MajorAxisLength",
    "original_shape_LeastAxisLength",
    "original_shape_Elongation",
    "original_shape_Maximum2DDiameterColumn",
]

META_COLS = {
    "MASK",
    "disc_degree",
    "disc_id",
    "patient_id",
    "level",
    "pfirrmann",
    "holdout",
    "quality",
    "id",
}

# Columns that are themselves volume (or near-identical) — exclude from "feature vs volume" ρ
VOLUME_SELF = {VOLUME_COL, VOLUME_ALT}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_class(name: str) -> str:
    n = str(name)
    if "_shape_" in n or n.startswith("original_shape_"):
        return "shape"
    for t in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        if f"_{t}_" in n:
            return t
    return "other"


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
        "kappa_quadratic": float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
        ),
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


def load_merged() -> pd.DataFrame:
    assign = pd.read_csv(SPLITS)
    labels = pd.read_csv(LABELS)
    ex = pd.read_excel(EXTRACTED)
    assert len(ex) == len(labels) == 630
    ex = ex.copy()
    ex["disc_id"] = labels["disc_id"].values
    ex["patient_id"] = labels["patient_id"].values
    ex["level"] = labels["level"].values
    ex["pfirrmann"] = labels["pfirrmann"].values
    m = assign[["disc_id", "holdout"]].merge(ex, on="disc_id", how="inner")
    assert len(m) == 630, f"merge rows {len(m)}"
    return m


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in META_COLS or c in VOLUME_SELF:
            continue
        if str(c).startswith("cv_rep"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def spearman_vs_volume(df: pd.DataFrame, feat_cols: list[str], volume: pd.Series) -> pd.DataFrame:
    rows = []
    vol = pd.to_numeric(volume, errors="coerce")
    for c in feat_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        msk = x.notna() & vol.notna()
        n = int(msk.sum())
        if n < 10:
            rows.append(
                dict(
                    feature=c,
                    feature_class=feature_class(c),
                    n=n,
                    spearman_rho=np.nan,
                    spearman_p=np.nan,
                    abs_rho=np.nan,
                )
            )
            continue
        rho, p = stats.spearmanr(x[msk], vol[msk])
        rows.append(
            dict(
                feature=c,
                feature_class=feature_class(c),
                n=n,
                spearman_rho=float(rho) if np.isfinite(rho) else np.nan,
                spearman_p=float(p) if np.isfinite(p) else np.nan,
                abs_rho=float(abs(rho)) if np.isfinite(rho) else np.nan,
            )
        )
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False)


def summarize_rho(sp: pd.DataFrame, label: str) -> dict:
    valid = sp.dropna(subset=["abs_rho"])
    n = len(valid)
    if n == 0:
        return {"scope": label, "n_features": 0}
    abs_r = valid["abs_rho"].to_numpy()
    return {
        "scope": label,
        "n_features": n,
        "median_abs_rho": float(np.median(abs_r)),
        "mean_abs_rho": float(np.mean(abs_r)),
        "p25_abs_rho": float(np.percentile(abs_r, 25)),
        "p75_abs_rho": float(np.percentile(abs_r, 75)),
        "max_abs_rho": float(np.max(abs_r)),
        "frac_abs_rho_gt_0.5": float(np.mean(abs_r > 0.5)),
        "n_abs_rho_gt_0.5": int(np.sum(abs_r > 0.5)),
        "frac_abs_rho_gt_0.7": float(np.mean(abs_r > 0.7)),
        "n_abs_rho_gt_0.7": int(np.sum(abs_r > 0.7)),
        "frac_abs_rho_gt_0.9": float(np.mean(abs_r > 0.9)),
        "n_abs_rho_gt_0.9": int(np.sum(abs_r > 0.9)),
    }


def by_class_summary(sp: pd.DataFrame) -> list[dict]:
    rows = []
    for cls, g in sp.dropna(subset=["abs_rho"]).groupby("feature_class"):
        abs_r = g["abs_rho"].to_numpy()
        rows.append(
            {
                "feature_class": cls,
                "n": len(g),
                "median_abs_rho": float(np.median(abs_r)),
                "frac_abs_rho_gt_0.5": float(np.mean(abs_r > 0.5)),
                "n_abs_rho_gt_0.5": int(np.sum(abs_r > 0.5)),
            }
        )
    return sorted(rows, key=lambda r: -r["frac_abs_rho_gt_0.5"])


def fit_volume_only(df: pd.DataFrame) -> dict:
    train = df[df["holdout"] == "train"].copy()
    test = df[df["holdout"] == "test"].copy()
    Xtr = pd.to_numeric(train[VOLUME_COL], errors="coerce")
    Xte = pd.to_numeric(test[VOLUME_COL], errors="coerce")
    med = Xtr.median()
    Xtr = Xtr.fillna(med).to_numpy().reshape(-1, 1)
    Xte = Xte.fillna(med).to_numpy().reshape(-1, 1)
    ytr = train["pfirrmann"].astype(int).to_numpy()
    yte = test["pfirrmann"].astype(int).to_numpy()

    bc = np.bincount(ytr, minlength=6)[1:6].astype(float)
    w = (bc.sum() / (5.0 * np.maximum(bc, 1))).tolist()
    class_weights = {i + 1: w[i] for i in range(5)}

    clf = CatBoostClassifier(**CB_PARAMS, class_weights=class_weights)
    clf.fit(Xtr, ytr)
    pr = clf.predict_proba(Xte)
    proba = np.zeros((len(yte), 5))
    for j, c in enumerate(clf.classes_):
        proba[:, int(c) - 1] = pr[:, j]
    pred = proba.argmax(1) + 1

    m = metrics_block(yte, pred, proba)
    m.update(bootstrap_ci(yte, pred, proba))
    m.update(
        dict(
            config="volume_only_MeshVolume",
            volume_feature=VOLUME_COL,
            n_features=1,
            n_train=len(train),
            n_test=len(test),
        )
    )

    # also Spearman volume vs grade (full cohort)
    vol = pd.to_numeric(df[VOLUME_COL], errors="coerce")
    g = df["pfirrmann"].astype(int)
    msk = vol.notna()
    rho, p = stats.spearmanr(vol[msk], g[msk])
    m["spearman_volume_vs_grade"] = float(rho)
    m["spearman_volume_vs_grade_p"] = float(p)
    return m


def primary_importance_and_rho(
    df: pd.DataFrame, selected: list[str], sp_map: dict[str, float]
) -> pd.DataFrame:
    """Re-fit primary CatBoost on 517 features to rank importance; join volume ρ."""
    use = [c for c in selected if c in df.columns]
    train = df[df["holdout"] == "train"].copy()
    Xtr = train[use].apply(pd.to_numeric, errors="coerce")
    med = Xtr.median()
    Xtr = Xtr.fillna(med)
    # drop constant
    keep = [c for c in use if float(Xtr[c].std() or 0) > 1e-12]
    Xtr = Xtr[keep]
    ytr = train["pfirrmann"].astype(int).to_numpy()

    bc = np.bincount(ytr, minlength=6)[1:6].astype(float)
    w = (bc.sum() / (5.0 * np.maximum(bc, 1))).tolist()
    class_weights = {i + 1: w[i] for i in range(5)}
    clf = CatBoostClassifier(**CB_PARAMS, class_weights=class_weights)
    clf.fit(Xtr, ytr)
    imp = pd.DataFrame(
        {
            "feature": keep,
            "importance": clf.get_feature_importance(),
            "in_primary_517": True,
            "spearman_rho_vs_volume": [sp_map.get(c, np.nan) for c in keep],
        }
    )
    imp["abs_rho_vs_volume"] = imp["spearman_rho_vs_volume"].abs()
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["importance_rank"] = np.arange(1, len(imp) + 1)
    return imp


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading data …")
    df = load_merged()
    selected = pd.read_csv(SEL3D)["Feature"].tolist()
    selected = [c for c in selected if c in df.columns]
    print(f"  n discs={len(df)}, n selected in matrix={len(selected)}")

    if VOLUME_COL not in df.columns:
        raise SystemExit(f"missing {VOLUME_COL}")

    # Mesh vs Voxel sanity
    mesh = pd.to_numeric(df[VOLUME_COL], errors="coerce")
    voxel = pd.to_numeric(df[VOLUME_ALT], errors="coerce")
    r_mv = float(stats.spearmanr(mesh, voxel).correlation)

    # --- 1) Spearman feature vs volume ---
    print("Spearman feature vs MeshVolume …")
    all_feats = numeric_feature_cols(df)
    # include only radiomics-like: exclude any leftover ints that aren't features
    # already filtered META
    sp_all = spearman_vs_volume(df, all_feats, mesh)
    sp_all["in_primary_517"] = sp_all["feature"].isin(selected)
    sp_all.to_csv(OUT / "feature_volume_spearman.csv", index=False)

    sp_517 = sp_all[sp_all["in_primary_517"]].copy()
    sum_all = summarize_rho(sp_all, "all_features_excl_volume")
    sum_517 = summarize_rho(sp_517, "primary_517")
    by_cls = by_class_summary(sp_all)
    by_cls_517 = by_class_summary(sp_517)

    print(
        f"  all: |ρ|>0.5 = {sum_all['frac_abs_rho_gt_0.5']:.3f} "
        f"({sum_all['n_abs_rho_gt_0.5']}/{sum_all['n_features']}); "
        f"median |ρ|={sum_all['median_abs_rho']:.3f}"
    )
    print(
        f"  517: |ρ|>0.5 = {sum_517['frac_abs_rho_gt_0.5']:.3f} "
        f"({sum_517['n_abs_rho_gt_0.5']}/{sum_517['n_features']}); "
        f"median |ρ|={sum_517['median_abs_rho']:.3f}"
    )

    # --- 2) volume-only model ---
    print("Volume-only CatBoost …")
    vol_perf = fit_volume_only(df)
    print(
        f"  macro AUC={vol_perf['macro_AUC']:.3f} "
        f"({vol_perf['macro_AUC_ci_low']:.3f}–{vol_perf['macro_AUC_ci_high']:.3f})"
    )
    pd.DataFrame([vol_perf]).to_csv(OUT / "volume_only_performance.csv", index=False)

    # primary ref row
    primary_auc = None
    if PRIMARY_PERF.exists():
        pref = pd.read_csv(PRIMARY_PERF)
        row = pref[pref["config"] == "3D_primary"]
        if len(row):
            primary_auc = float(row.iloc[0]["macro_AUC"])
            ref = {
                "config": "3D_primary_ref",
                "macro_AUC": primary_auc,
                "macro_AUC_ci_low": float(row.iloc[0].get("macro_AUC_ci_low", np.nan)),
                "macro_AUC_ci_high": float(row.iloc[0].get("macro_AUC_ci_high", np.nan)),
                "accuracy": float(row.iloc[0].get("accuracy", np.nan)),
                "kappa_quadratic": float(row.iloc[0].get("kappa_quadratic", np.nan)),
                "n_features": int(row.iloc[0].get("n_features_selected", 517)),
            }
            pd.DataFrame([vol_perf, ref]).to_csv(
                OUT / "volume_vs_primary.csv", index=False
            )

    # --- 3) primary importance + top features vs volume ---
    print("Primary CatBoost importance + volume ρ …")
    sp_map = {
        r.feature: r.spearman_rho
        for r in sp_all.itertuples()
        if np.isfinite(getattr(r, "spearman_rho", np.nan))
    }
    # volume self not in sp_map — add for anchors if needed
    imp = primary_importance_and_rho(df, selected, sp_map)
    imp.to_csv(OUT / "primary517_importance_and_rho.csv", index=False)

    top3 = imp.head(3).copy()
    top10 = imp.head(10).copy()

    # Anchor shape features (always report)
    anchor_rows = []
    for f in ANCHOR_FEATURES:
        if f not in df.columns:
            continue
        rho = sp_map.get(f, np.nan)
        if f in VOLUME_SELF:
            rho, _ = stats.spearmanr(
                pd.to_numeric(df[f], errors="coerce"), mesh
            )
            rho = float(rho)
        rank_row = imp[imp["feature"] == f]
        rank = int(rank_row["importance_rank"].iloc[0]) if len(rank_row) else None
        imp_val = float(rank_row["importance"].iloc[0]) if len(rank_row) else np.nan
        in517 = f in selected
        anchor_rows.append(
            dict(
                feature=f,
                in_primary_517=in517,
                importance_rank=rank,
                importance=imp_val,
                spearman_rho_vs_volume=float(rho) if np.isfinite(rho) else np.nan,
                abs_rho_vs_volume=float(abs(rho)) if np.isfinite(rho) else np.nan,
                still_high_importance=bool(rank is not None and rank <= 10),
            )
        )
    # also MeshVolume itself: is it in 517?
    anchor_rows.insert(
        0,
        dict(
            feature=VOLUME_COL,
            in_primary_517=VOLUME_COL in selected,
            importance_rank=None,
            importance=np.nan,
            spearman_rho_vs_volume=1.0,
            abs_rho_vs_volume=1.0,
            still_high_importance=False,
            note="volume itself (reference; excluded from feature–volume table)",
        ),
    )
    anchors = pd.DataFrame(anchor_rows)
    anchors.to_csv(OUT / "anchor_features_volume_rho.csv", index=False)

    top3.to_csv(OUT / "top3_features_volume_rho.csv", index=False)

    # Highest |ρ| among 517
    high517 = sp_517.dropna(subset=["abs_rho"]).head(15)
    high517.to_csv(OUT / "primary517_highest_volume_rho.csv", index=False)

    # Summary JSON
    summary = {
        "volume_definition": VOLUME_COL,
        "volume_alt": VOLUME_ALT,
        "mesh_vs_voxel_spearman": r_mv,
        "volume_in_primary_517": VOLUME_COL in selected,
        "voxelvolume_in_primary_517": VOLUME_ALT in selected,
        "spearman_all": sum_all,
        "spearman_primary_517": sum_517,
        "by_class_all": by_cls,
        "by_class_primary_517": by_cls_517,
        "volume_only": {
            "macro_AUC": vol_perf["macro_AUC"],
            "macro_AUC_ci": [
                vol_perf["macro_AUC_ci_low"],
                vol_perf["macro_AUC_ci_high"],
            ],
            "accuracy": vol_perf["accuracy"],
            "kappa_quadratic": vol_perf["kappa_quadratic"],
            "spearman_volume_vs_grade": vol_perf["spearman_volume_vs_grade"],
        },
        "primary_ref_macro_AUC": primary_auc,
        "delta_primary_minus_volume_only": (
            None if primary_auc is None else float(primary_auc - vol_perf["macro_AUC"])
        ),
        "top3_by_importance": top3[
            ["feature", "importance", "importance_rank", "spearman_rho_vs_volume", "abs_rho_vs_volume"]
        ].to_dict(orient="records"),
        "top10_by_importance": top10[
            ["feature", "importance", "importance_rank", "spearman_rho_vs_volume", "abs_rho_vs_volume"]
        ].to_dict(orient="records"),
        "anchors": anchors.to_dict(orient="records"),
        "narrative_hints": {
            "volume_proxy_strong_if": "volume-only AUC close to primary OR many top features |ρ|>0.5",
            "volume_not_dominant_if": "volume-only AUC << primary AND top features modest |ρ|",
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    meta = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "catboost_params": CB_PARAMS,
        "volume_col": VOLUME_COL,
        "split": "revision/splits/assignments.csv",
        "seconds": round(time.time() - t0, 1),
        "inputs_md5": {
            "assignments.csv": md5(SPLITS),
            "extracted_data.xlsx": md5(EXTRACTED),
            "labels.csv": md5(LABELS),
            "selected_features_3d.csv": md5(SEL3D),
        },
        "outputs": sorted(p.name for p in OUT.glob("C2_*")),
                "policy": "v3 full C2: Spearman + volume-only model + top/anchor features vs volume; no partial-correlation retrain of full pipeline",
    }
    (OUT / "volume_confounding.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    readme = f"""# 08_volume_confound

Volume = `{VOLUME_COL}` (mm3). Volume-only macro AUC {vol_perf['macro_AUC']:.3f}
({vol_perf['macro_AUC_ci_low']:.3f}–{vol_perf['macro_AUC_ci_high']:.3f}).
Primary 3D AUC {primary_auc}. See summary.json.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print("\n=== C2 done ===")
    print(json.dumps(
        {
            "all_|rho|>0.5": sum_all["frac_abs_rho_gt_0.5"],
            "517_|rho|>0.5": sum_517["frac_abs_rho_gt_0.5"],
            "volume_only_AUC": vol_perf["macro_AUC"],
            "primary_AUC": primary_auc,
            "top3": top3[["feature", "importance", "spearman_rho_vs_volume"]].to_dict(
                "records"
            ),
        },
        indent=2,
        default=str,
    ))
    print(f"Wrote {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
