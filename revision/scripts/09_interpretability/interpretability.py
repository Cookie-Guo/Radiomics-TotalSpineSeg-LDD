#!/usr/bin/env python3
"""
Grade-wise trends, correlation with simple measurements, native TreeSHAP, Top-k ablation.
Uses the primary 517-feature CatBoost; writes results/09_interpretability/.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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
OUT = ROOT / "results" / "09_interpretability"
FIG = OUT / "figures"
DATA = ROOT / "data"
SPLITS = ROOT / "splits" / "assignments.csv"
EXTRACTED = DATA / "extracted_data.xlsx"
LABELS = DATA / "labels.csv"
PRIMARY = ROOT / "results" / "02_primary"
SEL3D = PRIMARY / "selected_features_3d.csv"
PRIMARY_PERF = PRIMARY / "primary_performance.csv"
SIMPLE = ROOT / "results" / "04_compare_simple" / "features.csv"

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

SIMPLE_COLS = {
    "dhi": "DHI",
    "delta_peak_si_norm": "norm_peak_SI_diff",
    "disc_csf_mean_ratio": "disc_CSF_ratio",
    "area_mm2": "midsag_area",
    "sphericity_2d": "sphericity_2d",
}

# Always include if present in the 517 (manuscript / biology anchors)
ANCHOR_FEATURES = [
    "original_shape_Sphericity",
    "original_shape_Flatness",
    "original_firstorder_Median",
    "original_firstorder_Mean",
    "original_firstorder_10Percentile",
    "log-sigma-4-0-mm-3D_firstorder_10Percentile",
    "log-sigma-4-0-mm-3D_firstorder_Minimum",
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

GRADE_COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
GRADE_LABELS = ["I", "II", "III", "IV", "V"]


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


def short_name(f: str, max_len: int = 42) -> str:
    s = str(f)
    repl = (
        ("log-sigma-3-0-mm-3D_", "LoG3mm_"),
        ("log-sigma-4-0-mm-3D_", "LoG4mm_"),
        ("log-sigma-5-0-mm-3D_", "LoG5mm_"),
        ("original_shape_", "shape_"),
        ("original_firstorder_", "FO_"),
        ("lbp-3D-m2_", "LBPm2_"),
        ("lbp-3D-m1_", "LBPm1_"),
        ("lbp-3D-k_", "LBPk_"),
        ("wavelet-LLL_", "wavLLL_"),
        ("wavelet-LLH_", "wavLLH_"),
        ("wavelet-HLH_", "wavHLH_"),
        ("exponential_", "exp_"),
        ("logarithm_", "log_"),
        ("square_", "sq_"),
        ("squareroot_", "sqrt_"),
    )
    for a, b in repl:
        s = s.replace(a, b)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


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


def class_weights_from_y(ytr: np.ndarray) -> dict:
    bc = np.bincount(ytr, minlength=6)[1:6].astype(float)
    w = (bc.sum() / (5.0 * np.maximum(bc, 1))).tolist()
    return {i + 1: w[i] for i in range(5)}


def fit_predict(Xtr, ytr, Xte, yte) -> tuple[dict, np.ndarray, np.ndarray, CatBoostClassifier]:
    clf = CatBoostClassifier(**CB_PARAMS, class_weights=class_weights_from_y(ytr))
    clf.fit(Xtr, ytr)
    pr = clf.predict_proba(Xte)
    proba = np.zeros((len(yte), 5))
    for j, c in enumerate(clf.classes_):
        proba[:, int(c) - 1] = pr[:, j]
    pred = proba.argmax(1) + 1
    m = metrics_block(yte, pred, proba)
    m.update(bootstrap_ci(yte, pred, proba))
    return m, pred, proba, clf


def parse_shap(raw, n_features: int) -> np.ndarray:
    """Return (n_samples, n_classes, n_features)."""
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] == n_features + 1:
            return arr[:, :-1][:, None, :]
        raise ValueError(f"2D shap shape {arr.shape} vs n_features={n_features}")
    if arr.ndim != 3:
        raise ValueError(f"unexpected shap ndim {arr.ndim}, shape={arr.shape}")
    # (n, classes, feat+1)
    if arr.shape[2] == n_features + 1:
        return arr[:, :, :-1]
    # (n, feat+1, classes)
    if arr.shape[1] == n_features + 1:
        return np.transpose(arr[:, :-1, :], (0, 2, 1))
    if arr.shape[2] == n_features:
        return arr
    if arr.shape[1] == n_features:
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f"unexpected 3D shap shape {arr.shape} vs n_features={n_features}")


def spearman_pair(x, y) -> tuple[float, float, int]:
    xx = pd.to_numeric(pd.Series(x), errors="coerce")
    yy = pd.to_numeric(pd.Series(y), errors="coerce")
    msk = xx.notna() & yy.notna()
    n = int(msk.sum())
    if n < 10:
        return float("nan"), float("nan"), n
    rho, p = stats.spearmanr(xx[msk], yy[msk])
    return (
        float(rho) if np.isfinite(rho) else float("nan"),
        float(p) if np.isfinite(p) else float("nan"),
        n,
    )


def median_trend(meds: list[float]) -> str:
    vals = [m for m in meds if np.isfinite(m)]
    if len(vals) < 3:
        return "insufficient"
    d = np.diff(vals)
    pos = np.sum(d > 0)
    neg = np.sum(d < 0)
    if neg == 0 and pos > 0:
        return "monotone_up"
    if pos == 0 and neg > 0:
        return "monotone_down"
    if pos >= 3 and neg == 0:
        return "monotone_up"
    if neg >= 3 and pos == 0:
        return "monotone_down"
    return "mixed"


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
    simple = pd.read_csv(SIMPLE)
    keep = ["disc_id"] + [c for c in SIMPLE_COLS if c in simple.columns]
    m = m.merge(simple[keep], on="disc_id", how="left")
    assert len(m) == 630
    return m


def impute_train_median(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    Xtr = train[cols].apply(pd.to_numeric, errors="coerce")
    Xte = test[cols].apply(pd.to_numeric, errors="coerce")
    med = Xtr.median()
    return Xtr.fillna(med), Xte.fillna(med), med


def savefig(fig, name: str) -> None:
    p = FIG / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame, feats: list[str]) -> None:
    n = len(feats)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.1 * nrows), squeeze=False)
    grades = [1, 2, 3, 4, 5]
    for i, f in enumerate(feats):
        ax = axes[i // ncols][i % ncols]
        data = [
            pd.to_numeric(df.loc[df["pfirrmann"] == g, f], errors="coerce").dropna().to_numpy()
            for g in grades
        ]
        bp = ax.boxplot(
            data,
            tick_labels=GRADE_LABELS,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color="#444444"),
            capprops=dict(color="#444444"),
        )
        for patch, c in zip(bp["boxes"], GRADE_COLORS):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        ax.set_title(short_name(f, 36), fontsize=9)
        ax.set_xlabel("Pfirrmann")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Primary radiomic features by Pfirrmann grade (full cohort)", fontsize=11, y=1.01)
    fig.tight_layout()
    savefig(fig, "top_features_by_grade_boxplots.png")


def plot_heatmap(mat: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.38 * len(mat) + 1.6)))
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(list(mat.columns), rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([short_name(i, 40) for i in mat.index], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iloc[i, j]
            if not np.isfinite(v):
                continue
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(v) > 0.55 else "#222222",
            )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Spearman ρ")
    ax.set_title("Top radiomic features vs simple biomarkers")
    fig.tight_layout()
    savefig(fig, "radiomics_vs_simple_heatmap.png")


def plot_shap_bars(shap_df: pd.DataFrame, top_n: int = 15) -> None:
    sub = shap_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 0.38 * top_n + 1.4))
    ax.barh(range(len(sub)), sub["mean_abs_shap_all"].to_numpy(), color="#3d5a80")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([short_name(f, 44) for f in sub["feature"]], fontsize=8)
    ax.set_xlabel("Mean |SHAP| (average over classes, all discs)")
    ax.set_title("CatBoost TreeSHAP — global feature contribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, "shap_meanabs_top15.png")


def plot_shap_dependence(X: pd.DataFrame, shap_ncf: np.ndarray, feats: list[str], y: np.ndarray) -> None:
    """shap_ncf: (n, classes, features) aligned to X columns."""
    cols = list(X.columns)
    k = len(feats)
    fig, axes = plt.subplots(1, k, figsize=(4.2 * k, 3.5), squeeze=False, constrained_layout=True)
    sc = None
    for i, f in enumerate(feats):
        ax = axes[0][i]
        j = cols.index(f)
        xv = pd.to_numeric(X[f], errors="coerce").to_numpy()
        sv = shap_ncf[:, :, j].mean(axis=1)
        sc = ax.scatter(xv, sv, c=y, cmap="RdYlBu_r", s=10, alpha=0.7, vmin=1, vmax=5)
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_xlabel(short_name(f, 36), fontsize=8)
        ax.set_ylabel("Mean SHAP (across classes)" if i == 0 else "")
        ax.set_title(short_name(f, 32), fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Pfirrmann")
    fig.suptitle("SHAP dependence (official 517 CatBoost)", fontsize=11)
    savefig(fig, "shap_dependence_top3.png")


def plot_ablation(rows: list[dict], primary_auc: float) -> None:
    labels = [r["config"] for r in rows]
    aucs = [r["macro_AUC"] for r in rows]
    lo = [r.get("macro_AUC_ci_low", np.nan) for r in rows]
    hi = [r.get("macro_AUC_ci_high", np.nan) for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(labels))
    yerr = None
    if all(np.isfinite(lo)) and all(np.isfinite(hi)):
        yerr = np.vstack([np.array(aucs) - np.array(lo), np.array(hi) - np.array(aucs)])
    ax.bar(x, aucs, color=["#1b4965" if "primary" in c else "#5fa8d3" for c in labels], yerr=yerr, capsize=3)
    ax.axhline(primary_auc, color="#e63946", ls="--", lw=1, label=f"official primary {primary_auc:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("macro AUC")
    ax.set_title("Top-k ablation vs official 517-feature primary")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, "ablation_auc.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading data …")
    df = load_merged()
    selected = pd.read_csv(SEL3D)["Feature"].tolist()
    selected = [c for c in selected if c in df.columns]
    print(f"  n discs={len(df)}, n selected in matrix={len(selected)}")

    train = df[df["holdout"] == "train"].copy()
    test = df[df["holdout"] == "test"].copy()
    ytr = train["pfirrmann"].astype(int).to_numpy()
    yte = test["pfirrmann"].astype(int).to_numpy()
    y_all = df["pfirrmann"].astype(int).to_numpy()

    Xtr, Xte, med = impute_train_median(train, test, selected)
    keep = [c for c in selected if float(Xtr[c].std() or 0) > 1e-12]
    Xtr, Xte = Xtr[keep], Xte[keep]
    X_all = df[keep].apply(pd.to_numeric, errors="coerce").fillna(med)

    # --- 1) Fit official-spec CatBoost on 517 (for importance + SHAP) ---
    print("Fitting primary-spec CatBoost on 517 …")
    m517, _, _, clf = fit_predict(Xtr, ytr, Xte, yte)
    print(
        f"  refit 517 macro AUC={m517['macro_AUC']:.3f} "
        f"({m517['macro_AUC_ci_low']:.3f}–{m517['macro_AUC_ci_high']:.3f})"
    )

    imp = pd.DataFrame(
        {
            "feature": keep,
            "importance": clf.get_feature_importance(),
            "feature_class": [feature_class(c) for c in keep],
        }
    )
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["importance_rank"] = np.arange(1, len(imp) + 1)

    # --- 2) Feature vs grade Spearman (all 517) ---
    print("Spearman feature vs Pfirrmann …")
    grade_rows = []
    for c in keep:
        rho, p, n = spearman_pair(df[c], y_all)
        grade_rows.append(
            dict(
                feature=c,
                feature_class=feature_class(c),
                n=n,
                spearman_rho_vs_grade=rho,
                spearman_p=p,
                abs_rho=abs(rho) if np.isfinite(rho) else np.nan,
            )
        )
    sp_grade = pd.DataFrame(grade_rows)
    n_tests = int(sp_grade["spearman_p"].notna().sum())
    bonf = 0.05 / max(n_tests, 1)
    sp_grade["bonferroni_alpha"] = bonf
    sp_grade["significant_bonferroni"] = sp_grade["spearman_p"] < bonf
    sp_grade = sp_grade.merge(
        imp[["feature", "importance", "importance_rank"]], on="feature", how="left"
    )
    sp_grade = sp_grade.sort_values("abs_rho", ascending=False)
    sp_grade.to_csv(OUT / "feature_vs_grade_spearman.csv", index=False)

    valid = sp_grade.dropna(subset=["abs_rho"])
    abs_r = valid["abs_rho"].to_numpy()
    grade_sum = {
        "n_features": int(len(valid)),
        "median_abs_rho": float(np.median(abs_r)),
        "mean_abs_rho": float(np.mean(abs_r)),
        "frac_abs_rho_gt_0.3": float(np.mean(abs_r > 0.3)),
        "n_abs_rho_gt_0.3": int(np.sum(abs_r > 0.3)),
        "frac_abs_rho_gt_0.5": float(np.mean(abs_r > 0.5)),
        "n_abs_rho_gt_0.5": int(np.sum(abs_r > 0.5)),
        "n_bonferroni_sig": int(sp_grade["significant_bonferroni"].fillna(False).sum()),
        "bonferroni_alpha": bonf,
    }
    print(
        f"  517 vs grade: median |ρ|={grade_sum['median_abs_rho']:.3f}; "
        f"|ρ|>0.5={grade_sum['frac_abs_rho_gt_0.5']:.3f} "
        f"({grade_sum['n_abs_rho_gt_0.5']}/{grade_sum['n_features']}); "
        f"Bonferroni sig={grade_sum['n_bonferroni_sig']}"
    )

    # Focus features: top-10 importance + anchors present
    top10 = imp.head(10)["feature"].tolist()
    anchors_present = [f for f in ANCHOR_FEATURES if f in keep]
    focus = list(dict.fromkeys(top10 + anchors_present))

    # Per-grade stats for focus features
    stats_rows = []
    for f in focus:
        x = pd.to_numeric(df[f], errors="coerce")
        meds = []
        row = {
            "feature": f,
            "short_name": short_name(f),
            "feature_class": feature_class(f),
            "importance_rank": int(imp.loc[imp["feature"] == f, "importance_rank"].iloc[0]),
            "importance": float(imp.loc[imp["feature"] == f, "importance"].iloc[0]),
        }
        for g in range(1, 6):
            xv = x[df["pfirrmann"] == g].dropna()
            med_g = float(xv.median()) if len(xv) else float("nan")
            meds.append(med_g)
            row[f"n_grade_{g}"] = int(len(xv))
            row[f"median_grade_{g}"] = med_g
            row[f"q25_grade_{g}"] = float(xv.quantile(0.25)) if len(xv) else float("nan")
            row[f"q75_grade_{g}"] = float(xv.quantile(0.75)) if len(xv) else float("nan")
        rho, p, n = spearman_pair(x, y_all)
        row["spearman_rho_vs_grade"] = rho
        row["spearman_p"] = p
        row["n_spearman"] = n
        row["median_trend"] = median_trend(meds)
        stats_rows.append(row)
    focus_stats = pd.DataFrame(stats_rows)
    focus_stats.to_csv(OUT / "focus_features_grade_stats.csv", index=False)

    # --- 3) vs simple features ---
    print("Spearman radiomics vs simple features …")
    simple_avail = [c for c in SIMPLE_COLS if c in df.columns]
    vs_simple_long = []
    heat = pd.DataFrame(index=focus, columns=[SIMPLE_COLS[c] for c in simple_avail], dtype=float)
    for f in focus:
        best_s, best_abs, best_rho = None, -1.0, float("nan")
        for s in simple_avail:
            rho, p, n = spearman_pair(df[f], df[s])
            vs_simple_long.append(
                dict(
                    radiomic_feature=f,
                    simple_feature=s,
                    simple_label=SIMPLE_COLS[s],
                    spearman_rho=rho,
                    spearman_p=p,
                    n=n,
                    abs_rho=abs(rho) if np.isfinite(rho) else np.nan,
                )
            )
            heat.loc[f, SIMPLE_COLS[s]] = rho
            if np.isfinite(rho) and abs(rho) > best_abs:
                best_abs, best_rho, best_s = abs(rho), rho, s
        # attach to focus_stats later
        focus_stats.loc[focus_stats["feature"] == f, "best_simple"] = best_s
        focus_stats.loc[focus_stats["feature"] == f, "best_simple_rho"] = best_rho
    vs_simple = pd.DataFrame(vs_simple_long)
    vs_simple.to_csv(OUT / "focus_vs_simple_spearman.csv", index=False)
    focus_stats.to_csv(OUT / "focus_features_grade_stats.csv", index=False)

    # All 517 vs each simple (overlap summary)
    overlap_rows = []
    for s in simple_avail:
        rhos = []
        for c in keep:
            rho, p, n = spearman_pair(df[c], df[s])
            if np.isfinite(rho):
                rhos.append(abs(rho))
        rhos = np.asarray(rhos, dtype=float)
        overlap_rows.append(
            {
                "simple_feature": s,
                "simple_label": SIMPLE_COLS[s],
                "n_radiomics": int(len(rhos)),
                "median_abs_rho": float(np.median(rhos)),
                "mean_abs_rho": float(np.mean(rhos)),
                "frac_abs_rho_gt_0.5": float(np.mean(rhos > 0.5)),
                "n_abs_rho_gt_0.5": int(np.sum(rhos > 0.5)),
                "frac_abs_rho_gt_0.7": float(np.mean(rhos > 0.7)),
                "n_abs_rho_gt_0.7": int(np.sum(rhos > 0.7)),
                "max_abs_rho": float(np.max(rhos)),
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(OUT / "517_vs_simple_overlap.csv", index=False)
    print("  517 vs simple |ρ|>0.5:")
    for rec in overlap.to_dict("records"):
        print(
            f"    {rec['simple_label']}: {rec['frac_abs_rho_gt_0.5']:.3f} "
            f"(median |ρ|={rec['median_abs_rho']:.3f})"
        )

    # Concordance table (manuscript-facing)
    conc = focus_stats[
        [
            "feature",
            "short_name",
            "feature_class",
            "importance_rank",
            "importance",
            "spearman_rho_vs_grade",
            "spearman_p",
            "median_trend",
            "best_simple",
            "best_simple_rho",
        ]
    ].copy()
    conc.to_csv(OUT / "concordance_table.csv", index=False)

    # --- 4) SHAP ---
    print("Computing CatBoost ShapValues on all 630 …")
    t_shap = time.time()
    pool_all = Pool(X_all, y_all)
    raw_shap = clf.get_feature_importance(type="ShapValues", data=pool_all)
    shap_ncf = parse_shap(raw_shap, n_features=len(keep))
    print(f"  shap array {shap_ncf.shape} in {time.time() - t_shap:.1f}s")

    test_mask = (df["holdout"] == "test").to_numpy()
    mean_abs_all = np.mean(np.abs(shap_ncf), axis=(0, 1))
    mean_abs_test = np.mean(np.abs(shap_ncf[test_mask]), axis=(0, 1))
    shap_df = pd.DataFrame(
        {
            "feature": keep,
            "mean_abs_shap_all": mean_abs_all,
            "mean_abs_shap_test": mean_abs_test,
        }
    )
    for k, g in enumerate([1, 2, 3, 4, 5]):
        shap_df[f"mean_abs_shap_class_{g}"] = np.mean(np.abs(shap_ncf[:, k, :]), axis=0)
    shap_df = shap_df.merge(
        imp[["feature", "importance", "importance_rank", "feature_class"]],
        on="feature",
        how="left",
    )
    shap_df = shap_df.sort_values("mean_abs_shap_all", ascending=False).reset_index(drop=True)
    shap_df["shap_rank"] = np.arange(1, len(shap_df) + 1)
    shap_df.to_csv(OUT / "shap_mean_abs.csv", index=False)

    # Rank agreement: PVC importance vs mean |SHAP|
    rho_imp_shap, p_imp_shap, _ = spearman_pair(
        imp.set_index("feature").loc[keep, "importance"],
        shap_df.set_index("feature").loc[keep, "mean_abs_shap_all"],
    )
    print(f"  importance vs mean|SHAP| Spearman ρ={rho_imp_shap:.3f}")

    # --- 5) Top-k ablation ---
    print("Top-k ablation …")
    primary_auc = None
    primary_row = {}
    if PRIMARY_PERF.exists():
        pref = pd.read_csv(PRIMARY_PERF)
        prow = pref[pref["config"] == "3D_primary"]
        if len(prow):
            primary_auc = float(prow.iloc[0]["macro_AUC"])
            primary_row = {
                "config": "3D_primary_official",
                "n_features": int(prow.iloc[0].get("n_features_selected", 517)),
                "macro_AUC": primary_auc,
                "macro_AUC_ci_low": float(prow.iloc[0].get("macro_AUC_ci_low", np.nan)),
                "macro_AUC_ci_high": float(prow.iloc[0].get("macro_AUC_ci_high", np.nan)),
                "accuracy": float(prow.iloc[0].get("accuracy", np.nan)),
                "kappa_quadratic": float(prow.iloc[0].get("kappa_quadratic", np.nan)),
                "note": "official primary; not refit in C3",
            }

    ablation_rows = []
    if primary_row:
        ablation_rows.append(primary_row)
    refit_row = dict(config="refit_517_sanity", n_features=len(keep), **m517)
    refit_row["note"] = "same hyperparams/split; sanity check only"
    ablation_rows.append(refit_row)

    for k in (5, 10):
        feats_k = top10[:k]
        Xtr_k, Xte_k = Xtr[feats_k], Xte[feats_k]
        mk, _, _, _ = fit_predict(Xtr_k, ytr, Xte_k, yte)
        mk.update(
            dict(
                config=f"top{k}_importance",
                n_features=k,
                features=",".join(feats_k),
                note="subset of train-fit PredictionValuesChange ranks",
            )
        )
        ablation_rows.append(mk)
        print(
            f"  top{k} macro AUC={mk['macro_AUC']:.3f} "
            f"({mk['macro_AUC_ci_low']:.3f}–{mk['macro_AUC_ci_high']:.3f})"
        )

    abl = pd.DataFrame(ablation_rows)
    # stable column order
    front = [
        "config",
        "n_features",
        "macro_AUC",
        "macro_AUC_ci_low",
        "macro_AUC_ci_high",
        "accuracy",
        "macro_sensitivity",
        "macro_specificity",
        "kappa",
        "kappa_quadratic",
    ]
    rest = [c for c in abl.columns if c not in front]
    abl = abl[front + rest]
    abl.to_csv(OUT / "ablation_performance.csv", index=False)

    top5_auc = float(abl.loc[abl["config"] == "top5_importance", "macro_AUC"].iloc[0])
    top10_auc = float(abl.loc[abl["config"] == "top10_importance", "macro_AUC"].iloc[0])

    # --- figures ---
    print("Writing figures …")
    box_feats = list(dict.fromkeys(top10[:8] + [a for a in anchors_present if a in top10 or a.startswith("original_shape")]))
    # keep ≤8 panels
    box_feats = box_feats[:8]
    plot_boxplots(df, box_feats)
    plot_heatmap(heat.astype(float))
    plot_shap_bars(shap_df, top_n=15)
    plot_shap_dependence(X_all, shap_ncf, top10[:3], y_all)
    plot_ablation(
        [
            r
            for r in ablation_rows
            if r["config"] in ("3D_primary_official", "top5_importance", "top10_importance", "refit_517_sanity")
        ],
        primary_auc if primary_auc is not None else m517["macro_AUC"],
    )

    # Top-3 / top-10 payloads
    top3 = []
    for f in top10[:3]:
        r = focus_stats.loc[focus_stats["feature"] == f].iloc[0]
        top3.append(
            dict(
                feature=f,
                importance_rank=int(r["importance_rank"]),
                importance=float(r["importance"]),
                spearman_rho_vs_grade=float(r["spearman_rho_vs_grade"]),
                median_trend=r["median_trend"],
                best_simple=r.get("best_simple"),
                best_simple_rho=None
                if pd.isna(r.get("best_simple_rho"))
                else float(r["best_simple_rho"]),
                medians=[float(r[f"median_grade_{g}"]) for g in range(1, 6)],
            )
        )

    shap_top5 = shap_df.head(5)[["feature", "shap_rank", "mean_abs_shap_all", "importance_rank"]].to_dict(
        "records"
    )

    summary = {
        "n_discs": 630,
        "n_primary_features": len(keep),
        "refit_517_macro_AUC": m517["macro_AUC"],
        "official_primary_macro_AUC": primary_auc,
        "feature_vs_grade": grade_sum,
        "top10_importance": top10,
        "anchors_present": anchors_present,
        "focus_features": focus,
        "top3": top3,
        "overlap_517_vs_simple": overlap.to_dict("records"),
        "importance_vs_meanabs_shap_spearman": rho_imp_shap,
        "shap_top5": shap_top5,
        "ablation": {
            "top5_macro_AUC": top5_auc,
            "top10_macro_AUC": top10_auc,
            "official_primary_macro_AUC": primary_auc,
            "delta_primary_minus_top5": None
            if primary_auc is None
            else float(primary_auc - top5_auc),
            "delta_primary_minus_top10": None
            if primary_auc is None
            else float(primary_auc - top10_auc),
        },
        "shap_shape": list(shap_ncf.shape),
        "legacy_shap_note": (
            "Do not use results/legacy_superseded/patient_level_R_20260810/ SHAP PDFs "
            "as official figures; they belong to the superseded disc-level/R pipeline."
        ),
        "what_this_does_not_prove": [
            "automatic vs manual segmentation",
            "causal biomarker status",
            "external validity",
        ],
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
        "split": "revision/splits/assignments.csv",
        "seconds": round(time.time() - t0, 1),
        "inputs_md5": {
            "assignments.csv": md5(SPLITS),
            "extracted_data.xlsx": md5(EXTRACTED),
            "labels.csv": md5(LABELS),
            "selected_features_3d.csv": md5(SEL3D),
            "features.csv": md5(SIMPLE),
        },
        "outputs": sorted(p.name for p in OUT.glob("C3_*")),
        "figures": sorted(p.name for p in FIG.glob("*.png")),
                "policy": (
            "v3 full C3: grade trends + vs B2 simple features + native TreeSHAP "
            "+ Top-5/10 ablation; official primary AUC remains 0.936"
        ),
    }
    (OUT / "interpretability.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # human README with this run's numbers
    ov_lines = "\n".join(
        f"- vs {r['simple_label']}: |ρ|>0.5 = {r['frac_abs_rho_gt_0.5']:.3f} "
        f"({r['n_abs_rho_gt_0.5']}/{r['n_radiomics']}), median |ρ|={r['median_abs_rho']:.3f}"
        for r in overlap.to_dict("records")
    )
    top3_lines = "\n".join(
        f"- #{t['importance_rank']} `{t['feature']}`: ρ vs grade={t['spearman_rho_vs_grade']:.3f}, "
        f"trend={t['median_trend']}, best simple={t['best_simple']} ρ={t['best_simple_rho']}"
        for t in top3
    )
    readme = f"""# 09_interpretability

517 vs grade: median |rho|={grade_sum['median_abs_rho']:.3f}.
Ablation macro AUC Top-5={top5_auc:.3f}, Top-10={top10_auc:.3f}; primary={primary_auc}.
See summary.json and concordance_table.csv.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print("\n=== C3 done ===")
    print(
        json.dumps(
            {
                "refit_517_AUC": m517["macro_AUC"],
                "top5_AUC": top5_auc,
                "top10_AUC": top10_auc,
                "primary_AUC": primary_auc,
                "median_abs_rho_vs_grade": grade_sum["median_abs_rho"],
                "frac_|rho|>0.5_vs_grade": grade_sum["frac_abs_rho_gt_0.5"],
                "imp_vs_shap_rho": rho_imp_shap,
                "top3": top3,
            },
            indent=2,
            default=str,
        )
    )
    print(f"Wrote {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
