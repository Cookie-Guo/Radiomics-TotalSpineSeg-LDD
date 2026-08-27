#!/usr/bin/env python3
"""
Paired bootstrap comparisons among the ten models on the saved test probabilities.
1000 unstratified resamples, seed 4321; writes ten_models_pairwise_*.csv.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ten_models_unified import (  # noqa: E402
    CLASSES,
    EXTRACTED,
    FEATS,
    N_BOOT,
    OUT,
    SEED,
    SPLITS,
    get_classes,
    load_matrix,
    md5,
    metrics_block,
    proba_5,
)

MODELS = [
    "DecisionTree",
    "RandomForest",
    "XGBoost",
    "Lasso",
    "Ridge",
    "NeuralNetwork",
    "MultinomialLogistic",
    "KNN",
    "NaiveBayes",
    "CatBoost",
]


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


def fit_locked(name: str, Xtr, ytr):
    rng = SEED
    if name == "DecisionTree":
        m = DecisionTreeClassifier(
            ccp_alpha=0.05, random_state=rng, class_weight="balanced"
        )
        m.fit(Xtr, ytr)
        return m
    if name == "RandomForest":
        m = RandomForestClassifier(
            n_estimators=300, max_features=10, random_state=rng,
            class_weight="balanced_subsample", n_jobs=-1,
        )
        m.fit(Xtr, ytr)
        return m
    if name == "XGBoost":
        m = XGBClassifier(
            objective="multi:softprob", num_class=5,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1, gamma=0,
            max_depth=4, learning_rate=0.05, n_estimators=300,
            random_state=rng, n_jobs=-1, verbosity=0,
        )
        m.fit(Xtr, ytr - 1)
        return m
    if name == "Lasso":
        m = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1", solver="saga", C=1.0, max_iter=5000,
                class_weight="balanced", random_state=rng,
            )),
        ])
        m.fit(Xtr, ytr)
        return m
    if name == "Ridge":
        m = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2", solver="lbfgs", C=0.1, max_iter=5000,
                class_weight="balanced", random_state=rng,
            )),
        ])
        m.fit(Xtr, ytr)
        return m
    if name == "NeuralNetwork":
        m = Pipeline([
            ("sc", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(20,), alpha=0.01, max_iter=500,
                random_state=rng, early_stopping=True,
            )),
        ])
        m.fit(Xtr, ytr)
        return m
    if name == "MultinomialLogistic":
        C = 1.0 / (1.0 + 1e-3)
        m = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2", solver="lbfgs", C=C, max_iter=5000,
                class_weight="balanced", random_state=rng,
            )),
        ])
        m.fit(Xtr, ytr)
        return m
    if name == "KNN":
        m = Pipeline([
            ("sc", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=11)),
        ])
        m.fit(Xtr, ytr)
        return m
    if name == "NaiveBayes":
        m = Pipeline([
            ("sc", StandardScaler()),
            ("clf", GaussianNB(var_smoothing=1e-6)),
        ])
        m.fit(Xtr, ytr)
        return m
    if name == "CatBoost":
        bc = np.bincount(ytr, minlength=6)[1:6].astype(float)
        w = (bc.sum() / (5.0 * np.maximum(bc, 1))).tolist()
        cw = {i + 1: w[i] for i in range(5)}
        m = CatBoostClassifier(
            depth=2, learning_rate=0.05, l2_leaf_reg=1, iterations=223,
            loss_function="MultiClass", random_seed=SEED, verbose=False,
            allow_writing_files=False, class_weights=cw,
        )
        m.fit(Xtr, ytr)
        return m
    raise ValueError(name)


def predict_pack(name: str, model, Xte):
    if name == "XGBoost":
        proba = model.predict_proba(Xte)
    else:
        proba = proba_5(model, Xte, get_classes(model))
    pred = proba.argmax(axis=1) + 1
    return pred, proba


def compare_metric(boot_a: np.ndarray, boot_b: np.ndarray, obs_a: float, obs_b: float) -> dict:
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
        "Value_Model1": obs_a,
        "Value_Model2": obs_b,
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


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df, feats = load_matrix()
    train = df[df["holdout"] == "train"].copy().reset_index(drop=True)
    test = df[df["holdout"] == "test"].copy().reset_index(drop=True)
    med = train[feats].median()
    Xtr = train[feats].fillna(med).values.astype(float)
    Xte = test[feats].fillna(med).values.astype(float)
    ytr = train["pfirrmann"].astype(int).values
    yte = test["pfirrmann"].astype(int).values

    ref = pd.read_csv(OUT / "ten_models_test_performance.csv").set_index("Model")

    store = {}
    pred_rows = []
    for name in MODELS:
        print(f"fit {name}", flush=True)
        model = fit_locked(name, Xtr, ytr)
        pred, proba = predict_pack(name, model, Xte)
        m = metrics_block(yte, pred, proba)
        ref_auc = float(ref.loc[name, "macro_AUC"])
        if abs(m["macro_AUC"] - ref_auc) > 1e-6:
            raise SystemExit(
                f"{name} AUC {m['macro_AUC']:.8f} != table {ref_auc:.8f}"
            )
        store[name] = {"pred": pred, "proba": proba, "metrics": m}
        tmp = pd.DataFrame({
            "disc_id": test["disc_id"].values,
            "patient_id": test["patient_id"].values,
            "Model": name,
            "y_true": yte,
            "y_pred": pred,
        })
        for i, g in enumerate(CLASSES):
            tmp[f"prob_{g}"] = proba[:, i]
        pred_rows.append(tmp)

    pd.concat(pred_rows, ignore_index=True).to_csv(
        OUT / "ten_models_test_predictions.csv", index=False
    )

    n = len(yte)
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]

    boot = {name: {"macro_AUC": np.empty(N_BOOT), "macro_sensitivity": np.empty(N_BOOT)}
            for name in MODELS}
    for b, idx in enumerate(boot_idx):
        yb = yte[idx]
        for name in MODELS:
            mb = metrics_block(yb, store[name]["pred"][idx], store[name]["proba"][idx])
            boot[name]["macro_AUC"][b] = mb["macro_AUC"]
            boot[name]["macro_sensitivity"][b] = mb["macro_sensitivity"]

    def dump_pairs(metric: str, path: Path):
        rows = []
        mat_wald = pd.DataFrame(1.0, index=MODELS, columns=MODELS)
        mat_boot = pd.DataFrame(1.0, index=MODELS, columns=MODELS)
        for i, a in enumerate(MODELS):
            for b in MODELS[i + 1:]:
                rec = compare_metric(
                    boot[a][metric], boot[b][metric],
                    store[a]["metrics"][metric], store[b]["metrics"][metric],
                )
                rec["Model1"] = a
                rec["Model2"] = b
                rec["Metric"] = metric
                rows.append(rec)
                mat_wald.loc[a, b] = rec["P_Value"]
                mat_wald.loc[b, a] = rec["P_Value"]
                mat_boot.loc[a, b] = rec["P_bootstrap"]
                mat_boot.loc[b, a] = rec["P_bootstrap"]
        out = pd.DataFrame(rows).sort_values("P_Value")
        cols = [
            "Model1", "Model2", "Metric", "Value_Model1", "Value_Model2",
            "Difference", "CI_Lower", "CI_Upper", "SE",
            "P_Value", "Significance", "P_bootstrap", "Significance_bootstrap",
            "CI_excludes_0",
        ]
        out[cols].to_csv(path, index=False)
        return out, mat_wald, mat_boot

    auc_df, auc_pw, auc_pb = dump_pairs(
        "macro_AUC", OUT / "ten_models_pairwise_auc.csv"
    )
    sens_df, sens_pw, sens_pb = dump_pairs(
        "macro_sensitivity", OUT / "ten_models_pairwise_sensitivity.csv"
    )
    auc_pw.to_csv(OUT / "ten_models_pairwise_auc_p_matrix.csv")
    auc_pb.to_csv(OUT / "ten_models_pairwise_auc_p_bootstrap_matrix.csv")
    sens_pw.to_csv(OUT / "ten_models_pairwise_sensitivity_p_matrix.csv")

    vs_rows = []
    for name in MODELS:
        if name == "CatBoost":
            continue
        rec = compare_metric(
            boot[name]["macro_AUC"], boot["CatBoost"]["macro_AUC"],
            store[name]["metrics"]["macro_AUC"],
            store["CatBoost"]["metrics"]["macro_AUC"],
        )
        rec_s = compare_metric(
            boot[name]["macro_sensitivity"], boot["CatBoost"]["macro_sensitivity"],
            store[name]["metrics"]["macro_sensitivity"],
            store["CatBoost"]["metrics"]["macro_sensitivity"],
        )
        vs_rows.append({
            "Model": name,
            "AUC": rec["Value_Model1"],
            "AUC_CatBoost": rec["Value_Model2"],
            "AUC_delta_vs_CatBoost": rec["Difference"],
            "AUC_CI_low": rec["CI_Lower"],
            "AUC_CI_high": rec["CI_Upper"],
            "AUC_P_wald": rec["P_Value"],
            "AUC_P_bootstrap": rec["P_bootstrap"],
            "AUC_sig": rec["Significance"],
            "Sens": rec_s["Value_Model1"],
            "Sens_CatBoost": rec_s["Value_Model2"],
            "Sens_delta_vs_CatBoost": rec_s["Difference"],
            "Sens_CI_low": rec_s["CI_Lower"],
            "Sens_CI_high": rec_s["CI_Upper"],
            "Sens_P_wald": rec_s["P_Value"],
            "Sens_P_bootstrap": rec_s["P_bootstrap"],
            "Sens_sig": rec_s["Significance"],
        })
    vs = pd.DataFrame(vs_rows).sort_values("AUC", ascending=False)
    vs.to_csv(OUT / "ten_models_pairwise_vs_catboost.csv", index=False)

    # top tier: not significantly different from CatBoost on AUC (wald p>=0.05)
    tied = ["CatBoost"] + vs.loc[vs["AUC_P_wald"] >= 0.05, "Model"].tolist()
    beaten = vs.loc[vs["AUC_P_wald"] < 0.05, "Model"].tolist()
    # models that beat CatBoost point estimate
    higher = vs.loc[vs["AUC_delta_vs_CatBoost"] > 0, "Model"].tolist()

    meta = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "bootstrap": "paired unstratified disc-level; same indices across models",
        "p_primary": "Wald 2*(1-Phi(|d|/SE)) matching original Bootstrap_*_Comparison.csv",
        "p_secondary": "two-sided bootstrap tail 2*min(P(d<=0), P(d>=0))",
        "metric_auc": "one-vs-rest macro AUC (same as primary 0.936)",
        "n_test": int(n),
        "seconds": round(time.time() - t0, 1),
        "auc_tied_with_catboost_p_ge_0.05": tied,
        "auc_significantly_different_from_catboost": beaten,
        "auc_point_estimate_higher_than_catboost": higher,
        "catboost_macro_AUC": float(store["CatBoost"]["metrics"]["macro_AUC"]),
        "rf_macro_AUC": float(store["RandomForest"]["metrics"]["macro_AUC"]),
        "inputs_md5": {
            "assignments.csv": md5(SPLITS),
            "selected_features_3d.csv": md5(FEATS),
            "extracted_data.xlsx": md5(EXTRACTED),
        },
    }
    (OUT / "ten_models_pairwise.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(vs.to_string(index=False))
    print("tied (p_wald>=0.05):", tied)
    print("different:", beaten)
    print(f"Wrote pairwise tables in {OUT} ({meta['seconds']}s)")


if __name__ == "__main__":
    main()
