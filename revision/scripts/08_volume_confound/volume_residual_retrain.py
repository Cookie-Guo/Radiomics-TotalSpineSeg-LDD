#!/usr/bin/env python3
"""
Sensitivity: residualize the 517 features on MeshVolume (train-only coefficients) and refit CatBoost.
Does not replace the primary 0.936.
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
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from volume_confounding import (  # noqa: E402
    CB_PARAMS,
    EXTRACTED,
    LABELS,
    OUT,
    PRIMARY_PERF,
    SEED,
    SEL3D,
    SPLITS,
    VOLUME_COL,
    bootstrap_ci,
    load_merged,
    md5,
    metrics_block,
)


def residualize(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], vol_col: str):
    vtr = pd.to_numeric(train[vol_col], errors="coerce")
    vte = pd.to_numeric(test[vol_col], errors="coerce")
    vmed = float(vtr.median())
    vtr = vtr.fillna(vmed).to_numpy().reshape(-1, 1)
    vte = vte.fillna(vmed).to_numpy().reshape(-1, 1)

    Xtr = np.zeros((len(train), len(feats)), dtype=float)
    Xte = np.zeros((len(test), len(feats)), dtype=float)
    slopes = []
    intercepts = []
    r2s = []
    for j, f in enumerate(feats):
        ytr = pd.to_numeric(train[f], errors="coerce")
        yte = pd.to_numeric(test[f], errors="coerce")
        med = float(ytr.median())
        ytr = ytr.fillna(med).to_numpy()
        yte = yte.fillna(med).to_numpy()
        lr = LinearRegression()
        lr.fit(vtr, ytr)
        Xtr[:, j] = ytr - lr.predict(vtr)
        Xte[:, j] = yte - lr.predict(vte)
        slopes.append(float(lr.coef_[0]))
        intercepts.append(float(lr.intercept_))
        r2s.append(float(lr.score(vtr, ytr)))
    keep = [j for j in range(len(feats)) if float(np.std(Xtr[:, j])) > 1e-12]
    return Xtr[:, keep], Xte[:, keep], [feats[j] for j in keep], slopes, intercepts, r2s


def fit_eval(Xtr, ytr, Xte, yte, config: str, n_feat: int) -> dict:
    bc = np.bincount(ytr, minlength=6)[1:6].astype(float)
    w = (bc.sum() / (5.0 * np.maximum(bc, 1))).tolist()
    cw = {i + 1: w[i] for i in range(5)}
    clf = CatBoostClassifier(**CB_PARAMS, class_weights=cw)
    clf.fit(Xtr, ytr)
    pr = clf.predict_proba(Xte)
    proba = np.zeros((len(yte), 5))
    for j, c in enumerate(clf.classes_):
        proba[:, int(c) - 1] = pr[:, j]
    pred = proba.argmax(1) + 1
    m = metrics_block(yte, pred, proba)
    m.update(bootstrap_ci(yte, pred, proba))
    m["config"] = config
    m["n_features"] = n_feat
    m["n_train"] = int(len(ytr))
    m["n_test"] = int(len(yte))
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load_merged()
    feats = pd.read_csv(SEL3D)["Feature"].tolist()
    feats = [f for f in feats if f in df.columns and f != VOLUME_COL]
    train = df[df["holdout"] == "train"].copy()
    test = df[df["holdout"] == "test"].copy()
    ytr = train["pfirrmann"].astype(int).to_numpy()
    yte = test["pfirrmann"].astype(int).to_numpy()

    # unadjusted sanity
    med = train[feats].apply(pd.to_numeric, errors="coerce").median()
    Xtr0 = train[feats].apply(pd.to_numeric, errors="coerce").fillna(med).to_numpy()
    Xte0 = test[feats].apply(pd.to_numeric, errors="coerce").fillna(med).to_numpy()
    m0 = fit_eval(Xtr0, ytr, Xte0, yte, "unadjusted_517_sanity", len(feats))

    Xtr, Xte, kept, slopes, intercepts, r2s = residualize(train, test, feats, VOLUME_COL)
    m1 = fit_eval(Xtr, ytr, Xte, yte, "volume_residual_517", len(kept))

    primary = pd.read_csv(PRIMARY_PERF)
    p3 = float(primary.loc[primary["config"] == "3D_primary", "macro_AUC"].iloc[0])
    rows = [m0, m1]
    out = pd.DataFrame(rows)
    out["primary_3d_macro_AUC"] = p3
    out["delta_vs_primary"] = out["macro_AUC"] - p3
    out.to_csv(OUT / "volume_residual_retrain.csv", index=False)
    pd.DataFrame(
        {"feature": feats, "slope_vs_MeshVolume": slopes, "intercept": intercepts, "train_r2": r2s}
    ).to_csv(OUT / "volume_residual_coefficients.csv", index=False)

    meta = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "seed": SEED,
        "volume": VOLUME_COL,
        "residualization": "OLS feature ~ MeshVolume, coefficients from train only",
        "role": "sensitivity; does not replace primary 0.936",
        "n_features_input": len(feats),
        "n_features_after_residual": len(kept),
        "unadjusted_macro_AUC": m0["macro_AUC"],
        "residual_macro_AUC": m1["macro_AUC"],
        "primary_3d_macro_AUC": p3,
        "seconds": round(time.time() - t0, 1),
        "inputs_md5": {
            "assignments.csv": md5(SPLITS),
            "selected_features_3d.csv": md5(SEL3D),
            "extracted_data.xlsx": md5(EXTRACTED),
            "labels.csv": md5(LABELS),
        },
    }
    (OUT / "volume_residual_retrain.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out[["config", "macro_AUC", "macro_AUC_ci_low", "macro_AUC_ci_high", "accuracy", "kappa_quadratic", "delta_vs_primary"]].to_string(index=False))
    print(f"Wrote {OUT / 'volume_residual_retrain.csv'} ({meta['seconds']}s)")


if __name__ == "__main__":
    main()
