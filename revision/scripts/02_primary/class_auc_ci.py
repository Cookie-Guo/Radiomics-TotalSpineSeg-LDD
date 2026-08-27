#!/usr/bin/env python3
"""Grade-wise one-vs-rest AUC 95% CI (unstratified bootstrap, seed 4321, 1000).

Uses saved 02_primary test probabilities. Same CI policy as macro AUC.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "02_primary"
SEED = 4321
N_BOOT = 1000
CLASSES = [1, 2, 3, 4, 5]


def ovr_aucs(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    yb = label_binarize(y, classes=CLASSES)
    out = np.full(5, np.nan)
    for i in range(5):
        n_pos = int(yb[:, i].sum())
        if 0 < n_pos < len(y):
            out[i] = roc_auc_score(yb[:, i], proba[:, i])
    return out


def bootstrap_class(y: np.ndarray, proba: np.ndarray):
    rng = np.random.default_rng(SEED)
    n = len(y)
    store = [[] for _ in range(5)]
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        aucs = ovr_aucs(y[idx], proba[idx])
        for i in range(5):
            if np.isfinite(aucs[i]):
                store[i].append(aucs[i])
    lo, hi, n_ok = [], [], []
    for i in range(5):
        arr = np.asarray(store[i], float)
        n_ok.append(len(arr))
        lo.append(float(np.percentile(arr, 2.5)) if len(arr) else float("nan"))
        hi.append(float(np.percentile(arr, 97.5)) if len(arr) else float("nan"))
    return lo, hi, n_ok


def one_config(path: Path, config: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    y = df["y_true"].astype(int).values
    proba = df[[f"prob_{g}" for g in CLASSES]].values.astype(float)
    point = ovr_aucs(y, proba)
    lo, hi, n_ok = bootstrap_class(y, proba)
    yb = label_binarize(y, classes=CLASSES)
    rows = []
    for i, g in enumerate(CLASSES):
        rows.append({
            "config": config,
            "class": g,
            "AUC": float(point[i]),
            "AUC_ci_low": lo[i],
            "AUC_ci_high": hi[i],
            "n_test_pos": int(yb[:, i].sum()),
            "n_test_total": len(y),
            "n_boot_valid": n_ok[i],
        })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    d3 = one_config(OUT / "test_predictions_3d.csv", "3D_primary")
    d2 = one_config(OUT / "test_predictions_2d.csv", "2D_midsagittal")
    out = pd.concat([d3, d2], ignore_index=True)
    out.to_csv(OUT / "class_auc.csv", index=False)
    meta = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "bootstrap": "unstratified disc-level, same as primary macro AUC",
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "class_auc.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Wrote {OUT / 'class_auc.csv'}")


if __name__ == "__main__":
    main()
