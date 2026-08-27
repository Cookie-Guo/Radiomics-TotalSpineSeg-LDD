#!/usr/bin/env python3
"""
ICC(2,1) across original / erode-1 / dilate-1 mask conditions.
Reads results/07_perturbation_icc/features_{original,erode1,dilate1}.csv;
writes feature_icc.csv and summary.json.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "07_perturbation_icc"
PRIMARY_SEL = ROOT / "results" / "02_primary" / "selected_features_3d.csv"
CONDITIONS = ("original", "erode1", "dilate1")
META_COLS = {
    "disc_id", "patient_id", "level", "pfirrmann", "condition", "n_mask_voxels", "error",
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def icc_2_1(X: np.ndarray) -> float:
    """
    ICC(A,1) / ICC(2,1): two-way random effects, absolute agreement, single measure.
    X: shape (n_subjects, k_raters), no NaN.
    Shrout & Fleiss (1979) / McGraw & Wong.
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if n < 3 or k < 2:
        return float("nan")
    mean_un = X.mean()
    mean_n = X.mean(axis=1)
    mean_k = X.mean(axis=0)
    ss_r = k * np.sum((mean_n - mean_un) ** 2)
    ss_c = n * np.sum((mean_k - mean_un) ** 2)
    ss_t = np.sum((X - mean_un) ** 2)
    ss_e = ss_t - ss_r - ss_c
    ms_r = ss_r / (n - 1)
    ms_c = ss_c / (k - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float((ms_r - ms_e) / denom)


def feature_class(name: str) -> str:
    # e.g. original_firstorder_Mean, wavelet-LLH_glcm_...
    parts = name.split("_")
    if len(parts) < 2:
        return "other"
    # imageType_featureClass_feature
    # original_shape_Sphericity
    if parts[0] == "original" or parts[0].startswith("log") or parts[0].startswith("wavelet") \
            or parts[0] in ("square", "squareroot", "logarithm", "exponential", "gradient") \
            or parts[0].startswith("lbp"):
        # find class token
        for tok in ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
            if tok in parts:
                return tok
    for tok in ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        if tok in name:
            return tok
    return "other"


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for c in CONDITIONS:
        p = OUT / f"features_{c}.csv"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        dfs[c] = pd.read_csv(p)
        print(f"loaded {c}: {len(dfs[c])} rows, {dfs[c].shape[1]} cols")

    # intersect disc_ids present in all conditions
    ids = set(dfs["original"]["disc_id"].astype(str))
    for c in CONDITIONS[1:]:
        ids &= set(dfs[c]["disc_id"].astype(str))
    ids = sorted(ids)
    print(f"discs in all 3 conditions: {len(ids)}")

    # align matrices
    for c in CONDITIONS:
        d = dfs[c].copy()
        d["disc_id"] = d["disc_id"].astype(str)
        d = d[d["disc_id"].isin(ids)].drop_duplicates("disc_id")
        d = d.set_index("disc_id").loc[ids]
        dfs[c] = d

    feat_cols = [
        c for c in dfs["original"].columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(dfs["original"][c])
    ]
    # only features present in all
    for cond in CONDITIONS[1:]:
        feat_cols = [f for f in feat_cols if f in dfs[cond].columns]
    print(f"numeric features: {len(feat_cols)}")

    rows = []
    for f in feat_cols:
        mats = []
        ok = True
        for cond in CONDITIONS:
            v = pd.to_numeric(dfs[cond][f], errors="coerce").to_numpy(dtype=float)
            mats.append(v)
            if np.isnan(v).all():
                ok = False
        if not ok:
            rows.append({"feature": f, "ICC": np.nan, "n": 0, "feature_class": feature_class(f)})
            continue
        X = np.column_stack(mats)
        # drop subjects with any nan
        msk = np.isfinite(X).all(axis=1)
        X = X[msk]
        # drop near-constant features
        if X.shape[0] < 10 or np.nanstd(X) < 1e-12:
            icc = float("nan")
        else:
            icc = icc_2_1(X)
        rows.append({
            "feature": f,
            "ICC": icc,
            "n": int(X.shape[0]),
            "k_conditions": 3,
            "feature_class": feature_class(f),
            "pass_0.75": bool(np.isfinite(icc) and icc > 0.75),
            "pass_0.90": bool(np.isfinite(icc) and icc > 0.90),
        })

    icc_df = pd.DataFrame(rows)
    icc_df.to_csv(OUT / "feature_icc.csv", index=False)

    valid = icc_df[np.isfinite(icc_df["ICC"])]
    summary = {
        "n_features_evaluated": int(len(icc_df)),
        "n_features_valid_icc": int(len(valid)),
        "n_discs": int(len(ids)),
        "conditions": list(CONDITIONS),
        "icc_gt_0.75": {
            "n": int((valid["ICC"] > 0.75).sum()),
            "proportion": float((valid["ICC"] > 0.75).mean()) if len(valid) else None,
        },
        "icc_gt_0.90": {
            "n": int((valid["ICC"] > 0.90).sum()),
            "proportion": float((valid["ICC"] > 0.90).mean()) if len(valid) else None,
        },
        "icc_median": float(valid["ICC"].median()) if len(valid) else None,
        "icc_mean": float(valid["ICC"].mean()) if len(valid) else None,
        "icc_q25": float(valid["ICC"].quantile(0.25)) if len(valid) else None,
        "icc_q75": float(valid["ICC"].quantile(0.75)) if len(valid) else None,
        "by_class": {},
    }
    for cls, g in valid.groupby("feature_class"):
        summary["by_class"][cls] = {
            "n": int(len(g)),
            "prop_gt_0.75": float((g["ICC"] > 0.75).mean()),
            "median_icc": float(g["ICC"].median()),
        }

    # primary 517
    if PRIMARY_SEL.exists():
        sel = pd.read_csv(PRIMARY_SEL)["Feature"].astype(str).tolist()
        sub = icc_df[icc_df["feature"].isin(sel)].copy()
        # name matching: selected may use slightly different prefixes
        if len(sub) < len(sel) * 0.5:
            # try case-insensitive / strip
            icc_map = {f.lower(): f for f in icc_df["feature"]}
            matched = []
            for s in sel:
                if s in set(icc_df["feature"]):
                    matched.append(s)
                elif s.lower() in icc_map:
                    matched.append(icc_map[s.lower()])
            sub = icc_df[icc_df["feature"].isin(matched)].copy()
        sub.to_csv(OUT / "icc_on_primary517.csv", index=False)
        v517 = sub[np.isfinite(sub["ICC"])]
        summary["primary_517"] = {
            "n_matched": int(len(sub)),
            "n_valid": int(len(v517)),
            "prop_gt_0.75": float((v517["ICC"] > 0.75).mean()) if len(v517) else None,
            "prop_gt_0.90": float((v517["ICC"] > 0.90).mean()) if len(v517) else None,
            "median_icc": float(v517["ICC"].median()) if len(v517) else None,
            "mean_icc": float(v517["ICC"].mean()) if len(v517) else None,
        }

    summary["elapsed_sec"] = round(time.time() - t0, 2)
    summary["timestamp"] = datetime.now().isoformat(timespec="seconds")
    summary["inputs_md5"] = {f"features_{c}.csv": md5(OUT / f"features_{c}.csv") for c in CONDITIONS}

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # methods note
    note = f"""# Mask-perturbation ICC

ICC(2,1) across original / erode-1-voxel / dilate-1-voxel masks on the same image.
n_discs={summary['n_discs']}; n_features={summary['n_features_valid_icc']};
ICC>0.75 proportion={summary['icc_gt_0.75']['proportion']:.3f}; median={summary['icc_median']:.3f}.
"""
    if "primary_517" in summary and summary["primary_517"].get("prop_gt_0.75") is not None:
        p = summary["primary_517"]
        note += (
            f"Primary 517 subset: matched={p['n_matched']}, "
            f"ICC>0.75={p['prop_gt_0.75']:.3f}, median={p['median_icc']:.3f}.\n"
        )
    note += "Limits: no repeat scans; perturbation is morphological +/-1 voxel.\n"

    (OUT / "computational_repro_note.md").write_text(note, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Done {summary['elapsed_sec']}s")


if __name__ == "__main__":
    main()
