#!/usr/bin/env python3
"""
Create or verify the sole patient-level split (splits/assignments.csv, seed 4321).
Every modelling script must read this file; do not call train_test_split.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labels.csv"
SPLIT_DIR = ROOT / "splits"
RESULTS = ROOT / "results"

SEED = 4321
N_FOLDS = 5
N_REPEATS = 10

REQUIRED_COLS = ["disc_id", "patient_id", "level", "pfirrmann"]


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def load_labels(path: Path = LABELS) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[error] missing {path}. generate labels.csv first.")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"[error] labels.csv missing columns: {missing}")

    df["pfirrmann"] = df["pfirrmann"].astype(int)
    if df["disc_id"].duplicated().any():
        sys.exit("[error] duplicate disc_id.")
    if not df["pfirrmann"].between(1, 5).all():
        sys.exit("[error] pfirrmann must be integers 1..5.")
    return df


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def audit(df: pd.DataFrame) -> pd.DataFrame:
    """How many patients contribute to each rare Pfirrmann grade."""
    rows = []
    for g in sorted(df["pfirrmann"].unique()):
        sub = df[df["pfirrmann"] == g]
        n_pat = sub["patient_id"].nunique()
        rows.append({
            "pfirrmann": g,
            "n_discs": len(sub),
            "n_patients_contributing": n_pat,
            "discs_per_patient": round(len(sub) / n_pat, 2) if n_pat else np.nan,
            "expected_test_discs_20pct": round(len(sub) * 0.2, 1),
            "expected_test_patients_20pct": round(n_pat * 0.2, 1),
        })
    audit_df = pd.DataFrame(rows)

    span = df.groupby("patient_id")["pfirrmann"].agg(["min", "max", "nunique"])
    span["range"] = span["max"] - span["min"]

    print("\n=== patients contributing to each Pfirrmann grade ===")
    print(audit_df.to_string(index=False))
    print("\n=== within-patient grade consistency ===")
    print(f"patients with three identical grades: {(span['nunique'] == 1).sum()} / {len(span)} "
          f"（{100 * (span['nunique'] == 1).mean():.1f}%）")
    print(f"patients with grade span <= 1: {(span['range'] <= 1).sum()} / {len(span)} "
          f"（{100 * (span['range'] <= 1).mean():.1f}%）")
    print(f"mean grade span: {span['range'].mean():.2f}")

    for g in audit_df.itertuples():
        if g.n_patients_contributing < 15:
            print(f"\n[warn] Grade {g.pfirrmann} from {g.n_patients_contributing} patients; "
                  f"expected test patients ~{g.expected_test_patients_20pct}."
                  f"Repeated grouped CV OOF is needed; a single holdout is thin for this grade.")

    RESULTS.mkdir(exist_ok=True, parents=True)
    audit_df.to_csv(RESULTS / "A1_cohort_audit.csv", index=False)
    span.to_csv(RESULTS / "A1_within_patient_grade_span.csv")
    print(f"\nwrote {RESULTS / 'A1_cohort_audit.csv'}")
    return audit_df


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def make_holdout(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Patient-level holdout: first StratifiedGroupKFold fold as test (~20%)."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, y=df["pfirrmann"], groups=df["patient_id"]))
    holdout = pd.Series("train", index=df.index, name="holdout")
    holdout.iloc[test_idx] = "test"
    return holdout


def make_repeated_cv(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Repeated grouped CV on all 630 discs so OOF covers every grade I and V disc."""
    out = {}
    for r in range(1, N_REPEATS + 1):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed + r)
        fold = np.empty(len(df), dtype=int)
        for k, (_, va) in enumerate(sgkf.split(df, y=df["pfirrmann"], groups=df["patient_id"]), 1):
            fold[va] = k
        out[f"cv_rep{r:02d}"] = fold
    return pd.DataFrame(out, index=df.index)


def freeze(seed: int = SEED) -> pd.DataFrame:
    df = load_labels()
    audit(df)

    asg = df[REQUIRED_COLS].copy()
    asg["holdout"] = make_holdout(df, seed).values
    asg = pd.concat([asg, make_repeated_cv(df, seed)], axis=1)

    verify_frame(asg)

    SPLIT_DIR.mkdir(exist_ok=True, parents=True)
    asg.to_csv(SPLIT_DIR / "assignments.csv", index=False)

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "n_folds": N_FOLDS,
        "n_repeats": N_REPEATS,
        "grouping": "patient_id",
        "stratification": "pfirrmann (approximate, via StratifiedGroupKFold)",
        "n_discs": int(len(asg)),
        "n_patients": int(asg["patient_id"].nunique()),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "note": "Sole split used by downstream scripts. Do not re-split.",
    }
    (SPLIT_DIR / "assignments.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== holdout grade counts ===")
    print(pd.crosstab(asg["pfirrmann"], asg["holdout"], margins=True).to_string())
    print(f"\ntrain patients {asg.loc[asg.holdout=='train','patient_id'].nunique()} / "
          f"test patients {asg.loc[asg.holdout=='test','patient_id'].nunique()}")
    print(f"wrote {SPLIT_DIR / 'assignments.csv'}")
    return asg


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def verify_frame(asg: pd.DataFrame) -> None:
    tr = set(asg.loc[asg.holdout == "train", "patient_id"])
    te = set(asg.loc[asg.holdout == "test", "patient_id"])
    overlap = tr & te
    if overlap:
        raise AssertionError(f"[leak] {len(overlap)} patients in both train and test: {sorted(overlap)[:10]}")

    for col in [c for c in asg.columns if c.startswith("cv_rep")]:
        bad = asg.groupby("patient_id")[col].nunique()
        if (bad > 1).any():
            raise AssertionError(f"[leak] {col}: {(bad > 1).sum()} patients split across folds.")

    print("verify ok: no patient in both sets or across folds.")


def verify_file() -> None:
    p = SPLIT_DIR / "assignments.csv"
    if not p.exists():
        sys.exit("[error] splits/assignments.csv missing; run --freeze first")
    verify_frame(pd.read_csv(p))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true", help="cohort audit only")
    ap.add_argument("--freeze", action="store_true", help="write the split")
    ap.add_argument("--verify", action="store_true", help="verify the frozen split")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    if a.audit:
        audit(load_labels())
    elif a.freeze:
        freeze(a.seed)
    elif a.verify:
        verify_file()
    else:
        ap.print_help()
