#!/usr/bin/env python3
"""
Build de-identified labels.csv and original_split.csv from extracted_data.xlsx.
Writes revision/data/labels.csv, original_split.csv and labels.meta.json.
The offline name-to-ID map is written only if PHI_MAP is set.
"""

from __future__ import annotations
import os

import hashlib
import json
import platform
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PHI_DIR = Path(os.environ.get("PHI_MAP", "<not published>")).parent if os.environ.get("PHI_MAP") else Path("<not published>")

EXTRACTED = DATA / "extracted_data.xlsx"
TRAIN = DATA / "data.train.xlsx"
TEST = DATA / "data.test.xlsx"

N_DISCS, N_PATIENTS, N_TRAIN, N_TEST = 630, 210, 504, 126
GRADE_COUNTS = {1: 30, 2: 178, 3: 167, 4: 228, 5: 27}

LEVEL_MAP = {"L3-4": "L3-L4", "L4-5": "L4-L5", "L5-S1": "L5-S1"}
LEVELS = ["L3-L4", "L4-L5", "L5-S1"]
DISC_ID_RE = re.compile(r"^P\d{3}_(L3-L4|L4-L5|L5-S1)$")

ANOMALY_BY_PID = {"P108": ("T2", "L5-S1")}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def split_mask(m: str) -> tuple[str, str]:
    """Split MASK on the last underscore into name / original level token."""
    i = m.rfind("_")
    if i < 0:
        raise AssertionError(f"[error] MASK has no underscore：{m!r}")
    return m[:i].strip(), m[i + 1:].strip()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    ex = pd.read_excel(EXTRACTED, usecols=[0, 1])
    ex.columns = ["MASK", "disc_degree"]
    if len(ex) != N_DISCS:
        raise AssertionError(f"[error] extracted_data.xlsx should have {N_DISCS}  rows, got {len(ex)}")
    if ex["MASK"].duplicated().any():
        raise AssertionError("[error] duplicate MASK values")

    parsed = ex["MASK"].map(split_mask)
    ex["name"] = [a for a, _ in parsed]
    ex["raw_level"] = [b for _, b in parsed]

    order = list(dict.fromkeys(ex["name"]))
    if len(order) != N_PATIENTS:
        raise AssertionError(f"[error] unique names should be {N_PATIENTS}, got {len(order)}")
    pid = {n: f"P{i:03d}" for i, n in enumerate(order, 1)}
    ex["patient_id"] = ex["name"].map(pid)

    anomalies = []
    def to_level(r) -> str:
        raw = r["raw_level"]
        if raw in LEVEL_MAP:
            return LEVEL_MAP[raw]
        spec = ANOMALY_BY_PID.get(r["patient_id"])
        if spec and raw == spec[0]:
            anomalies.append({
                "orig_mask_anonymized": f"{r['patient_id']}_{raw}",
                "patient_id": r["patient_id"],
                "orig_level_token": raw,
                "assigned_level": spec[1],
                "inference_basis": "The other two discs for this patient are L3-4 and L4-5; cohort L5-S1 count is short by 1"
                                   "(L3-4=210, L4-5=210, L5-S1=209, T2=1); assigned L5-S1 by exclusion",
                "evidence_type": "inferred by exclusion, not a direct label",
            })
            return spec[1]
        raise AssertionError(f"[error] unrecognised level token {raw!r}（MASK={r['MASK']!r}）")

    ex["level"] = ex.apply(to_level, axis=1)
    ex["pfirrmann"] = ex["disc_degree"].astype(int)
    ex["disc_id"] = ex["patient_id"] + "_" + ex["level"]

    labels = ex[["disc_id", "patient_id", "level", "pfirrmann"]].copy()

    if len(labels) != N_DISCS:
        raise AssertionError(f"[error] labels rows {len(labels)} != {N_DISCS}")
    if labels["patient_id"].nunique() != N_PATIENTS:
        raise AssertionError(f"[error] n patients {labels['patient_id'].nunique()} != {N_PATIENTS}")
    per = labels.groupby("patient_id").size()
    if not (per == 3).all():
        raise AssertionError(f"[error] {(per != 3).sum()} patients do not have 3 discs")
    bad = labels.groupby("patient_id")["level"].agg(lambda s: sorted(s) != LEVELS)
    if bad.any():
        raise AssertionError(f"[error] {int(bad.sum())} patients do not have level set {LEVELS}")
    if labels["disc_id"].duplicated().any():
        raise AssertionError("[error] duplicate disc_id")
    ill = labels.loc[~labels["disc_id"].map(lambda s: bool(DISC_ID_RE.match(s))), "disc_id"]
    if len(ill):
        raise AssertionError(f"[error] disc_id format invalid：{list(ill)[:5]}")
    if not labels["pfirrmann"].between(1, 5).all():
        raise AssertionError("[error] pfirrmann out of range")
    got = labels["pfirrmann"].value_counts().sort_index().to_dict()
    if got != GRADE_COUNTS:
        raise AssertionError(f"[error] grade counts {got} != {GRADE_COUNTS}")

    def read_ids(p: Path, n: int, tag: str) -> pd.Series:
        d = pd.read_excel(p, usecols=[0])
        d.columns = ["id"]
        if len(d) != n:
            raise AssertionError(f"[error] {p.name} should have {n}  rows, got {len(d)}")
        return d["id"].astype(str).str.strip()

    tr, te = read_ids(TRAIN, N_TRAIN, "train"), read_ids(TEST, N_TEST, "test")
    if set(tr) & set(te):
        raise AssertionError(f"[error] original train/test share {len(set(tr) & set(te))}  discs")
    mask2disc = dict(zip(ex["MASK"], ex["disc_id"]))
    if set(tr) | set(te) != set(ex["MASK"]):
        raise AssertionError("[error] original train∪test MASK set != extracted")

    osplit = pd.DataFrame({
        "disc_id": [mask2disc[m] for m in list(tr) + list(te)],
        "original_holdout": ["train"] * len(tr) + ["test"] * len(te),
    }).sort_values("disc_id").reset_index(drop=True)
    if set(osplit["disc_id"]) != set(labels["disc_id"]) or len(osplit) != N_DISCS:
        raise AssertionError("[error] original_split disc_id does not match labels")

    j = osplit.merge(labels[["disc_id", "patient_id", "pfirrmann"]], on="disc_id")
    per_pat = j.groupby("patient_id")["original_holdout"].nunique()
    n_span = int((per_pat > 1).sum())
    leak = (j.groupby("patient_id")
              .agg(n_discs=("disc_id", "size"),
                   n_train=("original_holdout", lambda s: int((s == "train").sum())),
                   n_test=("original_holdout", lambda s: int((s == "test").sum())))
              .reset_index())
    leak["spans_both"] = (leak["n_train"] > 0) & (leak["n_test"] > 0)
    leak.to_csv(RESULTS / "A1_original_split_leakage.csv", index=False, encoding="utf-8-sig")

    variants = set()
    for n in order:
        variants |= {n, n.lower(), n.replace(" ", ""), n.replace(" ", "").lower()}

    def scan(df: pd.DataFrame, tag: str) -> int:
        txt = df.to_csv(index=False)
        hits = sorted(v for v in variants if v and v in txt)
        if hits:
            raise AssertionError(f"[error] {tag} residual names {len(hits)} : {hits[:5]}")
        non_ascii = sorted({c for c in txt if ord(c) > 127})
        if non_ascii:
            raise AssertionError(f"[error] {tag} contains non-ASCII：{non_ascii[:10]}")
        return 0

    scan(labels, "labels.csv")
    scan(osplit, "original_split.csv")

    phi_out = os.environ.get("PHI_MAP")
    if phi_out:
        Path(phi_out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"patient_id": [pid[n] for n in order],
                      "name": order,
                      "first_row_index": [int(ex.index[ex["name"] == n][0]) + 1 for n in order]}
                     ).to_excel(phi_out, index=False)

    labels.to_csv(DATA / "labels.csv", index=False, encoding="ascii")
    osplit.to_csv(DATA / "original_split.csv", index=False, encoding="ascii")
    pd.DataFrame(anomalies).to_csv(RESULTS / "A1_anomalies.csv", index=False, encoding="utf-8-sig")

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "seed": None,
        "seed_reason": "deterministic; no random component",
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "id_rule": "sequential anonymous identifiers",
        "level_map": LEVEL_MAP,
        "anomalies": anomalies,
        "n_discs": int(len(labels)), "n_patients": int(labels["patient_id"].nunique()),
        "original_split": {"train": N_TRAIN, "test": N_TEST,
                           "patients_spanning_both_sets": n_span,
                           "patients_total": N_PATIENTS},
        "inputs": {p.name: {"md5": md5(p)} for p in (EXTRACTED, TRAIN, TEST)},
        "phi_note": "The name-to-ID map is held offline and is not published.",
    }
    (DATA / "labels.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== labels.csv ===")
    print(f"  {len(labels)} rows / {labels['patient_id'].nunique()} patients / 3 discs each")
    print(f"  grade counts {got}")
    print(f"  anomalies {len(anomalies)} -> results/A1_anomalies.csv")
    print("\n=== original_split.csv ===")
    print(f"  train {N_TRAIN} / test {N_TEST} (aligned with labels)")
    print(f"  >>> 210 patients; {n_span} have discs in both original train and test"
          f"（{100 * n_span / N_PATIENTS:.1f}%）")
    print("\nname scan: labels.csv and original_split.csv are ASCII-only")
    print(f"\nwrote {DATA / 'labels.csv'}")
    print(f"wrote {DATA / 'original_split.csv'}")
    if os.environ.get("PHI_MAP"):
        print(f"wrote offline name-to-ID map")


if __name__ == "__main__":
    main()
