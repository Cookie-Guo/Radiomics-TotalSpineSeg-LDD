#!/usr/bin/env python3
"""
Build the SPIDER disc inventory: conventional T2, IVD 1/2/3 → L5–S1/L4–L5/L3–L4, QC.
Reads the local SPIDER download; writes inventory.csv and qc.json.
"""

from __future__ import annotations
import os

import json
import platform
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "10_external_spider"
RAW = Path(os.environ.get("SPIDER_ROOT", "<spider_root>"))
OV = RAW / "overview.csv"
GR = RAW / "radiological_gradings.csv"

LEVEL_FROM_IVD = {1: "L5-S1", 2: "L4-L5", 3: "L3-L4"}
OK_N_VERT = {6, 7, 8}


def parse_file_name(name: str) -> tuple[int | None, str]:
    s = str(name).strip()
    if s.endswith("_t2_SPACE"):
        return int(s[: -len("_t2_SPACE")]), "t2_space"
    if s.endswith("_t2"):
        return int(s[: -len("_t2")]), "t2"
    if s.endswith("_t1"):
        return int(s[: -len("_t1")]), "t1"
    return None, "other"


def find_volume(folder: Path, stem: str) -> str:
    if not folder.exists():
        return ""
    for ext in (".mha", ".nii.gz", ".nii", ".nrrd"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return str(p)
    hits = list(folder.rglob(f"{stem}.*"))
    if hits:
        return str(hits[0])
    return ""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ov = pd.read_csv(OV)
    gr = pd.read_csv(GR)
    gr = gr.rename(columns={"Pfirrman grade": "pfirrmann", "IVD label": "ivd_label", "Patient": "patient_id"})

    parsed = ov["new_file_name"].map(parse_file_name)
    ov["patient_id"] = [a for a, _ in parsed]
    ov["contrast"] = [b for _, b in parsed]
    ov["manufacturer"] = ov["Manufacturer"].astype(str).str.replace("Philips Medical Systems", "Philips", regex=False)
    ov["manufacturer"] = ov["manufacturer"].str.replace("Philips Healthcare", "Philips", regex=False)
    ov["field_T"] = pd.to_numeric(ov["MagneticFieldStrength"], errors="coerce")

    img_dir = RAW / "images"
    msk_dir = RAW / "masks"
    # unzip may nest one extra folder
    if img_dir.exists() and not any(img_dir.glob("*.mha")) and not any(img_dir.glob("*.nii*")):
        sub = [p for p in img_dir.iterdir() if p.is_dir()]
        if len(sub) == 1:
            img_dir = sub[0]
    if msk_dir.exists() and not any(msk_dir.glob("*.mha")) and not any(msk_dir.glob("*.nii*")):
        sub = [p for p in msk_dir.iterdir() if p.is_dir()]
        if len(sub) == 1:
            msk_dir = sub[0]

    rows = []
    for r in ov.itertuples(index=False):
        pid, contrast = parse_file_name(r.new_file_name)
        if pid is None:
            continue
        stem = str(r.new_file_name)
        img = find_volume(img_dir, stem)
        msk = find_volume(msk_dir, stem)
        nvert = int(r.num_vertebrae) if pd.notna(r.num_vertebrae) else -1
        anatomy_ok = nvert in OK_N_VERT
        g = gr[gr["patient_id"] == pid]
        for ivd, level in LEVEL_FROM_IVD.items():
            gg = g[g["ivd_label"] == ivd]
            pf = int(gg["pfirrmann"].iloc[0]) if len(gg) else None
            exclude = []
            if contrast != "t2":
                exclude.append(f"contrast={contrast}")
            if not anatomy_ok:
                exclude.append(f"num_vertebrae={nvert}")
            if pf is None:
                exclude.append("missing_pfirrmann")
            elif pf not in (1, 2, 3, 4, 5):
                exclude.append(f"pfirrmann={pf}")
            files_ok = bool(img) and bool(msk)
            include_eligible = len(exclude) == 0
            include = include_eligible and files_ok
            if include_eligible and not files_ok:
                exclude.append("files_pending")
            rows.append(
                {
                    "disc_id": f"S{pid:03d}_{level}_{contrast}",
                    "patient_id": f"S{pid:03d}",
                    "spider_patient": pid,
                    "file_stem": stem,
                    "contrast": contrast,
                    "ivd_label": ivd,
                    "mask_label": 200 + ivd,
                    "mapped_level": level,
                    "level_uncertain": not anatomy_ok,
                    "num_vertebrae": nvert,
                    "num_discs": int(r.num_discs) if pd.notna(r.num_discs) else None,
                    "pfirrmann": pf,
                    "manufacturer": r.manufacturer,
                    "model_name": r.ManufacturerModelName,
                    "field_T": r.field_T,
                    "series_description": r.SeriesDescription,
                    "acq_type": r.MRAcquisitionType,
                    "slice_thickness_mm": r.SliceThickness,
                    "spacing_between_slices_mm": r.SpacingBetweenSlices,
                    "echo_time_ms": r.EchoTime,
                    "image_path": img,
                    "mask_path": msk,
                    "include_eligible": include_eligible,
                    "include_primary": include,
                    "exclude_reason": ";".join(exclude),
                }
            )

    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)

    elig = inv[inv["include_eligible"]]
    pri = inv[inv["include_primary"]]
    qc = {
        "n_overview_series": int(len(ov)),
        "n_inventory_rows": int(len(inv)),
        "by_contrast": inv.groupby("contrast")["file_stem"].nunique().to_dict(),
        "eligible_before_files": {
            "n_discs": int(len(elig)),
            "n_patients": int(elig["patient_id"].nunique()),
            "grade_n": elig["pfirrmann"].value_counts().sort_index().to_dict(),
        },
        "primary": {
            "n_discs": int(len(pri)),
            "n_patients": int(pri["patient_id"].nunique()),
            "grade_n": pri["pfirrmann"].value_counts().sort_index().to_dict(),
            "level_n": pri["mapped_level"].value_counts().to_dict(),
            "manufacturer_n": pri.drop_duplicates("patient_id")["manufacturer"].value_counts().to_dict(),
            "field_n": pri.drop_duplicates("patient_id")["field_T"].value_counts().to_dict(),
        },
        "excluded_reason_counts": inv.loc[~inv["include_primary"], "exclude_reason"]
        .value_counts()
        .head(20)
        .to_dict(),
        "mapping_rule": "IVD 1/2/3 → L5-S1 / L4-L5 / L3-L4 if lowest vertebra assumed L5; num_vertebrae in {6,7,8}",
        "space_excluded_from_primary": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "image_dir": str(img_dir),
        "mask_dir": str(msk_dir),
        "paths_resolved": bool(pri["image_path"].astype(str).str.len().gt(0).any()) if len(pri) else False,
    }
    (OUT / "qc.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    meta = {
        "script": Path(__file__).name,
        "generated_at": qc["generated_at"],
        "python": qc["python"],
        "raw": str(RAW),
        "n_primary": qc["primary"]["n_discs"],
        "n_eligible": qc["eligible_before_files"]["n_discs"],
    }
    (OUT / "inventory.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(qc["primary"], indent=2, default=str))
    print("excluded top:", qc["excluded_reason_counts"])
    print("wrote", OUT / "inventory.csv")


if __name__ == "__main__":
    main()
