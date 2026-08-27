#!/usr/bin/env python3
"""
Re-extract 3D radiomics after ±1-voxel morphological mask perturbation.
Same extractor as the primary (3×3×3 mm, binWidth=5). Requires IMAGE_ROOT.
"""

from __future__ import annotations
import os

import argparse
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
import SimpleITK as sitk
from radiomics import featureextractor
import radiomics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_common"))
from anonymize import Redactor  # noqa: E402

radiomics.setVerbosity(60)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "07_perturbation_icc"
SPLITS = ROOT / "splits" / "assignments.csv"
VARIANTS = ROOT / "results" / "audit" / "A2_acquisition_resampling" / "A2_acquisition_variants.csv"
RAD = Path(os.environ.get("IMAGE_ROOT", "<image_root>"))
SEED = 4321
CONDITIONS = ("original", "erode1", "dilate1")
MIN_VOXELS = 10
LEVEL_TO_FOLDER = {"L3-L4": "L3-4", "L4-L5": "L4-5", "L5-S1": "L5-S1"}


def resolve_image_mask(pid: str, level_std: str, batch: str, series: str, red: Redactor):
    """Locate N4 image and segment mask (runtime PHI only)."""
    name = red.name_of(pid)
    lv = LEVEL_TO_FOLDER[level_std]
    md = RAD / batch / f"{batch}_{lv}_label"
    if not md.is_dir():
        raise FileNotFoundError(f"mask dir missing: {md}")
    cands = sorted(md.glob(f"{name}_{lv}_*.nrrd"))
    if not cands:
        cands = sorted(
            p for p in md.glob(f"{name}_*.nrrd")
            if lv in p.stem or level_std.replace("-", "") in p.stem
        )
    if not cands:
        raise FileNotFoundError(f"no mask for {pid} {lv} in {batch}")
    ser_norm = series.rstrip("_").rstrip(".")

    def score(p: Path):
        stem = p.stem
        exact = int(ser_norm in stem and lv in stem)
        contain = int(ser_norm in stem or ser_norm.replace("_", "") in stem.replace("_", ""))
        return (-exact, -contain)

    cands = sorted(cands, key=score)
    msk = cands[0]
    stem_series = msk.stem
    if stem_series.startswith(f"{name}_{lv}_"):
        stem_series = stem_series[len(f"{name}_{lv}_") :]
    else:
        stem_series = stem_series[len(name) :].lstrip("_").replace(f"{lv}_", "").replace(lv, "")
        stem_series = stem_series.strip("_")
    img_dir = RAD / batch / f"{batch}_img_N4"
    ser_clean = series.rstrip("_").rstrip(".")
    stem_clean = stem_series.rstrip("_").rstrip(".")
    for img in [
        img_dir / f"{name}_{stem_series}__0000.nii.gz",
        img_dir / f"{name}_{stem_series}_0000.nii.gz",
        img_dir / f"{name}_{stem_clean}__0000.nii.gz",
        img_dir / f"{name}_{series}__0000.nii.gz",
        img_dir / f"{name}_{ser_clean}__0000.nii.gz",
        img_dir / f"{name}{stem_series}__0000.nii.gz",
        img_dir / f"{name}{stem_clean}__0000.nii.gz",
        img_dir / f"{name}{ser_clean}__0000.nii.gz",
        img_dir / f"{name} _{stem_series}__0000.nii.gz",
        img_dir / f"{name} _{ser_clean}__0000.nii.gz",
    ]:
        if img.exists():
            return img, msk
    loose = sorted(p for p in img_dir.glob("*.nii.gz") if p.name.startswith(name))
    if len(loose) == 1:
        return loose[0], msk
    hit = [p for p in loose if ser_clean in p.name or stem_clean in p.name]
    if hit:
        return hit[0], msk
    raise FileNotFoundError(f"image not found for {pid} series={series} batch={batch}")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    settings = dict(
        normalize=True,
        normalizeScale=200,
        interpolator="sitkBSpline",
        resampledPixelSpacing=[3, 3, 3],
        padDistance=10,
        label=1,
        binWidth=5,
        geometryTolerance=1e-4,
    )
    ex = featureextractor.RadiomicsFeatureExtractor(**settings)
    ex.enableImageTypeByName("Original")
    ex.enableImageTypeByName("LoG", customArgs={"sigma": [3.0, 4.0, 5.0]})
    ex.enableImageTypeByName("Wavelet")
    for t in ("Square", "SquareRoot", "Logarithm", "Exponential", "Gradient"):
        ex.enableImageTypeByName(t)
    try:
        ex.enableImageTypeByName("LBP3D")
    except Exception:  # noqa: BLE001
        pass
    ex.addProvenance(False)
    return ex


def morph_mask(mask: sitk.Image, condition: str) -> sitk.Image:
    if condition == "original":
        return mask
    # binary; radius 1 voxel in image space
    if condition == "erode1":
        out = sitk.BinaryErode(mask, (1, 1, 1))
    elif condition == "dilate1":
        out = sitk.BinaryDilate(mask, (1, 1, 1))
    else:
        raise ValueError(condition)
    # ensure label 1
    out = sitk.Cast(out > 0, sitk.sitkUInt8)
    return out


def n_mask_voxels(mask: sitk.Image) -> int:
    arr = sitk.GetArrayFromImage(mask)
    return int((arr > 0).sum())


def extract_one(ex, img: sitk.Image, mask: sitk.Image) -> dict:
    fv = ex.execute(img, mask)
    return {k: float(v) for k, v in fv.items() if not str(k).startswith("diagnostics_")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max discs (0=all)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    red = Redactor()
    assign = pd.read_csv(SPLITS)
    variants = pd.read_csv(VARIANTS).set_index("patient_id")
    assert len(assign) == 630

    work = assign.copy()
    if args.smoke:
        work = work.head(9)
    elif args.limit > 0:
        work = work.head(args.limit)

    ex = build_extractor()
    buffers = {c: [] for c in CONDITIONS}
    errors = []

    # resume: skip disc_ids already in original file
    done = set()
    out_orig = OUT / "features_original.csv"
    if args.resume and out_orig.exists():
        done = set(pd.read_csv(out_orig, usecols=["disc_id"])["disc_id"].astype(str))
        for c in CONDITIONS:
            p = OUT / f"features_{c}.csv"
            if p.exists():
                buffers[c] = pd.read_csv(p).to_dict("records")

    for i, r in work.iterrows():
        disc_id = str(r["disc_id"])
        if disc_id in done:
            continue
        pid = r["patient_id"]
        lv = r["level"]
        try:
            v = variants.loc[pid]
            batch, series = str(v["batch"]), str(v["series"])
            img_p, msk_p = resolve_image_mask(pid, lv, batch, series, red)
            img = sitk.ReadImage(str(img_p))
            msk0 = sitk.ReadImage(str(msk_p))
            msk_bin = sitk.BinaryThreshold(msk0, 1, 255, 1, 0)

            for cond in CONDITIONS:
                m = morph_mask(msk_bin, cond)
                nv = n_mask_voxels(m)
                if nv < MIN_VOXELS:
                    errors.append({
                        "disc_id": disc_id, "condition": cond,
                        "error": f"mask_too_small n={nv}",
                    })
                    continue
                try:
                    feats = extract_one(ex, img, m)
                    row = {
                        "disc_id": disc_id,
                        "patient_id": pid,
                        "level": lv,
                        "pfirrmann": int(r["pfirrmann"]),
                        "condition": cond,
                        "n_mask_voxels": nv,
                        **feats,
                    }
                    # redact string fields
                    row["patient_id"] = pid  # already anonymous id
                    buffers[cond].append(row)
                except Exception as e:  # noqa: BLE001
                    errors.append({
                        "disc_id": disc_id, "condition": cond, "error": str(e)[:200],
                    })
        except Exception as e:  # noqa: BLE001
            errors.append({"disc_id": disc_id, "condition": "all", "error": str(e)[:200]})

        if (len(buffers["original"]) % 30 == 0) or args.smoke:
            n_o = len(buffers["original"])
            print(f"  progress original_rows={n_o} elapsed={time.time()-t0:.0f}s")
            # intermediate save
            for c in CONDITIONS:
                if buffers[c]:
                    df = pd.DataFrame(buffers[c])
                    for col in df.columns:
                        if df[col].dtype == object:
                            df[col] = df[col].map(lambda x: red.text(x) if isinstance(x, str) else x)
                    df.to_csv(OUT / f"features_{c}.csv", index=False)

    for c in CONDITIONS:
        df = pd.DataFrame(buffers[c])
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(lambda x: red.text(x) if isinstance(x, str) else x)
        path = OUT / f"features_{c}.csv"
        df.to_csv(path, index=False)
        blob = path.read_text(encoding="utf-8", errors="ignore")
        hits = red.check(blob)
        if hits:
            raise SystemExit(f"PHI in {path}: {hits[:3]}")
        print(f"  wrote {path.name} n={len(df)}")

    err_df = pd.DataFrame(errors)
    err_df.to_csv(OUT / "extract_errors.csv", index=False)

    meta = {
        "script": "extract_perturbed.py",
        "seed": SEED,
        "conditions": list(CONDITIONS),
        "settings": {
            "resampledPixelSpacing": [3, 3, 3],
            "binWidth": 5,
            "normalizeScale": 200,
            "padDistance": 10,
            "normalize": True,
        },
        "morphology": "BinaryErode/Dilate kernel radius (1,1,1) voxel",
        "n_discs_requested": len(work),
        "n_per_condition": {c: len(buffers[c]) for c in CONDITIONS},
        "n_errors": len(errors),
        "smoke": bool(args.smoke),
        "elapsed_sec": round(time.time() - t0, 2),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "radiomics": radiomics.__version__,
        "inputs_md5": {
            "assignments.csv": md5(SPLITS),
            "A2_acquisition_variants.csv": md5(VARIANTS),
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT / "extract_perturbed.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta["n_per_condition"], indent=2), f"errors={len(errors)}", f"{meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
