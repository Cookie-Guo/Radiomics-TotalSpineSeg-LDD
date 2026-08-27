#!/usr/bin/env python3
"""
Extract mid-sagittal 2D PyRadiomics features (force2D, no 3 mm resampling).
Reads splits/assignments.csv and N4-corrected volumes under IMAGE_ROOT.
Writes revision/results/03_compare_2d/features_2d.csv.
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

import radiomics
from radiomics import featureextractor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_common"))
from anonymize import Redactor  # noqa: E402

radiomics.setVerbosity(60)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]  # revision/
RESULTS = ROOT / "results" / "03_compare_2d"
DATA = ROOT / "data"
SPLITS = ROOT / "splits" / "assignments.csv"
VARIANTS = ROOT / "results" / "A2_acquisition_variants.csv"
RAD = Path(os.environ.get("IMAGE_ROOT", "<image_root>"))

LEVEL_TO_FOLDER = {"L3-L4": "L3-4", "L4-L5": "L4-5", "L5-S1": "L5-S1"}
SEED = 4321


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def resolve_image_mask(pid: str, level_std: str, batch: str, series: str, red: Redactor) -> tuple[Path, Path]:
    """Locate N4 image and level mask. Filenames may contain names; runtime only."""
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

    def score(p: Path) -> tuple[int, int]:
        stem = p.stem
        s_norm = stem.replace(name, "").strip("_")
        ser_norm = series.rstrip("_").rstrip(".")
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
    img_candidates = [
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
        img_dir / f"{name} _T2W_TSE_CLEAR_301__0000.nii.gz",
    ]
    for img in img_candidates:
        if img.exists():
            return img, msk
    loose = sorted(p for p in img_dir.glob("*.nii.gz") if p.name.startswith(name))
    if len(loose) == 1:
        return loose[0], msk
    ser = ser_clean
    hit = [p for p in loose if ser in p.name or stem_clean in p.name]
    if len(hit) == 1:
        return hit[0], msk
    if len(loose) > 1 and hit:
        return hit[0], msk
    raise FileNotFoundError(f"image not found for {pid} series={series} batch={batch}")


def mid_sagittal_mask(mask: sitk.Image) -> tuple[sitk.Image, dict]:
    """Keep the largest-area slice along the shortest axis."""
    arr = sitk.GetArrayFromImage(mask)  # z,y,x
    size = mask.GetSize()  # x,y,z
    sitk_axis = int(np.argmin(size))
    np_axis = (2, 1, 0)[sitk_axis]
    n = arr.shape[np_axis]
    areas = np.array(
        [(np.take(arr, i, axis=np_axis) > 0).sum() for i in range(n)],
        dtype=np.int64,
    )
    if areas.max() <= 0:
        raise ValueError("empty mask")
    best = int(areas.argmax())
    new = np.zeros_like(arr)
    idx = [slice(None)] * 3
    idx[np_axis] = best
    new[tuple(idx)] = arr[tuple(idx)]
    out = sitk.GetImageFromArray(new)
    out.CopyInformation(mask)
    meta = {
        "sitk_axis": sitk_axis,
        "np_axis": np_axis,
        "slice_index": best,
        "slice_area_voxels": int(areas[best]),
        "axis_size": int(size[sitk_axis]),
        "image_size_xyz": [int(x) for x in size],
    }
    return out, meta


def roi_intensity_stats(img: sitk.Image, msk: sitk.Image) -> dict:
    ia = sitk.GetArrayFromImage(img).astype(np.float64)
    ma = sitk.GetArrayFromImage(msk) > 0
    if not ma.any():
        return {"n": 0}
    v = ia[ma]
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "max": float(v.max()),
        "p1": float(np.percentile(v, 1)),
        "p99": float(np.percentile(v, 99)),
        "mean": float(v.mean()),
    }


def calibrate_binwidth(assignments: pd.DataFrame, variants: pd.DataFrame, red: Redactor, n_sample: int = 30) -> tuple[float, dict]:
    """Estimate binWidth on train discs after normalisation; target ~64 bins."""
    train = assignments[assignments["holdout"] == "train"].copy()
    rng = np.random.default_rng(SEED)
    pick = train.sample(n=min(n_sample, len(train)), random_state=SEED)
    vmap = variants.set_index("patient_id")

    ranges = []
    ok = 0
    for _, r in pick.iterrows():
        try:
            row = vmap.loc[r["patient_id"]]
            img_p, msk_p = resolve_image_mask(
                r["patient_id"], r["level"], row["batch"], str(row["series"]), red
            )
            img = sitk.ReadImage(str(img_p))
            msk = sitk.ReadImage(str(msk_p))
            msk2, _ = mid_sagittal_mask(msk)
            ia = sitk.GetArrayFromImage(img).astype(np.float64)
            ma = sitk.GetArrayFromImage(msk2) > 0
            # pyradiomics normalize uses whole image by default
            mu, sd = ia.mean(), ia.std()
            if sd < 1e-8:
                continue
            scale = 100.0
            vn = (ia - mu) / sd * scale
            vv = vn[ma]
            if vv.size < 10:
                continue
            ranges.append(float(np.percentile(vv, 99) - np.percentile(vv, 1)))
            ok += 1
        except Exception:  # noqa: BLE001
            continue

    if not ranges:
        raise RuntimeError("binWidth calibration failed: no valid train samples")
    med_range = float(np.median(ranges))
    target_bins = 64
    bw = max(med_range / target_bins, 0.5)
    exp_bins = med_range / bw
    if exp_bins > 128:
        bw = med_range / 128
    if exp_bins < 16:
        bw = max(med_range / 16, 0.1)
    info = {
        "n_ok": ok,
        "median_p1_p99_range_norm": med_range,
        "binWidth": float(bw),
        "expected_bins_at_median": float(med_range / bw),
        "normalizeScale": 100.0,
        "target_bins": target_bins,
    }
    return float(bw), info


def build_extractor(bin_width: float, force2d_dim: int) -> featureextractor.RadiomicsFeatureExtractor:
    settings = {
        "normalize": True,
        "normalizeScale": 100,
        "interpolator": "sitkBSpline",
        "padDistance": 10,
        "label": 1,
        "geometryTolerance": 1e-4,
        "binWidth": bin_width,
        "force2D": True,
        "force2Ddimension": int(force2d_dim),
    }
    ex = featureextractor.RadiomicsFeatureExtractor(**settings)
    ex.disableAllFeatures()
    for cls in ("shape2D", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        try:
            ex.enableFeatureClassByName(cls)
        except Exception:  # noqa: BLE001
            pass
    ex.enableImageTypeByName("Original")
    ex.enableImageTypeByName("LoG", customArgs={"sigma": [3.0, 4.0, 5.0]})
    ex.enableImageTypeByName("Wavelet")
    for t in ("Square", "SquareRoot", "Logarithm", "Exponential", "Gradient"):
        ex.enableImageTypeByName(t)
    ex.addProvenance(False)
    return ex


def extract_one(img_p: Path, msk_p: Path, bin_width: float) -> tuple[dict, dict]:
    img = sitk.ReadImage(str(img_p))
    msk = sitk.ReadImage(str(msk_p))
    msk2, smeta = mid_sagittal_mask(msk)
    ex = build_extractor(bin_width, smeta["sitk_axis"])
    fv = ex.execute(img, msk2)
    feats = {k: float(v) for k, v in fv.items() if not k.startswith("diagnostics_")}
    smeta["n_features"] = len(feats)
    smeta["intensity"] = roi_intensity_stats(img, msk2)
    return feats, smeta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3-disc smoke run")
    ap.add_argument("--limit", type=int, default=0, help="max discs (0=all)")
    ap.add_argument("--resume", action="store_true", help="skip disc_id already written")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    red = Redactor()

    assign = pd.read_csv(SPLITS)
    variants = pd.read_csv(VARIANTS)
    assert len(assign) == 630
    vmap = variants.set_index("patient_id")

    out_feat = RESULTS / "features_2d.csv"
    out_slice = RESULTS / "B1_2d_slice_meta.csv"
    out_err = RESULTS / "extract_errors.csv"

    done = set()
    if args.resume and out_feat.exists():
        prev = pd.read_csv(out_feat)
        if "disc_id" in prev.columns:
            done = set(prev["disc_id"].astype(str))

    print("calibrating binWidth ...")
    bw, bw_info = calibrate_binwidth(assign, variants, red)
    print(f"  binWidth={bw:.4f}, expected_bins≈{bw_info['expected_bins_at_median']:.1f}, n_ok={bw_info['n_ok']}")

    rows_feat: list[dict] = []
    rows_slice: list[dict] = []
    rows_err: list[dict] = []

    if out_feat.exists() and args.resume and done:
        rows_feat = pd.read_csv(out_feat).to_dict("records")
        if out_slice.exists():
            rows_slice = pd.read_csv(out_slice).to_dict("records")

    work = assign.copy()
    if args.smoke:
        work = assign.groupby("holdout", group_keys=False).head(2).head(3)
    elif args.limit > 0:
        work = assign.head(args.limit)

    n = len(work)
    for i, (_, r) in enumerate(work.iterrows(), 1):
        disc = r["disc_id"]
        if disc in done:
            continue
        pid, level = r["patient_id"], r["level"]
        try:
            rowv = vmap.loc[pid]
            img_p, msk_p = resolve_image_mask(pid, level, rowv["batch"], str(rowv["series"]), red)
            feats, smeta = extract_one(img_p, msk_p, bw)
            rec = {"disc_id": disc, "patient_id": pid, "level": level, "pfirrmann": int(r["pfirrmann"])}
            rec.update(feats)
            rows_feat.append(rec)
            srec = {
                "disc_id": disc,
                "patient_id": pid,
                "level": level,
                "batch": rowv["batch"],
                "series": str(rowv["series"]),
                "image": red.text(str(img_p)),
                "mask": red.text(str(msk_p)),
                **{k: smeta[k] for k in ("sitk_axis", "slice_index", "slice_area_voxels", "axis_size", "n_features")},
                "image_size_xyz": json.dumps(smeta["image_size_xyz"]),
            }
            rows_slice.append(srec)
            done.add(disc)
        except Exception as e:  # noqa: BLE001
            rows_err.append({
                "disc_id": disc,
                "patient_id": pid,
                "level": level,
                "error": red.text(f"{type(e).__name__}: {e}"),
            })
            print(f"[{i}/{n}] FAIL {disc}: {type(e).__name__}: {e}")
        if i % 10 == 0 or i == n:
            print(f"[{i}/{n}] ok={len(rows_feat)} err={len(rows_err)}")
            if rows_feat:
                pd.DataFrame(rows_feat).to_csv(out_feat, index=False)
                pd.DataFrame(rows_slice).to_csv(out_slice, index=False)
            if rows_err:
                pd.DataFrame(rows_err).to_csv(out_err, index=False)

    if not rows_feat:
        raise SystemExit("no successful extractions")

    df = pd.DataFrame(rows_feat)
    red_chk = Redactor()
    blob = df.to_csv(index=False)
    hits = red_chk.check(blob)
    if hits:
        raise SystemExit(f"de-identification failed, names hit: {hits[:5]}")

    df.to_csv(out_feat, index=False)
    pd.DataFrame(rows_slice).to_csv(out_slice, index=False)
    if rows_err:
        pd.DataFrame(rows_err).to_csv(out_err, index=False)

    meta = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "pyradiomics": radiomics.__version__,
        "numpy": np.__version__,
        "SimpleITK": sitk.Version_VersionString(),
        "seed": SEED,
        "binWidth_calibration": bw_info,
        "n_requested": int(n),
        "n_success": int(len(df)),
        "n_error": int(len(rows_err)),
        "n_features_per_disc_median": float(pd.DataFrame(rows_slice)["n_features"].median()) if rows_slice else None,
        "seconds": round(time.time() - t0, 1),
        "inputs": {
            "assignments.csv": md5(SPLITS),
            "A2_acquisition_variants.csv": md5(VARIANTS),
        },
        "settings_note": {
            "force2D": True,
            "resampledPixelSpacing": None,
            "normalizeScale": 100,
            "image_types": ["Original", "LoG", "Wavelet", "Square", "SquareRoot", "Logarithm", "Exponential", "Gradient"],
            "skipped": ["LBP3D"],
        },
        "smoke": bool(args.smoke),
    }
    (RESULTS / "features_2d.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"done: {out_feat}  rows={len(df)}  features≈{meta['n_features_per_disc_median']}  {meta['seconds']}s")


if __name__ == "__main__":
    main()
