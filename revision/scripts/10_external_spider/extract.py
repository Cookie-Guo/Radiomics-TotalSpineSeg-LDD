#!/usr/bin/env python3
"""
Extract 3D PyRadiomics features from SPIDER expert masks (same parameters as primary).
Reads inventory.csv; writes results/10_external_spider/features.csv.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import radiomics

radiomics.setVerbosity(60)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "10_external_spider"
INV = OUT / "inventory.csv"
MIN_VOXELS = 10


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


def isolate_label(mask: sitk.Image, label: int) -> sitk.Image:
    arr = sitk.GetArrayFromImage(mask)
    binary = (arr == int(label)).astype(np.uint8)
    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(mask)
    return out


def n_vox(mask: sitk.Image) -> int:
    return int((sitk.GetArrayFromImage(mask) > 0).sum())


def pick_disc_mask(mask: sitk.Image, ivd: int, preferred: int) -> tuple[sitk.Image, int, int]:
    """Return binary disc mask, label used, voxel count."""
    for lab in (preferred, ivd, 100 + ivd):
        m = isolate_label(mask, lab)
        nv = n_vox(m)
        if nv >= MIN_VOXELS:
            return m, lab, nv
    return isolate_label(mask, preferred), preferred, 0


def extract_one(ex, img: sitk.Image, mask: sitk.Image) -> dict:
    fv = ex.execute(img, mask)
    return {k: float(v) for k, v in fv.items() if not str(k).startswith("diagnostics_")}


def geometry_ok(img: sitk.Image, mask: sitk.Image, tol: float = 1e-3) -> tuple[bool, str]:
    if img.GetSize() != mask.GetSize():
        return False, f"size {img.GetSize()} vs {mask.GetSize()}"
    isp = np.array(img.GetSpacing(), dtype=float)
    msp = np.array(mask.GetSpacing(), dtype=float)
    if np.max(np.abs(isp - msp)) > max(tol, 0.05 * np.max(isp)):
        return False, f"spacing {tuple(isp)} vs {tuple(msp)}"
    return True, "ok"


META_COLS = {
    "disc_id", "patient_id", "mapped_level", "pfirrmann", "manufacturer",
    "field_T", "mask_label_used", "n_voxels",
}
SEL3D = ROOT / "results" / "02_primary" / "selected_features_3d.csv"


def unique_labels(mask: sitk.Image) -> list[int]:
    arr = sitk.GetArrayFromImage(mask)
    return [int(v) for v in np.unique(arr) if int(v) != 0]


def qc_geometry(inv: pd.DataFrame, n_series: int = 10) -> dict:
    """Spot-check image/mask geometry and disc-label occupancy on n series."""
    work = inv[inv["include_primary"]].copy() if "include_primary" in inv.columns else inv
    work = work[work["image_path"].astype(str).str.len() > 0]
    stems = work.drop_duplicates("file_stem")
    if len(stems) == 0:
        work = inv[inv["image_path"].astype(str).str.len() > 0]
        stems = work.drop_duplicates("file_stem")
    sample = stems.head(n_series)
    rows = []
    n_ok = 0
    for r in sample.itertuples(index=False):
        rec = {"file_stem": r.file_stem, "ok": False, "why": "", "labels": "", "vox_201": 0, "vox_1": 0}
        try:
            img = sitk.ReadImage(str(r.image_path))
            raw = sitk.ReadImage(str(r.mask_path))
            ok, why = geometry_ok(img, raw)
            labs = unique_labels(raw)
            rec["why"] = why
            rec["labels"] = ",".join(str(x) for x in labs[:40])
            rec["n_labels"] = len(labs)
            rec["size"] = str(img.GetSize())
            rec["spacing"] = str(tuple(round(s, 4) for s in img.GetSpacing()))
            rec["vox_201"] = n_vox(isolate_label(raw, 201))
            rec["vox_202"] = n_vox(isolate_label(raw, 202))
            rec["vox_203"] = n_vox(isolate_label(raw, 203))
            rec["vox_1"] = n_vox(isolate_label(raw, 1))
            rec["has_200plus"] = any(200 <= x < 300 for x in labs)
            rec["ok"] = bool(ok)
            if ok:
                n_ok += 1
        except Exception as e:  # noqa: BLE001
            rec["why"] = str(e)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "geometry_qc.csv", index=False)
    summary = {
        "n_checked": int(len(out)),
        "n_geometry_ok": int(n_ok),
        "frac_200plus": float(out["has_200plus"].mean()) if "has_200plus" in out and len(out) else None,
        "median_vox_201": float(out["vox_201"].median()) if len(out) else None,
        "median_vox_1": float(out["vox_1"].median()) if len(out) else None,
    }
    (OUT / "geometry_qc.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("geometry QC", json.dumps(summary, indent=2))
    return summary


def write_alignment(feat_cols: list[str], tag: str) -> dict:
    if not SEL3D.exists():
        return {}
    selected = pd.read_csv(SEL3D)["Feature"].tolist()
    present = [c for c in selected if c in feat_cols]
    missing = [c for c in selected if c not in feat_cols]
    extra = [c for c in feat_cols if c not in selected]
    aln = {
        "n_selected_517": len(selected),
        "n_extracted": len(feat_cols),
        "n_present": len(present),
        "n_missing": len(missing),
        "frac_present": round(len(present) / max(len(selected), 1), 4),
        "missing_head": missing[:20],
        "n_extra": len(extra),
    }
    (OUT / f"D_alignment_{tag}.json").write_text(
        json.dumps(aln, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("alignment", json.dumps(aln, indent=2))
    return aln


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--qc-geometry", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(INV)
    if args.qc_geometry:
        qc_geometry(inv, n_series=10)
        return

    work = inv[inv["include_eligible"]].copy()
    work = work[work["image_path"].astype(str).str.len() > 0]
    work = work[work["mask_path"].astype(str).str.len() > 0]
    work = work.reset_index(drop=True)
    if args.smoke:
        # three discs, preferably one series so geometry + labels are shared
        stems = work["file_stem"].drop_duplicates().head(1)
        work = work[work["file_stem"].isin(stems)].head(3)
    elif args.limit > 0:
        work = work.head(args.limit)

    out_csv = OUT / ("D_features_smoke.csv" if args.smoke else "features.csv")
    err_csv = OUT / ("D_extract_errors_smoke.csv" if args.smoke else "D_extract_errors.csv")

    done = set()
    rows = []
    if args.resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(prev["disc_id"].astype(str))
        rows = prev.to_dict("records")
        print(f"resume: {len(done)} already extracted", flush=True)

    ex = build_extractor()
    errors = []
    t0 = time.time()
    n_ok = 0
    cache: dict[str, tuple[sitk.Image, sitk.Image]] = {}

    grouped = list(work.groupby("file_stem", sort=False))
    for gi, (stem, gdf) in enumerate(grouped):
        pending = [r for r in gdf.itertuples(index=False) if str(r.disc_id) not in done]
        if not pending:
            continue
        try:
            if stem not in cache:
                img = sitk.ReadImage(str(pending[0].image_path))
                raw = sitk.ReadImage(str(pending[0].mask_path))
                ok, why = geometry_ok(img, raw)
                if not ok:
                    raise RuntimeError(f"geometry {why}")
                cache[stem] = (img, raw)
            img, raw = cache[stem]
        except Exception as e:  # noqa: BLE001
            for r in pending:
                errors.append({"disc_id": str(r.disc_id), "error": str(e)})
                print(f"  FAIL {r.disc_id}: {e}", flush=True)
            continue

        for r in pending:
            did = str(r.disc_id)
            try:
                disc, lab, nv = pick_disc_mask(raw, int(r.ivd_label), int(r.mask_label))
                if nv < MIN_VOXELS:
                    raise RuntimeError(f"empty disc mask labels tried; vox={nv}")
                feats = extract_one(ex, img, disc)
                rec = {
                    "disc_id": did,
                    "patient_id": r.patient_id,
                    "mapped_level": r.mapped_level,
                    "pfirrmann": int(r.pfirrmann),
                    "manufacturer": r.manufacturer,
                    "field_T": r.field_T,
                    "mask_label_used": lab,
                    "n_voxels": nv,
                }
                rec.update(feats)
                rows.append(rec)
                n_ok += 1
                if n_ok % 5 == 0 or args.smoke:
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
                    print(
                        f"  {n_ok} new / series {gi+1}/{len(grouped)} last={did} vox={nv} nfeat={len(feats)}",
                        flush=True,
                    )
            except Exception as e:  # noqa: BLE001
                errors.append({"disc_id": did, "error": str(e)})
                print(f"  FAIL {did}: {e}", flush=True)
        # keep only last series in cache
        cache = {stem: cache[stem]} if stem in cache else {}

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(err_csv, index=False)

    feat_cols = [c for c in pd.DataFrame(rows).columns if c not in META_COLS] if rows else []
    aln = write_alignment(feat_cols, "smoke" if args.smoke else "full")
    meta = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "radiomics": getattr(radiomics, "__version__", "unknown"),
        "smoke": bool(args.smoke),
        "n_ok": len(rows),
        "n_fail": len(errors),
        "n_features": len(feat_cols),
        "seconds": round(time.time() - t0, 1),
        "alignment": aln,
        "params": {
            "normalizeScale": 200,
            "resampledPixelSpacing": [3, 3, 3],
            "binWidth": 5,
        },
    }
    tag = "smoke" if args.smoke else "full"
    (OUT / f"d2_extract_{tag}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("D2 done", json.dumps({k: meta[k] for k in ("n_ok", "n_fail", "n_features", "seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
