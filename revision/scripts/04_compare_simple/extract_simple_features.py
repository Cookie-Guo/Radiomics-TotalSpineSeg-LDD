#!/usr/bin/env python3
"""
Extract simple disc measurements (DHI, disc/CSF ratio, area, 2D sphericity).
Reads N4-corrected volumes under IMAGE_ROOT; writes results/04_compare_simple/features.csv.
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
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_common"))
from anonymize import Redactor  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "04_compare_simple"
SPLITS = ROOT / "splits" / "assignments.csv"
VARIANTS = ROOT / "results" / "audit" / "A2_acquisition_resampling" / "A2_acquisition_variants.csv"
RAD = Path(os.environ.get("IMAGE_ROOT", "<image_root>"))
SEED = 4321
LEVELS = ["L3-L4", "L4-L5", "L5-S1"]
LEVEL_TO_FOLDER = {"L3-L4": "L3-4", "L4-L5": "L4-5", "L5-S1": "L5-S1"}


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
    hit = [p for p in loose if ser_clean in p.name or stem_clean in p.name]
    if len(hit) == 1:
        return hit[0], msk
    if len(loose) > 1 and hit:
        return hit[0], msk
    raise FileNotFoundError(f"image not found for {pid} series={series} batch={batch}")


def best_lr_index(mask_arr: np.ndarray) -> int:
    """mask_arr: z,y,x bool/int → max-area LR index (axis=2)."""
    areas = mask_arr.sum(axis=(0, 1))
    if areas.max() <= 0:
        raise ValueError("empty mask")
    return int(np.argmax(areas))


def slice2d(arr3: np.ndarray, x: int) -> np.ndarray:
    return arr3[:, :, x]


def hist_peak(values: np.ndarray, n_bins: int = 64) -> float:
    """Intensity mode via histogram peak (Waldenberg-style peak SI)."""
    v = values[np.isfinite(values)]
    if v.size < 5:
        return float("nan")
    lo, hi = np.percentile(v, [1, 99])
    if hi <= lo:
        return float(np.median(v))
    hist, edges = np.histogram(v, bins=n_bins, range=(float(lo), float(hi)))
    i = int(np.argmax(hist))
    return float(0.5 * (edges[i] + edges[i + 1]))


def disc_height_amp_mm(mask2d: np.ndarray, spacing_si: float, spacing_ap: float) -> dict:
    """
    mask2d rows=SI (axis0), cols=AP (axis1).
    Anterior / middle / posterior thirds along AP → mean SI span (mm).
    """
    ys, xs = np.where(mask2d)
    if ys.size == 0:
        return {"disc_height_mm": float("nan"), "h_ant": np.nan, "h_mid": np.nan, "h_post": np.nan}
    c0, c1 = int(xs.min()), int(xs.max())
    width = max(c1 - c0, 1)
    heights = []
    labels = []
    for lab, t0, t1 in (("h_ant", 0.0, 1 / 3), ("h_mid", 1 / 3, 2 / 3), ("h_post", 2 / 3, 1.0)):
        lo = c0 + width * t0
        hi = c0 + width * t1
        sel = (xs >= lo) & (xs <= hi + 1e-9)
        if not np.any(sel):
            heights.append(np.nan)
            labels.append(lab)
            continue
        r = ys[sel]
        heights.append((r.max() - r.min() + 1) * spacing_si)
        labels.append(lab)
    out = {lab: float(h) if np.isfinite(h) else float("nan") for lab, h in zip(labels, heights)}
    out["disc_height_mm"] = float(np.nanmean(heights))
    return out


def perimeter_area_sphericity(mask2d: np.ndarray, sp_si: float, sp_ap: float) -> dict:
    """2D area (mm²) and circularity (sphericity_2d = 4πA/P²)."""
    m = mask2d.astype(bool)
    n = int(m.sum())
    if n == 0:
        return {"area_mm2": float("nan"), "perimeter_mm": float("nan"), "sphericity_2d": float("nan")}
    area = n * sp_si * sp_ap
    # perimeter via binary erosion boundary count, scaled by mean spacing
    eroded = ndimage.binary_erosion(m)
    boundary = m & ~eroded
    # approximate perimeter: boundary voxels * mean in-plane spacing
    perim = float(boundary.sum()) * float(np.mean([sp_si, sp_ap]))
    if perim < 1e-6:
        sph = float("nan")
    else:
        sph = float(4.0 * np.pi * area / (perim ** 2))
        sph = min(sph, 1.0)  # numerical clip
    return {"area_mm2": float(area), "perimeter_mm": perim, "sphericity_2d": sph}


def si_margins(mask2d: np.ndarray) -> tuple[float, float]:
    """Return (si_superior, si_inferior) as row indices; larger row = superior in this cohort."""
    ys, _ = np.where(mask2d)
    if ys.size == 0:
        return float("nan"), float("nan")
    return float(ys.max()), float(ys.min())


def csf_roi(plane: np.ndarray, disc2d: np.ndarray, side: str = "low_col") -> np.ndarray:
    """
    Heuristic CSF: on mid-sagittal plane, SI band of disc, AP strip on posterior side.
    side='low_col' → smaller column indices (validated as brighter/CSF side in cohort).
    """
    ys, xs = np.where(disc2d)
    if ys.size == 0:
        return np.zeros_like(disc2d, dtype=bool)
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    H, W = plane.shape
    # pad SI slightly
    r0p, r1p = max(0, r0 - 2), min(H - 1, r1 + 2)
    band_w = max(8, int(0.6 * (c1 - c0 + 1)))
    if side == "low_col":
        a0, a1 = max(0, c0 - band_w - 2), max(0, c0 - 2)
    else:
        a0, a1 = min(W, c1 + 2), min(W, c1 + band_w + 2)
    if a1 <= a0:
        return np.zeros_like(disc2d, dtype=bool)
    strip = np.zeros_like(disc2d, dtype=bool)
    strip[r0p : r1p + 1, a0:a1] = True
    strip &= ~disc2d
    vals = plane[strip]
    if vals.size < 10:
        return strip
    thr = np.percentile(vals, 75)
    return strip & (plane >= thr)


def features_for_disc(
    plane: np.ndarray,
    disc2d: np.ndarray,
    sp_si: float,
    sp_ap: float,
) -> dict:
    h = disc_height_amp_mm(disc2d, sp_si, sp_ap)
    sh = perimeter_area_sphericity(disc2d, sp_si, sp_ap)
    disc_vals = plane[disc2d].astype(np.float64)
    csf = csf_roi(plane, disc2d, side="low_col")
    csf_vals = plane[csf].astype(np.float64)
    # if too few CSF voxels, try other side
    if csf_vals.size < 10:
        csf = csf_roi(plane, disc2d, side="high_col")
        csf_vals = plane[csf].astype(np.float64)
        csf_side = "high_col"
    else:
        csf_side = "low_col"

    peak_d = hist_peak(disc_vals)
    peak_c = hist_peak(csf_vals) if csf_vals.size >= 5 else float("nan")
    mean_d = float(np.mean(disc_vals)) if disc_vals.size else float("nan")
    mean_c = float(np.mean(csf_vals)) if csf_vals.size else float("nan")
    delta = peak_d - peak_c if np.isfinite(peak_d) and np.isfinite(peak_c) else float("nan")
    delta_n = (
        delta / abs(peak_c)
        if np.isfinite(delta) and abs(peak_c) > 1e-6
        else float("nan")
    )
    ratio = mean_d / mean_c if np.isfinite(mean_c) and abs(mean_c) > 1e-6 else float("nan")

    sup, inf = si_margins(disc2d)
    out = {
        **h,
        **sh,
        "peak_si_disc": peak_d,
        "peak_si_csf": peak_c,
        "delta_peak_si": float(delta) if np.isfinite(delta) else float("nan"),
        "delta_peak_si_norm": float(delta_n) if np.isfinite(delta_n) else float("nan"),
        "disc_mean_si": mean_d,
        "csf_mean_si": mean_c,
        "disc_csf_mean_ratio": float(ratio) if np.isfinite(ratio) else float("nan"),
        "n_disc_voxels": int(disc2d.sum()),
        "n_csf_voxels": int(csf.sum()),
        "csf_side": csf_side,
        "si_superior_row": sup,
        "si_inferior_row": inf,
    }
    return out


def dhi_for_patient(
    masks2d: dict[str, np.ndarray],
    heights: dict[str, float],
    sp_si: float,
) -> dict[str, float]:
    """
    masks2d / heights keyed by level_std L3-L4 etc.
    VB between superior disc A and inferior disc B:
      gap = si_inferior(A) - si_superior(B) - 1  (rows), * sp_si
    larger row = superior.
    """
    order = ["L3-L4", "L4-L5", "L5-S1"]
    margins = {}
    for lv in order:
        if lv not in masks2d or masks2d[lv] is None:
            margins[lv] = (np.nan, np.nan)
        else:
            margins[lv] = si_margins(masks2d[lv])

    def gap(sup_lv: str, inf_lv: str) -> float:
        # inferior margin of superior disc, superior margin of inferior disc
        _, inf_of_sup = margins[sup_lv]
        sup_of_inf, _ = margins[inf_lv]
        if not (np.isfinite(inf_of_sup) and np.isfinite(sup_of_inf)):
            return float("nan")
        g = (inf_of_sup - sup_of_inf - 1.0) * sp_si
        return float(g) if g > 0 else float("nan")

    vb_L4 = gap("L3-L4", "L4-L5")  # between L3-4 and L4-5
    vb_L5 = gap("L4-L5", "L5-S1")  # between L4-5 and L5-S1

    # map adjacent VBs per disc level
    adj = {
        "L3-L4": [vb_L4],           # only inferior VB reliably
        "L4-L5": [vb_L4, vb_L5],
        "L5-S1": [vb_L5],
    }
    dhi = {}
    for lv in order:
        dh = heights.get(lv, float("nan"))
        vbs = [v for v in adj[lv] if np.isfinite(v) and v > 0.5]
        if not np.isfinite(dh) or not vbs:
            dhi[lv] = float("nan")
        else:
            dhi[lv] = float(dh / np.mean(vbs))
    return {
        "dhi": dhi,
        "vb_L4_mm": vb_L4,
        "vb_L5_mm": vb_L5,
    }


def process_patient(
    pid: str,
    batch: str,
    series: str,
    red: Redactor,
    assign_levels: pd.DataFrame,
) -> list[dict]:
    """assign_levels: rows for this patient with disc_id, level, pfirrmann."""
    # load image once via first level
    img_p, _ = resolve_image_mask(pid, "L4-L5", batch, series, red)
    img = sitk.ReadImage(str(img_p))
    ia = sitk.GetArrayFromImage(img).astype(np.float64)
    sp = img.GetSpacing()  # x,y,z
    # arr z,y,x ; SI=axis0≈z-sitk, AP=axis1≈y-sitk after GetArray
    # sitk spacing: (sx,sy,sz) for x,y,z → array axes (sz, sy, sx) for (z,y,x)
    sp_si = float(sp[2])  # z
    sp_ap = float(sp[1])  # y
    sp_lr = float(sp[0])

    masks3 = {}
    own_x = {}
    for lv in LEVELS:
        try:
            _, msk_p = resolve_image_mask(pid, lv, batch, series, red)
            m = sitk.ReadImage(str(msk_p))
            ma = sitk.GetArrayFromImage(m) > 0
            masks3[lv] = ma
            own_x[lv] = best_lr_index(ma)
        except Exception as e:  # noqa: BLE001
            masks3[lv] = None
            own_x[lv] = None

    # common mid-sag for DHI: prefer L4-5
    if own_x.get("L4-L5") is not None:
        x_dhi = own_x["L4-L5"]
    else:
        xs = [v for v in own_x.values() if v is not None]
        if not xs:
            raise FileNotFoundError(f"no masks for {pid}")
        x_dhi = int(np.median(xs))

    plane_dhi = slice2d(ia, x_dhi)
    masks2d_dhi = {}
    heights = {}
    for lv in LEVELS:
        if masks3[lv] is None:
            masks2d_dhi[lv] = None
            heights[lv] = float("nan")
            continue
        m2 = slice2d(masks3[lv], x_dhi)
        masks2d_dhi[lv] = m2
        heights[lv] = disc_height_amp_mm(m2, sp_si, sp_ap)["disc_height_mm"]

    dhi_pack = dhi_for_patient(masks2d_dhi, heights, sp_si)

    rows = []
    for _, r in assign_levels.iterrows():
        lv = r["level"]
        if masks3.get(lv) is None:
            rows.append({
                "disc_id": r["disc_id"],
                "patient_id": pid,
                "level": lv,
                "pfirrmann": int(r["pfirrmann"]),
                "error": "mask_missing",
            })
            continue
        x = own_x[lv]
        plane = slice2d(ia, x)
        disc2d = slice2d(masks3[lv], x)
        try:
            feats = features_for_disc(plane, disc2d, sp_si, sp_ap)
            # DHI from common plane geometry
            feats["dhi"] = dhi_pack["dhi"].get(lv, float("nan"))
            feats["vb_L4_mm"] = dhi_pack["vb_L4_mm"]
            feats["vb_L5_mm"] = dhi_pack["vb_L5_mm"]
            feats["disc_height_dhi_plane_mm"] = heights.get(lv, float("nan"))
            feats["midsag_x_own"] = int(x)
            feats["midsag_x_dhi"] = int(x_dhi)
            feats["spacing_si_mm"] = sp_si
            feats["spacing_ap_mm"] = sp_ap
            feats["spacing_lr_mm"] = sp_lr
            row = {
                "disc_id": r["disc_id"],
                "patient_id": pid,
                "level": lv,
                "pfirrmann": int(r["pfirrmann"]),
                "batch": batch,
                "error": "",
                **feats,
            }
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            rows.append({
                "disc_id": r["disc_id"],
                "patient_id": pid,
                "level": lv,
                "pfirrmann": int(r["pfirrmann"]),
                "error": str(e)[:200],
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit_patients", type=int, default=0)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    red = Redactor()
    assign = pd.read_csv(SPLITS)
    variants = pd.read_csv(VARIANTS).set_index("patient_id")
    assert len(assign) == 630

    patients = assign["patient_id"].drop_duplicates().tolist()
    if args.smoke:
        patients = patients[:3]
    elif args.limit_patients > 0:
        patients = patients[: args.limit_patients]

    rows: list[dict] = []
    errors: list[dict] = []
    for i, pid in enumerate(patients):
        sub = assign[assign["patient_id"] == pid]
        try:
            v = variants.loc[pid]
            batch, series = str(v["batch"]), str(v["series"])
            pr = process_patient(pid, batch, series, red, sub)
            for row in pr:
                if row.get("error"):
                    errors.append({"disc_id": row["disc_id"], "patient_id": pid, "error": row["error"]})
                rows.append(row)
        except Exception as e:  # noqa: BLE001
            for _, r in sub.iterrows():
                errors.append({"disc_id": r["disc_id"], "patient_id": pid, "error": str(e)[:200]})
                rows.append({
                    "disc_id": r["disc_id"],
                    "patient_id": pid,
                    "level": r["level"],
                    "pfirrmann": int(r["pfirrmann"]),
                    "error": str(e)[:200],
                })
        if (i + 1) % 20 == 0 or args.smoke:
            print(f"  {i+1}/{len(patients)} patients, rows={len(rows)}")

    df = pd.DataFrame(rows)
    # drop raw error text from main feature table if empty; keep column
    # redact any accidental PHI (should be none)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: red.text(x) if isinstance(x, str) else x)

    out_feat = RESULTS / ("B2_simple_features_smoke.csv" if args.smoke else "features.csv")
    df.to_csv(out_feat, index=False)
    err_df = pd.DataFrame(errors)
    out_err = RESULTS / ("B2_simple_errors_smoke.csv" if args.smoke else "extract_errors.csv")
    err_df.to_csv(out_err, index=False)

    # residual PHI assert
    blob = out_feat.read_text(encoding="utf-8", errors="ignore")
    hits = red.check(blob)
    if hits:
        raise SystemExit(f"PHI residual in output: {hits[:5]}")

    ok = df["error"].fillna("").eq("").sum() if "error" in df.columns else len(df)
    meta = {
        "script": "extract_simple_features.py",
        "seed": SEED,
        "n_patients": len(patients),
        "n_rows": len(df),
        "n_ok": int(ok),
        "n_errors": len(errors),
        "smoke": bool(args.smoke),
        "definitions": {
            "dhi": "mean(ant/mid/post disc height) / mean(adjacent inter-disc VB gaps on mid-sagittal)",
            "delta_peak_si": "hist_peak(disc) - hist_peak(CSF); Waldenberg 2018 operationalized",
            "delta_peak_si_norm": "(peak_disc - peak_csf) / |peak_csf|",
            "disc_csf_mean_ratio": "mean(disc) / mean(CSF)",
            "sphericity_2d": "4*pi*area/perimeter^2 on mid-sagittal disc mask",
            "csf_roi": "mid-sagittal SI band of disc, AP strip on low-column (posterior) side, >=P75",
            "reference": "Waldenberg et al. Eur Spine J 2018;27:1042-1048",
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "inputs_md5": {
            "assignments.csv": md5(SPLITS),
            "A2_acquisition_variants.csv": md5(VARIANTS),
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = RESULTS / ("B2_simple_features_smoke.meta.json" if args.smoke else "features.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_feat} ok={ok}/{len(df)} errors={len(errors)}  ({meta['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
