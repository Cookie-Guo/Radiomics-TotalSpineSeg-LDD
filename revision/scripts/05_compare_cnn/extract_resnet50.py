#!/usr/bin/env python3
"""
Frozen ResNet50 (ImageNet) embeddings for mid-sagittal disc ROIs.
Reads splits/assignments.csv and the N4-corrected volumes; writes
results/05_compare_cnn/imagenet_features.csv.
"""

from __future__ import annotations

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
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_common"))
from anonymize import Redactor  # noqa: E402

# path helpers (same as B2, no radiomics import)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_compare_simple"))
from extract_simple_features import (  # noqa: E402
    LEVELS,
    LEVEL_TO_FOLDER,
    RAD,
    best_lr_index,
    resolve_image_mask,
    slice2d,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "05_compare_cnn"
SPLITS = ROOT / "splits" / "assignments.csv"
VARIANTS = ROOT / "results" / "audit" / "A2_acquisition_resampling" / "A2_acquisition_variants.csv"
SEED = 4321
PAD = 8  # voxels around disc bbox on mid-sagittal
OUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_backbone(device: torch.device) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    # drop FC → 2048-d after avgpool
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


def roi_patch(plane: np.ndarray, disc2d: np.ndarray, pad: int = PAD) -> tuple[np.ndarray, dict]:
    """Extract padded bbox around disc on mid-sagittal; return float32 HxW and meta."""
    ys, xs = np.where(disc2d)
    if ys.size == 0:
        raise ValueError("empty disc mask on slice")
    H, W = plane.shape
    r0 = max(0, int(ys.min()) - pad)
    r1 = min(H, int(ys.max()) + pad + 1)
    c0 = max(0, int(xs.min()) - pad)
    c1 = min(W, int(xs.max()) + pad + 1)
    patch = plane[r0:r1, c0:c1].astype(np.float64)
    meta = {
        "bbox": [r0, r1, c0, c1],
        "patch_h": int(r1 - r0),
        "patch_w": int(c1 - c0),
        "n_disc_voxels": int(disc2d.sum()),
    }
    return patch, meta


def patch_to_tensor(patch: np.ndarray) -> torch.Tensor:
    """
    Per-ROI percentile stretch → [0,1] → 3ch ImageNet normalize → 1x3x224x224.
    No horizontal flip.
    """
    p1, p99 = np.percentile(patch, [1, 99])
    if p99 <= p1:
        p1, p99 = float(patch.min()), float(patch.max() + 1e-6)
    x = np.clip((patch - p1) / (p99 - p1 + 1e-8), 0.0, 1.0).astype(np.float32)
    # resize with torch bilinear
    t = torch.from_numpy(x)[None, None, :, :]  # 1,1,H,W
    t = torch.nn.functional.interpolate(t, size=(OUT_SIZE, OUT_SIZE), mode="bilinear", align_corners=False)
    t = t.repeat(1, 3, 1, 1)  # 1,3,224,224
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t


@torch.inference_mode()
def extract_embedding(model: nn.Module, patch: np.ndarray, device: torch.device) -> np.ndarray:
    t = patch_to_tensor(patch).to(device)
    feat = model(t)  # 1, 2048
    return feat.squeeze(0).cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit_patients", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    red = Redactor()
    assign = pd.read_csv(SPLITS)
    variants = pd.read_csv(VARIANTS).set_index("patient_id")
    assert len(assign) == 630

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"device={device}; loading ResNet50 ImageNet weights …")
    model = build_backbone(device)

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
            # load image once via L4-L5
            img_p, _ = resolve_image_mask(pid, "L4-L5", batch, series, red)
            img = sitk.ReadImage(str(img_p))
            ia = sitk.GetArrayFromImage(img).astype(np.float64)

            for _, r in sub.iterrows():
                lv = r["level"]
                disc_id = r["disc_id"]
                try:
                    _, msk_p = resolve_image_mask(pid, lv, batch, series, red)
                    ma = sitk.GetArrayFromImage(sitk.ReadImage(str(msk_p))) > 0
                    x = best_lr_index(ma)
                    plane = slice2d(ia, x)
                    disc2d = slice2d(ma, x)
                    patch, pmeta = roi_patch(plane, disc2d, pad=PAD)
                    emb = extract_embedding(model, patch, device)
                    row = {
                        "disc_id": disc_id,
                        "patient_id": pid,
                        "level": lv,
                        "pfirrmann": int(r["pfirrmann"]),
                        "batch": batch,
                        "midsag_x": int(x),
                        "error": "",
                        **{f"resnet50_{j:04d}": float(emb[j]) for j in range(emb.shape[0])},
                    }
                    row.update({f"meta_{k}": pmeta[k] for k in ("patch_h", "patch_w", "n_disc_voxels")})
                    rows.append(row)
                except Exception as e:  # noqa: BLE001
                    errors.append({"disc_id": disc_id, "patient_id": pid, "error": str(e)[:200]})
                    rows.append({
                        "disc_id": disc_id,
                        "patient_id": pid,
                        "level": lv,
                        "pfirrmann": int(r["pfirrmann"]),
                        "error": str(e)[:200],
                    })
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
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: red.text(x) if isinstance(x, str) else x)

    tag = "smoke" if args.smoke else "full"
    out_feat = RESULTS / (f"B3_resnet50_features_{tag}.csv" if args.smoke else "imagenet_features.csv")
    # for full run use standard name
    if not args.smoke:
        out_feat = RESULTS / "imagenet_features.csv"
    df.to_csv(out_feat, index=False)
    err_path = RESULTS / ("B3_resnet50_errors_smoke.csv" if args.smoke else "imagenet_extract_errors.csv")
    pd.DataFrame(errors).to_csv(err_path, index=False)

    blob = out_feat.read_text(encoding="utf-8", errors="ignore")
    hits = red.check(blob)
    if hits:
        raise SystemExit(f"PHI residual: {hits[:5]}")

    ok = int((df["error"].fillna("") == "").sum()) if "error" in df.columns else len(df)
    meta = {
        "script": "extract_resnet50.py",
        "seed": SEED,
        "backbone": "torchvision resnet50 IMAGENET1K_V2",
        "embedding_dim": 2048,
        "out_size": OUT_SIZE,
        "pad_voxels": PAD,
        "no_horizontal_flip": True,
        "no_finetune": True,
        "device": str(device),
        "n_patients": len(patients),
        "n_rows": len(df),
        "n_ok": ok,
        "n_errors": len(errors),
        "smoke": bool(args.smoke),
        "elapsed_sec": round(time.time() - t0, 2),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "inputs_md5": {
            "assignments.csv": md5(SPLITS),
            "A2_acquisition_variants.csv": md5(VARIANTS),
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = RESULTS / ("B3_resnet50_features_smoke.meta.json" if args.smoke else "imagenet_features.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_feat} ok={ok}/{len(df)} errors={len(errors)} ({meta['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
