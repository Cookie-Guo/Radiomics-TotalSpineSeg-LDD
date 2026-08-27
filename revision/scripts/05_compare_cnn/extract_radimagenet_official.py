#!/usr/bin/env python3
"""
Frozen official RadImageNet ResNet50 embeddings (Keras forward, rescale_0_1).
Reads splits/assignments.csv and N4-corrected volumes; writes
results/05_compare_cnn/radimagenet_features.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]
B_COMP = ROOT / "results" / "05_compare_cnn"
WEIGHTS_DIR = B_COMP / "weights"
OUT = ROOT / "results" / "05_compare_cnn"
SPLITS = ROOT / "splits" / "assignments.csv"
VARIANTS = ROOT / "results" / "audit" / "A2_acquisition_resampling" / "A2_acquisition_variants.csv"
RAD = Path(os.environ.get("IMAGE_ROOT", "<image_root>"))
RAD_REAL = Path(os.environ.get("IMAGE_ROOT_REAL", os.environ.get("IMAGE_ROOT", "<image_root>")))
PHI_MAP = Path(os.environ.get("PHI_MAP", "<not published>"))

H5 = WEIGHTS_DIR / "RadImageNet-ResNet50_notop.h5"
PTH = WEIGHTS_DIR / "RadImageNet-ResNet50_notop.pth"

SEED = 4321
PAD = 8
OUT_SIZE = 224
MAIN_PREPROCESS = "rescale_0_1"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CAFFE_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(ROOT / "scripts" / "00_common"))
sys.path.insert(0, str(ROOT / "scripts" / "04_compare_simple"))
from anonymize import Redactor  # noqa: E402
import extract_simple_features as b2mod  # noqa: E402

b2mod.RAD = RAD


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_keras_backbone():
    """Official Keras ResNet50 with official notop weights."""
    import keras
    m = keras.applications.ResNet50(include_top=False, pooling="avg", weights=None)
    if not H5.is_file():
        raise SystemExit(f"missing official Keras weights: {H5}\n"
                         "run revision/scripts/05_compare_cnn/fetch_official_radimagenet_h5.py first")
    m.load_weights(str(H5))
    if m.output_shape[-1] != 2048:
        raise SystemExit(f"unexpected embedding dim {m.output_shape}")
    return m


def verify_h5_vs_pth() -> dict:
    """Cross-check official .h5 vs local official .pth tensors (Keras HWIO vs torch OIHW)."""
    import torch
    m = build_keras_backbone()
    pth = torch.load(PTH, map_location="cpu", weights_only=False)
    torch_np = {k: v.numpy() for k, v in pth.items()}

    checked = mismatch = 0
    worst = 0.0
    details = []
    for layer in m.layers:
        for w in layer.weights:
            a = w.numpy()
            path = w.path
            # Keras conv kernel HWIO -> torch OIHW
            if a.ndim == 4:
                a = np.transpose(a, (3, 2, 0, 1))
            cands = [k for k, v in torch_np.items() if v.shape == a.shape]
            hit = None
            for k in cands:
                d = float(np.abs(torch_np[k] - a).max())
                if d == 0.0:
                    hit = (k, d)
                    break
            if hit is None:
                best = min(((k, float(np.abs(torch_np[k] - a).max())) for k in cands),
                           key=lambda t: t[1], default=(None, float("inf")))
                mismatch += 1
                worst = max(worst, best[1])
                details.append({"keras": path, "best_torch": best[0], "max_abs_diff": best[1]})
            checked += 1
    return {
        "keras_weight_tensors": checked,
        "torch_tensors": len(torch_np),
        "torch_minus_keras": len(torch_np) - checked,
        "torch_extra_are_num_batches_tracked": sum(1 for k in torch_np if "num_batches_tracked" in k),
        "tensors_with_an_exact_match_in_pth": checked - mismatch,
        "tensors_without_exact_match": mismatch,
        "worst_max_abs_diff": worst,
        "unmatched_details": details[:10],
        "verdict": ("official .h5 and local .pth are the SAME weights (bit-exact); the PyTorch "
                    "port is faithful") if mismatch == 0 else "MISMATCH - inspect",
    }


def roi_patch(plane: np.ndarray, disc2d: np.ndarray, pad: int = PAD):
    ys, xs = np.where(disc2d)
    if ys.size == 0:
        raise ValueError("empty disc mask on slice")
    H, W = plane.shape
    r0 = max(0, int(ys.min()) - pad)
    r1 = min(H, int(ys.max()) + pad + 1)
    c0 = max(0, int(xs.min()) - pad)
    c1 = min(W, int(xs.max()) + pad + 1)
    return plane[r0:r1, c0:c1].astype(np.float64), {
        "patch_h": int(r1 - r0), "patch_w": int(c1 - c0), "n_disc_voxels": int(disc2d.sum())}


def patch_to_nhwc(patch: np.ndarray, preprocess: str) -> np.ndarray:
    """Percentile stretch to [0,1], 224^2, 3ch, checkpoint normalisation. No flip. NHWC."""
    import tensorflow as tf
    p1, p99 = np.percentile(patch, [1, 99])
    if p99 <= p1:
        p1, p99 = float(patch.min()), float(patch.max() + 1e-6)
    x = np.clip((patch - p1) / (p99 - p1 + 1e-8), 0.0, 1.0).astype(np.float32)
    t = tf.image.resize(x[None, :, :, None], (OUT_SIZE, OUT_SIZE), method="bilinear").numpy()
    t = np.repeat(t, 3, axis=3)                              # 1,224,224,3 in [0,1]
    if preprocess == "rescale_0_1":
        return t
    if preprocess == "imagenet_mean_std":
        return (t - IMAGENET_MEAN.reshape(1, 1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 1, 3)
    if preprocess == "caffe":
        t = (t * 255.0)[..., ::-1]                           # RGB replica -> BGR
        return t - CAFFE_MEAN_BGR.reshape(1, 1, 1, 3)
    raise ValueError(f"unknown preprocess {preprocess}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify", action="store_true", help="only run the .h5 <-> .pth lineage check")
    ap.add_argument("--limit_patients", type=int, default=0)
    ap.add_argument("--preprocess", default=MAIN_PREPROCESS,
                    choices=["rescale_0_1", "imagenet_mean_std", "caffe"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.verify:
        v = verify_h5_vs_pth()
        (OUT / "official_h5_vs_pth_lineage.json").write_text(
            json.dumps(v, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(v, indent=2, ensure_ascii=False))
        return

    if not PHI_MAP.is_file():
        raise SystemExit(f"missing PHI map: {PHI_MAP}")
    if not RAD.is_dir():
        raise SystemExit(f"missing image root: {RAD}")

    red = Redactor(phi_map=PHI_MAP)
    assign = pd.read_csv(SPLITS)
    variants = pd.read_csv(VARIANTS).set_index("patient_id")
    assert len(assign) == 630, f"expected 630 discs, got {len(assign)}"

    print(f"backbone: official Keras RadImageNet ResNet50 | preprocess={args.preprocess}")
    model = build_keras_backbone()

    patients = assign["patient_id"].drop_duplicates().tolist()
    if args.smoke:
        patients = patients[:3]
    elif args.limit_patients > 0:
        patients = patients[: args.limit_patients]

    rows: list[dict] = []
    errors: list[dict] = []
    first_emb = None

    for i, pid in enumerate(patients):
        sub = assign[assign["patient_id"] == pid]
        try:
            v = variants.loc[pid]
            batch, series = str(v["batch"]), str(v["series"])
            img_p, _ = b2mod.resolve_image_mask(pid, "L4-L5", batch, series, red)
            ia = sitk.GetArrayFromImage(sitk.ReadImage(str(img_p))).astype(np.float64)

            for _, r in sub.iterrows():
                lv, disc_id = r["level"], r["disc_id"]
                try:
                    _, msk_p = b2mod.resolve_image_mask(pid, lv, batch, series, red)
                    ma = sitk.GetArrayFromImage(sitk.ReadImage(str(msk_p))) > 0
                    x = b2mod.best_lr_index(ma)
                    patch, pmeta = roi_patch(b2mod.slice2d(ia, x), b2mod.slice2d(ma, x), pad=PAD)
                    emb = model(patch_to_nhwc(patch, args.preprocess), training=False).numpy().ravel()
                    if emb.shape[0] != 2048:
                        raise RuntimeError(f"expected 2048-d, got {emb.shape}")
                    if first_emb is None:
                        first_emb = emb
                    row = {
                        "disc_id": disc_id, "patient_id": pid, "level": lv,
                        "pfirrmann": int(r["pfirrmann"]), "batch": batch,
                        "midsag_x": int(x), "error": "",
                        **{f"resnet50_{j:04d}": float(emb[j]) for j in range(emb.shape[0])},
                    }
                    row.update({f"meta_{k}": pmeta[k] for k in ("patch_h", "patch_w", "n_disc_voxels")})
                    rows.append(row)
                except Exception as e:  # noqa: BLE001
                    errors.append({"disc_id": disc_id, "patient_id": pid, "error": str(e)[:200]})
                    rows.append({"disc_id": disc_id, "patient_id": pid, "level": lv,
                                 "pfirrmann": int(r["pfirrmann"]), "error": str(e)[:200]})
        except Exception as e:  # noqa: BLE001
            for _, r in sub.iterrows():
                errors.append({"disc_id": r["disc_id"], "patient_id": pid, "error": str(e)[:200]})
                rows.append({"disc_id": r["disc_id"], "patient_id": pid, "level": r["level"],
                             "pfirrmann": int(r["pfirrmann"]), "error": str(e)[:200]})
        if (i + 1) % 20 == 0 or args.smoke:
            print(f"  {i+1}/{len(patients)} patients, rows={len(rows)}")

    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: red.text(x) if isinstance(x, str) else x)

    suffix = "" if args.preprocess == MAIN_PREPROCESS else f"_{args.preprocess}"
    stem = f"B3_radimagenet_official{suffix}"
    out_feat = OUT / (f"{stem}_features_smoke.csv" if args.smoke else f"{stem}_features.csv")
    df.to_csv(out_feat, index=False)
    pd.DataFrame(errors).to_csv(
        OUT / (f"{stem}_errors_smoke.csv" if args.smoke else f"{stem}_errors.csv"), index=False)

    hits = red.check(out_feat.read_text(encoding="utf-8", errors="ignore"))
    if hits:
        raise SystemExit(f"PHI residual: {hits[:5]}")

    ok = int((df["error"].fillna("") == "").sum()) if "error" in df.columns else len(df)
    fcols = [c for c in df.columns if c.startswith("resnet50_")]
    feat_std = float(df[fcols].std().mean()) if fcols and ok else float("nan")
    if first_emb is None or not np.isfinite(first_emb).all() or float(first_emb.std()) == 0.0:
        raise SystemExit("sanity failed: first embedding missing, NaN or zero-variance")

    import keras
    import tensorflow as tf
    meta = {
        "script": "extract_radimagenet_official.py",
        "seed": SEED,
        "backbone": "official RadImageNet ResNet50 (frozen), Keras forward",
        "weights_file": str(H5),
        "weights_name": H5.name,
        "weights_sha256": sha256(H5),
        "weights_bytes": H5.stat().st_size,
        "weights_source": "https://github.com/BMEII-AI/RadImageNet -> Google Drive "
                          "RadImageNet_models-20230414T114049Z-001.zip (member "
                          "RadImageNet_models/RadImageNet-ResNet50_notop.h5)",
        "weights_provenance_file": str(OUT / "official_weights_provenance.json"),
        "architecture": "keras.applications.ResNet50(include_top=False, pooling='avg')",
        "keras_h5_keras_version": "2.4.0",
        "preprocess": args.preprocess,
        "preprocess_is_pre_specified_main_arm": args.preprocess == MAIN_PREPROCESS,
        "preprocess_formula": {
            "rescale_0_1": "percentile[1,99]->[0,1]; bilinear 224; 3ch replicate; no further scaling",
            "imagenet_mean_std": "percentile[1,99]->[0,1]; bilinear 224; 3ch; (x-mean)/std ImageNet",
            "caffe": "percentile[1,99]->[0,1]; bilinear 224; 3ch; x255, RGB->BGR, -[103.939,116.779,123.68]",
        }[args.preprocess],
        "preprocess_decision_rule": "normalisation chosen by agreement between observed activation "
                                    "statistics and the checkpoint's stored BN running statistics "
                                    "(audit_radimagenet_weights.py), decided before any AUC",
        "embedding_dim": 2048,
        "out_size": OUT_SIZE,
        "pad_voxels": PAD,
        "no_horizontal_flip": True,
        "no_finetune": True,
        "not_used_lab_rasool": "ResNet50.pt shares zero tensors with the official release (audit A)",
        "image_root": str(RAD),
        "image_root_real": str(RAD_REAL),
        "n_patients": len(patients),
        "n_rows": len(df),
        "n_ok": ok,
        "n_errors": len(errors),
        "mean_feature_std": feat_std,
        "smoke": bool(args.smoke),
        "elapsed_sec": round(time.time() - t0, 2),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "inputs_md5": {"assignments.csv": md5(SPLITS),
                       "A2_acquisition_variants.csv": md5(VARIANTS)},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT / (f"{stem}_features_smoke.meta.json" if args.smoke
            else f"{stem}_features.meta.json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_feat} ok={ok}/{len(df)} errors={len(errors)} "
          f"std={feat_std:.4f} ({meta['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
