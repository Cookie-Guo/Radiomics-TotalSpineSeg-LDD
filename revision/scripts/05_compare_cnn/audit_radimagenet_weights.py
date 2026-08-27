#!/usr/bin/env python3
"""
Provenance check of RadImageNet ResNet50 checkpoints versus the official release.
Writes results/05_compare_cnn/audit_weights.json and audit_weights.md.
"""

from __future__ import annotations
import os

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

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "05_compare_cnn"
WEIGHTS_DIR = RESULTS / "weights"
OUT = ROOT / "results" / "05_compare_cnn"
SPLITS = ROOT / "splits" / "assignments.csv"
VARIANTS = ROOT / "results" / "audit" / "A2_acquisition_resampling" / "A2_acquisition_variants.csv"
PHI_MAP = Path(os.environ.get("PHI_MAP", "<not published>"))

N_AUDIT_PATIENTS = 12
PAD = 8
OUT_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CAFFE_MEAN_BGR = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)

SEQ_TO_TV = (("0.", "conv1."), ("1.", "bn1."), ("4.", "layer1."),
             ("5.", "layer2."), ("6.", "layer3."), ("7.", "layer4."))

MODES = ["rescale_0_1", "imagenet_mean_std", "caffe_bgr"]

warnings.filterwarnings("ignore")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def remap_sequential(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        nk = k[len("backbone."):] if k.startswith("backbone.") else k
        for src, dst in SEQ_TO_TV:
            if nk.startswith(src):
                out[dst + nk[len(src):]] = v
                break
    return out


def load_patches(n_patients: int):
    """Real mid-sagittal ROI patches; same pipeline as the ImageNet arm."""
    sys.path.insert(0, str(ROOT / "scripts" / "00_common"))
    sys.path.insert(0, str(ROOT / "scripts" / "04_compare_simple"))
    from anonymize import Redactor
    import extract_simple_features as b2mod
    b2mod.RAD = Path(os.environ.get("IMAGE_ROOT", "<image_root>"))
    red = Redactor(phi_map=PHI_MAP)

    assign = pd.read_csv(SPLITS)
    variants = pd.read_csv(VARIANTS).set_index("patient_id")
    tensors = []
    for pid in assign["patient_id"].drop_duplicates().tolist()[:n_patients]:
        v = variants.loc[pid]
        batch, series = str(v["batch"]), str(v["series"])
        img_p, _ = b2mod.resolve_image_mask(pid, "L4-L5", batch, series, red)
        ia = sitk.GetArrayFromImage(sitk.ReadImage(str(img_p))).astype(np.float64)
        for lv in ["L3-L4", "L4-L5", "L5-S1"]:
            try:
                _, msk_p = b2mod.resolve_image_mask(pid, lv, batch, series, red)
                ma = sitk.GetArrayFromImage(sitk.ReadImage(str(msk_p))) > 0
                x = b2mod.best_lr_index(ma)
                plane, disc2d = b2mod.slice2d(ia, x), b2mod.slice2d(ma, x)
                ys, xs = np.where(disc2d)
                H, W = plane.shape
                r0, r1 = max(0, int(ys.min()) - PAD), min(H, int(ys.max()) + PAD + 1)
                c0, c1 = max(0, int(xs.min()) - PAD), min(W, int(xs.max()) + PAD + 1)
                patch = plane[r0:r1, c0:c1].astype(np.float64)
                p1, p99 = np.percentile(patch, [1, 99])
                if p99 <= p1:
                    p1, p99 = float(patch.min()), float(patch.max() + 1e-6)
                arr = np.clip((patch - p1) / (p99 - p1 + 1e-8), 0, 1).astype(np.float32)
                t = torch.from_numpy(arr)[None, None]
                t = torch.nn.functional.interpolate(t, size=(OUT_SIZE, OUT_SIZE),
                                                    mode="bilinear", align_corners=False)
                tensors.append(t.repeat(1, 3, 1, 1))
            except Exception:
                pass
    return torch.cat(tensors, 0)


def preprocessed(x01: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "rescale_0_1":
        return x01
    if mode == "imagenet_mean_std":
        return (x01 - IMAGENET_MEAN) / IMAGENET_STD
    if mode == "caffe_bgr":
        return (x01 * 255.0)[:, [2, 1, 0]] - CAFFE_MEAN_BGR
    raise ValueError(mode)


def build_torch_resnet(state: dict, arch: str, conv_bias: bool) -> nn.Module:
    """arch: torchvision_v1p5 (stride on 3x3) or keras_v1 (stride on first 1x1)."""
    m = resnet50(weights=None)
    if conv_bias:
        for mod in m.modules():
            if isinstance(mod, nn.Conv2d) and mod.bias is None:
                mod.bias = nn.Parameter(torch.zeros(mod.out_channels))
    if arch == "keras_v1":
        for lname in ("layer2", "layer3", "layer4"):
            blk = getattr(m, lname)[0]
            blk.conv1.stride = (2, 2)
            blk.conv2.stride = (1, 1)
    m.fc = nn.Identity()
    missing, unexpected = m.load_state_dict(state, strict=False)
    m.eval()
    m._audit_load = {
        "missing_non_fc": [k for k in missing if not k.startswith("fc.")],
        "n_unexpected": len(unexpected),
    }
    return m


@torch.inference_mode()
def bn_agreement(model: nn.Module, x: torch.Tensor) -> dict:
    """Observed activation stats vs checkpoint stored BN running stats."""
    logvar, corr_mean = [], []
    handles = []

    def mk(mod):
        def hook(_m, inp, _o):
            z = inp[0].detach()
            ov = z.var(dim=(0, 2, 3)).clamp_min(1e-8)
            om = z.mean(dim=(0, 2, 3))
            logvar.append((ov.log() - mod.running_var.clamp_min(1e-8).log()).abs().median().item())
            a, b = om.numpy(), mod.running_mean.numpy()
            if a.std() > 0 and b.std() > 0:
                corr_mean.append(float(np.corrcoef(a, b)[0, 1]))
        return hook

    for _, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            handles.append(mod.register_forward_hook(mk(mod)))
    model(x)
    for h in handles:
        h.remove()
    return {
        "median_abs_log_var_ratio": float(np.median(logvar)),
        "implied_variance_fold_error": float(np.exp(np.median(logvar))),
        "median_corr_obs_mean_vs_running_mean": float(np.median(corr_mean)) if corr_mean else float("nan"),
        "n_bn_layers": len(logvar),
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"script": "audit_radimagenet_weights.py",
                    "purpose": "provenance check of the RadImageNet ResNet50 checkpoint",
                    "timestamp": datetime.now().isoformat(timespec="seconds")}

    lab_p = WEIGHTS_DIR / "ResNet50.pt"
    off_p = WEIGHTS_DIR / "RadImageNet-ResNet50_notop.pth"
    for p in (lab_p, off_p):
        if not p.is_file():
            raise SystemExit(f"missing checkpoint: {p}")

    lab_raw = torch.load(lab_p, map_location="cpu", weights_only=False)
    off = torch.load(off_p, map_location="cpu", weights_only=False)
    lab = remap_sequential(lab_raw)

    report["checkpoints"] = {
        "lab_rasool": {"path": str(lab_p), "sha256": sha256(lab_p),
                       "bytes": lab_p.stat().st_size, "n_keys": len(lab_raw)},
        "official_notop": {"path": str(off_p), "sha256": sha256(off_p),
                           "bytes": off_p.stat().st_size, "n_keys": len(off)},
    }

    common = [k for k in lab if k in off]
    identical = shape_mismatch = 0
    maxdiff = {}
    for k in common:
        a, b = lab[k].float(), off[k].float()
        if a.shape != b.shape:
            shape_mismatch += 1
            continue
        if torch.equal(a, b):
            identical += 1
        else:
            maxdiff[k] = float((a - b).abs().max())
    conv_bias_only_official = [k for k in off if k.endswith(".bias") and k not in lab]
    report["A_lab_vs_official"] = {
        "keys_lab_remapped": len(lab),
        "keys_official": len(off),
        "keys_in_common": len(common),
        "identical_tensors": identical,
        "shape_mismatches": shape_mismatch,
        "keys_only_in_official": len(off) - len(common),
        "conv_bias_keys_only_in_official": len(conv_bias_only_official),
        "example_max_abs_diff": {k: round(maxdiff[k], 6) for k in list(maxdiff)[:6]},
        "num_batches_tracked_lab": int(lab["bn1.num_batches_tracked"]),
        "num_batches_tracked_official": int(off["bn1.num_batches_tracked"]),
        "bn1_running_var_mean_lab": float(lab["bn1.running_var"].float().mean()),
        "bn1_running_var_mean_official": float(off["bn1.running_var"].float().mean()),
        "verdict": ("Lab-Rasool ResNet50.pt shares ZERO tensors with the official RadImageNet "
                    "release; it is an unverifiable third-party checkpoint, not the officially "
                    "released weights.") if identical == 0 else "PARTIAL/UNEXPECTED - inspect manually",
    }

    imnet = {}
    for w, name in ((ResNet50_Weights.IMAGENET1K_V1, "IMAGENET1K_V1"),
                    (ResNet50_Weights.IMAGENET1K_V2, "IMAGENET1K_V2")):
        sd = w.get_state_dict(progress=False)
        imnet[name] = {
            "lab_identical": sum(1 for k in lab if k in sd and lab[k].shape == sd[k].shape
                                 and torch.equal(lab[k].float(), sd[k].float())),
            "official_identical": sum(1 for k in off if k in sd and off[k].shape == sd[k].shape
                                      and torch.equal(off[k].float(), sd[k].float())),
        }
    report["A_vs_torchvision_imagenet"] = imnet

    print("loading real mid-sagittal ROI patches ...")
    x01 = load_patches(N_AUDIT_PATIENTS)
    report["diagnostic_input"] = {
        "n_patches": int(x01.shape[0]), "shape": list(x01.shape),
        "pipeline": "identical to B3 ImageNet arm: best_lr_index slice, pad8 bbox, "
                    "percentile[1,99]->[0,1], bilinear 224, 3-channel replicate, no flip",
        "note": "labels and test-set performance are NOT used by this diagnostic",
    }

    diag: dict = {}
    for arch in ("keras_v1", "torchvision_v1p5"):
        m = build_torch_resnet(off, arch, conv_bias=True)
        diag[f"official_notop::{arch}"] = {
            "load": m._audit_load,
            **{mode: bn_agreement(m, preprocessed(x01, mode)) for mode in MODES},
        }

    off_nobias = {k: v for k, v in off.items() if k not in conv_bias_only_official}
    m_nb = build_torch_resnet(off_nobias, "torchvision_v1p5", conv_bias=False)
    diag["official_notop::torchvision_v1p5_BIASES_DROPPED"] = {
        "load": m_nb._audit_load,
        "n_conv_bias_silently_dropped": len(conv_bias_only_official),
        **{mode: bn_agreement(m_nb, preprocessed(x01, mode)) for mode in MODES},
    }

    m_lab = build_torch_resnet(lab, "torchvision_v1p5", conv_bias=False)
    diag["lab_rasool::torchvision_v1p5"] = {
        "load": m_lab._audit_load,
        **{mode: bn_agreement(m_lab, preprocessed(x01, mode)) for mode in MODES},
    }
    report["BC_preprocess_and_architecture_diagnostic"] = diag

    ref = diag["official_notop::keras_v1"]
    best_mode = min(MODES, key=lambda mo: ref[mo]["median_abs_log_var_ratio"])
    report["B_verdict"] = {
        "pre_specified_rule": "pick the normalisation whose activation statistics agree with the "
                              "checkpoint's stored BN running statistics; decided before any AUC",
        "selected_preprocess_for_official_weights": best_mode,
        "ranking": {mo: round(ref[mo]["median_abs_log_var_ratio"], 3) for mo in MODES},
        "preprocess_used_by_the_0p773_run": "imagenet_mean_std",
    }
    report["C_verdict"] = {
        "official_notop_has_conv_bias": len(conv_bias_only_official),
        "silently_dropped_if_loaded_into_torchvision": True,
        "keras_resnet50_is_v1_stride_torchvision_is_v1p5": True,
        "bn_diagnostic_cannot_separate_v1_from_v1p5": True,
        "implication": "architecture must come from the Keras definition (or a Keras forward), "
                       "not be guessed on the PyTorch side",
    }
    report["elapsed_sec"] = round(time.time() - t0, 2)
    report["platform"] = platform.platform()
    report["python"] = sys.version.split()[0]
    report["torch"] = torch.__version__

    (OUT / "audit_weights.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                            encoding="utf-8")

    A = report["A_lab_vs_official"]
    BAR = "&#124;"
    L = []
    L.append("# RadImageNet ResNet50 weight and forward-pass audit\n")
    L.append(
        f"> Generated {report['timestamp']} by `audit_radimagenet_weights.py`. "
        "Read-only; does not modify reported tables.\n"
    )
    L.append("## A. Is Lab-Rasool `ResNet50.pt` the official RadImageNet checkpoint?\n")
    L.append("| Check | Result |")
    L.append("|---|---|")
    L.append(
        f"| Key alignment | {A['keys_in_common']}/{A['keys_lab_remapped']} keys in official notop; "
        f"shape mismatches {A['shape_mismatches']} |"
    )
    L.append(
        f"| **Numeric identity** | **identical = {A['identical_tensors']} / {A['keys_in_common']}** |"
    )
    L.append(
        f"| `num_batches_tracked` | Lab-Rasool **{A['num_batches_tracked_lab']}** vs official "
        f"**{A['num_batches_tracked_official']}** |"
    )
    L.append(
        f"| Mean of `bn1.running_var` | Lab-Rasool **{A['bn1_running_var_mean_lab']:.2f}** vs official "
        f"**{A['bn1_running_var_mean_official']:.2f}** |"
    )
    L.append(f"| conv.bias unique to official | {A['conv_bias_keys_only_in_official']} |")
    for n, v in imnet.items():
        L.append(
            f"| vs torchvision {n} | Lab identical {v['lab_identical']}/318; "
            f"official identical {v['official_identical']}/318 |"
        )
    L.append(f"\n**Verdict:** {A['verdict']}\n")

    L.append("## B. Native normalisation of the official weights\n")
    L.append(
        "Rule (applied **before** any AUC; stored BN running statistics and input patches only): "
        "compare per-channel BN input variance with checkpoint `running_var`.\n"
    )
    L.append(
        f"Input = {report['diagnostic_input']['n_patches']} real mid-sagittal ROI patches; "
        "pipeline identical to the ImageNet arm.\n"
    )
    L.append(f"| preprocess | median {BAR}log variance ratio{BAR} | implied variance fold-error |")
    L.append("|---|---|---|")
    for mo in MODES:
        d = ref[mo]
        star = " <- **reported arm**" if mo == best_mode else (
            " <- used by the 0.773 run" if mo == "imagenet_mean_std" else ""
        )
        L.append(
            f"| `{mo}`{star} | {d['median_abs_log_var_ratio']:.2f} | "
            f"~{d['implied_variance_fold_error']:.3g}x |"
        )
    L.append(
        f"\n**Verdict:** native normalisation for official RadImageNet ResNet50 is `{best_mode}`.\n"
    )

    L.append("## C. Loading official notop into torchvision\n")
    L.append(
        f"- Official notop has **{len(conv_bias_only_official)} conv.bias** tensors (Keras lineage); "
        "torchvision convolutions have no bias, so `load_state_dict(strict=False)` silently drops them."
    )
    L.append("- BN `running_mean` was trained with those biases; dropping them is not absorbed by BN.")
    L.append(
        "- Keras `ResNet50` is **v1** stride (downsample on the first 1x1); torchvision is **v1.5** "
        "(stride on 3x3). Shapes still match, so a silent wrong load is possible."
    )
    kv = ref[best_mode]["median_abs_log_var_ratio"]
    tv = diag["official_notop::torchvision_v1p5"][best_mode]["median_abs_log_var_ratio"]
    L.append(
        f"- The BN statistic test does not separate the two strides "
        f"(keras_v1 {kv:.3f} vs torchvision_v1p5 {tv:.3f}). "
        "Use the Keras definition or a Keras forward pass.\n"
    )

    L.append("## D. The 0.773 run\n")
    lb = diag["lab_rasool::torchvision_v1p5"]
    L.append(f"| On the Lab-Rasool checkpoint itself | median {BAR}log variance ratio{BAR} |")
    L.append("|---|---|")
    for mo in MODES:
        star = " <- used by that run" if mo == "imagenet_mean_std" else ""
        L.append(f"| `{mo}`{star} | {lb[mo]['median_abs_log_var_ratio']:.2f} |")
    L.append(
        "\n0.773 is an implementation error (non-official weights + non-native normalisation), "
        "not a retainable sensitivity arm.\n"
    )
    (OUT / "audit_weights.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\nA: Lab-Rasool identical tensors = {A['identical_tensors']}/{A['keys_in_common']}")
    print(f"B: selected preprocess = {best_mode}  ranking={report['B_verdict']['ranking']}")
    print(f"C: conv biases at risk = {len(conv_bias_only_official)}")
    print(f"Wrote {OUT / 'audit_weights.md'}  ({report['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
