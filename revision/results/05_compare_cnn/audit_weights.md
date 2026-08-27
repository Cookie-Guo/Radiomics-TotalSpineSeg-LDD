# RadImageNet ResNet50 weight and forward-pass audit

Generated 2026-08-27T06:02:46 by `audit_radimagenet_weights.py`. Read-only; does not modify reported tables.

## A. Is Lab-Rasool `ResNet50.pt` the official RadImageNet checkpoint?

| Check | Result |
|---|---|
| Key alignment | 318/318 keys present in official notop; shape mismatches 0 |
| **Numeric identity** | **identical = 0 / 318** |
| `num_batches_tracked` | Lab-Rasool **96900** vs official **0** |
| Mean of `bn1.running_var` | Lab-Rasool **42.00** vs official **1.30** |
| conv.bias unique to official | 53 |
| vs torchvision IMAGENET1K_V1 | Lab identical 0/318; official identical 0/318 |
| vs torchvision IMAGENET1K_V2 | Lab identical 0/318; official identical 0/318 |

**Verdict:** Lab-Rasool ResNet50.pt shares ZERO tensors with the official RadImageNet release; it is an unverifiable third-party checkpoint, not the officially released weights.

## B. Native normalisation of the official weights

Rule (applied **before** any AUC; uses stored BN running statistics and input patches only; no labels or test performance):
forward real ROI patches and compare per-channel BN input variance with the checkpoint `running_var`.

Input = 36 real mid-sagittal ROI patches; pipeline identical to the ImageNet arm.

| preprocess | median \|log variance ratio\| | implied variance fold-error |
|---|---|---|
| `rescale_0_1` ← **reported arm** | 0.51 | ~1.66× |
| `imagenet_mean_std` | 4.76 | ~116× |
| `caffe_bgr` | 13.58 | ~7.86e+05× |

**Verdict:** native normalisation for official RadImageNet ResNet50 is `rescale_0_1`.

## C. Loading official notop into torchvision

- Official notop has **53 conv.bias** tensors (Keras lineage); torchvision resnet50 convolutions have no bias, so `load_state_dict(strict=False)` **silently drops** them.
- BN `running_mean` was trained with those biases; dropping them is not absorbed by BN.
- Keras `ResNet50` is **v1** stride (downsample on the first 1×1); torchvision is **v1.5** (stride on 3×3). Shapes still match, so a silent wrong load is possible.
- The BN statistic test does **not** separate the two strides (keras_v1 0.505 vs torchvision_v1p5 0.537). Use the Keras definition or a Keras forward pass.

## D. The 0.773 run

| On the Lab-Rasool checkpoint itself | median \|log variance ratio\| |
|---|---|
| `rescale_0_1` | 1.05 |
| `imagenet_mean_std` ← used by that run | 1.59 |
| `caffe_bgr` | 12.04 |

0.773 is an implementation error (non-official weights + non-native normalisation), not a retainable sensitivity arm.
