| Model | Folds | Dice RV | Dice MYO | Dice LV | **Mean Dice** (± SD) | Paper Dice | Gap (in fold SDs) | NSD @2 mm (± SD) | Paper NSD |
|---|---|---|---|---|---|---|---|---|---|
| nnU-Net (org.) | 5/5 | 90.12 | 89.96 | 94.61 | **91.56 ± 0.63** | 91.54 | +0.02 (0.03 SD) | 94.94 ± 0.73 | 94.93 |
| nnU-Net ResEnc M | 5/5 | 90.77 | 90.27 | 94.90 | **91.98 ± 0.57** | 91.99 | -0.01 (0.02 SD) | 95.53 ± 0.69 | 95.50 |
| nnU-Net ResEnc L | 5/5 | 90.50 | 89.97 | 94.69 | **91.72 ± 0.47** | 91.69 | +0.03 (0.06 SD) | 95.19 ± 0.55 | 95.11 |
| U-Mamba Bot | 5/5 | 90.31 | 89.95 | 94.62 | **91.63 ± 0.58** | 91.79 | -0.16 (0.28 SD) | 95.06 ± 0.66 | 95.40 |
| U-Mamba Enc | 0/5 | – | – | – | – | 91.22 | – | – | 94.33 |
| SwinUNETR | 5/5 | 85.52 | 86.67 | 93.04 | **88.41 ± 1.05** | 91.29 | -2.88 (2.74 SD) | 90.96 ± 1.07 | 94.12 |

Paired comparison with nnU-Net (org.) (same folds, mean Dice per fold):

| Model | Folds compared | Mean difference | Folds better | Paper difference | Larger than fold SD? |
|---|---|---|---|---|---|
| nnU-Net ResEnc M | 5 | +0.42 | 5/5 | +0.45 | no |
| nnU-Net ResEnc L | 5 | +0.16 | 4/5 | +0.15 | no |
| U-Mamba Bot | 5 | +0.07 | 3/5 | +0.25 | no |
| SwinUNETR | 5 | -3.15 | 0/5 | -0.25 | yes |

Dice/NSD in %, 5-fold CV on the 100 ACDC training patients (200 ED/ES volumes), official nnU-Net split. Cross-check vs each model's own nnU-Net summary.json: max per-fold Dice difference = 0.000 points.
