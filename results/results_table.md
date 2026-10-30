| Model | Folds | Dice RV | Dice MYO | Dice LV | Mean Dice (± SD over folds) | NSD @2 mm | Paper Dice |
|---|---|---|---|---|---|---|---|
| nnU-Net (org.) | 5/5 | 90.12 | 89.96 | 94.61 | **91.56 ± 0.63** | 94.94 ± 0.73 | 91.54 |
| nnU-Net ResEnc M | 5/5 | 90.77 | 90.27 | 94.90 | **91.98 ± 0.57** | 95.53 ± 0.69 | 91.99 |
| nnU-Net ResEnc L | 5/5 | 90.50 | 89.97 | 94.69 | **91.72 ± 0.47** | 95.19 ± 0.55 | 91.69 |

Dice/NSD in %, 5-fold CV on the 100 ACDC training patients (200 ED/ES volumes), per-fold mean over cases and classes, then mean over folds. Cross-check vs nnU-Net's own summary.json: max per-fold Dice difference = 0.000 points.
