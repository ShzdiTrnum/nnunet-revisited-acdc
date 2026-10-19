#!/usr/bin/env bash
# Step 2b: convert ACDC to nnU-Net format, check the split, verify dataset integrity.
# Safe to re-run. Usage:  bash ~/nnunet_repro/scripts/step2_convert.sh <ACDC root containing training/>
set -euo pipefail
source ~/nnunet_repro/env.sh
ACDC_ROOT="${1:?give the ACDC root folder printed by download_acdc.py}"
DS=Dataset027_ACDC
LOG=~/nnunet_repro/logs/step2_convert_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "=== Step 2 conversion started $(date) ==="

# 1. convert (official nnU-Net converter via wrapper; also writes the patient-level 5-fold split)
n_lab=$(ls "$nnUNet_raw/$DS/labelsTr" 2>/dev/null | wc -l || true)
if [ -f "$nnUNet_raw/$DS/dataset.json" ] && [ "$n_lab" -eq 200 ] && [ -f "$nnUNet_preprocessed/$DS/splits_final.json" ]; then
  echo "[skip] $DS already converted (200 labels + splits_final.json)"
else
  python ~/nnunet_repro/scripts/convert_acdc.py -i "$ACDC_ROOT" -d 27
fi
echo "imagesTr: $(ls "$nnUNet_raw/$DS/imagesTr" | wc -l) | labelsTr: $(ls "$nnUNet_raw/$DS/labelsTr" | wc -l)  (expected 200 / 200)"
cat "$nnUNet_raw/$DS/dataset.json"

# 2. check the split: 5 folds, 20 patients (40 cases) per validation fold, no patient leakage
python - << 'PY'
import json, os
s = json.load(open(os.path.join(os.environ["nnUNet_preprocessed"], "Dataset027_ACDC", "splits_final.json")))
pat = lambda cases: {c[:10] for c in cases}          # 'patient001_frame01' -> 'patient001'
all_val = []
assert len(s) == 5, f"expected 5 folds, got {len(s)}"
for k, f in enumerate(s):
    leak = pat(f["train"]) & pat(f["val"])
    print(f"fold {k}: train {len(f['train'])} cases / {len(pat(f['train']))} patients | "
          f"val {len(f['val'])} cases / {len(pat(f['val']))} patients | leakage: {sorted(leak) or 'none'}")
    assert not leak and len(f["val"]) == 40 and len(f["train"]) == 160
    all_val += f["val"]
assert len(all_val) == len(set(all_val)) == 200, "validation folds must cover all 200 cases exactly once"
print("SPLIT OK: patient-level 5-fold CV, every case validated exactly once")
PY

# 3. integrity check + dataset fingerprint (spacings, intensities) -- needed by the planners in Step 3
nnUNetv2_extract_fingerprint -d 27 --verify_dataset_integrity -np 8
python - << 'PY'
import json, os
import numpy as np
fp = json.load(open(os.path.join(os.environ["nnUNet_preprocessed"], "Dataset027_ACDC", "dataset_fingerprint.json")))
sp = np.array(fp["spacings"]); sh = np.array(fp["shapes_after_crop"])
print("cases:", len(sp))
print("median spacing (z,y,x) mm:", np.median(sp, 0).round(3), "| min:", sp.min(0).round(3), "| max:", sp.max(0).round(3))
print("median shape after crop:", np.median(sh, 0).astype(int))
PY
echo "=== STEP 2 COMPLETE $(date). Log: $LOG ==="
