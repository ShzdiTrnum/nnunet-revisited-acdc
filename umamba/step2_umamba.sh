#!/usr/bin/env bash
# U-Mamba / SwinUNETR on ACDC, Step 2: plan + preprocess with U-Mamba's nnU-Net 2.1.1, compare the plan
# with the nnU-Net 2.8.1 runs, and check memory/speed of the three models. Safe to re-run.
# Run INSIDE tmux:  bash ~/umamba_repro/step2_umamba.sh
set -euo pipefail
source ~/umamba_repro/env_umamba.sh
R=~/umamba_repro; DS=Dataset027_ACDC; PRE="$nnUNet_preprocessed/$DS"
LOG=$R/logs/step2_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "=== U-Mamba Step 2 started $(date) ==="

# guard: never run this with the nnU-Net 2.8.1 environment
python -c "import nnunetv2,sys; sys.exit(0 if 'U-Mamba' in nnunetv2.__file__ else 'wrong environment: '+nnunetv2.__file__)"
SPLIT_SRC=~/nnunet_repro/data/nnUNet_preprocessed/$DS/splits_final.json

# 1. fingerprint + plan + preprocess 3d_fullres (the configuration the paper reports)
n_done=$(ls "$PRE"/nnUNetPlans_3d_fullres/*.pkl 2>/dev/null | wc -l || true)
if [ "$n_done" -eq 200 ]; then echo "[skip] 3d_fullres already preprocessed (200 cases)"
else nnUNetv2_plan_and_preprocess -d 27 -c 3d_fullres -np 8 --verify_dataset_integrity; fi
echo "preprocessed cases: $(ls "$PRE"/nnUNetPlans_3d_fullres/*.pkl | wc -l) (expected 200)"

# 2. the official split must still be the one used for the nnU-Net runs
if cmp -s "$SPLIT_SRC" "$PRE/splits_final.json"; then echo "[ok] splits_final.json unchanged (same folds as the nnU-Net runs)"
else echo "ERROR: split differs from the nnU-Net runs"; exit 1; fi

# 3. compare U-Mamba's 2.1.1 plan with the 2.8.1 plan used for nnU-Net (org.)
python - << 'PY'
import json, os
new = json.load(open(os.path.join(os.environ["nnUNet_preprocessed"], "Dataset027_ACDC", "nnUNetPlans.json")))["configurations"]["3d_fullres"]
old = json.load(open(os.path.expanduser("~/nnunet_repro/data/nnUNet_preprocessed/Dataset027_ACDC/nnUNetPlans.json")))["configurations"]["3d_fullres"]
def get(c, k):
    a = c.get("architecture", {}).get("arch_kwargs", {})
    return {"patch_size": c["patch_size"], "batch_size": c["batch_size"], "spacing": [round(s, 4) for s in c["spacing"]],
            "n_stages": a.get("n_stages", len(c.get("conv_kernel_sizes", []))),
            "features": a.get("features_per_stage", c.get("unet_max_num_features"))}[k]
print(f"\n{'3d_fullres plan':16s} {'U-Mamba (nnU-Net 2.1.1)':32s} {'nnU-Net runs (2.8.1)':32s} same?")
for k in ("patch_size", "batch_size", "spacing", "n_stages"):
    a, b = get(new, k), get(old, k)
    print(f"{k:16s} {str(a):32s} {str(b):32s} {'yes' if a == b else 'DIFFERENT'}")
PY

# 4. memory + speed of the three models at the planned settings (GPU 1)
CUDA_VISIBLE_DEVICES=1 python $R/smoke_umamba.py
echo "=== U-Mamba STEP 2 COMPLETE $(date). Log: $LOG ==="
