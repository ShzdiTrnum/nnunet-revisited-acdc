#!/usr/bin/env bash
# Step 3: experiment planning (3 models) + preprocessing for ACDC (Dataset027).
# Safe to re-run: finished parts are skipped. Usage: bash ~/nnunet_repro/scripts/step3_plan_preprocess.sh
set -euo pipefail
source ~/nnunet_repro/env.sh
D=27
DS=Dataset027_ACDC
PRE="$nnUNet_preprocessed/$DS"
LOG=~/nnunet_repro/logs/step3_plan_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "=== Step 3 started $(date) ==="

# 1. Plan the three models compared in the paper.
#    planner class                -> plans file (used later with -p)
#    ExperimentPlanner (default)  -> nnUNetPlans              = "nnU-Net (org.)"
#    nnUNetPlannerResEncM         -> nnUNetResEncUNetMPlans   = "nnU-Net ResEnc M"
#    nnUNetPlannerResEncL         -> nnUNetResEncUNetLPlans   = "nnU-Net ResEnc L"
plan() {  # $1 planner, $2 plans name
  if [ -f "$PRE/$2.json" ]; then echo "[skip] $2.json exists"
  else nnUNetv2_plan_experiment -d $D -pl "$1"; fi
}
plan ExperimentPlanner      nnUNetPlans
plan nnUNetPlannerResEncM   nnUNetResEncUNetMPlans
plan nnUNetPlannerResEncL   nnUNetResEncUNetLPlans

# 2. Preprocess 3d_fullres once. The ResEnc presets reuse the default plans' 3d_fullres data
#    (same spacing + normalization), so one preprocessed copy serves all three models.
OUT="$PRE/nnUNetPlans_3d_fullres"
n_done=$(ls "$OUT"/*.pkl 2>/dev/null | wc -l || true)
if [ "$n_done" -eq 200 ]; then echo "[skip] 3d_fullres already preprocessed (200 cases)"
else nnUNetv2_preprocess -d $D -plans_name nnUNetPlans -c 3d_fullres -np 8; fi
echo "preprocessed cases: $(ls "$OUT"/*.pkl | wc -l) (expected 200) | size: $(du -sh "$OUT" | cut -f1)"

# 3. Summarize what nnU-Net chose for each model (record this in your README).
python - << 'PY'
import json, os
pre = os.path.join(os.environ["nnUNet_preprocessed"], "Dataset027_ACDC")
for label, name in [("nnU-Net (org.)", "nnUNetPlans"),
                    ("ResEnc M", "nnUNetResEncUNetMPlans"),
                    ("ResEnc L", "nnUNetResEncUNetLPlans")]:
    p = json.load(open(os.path.join(pre, name + ".json")))
    c = p["configurations"]["3d_fullres"]
    a = c["architecture"]; k = a["arch_kwargs"]
    blocks = k.get("n_blocks_per_stage", k.get("n_conv_per_stage"))
    print(f"\n[{label}]  plans: {name}")
    print(f"  network        : {a['network_class_name'].split('.')[-1]}")
    print(f"  patch size     : {c['patch_size']}   batch size: {c['batch_size']}")
    print(f"  target spacing : {[round(s, 4) for s in c['spacing']]}  (z, y, x in mm)")
    print(f"  stages         : {k['n_stages']}   features: {k['features_per_stage']}")
    print(f"  encoder blocks : {blocks}   decoder convs: {k.get('n_conv_per_stage_decoder')}")
    print(f"  data identifier: {c['data_identifier']}")
    print(f"  other configs  : {[x for x in p['configurations'] if x != '3d_fullres']}")
PY
echo "=== STEP 3 COMPLETE $(date). Log: $LOG ==="
