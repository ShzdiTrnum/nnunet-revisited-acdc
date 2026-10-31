#!/usr/bin/env bash
# U-Mamba / SwinUNETR on ACDC, Step 3: resumable training queue (one worker per GPU, inside tmux).
# Usage:  bash ~/umamba_repro/train_queue_umamba.sh <gpu_id> [models...]
#   models (in queue order): bot (default) | enc | swin      e.g.  ... 0 bot swin
#   bot  = nnUNetTrainerUMambaBot,  plans nnUNetPlans   (batch 4, fp16 AMP, as planned)
#   enc  = nnUNetTrainerUMambaEnc,  plans nnUNetPlans   (batch 4, fp16 AMP, as planned)
#   swin = nnUNetTrainerSwinUNETR,  plans nnUNetPlans_bs3 (batch 3 instead of 4: fp32 batch 4 does not fit in 48 GB)
# Same safety as the nnU-Net queue: a fold is DONE when fold_X/validation/summary.json exists; flock prevents two
# workers training the same fold; --c resumes from checkpoint_latest.pth (every 50 epochs); 5 failed attempts -> skipped.
set -uo pipefail
source ~/umamba_repro/env_umamba.sh
GPU="${1:?usage: train_queue_umamba.sh <gpu_id> [bot] [enc] [swin]}"; shift
MODELS=("${@:-bot}"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(bot)
D=27; DS=Dataset027_ACDC; CFG=3d_fullres; MAX_ATTEMPTS=5
PRE="$nnUNet_preprocessed/$DS"; LOGDIR=~/umamba_repro/logs/train; LOCKDIR=$LOGDIR/locks
mkdir -p "$LOCKDIR"; QLOG=$LOGDIR/queue_gpu${GPU}.log
say() { echo "$(date '+%F %T') [GPU $GPU] $*" | tee -a "$QLOG"; }
python -c "import nnunetv2,sys; sys.exit(0 if 'U-Mamba' in nnunetv2.__file__ else 'wrong environment')" || exit 1

job_spec() { case "$1" in
  bot)  echo "nnUNetTrainerUMambaBot nnUNetPlans";;
  enc)  echo "nnUNetTrainerUMambaEnc nnUNetPlans";;
  swin) echo "nnUNetTrainerSwinUNETR nnUNetPlans_bs3";;
  *) echo "unknown model $1" >&2; exit 1;; esac; }
is_done() { [ -f "$nnUNet_results/$DS/${1}__${2}__${CFG}/fold_$3/validation/summary.json" ]; }

# one-time preparation, done by whichever worker gets the lock first (the other waits) ------------------
(
  flock 8
  # (a) unpack the preprocessed data once, so parallel trainings never read half-written files
  if [ ! -f "$PRE/nnUNetPlans_3d_fullres/.unpacked_ok" ]; then
    say "unpacking preprocessed data once (both workers share it)..."
    python -c "from nnunetv2.training.dataloading.utils import unpack_dataset; unpack_dataset('$PRE/nnUNetPlans_3d_fullres', True, False, 8)"
    n_img=$(ls "$PRE"/nnUNetPlans_3d_fullres/*.npy | grep -vc _seg.npy); n_seg=$(ls "$PRE"/nnUNetPlans_3d_fullres/*_seg.npy | wc -l)
    [ "$n_img" -eq 200 ] && [ "$n_seg" -eq 200 ] && touch "$PRE/nnUNetPlans_3d_fullres/.unpacked_ok" && say "unpacked 200 images + 200 labels"
  fi
  # (b) SwinUNETR plans: identical to nnUNetPlans except batch_size 3 (same preprocessed data)
  if [ ! -f "$PRE/nnUNetPlans_bs3.json" ]; then
    python - << PY
import json
p = json.load(open("$PRE/nnUNetPlans.json"))
p["plans_name"] = "nnUNetPlans_bs3"
p["configurations"]["3d_fullres"]["batch_size"] = 3
json.dump(p, open("$PRE/nnUNetPlans_bs3.json", "w"), indent=2)
PY
    say "created nnUNetPlans_bs3.json (batch_size 3, everything else identical)"
  fi
) 8>"$LOCKDIR/prepare.lock"
[ -f "$PRE/nnUNetPlans_3d_fullres/.unpacked_ok" ] || { say "ERROR: unpacking failed"; exit 1; }

say "worker started (pid $$), models: ${MODELS[*]}"
while true; do
  launched=0; open=0
  for m in "${MODELS[@]}"; do
    read -r TR PL <<< "$(job_spec "$m")"
    for f in 0 1 2 3 4; do
      is_done "$TR" "$PL" "$f" && continue
      open=$((open + 1)); job="${m}_fold${f}"
      att=$(cat "$LOCKDIR/$job.attempts" 2>/dev/null || echo 0)
      [ "$att" -ge "$MAX_ATTEMPTS" ] && continue
      exec 9>"$LOCKDIR/$job.lock"
      if ! flock -n 9; then exec 9>&-; continue; fi
      if is_done "$TR" "$PL" "$f"; then exec 9>&-; continue; fi
      echo $((att + 1)) > "$LOCKDIR/$job.attempts"
      say "START $job ($TR, $PL) attempt $((att + 1))/$MAX_ATTEMPTS -> log: $LOGDIR/$job.log"
      CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_train $D $CFG $f -tr "$TR" -p "$PL" --c >> "$LOGDIR/$job.log" 2>&1
      rc=$?; exec 9>&-
      if is_done "$TR" "$PL" "$f"; then rm -f "$LOCKDIR/$job.attempts"; say "DONE  $job"
      else say "FAIL  $job (exit code $rc) -- last lines:"; tail -n 5 "$LOGDIR/$job.log" | tee -a "$QLOG"; sleep 60; fi
      launched=1; break 2
    done
  done
  if [ "$launched" -eq 0 ]; then
    if [ "$open" -eq 0 ]; then say "ALL JOBS DONE for: ${MODELS[*]}"
    else say "no free job for this GPU ($open unfinished: running on the other GPU or failed $MAX_ATTEMPTS times)"; fi
    break
  fi
done
say "worker exiting"
