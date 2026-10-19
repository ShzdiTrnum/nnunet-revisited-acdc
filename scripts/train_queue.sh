#!/usr/bin/env bash
# Step 4: resumable training queue for the nnU-Net Revisited ACDC reproduction.
# Start one worker per GPU (each inside tmux). Workers share one job list:
#   3 models x 5 folds = 15 jobs, ordered nnU-Net (org.) -> ResEnc M -> ResEnc L.
# - A job is DONE when fold_X/validation/summary.json exists (training + validation finished).
# - A job being trained is locked (flock), so two workers never train the same fold.
#   The lock disappears automatically if the worker dies (reboot, crash, kill).
# - Every launch uses --c: continue from checkpoint_latest.pth (saved every 50 epochs),
#   start fresh if no checkpoint exists, or just run validation if training already finished.
# - A job that fails MAX_ATTEMPTS times in a row is skipped and reported (no endless loop).
# Safe to re-run at any time.   Usage:  bash ~/nnunet_repro/scripts/train_queue.sh <gpu_id>
set -uo pipefail
source ~/nnunet_repro/env.sh
GPU="${1:?usage: train_queue.sh <gpu_id>   (0 or 1)}"
D=27; DS=Dataset027_ACDC; CFG=3d_fullres; MAX_ATTEMPTS=5
PLANS=(nnUNetPlans nnUNetResEncUNetMPlans nnUNetResEncUNetLPlans)
LOGDIR=~/nnunet_repro/logs/train; LOCKDIR=$LOGDIR/locks
mkdir -p "$LOCKDIR"
QLOG=$LOGDIR/queue_gpu${GPU}.log
say() { echo "$(date '+%F %T') [GPU $GPU] $*" | tee -a "$QLOG"; }
is_done() { [ -f "$nnUNet_results/$DS/nnUNetTrainer__${1}__${CFG}/fold_$2/validation/summary.json" ]; }

say "worker started (pid $$)"
while true; do
  launched=0; open=0
  for p in "${PLANS[@]}"; do
    for f in 0 1 2 3 4; do
      is_done "$p" "$f" && continue
      open=$((open + 1))
      job="${p}_fold${f}"
      att=$(cat "$LOCKDIR/$job.attempts" 2>/dev/null || echo 0)
      if [ "$att" -ge "$MAX_ATTEMPTS" ]; then continue; fi
      exec 9>"$LOCKDIR/$job.lock"
      if ! flock -n 9; then exec 9>&-; continue; fi      # another worker is training it
      if is_done "$p" "$f"; then exec 9>&-; continue; fi   # finished meanwhile
      echo $((att + 1)) > "$LOCKDIR/$job.attempts"
      say "START $job (attempt $((att + 1))/$MAX_ATTEMPTS) -> log: $LOGDIR/$job.log"
      CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_train $D $CFG $f -p "$p" --c >> "$LOGDIR/$job.log" 2>&1
      rc=$?
      exec 9>&-
      if is_done "$p" "$f"; then
        rm -f "$LOCKDIR/$job.attempts"; say "DONE  $job"
      else
        say "FAIL  $job (exit code $rc) -- last lines:"; tail -n 5 "$LOGDIR/$job.log" | tee -a "$QLOG"
        sleep 60
      fi
      launched=1; break 2                                  # rescan the list from the top
    done
  done
  if [ "$launched" -eq 0 ]; then
    if [ "$open" -eq 0 ]; then say "ALL 15 JOBS DONE"; else
      say "no free job for this GPU ($open unfinished: running on the other GPU or failed $MAX_ATTEMPTS times)"; fi
    break
  fi
done
say "worker exiting"
