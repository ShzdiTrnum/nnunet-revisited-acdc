"""Status of the U-Mamba / SwinUNETR training jobs (read-only; safe any time).
Usage (after source ~/umamba_repro/env_umamba.sh): python ~/umamba_repro/train_status_umamba.py
"""
import glob, json, os, re, subprocess, time

RES = os.path.join(os.environ["nnUNet_results"], "Dataset027_ACDC")
LOCKS = os.path.expanduser("~/umamba_repro/logs/train/locks")
MODELS = [("U-Mamba Bot", "bot", "nnUNetTrainerUMambaBot", "nnUNetPlans"),
          ("U-Mamba Enc", "enc", "nnUNetTrainerUMambaEnc", "nnUNetPlans"),
          ("SwinUNETR", "swin", "nnUNetTrainerSwinUNETR", "nnUNetPlans_bs3")]
EPOCHS = 1000


def fold_state(key, tr, pl, f):
    d = os.path.join(RES, f"{tr}__{pl}__3d_fullres", f"fold_{f}")
    summ = os.path.join(d, "validation", "summary.json")
    if os.path.isfile(summ):
        return "done", f"val Dice {100 * json.load(open(summ))['foreground_mean']['Dice']:.2f}"
    att_file = os.path.join(LOCKS, f"{key}_fold{f}.attempts")
    att = open(att_file).read().strip() if os.path.isfile(att_file) else "0"
    logs = sorted(glob.glob(os.path.join(d, "training_log_*.txt")), key=os.path.getmtime)
    if not logs:
        return ("pending", f"attempts {att}" if att != "0" else "") if os.path.isdir(os.path.dirname(d)) or att != "0" else (None, "")
    txt = open(logs[-1], errors="ignore").read()
    ep = [int(x) for x in re.findall(r"Epoch (\d+)\s*$", txt, re.M)]
    times = [float(x) for x in re.findall(r"Epoch time: ([\d.]+) s", txt)]
    best = re.findall(r"New best EMA pseudo Dice: ([\d.]+)", txt)
    age = (time.time() - os.path.getmtime(logs[-1])) / 60
    epoch = ep[-1] if ep else 0
    t = sum(times[-20:]) / len(times[-20:]) if times else 0
    info = f"epoch {epoch}/{EPOCHS} | {t:.0f} s/epoch | ETA {(EPOCHS - epoch) * t / 3600:.1f} h"
    if best:
        info += f" | best EMA pseudo-Dice {float(best[-1]):.4f}"
    if age >= 15:
        info += f" | log silent {age:.0f} min, attempts {att}"
    return ("RUNNING" if age < 15 else "stopped"), info


print(time.strftime("%F %T"))
for label, key, tr, pl in MODELS:
    rows = [fold_state(key, tr, pl, f) for f in range(5)]
    if all(s is None for s, _ in rows):
        print(f"{label:12s} not queued")
        continue
    for f, (s, info) in enumerate(rows):
        print(f"{label:12s} fold {f}: {s or 'pending':8s} {info}")
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
except FileNotFoundError:
    pass
