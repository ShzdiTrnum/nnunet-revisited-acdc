"""Show the state of all 15 training jobs (read-only; safe to run any time).
Usage: python ~/nnunet_repro/scripts/train_status.py
"""
import glob, json, os, re, subprocess, time

RES = os.path.join(os.environ["nnUNet_results"], "Dataset027_ACDC")
LOCKS = os.path.expanduser("~/nnunet_repro/logs/train/locks")
MODELS = [("org.", "nnUNetPlans"), ("ResEnc M", "nnUNetResEncUNetMPlans"), ("ResEnc L", "nnUNetResEncUNetLPlans")]
EPOCHS = 1000

def fold_state(plans, f):
    d = os.path.join(RES, f"nnUNetTrainer__{plans}__3d_fullres", f"fold_{f}")
    summ = os.path.join(d, "validation", "summary.json")
    if os.path.isfile(summ):
        dice = json.load(open(summ))["foreground_mean"]["Dice"]
        return "done", f"val Dice {100 * dice:.2f}"
    logs = sorted(glob.glob(os.path.join(d, "training_log_*.txt")), key=os.path.getmtime)
    att_file = os.path.join(LOCKS, f"{plans}_fold{f}.attempts")
    att = open(att_file).read().strip() if os.path.isfile(att_file) else "0"
    if not logs:
        return "pending", f"attempts {att}" if att != "0" else ""
    txt = open(logs[-1], errors="ignore").read()
    ep = [int(x) for x in re.findall(r"Epoch (\d+)\s*$", txt, re.M)]
    times = [float(x) for x in re.findall(r"Epoch time: ([\d.]+) s", txt)]
    best = re.findall(r"New best EMA pseudo Dice: ([\d.]+)", txt)
    age_min = (time.time() - os.path.getmtime(logs[-1])) / 60
    state = "RUNNING" if age_min < 15 else "stopped"
    epoch = ep[-1] if ep else 0
    t = sum(times[-20:]) / len(times[-20:]) if times else 0
    eta_h = (EPOCHS - epoch) * t / 3600 if t else 0
    info = f"epoch {epoch}/{EPOCHS} | {t:.0f} s/epoch | ETA {eta_h:.1f} h"
    if best:
        info += f" | best EMA pseudo-Dice {float(best[-1]):.4f}"
    if state == "stopped":
        info += f" | log silent {age_min:.0f} min, attempts {att}"
    return state, info

print(time.strftime("%F %T"))
for label, plans in MODELS:
    for f in range(5):
        s, info = fold_state(plans, f)
        print(f"{label:9s} fold {f}: {s:8s} {info}")
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
except FileNotFoundError:
    pass
