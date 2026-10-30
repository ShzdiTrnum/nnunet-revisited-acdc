"""Make the two README figures from the final evaluation (CPU only, a few seconds).

  1. per_fold_dice.png      : mean Dice per fold for the three models (shows the fold-wise consistency)
  2. example_segmentation.png : MRI + ground truth + the three models' predictions for a typical case
                                (median ResEnc M Dice) and a difficult case (10th percentile)

Usage: python make_figures.py --results ~/nnunet_repro/results
Reads results_summary.json + per_case_metrics.csv written by evaluate_acdc.py, the ACDC images/labels in
nnUNet_raw and each model's own cross-validation predictions in nnUNet_results.
"""
import argparse, csv, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import SimpleITK as sitk

RAW = os.path.join(os.environ["nnUNet_raw"], "Dataset027_ACDC")
RES = os.path.join(os.environ["nnUNet_results"], "Dataset027_ACDC")
MODELS = [("nnU-Net (org.)", "nnUNetPlans"), ("nnU-Net ResEnc M", "nnUNetResEncUNetMPlans"),
          ("nnU-Net ResEnc L", "nnUNetResEncUNetLPlans")]
SHORT = {"nnU-Net (org.)": "nnU-Net (org.)", "nnU-Net ResEnc M": "ResEnc M", "nnU-Net ResEnc L": "ResEnc L"}
# validated categorical palette (models) and a second validated triple for the anatomy classes
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
CLASS_COLORS = {1: ("RV", "#eda100"), 2: ("MYO", "#e87ba4"), 3: ("LV", "#008300")}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a65", "#e4e3de", "#fcfcfb"

plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def per_fold_plot(summary, out_png):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ends = []
    for (label, _), color in zip(MODELS, MODEL_COLORS):
        if label not in summary:
            continue
        pf = summary[label]["per_fold"]
        folds = sorted(int(f) for f in pf)
        y = [100 * pf[str(f)]["Dice"] for f in folds]
        ax.plot(folds, y, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, label=SHORT[label])
        ends.append([y[-1], folds[-1], f"{SHORT[label]}  {summary[label]['Dice'][0]:.2f}"])
    # direct labels at the line ends, pushed apart so they never overlap
    lo, hi = ax.get_ylim(); gap = (hi - lo) * 0.075
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    for ylab, x, text in ends:
        ax.annotate(text, xy=(x, ylab), xytext=(12, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_ylim(lo, max(hi, ends[-1][0] + gap) if ends else hi)
    ax.set_xticks(range(5), [f"fold {f}" for f in range(5)])
    ax.set_ylabel("Mean Dice (%)")
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_xlim(-0.3, 5.7)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9, ncol=3)
    fig.suptitle("ACDC 5-fold cross-validation: mean Dice per fold (end label = 5-fold mean)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out_png, dpi=200); plt.close(fig)


def load(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def pick_cases(rows):
    m = [r for r in rows if r["model"] == "nnU-Net ResEnc M"] or rows
    score = lambda r: np.nanmean([float(r[f"Dice_{c}"]) for c in ("RV", "MYO", "LV")])
    m = sorted(m, key=score)
    return [("typical case (median)", m[len(m) // 2]), ("difficult case (10th percentile)", m[len(m) // 10])]


def case_dice(rows, model, case):
    for r in rows:
        if r["model"] == model and r["case"] == case:
            return 100 * np.nanmean([float(r[f"Dice_{c}"]) for c in ("RV", "MYO", "LV")])
    return float("nan")


def example_plot(rows, out_png):
    cmap = ListedColormap(["#00000000"] + [CLASS_COLORS[k][1] for k in (1, 2, 3)])
    cases = pick_cases(rows)
    fig, axes = plt.subplots(len(cases), 2 + len(MODELS), figsize=(13, 3.4 * len(cases)), gridspec_kw={"hspace": 0.35})
    for i, (title, row) in enumerate(cases):
        case, fold = row["case"], row["fold"]
        img = load(os.path.join(RAW, "imagesTr", case + "_0000.nii.gz")).astype(float)
        gt = load(os.path.join(RAW, "labelsTr", case + ".nii.gz"))
        z = int(np.argmax((gt > 0).reshape(gt.shape[0], -1).sum(1)))        # slice with most foreground
        lo, hi = np.percentile(img[z], (1, 99))
        sl = np.clip((img[z] - lo) / (hi - lo + 1e-8), 0, 1)
        panels = [("MRI", None, None), ("Ground truth", gt, None)]
        for label, plans in MODELS:
            p = os.path.join(RES, f"nnUNetTrainer__{plans}__3d_fullres", f"fold_{fold}", "validation", case + ".nii.gz")
            if os.path.isfile(p):
                panels.append((SHORT[label], load(p), case_dice(rows, label, case)))
        for j, ax in enumerate(axes[i]):
            ax.axis("off")
            if j >= len(panels):
                continue
            name, seg, d = panels[j]
            ax.imshow(sl, cmap="gray", interpolation="nearest")
            if seg is not None:
                ax.imshow(np.ma.masked_equal(seg[z], 0), cmap=cmap, vmin=0, vmax=3, alpha=0.5, interpolation="nearest")
            ax.set_title(name if d is None else f"{name}\nDice {d:.1f}", fontsize=9)
        axes[i][0].text(-0.04, 0.5, f"{title}\n{case} (fold {fold})", transform=axes[i][0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=9, color=MUTED)
    handles = [Patch(color=c, alpha=0.7, label=n) for n, c in CLASS_COLORS.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("ACDC examples (mid-ventricular slice): ground truth vs. each model's cross-validation prediction",
                 x=0.01, ha="left", fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.08, wspace=0.05); fig.savefig(out_png, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    res = os.path.expanduser(ap.parse_args().results)
    summary = json.load(open(os.path.join(res, "results_summary.json")))
    rows = list(csv.DictReader(open(os.path.join(res, "per_case_metrics.csv"))))
    os.makedirs(os.path.join(res, "figures"), exist_ok=True)
    per_fold_plot(summary, os.path.join(res, "figures", "per_fold_dice.png"))
    example_plot(rows, os.path.join(res, "figures", "example_segmentation.png"))
    print("written:", os.path.join(res, "figures", "per_fold_dice.png"),
          os.path.join(res, "figures", "example_segmentation.png"), sep="\n  ")


if __name__ == "__main__":
    main()
