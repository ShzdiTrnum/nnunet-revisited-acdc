"""Step 5: evaluate the 5-fold cross-validation exactly as reported in nnU-Net Revisited (Isensee et al., 2024).

Metrics (paper, Table 1 / Table 4):
  - Dice (DSC) per class: RV, MYO, LV
  - Normalized Surface Distance (NSD) at 2 mm tolerance, computed with google-deepmind/surface-distance
  - "results are averaged over all classes ... as well as over the five folds"
    -> per fold: mean over cases per class, then mean over the 3 classes; final = mean (+- SD) over folds.

Inputs: nnU-Net's own validation predictions (fold_X/validation/*.nii.gz, original image space)
        and the ground truth in nnUNet_raw/Dataset027_ACDC/labelsTr.
Only finished folds (validation/summary.json present) are evaluated, so it can run while training continues.
CPU only. Usage:  python evaluate_acdc.py --out ~/nnunet_repro/repo/results
"""
import argparse, csv, json, os
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk
import surface_distance as sd

RES = os.path.join(os.environ["nnUNet_results"], "Dataset027_ACDC")
GT_DIR = os.path.join(os.environ["nnUNet_raw"], "Dataset027_ACDC", "labelsTr")
CLASSES = {1: "RV", 2: "MYO", 3: "LV"}
MODELS = [("nnU-Net (org.)", "nnUNetPlans", 91.54),
          ("nnU-Net ResEnc M", "nnUNetResEncUNetMPlans", 91.99),
          ("nnU-Net ResEnc L", "nnUNetResEncUNetLPlans", 91.69)]
TOL_MM = 2.0


def read(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img), tuple(float(s) for s in reversed(img.GetSpacing()))  # (z, y, x)


def eval_case(args):
    model, fold, pred_path = args
    case = os.path.basename(pred_path)[:-len(".nii.gz")]
    pred, sp_p = read(pred_path)
    gt, sp_g = read(os.path.join(GT_DIR, case + ".nii.gz"))
    assert pred.shape == gt.shape, f"{case}: shape mismatch {pred.shape} vs {gt.shape}"
    assert np.allclose(sp_p, sp_g, atol=1e-3), f"{case}: spacing mismatch {sp_p} vs {sp_g}"
    row = {"model": model, "fold": fold, "case": case}
    for lab, name in CLASSES.items():
        g, p = gt == lab, pred == lab
        if not g.any() and not p.any():                 # class absent in both: undefined, ignored in means
            row[f"Dice_{name}"], row[f"NSD_{name}"] = np.nan, np.nan
            continue
        inter = np.logical_and(g, p).sum()
        row[f"Dice_{name}"] = 2.0 * inter / (g.sum() + p.sum())
        if not g.any() or not p.any():                  # one of them empty: no overlap of surfaces
            row[f"NSD_{name}"] = 0.0
            continue
        dists = sd.compute_surface_distances(g, p, spacing_mm=sp_g)
        row[f"NSD_{name}"] = sd.compute_surface_dice_at_tolerance(dists, TOL_MM)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("-np", type=int, default=4)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)

    jobs, nnunet_dice = [], {}
    for label, plans, _ in MODELS:
        for f in range(5):
            vdir = os.path.join(RES, f"nnUNetTrainer__{plans}__3d_fullres", f"fold_{f}", "validation")
            if not os.path.isfile(os.path.join(vdir, "summary.json")):
                continue
            nnunet_dice[(label, f)] = json.load(open(os.path.join(vdir, "summary.json")))["foreground_mean"]["Dice"]
            jobs += [(label, f, os.path.join(vdir, x)) for x in sorted(os.listdir(vdir)) if x.endswith(".nii.gz")]
    print(f"evaluating {len(jobs)} predictions from {len(nnunet_dice)} finished folds ...", flush=True)
    with Pool(a.np) as pool:
        rows = pool.map(eval_case, jobs, chunksize=4)

    keys = ["model", "fold", "case"] + [f"{m}_{c}" for m in ("Dice", "NSD") for c in CLASSES.values()]
    with open(os.path.join(out, "per_case_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

    summary, lines, max_diff = {}, [], 0.0
    for label, plans, paper in MODELS:
        folds = sorted({r["fold"] for r in rows if r["model"] == label})
        if not folds:
            continue
        per_fold = {}
        for f in folds:
            rs = [r for r in rows if r["model"] == label and r["fold"] == f]
            per_fold[f] = {k: float(np.nanmean([r[k] for r in rs])) for k in keys[3:]}
            per_fold[f]["Dice"] = float(np.mean([per_fold[f][f"Dice_{c}"] for c in CLASSES.values()]))
            per_fold[f]["NSD"] = float(np.mean([per_fold[f][f"NSD_{c}"] for c in CLASSES.values()]))
            max_diff = max(max_diff, abs(per_fold[f]["Dice"] - nnunet_dice[(label, f)]))
        agg = lambda k: (100 * np.mean([per_fold[f][k] for f in folds]), 100 * np.std([per_fold[f][k] for f in folds], ddof=1) if len(folds) > 1 else 0.0)
        summary[label] = {"folds": folds, "per_fold": per_fold, "paper_DSC": paper,
                          **{k: agg(k) for k in ["Dice", "NSD"] + keys[3:]}}
        s = summary[label]
        lines.append(f"| {label} | {len(folds)}/5 | {s['Dice_RV'][0]:.2f} | {s['Dice_MYO'][0]:.2f} | {s['Dice_LV'][0]:.2f} | "
                     f"**{s['Dice'][0]:.2f} ± {s['Dice'][1]:.2f}** | {s['NSD'][0]:.2f} ± {s['NSD'][1]:.2f} | {paper:.2f} |")

    table = ("| Model | Folds | Dice RV | Dice MYO | Dice LV | Mean Dice (± SD over folds) | NSD @2 mm | Paper Dice |\n"
             "|---|---|---|---|---|---|---|---|\n" + "\n".join(lines))
    note = (f"\nDice/NSD in %, 5-fold CV on the 100 ACDC training patients (200 ED/ES volumes), "
            f"per-fold mean over cases and classes, then mean over folds. "
            f"Cross-check vs nnU-Net's own summary.json: max per-fold Dice difference = {100 * max_diff:.3f} points.\n")
    open(os.path.join(out, "results_table.md"), "w").write(table + "\n" + note)
    json.dump(summary, open(os.path.join(out, "results_summary.json"), "w"), indent=1, default=float)
    print(table); print(note)
    print("written:", *(os.path.join(out, x) for x in ("results_table.md", "results_summary.json", "per_case_metrics.csv")), sep="\n  ")


if __name__ == "__main__":
    main()
