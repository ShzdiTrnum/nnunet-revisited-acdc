"""Evaluate all reproduced models on ACDC in one table: nnU-Net (org.), ResEnc M/L, U-Mamba Bot/Enc, SwinUNETR.

Same metrics and aggregation as evaluate_acdc.py (and as nnU-Net Revisited): Dice per class and NSD @ 2 mm
(google-deepmind/surface-distance); per fold mean over cases and the 3 classes, then mean +- SD over folds.
Only folds whose training + validation finished (validation/summary.json) are used, so it is safe to run while
other models are still training. Read-only on all result folders; CPU only; writes to --out.

Run in the nnU-Net environment (it has SimpleITK + surface-distance), at low priority:
  source ~/nnunet_repro/env.sh
  nice -n 19 python ~/nnunet_repro/scripts/evaluate_all.py --out ~/nnunet_repro/results_all -np 4
"""
import argparse, csv, json, os
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk
import surface_distance as sd

CLASSES = {1: "RV", 2: "MYO", 3: "LV"}
TOL_MM = 2.0
# label, project ("nnunet" | "umamba"), results sub-folder, paper Dice (Table 1), paper NSD (Table 4)
MODELS = [
    ("nnU-Net (org.)",   "nnunet", "nnUNetTrainer__nnUNetPlans__3d_fullres",              91.54, 94.93),
    ("nnU-Net ResEnc M", "nnunet", "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",   91.99, 95.50),
    ("nnU-Net ResEnc L", "nnunet", "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",   91.69, 95.11),
    ("U-Mamba Bot",      "umamba", "nnUNetTrainerUMambaBot__nnUNetPlans__3d_fullres",     91.79, 95.40),
    ("U-Mamba Enc",      "umamba", "nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres",     91.22, 94.33),
    ("SwinUNETR",        "umamba", "nnUNetTrainerSwinUNETR__nnUNetPlans_bs3__3d_fullres", 91.29, 94.12),
]
BASELINE = "nnU-Net (org.)"


def read(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img), tuple(float(s) for s in reversed(img.GetSpacing()))  # (z, y, x)


def eval_case(args):
    model, fold, pred_path, gt_dir = args
    case = os.path.basename(pred_path)[:-len(".nii.gz")]
    pred, sp_p = read(pred_path)
    gt, sp_g = read(os.path.join(gt_dir, case + ".nii.gz"))
    assert pred.shape == gt.shape, f"{model} {case}: shape mismatch {pred.shape} vs {gt.shape}"
    assert np.allclose(sp_p, sp_g, atol=1e-3), f"{model} {case}: spacing mismatch {sp_p} vs {sp_g}"
    row = {"model": model, "fold": fold, "case": case}
    for lab, name in CLASSES.items():
        g, p = gt == lab, pred == lab
        if not g.any() and not p.any():
            row[f"Dice_{name}"], row[f"NSD_{name}"] = np.nan, np.nan
            continue
        row[f"Dice_{name}"] = 2.0 * np.logical_and(g, p).sum() / (g.sum() + p.sum())
        if not g.any() or not p.any():
            row[f"NSD_{name}"] = 0.0
            continue
        row[f"NSD_{name}"] = sd.compute_surface_dice_at_tolerance(sd.compute_surface_distances(g, p, spacing_mm=sp_g), TOL_MM)
    return row


def main():
    home = os.path.expanduser("~")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("-np", type=int, default=4)
    ap.add_argument("--nnunet_results", default=os.path.join(home, "nnunet_repro/data/nnUNet_results/Dataset027_ACDC"))
    ap.add_argument("--umamba_results", default=os.path.join(home, "umamba_repro/data/nnUNet_results/Dataset027_ACDC"))
    ap.add_argument("--gt", default=os.path.join(home, "nnunet_repro/data/nnUNet_raw/Dataset027_ACDC/labelsTr"))
    a = ap.parse_args()
    out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True)
    roots = {"nnunet": a.nnunet_results, "umamba": a.umamba_results}

    jobs, own_dice, val_sets = [], {}, {}
    for label, proj, sub, _, _ in MODELS:
        for f in range(5):
            vdir = os.path.join(roots[proj], sub, f"fold_{f}", "validation")
            if not os.path.isfile(os.path.join(vdir, "summary.json")):
                continue
            own_dice[(label, f)] = json.load(open(os.path.join(vdir, "summary.json")))["foreground_mean"]["Dice"]
            cases = sorted(x for x in os.listdir(vdir) if x.endswith(".nii.gz"))
            val_sets.setdefault(f, set(cases))
            assert set(cases) == val_sets[f], f"{label} fold {f}: different validation cases than other models"
            jobs += [(label, f, os.path.join(vdir, x), a.gt) for x in cases]
    print(f"evaluating {len(jobs)} predictions from {len(own_dice)} finished folds "
          f"(all models share the same validation cases per fold: checked) ...", flush=True)
    with Pool(a.np) as pool:
        rows = pool.map(eval_case, jobs, chunksize=4)

    keys = ["model", "fold", "case"] + [f"{m}_{c}" for m in ("Dice", "NSD") for c in CLASSES.values()]
    with open(os.path.join(out, "per_case_metrics_all.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

    summary, max_diff = {}, 0.0
    for label, proj, sub, pd, pn in MODELS:
        folds = sorted({r["fold"] for r in rows if r["model"] == label})
        if not folds:
            continue
        pf = {}
        for f in folds:
            rs = [r for r in rows if r["model"] == label and r["fold"] == f]
            pf[f] = {k: float(np.nanmean([r[k] for r in rs])) for k in keys[3:]}
            pf[f]["Dice"] = float(np.mean([pf[f][f"Dice_{c}"] for c in CLASSES.values()]))
            pf[f]["NSD"] = float(np.mean([pf[f][f"NSD_{c}"] for c in CLASSES.values()]))
            max_diff = max(max_diff, abs(pf[f]["Dice"] - own_dice[(label, f)]))
        ms = lambda k: (100 * np.mean([pf[f][k] for f in folds]),
                        100 * np.std([pf[f][k] for f in folds], ddof=1) if len(folds) > 1 else float("nan"))
        summary[label] = {"folds": folds, "per_fold": pf, "paper_Dice": pd, "paper_NSD": pn,
                          **{k: ms(k) for k in ["Dice", "NSD"] + keys[3:]}}

    L = []
    L.append("| Model | Folds | Dice RV | Dice MYO | Dice LV | **Mean Dice** (± SD) | Paper Dice | Gap (in fold SDs) | NSD @2 mm (± SD) | Paper NSD |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for label, *_ in MODELS:
        if label not in summary:
            L.append(f"| {label} | 0/5 | – | – | – | – | {[m for m in MODELS if m[0] == label][0][3]:.2f} | – | – | {[m for m in MODELS if m[0] == label][0][4]:.2f} |")
            continue
        s = summary[label]; d, dsd = s["Dice"]; n, nsd = s["NSD"]; gap = d - s["paper_Dice"]
        gap_sd = f"{gap:+.2f} ({abs(gap) / dsd:.2f} SD)" if dsd == dsd and dsd > 0 else f"{gap:+.2f}"
        pm = lambda m, v: f"{m:.2f} ± {v:.2f}" if v == v else f"{m:.2f} (1 fold)"
        L.append(f"| {label} | {len(s['folds'])}/5 | {s['Dice_RV'][0]:.2f} | {s['Dice_MYO'][0]:.2f} | {s['Dice_LV'][0]:.2f} | "
                 f"**{pm(d, dsd)}** | {s['paper_Dice']:.2f} | {gap_sd} | {pm(n, nsd)} | {s['paper_NSD']:.2f} |")

    # paired comparison with the baseline on the folds both have finished
    P = ["", f"Paired comparison with {BASELINE} (same folds, mean Dice per fold):", "",
         "| Model | Folds compared | Mean difference | Folds better | Paper difference | Larger than fold SD? |", "|---|---|---|---|---|---|"]
    base = summary.get(BASELINE)
    for label, *_ in MODELS:
        if label == BASELINE or label not in summary or base is None:
            continue
        common = [f for f in summary[label]["folds"] if f in base["folds"]]
        if not common:
            continue
        diffs = [100 * (summary[label]["per_fold"][f]["Dice"] - base["per_fold"][f]["Dice"]) for f in common]
        paper = summary[label]["paper_Dice"] - base["paper_Dice"]
        P.append(f"| {label} | {len(common)} | {np.mean(diffs):+.2f} | {sum(x > 0 for x in diffs)}/{len(common)} | {paper:+.2f} | "
                 f"{'yes' if abs(np.mean(diffs)) > base['Dice'][1] else 'no'} |")

    note = (f"\nDice/NSD in %, 5-fold CV on the 100 ACDC training patients (200 ED/ES volumes), official nnU-Net split. "
            f"Cross-check vs each model's own nnU-Net summary.json: max per-fold Dice difference = {100 * max_diff:.3f} points.\n")
    text = "\n".join(L + P) + "\n" + note
    open(os.path.join(out, "results_table_all.md"), "w").write(text)
    json.dump(summary, open(os.path.join(out, "results_summary_all.json"), "w"), indent=1, default=float)
    print(text)
    print("written to", out)


if __name__ == "__main__":
    main()
