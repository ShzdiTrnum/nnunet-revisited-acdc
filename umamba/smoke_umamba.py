"""Step 2b: will each model fit and how fast is it? (one GPU, ~5 min, no real training)

Builds U-Mamba Bot, U-Mamba Enc and SwinUNETR exactly as their nnU-Net trainers do (same plans, same
patch size, same batch size, same precision) and runs a few forward/backward passes on random data.
Reports peak GPU memory and seconds per iteration -> rough time per epoch (250 iterations) and per fold
(1000 epochs). If a model does not fit at the planned batch size, it also reports the largest batch that fits.
Usage: CUDA_VISIBLE_DEVICES=1 python ~/umamba_repro/smoke_umamba.py
"""
import json, os, time
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEnc import nnUNetTrainerUMambaEnc
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinUNETR import nnUNetTrainerSwinUNETR

PRE = os.path.join(os.environ["nnUNet_preprocessed"], "Dataset027_ACDC")
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def pad32(ps):  # what nnUNetTrainerSwinUNETR does to the patch size
    return [p if (p / 32) >= 1 and (p / 32) % 1 == 0 else round(p / 32 + 0.5) * 32 for p in ps]


def run(name, trainer_cls, batch, amp, deep_sup, swin):
    pm = PlansManager(os.path.join(PRE, "nnUNetPlans.json"))
    cm = pm.get_configuration("3d_fullres")
    if swin:
        cm.configuration["patch_size"] = pad32(cm.patch_size)
    ds_json = load_json(os.path.join(PRE, "dataset.json"))
    net = trainer_cls.build_network_architecture(pm, ds_json, cm, 1, enable_deep_supervision=deep_sup).cuda()
    opt = torch.optim.SGD(net.parameters(), lr=1e-3)
    x = torch.randn(batch, 1, *cm.patch_size, device="cuda")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(6):
        torch.cuda.synchronize(); t = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp):
            out = net(x)
            outs = out if isinstance(out, (list, tuple)) else [out]
            loss = sum(o.float().mean() for o in outs)
        loss.backward(); opt.step()
        torch.cuda.synchronize()
        if i >= 2:                                   # skip warm-up iterations
            times.append(time.time() - t)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    del net, opt, x, out, outs, loss
    torch.cuda.empty_cache()
    return cm.patch_size, peak, sum(times) / len(times)


plans = json.load(open(os.path.join(PRE, "nnUNetPlans.json")))["configurations"]["3d_fullres"]
B = plans["batch_size"]
print(f"GPU: {torch.cuda.get_device_name(0)} ({GPU_GB:.1f} GB) | planned 3d_fullres batch {B}, patch {plans['patch_size']}\n")
MODELS = [("U-Mamba Bot", nnUNetTrainerUMambaBot, True, True, False),
          ("U-Mamba Enc", nnUNetTrainerUMambaEnc, True, True, False),
          ("SwinUNETR",   nnUNetTrainerSwinUNETR, False, False, True)]   # U-Mamba's SwinUNETR trainer: fp32, no deep supervision
for name, cls, amp, ds, swin in MODELS:
    b = B
    while b >= 1:
        try:
            ps, peak, it = run(name, cls, b, amp, ds, swin)
            note = "planned batch" if b == B else f"DOES NOT FIT at planned batch {B}; largest batch that fits"
            print(f"{name:12s} batch {b} ({note}) | patch {ps} | {'fp16 AMP' if amp else 'fp32'} | "
                  f"peak {peak:.1f} GB of {GPU_GB:.1f} | {it:.2f} s/iter -> ~{250 * it:.0f} s/epoch "
                  f"-> ~{250 * it * 1000 / 3600:.0f} h per fold (GPU compute only)", flush=True)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{name:12s} batch {b}: out of GPU memory", flush=True)
            b -= 1
print("\nSMOKE TEST DONE")
