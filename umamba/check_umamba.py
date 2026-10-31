"""Feasibility check for U-Mamba / SwinUNETR on ACDC (runs on one GPU in about a minute, no training).
Usage (after source ~/umamba_repro/env_umamba.sh):  CUDA_VISIBLE_DEVICES=1 python ~/umamba_repro/check_umamba.py
"""
import os, sys, traceback

ok = True
def report(name, passed, info=""):
    global ok
    ok &= bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {info}", flush=True)

def step(name, fn):
    try:
        report(name, True, fn() or "")
    except Exception as e:
        report(name, False, f"{type(e).__name__}: {e}")
        traceback.print_exc(limit=2)

import torch, numpy
report("PyTorch + CUDA", torch.cuda.is_available(), f"torch {torch.__version__}, numpy {numpy.__version__}")

# 1. the nnunetv2 in this env must be U-Mamba's 2.1.1 copy, with paths taken from the environment
def nnunet_copy():
    import nnunetv2, nnunetv2.paths as p
    from importlib.metadata import version
    loc = os.path.dirname(nnunetv2.__file__)
    assert "U-Mamba" in loc, f"wrong nnunetv2 imported from {loc}"
    for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        assert getattr(p, k) == os.environ.get(k), f"{k}: paths.py={getattr(p, k)} env={os.environ.get(k)}"
    return f"nnunetv2 {version('nnunetv2')} from {loc}; paths follow the environment"
step("U-Mamba's nnU-Net + paths", nnunet_copy)

# 2. same split as the nnU-Net runs, and the raw ACDC data is visible
def split():
    import json
    s = json.load(open(os.path.join(os.environ["nnUNet_preprocessed"], "Dataset027_ACDC", "splits_final.json")))
    n = len(os.listdir(os.path.join(os.environ["nnUNet_raw"], "Dataset027_ACDC", "labelsTr")))
    assert len(s) == 5 and all(len(f["val"]) == 40 for f in s) and n == 200
    return "5 folds x 40 validation cases, 200 labels in nnUNet_raw"
step("official split + raw data", split)

# 3. the three trainers import
def trainers():
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEnc import nnUNetTrainerUMambaEnc
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinUNETR import nnUNetTrainerSwinUNETR
    return "nnUNetTrainerUMambaBot, nnUNetTrainerUMambaEnc, nnUNetTrainerSwinUNETR"
step("trainers import", trainers)

# 4. Mamba CUDA kernels: forward + backward on the GPU
def mamba():
    from mamba_ssm import Mamba
    import mamba_ssm, causal_conv1d
    m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
    x = torch.randn(2, 512, 64, device="cuda", requires_grad=True)
    y = m(x); y.mean().backward()
    assert torch.isfinite(y).all() and x.grad is not None
    return f"mamba_ssm {mamba_ssm.__version__}, causal_conv1d {causal_conv1d.__version__}, output {tuple(y.shape)}"
step("Mamba layer on GPU", mamba)

# 5. MONAI SwinUNETR at the padded ACDC patch size used by U-Mamba's trainer (20 -> 32 slices)
def swin():
    import monai
    from monai.networks.nets import SwinUNETR
    net = SwinUNETR(img_size=(32, 256, 224), in_channels=1, out_channels=4, feature_size=48,
                    norm_name="instance", spatial_dims=3, downsample="merging", use_v2=False).cuda()
    x = torch.randn(1, 1, 32, 256, 224, device="cuda")
    y = net(x); y.mean().backward()
    mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    return f"monai {monai.__version__}, output {tuple(y.shape)}, peak memory {mem:.1f} GB (batch 1)"
step("SwinUNETR on GPU", swin)

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED - send me the output above")
sys.exit(0 if ok else 1)
