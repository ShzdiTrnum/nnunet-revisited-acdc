#!/usr/bin/env bash
# U-Mamba / SwinUNETR on ACDC, Step 1: separate environment + feasibility check (no training).
# Safe to re-run: every part checks whether it is already done and skips it.
# Run INSIDE tmux:  bash ~/umamba_repro/setup_umamba.sh
# Does not touch ~/nnunet_repro (only reads its raw ACDC data and its official split).
set -euo pipefail

R="$HOME/umamba_repro"
SRC_ENV="nnunet_repro"          # finished nnU-Net 2.8.1 env (PyTorch 2.5.1, CUDA 12.1, Python 3.10)
ENV_NAME="umamba_repro"
UMAMBA_COMMIT="28459e3"         # bowang-lab/U-Mamba main, 2024-04-04
MAMBA_V="2.2.4"; CCONV_V="1.5.0.post8"
mkdir -p "$R"/{wheels,logs,data/nnUNet_preprocessed,data/nnUNet_results}
LOG="$R/logs/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== U-Mamba setup started $(date) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. separate conda env: offline clone of the working nnU-Net env (no PyTorch download) ----------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then echo "[skip] env $ENV_NAME exists"
else conda create -n "$ENV_NAME" --clone "$SRC_ENV" --offline -y; fi
conda activate "$ENV_NAME"
python - << 'PY'
import sys, torch, numpy
print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA build", torch.version.cuda,
      "| cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI, "| numpy", numpy.__version__)
PY
ABI=$(python -c "import torch;print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")

# 2. prebuilt Mamba CUDA kernels (no compiler needed), resumable download ---------------------------
TAG="cu12torch2.5cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"
W1="mamba_ssm-${MAMBA_V}+${TAG}"; W2="causal_conv1d-${CCONV_V}+${TAG}"
wget -c -nv --show-progress -P "$R/wheels" "https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CCONV_V}/${W2}"
wget -c -nv --show-progress -P "$R/wheels" "https://github.com/state-spaces/mamba/releases/download/v${MAMBA_V}/${W1}"
if python -c "import mamba_ssm, causal_conv1d" 2>/dev/null; then echo "[skip] mamba_ssm + causal_conv1d installed"
else pip install "$R/wheels/$W2" "$R/wheels/$W1"; fi

# 3. U-Mamba code (bundles nnU-Net 2.1.1), pinned; make paths.py read the environment variables -------
if [ ! -d "$R/U-Mamba/.git" ]; then git clone https://github.com/bowang-lab/U-Mamba.git "$R/U-Mamba"; fi
git -C "$R/U-Mamba" checkout -q "$UMAMBA_COMMIT"
P="$R/U-Mamba/umamba/nnunetv2/paths.py"
if grep -q "^nnUNet_raw = os.environ" "$P"; then echo "[skip] paths.py already patched"
else
  sed -i -E "s#^(nnUNet_raw|nnUNet_preprocessed|nnUNet_results) = join\(base, '[A-Za-z_]+'\).*\$#\1 = os.environ.get('\1')#" "$P"
  echo "[done] paths.py now reads nnUNet_raw / nnUNet_preprocessed / nnUNet_results from the environment"
fi
grep -n "^nnUNet_" "$P"
if pip show nnunetv2 2>/dev/null | grep -q "^Version: 2.1.1"; then echo "[skip] U-Mamba's nnunetv2 2.1.1 installed"
else pip uninstall -y nnunetv2 || true; pip install -e "$R/U-Mamba/umamba"; fi

# 4. environment file + data folders (ACDC raw data is shared read-only with ~/nnunet_repro) ---------
cat > "$R/env_umamba.sh" << EOF
# Load with: source ~/umamba_repro/env_umamba.sh   (never in the same terminal as ~/nnunet_repro/env.sh)
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
export nnUNet_raw="$HOME/nnunet_repro/data/nnUNet_raw"
export nnUNet_preprocessed="$R/data/nnUNet_preprocessed"
export nnUNet_results="$R/data/nnUNet_results"
export nnUNet_n_proc_DA=12
export CUDA_DEVICE_ORDER=PCI_BUS_ID
EOF
source "$R/env_umamba.sh"

# 5. the SAME official patient-level split as the nnU-Net runs (must exist before any training) -----
mkdir -p "$nnUNet_preprocessed/Dataset027_ACDC"
SPLIT_SRC="$HOME/nnunet_repro/data/nnUNet_preprocessed/Dataset027_ACDC/splits_final.json"
SPLIT_DST="$nnUNet_preprocessed/Dataset027_ACDC/splits_final.json"
[ -f "$SPLIT_DST" ] || cp "$SPLIT_SRC" "$SPLIT_DST"
if cmp -s "$SPLIT_SRC" "$SPLIT_DST"; then echo "[ok] splits_final.json identical to the nnU-Net runs"
else echo "ERROR: splits_final.json differs from the nnU-Net runs"; exit 1; fi

# 6. record environment and run the checks on GPU 1 -------------------------------------------------
{ date; echo "U-Mamba commit: $(git -C "$R/U-Mamba" rev-parse HEAD)"
  pip list 2>/dev/null | grep -iE "^torch |^numpy |nnunetv2|mamba|causal|monai|dynamic-network|batchgenerators|acvl|triton|transformers"
} > "$R/logs/env_info.txt"; cat "$R/logs/env_info.txt"
CUDA_VISIBLE_DEVICES=1 python "$R/check_umamba.py"
echo "=== U-Mamba setup finished $(date). Log: $LOG ==="
