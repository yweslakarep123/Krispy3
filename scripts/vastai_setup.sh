#!/bin/bash
# ================================================================
# Vast.ai Instance Setup Script (tanpa Docker)
# Jalankan ini setelah SSH ke instance vast.ai
# ================================================================
set -e

echo "========================================"
echo " FlowPolicy - Vast.ai Setup"
echo "========================================"

apt-get update && apt-get install -y --no-install-recommends \
    wget curl git build-essential gcc g++ \
    libx11-dev libgl1-mesa-dev libegl1-mesa-dev libglew-dev \
    libosmesa6-dev libglu1-mesa-dev \
    patchelf unzip ca-certificates

echo "[1/8] Installing MuJoCo 2.1.0 ..."
mkdir -p ~/.mujoco
if [ ! -d ~/.mujoco/mujoco210 ]; then
    wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
        -O /tmp/mujoco210.tar.gz --no-check-certificate
    tar -xzf /tmp/mujoco210.tar.gz -C ~/.mujoco/
    rm /tmp/mujoco210.tar.gz
fi

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1

cat >> ~/.bashrc << 'EOF'
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
EOF

echo "[2/8] Installing Miniconda ..."
if ! command -v conda &> /dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
else
    eval "$(conda shell.bash hook)"
fi

echo "[3/8] Creating conda environment ..."
conda create -n flowpolicy python=3.8 -y
conda activate flowpolicy

echo "[4/8] Installing PyTorch 2.0.1 + CUDA 11.7 ..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu117

echo "[5/8] Installing build tools and third-party packages ..."
pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}/third_party/mujoco-py-2.1.2.14" && pip install -e .
cd "${REPO_DIR}/third_party/gym-0.21.0" && pip install -e .
cd "${REPO_DIR}/third_party/Metaworld" && pip install -e .
cd "${REPO_DIR}/third_party/rrl-dependencies" && pip install -e mj_envs/. && pip install -e mjrl/.

echo "[6/8] Installing pytorch3d ..."
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit -y
cd "${REPO_DIR}/third_party/pytorch3d_simplified" && pip install -e .

echo "[7/8] Installing remaining packages ..."
pip install \
    zarr==2.12.0 wandb ipdb gpustat \
    "mujoco<3.0" "dm_control<1.0.15" \
    omegaconf hydra-core==1.2.0 \
    dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 "huggingface_hub<0.24" \
    numba==0.56.4 moviepy imageio av matplotlib termcolor natsort open3d

pip install git+https://github.com/Farama-Foundation/D4RL@master#egg=d4rl

echo "[8/8] Installing FlowPolicy ..."
cd "${REPO_DIR}/FlowPolicy" && pip install -e .

echo ""
echo "========================================"
echo " Setup complete!"
echo "========================================"
echo " conda activate flowpolicy"
echo " bash scripts/gen_demonstration_franka_kitchen.sh"
echo " bash scripts/train_policy.sh flowpolicy_kitchen franka_kitchen 0001 0 0"
echo "========================================"
