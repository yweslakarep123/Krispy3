#!/bin/bash
# =============================================================
# FlowPolicy + Franka Kitchen — Full Environment Setup
# =============================================================
# GPU  : NVIDIA GeForce GTX 1080 (compute 6.1)
# Driver: 580.82.09 → backward-compatible with CUDA 11.7
# =============================================================
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "========================================"
echo " FlowPolicy Environment Setup"
echo " Repo: ${REPO_DIR}"
echo "========================================"

# ----------------------------------------------------------
# 1. Create conda environment
# ----------------------------------------------------------
echo ""
echo "[1/10] Creating conda environment (python 3.8) ..."
conda create -n flowpolicy python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate flowpolicy

# ----------------------------------------------------------
# 2. Install PyTorch 2.0.1 + CUDA 11.7
# ----------------------------------------------------------
echo ""
echo "[2/10] Installing PyTorch 2.0.1 (cu117) ..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu117

# ----------------------------------------------------------
# 3. Install CUDA toolkit via conda (needed for pytorch3d)
# ----------------------------------------------------------
echo ""
echo "[3/10] Installing CUDA toolkit 11.7 via conda ..."
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit -y

# ----------------------------------------------------------
# 4. Install MuJoCo 2.1.0
# ----------------------------------------------------------
echo ""
echo "[4/10] Installing MuJoCo 2.1.0 ..."
mkdir -p ~/.mujoco
if [ ! -d ~/.mujoco/mujoco210 ]; then
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
        -O /tmp/mujoco210.tar.gz --no-check-certificate
    tar -xzf /tmp/mujoco210.tar.gz -C ~/.mujoco/
    rm /tmp/mujoco210.tar.gz
    echo "MuJoCo 2.1.0 installed to ~/.mujoco/mujoco210"
else
    echo "MuJoCo 2.1.0 already present — skipping"
fi

# environment variables for MuJoCo
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/nvidia
export MUJOCO_GL=egl

# ----------------------------------------------------------
# 5. Install build prerequisites + mujoco-py
# ----------------------------------------------------------
echo ""
echo "[5/10] Installing build tools and mujoco-py ..."
pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

cd "${REPO_DIR}/third_party/mujoco-py-2.1.2.14"
pip install -e .
cd "${REPO_DIR}"

# ----------------------------------------------------------
# 6. Install simulation environments (gym, Metaworld, rrl)
# ----------------------------------------------------------
echo ""
echo "[6/10] Installing simulation environments ..."
cd "${REPO_DIR}/third_party/gym-0.21.0"  && pip install -e . && cd "${REPO_DIR}"
cd "${REPO_DIR}/third_party/Metaworld"   && pip install -e . && cd "${REPO_DIR}"
cd "${REPO_DIR}/third_party/rrl-dependencies" \
    && pip install -e mj_envs/. \
    && pip install -e mjrl/.  \
    && cd "${REPO_DIR}"

# ----------------------------------------------------------
# 7. Install pytorch3d (simplified, from third_party)
# ----------------------------------------------------------
echo ""
echo "[7/10] Installing pytorch3d ..."
cd "${REPO_DIR}/third_party/pytorch3d_simplified"
pip install -e .
cd "${REPO_DIR}"

# ----------------------------------------------------------
# 8. Install remaining Python packages
# ----------------------------------------------------------
echo ""
echo "[8/10] Installing remaining Python packages ..."
pip install \
    zarr==2.12.0 \
    wandb \
    ipdb \
    gpustat \
    dm_control \
    omegaconf \
    hydra-core==1.2.0 \
    dill==0.3.5.1 \
    einops==0.4.1 \
    diffusers==0.11.1 \
    numba==0.56.4 \
    moviepy \
    imageio \
    av \
    matplotlib \
    termcolor \
    natsort \
    open3d

# ----------------------------------------------------------
# 9. Install D4RL (for Franka Kitchen)
# ----------------------------------------------------------
echo ""
echo "[9/10] Installing D4RL for Franka Kitchen ..."
pip install git+https://github.com/Farama-Foundation/D4RL@master#egg=d4rl

# ----------------------------------------------------------
# 10. Install FlowPolicy itself
# ----------------------------------------------------------
echo ""
echo "[10/10] Installing FlowPolicy ..."
cd "${REPO_DIR}/FlowPolicy"
pip install -e .
cd "${REPO_DIR}"

# ----------------------------------------------------------
# Append environment variables to ~/.bashrc (if not present)
# ----------------------------------------------------------
echo ""
echo "Setting up environment variables in ~/.bashrc ..."

MARKER="# >>> flowpolicy env >>>"
if ! grep -q "${MARKER}" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'BASHEOF'

# >>> flowpolicy env >>>
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/nvidia
export MUJOCO_GL=egl
# <<< flowpolicy env <<<
BASHEOF
    echo "Environment variables appended to ~/.bashrc"
else
    echo "Environment variables already in ~/.bashrc — skipping"
fi

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------
echo ""
echo "========================================"
echo " Setup complete!"
echo "========================================"
echo ""
echo "Activate the environment:"
echo "  conda activate flowpolicy"
echo ""
echo "Then generate Franka Kitchen demos:"
echo "  bash scripts/gen_demonstration_franka_kitchen.sh"
echo ""
echo "Then train:"
echo "  bash scripts/train_policy.sh flowpolicy franka_kitchen 0001 0 0"
echo ""
