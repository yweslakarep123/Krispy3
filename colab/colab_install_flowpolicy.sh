#!/usr/bin/env bash
# Jalankan di dalam conda env flowpolicy (Python 3.8), setelah apt + MuJoCo.
# Contoh Colab:
#   export REPO_DIR=/content/FlowPolicy
#   bash colab/colab_install_flowpolicy.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/FlowPolicy}"
cd "${REPO_DIR}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

echo "[colab] REPO_DIR=${REPO_DIR} MUJOCO_GL=${MUJOCO_GL}"

# PyTorch + CUDA (wheel cu118, kompatibel Python 3.8 & driver Colab)
pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -q setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

if [ -d "${REPO_DIR}/third_party/mujoco-py-2.1.2.14" ]; then
  (cd "${REPO_DIR}/third_party/mujoco-py-2.1.2.14" && pip install -q -e .)
else
  echo "Missing third_party/mujoco-py-2.1.2.14 — pastikan clone repo lengkap."
  exit 1
fi

(cd "${REPO_DIR}/third_party/gym-0.21.0" && pip install -q -e .)

if [ -d "${REPO_DIR}/third_party/Metaworld" ]; then
  (cd "${REPO_DIR}/third_party/Metaworld" && pip install -q -e .)
fi
if [ -d "${REPO_DIR}/third_party/rrl-dependencies" ]; then
  (cd "${REPO_DIR}/third_party/rrl-dependencies" && pip install -q -e mj_envs/. && pip install -q -e mjrl/.)
fi

(cd "${REPO_DIR}/third_party/pytorch3d_simplified" && pip install -q -e .)

pip install -q \
  zarr==2.12.0 wandb ipdb omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 \
  diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor natsort open3d

pip install -q "huggingface_hub<0.24"
pip install -q "mujoco<3.0" "dm_control<1.0.15"

pip install -q "git+https://github.com/Farama-Foundation/D4RL@master#egg=d4rl"

(cd "${REPO_DIR}/FlowPolicy" && pip install -q -e .)

echo "[colab] Selesai. Uji: python -c \"import gym; import d4rl; print('ok')\""
