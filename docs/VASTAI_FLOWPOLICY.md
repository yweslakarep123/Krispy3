# Menjalankan FlowPolicy di vast.ai

Panduan praktis untuk GPU sewa. vast.ai hanya menyediakan VM + GPU; Anda mengatur software sendiri.

## 1. Pilih instance

- **Image:** `PyTorch 2.x + CUDA 11.8/12.x` (NVIDIA official atau vast template) atau Ubuntu 22.04 + driver NVIDIA.
- **Disk:** minimal **50 GB** jika Anda akan build conda, MuJoCo, D4RL, dan menyimpan zarr + checkpoint.
- **Port:** buka **22** (SSH). Opsional: **8888** (Jupyter) jika dipakai.

## 2. SSH dan persiapan sistem

```bash
ssh -p <port> root@<host_vast>
# atau user yang diberikan vast

apt-get update && apt-get install -y git wget build-essential \
  libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev \
  libx11-dev patchelf ffmpeg
```

## 3. MuJoCo 2.1 (untuk mujoco-py + D4RL kitchen)

```bash
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
source ~/.bashrc
```

Tanpa display: `MUJOCO_GL=egl` atau `osmesa` (lebih lambat).

## 4. Miniconda + environment

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
conda create -n flowpolicy python=3.8 -y
conda activate flowpolicy
```

Clone repo (HTTPS atau SSH):

```bash
cd /workspace   # atau $HOME
git clone <URL_REPO_ANDA> FlowPolicy
cd FlowPolicy
```

Ikuti `setup_env.sh` atau install manual seperti yang sudah Anda pakai di laptop (PyTorch CUDA, `mujoco-py`, `d4rl`, `dm_control`, `gym==0.21`, `huggingface_hub<0.24`, dll.).

## 5. Memindahkan data zarr

Dari laptop:

```bash
rsync -avz -e "ssh -p <port>" \
  /home/dafa/Documents/FlowPolicy/data/franka_kitchen_expert.zarr \
  root@<host_vast>:/workspace/FlowPolicy/data/
```

Atau upload ke cloud lalu `wget`/`aws s3 cp` di VM.

## 6. Training

```bash
conda activate flowpolicy
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_GL=egl
cd /workspace/FlowPolicy/FlowPolicy
wandb login   # opsional

python train.py --config-name=flowpolicy_franka_kitchen.yaml \
  task.dataset.zarr_path=/workspace/FlowPolicy/data/franka_kitchen_expert.zarr \
  training.device=cuda:0
```

Ganti path zarr sesuai lokasi di VM. **Selalu gunakan path absolut** untuk `zarr_path` karena Hydra mengubah working directory ke folder output.

## 7. Mengambil checkpoint

```bash
rsync -avz -e "ssh -p <port>" \
  root@<host_vast>:/workspace/FlowPolicy/data/outputs/<run>/checkpoints/ \
  ./checkpoints_backup/
```

## 8. Troubleshooting

| Gejala | Tindakan |
|--------|----------|
| `ImportError` mujoco_py | Pastikan `LD_LIBRARY_PATH` dan tarball `mujoco210` benar. |
| D4RL kitchen XML error | Pakai `mujoco<3` dan `dm_control<1.0.15` seperti di mesin lokal. |
| CUDA out of memory | Kurangi `batch_size` atau `horizon` lewat override Hydra. |
| WandB timeout | `export WANDB_MODE=offline` atau nonaktifkan logger di config. |
