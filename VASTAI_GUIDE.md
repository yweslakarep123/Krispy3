# Panduan Vast.ai untuk FlowPolicy Franka Kitchen

## 1. Persiapan di Laptop Lokal

### Compress codebase (tanpa data besar)
```bash
cd ~/Documents
tar --exclude='*.zarr' --exclude='data/' --exclude='.git' \
    --exclude='__pycache__' --exclude='results/' \
    -czf FlowPolicy_code.tar.gz FlowPolicy/
```

### Jika sudah punya data zarr, compress terpisah
```bash
cd ~/Documents/FlowPolicy
tar -czf franka_kitchen_data.tar.gz data/franka_kitchen_expert.zarr
```

## 2. Rent Instance di Vast.ai

1. Buka https://vast.ai dan buat akun
2. Klik **Search** atau **Create** untuk mencari GPU
3. Filter rekomendasi:
   - **GPU**: RTX 3090 atau RTX 4090 (24GB VRAM)
   - **Disk**: minimal 50GB
   - **Image**: `nvidia/cuda:11.7.1-devel-ubuntu20.04`
   - Atau pilih **PyTorch** template
4. Klik **Rent** pada instance yang dipilih
5. Tunggu instance running, lalu klik **Connect** untuk mendapat SSH command

## 3. Upload Codebase ke Instance

```bash
# Dari laptop lokal:
scp -P <PORT> FlowPolicy_code.tar.gz root@<VAST_IP>:/workspace/
scp -P <PORT> franka_kitchen_data.tar.gz root@<VAST_IP>:/workspace/  # jika ada

# SSH ke instance:
ssh -p <PORT> root@<VAST_IP>
```

## 4. Setup di Instance Vast.ai

```bash
cd /workspace
tar -xzf FlowPolicy_code.tar.gz
cd FlowPolicy

# Jalankan setup otomatis
sudo bash scripts/vastai_setup.sh
```

Setup ini memakan waktu ~15-20 menit. Script akan:
- Install MuJoCo 2.1.0
- Buat conda env `flowpolicy` (Python 3.8)
- Install PyTorch 2.0.1 + CUDA 11.7
- Install semua dependensi (mujoco-py, gym, Metaworld, pytorch3d, d4rl, dll)

## 5. Generate Data Demonstrasi

```bash
conda activate flowpolicy
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl

# Generate dari 3 dataset D4RL (~30 menit)
bash scripts/gen_demonstration_franka_kitchen.sh
```

Atau jika sudah upload data zarr:
```bash
cd /workspace/FlowPolicy
tar -xzf /workspace/franka_kitchen_data.tar.gz
```

## 6. Training

```bash
conda activate flowpolicy
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl

# Login wandb (opsional, untuk monitoring)
wandb login

# Jalankan training dengan config kitchen
bash scripts/train_policy.sh flowpolicy_kitchen franka_kitchen 0001 0 0
```

### Parameter training:
- `flowpolicy_kitchen` = nama config (model kecil ~5M params, horizon 16)
- `franka_kitchen` = nama task
- `0001` = experiment ID
- `0` = seed
- `0` = GPU ID

### Monitoring via WandB:
Buka https://wandb.ai untuk melihat loss curve dan video evaluasi.

## 7. Download Checkpoint

Setelah training selesai, download checkpoint terbaik:

```bash
# Dari laptop lokal:
scp -P <PORT> root@<VAST_IP>:/workspace/FlowPolicy/FlowPolicy/data/outputs/*/checkpoints/latest.ckpt ./
```

## 8. Tips

- **Jangan lupa destroy instance** setelah selesai agar tidak terus bayar
- **Save checkpoint** secara berkala — instance bisa timeout
- Training ~5000 epoch dengan RTX 3090 diperkirakan **4-8 jam**
- Gunakan `tmux` atau `screen` agar training tidak mati saat SSH putus:
  ```bash
  tmux new -s train
  # jalankan training di dalam tmux
  # Ctrl+B lalu D untuk detach
  # tmux attach -t train untuk kembali
  ```
