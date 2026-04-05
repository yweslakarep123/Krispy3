# Franka Kitchen: data, arsitektur, dan pelatihan di vast.ai

Dokumen ini merangkum penyebab loss sangat kecil di awal epoch, strategi data & model, serta cara menjalankan FlowPolicy di GPU sewa (vast.ai).

## 1. Mengapa loss bisa turun sangat cepat ke ~1e-4?

**Kemungkinan utama (bukan saling mengecualikan):**

| Penyebab | Penjelasan |
|----------|------------|
| **Dataset kecil** | `kitchen-complete-v0` hanya puluhan ribu transisi; model besar (≈256M param) mudah **overfit** pada noise / pola repetitif. |
| **Normalisasi** | `LinearNormalizer` mode `limits` men-skala aksi dan observasi; error MSE dalam ruang ter-normalisasi bisa terlihat sangat kecil meski belum generalisasi. |
| **Flow / consistency loss** | Objektif FlowPolicy bukan BC MSE klasik; skala loss bergantung pada implementasi `compute_loss`. Angka kecil belum tentu berarti policy sudah optimal di sim. |
| **Kapasitas vs data** | Encoder PointNet-MLP + U-Net 1D sangat ekspresif; tanpa augmentasi, **memorization** cepat. |

**Yang disarankan:** gabungkan lebih banyak demonstrasi, aktifkan augmentasi (lihat `FrankaKitchenDataset`), pantau **rollout / `test_mean_score`** di WandB, bukan hanya `train_loss`.

## 2. Strategi data

1. **Gabung beberapa dataset D4RL Kitchen** (partial, mixed, complete) → beberapa zarr (satu per `gen_demonstration_franka_kitchen.py --env_name ...`), lalu **`scripts/merge_kitchen_zarr.py`** menggabungkan zarr menjadi satu file training.
2. **Augmentasi saat training** (noise titik awan, noise `agent_pos`, dropout titik) → variasi tanpa koleksi baru.
3. **(Opsional)** Generate ulang zarr dengan kamera / FPS sampling berbeda lalu merge.

## 3. Strategi arsitektur

File `flowpolicy_franka_kitchen.yaml` memakai:

- `condition_type: cross_attention_film` — kondisi spasial lebih kaya daripada FiLM murni untuk multi-obs-step.
- `encoder_output_dim` dan `down_dims` lebih besar — kapasitas lebih sebanding dengan tugas multi-subtask.

Anda bisa menyesuaikan lagi lewat override Hydra.

## 4. vast.ai — ringkasan alur

1. Buat instance dengan **CUDA** + image **PyTorch** atau **Ubuntu + NVIDIA driver**.
2. Clone repo, install **Miniconda**, buat env seperti di `setup_env.sh` (atau `environment.yml` jika tersedia).
3. Salin **`data/*.zarr`** (mis. dengan `rsync`, `scp`, atau volume).
4. Set variabel: `MUJOCO_GL=egl`, `LD_LIBRARY_PATH` ke `~/.mujoco/mujoco210/bin`.
5. `wandb login` jika pakai logging online.
6. Jalankan training dari folder `FlowPolicy/` dengan `--config-name=flowpolicy_franka_kitchen.yaml` dan override `task.dataset.zarr_path` ke path absolut di VM.

Detail perintah ada di bagian bawah file ini dan di `docs/VASTAI_FLOWPOLICY.md`.

## 5. Perintah cepat (lokal atau VM)

```bash
conda activate flowpolicy
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl

# (Opsional) merge beberapa sumber demonstrasi
python scripts/merge_kitchen_zarr.py \
  --inputs /path/to/kitchen_complete.zarr /path/to/kitchen_partial.zarr \
  --output_zarr /path/to/data/kitchen_merged.zarr

cd FlowPolicy
python train.py --config-name=flowpolicy_franka_kitchen.yaml \
  task.dataset.zarr_path=/path/to/data/kitchen_merged.zarr \
  hydra.run.dir=../data/outputs/my_run \
  training.device=cuda:0
```

Pastikan `zarr_path` adalah path **absolut** di mesin tempat training jalan (Hydra mengubah cwd ke output run).
