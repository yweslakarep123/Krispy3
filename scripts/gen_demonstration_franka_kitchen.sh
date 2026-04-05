# Generate Franka Kitchen demonstrations from D4RL dataset
# Usage: bash scripts/gen_demonstration_franka_kitchen.sh
#
# This script loads the D4RL kitchen-complete-v0 dataset,
# re-renders point clouds from MuJoCo states, and saves
# to zarr format for FlowPolicy training.

ROOT_DIR="data/"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

python scripts/gen_demonstration_franka_kitchen.py \
    --env_name kitchen-complete-v0 \
    --root_dir ${ROOT_DIR} \
    --num_points 512 \
    --image_size 128 \
    --use_point_crop
