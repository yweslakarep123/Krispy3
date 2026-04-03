# Generate Franka Kitchen demonstrations from all D4RL kitchen datasets.
# Combines complete + partial + mixed for maximum data diversity.
#
# Usage: bash scripts/gen_demonstration_franka_kitchen.sh

ROOT_DIR="data/"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1

python scripts/gen_demonstration_franka_kitchen.py \
    --env_names kitchen-complete-v0 kitchen-partial-v0 kitchen-mixed-v0 \
    --root_dir ${ROOT_DIR} \
    --num_points 512 \
    --image_size 128 \
    --use_point_crop \
    --output_name franka_kitchen_expert.zarr
