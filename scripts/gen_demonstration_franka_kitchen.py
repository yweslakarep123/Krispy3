"""
Generate demonstration data for Franka Kitchen from D4RL dataset.

Loads the D4RL kitchen-complete-v0 dataset, re-renders point clouds from
MuJoCo via mujoco_py, and saves everything in zarr format for FlowPolicy.

Usage:
    python scripts/gen_demonstration_franka_kitchen.py \
        --env_name kitchen-complete-v0 \
        --root_dir data/ \
        --num_points 512 \
        --image_size 128
"""

import os
import sys
import argparse
import numpy as np
import zarr
import gym
import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FlowPolicy'))

from flow_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator
from flow_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling


def get_kitchen_model_xml_path():
    import d4rl.kitchen
    kitchen_dir = os.path.dirname(d4rl.kitchen.__file__)
    return os.path.join(kitchen_dir, 'adept_envs', 'franka', 'assets',
                        'franka_kitchen_jntpos_act_ab.xml')


def create_mujoco_py_sim(xml_path):
    import mujoco_py
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)
    return sim


def set_mjpy_sim_state(mjpy_sim, qpos, qvel):
    state = mjpy_sim.get_state()
    new_qpos = state.qpos.copy()
    new_qvel = state.qvel.copy()
    qpos_len = min(len(qpos), len(new_qpos))
    qvel_len = min(len(qvel), len(new_qvel))
    new_qpos[:qpos_len] = qpos[:qpos_len]
    new_qvel[:qvel_len] = qvel[:qvel_len]
    new_state = state._replace(qpos=new_qpos, qvel=new_qvel)
    mjpy_sim.set_state(new_state)
    mjpy_sim.forward()


def segment_episodes(dataset):
    """Segment flat D4RL dataset into episodes using timeouts/terminals."""
    N = dataset['observations'].shape[0]
    timeouts = dataset.get('timeouts', np.zeros(N, dtype=bool))
    terminals = dataset.get('terminals', np.zeros(N, dtype=bool))

    episode_ends = []
    for i in range(N):
        if timeouts[i] or terminals[i] or i == N - 1:
            episode_ends.append(i + 1)
    return episode_ends


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='kitchen-complete-v0')
    parser.add_argument('--root_dir', type=str, default='data/')
    parser.add_argument('--num_points', type=int, default=512)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--use_point_crop', action='store_true', default=True)
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Path zarr lengkap (default: root_dir/franka_kitchen_expert.zarr)')
    args = parser.parse_args()

    os.makedirs(args.root_dir, exist_ok=True)

    import d4rl
    env = gym.make(args.env_name)
    dataset = env.get_dataset()

    xml_path = get_kitchen_model_xml_path()
    print(f"Kitchen model XML: {xml_path}")

    mjpy_sim = create_mujoco_py_sim(xml_path)
    cam_names = list(mjpy_sim.model.camera_names)
    print(f"Available cameras: {cam_names}")
    cam_name = cam_names[0] if cam_names else 'fixed'
    print(f"Using camera: {cam_name}")

    mjpy_sim.model.vis.map.znear = 0.1
    mjpy_sim.model.vis.map.zfar = 5.0

    pc_generator = PointCloudGenerator(
        sim=mjpy_sim, cam_names=[cam_name], img_size=args.image_size)

    min_bound = np.array([-2.0, -2.0, -0.5])
    max_bound = np.array([2.0, 2.0, 3.0])

    observations = dataset['observations']
    actions = dataset['actions']
    N = observations.shape[0]
    obs_dim = observations.shape[1]

    nq = mjpy_sim.model.nq  # 30
    nv = mjpy_sim.model.nv  # 29
    print(f"obs_dim={obs_dim}, nq={nq}, nv={nv}")

    episode_ends = segment_episodes(dataset)
    num_episodes = len(episode_ends)
    print(f"Found {num_episodes} episodes, {N} total timesteps")

    all_states = []
    all_actions = []
    all_point_clouds = []
    zarr_episode_ends = []

    current_idx = 0
    for ep_i in tqdm.tqdm(range(num_episodes), desc="Processing episodes"):
        ep_start = 0 if ep_i == 0 else episode_ends[ep_i - 1]
        ep_end = episode_ends[ep_i]
        ep_len = ep_end - ep_start

        for t in range(ep_start, ep_end):
            obs = observations[t]
            qpos = obs[:nq]
            qvel = obs[nq:nq + nv]

            set_mjpy_sim_state(mjpy_sim, qpos, qvel)

            pc_raw, depth = pc_generator.generateCroppedPointCloud(device_id=0)
            point_cloud = pc_raw[..., :3]

            if args.use_point_crop:
                mask = np.all(point_cloud[:, :3] > min_bound, axis=1)
                point_cloud = point_cloud[mask]
                mask = np.all(point_cloud[:, :3] < max_bound, axis=1)
                point_cloud = point_cloud[mask]

            point_cloud = point_cloud_sampling(
                point_cloud, args.num_points, 'fps')

            robot_state = qpos[:9].astype(np.float32)
            action = actions[t].astype(np.float32)

            all_states.append(robot_state)
            all_actions.append(action)
            all_point_clouds.append(point_cloud.astype(np.float32))

        current_idx += ep_len
        zarr_episode_ends.append(current_idx)

    all_states = np.array(all_states)
    all_actions = np.array(all_actions)
    all_point_clouds = np.array(all_point_clouds)
    zarr_episode_ends = np.array(zarr_episode_ends, dtype=np.int64)

    print(f"States shape: {all_states.shape}")
    print(f"Actions shape: {all_actions.shape}")
    print(f"Point clouds shape: {all_point_clouds.shape}")
    print(f"Episode ends: {zarr_episode_ends}")

    zarr_path = args.output_path or os.path.join(
        args.root_dir, 'franka_kitchen_expert.zarr')
    print(f"Saving to {zarr_path}")

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store, overwrite=True)

    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    data_group.create_dataset('state', data=all_states,
                              chunks=(min(1000, len(all_states)),) + all_states.shape[1:],
                              dtype=np.float32)
    data_group.create_dataset('action', data=all_actions,
                              chunks=(min(1000, len(all_actions)),) + all_actions.shape[1:],
                              dtype=np.float32)
    data_group.create_dataset('point_cloud', data=all_point_clouds,
                              chunks=(min(100, len(all_point_clouds)),) + all_point_clouds.shape[1:],
                              dtype=np.float32)
    meta_group.create_dataset('episode_ends', data=zarr_episode_ends,
                              dtype=np.int64)

    print(f"Done! Saved {num_episodes} episodes to {zarr_path}")
    print(f"  state: {all_states.shape}")
    print(f"  action: {all_actions.shape}")
    print(f"  point_cloud: {all_point_clouds.shape}")


if __name__ == '__main__':
    main()
