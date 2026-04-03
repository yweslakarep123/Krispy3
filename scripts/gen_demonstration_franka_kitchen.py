"""
Generate demonstration data for Franka Kitchen from D4RL datasets.

Combines multiple D4RL kitchen datasets (complete, partial, mixed),
re-renders point clouds from MuJoCo via mujoco_py, and saves
everything in a single zarr file for FlowPolicy training.

Usage:
    python scripts/gen_demonstration_franka_kitchen.py \
        --env_names kitchen-complete-v0 kitchen-partial-v0 kitchen-mixed-v0 \
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
    N = dataset['observations'].shape[0]
    timeouts = dataset.get('timeouts', np.zeros(N, dtype=bool))
    terminals = dataset.get('terminals', np.zeros(N, dtype=bool))

    episode_ends = []
    for i in range(N):
        if timeouts[i] or terminals[i] or i == N - 1:
            episode_ends.append(i + 1)
    return episode_ends


def process_dataset(env_name, mjpy_sim, pc_generator, nq, nv,
                    min_bound, max_bound, num_points, use_point_crop):
    """Load one D4RL dataset and render point clouds for all timesteps."""
    import d4rl
    print(f"\n{'='*60}")
    print(f"Processing: {env_name}")
    print(f"{'='*60}")

    env = gym.make(env_name)
    dataset = env.get_dataset()

    observations = dataset['observations']
    actions = dataset['actions']
    N = observations.shape[0]

    episode_ends = segment_episodes(dataset)
    num_episodes = len(episode_ends)
    print(f"  Episodes: {num_episodes}, Timesteps: {N}")

    states = []
    acts = []
    point_clouds = []
    ep_lengths = []

    for ep_i in tqdm.tqdm(range(num_episodes),
                          desc=f"  {env_name}", leave=True):
        ep_start = 0 if ep_i == 0 else episode_ends[ep_i - 1]
        ep_end = episode_ends[ep_i]

        for t in range(ep_start, ep_end):
            obs = observations[t]
            qpos = obs[:nq]
            qvel = obs[nq:nq + nv]

            set_mjpy_sim_state(mjpy_sim, qpos, qvel)

            pc_raw, _ = pc_generator.generateCroppedPointCloud(device_id=0)
            pc = pc_raw[..., :3]

            if use_point_crop:
                mask = np.all(pc[:, :3] > min_bound, axis=1)
                pc = pc[mask]
                mask = np.all(pc[:, :3] < max_bound, axis=1)
                pc = pc[mask]

            pc = point_cloud_sampling(pc, num_points, 'fps')

            states.append(qpos[:9].astype(np.float32))
            acts.append(actions[t].astype(np.float32))
            point_clouds.append(pc.astype(np.float32))

        ep_lengths.append(ep_end - ep_start)

    env.close()
    return states, acts, point_clouds, ep_lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_names', type=str, nargs='+',
                        default=['kitchen-complete-v0',
                                 'kitchen-partial-v0',
                                 'kitchen-mixed-v0'])
    parser.add_argument('--root_dir', type=str, default='data/')
    parser.add_argument('--num_points', type=int, default=512)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--use_point_crop', action='store_true', default=True)
    parser.add_argument('--output_name', type=str,
                        default='franka_kitchen_expert.zarr')
    args = parser.parse_args()

    os.makedirs(args.root_dir, exist_ok=True)

    import d4rl

    xml_path = get_kitchen_model_xml_path()
    print(f"Kitchen model XML: {xml_path}")

    mjpy_sim = create_mujoco_py_sim(xml_path)
    cam_names = list(mjpy_sim.model.camera_names)
    print(f"Available cameras: {cam_names}")
    cam_name = cam_names[0] if cam_names else 'fixed'
    print(f"Using camera: {cam_name}")

    mjpy_sim.model.vis.map.znear = 0.1
    mjpy_sim.model.vis.map.zfar = 5.0

    nq = mjpy_sim.model.nq
    nv = mjpy_sim.model.nv
    print(f"nq={nq}, nv={nv}")

    pc_generator = PointCloudGenerator(
        sim=mjpy_sim, cam_names=[cam_name], img_size=args.image_size)

    min_bound = np.array([-2.0, -2.0, -0.5])
    max_bound = np.array([2.0, 2.0, 3.0])

    all_states = []
    all_actions = []
    all_point_clouds = []
    zarr_episode_ends = []
    current_idx = 0
    total_episodes = 0

    for env_name in args.env_names:
        states, acts, pcs, ep_lengths = process_dataset(
            env_name, mjpy_sim, pc_generator, nq, nv,
            min_bound, max_bound, args.num_points, args.use_point_crop)

        all_states.extend(states)
        all_actions.extend(acts)
        all_point_clouds.extend(pcs)

        for ep_len in ep_lengths:
            current_idx += ep_len
            zarr_episode_ends.append(current_idx)
            total_episodes += 1

    all_states = np.array(all_states)
    all_actions = np.array(all_actions)
    all_point_clouds = np.array(all_point_clouds)
    zarr_episode_ends = np.array(zarr_episode_ends, dtype=np.int64)

    print(f"\n{'='*60}")
    print(f"Combined dataset summary:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  States:       {all_states.shape}")
    print(f"  Actions:      {all_actions.shape}")
    print(f"  Point clouds: {all_point_clouds.shape}")
    print(f"{'='*60}")

    zarr_path = os.path.join(args.root_dir, args.output_name)
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

    print(f"Done! Saved {total_episodes} episodes to {zarr_path}")


if __name__ == '__main__':
    main()
