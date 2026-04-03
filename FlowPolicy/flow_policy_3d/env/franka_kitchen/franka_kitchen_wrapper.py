import os
import gym
import numpy as np
import mujoco_py

from termcolor import cprint
from gym import spaces
from flow_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator
from flow_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling


OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
}

OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0.0, 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}

TASK_ELEMENTS = ['microwave', 'kettle', 'light switch', 'slide cabinet']

COMPLETION_THRESHOLD = 0.3


def _get_kitchen_model_xml_path():
    import d4rl.kitchen
    kitchen_dir = os.path.dirname(d4rl.kitchen.__file__)
    return os.path.join(kitchen_dir, 'adept_envs', 'franka', 'assets',
                        'franka_kitchen_jntpos_act_ab.xml')


class FrankaKitchenEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, device="cuda",
                 use_point_crop=True,
                 num_points=512,
                 env_name='kitchen-complete-v0',
                 image_size=128,
                 ):
        super(FrankaKitchenEnv, self).__init__()

        import d4rl
        self.env = gym.make(env_name)
        self.env.reset()

        xml_path = _get_kitchen_model_xml_path()
        model = mujoco_py.load_model_from_path(xml_path)
        self._mjpy_sim = mujoco_py.MjSim(model)

        cam_names = list(self._mjpy_sim.model.camera_names)
        cprint(f"[FrankaKitchenEnv] available cameras: {cam_names}", "cyan")
        self._cam_name = cam_names[0] if cam_names else 'fixed'
        cprint(f"[FrankaKitchenEnv] using camera: {self._cam_name}", "cyan")

        self.device_id = 0
        self.image_size = image_size

        self._mjpy_sim.model.vis.map.znear = 0.1
        self._mjpy_sim.model.vis.map.zfar = 5.0

        self.pc_generator = PointCloudGenerator(
            sim=self._mjpy_sim, cam_names=[self._cam_name],
            img_size=self.image_size)
        self.use_point_crop = use_point_crop
        cprint(f"[FrankaKitchenEnv] use_point_crop: {self.use_point_crop}", "cyan")
        self.num_points = num_points

        self.pc_transform = None
        self.pc_scale = np.array([1, 1, 1])
        self.pc_offset = np.array([0, 0, 0])
        self.min_bound = [-2.0, -2.0, -0.5]
        self.max_bound = [2.0, 2.0, 3.0]

        self.episode_length = self._max_episode_steps = 280
        self.action_space = self.env.action_space
        self.obs_sensor_dim = 9

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32
            ),
            'depth': spaces.Box(
                low=0, high=255,
                shape=(self.image_size, self.image_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'full_state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(60,),
                dtype=np.float32
            ),
        })

        self.tasks = TASK_ELEMENTS
        self.completed_tasks = set()

    def _sync_mjpy_sim(self):
        """Copy state from D4RL dm_control env to mujoco_py sim for rendering."""
        dm_sim = self.env.sim
        qpos = np.array(dm_sim.data.qpos[:]).copy()
        qvel = np.array(dm_sim.data.qvel[:]).copy()

        state = self._mjpy_sim.get_state()
        new_qpos = state.qpos.copy()
        new_qvel = state.qvel.copy()
        nq = min(len(qpos), len(new_qpos))
        nv = min(len(qvel), len(new_qvel))
        new_qpos[:nq] = qpos[:nq]
        new_qvel[:nv] = qvel[:nv]
        new_state = state._replace(qpos=new_qpos, qvel=new_qvel)
        self._mjpy_sim.set_state(new_state)
        self._mjpy_sim.forward()

    def get_robot_state(self):
        qpos = np.array(self.env.sim.data.qpos[:9]).copy()
        return qpos.astype(np.float32)

    def get_rgb(self):
        self._sync_mjpy_sim()
        img = self._mjpy_sim.render(
            width=self.image_size, height=self.image_size,
            camera_name=self._cam_name, device_id=self.device_id)
        return img

    def get_point_cloud(self, use_rgb=False):
        self._sync_mjpy_sim()
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(
            device_id=self.device_id)

        if not use_rgb:
            point_cloud = point_cloud[..., :3]

        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        if self.pc_offset is not None:
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset

        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps')
        depth = depth[::-1]

        return point_cloud, depth

    def _check_task_completion(self):
        qpos = np.array(self.env.sim.data.qpos[:]).copy()
        newly_completed = []
        for task in self.tasks:
            if task in self.completed_tasks:
                continue
            indices = OBS_ELEMENT_INDICES[task]
            goal = OBS_ELEMENT_GOALS[task]
            current = qpos[indices]
            dist = np.linalg.norm(current - goal)
            if dist < COMPLETION_THRESHOLD:
                self.completed_tasks.add(task)
                newly_completed.append(task)
        return newly_completed

    def step(self, action: np.array):
        raw_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1

        newly_completed = self._check_task_completion()

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_state,
        }

        env_info['completed_tasks'] = list(self.completed_tasks)
        env_info['num_completed'] = len(self.completed_tasks)
        env_info['newly_completed'] = newly_completed
        env_info['all_completed'] = len(self.completed_tasks) == len(self.tasks)
        env_info['success'] = float(env_info['all_completed'])

        done = done or self.cur_step >= self.episode_length

        return obs_dict, reward, done, env_info

    def reset(self):
        raw_obs = self.env.reset()
        self.cur_step = 0
        self.completed_tasks = set()

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_obs,
        }

        return obs_dict

    def seed(self, seed=None):
        if seed is not None:
            self.env.seed(seed)

    def render(self, mode='rgb_array'):
        img = self.get_rgb()
        return img

    def close(self):
        self.env.close()
