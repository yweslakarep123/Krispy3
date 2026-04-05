from typing import Dict
import copy
import numpy as np
import torch
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.common.replay_buffer import ReplayBuffer
from flow_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from flow_policy_3d.model.common.normalizer import LinearNormalizer
from flow_policy_3d.dataset.base_dataset import BaseDataset


class FrankaKitchenDataset(BaseDataset):
    """Zarr dataset: state (9-d qpos), action, point_cloud — optional velocity concat for agent_pos."""

    def __init__(
            self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            use_velocity=True,
            augment=False,
            pc_jitter_std=0.0,
            action_noise_std=0.0,
            pc_dropout_prob=0.0,
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_velocity = use_velocity
        self.augment = augment
        self.pc_jitter_std = pc_jitter_std
        self.action_noise_std = action_noise_std
        self.pc_dropout_prob = pc_dropout_prob

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        val_set.augment = False
        return val_set

    def _state_to_agent_pos(self, state: np.ndarray) -> np.ndarray:
        state = state.astype(np.float32)
        if not self.use_velocity:
            return state
        vel = np.zeros_like(state)
        vel[1:] = state[1:] - state[:-1]
        return np.concatenate([state, vel], axis=-1)

    def get_normalizer(self, mode='limits', **kwargs):
        buf = self.replay_buffer
        st = buf['state'][:].astype(np.float32)
        if self.use_velocity:
            vel = np.zeros_like(st)
            ends = np.asarray(buf.episode_ends[:], dtype=np.int64)
            start = 0
            for end in ends:
                seg = st[start:end]
                if len(seg) > 1:
                    vel[start:end][1:] = seg[1:] - seg[:-1]
                start = end
            agent = np.concatenate([st, vel], axis=-1)
        else:
            agent = st
        data = {
            'action': buf['action'],
            'agent_pos': agent,
            'point_cloud': buf['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _maybe_augment(self, point_cloud: np.ndarray, action: np.ndarray):
        if not self.augment:
            return point_cloud, action
        if self.pc_jitter_std > 0:
            point_cloud = point_cloud + self.pc_jitter_std * np.random.randn(
                *point_cloud.shape).astype(np.float32)
        if self.action_noise_std > 0:
            action = action + self.action_noise_std * np.random.randn(
                *action.shape).astype(np.float32)
        if self.pc_dropout_prob > 0 and point_cloud.shape[0] > 1:
            mask = np.random.rand(point_cloud.shape[0]) > self.pc_dropout_prob
            if mask.sum() < 4:
                return point_cloud, action
            pc = point_cloud[mask]
            if pc.shape[0] < point_cloud.shape[0]:
                pad = point_cloud.shape[0] - pc.shape[0]
                idx = np.random.choice(pc.shape[0], size=pad, replace=True)
                pc = np.concatenate([pc, pc[idx]], axis=0)
            point_cloud = pc.astype(np.float32)
        return point_cloud, action

    def _sample_to_data(self, sample):
        agent_pos = self._state_to_agent_pos(sample['state'][:,])
        point_cloud = sample['point_cloud'][:,].astype(np.float32)
        action = sample['action'].astype(np.float32)
        point_cloud, action = self._maybe_augment(point_cloud, action)
        return {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
            },
            'action': action,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
