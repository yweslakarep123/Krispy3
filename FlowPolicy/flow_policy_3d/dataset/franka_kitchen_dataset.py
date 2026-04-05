from typing import Dict
import torch
import numpy as np
import copy
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.common.replay_buffer import ReplayBuffer
from flow_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from flow_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from flow_policy_3d.dataset.base_dataset import BaseDataset


class FrankaKitchenDataset(BaseDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 augment=False,
                 pc_noise_std=0.0,
                 pc_point_dropout_ratio=0.0,
                 agent_pos_noise_std=0.0,
                 ):
        super().__init__()
        self.augment = augment
        self.pc_noise_std = float(pc_noise_std)
        self.pc_point_dropout_ratio = float(pc_point_dropout_ratio)
        self.agent_pos_noise_std = float(agent_pos_noise_std)
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

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][..., :],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:, ].astype(np.float32)
        point_cloud = sample['point_cloud'][:, ].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
            },
            'action': sample['action'].astype(np.float32)
        }
        return data

    def _augment_obs(self, data: Dict) -> Dict:
        if not self.augment:
            return data
        obs = data['obs']
        pc = obs['point_cloud'].copy()
        ap = obs['agent_pos'].copy()
        if self.pc_noise_std > 0:
            pc = pc + np.random.randn(*pc.shape).astype(np.float32) * self.pc_noise_std
        if self.pc_point_dropout_ratio > 0 and pc.ndim == 3:
            h, n, _ = pc.shape
            k = max(1, int(n * self.pc_point_dropout_ratio))
            for t in range(h):
                repl = np.random.choice(n, size=k, replace=False)
                src = np.random.randint(0, n, size=k)
                jitter = np.random.randn(k, 3).astype(np.float32) * max(
                    self.pc_noise_std, 1e-4)
                pc[t, repl] = pc[t, src] + jitter
        if self.agent_pos_noise_std > 0:
            ap = ap + np.random.randn(*ap.shape).astype(np.float32) * self.agent_pos_noise_std
        data = dict(data)
        data['obs'] = {'point_cloud': pc, 'agent_pos': ap}
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        data = self._augment_obs(data)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
