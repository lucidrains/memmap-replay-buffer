from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, cat
from torch.utils.data import ConcatDataset, Dataset

from memmap_replay_buffer.replay_buffer import ReplayBuffer, default, exists, pad_right_at_dim

class ConcatReplayBuffer:
    def __init__(self, folders: list[str | Path]):
        if len(folders) == 0:
            raise ValueError('ConcatReplayBuffer requires at least one folder')

        self.buffers = [ReplayBuffer.from_folder(folder, read_only = True) for folder in folders]

        first_buffer = self.buffers[0]
        self.fieldnames = first_buffer.fieldnames
        self.meta_fieldnames = first_buffer.meta_fieldnames

        for buffer in self.buffers[1:]:
            if buffer.fieldnames != self.fieldnames:
                raise ValueError(f'All buffers must have the same fieldnames - got {buffer.fieldnames} and {self.fieldnames}')

            if buffer.meta_fieldnames != self.meta_fieldnames:
                raise ValueError(f'All buffers must have the same meta_fieldnames - got {buffer.meta_fieldnames} and {self.meta_fieldnames}')

        self.read_only = True

    @property
    def num_episodes(self):
        return sum(buffer.num_episodes for buffer in self.buffers)

    @property
    def max_episodes(self):
        return sum(buffer.max_episodes for buffer in self.buffers)

    @property
    def max_timesteps(self):
        return max(buffer.max_timesteps for buffer in self.buffers)

    def __len__(self):
        return sum(len(buffer) for buffer in self.buffers)

    def dataset(self, **kwargs) -> Dataset:
        if len(self) == 0:
            raise ValueError('replay buffer is empty')

        datasets = [buffer.dataset(**kwargs) for buffer in self.buffers if len(buffer) > 0]
        return ConcatDataset(datasets)

    def dataloader(self, *args, **kwargs):
        return ReplayBuffer.dataloader(self, *args, **kwargs)

    def get_all_data(
        self,
        fields: tuple[str, ...] | None = None,
        meta_fields: tuple[str, ...] | None = None,
        device: torch.device | str | None = None
    ) -> dict[str, Tensor]:

        all_data = [buffer.get_all_data(fields = fields, meta_fields = meta_fields, device = device) for buffer in self.buffers]
        all_data = [data for data in all_data if len(data) > 0]

        if not all_data:
            return dict()

        data_fields = self.fieldnames if not exists(fields) and not exists(meta_fields) else default(fields, ())

        out = dict()

        for key in all_data[0].keys():
            tensors = [data[key] for data in all_data if key in data]

            if key in data_fields and tensors[0].ndim > 1:
                # pad trajectories to the longest across buffers
                max_time = max(t.shape[1] for t in tensors)
                tensors = [pad_right_at_dim(t, max_time - t.shape[1], dim = 1) for t in tensors]

            out[key] = cat(tensors, dim = 0)

        return out

    def flush(self):
        pass

    def _read_only(self, *args, **kwargs):
        raise NotImplementedError('ConcatReplayBuffer is read-only')

    clear = reset_ = advance_episode = store_batch = store_meta_batch = \
    one_episode = batched_episode = store_datapoint = store_meta_datapoint = \
    store_batch_datapoint = store_batch_meta_datapoint = store = store_episode = update = _read_only
