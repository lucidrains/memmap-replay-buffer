from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, cat
from torch.utils.data import ConcatDataset, Dataset

from memmap_replay_buffer.replay_buffer import ReplayBuffer, default, exists, pad_at_dim


class ConcatReplayBuffer:
    def __init__(self, folders: list[str | Path]):
        self.buffers = [ReplayBuffer.from_folder(f, read_only=True) for f in folders]

        if len(self.buffers) == 0:
            raise ValueError("ConcatReplayBuffer requires at least one folder")

        # Verify compatibility
        first_buf = self.buffers[0]
        self.fieldnames = first_buf.fieldnames
        self.meta_fieldnames = first_buf.meta_fieldnames

        for b in self.buffers[1:]:
            if b.fieldnames != self.fieldnames:
                raise ValueError(f"All buffers must have the same fieldnames - got {b.fieldnames} and {self.fieldnames}")
            if b.meta_fieldnames != self.meta_fieldnames:
                raise ValueError(f"All buffers must have the same meta_fieldnames - got {b.meta_fieldnames} and {self.meta_fieldnames}")

        self.read_only = True

    @property
    def num_episodes(self):
        return sum(b.num_episodes for b in self.buffers)

    @property
    def max_episodes(self):
        return sum(b.max_episodes for b in self.buffers)

    @property
    def max_timesteps(self):
        return max(b.max_timesteps for b in self.buffers)

    def __len__(self):
        return sum(len(b) for b in self.buffers)

    def dataset(self, **kwargs) -> Dataset:
        if len(self) == 0:
            raise ValueError('replay buffer is empty')
        datasets = [b.dataset(**kwargs) for b in self.buffers if len(b) > 0]
        return ConcatDataset(datasets)

    dataloader = ReplayBuffer.dataloader

    def get_all_data(
        self,
        fields: tuple[str, ...] | None = None,
        meta_fields: tuple[str, ...] | None = None,
        device: torch.device | str | None = None
    ) -> dict[str, Tensor]:

        all_data = [b.get_all_data(fields=fields, meta_fields=meta_fields, device=device) for b in self.buffers]
        all_data = [d for d in all_data if len(d) > 0]

        if not all_data:
            return dict()

        keys = all_data[0].keys()
        data_fields = self.fieldnames if not exists(fields) and not exists(meta_fields) else default(fields, ())

        out = dict()

        for key in keys:
            tensors = [d[key] for d in all_data if key in d]
            if not tensors:
                continue

            if key in data_fields and tensors[0].ndim > 1:
                max_time = max(t.shape[1] for t in tensors)
                tensors = [pad_at_dim(t, (0, max_time - t.shape[1]), dim=1) if t.shape[1] < max_time else t for t in tensors]

            out[key] = cat(tensors, dim=0)

        return out

    def flush(self):
        pass

    def _read_only(self, *args, **kwargs):
        raise NotImplementedError("ConcatReplayBuffer is read-only")

    clear = reset_ = advance_episode = store_batch = store_meta_batch = \
    one_episode = batched_episode = store_datapoint = store_meta_datapoint = \
    store_batch_datapoint = store_batch_meta_datapoint = store = store_episode = update = _read_only
