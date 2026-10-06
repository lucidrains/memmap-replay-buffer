from __future__ import annotations

from collections import defaultdict, namedtuple
from typing import Any

import numpy as np
from beartype import beartype
from numpy import ndarray
import torch
from torch import is_tensor
from torch.utils.data import DataLoader, Dataset, default_collate

from memmap_replay_buffer.replay_buffer import tree_map_to_device

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_group_batched(value, num_groups):
    if is_tensor(value) or isinstance(value, ndarray):
        return value.ndim > 0 and value.shape[0] == num_groups

    if isinstance(value, (list, tuple)):
        return len(value) == num_groups

    return False

def stack(values, dim = 0):
    if is_tensor(values[0]):
        return torch.stack(values, dim = dim)

    return np.stack(values, axis = dim)

# dataset

class TensorDictDataset(Dataset):
    def __init__(self, data: dict[str, torch.Tensor]):
        self.data = data
        self.num_samples = len(next(iter(data.values())))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {name: value[index] for name, value in self.data.items()}

# class

class EpisodeCollector:
    """ accumulates a group-batched rollout step by step, then either stores it or trains on it in memory

    layouts are explicit, so the time and group axes can never be silently swapped

        `append` - one step, group-batched (num_groups, *shape) or (num_groups, 1, *shape)
        `extend` - whole trajectories, time-major (time, num_groups, *shape), e.g. returns / advantages
        `env_major` / `time_major` - read back (num_groups, time, ...) / (time, num_groups, ...)
        `all_data` / `dataloader` - env-major tensors / flattened minibatches, entirely in memory
        `store` - write one episode per group, then `clear`
        `clear` - discard the current rollout
    """

    @beartype
    def __init__(
        self,
        replay_buffer: Any,        # ReplayBuffer | ReplayBufferH5PY, without a hard dependency on h5py
        num_groups: int,
        fieldnames: tuple[str, ...] | None = None,
        meta_fieldnames: tuple[str, ...] | None = None,
        strict: bool = True
    ):
        self.replay_buffer = replay_buffer
        self.num_groups = num_groups

        self.fieldnames = set(default(fieldnames, tuple(replay_buffer.fieldnames)))
        self.meta_fieldnames = set(default(meta_fieldnames, tuple(replay_buffer.meta_fieldnames)))

        self.shapes = replay_buffer.shapes
        self.meta_shapes = replay_buffer.meta_shapes
        self.strict = strict

        self.clear()

    def __len__(self):
        return self.step_count

    def clear(self):
        self.episode_data = [defaultdict(list) for _ in range(self.num_groups)]
        self.step_count = 0

    # fields

    def field_kind(self, name):
        if name in self.fieldnames:
            return 'data'

        if name in self.meta_fieldnames:
            return 'meta'

        if self.strict:
            raise KeyError(f'unknown field {name} - valid fields are {sorted(self.fieldnames | self.meta_fieldnames)}')

        return None

    def field_shape(self, name):
        return self.meta_shapes[name] if name in self.meta_fieldnames else self.shapes[name]

    def per_group(self, name, value):
        # (num_groups, *shape) / (num_groups, 1, *shape) -> num_groups values of shape (*shape)

        if isinstance(value, (list, tuple)):
            return list(value)

        shape = tuple(self.field_shape(name))
        rest = tuple(value.shape[1:])

        assert rest in (shape, (1, *shape)), f'field {name} - per-group shape {rest} should be {shape} or (1, *shape)'

        if rest != shape:
            value = value[:, 0]

        return list(value)

    # adding data

    @beartype
    def append(self, **data):
        # one step, group-batched values

        for name, value in data.items():
            kind = self.field_kind(name)

            if not exists(kind):
                continue

            if kind == 'meta':
                values = [value] * self.num_groups
            else:
                shape = tuple(self.field_shape(name))

                assert is_group_batched(value, self.num_groups), (
                    f'field {name} - `append` expects per-step group-batched values of shape ({self.num_groups}, *{shape}) '
                    f'or ({self.num_groups}, 1, *{shape}) - for whole trajectories in time-major (time, {self.num_groups}, *{shape}), use `extend`'
                )

                values = self.per_group(name, value)

            for group_index, group_value in enumerate(values):
                self.episode_data[group_index][name].append(group_value)

        self.step_count += 1

    @beartype
    def extend(self, **trajectories):
        # whole trajectories, time-major (time, num_groups, *shape)

        for name, value in trajectories.items():
            kind = self.field_kind(name)

            if not exists(kind):
                continue

            assert is_tensor(value) or isinstance(value, ndarray), f'field {name} - `extend` expects a tensor / ndarray of shape (time, {self.num_groups}, *shape)'

            time = value.shape[0]

            assert value.ndim > 1 and value.shape[1] == self.num_groups, (
                f'field {name} - `extend` expects time-major (time, {self.num_groups}, *shape), got {tuple(value.shape)}'
            )

            if len(self) == 0:
                self.step_count = time
            else:
                assert time == len(self), f'field {name} - time dim {time} does not match the {len(self)} collected steps'

            shape = tuple(self.field_shape(name))
            assert tuple(value.shape[2:]) == shape, f'field {name} - feature shape {tuple(value.shape[2:])} does not match {shape}'

            for group_index in range(self.num_groups):
                self.episode_data[group_index][name].extend(list(value[:, group_index]))

    # reading back

    def groups(self, name):
        # one value per group, i.e. a list of num_groups tensors / arrays

        if self.field_kind(name) == 'meta':
            return [data[name][-1] for data in self.episode_data]

        return [stack(data[name], dim = 0) for data in self.episode_data]

    def time_major(self, name):
        # (time, num_groups, *shape) for data fields, (num_groups,) for meta fields

        return stack(self.groups(name), dim = 1 if self.field_kind(name) == 'data' else 0)

    def env_major(self, name):
        # (num_groups, time, *shape), the layout generalized advantage estimation expects

        values = self.time_major(name)

        if self.field_kind(name) == 'meta':
            return values

        return values.swapaxes(0, 1)

    # in-memory training - no storing required

    def all_data(self, fields = None, device: torch.device | str | None = None):
        # dict of env-major tensors, (num_groups, time, *shape), for the collected data fields

        if len(self) == 0:
            raise ValueError('collector is empty')

        collected = tuple(name for name in self.episode_data[0] if name in self.fieldnames)

        data = {name: self.env_major(name) for name in default(fields, collected)}
        return tree_map_to_device(data, device)

    def dataloader(
        self,
        batch_size,
        fields = None,
        filter_fields = None,
        to_named_tuple = None,
        shuffle = False,
        device: torch.device | str | None = None,
        **kwargs
    ):
        # minibatches flattened over (num_groups, time) for the current rollout, entirely in memory

        data = {name: torch.as_tensor(value).reshape(-1, *value.shape[2:]) for name, value in self.all_data().items()}

        if exists(filter_fields) and len(filter_fields) > 0:
            mask = None
            for name, filter_value in filter_fields.items():
                field_mask = data[name] == filter_value
                mask = field_mask if not exists(mask) else (mask & field_mask)

            data = {name: value[mask] for name, value in data.items()}

        fields = default(fields, tuple(data.keys()))
        data = {name: data[name] for name in fields}

        dataset = TensorDictDataset(data)

        NamedTupleCls = None
        if exists(to_named_tuple):
            sanitized_fields = tuple(f.lstrip('_') if f.startswith('_') else f for f in to_named_tuple)
            NamedTupleCls = namedtuple('Batch', sanitized_fields)

        def collate_fn(data):
            batch = default_collate(data)

            if exists(NamedTupleCls):
                for field in to_named_tuple:
                    if field not in batch:
                        raise ValueError(f'field `{field}` not found in batch. available fields: {list(batch.keys())}')

                batch = NamedTupleCls(**{san: batch[orig] for orig, san in zip(to_named_tuple, sanitized_fields)})

            return tree_map_to_device(batch, device)

        return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn, **kwargs)

    # storing

    @beartype
    def store(self, **meta_data):
        for group_index, group_data in enumerate(self.episode_data):
            store_kwargs = {name: stack(values, dim = 0) if name in self.fieldnames else values[-1] for name, values in group_data.items()}

            for name, value in meta_data.items():
                if name not in self.meta_fieldnames:
                    continue

                store_kwargs[name] = value[group_index] if is_group_batched(value, self.num_groups) else value

            self.replay_buffer.store_episode(**store_kwargs)

        self.clear()
