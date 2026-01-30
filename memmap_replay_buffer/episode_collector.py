from __future__ import annotations
from typing import Any, Sequence
from collections import defaultdict

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor, cat, is_tensor

from beartype import beartype

from memmap_replay_buffer.replay_buffer import ReplayBuffer
from memmap_replay_buffer.replay_buffer_h5py import ReplayBufferH5PY

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_group_batched(value, num_groups):
    if is_tensor(value) or isinstance(value, ndarray):
        return value.shape[0] == num_groups
    
    if isinstance(value, (list, tuple)):
        return len(value) == num_groups
    
    return False

def combine_trajectory(values):
    first = values[0]
    if is_tensor(first):
        return cat(values, dim = 0)
    
    if isinstance(first, ndarray):
        return np.concatenate(values, axis = 0)
    
    return np.array(values)

class EpisodeCollector:
    @beartype
    def __init__(
        self,
        replay_buffer: ReplayBuffer | ReplayBufferH5PY,
        num_groups: int,
        fieldnames: tuple[str, ...] | None = None,
        meta_fieldnames: tuple[str, ...] | None = None
    ):
        self.replay_buffer = replay_buffer
        self.num_groups = num_groups

        self.fieldnames = default(fieldnames, tuple(replay_buffer.fieldnames))
        self.meta_fieldnames = default(meta_fieldnames, tuple(replay_buffer.meta_fieldnames))

        self.reset()

    def reset(self):
        self.episode_data = [defaultdict(list) for _ in range(self.num_groups)]

    @beartype
    def append(
        self,
        **data
    ):
        for name, value in data.items():
            if name not in self.fieldnames and name not in self.meta_fieldnames:
                continue

            if is_group_batched(value, self.num_groups):
                for i in range(self.num_groups):
                    self.episode_data[i][name].append(value[i])
            else:
                for i in range(self.num_groups):
                    self.episode_data[i][name].append(value)

    @beartype
    def store(self, **meta_data):
        for i in range(self.num_groups):
            group_data = self.episode_data[i]
            
            store_kwargs = dict()

            # Handle collected data
            for name, values in group_data.items():
                if name in self.fieldnames:
                    combined = combine_trajectory(values)
                else:
                    # Meta fields: take the last appended value
                    combined = values[-1]

                store_kwargs[name] = combined

            # Merge with extra meta data provided to store()
            for name, value in meta_data.items():
                if name not in self.meta_fieldnames:
                    continue

                if is_group_batched(value, self.num_groups):
                    store_kwargs[name] = value[i]
                else:
                    store_kwargs[name] = value

            self.replay_buffer.store_episode(**store_kwargs)

        self.reset()
