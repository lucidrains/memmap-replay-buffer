from __future__ import annotations
from typing import Callable
from beartype import beartype
from beartype.door import is_bearable

from pathlib import Path
from shutil import rmtree
from contextlib import contextmanager
from collections import namedtuple

import numpy as np
from numpy import ndarray
from numpy.lib.format import open_memmap

import torch
from torch import tensor, from_numpy, stack, cat, is_tensor, Tensor, arange
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# constants

PrimitiveType = int | float | bool

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

def xnor(x, y):
    return not (x ^ y)

def is_empty(t):
    return t.numel() == 0

def pad_at_dim(
    t,
    pad: tuple[int, int],
    dim = -1,
    value = 0.
):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# data

def collate_var_time(data):

    datum = first(data)
    keys = datum.keys()

    all_tensors = zip(*[datum.values() for datum in data])

    collated_values = []

    for key, tensors in zip(keys, all_tensors):

        # the episode lens have zero dimension - think of a cleaner way to handle this later

        if key != '_lens':

            times = [t.shape[0] for t in tensors]
            max_time = max(times)
            tensors = [pad_at_dim(t, (0, max_time - t.shape[0]), dim = 0) for t in tensors]

        collated_values.append(stack(tensors))

    return dict(zip(keys, collated_values))

class ReplayDataset(Dataset):
    def __init__(
        self,
        folder: str | Path,
        fields: tuple[str, ...] | None = None
    ):
        if isinstance(folder, str):
            folder = Path(folder)

        episode_lens = folder / 'episode_lens.data.meta.npy'
        self.episode_lens = open_memmap(str(episode_lens), mode = 'r')

        # get indices of non-zero lengthed episodes

        nonzero_episodes = self.episode_lens > 0
        self.indices = np.arange(self.episode_lens.shape[-1])[nonzero_episodes]

        # get all data files

        filepaths = [*folder.glob('*.data.npy')]
        assert len(filepaths) > 0

        fieldname_to_filepath = {path.name.split('.')[0]: path for path in filepaths}

        fieldnames_from_files = set(fieldname_to_filepath.keys())

        fields = default(fields, fieldnames_from_files)

        self.memmaps = dict()

        for field in fields:
            assert field in fieldnames_from_files, f'invalid field {field} - must be one of {fieldnames_from_files}'

            path = fieldname_to_filepath[field]

            self.memmaps[field] = open_memmap(str(path), mode = 'r')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_index = self.indices[idx]

        episode_len = self.episode_lens[episode_index]

        data = {field: from_numpy(memmap[episode_index, :episode_len].copy()) for field, memmap in self.memmaps.items()}

        data['_lens'] = tensor(episode_len)
        return data

class ReplayBuffer:

    @beartype
    def __init__(
        self,
        folder: str | Path,
        max_episodes: int,
        max_timesteps: int,
        fields: dict[
            str,
            str | tuple[str, int | tuple[int, ...]]
        ],
        meta_fields: dict[
            str,
            str | tuple[str, int | tuple[int, ...]]
        ] = dict()
    ):

        # folder for data

        if not isinstance(folder, Path):
            folder = Path(folder)
            folder.mkdir(exist_ok = True, parents = True)

        self.folder = folder
        assert folder.is_dir()

        # keeping track of episode length

        self.episode_index = 0
        self.timestep_index = 0

        self.num_episodes = 0
        self.max_episodes = max_episodes
        self.max_timesteps= max_timesteps

        assert not 'episode_lens' in meta_fields
        meta_fields.update(episode_lens = 'int')

        # create the memmap for meta data tracks

        self.meta_shapes = dict()
        self.meta_dtypes = dict()
        self.meta_memmaps = dict()
        self.meta_fieldnames = set(meta_fields.keys())

        def parse_field_info(field_info):
            # some flexibility

            field_info = (field_info, ()) if isinstance(field_info, str) else field_info

            dtype_str, shape = field_info
            assert dtype_str in {'int', 'float', 'bool'}

            dtype = dict(int = np.int32, float = np.float32, bool = np.bool_)[dtype_str]
            return dtype, shape

        for field_name, field_info in meta_fields.items():

            dtype, shape = parse_field_info(field_info)

            # memmap file

            filepath = folder / f'{field_name}.data.meta.npy'

            if isinstance(shape, int):
                shape = (shape,)

            memmap = open_memmap(str(filepath), mode = 'w+', dtype = dtype, shape = (max_episodes, *shape))

            self.meta_memmaps[field_name] = memmap
            self.meta_shapes[field_name] = shape
            self.meta_dtypes[field_name] = dtype

        # create the memmap for individual data tracks

        self.shapes = dict()
        self.dtypes = dict()
        self.memmaps = dict()
        self.fieldnames = set(fields.keys())

        for field_name, field_info in fields.items():

            dtype, shape = parse_field_info(field_info)

            # memmap file

            filepath = folder / f'{field_name}.data.npy'

            if isinstance(shape, int):
                shape = (shape,)

            memmap = open_memmap(str(filepath), mode = 'w+', dtype = dtype, shape = (max_episodes, max_timesteps, *shape))

            self.memmaps[field_name] = memmap
            self.shapes[field_name] = shape
            self.dtypes[field_name] = dtype

        self.memory_namedtuple = namedtuple('Memory', list(fields.keys()))

    def __len__(self):
        return (self.episode_lens > 0).sum().item()

    def clear(self):
        rmtree(str(self.folder), ignore_errors = True)

    @property
    def episode_lens(self):
        return self.meta_memmaps['episode_lens']

    def reset_(self):
        self.episode_lens[:] = 0
        self.episode_index = 0
        self.timestep_index = 0

    def advance_episode(self):
        self.episode_index = (self.episode_index + 1) % self.max_episodes
        self.timestep_index = 0
        self.num_episodes += 1

    def flush(self):
        self.episode_lens[self.episode_index] = self.timestep_index

        for memmap in self.memmaps.values():
            memmap.flush()

        self.episode_lens.flush()

    @contextmanager
    def one_episode(self):

        # storing data before exiting the context

        final_meta_data_store = dict()

        yield final_meta_data_store

        # store meta data for use in constructing sequences for learning

        for key, value in final_meta_data_store.items():
            assert key in self.meta_memmaps, f'{key} not defined in `meta_fields` on init'

            self.meta_memmaps[key][self.episode_index] = value

        # flush and advance

        self.flush()
        self.advance_episode()

    @beartype
    def store_datapoint(
        self,
        episode_index: int,
        timestep_index: int,
        name: str,
        datapoint: PrimitiveType | Tensor | ndarray
    ):
        assert 0 <= episode_index < self.max_episodes
        assert 0 <= timestep_index < self.max_timesteps

        if is_bearable(datapoint, PrimitiveType):
            datapoint = tensor(datapoint)
            
        if is_tensor(datapoint):
            datapoint = datapoint.detach().cpu().numpy()

        assert name in self.fieldnames, f'invalid field name {name} - must be one of {self.fieldnames}'

        assert datapoint.shape == self.shapes[name], f'field {name} - invalid shape {datapoint.shape} - shape must be {self.shapes[name]}'

        self.memmaps[name][episode_index, timestep_index] = datapoint

    def store(
        self,
        **data
    ):
        assert not self.timestep_index >= self.max_timesteps, 'you exceeded the `max_timesteps` set on the replay buffer'

        # filter to only what is defined in the namedtuple, and store those that are present

        store_data = dict()

        for name in self.memory_namedtuple._fields:
            datapoint = data.get(name)
            store_data[name] = datapoint

            if exists(datapoint):
                self.store_datapoint(self.episode_index, self.timestep_index, name, datapoint)

        self.timestep_index += 1

        return self.memory_namedtuple(**store_data)

    def dataset(
        self,
        fields: tuple[str, ...] | None = None
    ) -> Dataset:
        self.flush()

        dataset = ReplayDataset(self.folder, fields)
        return dataset

    def dataloader(
        self,
        batch_size,
        dataset: Dataset | None = None,
        fields: tuple[str, ...] | None = None,
        **kwargs
    ) -> DataLoader:
        self.flush()

        if not exists(dataset):
            dataset = self.dataset(fields)

        return DataLoader(dataset, batch_size = batch_size, collate_fn = collate_var_time, **kwargs)
