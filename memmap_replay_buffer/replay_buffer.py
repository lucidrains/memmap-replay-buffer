from __future__ import annotations
import pickle
from typing import Callable
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Literal

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

PrimitiveTypeStr = Literal['int', 'float', 'bool']

FieldInfo = (
    PrimitiveTypeStr |
    tuple[PrimitiveTypeStr, int | tuple[int, ...]] |
    tuple[PrimitiveTypeStr, int | tuple[int, ...], PrimitiveType]
)

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

        episode_lens_path = folder / 'episode_lens.data.meta.npy'
        self.episode_lens = open_memmap(str(episode_lens_path), mode = 'r')

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
        fields: dict[str, FieldInfo],
        meta_fields: dict[str, FieldInfo] = dict(),
        circular = False,
        overwrite = True
    ):

        # folder for data

        if not isinstance(folder, Path):
            folder = Path(folder)
            
        folder.mkdir(exist_ok = True, parents = True)

        self.folder = folder
        assert folder.is_dir()

        # save the config if not exists

        config_path = folder / 'metadata.pkl'

        if not config_path.exists() or overwrite:
            config = dict(
                max_episodes = max_episodes,
                max_timesteps = max_timesteps,
                fields = fields,
                meta_fields = meta_fields,
                circular = circular
            )

            with open(str(config_path), 'wb') as f:
                pickle.dump(config, f)

        # keeping track of state

        num_episodes_path = folder / 'num_episodes.state.npy'
        episode_index_path = folder / 'episode_index.state.npy'
        timestep_index_path = folder / 'timestep_index.state.npy'

        self._num_episodes = open_memmap(str(num_episodes_path), mode = 'w+' if not num_episodes_path.exists() or overwrite else 'r+', dtype = np.int32, shape = ())
        self._episode_index = open_memmap(str(episode_index_path), mode = 'w+' if not episode_index_path.exists() or overwrite else 'r+', dtype = np.int32, shape = ())
        self._timestep_index = open_memmap(str(timestep_index_path), mode = 'w+' if not timestep_index_path.exists() or overwrite else 'r+', dtype = np.int32, shape = ())

        if overwrite:
            self.num_episodes = 0
            self.episode_index = 0
            self.timestep_index = 0

        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.circular = circular

        if 'episode_lens' not in meta_fields:
            meta_fields = meta_fields.copy()
            meta_fields.update(episode_lens = 'int')

        # create the memmap for meta data tracks

        self.meta_shapes = dict()
        self.meta_dtypes = dict()
        self.meta_memmaps = dict()
        self.meta_defaults = dict()
        self.meta_fieldnames = set(meta_fields.keys())

        def parse_field_info(field_info):
            if isinstance(field_info, str):
                field_info = (field_info, (), None)

            elif is_bearable(field_info, tuple[str, int | tuple[int, ...]]):
                field_info = (*field_info, None)

            dtype_str, shape, default_value = field_info
            assert dtype_str in {'int', 'float', 'bool'}

            dtype = dict(int = np.int32, float = np.float32, bool = np.bool_)[dtype_str]

            if isinstance(shape, int):
                shape = (shape,)

            return dtype, shape, default_value

        for field_name, field_info in meta_fields.items():

            dtype, shape, default_value = parse_field_info(field_info)

            # memmap file

            filepath = folder / f'{field_name}.data.meta.npy'

            memmap = open_memmap(str(filepath), mode = 'w+' if overwrite or not filepath.exists() else 'r+', dtype = dtype, shape = (max_episodes, *shape))

            if overwrite:
                if exists(default_value):
                    memmap[:] = default_value
                else:
                    memmap[:] = 0

            self.meta_memmaps[field_name] = memmap
            self.meta_shapes[field_name] = shape
            self.meta_dtypes[field_name] = dtype
            self.meta_defaults[field_name] = default_value

        # create the memmap for individual data tracks

        self.shapes = dict()
        self.dtypes = dict()
        self.memmaps = dict()
        self.defaults = dict()
        self.fieldnames = set(fields.keys())

        for field_name, field_info in fields.items():

            dtype, shape, default_value = parse_field_info(field_info)

            # memmap file

            filepath = folder / f'{field_name}.data.npy'

            memmap = open_memmap(str(filepath), mode = 'w+' if overwrite or not filepath.exists() else 'r+', dtype = dtype, shape = (max_episodes, max_timesteps, *shape))

            if overwrite:
                if exists(default_value):
                    memmap[:] = default_value
                else:
                    memmap[:] = 0

            self.memmaps[field_name] = memmap
            self.shapes[field_name] = shape
            self.dtypes[field_name] = dtype
            self.defaults[field_name] = default_value

        self.memory_namedtuple = namedtuple('Memory', list(fields.keys()))

    @classmethod
    def from_config(cls, folder: str | Path):
        if isinstance(folder, str):
            folder = Path(folder)

        config_path = folder / 'metadata.pkl'
        assert config_path.exists(), f'metadata.pkl not found in {folder}'

        with open(str(config_path), 'rb') as f:
            config = pickle.load(f)

        return cls(folder = folder, overwrite = False, **config)

    @property
    def num_episodes(self):
        return self._num_episodes.item()

    @num_episodes.setter
    def num_episodes(self, value):
        self._num_episodes[()] = value
        self._num_episodes.flush()

    @property
    def episode_index(self):
        return self._episode_index.item()

    @episode_index.setter
    def episode_index(self, value):
        self._episode_index[()] = value
        self._episode_index.flush()

    @property
    def timestep_index(self):
        return self._timestep_index.item()

    @timestep_index.setter
    def timestep_index(self, value):
        self._timestep_index[()] = value
        self._timestep_index.flush()

    def __len__(self):
        return (self.episode_lens > 0).sum().item()

    def clear(self):
        for name, memmap in self.memmaps.items():
            default_value = self.defaults[name]
            if exists(default_value):
                memmap[:] = default_value
            else:
                memmap[:] = 0

        for name, memmap in self.meta_memmaps.items():
            default_value = self.meta_defaults[name]
            if exists(default_value):
                memmap[:] = default_value
            else:
                memmap[:] = 0

        self.reset_()
        self.flush()

    @property
    def episode_lens(self):
        return self.meta_memmaps['episode_lens']

    def reset_(self):
        self.episode_lens[:] = 0
        self.num_episodes = 0
        self.episode_index = 0
        self.timestep_index = 0

    def advance_episode(self):
        if not self.circular and self.num_episodes >= self.max_episodes:
            raise ValueError(f'The replay buffer is full ({self.max_episodes} episodes) and is not set to be circular. Please set `circular = True` or clear the buffer.')

        self.episode_lens[self.episode_index] = self.timestep_index

        self.episode_index = (self.episode_index + 1) % self.max_episodes
        self.timestep_index = 0
        self.num_episodes += 1

        if self.circular:
            self.num_episodes = min(self.num_episodes, self.max_episodes)

    def flush(self):
        self.episode_lens[self.episode_index] = self.timestep_index

        for memmap in self.memmaps.values():
            memmap.flush()

        for memmap in self.meta_memmaps.values():
            memmap.flush()

        self._num_episodes.flush()
        self._episode_index.flush()
        self._timestep_index.flush()

    @contextmanager
    def one_episode(self, **meta_data):
        if not self.circular and self.num_episodes >= self.max_episodes:
            raise ValueError(f'The replay buffer is full ({self.max_episodes} episodes) and is not set to be circular. Please set `circular = True` or clear the buffer.')

        for name, value in meta_data.items():
            self.store_meta_datapoint(self.episode_index, name, value)

        final_meta_data_store = dict()

        yield final_meta_data_store

        for name, value in final_meta_data_store.items():
            self.store_meta_datapoint(self.episode_index, name, value)

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

        if is_tensor(datapoint):
            datapoint = datapoint.detach().cpu().numpy()

        if is_bearable(datapoint, PrimitiveType):
            datapoint = np.array(datapoint)

        assert name in self.fieldnames, f'invalid field name {name} - must be one of {self.fieldnames}'

        assert datapoint.shape == self.shapes[name], f'field {name} - invalid shape {datapoint.shape} - shape must be {self.shapes[name]}'

        self.memmaps[name][episode_index, timestep_index] = datapoint

    @beartype
    def store_meta_datapoint(
        self,
        episode_index: int,
        name: str,
        datapoint: PrimitiveType | Tensor | ndarray
    ):
        assert 0 <= episode_index < self.max_episodes

        if is_tensor(datapoint):
            datapoint = datapoint.detach().cpu().numpy()

        if is_bearable(datapoint, PrimitiveType):
            datapoint = np.array(datapoint)

        assert name in self.meta_fieldnames, f'invalid field name {name} - must be one of {self.meta_fieldnames}'

        assert datapoint.shape == self.meta_shapes[name], f'field {name} - invalid shape {datapoint.shape} - shape must be {self.meta_shapes[name]}'

        self.meta_memmaps[name][episode_index] = datapoint

    def store(
        self,
        **data
    ):
        if self.timestep_index >= self.max_timesteps:
            raise ValueError(f'You exceeded the `max_timesteps` ({self.max_timesteps}) set on the replay buffer. Please increase it on init.')

        # filter to only what is defined in the namedtuple, and store those that are present

        store_data = dict()

        for name in self.memory_namedtuple._fields:
            datapoint = data.get(name)

            if not exists(datapoint):
                default_value = self.defaults[name]

                if exists(default_value):
                    datapoint = default_value
                else:
                    datapoint = np.zeros(self.shapes[name], dtype = self.dtypes[name])

            if is_bearable(datapoint, PrimitiveType) or np.isscalar(datapoint):
                datapoint = np.full(self.shapes[name], datapoint, dtype = self.dtypes[name])

            store_data[name] = datapoint
            self.store_datapoint(self.episode_index, self.timestep_index, name, datapoint)

        self.timestep_index += 1

        return self.memory_namedtuple(**store_data)

    @beartype
    def dataset(
        self,
        fields: tuple[str, ...] | None = None
    ) -> Dataset:
        self.flush()

        dataset = ReplayDataset(self.folder, fields)
        return dataset

    @beartype
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
