from __future__ import annotations
from typing import Callable
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Literal

import pickle
from functools import partial
from pathlib import Path
from shutil import rmtree
from contextlib import contextmanager
from collections import namedtuple

import numpy as np
from numpy import ndarray
from numpy.lib.format import open_memmap

import torch
from torch import tensor, from_numpy as torch_from_numpy, stack, cat, is_tensor, Tensor, arange, broadcast_tensors
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, default_collate
from torch.utils._pytree import tree_map

import einx
from einops import rearrange

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

def from_numpy(arr: ndarray):
    if np.isscalar(arr) or arr.ndim == 0:
        arr = np.array(arr)
    return torch_from_numpy(arr)

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

def tree_map_to_device(batch, device):
    if not exists(device):
        return batch
    return tree_map(lambda t: t.to(device) if is_tensor(t) else t, batch)

# data

def collate_var_time(data, fields_to_pad = None):

    datum = first(data)
    keys = datum.keys()

    all_tensors = zip(*[datum.values() for datum in data])

    collated_values = []

    for key, tensors in zip(keys, all_tensors):
        tensors = list(tensors)

        # only pad fields that are explicitly requested as trajectories
        # and have more than 0 dimensions
        
        is_trajectory = exists(fields_to_pad) and key in fields_to_pad
        
        if is_trajectory and tensors[0].ndim > 0:

            times = [t.shape[0] for t in tensors]
            max_time = max(times)
            tensors = [pad_at_dim(t, (0, max_time - t.shape[0]), dim = 0) for t in tensors]

        collated_values.append(stack(tensors))

    return dict(zip(keys, collated_values))

class ReplayDatasetTrajectory(Dataset):
    """Dataset that returns full episodes (variable length trajectories)."""
    
    def __init__(
        self,
        replay_buffer: str | Path | ReplayBuffer,
        fields: tuple[str, ...] | None = None,
        fieldname_map: dict[str, str] | None = None,
        include_metadata: bool = True,
        filter_meta: dict | None = None,
        filter_fields: dict | None = None,
        return_indices: bool = False,
        **kwargs
    ):
        # accept either folder path or ReplayBuffer instance

        if isinstance(replay_buffer, (str, Path)):
            self.replay_buffer = ReplayBuffer.from_folder(replay_buffer)
        else:
            self.replay_buffer = replay_buffer

        self.return_indices = return_indices
        self.fieldname_map = default(fieldname_map, {})

        # start with non-zero length episodes
        
        valid_mask = self.replay_buffer.episode_lens > 0

        # apply meta field filters

        if exists(filter_meta):
            for field_name, filter_value in filter_meta.items():
                field_data = self.replay_buffer.meta_data[field_name]
                if isinstance(filter_value, bool):
                    if filter_value:
                        valid_mask = valid_mask & field_data.astype(bool)
                    else:
                        valid_mask = valid_mask & ~field_data.astype(bool)
                else:
                    valid_mask = valid_mask & (field_data == filter_value)

        self.indices = np.arange(self.replay_buffer.episode_lens.shape[-1])[valid_mask]

        # filter to requested fields

        if exists(fields):
            self.fields = fields
        else:
            self.fields = tuple(self.replay_buffer.fieldnames)

        # apply field filters (trajectory-level, matches if any timestep passes)
        # only supported if timestep_level is False (which it is here)

        if exists(filter_fields):
            for field_name, filter_value in filter_fields.items():
                field_data = self.replay_buffer.data[field_name]
                if isinstance(filter_value, bool):
                    matches = field_data.astype(bool) if filter_value else ~field_data.astype(bool)
                else:
                    matches = field_data == filter_value

                # for trajectory level, we include the episode if ANY timestep matches the filter
                # common use case: filter episodes that have a 'learnable' timestep
                ep_matches = np.any(matches, axis = 1)
                valid_mask = valid_mask & ep_matches[:self.replay_buffer.max_episodes]

        self.indices = np.arange(self.replay_buffer.max_episodes)[valid_mask]

        # exclude episode_lens from meta_data (it's handled separately as _lens)

        self.include_metadata = include_metadata

        if include_metadata:
            self.meta_data = {k: v for k, v in self.replay_buffer.meta_data.items() if k != 'episode_lens'}
        else:
            self.meta_data = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_index = self.indices[idx]

        episode_len = self.replay_buffer.episode_lens[episode_index]

        data = dict()
        for field in self.fields:
            memmap = self.replay_buffer.data[field]
            mapped_name = self.fieldname_map.get(field, field)
            data[mapped_name] = from_numpy(memmap[episode_index, :episode_len].copy())

        # add meta fields (without time dimension)

        for field, memmap in self.meta_data.items():
            mapped_name = self.fieldname_map.get(field, field)
            data[mapped_name] = from_numpy(memmap[episode_index].copy())

        data['_lens'] = tensor(episode_len)

        if self.return_indices:
            data['_index'] = tensor(episode_index)

        return data


class ReplayDatasetTimestep(Dataset):
    """Dataset that returns individual timesteps."""
    
    def __init__(
        self,
        replay_buffer: 'ReplayBuffer',
        fields: tuple[str, ...] | None = None,
        fieldname_map: dict[str, str] | None = None,
        return_indices: bool = False,
        include_metadata: bool = True,
        filter_meta: dict | None = None,
        filter_fields: dict | None = None,
        **kwargs
    ):
        self.replay_buffer = replay_buffer
        self.return_indices = return_indices
        self.fieldname_map = default(fieldname_map, {})

        self.include_metadata = include_metadata

        if include_metadata:
            self.meta_data = {k: v for k, v in self.replay_buffer.meta_data.items() if k != 'episode_lens'}
        else:
            self.meta_data = {}

        episode_ids = arange(replay_buffer.max_episodes)
        episode_lens = from_numpy(replay_buffer.episode_lens)

        max_episode_len = episode_lens.amax().item()

        # start with non-zero length episodes
        
        valid_mask = episode_lens > 0

        # apply meta field filters (episode-level)

        if exists(filter_meta):
            for field_name, filter_value in filter_meta.items():
                field_data = from_numpy(replay_buffer.meta_data[field_name])
                if isinstance(filter_value, bool):
                    if filter_value:
                        valid_mask = valid_mask & field_data.bool()
                    else:
                        valid_mask = valid_mask & ~field_data.bool()
                else:
                    valid_mask = valid_mask & (field_data == filter_value)

        valid_episodes = episode_ids[valid_mask]
        self.valid_episodes = valid_episodes
        valid_episode_lens = episode_lens[valid_mask]

        timesteps = arange(max_episode_len)

        episode_timesteps = stack(broadcast_tensors(
            rearrange(valid_episodes, 'e -> e 1'),
            rearrange(timesteps, 't -> 1 t')
        ), dim = -1)

        valid_timesteps = einx.less('j, i -> i j', timesteps, valid_episode_lens)

        # apply field filters (timestep-level)

        if exists(filter_fields):
            for field_name, filter_value in filter_fields.items():
                field_data = from_numpy(replay_buffer.data[field_name][valid_episodes.numpy(), :max_episode_len])
                if isinstance(filter_value, bool):
                    if filter_value:
                        valid_timesteps = valid_timesteps & field_data.bool()
                    else:
                        valid_timesteps = valid_timesteps & ~field_data.bool()
                else:
                    valid_timesteps = valid_timesteps & (field_data == filter_value)

        self.timepoints = episode_timesteps[valid_timesteps]

        self.fields = default(fields, tuple(replay_buffer.fieldnames))

    def __len__(self):
        return len(self.timepoints)

    def __getitem__(self, idx):
        episode_id, timestep_index = self.timepoints[idx].unbind(dim = -1)

        step_data = dict()

        for field in self.fields:
            data = self.replay_buffer.data[field]
            model_kwarg_name = self.fieldname_map.get(field, field)
            step_data[model_kwarg_name] = from_numpy(data[episode_id, timestep_index].copy())

        # add meta fields (without time dimension)

        for field, memmap in self.meta_data.items():
            mapped_name = self.fieldname_map.get(field, field)
            step_data[mapped_name] = from_numpy(memmap[episode_id].copy())

        if self.return_indices:
            step_data['_indices'] = self.timepoints[idx]

        return step_data


# backwards compatibility alias

ReplayDataset = ReplayDatasetTrajectory

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
        self.meta_data = dict()
        self.meta_defaults = dict()
        self.meta_fieldnames = set(meta_fields.keys())

        def parse_field_info(field_info):
            if isinstance(field_info, str):
                field_info = (field_info, (), None)

            elif isinstance(field_info, tuple) and len(field_info) == 2:
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

            self.meta_data[field_name] = memmap
            self.meta_shapes[field_name] = shape
            self.meta_dtypes[field_name] = dtype
            self.meta_defaults[field_name] = default_value

        # create the memmap for individual data tracks

        self.shapes = dict()
        self.dtypes = dict()
        self.data = dict()
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

            self.data[field_name] = memmap
            self.shapes[field_name] = shape
            self.dtypes[field_name] = dtype
            self.defaults[field_name] = default_value

        self.memory_namedtuple = namedtuple('Memory', list(fields.keys()))

    @classmethod
    def from_folder(cls, folder: str | Path):
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
        for name, memmap in self.data.items():
            default_value = self.defaults[name]
            if exists(default_value):
                memmap[:] = default_value
            else:
                memmap[:] = 0

        for name, memmap in self.meta_data.items():
            default_value = self.meta_defaults[name]
            if exists(default_value):
                memmap[:] = default_value
            else:
                memmap[:] = 0

        self.reset_()
        self.flush()

    @property
    def episode_lens(self):
        return self.meta_data['episode_lens']

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

        if self.timestep_index > 0:
            self.episode_lens[self.episode_index] = self.timestep_index

        for memmap in self.data.values():
            memmap.flush()

        for memmap in self.meta_data.values():
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

        self.data[name][episode_index, timestep_index] = datapoint

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

        self.meta_data[name][episode_index] = datapoint

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
        fields: tuple[str, ...] | None = None,
        timestep_level: bool = False,
        filter_meta: dict | None = None,
        filter_fields: dict | None = None,
        fieldname_map: dict[str, str] | None = None,
        **kwargs
    ) -> Dataset:
        self.flush()
        assert len(self) > 0, 'replay buffer is empty'

        dataset_klass = ReplayDatasetTimestep if timestep_level else ReplayDatasetTrajectory

        return dataset_klass(
            self,
            fields = fields,
            filter_meta = filter_meta,
            filter_fields = filter_fields,
            fieldname_map = fieldname_map,
            **kwargs
        )

    @beartype
    def dataloader(
        self,
        batch_size,
        dataset: Dataset | None = None,
        fields: tuple[str, ...] | None = None,
        filter_meta: dict | None = None,
        filter_fields: dict | None = None,
        fieldname_map: dict[str, str] | None = None,
        return_indices: bool = False,
        return_mask: bool = False,
        timestep_level: bool = False,
        to_named_tuple: tuple[str, ...] | None = None,
        shuffle = False,
        device: torch.device | str | None = None,
        dataset_kwargs: dict = {},
        **kwargs
    ) -> DataLoader:
        self.flush()
        assert len(self) > 0, 'replay buffer is empty'

        # if to_named_tuple is specified, don't filter dataset fields
        if exists(to_named_tuple):
            assert not exists(fields), 'cannot specify both fields and to_named_tuple'

        assert not (return_mask and timestep_level), 'return_mask is only supported for trajectory-level data'

        if not exists(dataset):
            dataset = self.dataset(
                fields = fields,
                timestep_level = timestep_level,
                return_indices = return_indices,
                filter_meta = filter_meta,
                filter_fields = filter_fields,
                fieldname_map = fieldname_map,
                **dataset_kwargs
            )

        # choose appropriate base collation

        if timestep_level:
            base_collate_fn = None  # default collation for fixed-size timesteps
        else:
            # only pad data fields (trajectories), not meta fields or special fields
            fields_to_pad = self.fieldnames
            if exists(fieldname_map):
                fields_to_pad = {fieldname_map.get(f, f) for f in fields_to_pad}

            base_collate_fn = partial(collate_var_time, fields_to_pad = fields_to_pad)

        # wrap collate to convert dict to namedtuple if requested

        NamedTupleCls = None
        if exists(to_named_tuple):
            sanitized_fields = tuple(f.lstrip('_') if f.startswith('_') else f for f in to_named_tuple)
            NamedTupleCls = namedtuple('Batch', sanitized_fields)

        def collate_fn(data):
            if exists(base_collate_fn):
                batch = base_collate_fn(data)
            else:
                batch = default_collate(data)

            if return_mask:
                lens = batch['_lens']
                max_len = lens.amax().item()
                batch['_mask'] = einx.less('j, i -> i j', arange(max_len, device = lens.device), lens)

            if exists(to_named_tuple):
                for field in to_named_tuple:
                    assert field in batch, f'field `{field}` not found in batch. available fields: {list(batch.keys())}'

                batch = NamedTupleCls(**{san: batch[orig] for orig, san in zip(to_named_tuple, sanitized_fields)})

            return tree_map_to_device(batch, device)

        return DataLoader(dataset, batch_size = batch_size, collate_fn = collate_fn, shuffle = shuffle, **kwargs)
