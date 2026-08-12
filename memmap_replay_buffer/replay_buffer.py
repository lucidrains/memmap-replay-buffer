from __future__ import annotations

import pickle
import warnings
from collections import namedtuple
from contextlib import contextmanager, suppress
from functools import partial, wraps
from pathlib import Path

import einx
import numpy as np
import torch
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Literal
from einops import rearrange
from numpy import ndarray
from numpy.lib.format import open_memmap
from torch import Tensor, arange, broadcast_tensors, is_tensor, stack, tensor
from torch import from_numpy as torch_from_numpy
from torch.nn.functional import pad as pad_tensor
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader, Dataset, default_collate

# constants

PrimitiveType = int | float | bool

PrimitiveTypeStr = Literal['int', 'float', 'bool', 'uint8']

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

def cast_to_target_shape(value, target_shape, is_time_varying = False):
    input_shape = value.shape[1:] if is_time_varying else value.shape

    if target_shape == () and input_shape == (1,):
        return value.squeeze(-1)
    elif target_shape == (1,) and input_shape == ():
        return np.expand_dims(value, -1)

    return value

def xnor(x, y):
    return not (x ^ y)

def is_empty(t):
    return t.numel() == 0

def divisible_by(num, den):
    return (num % den) == 0

def from_numpy(arr: ndarray):
    arr = np.asarray(arr)

    if arr.ndim == 0:
        arr = np.array(arr)

    if not arr.flags.writeable:
        arr = arr.copy()

    return torch_from_numpy(arr)

def pad_right_at_dim(t, amount, dim = 0):
    if amount <= 0:
        return t

    ndim = t.ndim
    pad = [0] * (2 * ndim)
    pad[2 * (ndim - 1 - dim) + 1] = amount
    return pad_tensor(t, pad)

def pad_at_dim(t, padding, dim = 0):
    left, right = padding

    if left != 0:
        raise ValueError(f'only right padding is supported, got left padding of {left}')

    return pad_right_at_dim(t, right, dim = dim)

def pad_right_at_dim_to(t, target, dim = 0):
    return pad_right_at_dim(t, target - t.shape[dim], dim = dim)

def tree_map_to_device(batch, device):
    if not exists(device):
        return batch
    return tree_map(lambda t: t.to(device) if is_tensor(t) else t, batch)

def can_write(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        if self.read_only:
            raise ValueError(f'cannot call `{fn.__name__}` in read-only mode')
        return fn(self, *args, **kwargs)
    return inner

# data

def collate_var_time(data, fields_to_pad = None):
    datum = first(data)
    keys = datum.keys()

    all_tensors = zip(*[datum.values() for datum in data])

    collated_values = []

    for key, tensors in zip(keys, all_tensors):
        tensors = list(tensors)

        is_trajectory = exists(fields_to_pad) and key in fields_to_pad

        if is_trajectory and tensors[0].ndim > 0:
            max_time = max(t.shape[0] for t in tensors)
            tensors = [pad_at_dim(t, (0, max_time - t.shape[0]), dim = 0) for t in tensors]

        collated_values.append(stack(tensors))

    return dict(zip(keys, collated_values))

class ReplayDatasetTrajectory(Dataset):

    def __init__(
        self,
        replay_buffer: str | Path | ReplayBuffer,
        fields: tuple[str, ...] | None = None,
        fieldname_map: dict[str, str] | None = None,
        include_metadata: bool = True,
        filter_meta: dict | None = None,
        filter_fields: dict | None = None,
        return_indices: bool = False,
        slice_by_episode_len: bool = True,
        return_episode_lens: bool = True,
        **kwargs
    ):
        if isinstance(replay_buffer, (str, Path)):
            self.replay_buffer = ReplayBuffer.from_folder(replay_buffer)
        else:
            self.replay_buffer = replay_buffer

        self.return_indices = return_indices
        self.slice_by_episode_len = slice_by_episode_len
        self.return_episode_lens = return_episode_lens
        self.fieldname_map = default(fieldname_map, {})
        self.meta_data = {k: v for k, v in self.replay_buffer.meta_data.items() if k not in self.replay_buffer.internal_meta_fieldnames} if include_metadata else {}
        self.fields = default(fields, tuple(self.replay_buffer.fieldnames))

        if exists(filter_fields):
            raise ValueError('filter_fields is only supported for timestep-level and n-step datasets')

        if not slice_by_episode_len and not return_episode_lens:
            raise ValueError('cannot turn off return_episode_lens if slice_by_episode_len is also turned off')

        episode_ids = arange(self.replay_buffer.max_episodes)
        episode_lens = from_numpy(self.replay_buffer.episode_lens)

        valid_mask = episode_lens > 0

        if exists(filter_meta):
            for field_name, filter_value in filter_meta.items():
                field_data = from_numpy(self.replay_buffer.meta_data[field_name])
                if isinstance(filter_value, bool):
                    field_data = field_data.bool()
                valid_mask &= field_data == filter_value

        self.indices = episode_ids[valid_mask]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_index = self.indices[idx].item()

        episode_len = self.replay_buffer.episode_lens[episode_index]

        data = dict()

        for field in self.fields:
            name = self.fieldname_map.get(field, field)

            arr = self.replay_buffer.data[field][episode_index]

            if self.slice_by_episode_len:
                arr = arr[:episode_len]

            data[name] = from_numpy(arr.copy())

        for field, memmap in self.meta_data.items():
            name = self.fieldname_map.get(field, field)
            data[name] = from_numpy(memmap[episode_index].copy())

        if self.return_episode_lens:
            data['_lens'] = tensor(episode_len)

        if self.return_indices:
            data['_index'] = tensor(episode_index)

        return data


class ReplayDatasetTimeWindow(Dataset):
    def __init__(
        self,
        replay_buffer: str | Path | ReplayBuffer,
        window_length: int,
        fields: tuple[str, ...] | None = None,
        fieldname_map: dict[str, str] | None = None,
        include_metadata: bool = True,
        filter_meta: dict | None = None,
        return_indices: bool = False,
        **kwargs
    ):
        if isinstance(replay_buffer, (str, Path)):
            self.replay_buffer = ReplayBuffer.from_folder(replay_buffer)
        else:
            self.replay_buffer = replay_buffer

        self.window_length = window_length
        self.return_indices = return_indices
        self.fieldname_map = default(fieldname_map, {})
        self.meta_data = {k: v for k, v in self.replay_buffer.meta_data.items() if k not in self.replay_buffer.internal_meta_fieldnames} if include_metadata else {}
        self.fields = default(fields, tuple(self.replay_buffer.fieldnames))

        episode_lens = from_numpy(self.replay_buffer.episode_lens)
        valid_mask = episode_lens > 0

        for field, value in default(filter_meta, {}).items():
            field_data = from_numpy(self.replay_buffer.meta_data[field])
            if isinstance(value, bool):
                field_data = field_data.bool()
            valid_mask &= (field_data == value)

        self.indices = arange(self.replay_buffer.max_episodes)[valid_mask]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_index = self.indices[idx].item()
        episode_len = self.replay_buffer.episode_lens[episode_index]
        window_len = self.window_length

        actual_len = min(window_len, episode_len)
        start = torch.randint(0, episode_len - actual_len + 1, (1,)).item()
        end = start + actual_len

        data = dict()

        for field in self.fields:
            name = self.fieldname_map.get(field, field)
            t = from_numpy(self.replay_buffer.data[field][episode_index, start:end].copy())

            if actual_len < window_len:
                t = pad_right_at_dim_to(t, window_len, dim = 0)

            data[name] = t

        for field, memmap in self.meta_data.items():
            name = self.fieldname_map.get(field, field)
            data[name] = from_numpy(memmap[episode_index].copy())

        data.update(
            _lens = tensor(actual_len, dtype = torch.long),
            _reaches_episode_end = tensor(end >= episode_len, dtype = torch.bool)
        )

        if self.return_indices:
            data.update(
                _index = tensor(episode_index),
                _start = tensor(start)
            )

        return data

class ReplayDatasetTimestep(Dataset):

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
        self.meta_data = {k: v for k, v in replay_buffer.meta_data.items() if k not in replay_buffer.internal_meta_fieldnames} if include_metadata else {}
        self.fields = default(fields, tuple(replay_buffer.fieldnames))

        episode_ids = arange(replay_buffer.max_episodes)
        episode_lens = from_numpy(replay_buffer.episode_lens)
        max_episode_len = episode_lens.amax().item()

        valid_mask = episode_lens > 0

        if exists(filter_meta):
            for field_name, filter_value in filter_meta.items():
                field_data = from_numpy(replay_buffer.meta_data[field_name])
                if isinstance(filter_value, bool):
                    field_data = field_data.bool()
                valid_mask &= field_data == filter_value

        valid_episodes = episode_ids[valid_mask]
        valid_episode_lens = episode_lens[valid_mask]

        self.valid_episodes = valid_episodes
        self.valid_episode_lens = valid_episode_lens

        timesteps = arange(max_episode_len)

        episode_timesteps = stack(broadcast_tensors(
            rearrange(valid_episodes, 'e -> e 1'),
            rearrange(timesteps, 't -> 1 t')
        ), dim = -1)

        valid_timesteps = einx.less('j, i -> i j', timesteps, valid_episode_lens)

        if exists(filter_fields):
            for field_name, filter_value in filter_fields.items():
                field_data = stack([from_numpy(replay_buffer.data[field_name][ep, :max_episode_len]) for ep in valid_episodes.numpy()])
                if isinstance(filter_value, bool):
                    field_data = field_data.bool()

                if field_data.ndim != 2:
                    raise ValueError(f'filter_fields only supports scalar fields, got shape {field_data.shape[2:]} for `{field_name}`')

                valid_timesteps &= field_data == filter_value

        self.timepoints = episode_timesteps[valid_timesteps]

    def __len__(self):
        return len(self.timepoints)

    def __getitem__(self, idx):
        ep, t = self.timepoints[idx].tolist()

        step_data = dict()

        for field in self.fields:
            name = self.fieldname_map.get(field, field)
            step_data[name] = from_numpy(self.replay_buffer.data[field][ep, t].copy())

        for field, memmap in self.meta_data.items():
            name = self.fieldname_map.get(field, field)
            step_data[name] = from_numpy(memmap[ep].copy())

        if self.return_indices:
            step_data['_indices'] = self.timepoints[idx]

        return step_data


class ReplayDatasetNStep(Dataset):

    def __init__(
        self,
        replay_buffer: 'ReplayBuffer',
        n_steps: int,
        current_fields: tuple[str, ...] | None = None,
        next_fields: tuple[str, ...] | None = None,
        sequence_fields: tuple[str, ...] | None = None,
        fieldname_map: dict[str, str] | None = None,
        return_indices: bool = False,
        include_metadata: bool = True,
        filter_meta: dict | None = None,
        filter_fields: dict | None = None,
        **kwargs
    ):
        self.replay_buffer = replay_buffer
        self.n_steps = n_steps
        self.return_indices = return_indices
        self.fieldname_map = default(fieldname_map, {})
        self.meta_data = {k: v for k, v in replay_buffer.meta_data.items() if k not in replay_buffer.internal_meta_fieldnames} if include_metadata else {}

        self.current_fields = default(current_fields, tuple(replay_buffer.fieldnames))
        self.next_fields = default(next_fields, tuple(replay_buffer.fieldnames))
        self.sequence_fields = default(sequence_fields, ())

        episode_ids = arange(replay_buffer.max_episodes)
        episode_lens = from_numpy(replay_buffer.episode_lens)

        # need at least 2 timesteps for a valid transition

        valid_mask = episode_lens >= 2

        if exists(filter_meta):
            for field_name, filter_value in filter_meta.items():
                field_data = from_numpy(replay_buffer.meta_data[field_name])
                if isinstance(filter_value, bool):
                    field_data = field_data.bool()
                valid_mask &= field_data == filter_value

        valid_episodes = episode_ids[valid_mask]
        valid_episode_lens = episode_lens[valid_mask]

        self.valid_episodes = valid_episodes
        self.valid_episode_lens = valid_episode_lens

        if len(valid_episodes) == 0:
            self.timepoints = tensor([], dtype = torch.long).reshape(0, 2)
        else:
            max_len = valid_episode_lens.amax().item()
            timesteps = arange(max_len - 1)

            episode_timesteps = stack(broadcast_tensors(
                rearrange(valid_episodes, 'e -> e 1'),
                rearrange(timesteps, 't -> 1 t')
            ), dim = -1)

            valid_timesteps = einx.less('j, i -> i j', timesteps, valid_episode_lens - 1)

            if exists(filter_fields):
                for field_name, filter_value in filter_fields.items():
                    field_data = from_numpy(replay_buffer.data[field_name][valid_episodes.numpy(), :max_len - 1])
                    if isinstance(filter_value, bool):
                        field_data = field_data.bool()

                    if field_data.ndim != 2:
                        raise ValueError(f'filter_fields only supports scalar fields, got shape {field_data.shape[2:]} for `{field_name}`')

                    valid_timesteps &= field_data == filter_value

            self.timepoints = episode_timesteps[valid_timesteps]

    def __len__(self):
        return len(self.timepoints)

    def __getitem__(self, idx):
        ep, t = self.timepoints[idx].tolist()

        episode_len = self.replay_buffer.episode_lens[ep]
        actual_n = min(self.n_steps, episode_len - t - 1)

        step_data = dict()

        for field in self.current_fields:
            name = self.fieldname_map.get(field, field)
            step_data[name] = from_numpy(self.replay_buffer.data[field][ep, t].copy())

        for field in self.next_fields:
            default_name = f'next_{field}'
            name = self.fieldname_map.get(default_name, default_name)
            step_data[name] = from_numpy(self.replay_buffer.data[field][ep, t + actual_n].copy())

        for field in self.sequence_fields:
            default_name = f'seq_{field}'
            name = self.fieldname_map.get(default_name, default_name)
            seq = from_numpy(self.replay_buffer.data[field][ep, t:t + actual_n].copy())
            step_data[name] = pad_right_at_dim_to(seq, self.n_steps, dim = 0)

        step_data[self.fieldname_map.get('n_step_lens', 'n_step_lens')] = tensor(actual_n, dtype = torch.long)

        for field, memmap in self.meta_data.items():
            name = self.fieldname_map.get(field, field)
            step_data[name] = from_numpy(memmap[ep].copy())

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
        meta_fields: dict[str, FieldInfo] | None = None,
        circular = False,
        overwrite = True,
        read_only = False,
        flush_every_store_step: int = 1
    ):
        meta_fields = default(meta_fields, dict())

        self.read_only = read_only

        assert not (read_only and overwrite), 'cannot overwrite a buffer in read-only mode'

        # folder for data

        if not isinstance(folder, Path):
            folder = Path(folder)

        if read_only:
            assert folder.is_dir(), f'cannot open folder `{folder}` in read-only mode - folder does not exist'
        else:
            folder.mkdir(exist_ok = True, parents = True)

        self.folder = folder
        assert folder.is_dir()

        # save the config if not exists, and validate against any existing one

        config_path = folder / 'metadata.pkl'

        if config_path.exists() and not overwrite:
            with open(str(config_path), 'rb') as f:
                stored_config = pickle.load(f)

            init_locals = locals()

            mismatched_keys = [key for key, value in stored_config.items() if init_locals[key] != value]

            if len(mismatched_keys) > 0:
                mismatch_lines = [f'  {key}: stored {stored_config[key]!r} vs passed {init_locals[key]!r}' for key in mismatched_keys]
                mismatch_str = '\n'.join(mismatch_lines)
                raise ValueError(f'buffer at `{folder}` was created with a different config:\n{mismatch_str}\nPass `overwrite = True` to recreate the buffer with the new config.')

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

        state_paths = (num_episodes_path, episode_index_path, timestep_index_path)

        if read_only:
            assert all(path.exists() for path in state_paths), f'state files not found in `{folder}` - cannot open in read-only mode'
            state_mode = 'r'
        else:
            state_mode = 'w+' if not num_episodes_path.exists() or overwrite else 'r+'

        self._num_episodes = open_memmap(str(num_episodes_path), mode = state_mode, dtype = np.int32, shape = ())
        self._episode_index = open_memmap(str(episode_index_path), mode = state_mode, dtype = np.int32, shape = ())
        self._timestep_index = open_memmap(str(timestep_index_path), mode = state_mode, dtype = np.int32, shape = ())

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

        if '_initted' not in meta_fields:
            meta_fields = meta_fields.copy()
            meta_fields.update(_initted = 'bool')

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
            assert dtype_str in {'int', 'float', 'bool', 'uint8'}

            dtype = dict(int = np.int32, float = np.float32, bool = np.bool_, uint8 = np.uint8)[dtype_str]

            if isinstance(shape, int):
                shape = (shape,)

            return dtype, shape, default_value

        for field_name, field_info in meta_fields.items():

            dtype, shape, default_value = parse_field_info(field_info)

            # memmap file

            filepath = folder / f'{field_name}.data.meta.npy'

            if read_only:
                assert filepath.exists(), f'`{filepath}` not found - cannot open in read-only mode'
                memmap_mode = 'r'
            else:
                memmap_mode = 'w+' if overwrite or not filepath.exists() else 'r+'

            memmap = open_memmap(str(filepath), mode = memmap_mode, dtype = dtype, shape = (max_episodes, *shape))

            self.meta_data[field_name] = memmap
            self.meta_shapes[field_name] = shape
            self.meta_dtypes[field_name] = dtype
            self.meta_defaults[field_name] = default_value

        self.internal_meta_fieldnames = {'episode_lens', '_initted'}

        # create the memmap for individual data tracks

        self.shapes = dict()
        self.dtypes = dict()
        self.data = dict()
        self.defaults = dict()
        self.fieldnames = set(fields.keys())

        if not self.fieldnames.isdisjoint(self.meta_fieldnames):
            raise ValueError(f'fields and meta_fields must be disjoint - shared {self.fieldnames & self.meta_fieldnames}')

        for field_name, field_info in fields.items():

            dtype, shape, default_value = parse_field_info(field_info)

            # memmap file

            filepath = folder / f'{field_name}.data.npy'

            if read_only:
                assert filepath.exists(), f'`{filepath}` not found - cannot open in read-only mode'
                memmap_mode = 'r'
            else:
                memmap_mode = 'w+' if overwrite or not filepath.exists() else 'r+'

            memmap = open_memmap(str(filepath), mode = memmap_mode, dtype = dtype, shape = (max_episodes, max_timesteps, *shape))

            self.data[field_name] = memmap
            self.shapes[field_name] = shape
            self.dtypes[field_name] = dtype
            self.defaults[field_name] = default_value

        self.memory_namedtuple = namedtuple('Memory', list(fields.keys()))

        # how often to flush for store

        if flush_every_store_step < 0:
            raise ValueError(f'flush_every_store_step must be >= 0, got {flush_every_store_step}')

        self.store_step = 0
        self.should_flush = flush_every_store_step > 0
        self.flush_every_store_step = flush_every_store_step

    def __del__(self):
        with suppress(Exception):
            if self.read_only:
                return
            self.flush()

    @classmethod
    def from_folder(cls, folder: str | Path, read_only: bool = False):
        if isinstance(folder, str):
            folder = Path(folder)

        config_path = folder / 'metadata.pkl'

        if not config_path.exists():
            raise ValueError(f'metadata.pkl not found in {folder}')

        with open(str(config_path), 'rb') as f:
            config = pickle.load(f)

        return cls(folder = folder, overwrite = False, read_only = read_only, **config)

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

    @property
    def is_new_episode(self):
        return self.timestep_index == 0

    def __len__(self):
        return (self.episode_lens > 0).sum().item()

    @can_write
    def clear(self):
        self.reset_()
        self.flush()

    @property
    def episode_lens(self):
        return self.meta_data['episode_lens']

    @property
    def _initted(self):
        return self.meta_data['_initted']

    @can_write
    def reset_(self):
        self.episode_lens[:] = 0
        self._initted[:] = False
        self.num_episodes = 0
        self.episode_index = 0
        self.timestep_index = 0

    @can_write
    def advance_episode(self, batch_size = 1):

        # if episode length is 0, and not batching, do not advance

        if self.is_new_episode and batch_size == 1:
            return

        if not self.circular and self.num_episodes + batch_size > self.max_episodes:
            raise ValueError(f'The replay buffer is full ({self.max_episodes} episodes) and is not set to be circular. Please set `circular = True` or clear the buffer.')

        indices = np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes

        self.episode_lens[indices] = self.timestep_index

        self.episode_index = (self.episode_index + batch_size) % self.max_episodes
        self.timestep_index = 0
        self.num_episodes += batch_size

        if self.circular:
            self.num_episodes = min(self.num_episodes, self.max_episodes)

        # mark next newly active episode(s) as uninitialized so they lazily overwrite old data with defaults
        next_indices = np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes
        self._initted[next_indices] = False

    @can_write
    def _lazy_init_episodes(self, indices: ndarray):
        is_initted = self._initted[indices]

        if is_initted.all():
            return

        uninit_indices = np.unique(indices[~is_initted])

        # fill meta fields with defaults

        for name, memmap in self.meta_data.items():
            if name in self.internal_meta_fieldnames:
                continue
            memmap[uninit_indices] = default(self.meta_defaults[name], 0)

        # fill data fields with defaults

        for name, memmap in self.data.items():
            memmap[uninit_indices] = default(self.defaults[name], 0)

        self._initted[uninit_indices] = True

    @can_write
    @beartype
    def _store_batch(
        self,
        data: dict[str, Tensor | ndarray | list | tuple],
        is_meta = False
    ):
        if len(data) == 0:
            raise ValueError(f'No data provided to {"store_meta_batch" if is_meta else "store_batch"}')

        fieldnames = self.meta_fieldnames if is_meta else self.fieldnames

        if not set(data.keys()).issubset(fieldnames):
            raise ValueError(f'invalid {"meta " if is_meta else ""}field names {set(data.keys()) - fieldnames} - must be a subset of {fieldnames}')

        # get batch size

        batch_size = None

        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                value = tensor(value)
                data[key] = value

            curr_batch_size = value.shape[0]

            if not exists(batch_size):
                batch_size = curr_batch_size

            if batch_size != curr_batch_size:
                raise ValueError(f'All data in batch must have the same batch size. Field {key} has batch size {curr_batch_size} while previous fields had {batch_size}.')

        if not exists(batch_size):
            raise ValueError('Could not determine batch size from data')

        if not is_meta and self.timestep_index >= self.max_timesteps:
            raise ValueError(f'You exceeded the `max_timesteps` ({self.max_timesteps}) set on the replay buffer. Please increase it on init.')

        # handle non-circular buffer constraints

        if not self.circular:
            remaining_episodes = self.max_episodes - self.num_episodes

            if remaining_episodes <= 0:
                raise ValueError(f'The replay buffer is full ({self.max_episodes} episodes) and is not set to be circular. Please set `circular = True` or clear the buffer.')

            if remaining_episodes < batch_size:
                # slice data before inserting
                data = {k: v[:remaining_episodes] for k, v in data.items()}
                batch_size = remaining_episodes

        # compute row indices

        indices = np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes

        if self.is_new_episode:
            self._lazy_init_episodes(indices)

        # store data

        for name, values in data.items():
            if is_meta:
                self.store_batch_meta_datapoint(indices, name, values)
            else:
                self.store_batch_datapoint(indices, self.timestep_index, name, values)

        # update state

        if not is_meta:
            self.episode_lens[indices] = self.timestep_index + 1
            self.timestep_index += 1

        if self.should_flush:
            self.flush()

    @can_write
    def store_batch(self, **data):
        return self._store_batch(data, is_meta = False)

    @can_write
    def store_meta_batch(self, **data):
        return self._store_batch(data, is_meta = True)

    def flush(self):
        if self.read_only:
            return

        if self.timestep_index > 0:
            self.episode_lens[self.episode_index] = self.timestep_index

        for memmap in self.data.values():
            memmap.flush()

        for memmap in self.meta_data.values():
            memmap.flush()

        self._num_episodes.flush()
        self._episode_index.flush()
        self._timestep_index.flush()

    @can_write
    @contextmanager
    def one_episode(self, **meta_data):

        if not self.circular and self.num_episodes >= self.max_episodes:
            raise ValueError(f'The replay buffer is full ({self.max_episodes} episodes) and is not set to be circular. Please set `circular = True` or clear the buffer.')

        self._lazy_init_episodes(np.array([self.episode_index]))

        for name, value in meta_data.items():
            self.store_meta_datapoint(self.episode_index, name, value)

        final_meta_data_store = dict()

        try:
            yield final_meta_data_store
        except Exception:
            self.timestep_index = 0
            raise

        for name, value in final_meta_data_store.items():
            self.store_meta_datapoint(self.episode_index, name, value)

        self.flush()
        self.advance_episode()

    @can_write
    @contextmanager
    def batched_episode(self, batch_size, **meta_batch):

        if not self.circular and self.num_episodes + batch_size > self.max_episodes:
            raise ValueError(f'The replay buffer is full ({self.max_episodes} episodes) and is not set to be circular. Please set `circular = True` or clear the buffer.')

        next_indices = np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes
        self._lazy_init_episodes(next_indices)

        if len(meta_batch) > 0:
            self.store_meta_batch(**meta_batch)

        try:
            yield
        except Exception:
            self.timestep_index = 0
            raise

        self.flush()
        self.advance_episode(batch_size = batch_size)

    @can_write
    def store_datapoint(
        self,
        episode_index: int,
        timestep_index: int,
        name: str,
        datapoint: PrimitiveType | Tensor | ndarray
    ):

        if not (0 <= episode_index < self.max_episodes):
            raise ValueError(f'episode_index {episode_index} out of range - must be in [0, {self.max_episodes})')

        if not (0 <= timestep_index < self.max_timesteps):
            raise ValueError(f'timestep_index {timestep_index} out of range - must be in [0, {self.max_timesteps})')

        if is_tensor(datapoint):
            datapoint = datapoint.detach().cpu().numpy()

        if is_bearable(datapoint, PrimitiveType):
            datapoint = np.array(datapoint)

        if name not in self.fieldnames:
            raise ValueError(f'invalid field name {name} - must be one of {self.fieldnames}')

        if datapoint.shape != self.shapes[name]:
            raise ValueError(f'field {name} - invalid shape {datapoint.shape} - shape must be {self.shapes[name]}')

        self.data[name][episode_index, timestep_index] = datapoint

    @can_write
    def store_meta_datapoint(
        self,
        episode_index: int,
        name: str,
        datapoint: PrimitiveType | Tensor | ndarray
    ):

        if not (0 <= episode_index < self.max_episodes):
            raise ValueError(f'episode_index {episode_index} out of range - must be in [0, {self.max_episodes})')

        if is_tensor(datapoint):
            datapoint = datapoint.detach().cpu().numpy()

        if is_bearable(datapoint, PrimitiveType):
            datapoint = np.array(datapoint)

        if name not in self.meta_fieldnames:
            raise ValueError(f'invalid field name {name} - must be one of {self.meta_fieldnames}')

        if datapoint.shape != self.meta_shapes[name]:
            raise ValueError(f'field {name} - invalid shape {datapoint.shape} - shape must be {self.meta_shapes[name]}')

        self.meta_data[name][episode_index] = datapoint

    @can_write
    def store_batch_datapoint(
        self,
        episode_indices: ndarray,
        timestep_index: int,
        name: str,
        datapoints: Tensor | ndarray
    ):
        if is_tensor(datapoints):
            datapoints = datapoints.detach().cpu().numpy()

        if name not in self.fieldnames:
            raise ValueError(f'invalid field name {name} - must be one of {self.fieldnames}')

        self.data[name][episode_indices, timestep_index] = datapoints

    @can_write
    def store_batch_meta_datapoint(
        self,
        episode_indices: ndarray,
        name: str,
        datapoints: Tensor | ndarray
    ):
        if is_tensor(datapoints):
            datapoints = datapoints.detach().cpu().numpy()

        if name not in self.meta_fieldnames:
            raise ValueError(f'invalid field name {name} - must be one of {self.meta_fieldnames}')

        self.meta_data[name][episode_indices] = datapoints

    @can_write
    def store(
        self,
        **data
    ):

        if self.timestep_index >= self.max_timesteps:
            raise ValueError(f'You exceeded the `max_timesteps` ({self.max_timesteps}) set on the replay buffer. Please increase it on init.')

        if self.is_new_episode:
            self._lazy_init_episodes(np.array([self.episode_index]))

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
        self.store_step += 1

        if self.should_flush and divisible_by(self.store_step, self.flush_every_store_step):
            self.flush()

        return self.memory_namedtuple(**store_data)

    @can_write
    def store_episode(
        self,
        **data
    ):
        if not self.is_new_episode:
            warnings.warn(f'timestep index is not 0 ({self.timestep_index}) when calling `store_episode`. This will overwrite the current episode from the beginning.')

        if len(data) == 0:
            raise ValueError('No data provided to `store_episode`')

        # lazy init uninitialized episode
        self._lazy_init_episodes(np.array([self.episode_index]))

        # validate all fields have same time dimension

        time_dim = None

        for name, value in data.items():
            if is_tensor(value):
                value = value.detach().cpu().numpy()

            if isinstance(value, (list, tuple)):
                value = np.array(value)

            if np.isscalar(value):
                value = np.array(value)

            is_time_varying = name in self.fieldnames
            is_meta = name in self.meta_fieldnames

            if not (is_time_varying or is_meta):
                raise ValueError(f'invalid field name {name} - must be one of {self.fieldnames} or {self.meta_fieldnames}')

            if is_time_varying:
                curr_time_dim = value.shape[0]

                if not exists(time_dim):
                    time_dim = curr_time_dim

                if time_dim != curr_time_dim:
                    raise ValueError(f'all fields must have the same time dimension. field {name} has {curr_time_dim} while previous fields had {time_dim}')

                # auto-squeeze/unsqueeze logic for shapes () and (1,)
                value = cast_to_target_shape(value, self.shapes[name], is_time_varying = True)

                if value.shape[1:] != self.shapes[name]:
                    raise ValueError(f'field {name} - invalid shape {value.shape[1:]} - shape must be {self.shapes[name]}')

                if time_dim > self.max_timesteps:
                    raise ValueError(f'You exceeded the `max_timesteps` ({self.max_timesteps}) set on the replay buffer. Please increase it on init.')

                self.data[name][self.episode_index, :time_dim] = value

            elif is_meta:
                # auto-squeeze/unsqueeze logic for shapes () and (1,)
                target_shape = self.shapes[name] if name in self.shapes else self.meta_shapes[name]
                value = cast_to_target_shape(value, target_shape, is_time_varying = False)

                if value.shape != self.meta_shapes[name]:
                    raise ValueError(f'meta field {name} - invalid shape {value.shape} - shape must be {self.meta_shapes[name]}')

                self.meta_data[name][self.episode_index] = value

        if not exists(time_dim):
            raise ValueError('At least one time-varying field must be provided to store_episode')

        self.timestep_index = time_dim
        self.advance_episode()

    @can_write
    @beartype
    def update(
        self,
        indices: int | list | ndarray | slice | None = None,
        **data
    ):
        if len(data) == 0:
            raise ValueError('No data provided to `update`')

        # normalize indices

        if not exists(indices):
            indices = np.where(self.episode_lens > 0)[0]
            scalar_index = False
        elif isinstance(indices, slice):
            indices = np.arange(*indices.indices(self.max_episodes))
            scalar_index = False
        elif np.isscalar(indices):
            indices = np.array([indices])
            scalar_index = True
        else:
            indices = np.atleast_1d(np.asarray(indices))
            scalar_index = False

        for name, value in data.items():
            if is_tensor(value):
                value = value.detach().cpu().numpy()

            if isinstance(value, (list, tuple)):
                value = np.array(value)

            if np.isscalar(value):
                value = np.array(value)

            if scalar_index:
                value = np.expand_dims(value, 0)

            is_time_varying = name in self.fieldnames
            is_meta = name in self.meta_fieldnames

            if not (is_time_varying or is_meta):
                raise ValueError(f'invalid field name `{name}`')

            if is_time_varying:
                time_dim = value.shape[1]
                value = cast_to_target_shape(value, self.shapes[name], is_time_varying = True)

                if time_dim > self.max_timesteps:
                    raise ValueError(f'You exceeded the `max_timesteps` ({self.max_timesteps}) set on the replay buffer. Please increase it on init.')

                self.data[name][indices, :time_dim] = value

            elif is_meta:
                target_shape = self.meta_shapes[name]
                value = cast_to_target_shape(value, target_shape, is_time_varying = False)
                self.meta_data[name][indices] = value

        if self.should_flush:
            self.flush()

    def get_all_data(
        self,
        fields: tuple[str, ...] | None = None,
        meta_fields: tuple[str, ...] | None = None,
        device: torch.device | str | None = None
    ):
        self.flush()

        n = self.num_episodes

        if n == 0:
            return dict()

        max_len = self.episode_lens[:n].max()

        all_data = dict()

        # sub-select fields and meta fields

        if not exists(fields) and not exists(meta_fields):
            data_fields = self.fieldnames
            meta_data_fields = tuple(f for f in self.meta_fieldnames if f != '_initted')
        else:
            data_fields = default(fields, ())
            meta_data_fields = default(meta_fields, ())

        for name in data_fields:
            memmap = self.data[name]
            all_data[name] = from_numpy(memmap[:n, :max_len].copy())

        for name in meta_data_fields:
            memmap = self.meta_data[name]
            all_data[name] = from_numpy(memmap[:n].copy())

        return tree_map_to_device(all_data, device)

    @beartype
    def dataset(
        self,
        n_steps: int | None = None,
        fields: tuple[str, ...] | None = None,
        current_fields: tuple[str, ...] | None = None,
        next_fields: tuple[str, ...] | None = None,
        sequence_fields: tuple[str, ...] | None = None,
        timestep_level: bool = False,
        filter_meta: dict | None = None,
        filter_fields: dict | None = None,
        fieldname_map: dict[str, str] | None = None,
        slice_by_episode_len: bool = True,
        return_episode_lens: bool = True,
        **kwargs
    ) -> Dataset:
        self.flush()

        if len(self) == 0:
            raise ValueError('replay buffer is empty')

        if exists(n_steps) and timestep_level:
            raise ValueError('cannot specify both n_steps and timestep_level')

        if exists(n_steps):
            return ReplayDatasetNStep(
                self,
                n_steps = n_steps,
                current_fields = current_fields,
                next_fields = next_fields,
                sequence_fields = sequence_fields,
                filter_meta = filter_meta,
                filter_fields = filter_fields,
                fieldname_map = fieldname_map,
                **kwargs
            )
        elif timestep_level:
            return ReplayDatasetTimestep(
                self,
                fields = fields,
                filter_meta = filter_meta,
                filter_fields = filter_fields,
                fieldname_map = fieldname_map,
                **kwargs
            )
        else:
            return ReplayDatasetTrajectory(
                self,
                fields = fields,
                filter_meta = filter_meta,
                filter_fields = filter_fields,
                fieldname_map = fieldname_map,
                slice_by_episode_len = slice_by_episode_len,
                return_episode_lens = return_episode_lens,
                **kwargs
            )

    @beartype
    def dataloader(
        self,
        batch_size,
        n_steps: int | None = None,
        dataset: Dataset | None = None,
        fields: tuple[str, ...] | None = None,
        current_fields: tuple[str, ...] | None = None,
        next_fields: tuple[str, ...] | None = None,
        sequence_fields: tuple[str, ...] | None = None,
        filter_meta: dict | None = None,
        filter_fields: dict | None = None,
        fieldname_map: dict[str, str] | None = None,
        return_indices: bool = False,
        return_mask: bool = False,
        timestep_level: bool = False,
        to_named_tuple: tuple[str, ...] | None = None,
        shuffle = False,
        device: torch.device | str | None = None,
        dataset_kwargs: dict | None = None,
        **kwargs
    ) -> DataLoader:
        dataset_kwargs = default(dataset_kwargs, dict())

        self.flush()

        if len(self) == 0:
            raise ValueError('replay buffer is empty')

        # if to_named_tuple is specified, don't filter dataset fields
        if exists(to_named_tuple) and exists(fields):
            raise ValueError('cannot specify both fields and to_named_tuple')

        if return_mask and (timestep_level or exists(n_steps)):
            raise ValueError('return_mask is only supported for trajectory-level data')

        if exists(n_steps) and timestep_level:
            raise ValueError('cannot specify both n_steps and timestep_level')

        if not exists(dataset):
            dataset = self.dataset(
                n_steps = n_steps,
                fields = fields,
                current_fields = current_fields,
                next_fields = next_fields,
                sequence_fields = sequence_fields,
                timestep_level = timestep_level,
                return_indices = return_indices,
                filter_meta = filter_meta,
                filter_fields = filter_fields,
                fieldname_map = fieldname_map,
                **dataset_kwargs
            )

        # choose appropriate base collation

        if exists(n_steps) or timestep_level:
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
                    if field not in batch:
                        raise ValueError(f'field `{field}` not found in batch. available fields: {list(batch.keys())}')

                batch = NamedTupleCls(**{san: batch[orig] for orig, san in zip(to_named_tuple, sanitized_fields)})

            return tree_map_to_device(batch, device)

        return DataLoader(dataset, batch_size = batch_size, collate_fn = collate_fn, shuffle = shuffle, **kwargs)

    def create_collector(
        self,
        num_groups: int,
        fieldnames: tuple[str, ...] | None = None,
        meta_fieldnames: tuple[str, ...] | None = None
    ):
        from memmap_replay_buffer.episode_collector import EpisodeCollector
        return EpisodeCollector(
            self,
            num_groups,
            fieldnames = fieldnames,
            meta_fieldnames = meta_fieldnames
        )
