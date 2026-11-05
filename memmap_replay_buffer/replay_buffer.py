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
from beartype.typing import Any, Literal, Sequence
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

DTYPES = dict(int = np.int32, float = np.float32, bool = np.bool_, uint8 = np.uint8)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def to_numpy(value):
    if is_tensor(value):
        return value.detach().cpu().numpy()

    if isinstance(value, (list, tuple)):
        return tensor(value).numpy()

    return np.asarray(value)

def parse_field_info(field_info):
    if isinstance(field_info, str):
        field_info = (field_info, (), None)

    elif isinstance(field_info, tuple) and len(field_info) == 2:
        field_info = (*field_info, None)

    dtype_str, shape, default_value = field_info
    assert dtype_str in DTYPES, f'invalid dtype `{dtype_str}` - must be one of {tuple(DTYPES)}'

    if isinstance(shape, int):
        shape = (shape,)

    return DTYPES[dtype_str], shape, default_value

def cast_to_target_shape(value, target_shape, num_leading_dims = 0):
    # squeeze / unsqueeze a trailing singleton, e.g. target_shape () <-> (1,)

    feature_shape = value.shape[num_leading_dims:]

    if target_shape == () and feature_shape == (1,):
        return value.squeeze(-1)

    if target_shape == (1,) and feature_shape == ():
        return value[..., None]

    return value

def broadcastable_to(shape, target_shape):
    try:
        return np.broadcast_shapes(shape, target_shape) == tuple(target_shape)
    except ValueError:
        return False

def expand_indices(indices, max_episodes = None):
    if not exists(indices):
        return None, False

    if isinstance(indices, slice):
        return np.arange(*indices.indices(max_episodes)), False

    if np.isscalar(indices):
        return np.array([indices]), True

    return np.atleast_1d(np.asarray(indices)), False

def from_numpy(arr: ndarray):
    arr = np.asarray(arr)

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

def pad_right_at_dim_to(t, target, dim = 0):
    return pad_right_at_dim(t, target - t.shape[dim], dim = dim)

def tree_map_to_device(batch, device):
    if not exists(device):
        return batch
    return tree_map(lambda t: t.to(device) if is_tensor(t) else t, batch)

def divisible_by(num, den):
    return (num % den) == 0

def can_write(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        if self.read_only:
            raise ValueError(f'cannot call `{fn.__name__}` in read-only mode')
        return fn(self, *args, **kwargs)
    return inner

def load_or_write_config(folder: Path, config: dict, overwrite: bool):
    config_path = folder / 'metadata.pkl'

    if config_path.exists() and not overwrite:
        with open(str(config_path), 'rb') as f:
            stored_config = pickle.load(f)

        mismatched_keys = [key for key, value in stored_config.items() if config.get(key) != value]

        if len(mismatched_keys) > 0:
            mismatch_lines = [f'  {key}: stored {stored_config[key]!r} vs passed {config.get(key)!r}' for key in mismatched_keys]
            mismatch_str = '\n'.join(mismatch_lines)
            raise ValueError(f'buffer at `{folder}` was created with a different config:\n{mismatch_str}\nPass `overwrite = True` to recreate the buffer with the new config.')

        return

    with open(str(config_path), 'wb') as f:
        pickle.dump(config, f)

# data

def collate_var_time(data, fields_to_pad = None):
    fields_to_pad = default(fields_to_pad, set())

    collated = dict()

    for key in data[0].keys():
        tensors = [d[key] for d in data]

        if key in fields_to_pad and tensors[0].ndim > 0:
            max_time = max(t.shape[0] for t in tensors)
            tensors = [pad_right_at_dim(t, max_time - t.shape[0]) for t in tensors]

        collated[key] = stack(tensors)

    return collated

def episode_filter_mask(replay_buffer: ReplayBufferBase, filter_meta: dict | None = None, min_episode_len = 1):
    episode_lens = from_numpy(replay_buffer.episode_lens)
    valid_mask = episode_lens >= min_episode_len

    for field_name, filter_value in default(filter_meta, {}).items():
        field_data = from_numpy(replay_buffer.meta_data[field_name])

        if isinstance(filter_value, bool):
            field_data = field_data.bool()

        valid_mask &= field_data == filter_value

    return valid_mask

def episode_timestep_grid(episode_indices, episode_lens, horizon = 1):
    # all (episode, timestep) pairs where timestep + horizon <= episode_len

    max_len = int(episode_lens.amax()) if episode_lens.numel() > 0 else 0
    num_timesteps = max(max_len - horizon + 1, 0)

    timesteps = arange(num_timesteps)

    grid = stack(broadcast_tensors(
        rearrange(episode_indices, 'e -> e 1'),
        rearrange(timesteps, 't -> 1 t')
    ), dim = -1)

    valid_mask = rearrange(timesteps, 't -> 1 t') < rearrange(episode_lens - horizon + 1, 'e -> e 1')

    return grid, valid_mask

def filter_fields_mask(data: dict, episode_indices, max_timesteps, filter_fields: dict):
    mask = None

    for field_name, filter_value in filter_fields.items():
        field_data = from_numpy(data[field_name][episode_indices.numpy(), :max_timesteps])

        if isinstance(filter_value, bool):
            field_data = field_data.bool()

        if field_data.ndim != 2:
            raise ValueError(f'filter_fields only supports scalar fields, got shape {tuple(field_data.shape[2:])} for `{field_name}`')

        field_mask = field_data == filter_value
        mask = field_mask if not exists(mask) else (mask & field_mask)

    return mask

class ReplayDatasetBase(Dataset):
    def __init__(
        self,
        replay_buffer: str | Path | ReplayBuffer,
        fields: tuple[str, ...] | None = None,
        fieldname_map: dict[str, str] | None = None,
        include_metadata = True,
        filter_meta: dict | None = None,
        min_episode_len = 1,
        **kwargs
    ):
        if isinstance(replay_buffer, (str, Path)):
            replay_buffer = ReplayBuffer.from_folder(replay_buffer)

        self.replay_buffer = replay_buffer
        self.fields = default(fields, tuple(replay_buffer.fieldnames))
        self.fieldname_map = default(fieldname_map, {})

        self.meta_data = {
            name: memmap for name, memmap in replay_buffer.meta_data.items()
            if name not in replay_buffer.internal_meta_fieldnames
        } if include_metadata else {}

        valid_mask = episode_filter_mask(replay_buffer, filter_meta, min_episode_len)
        self.indices = arange(replay_buffer.max_episodes)[valid_mask]

    def __len__(self):
        return len(self.indices)

    def out_name(self, name):
        return self.fieldname_map.get(name, name)

    def meta_values(self, episode_index):
        return {self.out_name(name): from_numpy(memmap[episode_index].copy()) for name, memmap in self.meta_data.items()}

class ReplayDatasetTrajectory(ReplayDatasetBase):
    def __init__(
        self,
        replay_buffer,
        fields: tuple[str, ...] | None = None,
        fieldname_map: dict[str, str] | None = None,
        include_metadata = True,
        filter_meta: dict | None = None,
        filter_fields: dict | None = None,
        return_indices = False,
        slice_by_episode_len = True,
        return_episode_lens = True,
        **kwargs
    ):
        super().__init__(
            replay_buffer,
            fields = fields,
            fieldname_map = fieldname_map,
            include_metadata = include_metadata,
            filter_meta = filter_meta,
            **kwargs
        )

        if exists(filter_fields):
            raise ValueError('filter_fields is only supported for timestep-level and n-step datasets')

        if not slice_by_episode_len and not return_episode_lens:
            raise ValueError('cannot turn off return_episode_lens if slice_by_episode_len is also turned off')

        self.return_indices = return_indices
        self.slice_by_episode_len = slice_by_episode_len
        self.return_episode_lens = return_episode_lens

    def __getitem__(self, idx):
        episode_index = self.indices[idx].item()
        episode_len = int(self.replay_buffer.episode_lens[episode_index])

        data = dict()

        for field in self.fields:
            if self.slice_by_episode_len:
                arr = self.replay_buffer.data[field][episode_index, :episode_len]
            else:
                arr = self.replay_buffer.data[field][episode_index]

            data[self.out_name(field)] = from_numpy(arr.copy())

        data.update(self.meta_values(episode_index))

        if self.return_episode_lens:
            data['_lens'] = tensor(episode_len)

        if self.return_indices:
            data['_index'] = tensor(episode_index)

        return data

class ReplayDatasetTimeWindow(ReplayDatasetBase):
    def __init__(
        self,
        replay_buffer,
        window_length: int,
        fields: tuple[str, ...] | None = None,
        fieldname_map: dict[str, str] | None = None,
        include_metadata = True,
        filter_meta: dict | None = None,
        return_indices = False,
        **kwargs
    ):
        super().__init__(
            replay_buffer,
            fields = fields,
            fieldname_map = fieldname_map,
            include_metadata = include_metadata,
            filter_meta = filter_meta,
            **kwargs
        )

        self.window_length = window_length
        self.return_indices = return_indices

    def __getitem__(self, idx):
        episode_index = self.indices[idx].item()
        episode_len = int(self.replay_buffer.episode_lens[episode_index])

        actual_len = min(self.window_length, episode_len)
        start = np.random.randint(0, episode_len - actual_len + 1)
        end = start + actual_len

        data = dict()

        for field in self.fields:
            t = from_numpy(self.replay_buffer.data[field][episode_index, start:end].copy())
            data[self.out_name(field)] = pad_right_at_dim_to(t, self.window_length)

        data.update(self.meta_values(episode_index))

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

class ReplayDatasetTimestep(ReplayDatasetBase):
    def __init__(
        self,
        replay_buffer,
        fields: tuple[str, ...] | None = None,
        fieldname_map: dict[str, str] | None = None,
        return_indices = False,
        include_metadata = True,
        filter_meta: dict | None = None,
        filter_fields: dict | None = None,
        **kwargs
    ):
        super().__init__(
            replay_buffer,
            fields = fields,
            fieldname_map = fieldname_map,
            include_metadata = include_metadata,
            filter_meta = filter_meta,
            **kwargs
        )

        self.return_indices = return_indices

        valid_episodes = self.indices
        valid_episode_lens = from_numpy(self.replay_buffer.episode_lens)[valid_episodes]

        episode_timesteps, valid_timesteps = episode_timestep_grid(valid_episodes, valid_episode_lens)

        if exists(filter_fields):
            valid_timesteps &= filter_fields_mask(self.replay_buffer.data, valid_episodes, valid_timesteps.shape[1], filter_fields)

        self.timepoints = episode_timesteps[valid_timesteps]

    def __len__(self):
        return len(self.timepoints)

    def __getitem__(self, idx):
        ep, t = self.timepoints[idx].tolist()

        data = dict()

        for field in self.fields:
            data[self.out_name(field)] = from_numpy(self.replay_buffer.data[field][ep, t].copy())

        data.update(self.meta_values(ep))

        if self.return_indices:
            data['_indices'] = self.timepoints[idx]

        return data

class ReplayDatasetNStep(ReplayDatasetBase):
    def __init__(
        self,
        replay_buffer,
        n_steps: int,
        current_fields: tuple[str, ...] | None = None,
        next_fields: tuple[str, ...] | None = None,
        sequence_fields: tuple[str, ...] | None = None,
        fieldname_map: dict[str, str] | None = None,
        return_indices = False,
        include_metadata = True,
        filter_meta: dict | None = None,
        filter_fields: dict | None = None,
        **kwargs
    ):
        # need at least 2 timesteps for a valid transition

        super().__init__(
            replay_buffer,
            fieldname_map = fieldname_map,
            include_metadata = include_metadata,
            filter_meta = filter_meta,
            min_episode_len = 2,
            **kwargs
        )

        self.n_steps = n_steps
        self.return_indices = return_indices

        self.current_fields = default(current_fields, tuple(replay_buffer.fieldnames))
        self.next_fields = default(next_fields, tuple(replay_buffer.fieldnames))
        self.sequence_fields = default(sequence_fields, ())

        valid_episodes = self.indices
        valid_episode_lens = from_numpy(self.replay_buffer.episode_lens)[valid_episodes]

        episode_timesteps, valid_timesteps = episode_timestep_grid(valid_episodes, valid_episode_lens, horizon = 2)

        if exists(filter_fields):
            valid_timesteps &= filter_fields_mask(self.replay_buffer.data, valid_episodes, valid_timesteps.shape[1], filter_fields)

        self.timepoints = episode_timesteps[valid_timesteps]

    def __len__(self):
        return len(self.timepoints)

    def __getitem__(self, idx):
        ep, t = self.timepoints[idx].tolist()

        episode_len = self.replay_buffer.episode_lens[ep]
        actual_n = min(self.n_steps, episode_len - t - 1)

        data = dict()

        for field in self.current_fields:
            data[self.out_name(field)] = from_numpy(self.replay_buffer.data[field][ep, t].copy())

        for field in self.next_fields:
            name = self.out_name(f'next_{field}')
            data[name] = from_numpy(self.replay_buffer.data[field][ep, t + actual_n].copy())

        for field in self.sequence_fields:
            name = self.out_name(f'seq_{field}')
            seq = from_numpy(self.replay_buffer.data[field][ep, t:t + actual_n].copy())
            data[name] = pad_right_at_dim_to(seq, self.n_steps)

        data[self.out_name('n_step_lens')] = tensor(actual_n, dtype = torch.long)

        data.update(self.meta_values(ep))

        if self.return_indices:
            data['_indices'] = self.timepoints[idx]

        return data

# backwards compatibility alias

ReplayDataset = ReplayDatasetTrajectory

# buffer

class ReplayBufferBase:
    """ shared logic for the memmap and h5py replay buffers - subclasses provide the storage

    `fields` are time-varying (episode, timestep, *shape), `meta_fields` are per-episode (episode, *shape)

    subclasses implement
        `_init_state(overwrite, extra_config)` - open / create the state tracks
        `_open_array(name, shape, dtype, is_meta, overwrite)` - open / create a data track
        `_fill_default(array, indices, default_value)` - write a default value into selected episodes
        `_flush_storage()` / `_close_storage()` - persist / release the backing store
        `num_episodes` / `episode_index` / `timestep_index` - state properties
    """

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
        flush_every_store_step: int = 1,
        **extra_config: Any
    ):
        meta_fields = default(meta_fields, dict())

        assert not (read_only and overwrite), 'cannot overwrite a buffer in read-only mode'

        if flush_every_store_step < 0:
            raise ValueError(f'flush_every_store_step must be >= 0, got {flush_every_store_step}')

        self.folder = Path(folder)
        self.read_only = read_only
        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.circular = circular

        self.store_step = 0
        self.should_flush = flush_every_store_step > 0
        self.flush_every_store_step = flush_every_store_step

        if read_only:
            assert self.folder.is_dir(), f'cannot open folder `{self.folder}` in read-only mode - folder does not exist'
        else:
            self.folder.mkdir(exist_ok = True, parents = True)

        assert self.folder.is_dir()

        config = dict(
            max_episodes = max_episodes,
            max_timesteps = max_timesteps,
            fields = fields,
            meta_fields = meta_fields,
            circular = circular,
            **extra_config
        )

        load_or_write_config(self.folder, config, overwrite)

        self._init_state(overwrite, extra_config)
        self._init_fields(fields, meta_fields, overwrite)

    @classmethod
    def from_folder(cls, folder: str | Path, read_only: bool = False):
        folder = Path(folder)
        config_path = folder / 'metadata.pkl'

        if not config_path.exists():
            raise ValueError(f'metadata.pkl not found in {folder}')

        with open(str(config_path), 'rb') as f:
            config = pickle.load(f)

        return cls(folder, overwrite = False, read_only = read_only, **config)

    def __del__(self):
        with suppress(Exception):
            self.flush()
            self._close_storage()

    # subclass hooks

    def _init_state(self, overwrite, extra_config):
        raise NotImplementedError

    def _open_array(self, name, shape, dtype, is_meta, overwrite):
        raise NotImplementedError

    def _fill_default(self, array, indices, default_value):
        array[indices] = np.full((len(indices), *array.shape[1:]), default_value, dtype = array.dtype)

    def _flush_storage(self):
        raise NotImplementedError

    def _close_storage(self):
        pass

    # internal setup

    def _init_fields(self, fields, meta_fields, overwrite):
        meta_fields = dict(meta_fields)
        meta_fields.setdefault('episode_lens', 'int')
        meta_fields.setdefault('_initted', 'bool')

        self.meta_shapes = dict()
        self.meta_dtypes = dict()
        self.meta_data = dict()
        self.meta_defaults = dict()
        self.meta_fieldnames = set(meta_fields.keys())

        self.shapes = dict()
        self.dtypes = dict()
        self.data = dict()
        self.defaults = dict()
        self.fieldnames = set(fields.keys())

        if not self.fieldnames.isdisjoint(self.meta_fieldnames):
            raise ValueError(f'fields and meta_fields must be disjoint - shared {self.fieldnames & self.meta_fieldnames}')

        for field_name, field_info in meta_fields.items():
            dtype, shape, default_value = parse_field_info(field_info)

            self.meta_data[field_name] = self._open_array(field_name, (self.max_episodes, *shape), dtype, is_meta = True, overwrite = overwrite)
            self.meta_shapes[field_name] = shape
            self.meta_dtypes[field_name] = dtype
            self.meta_defaults[field_name] = default_value

        self.internal_meta_fieldnames = {'episode_lens', '_initted'}

        for field_name, field_info in fields.items():
            dtype, shape, default_value = parse_field_info(field_info)

            self.data[field_name] = self._open_array(field_name, (self.max_episodes, self.max_timesteps, *shape), dtype, is_meta = False, overwrite = overwrite)
            self.shapes[field_name] = shape
            self.dtypes[field_name] = dtype
            self.defaults[field_name] = default_value

        self.memory_namedtuple = namedtuple('Memory', list(fields.keys()))

    def _full_message(self):
        return f'The replay buffer is full ({self.max_episodes} episodes) and is not set to be circular. Please set `circular = True` or clear the buffer.'

    def _check_room(self, batch_size = 1):
        if not self.circular and self.num_episodes + batch_size > self.max_episodes:
            raise ValueError(self._full_message())

    # state

    @property
    def is_new_episode(self):
        return self.timestep_index == 0

    @property
    def episode_lens(self):
        return self.meta_data['episode_lens']

    @property
    def _initted(self):
        return self.meta_data['_initted']

    def __len__(self):
        return int((self.episode_lens[:] > 0).sum())

    def flush(self):
        if self.read_only:
            return

        if self.timestep_index > 0:
            self.episode_lens[self.episode_index] = self.timestep_index

        self._flush_storage()

    @can_write
    def clear(self):
        self.reset_()
        self.flush()

    @can_write
    def reset_(self):
        self.episode_lens[:] = 0
        self._initted[:] = False
        self.num_episodes = 0
        self.episode_index = 0
        self.timestep_index = 0

    @can_write
    def advance_episode(self, batch_size = 1):
        # a fresh episode with nothing stored does not advance

        if self.is_new_episode and batch_size == 1:
            return

        self._check_room(batch_size)

        # ascending indices, as some backends require ordered fancy writes

        indices = np.sort(np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes)

        self.episode_lens[indices] = self.timestep_index

        self.episode_index = (self.episode_index + batch_size) % self.max_episodes
        self.timestep_index = 0
        self.num_episodes += batch_size

        if self.circular:
            self.num_episodes = min(self.num_episodes, self.max_episodes)

        # lazily overwrite the next episodes with defaults once first written to

        next_indices = np.sort(np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes)
        self._initted[next_indices] = False

    @can_write
    def _lazy_init_episodes(self, indices: ndarray):
        # ascending indices, as some backends require ordered fancy indexing

        indices = np.sort(indices)
        is_initted = self._initted[indices]

        if is_initted.all():
            return

        uninit_indices = np.unique(indices[~is_initted])

        for name, array in self.meta_data.items():
            if name in self.internal_meta_fieldnames:
                continue
            self._fill_default(array, uninit_indices, default(self.meta_defaults[name], 0))

        for name, array in self.data.items():
            self._fill_default(array, uninit_indices, default(self.defaults[name], 0))

        self._initted[uninit_indices] = True

    # storing

    @can_write
    @beartype
    def _store_batch(self, data: dict[str, Tensor | ndarray | list | tuple], is_meta = False):
        if len(data) == 0:
            raise ValueError(f'No data provided to {"store_meta_batch" if is_meta else "store_batch"}')

        fieldnames = self.meta_fieldnames if is_meta else self.fieldnames

        if not set(data.keys()).issubset(fieldnames):
            raise ValueError(f'invalid {"meta " if is_meta else ""}field names {set(data.keys()) - fieldnames} - must be a subset of {fieldnames}')

        batch_size = None

        for name, value in data.items():
            value = to_numpy(tensor(value) if isinstance(value, (list, tuple)) else value)
            data[name] = value

            if not exists(batch_size):
                batch_size = value.shape[0]

            if batch_size != value.shape[0]:
                raise ValueError(f'All data in batch must have the same batch size. Field {name} has batch size {value.shape[0]} while previous fields had {batch_size}.')

        if not is_meta and self.timestep_index >= self.max_timesteps:
            raise ValueError(f'You exceeded the `max_timesteps` ({self.max_timesteps}) set on the replay buffer. Please increase it on init.')

        if not self.circular:
            remaining_episodes = self.max_episodes - self.num_episodes

            if remaining_episodes <= 0:
                raise ValueError(self._full_message())

            # silently slice to the remaining capacity

            if remaining_episodes < batch_size:
                data = {name: value[:remaining_episodes] for name, value in data.items()}
                batch_size = remaining_episodes

        # ascending indices, as some backends require ordered fancy writes

        indices = np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes
        order = np.argsort(indices)
        indices = indices[order]

        if self.is_new_episode:
            self._lazy_init_episodes(indices)

        for name, values in data.items():
            if is_meta:
                self.store_batch_meta_datapoint(indices, name, values[order])
            else:
                self.store_batch_datapoint(indices, self.timestep_index, name, values[order])

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

    @can_write
    @contextmanager
    def one_episode(self, **meta_data):
        self._check_room()

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
        self._check_room(batch_size)

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
    def store_datapoint(self, episode_index: int, timestep_index: int, name: str, datapoint: PrimitiveType | Tensor | ndarray):
        if not (0 <= episode_index < self.max_episodes):
            raise ValueError(f'episode_index {episode_index} out of range - must be in [0, {self.max_episodes})')

        if not (0 <= timestep_index < self.max_timesteps):
            raise ValueError(f'timestep_index {timestep_index} out of range - must be in [0, {self.max_timesteps})')

        datapoint = to_numpy(datapoint)

        if name not in self.fieldnames:
            raise ValueError(f'invalid field name {name} - must be one of {self.fieldnames}')

        if datapoint.shape != self.shapes[name]:
            raise ValueError(f'field {name} - invalid shape {datapoint.shape} - shape must be {self.shapes[name]}')

        self.data[name][episode_index, timestep_index] = datapoint

    @can_write
    def store_meta_datapoint(self, episode_index: int, name: str, datapoint: PrimitiveType | Tensor | ndarray):
        if not (0 <= episode_index < self.max_episodes):
            raise ValueError(f'episode_index {episode_index} out of range - must be in [0, {self.max_episodes})')

        datapoint = to_numpy(datapoint)

        if name not in self.meta_fieldnames:
            raise ValueError(f'invalid field name {name} - must be one of {self.meta_fieldnames}')

        if datapoint.shape != self.meta_shapes[name]:
            raise ValueError(f'field {name} - invalid shape {datapoint.shape} - shape must be {self.meta_shapes[name]}')

        self.meta_data[name][episode_index] = datapoint

    @can_write
    def store_batch_datapoint(self, episode_indices: ndarray, timestep_index: int, name: str, datapoints: Tensor | ndarray):
        datapoints = to_numpy(datapoints)

        if name not in self.fieldnames:
            raise ValueError(f'invalid field name {name} - must be one of {self.fieldnames}')

        self.data[name][episode_indices, timestep_index] = datapoints

    @can_write
    def store_batch_meta_datapoint(self, episode_indices: ndarray, name: str, datapoints: Tensor | ndarray):
        datapoints = to_numpy(datapoints)

        if name not in self.meta_fieldnames:
            raise ValueError(f'invalid field name {name} - must be one of {self.meta_fieldnames}')

        self.meta_data[name][episode_indices] = datapoints

    @can_write
    def store(self, **data):
        if self.timestep_index >= self.max_timesteps:
            raise ValueError(f'You exceeded the `max_timesteps` ({self.max_timesteps}) set on the replay buffer. Please increase it on init.')

        if self.is_new_episode:
            self._lazy_init_episodes(np.array([self.episode_index]))

        # filter to only what is declared, filling in defaults for the rest

        store_data = dict()

        for name in self.memory_namedtuple._fields:
            datapoint = data.get(name)

            if not exists(datapoint):
                default_value = self.defaults[name]
                datapoint = default_value if exists(default_value) else np.zeros(self.shapes[name], dtype = self.dtypes[name])

            if np.isscalar(datapoint):
                datapoint = np.full(self.shapes[name], datapoint, dtype = self.dtypes[name])

            store_data[name] = datapoint
            self.store_datapoint(self.episode_index, self.timestep_index, name, datapoint)

        self.timestep_index += 1
        self.store_step += 1

        if self.should_flush and divisible_by(self.store_step, self.flush_every_store_step):
            self.flush()

        return self.memory_namedtuple(**store_data)

    @can_write
    def store_episode(self, **data):
        if not self.is_new_episode:
            warnings.warn(f'timestep index is not 0 ({self.timestep_index}) when calling `store_episode`. This will overwrite the current episode from the beginning.')

        if len(data) == 0:
            raise ValueError('No data provided to `store_episode`')

        self._lazy_init_episodes(np.array([self.episode_index]))

        time_dim = None

        for name, value in data.items():
            value = to_numpy(value)

            if name in self.fieldnames:
                if value.ndim == 0:
                    raise ValueError(f'time-varying field {name} must have a time dimension')

                curr_time_dim = value.shape[0]

                if not exists(time_dim):
                    time_dim = curr_time_dim

                if time_dim != curr_time_dim:
                    raise ValueError(f'all fields must have the same time dimension. field {name} has {curr_time_dim} while previous fields had {time_dim}')

                value = cast_to_target_shape(value, self.shapes[name], num_leading_dims = 1)

                if value.shape[1:] != self.shapes[name]:
                    raise ValueError(f'field {name} - invalid shape {value.shape[1:]} - shape must be {self.shapes[name]}')

                if time_dim > self.max_timesteps:
                    raise ValueError(f'You exceeded the `max_timesteps` ({self.max_timesteps}) set on the replay buffer. Please increase it on init.')

                self.data[name][self.episode_index, :time_dim] = value

            elif name in self.meta_fieldnames:
                value = cast_to_target_shape(value, self.meta_shapes[name], num_leading_dims = 0)

                if value.shape != self.meta_shapes[name]:
                    raise ValueError(f'meta field {name} - invalid shape {value.shape} - shape must be {self.meta_shapes[name]}')

                self.meta_data[name][self.episode_index] = value

            else:
                raise ValueError(f'invalid field name {name} - must be one of {self.fieldnames} or {self.meta_fieldnames}')

        if not exists(time_dim):
            raise ValueError('At least one time-varying field must be provided to store_episode')

        self.timestep_index = time_dim
        self.advance_episode()

    @can_write
    @beartype
    def update(
        self,
        indices: int | list | ndarray | slice | None = None,
        timestep_start: int = 0,
        **data
    ):
        if len(data) == 0:
            raise ValueError('No data provided to `update`')

        if timestep_start < 0:
            raise ValueError(f'timestep_start must be non-negative, got {timestep_start}')

        indices, scalar_index = expand_indices(indices, self.max_episodes)

        if not exists(indices):
            indices = np.where(self.episode_lens[:] > 0)[0]

        # ascending indices, as some backends require ordered fancy writes

        order = np.argsort(indices)
        indices = np.asarray(indices)[order]

        for name, value in data.items():
            is_time_varying = name in self.fieldnames
            is_meta = name in self.meta_fieldnames

            if not (is_time_varying or is_meta):
                raise ValueError(f'invalid field name `{name}`')

            value = to_numpy(value)

            # a scalar is a single timestep, broadcast over the selected episodes

            if value.ndim == 0:
                value = value[None, None]
            elif scalar_index:
                value = value[None]

            if is_time_varying:
                if value.ndim < 2:
                    raise ValueError(f'field {name} - update expects (num_episodes, time, *shape) values, got shape {value.shape}')

                value = cast_to_target_shape(value, self.shapes[name], num_leading_dims = 2)

                if not broadcastable_to(value.shape[2:], self.shapes[name]):
                    raise ValueError(f'field {name} - invalid shape {value.shape[2:]} - shape must be broadcastable to {self.shapes[name]}')

                time_dim = value.shape[1]

                if timestep_start + time_dim > self.max_timesteps:
                    raise ValueError(f'You exceeded the `max_timesteps` ({self.max_timesteps}) set on the replay buffer. Please increase it on init.')

                value = np.broadcast_to(value, (len(indices), time_dim, *self.shapes[name]))
                self.data[name][indices, timestep_start:timestep_start + time_dim] = value[order]

            else:
                value = cast_to_target_shape(value, self.meta_shapes[name], num_leading_dims = 1)

                if not broadcastable_to(value.shape[1:], self.meta_shapes[name]):
                    raise ValueError(f'meta field {name} - invalid shape {value.shape[1:]} - shape must be broadcastable to {self.meta_shapes[name]}')

                value = np.broadcast_to(value, (len(indices), *self.meta_shapes[name]))
                self.meta_data[name][indices] = value[order]

        if self.should_flush:
            self.flush()

    # reading

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
            all_data[name] = from_numpy(self.data[name][:n, :max_len].copy())

        for name in meta_data_fields:
            all_data[name] = from_numpy(self.meta_data[name][:n].copy())

        return tree_map_to_device(all_data, device)

    @beartype
    def dataset(
        self,
        n_steps: int | None = None,
        fields: tuple[str, ...] | None = None,
        current_fields: tuple[str, ...] | None = None,
        next_fields: tuple[str, ...] | None = None,
        sequence_fields: tuple[str, ...] | None = None,
        timestep_level = False,
        filter_meta: dict | None = None,
        filter_fields: dict | None = None,
        fieldname_map: dict[str, str] | None = None,
        slice_by_episode_len = True,
        return_episode_lens = True,
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

        if timestep_level:
            return ReplayDatasetTimestep(
                self,
                fields = fields,
                filter_meta = filter_meta,
                filter_fields = filter_fields,
                fieldname_map = fieldname_map,
                **kwargs
            )

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
        timestep_level = False,
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

        # only data fields are padded to the longest trajectory

        if exists(n_steps) or timestep_level:
            base_collate_fn = None
        else:
            fields_to_pad = self.fieldnames
            if exists(fieldname_map):
                fields_to_pad = {fieldname_map.get(f, f) for f in fields_to_pad}

            base_collate_fn = partial(collate_var_time, fields_to_pad = fields_to_pad)

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

    # video

    def image_fields(self):
        """ all fields that look like images, in field declaration order """
        return [name for name in self.shapes if len(self.shapes[name]) >= 2]

    def default_image_field(self):
        candidates = self.image_fields()

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) == 0:
            raise ValueError('no image-like field found - pass `field` explicitly')

        raise ValueError(f'multiple image-like fields {candidates} - pass `field` explicitly')

    def episode_frames(self, episode_index: int | Sequence[int] = 0, field: str | None = None):
        if not exists(field):
            field = self.default_image_field()

        if field not in self.fieldnames:
            raise ValueError(f'invalid field name `{field}` - must be one of {self.fieldnames}')

        episode_indices = [episode_index] if isinstance(episode_index, (int, np.integer)) else list(episode_index)

        if len(episode_indices) == 0:
            raise ValueError('at least one episode index is required')

        for index in episode_indices:
            if not (0 <= index < self.num_episodes):
                raise ValueError(f'episode_index {index} out of range - buffer has {self.num_episodes} episodes')

        frames = [self.data[field][index, :int(self.episode_lens[index])] for index in episode_indices]

        if len(frames) == 1:
            return frames[0].copy()

        return np.concatenate(frames, axis = 0)

    def save_video(self, path: str | Path, episode_index: int = 0, field: str | None = None, fps: int = 30, **kwargs):
        from memmap_replay_buffer.video import trajectory_to_video

        return trajectory_to_video(self.episode_frames(episode_index, field), path, fps = fps, **kwargs)

    # collector

    def create_collector(
        self,
        num_groups: int,
        fieldnames: tuple[str, ...] | None = None,
        meta_fieldnames: tuple[str, ...] | None = None,
        **collector_kwargs
    ):
        from memmap_replay_buffer.episode_collector import EpisodeCollector
        return EpisodeCollector(
            self,
            num_groups,
            fieldnames = fieldnames,
            meta_fieldnames = meta_fieldnames,
            **collector_kwargs
        )

    create_rollout_collector = create_collector

# memmap backed replay buffer

class ReplayBuffer(ReplayBufferBase):

    def _init_state(self, overwrite, extra_config):
        for name in ('num_episodes', 'episode_index', 'timestep_index'):
            path = self.folder / f'{name}.state.npy'

            if self.read_only:
                assert path.exists(), f'state file `{path}` not found - cannot open in read-only mode'
                mode = 'r'
            else:
                mode = 'w+' if overwrite or not path.exists() else 'r+'

            setattr(self, f'_{name}', open_memmap(str(path), mode = mode, dtype = np.int32, shape = ()))

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

    def _open_array(self, name, shape, dtype, is_meta, overwrite):
        filepath = self.folder / (f'{name}.data.meta.npy' if is_meta else f'{name}.data.npy')

        if self.read_only:
            assert filepath.exists(), f'`{filepath}` not found - cannot open in read-only mode'
            mode = 'r'
        else:
            mode = 'w+' if overwrite or not filepath.exists() else 'r+'

        return open_memmap(str(filepath), mode = mode, dtype = dtype, shape = shape)

    def _fill_default(self, array, indices, default_value):
        # memmaps take a scalar directly, no need to materialize the whole selection

        array[indices] = default_value

    def _flush_storage(self):
        for memmap in (*self.data.values(), *self.meta_data.values()):
            memmap.flush()

        self._num_episodes.flush()
        self._episode_index.flush()
        self._timestep_index.flush()
