from __future__ import annotations

import pickle
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager, suppress
from functools import partial
from pathlib import Path

import einx
import h5py
import numpy as np
import torch
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Any
from numpy import ndarray
from torch import Tensor, arange, is_tensor, tensor
from torch.utils.data import DataLoader, Dataset, default_collate

from memmap_replay_buffer.replay_buffer import (
    FieldInfo,
    PrimitiveType,
    ReplayDatasetTimestep,
    ReplayDatasetTrajectory,
    can_write,
    cast_to_target_shape,
    collate_var_time,
    default,
    divisible_by,
    exists,
    from_numpy,
    tree_map_to_device,
)

# h5py dataset proxy

class H5DatasetProxy:
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, key):
        return self.dset[key]

    def __setitem__(self, key, value):
        self.dset[key] = value

    def __getattr__(self, name):
        return getattr(self.dset, name)

    def __gt__(self, other):
        return self.dset[:] > other

    def __ge__(self, other):
        return self.dset[:] >= other

    def __lt__(self, other):
        return self.dset[:] < other

    def __le__(self, other):
        return self.dset[:] <= other

    def __eq__(self, other):
        return self.dset[:] == other

    def __ne__(self, other):
        return self.dset[:] != other

    @property
    def shape(self):
        return self.dset.shape

    @property
    def dtype(self):
        return self.dset.dtype

    @property
    def ndim(self):
        return self.dset.ndim

    def astype(self, dtype):
        return self.dset[:].astype(dtype)

# main class

class ReplayBufferH5PY:

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
        h5py_compression: str | None = None,
        h5py_compression_opts: int | Any | None = None
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

        self.h5_path = folder / 'data.h5'
        self.config_path = folder / 'metadata.pkl'

        if self.config_path.exists() and not overwrite:
            with open(str(self.config_path), 'rb') as f:
                stored_config = pickle.load(f)

            init_locals = locals()

            mismatched_keys = [key for key, value in stored_config.items() if init_locals[key] != value]

            if len(mismatched_keys) > 0:
                mismatch_lines = [f'  {key}: stored {stored_config[key]!r} vs passed {init_locals[key]!r}' for key in mismatched_keys]
                mismatch_str = '\n'.join(mismatch_lines)
                raise ValueError(f'buffer at `{folder}` was created with a different config:\n{mismatch_str}\nPass `overwrite = True` to recreate the buffer with the new config.')

        if not self.config_path.exists() or overwrite:
            config = dict(
                max_episodes = max_episodes,
                max_timesteps = max_timesteps,
                fields = fields,
                meta_fields = meta_fields,
                circular = circular,
                h5py_compression = h5py_compression,
                h5py_compression_opts = h5py_compression_opts
            )

            with open(str(self.config_path), 'wb') as f:
                pickle.dump(config, f)

        # open hdf5 file

        if read_only:
            assert self.h5_path.exists(), f'data.h5 not found in `{folder}` - cannot open in read-only mode'
            mode = 'r'
        else:
            mode = 'w' if overwrite or not self.h5_path.exists() else 'r+'

        self.file = h5py.File(str(self.h5_path), mode)

        if overwrite:
            for key in list(self.file):
                del self.file[key]

            for key in list(self.file.attrs):
                del self.file.attrs[key]

        # state management

        if overwrite or 'num_episodes' not in self.file.attrs:
            self.file.attrs['num_episodes'] = 0
            self.file.attrs['episode_index'] = 0
            self.file.attrs['timestep_index'] = 0

        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.circular = circular

        # compression settings

        self.h5py_compression = h5py_compression
        self.h5py_compression_opts = h5py_compression_opts

        if 'episode_lens' not in meta_fields:
            meta_fields = meta_fields.copy()
            meta_fields.update(episode_lens = 'int')

        if '_initted' not in meta_fields:
            meta_fields = meta_fields.copy()
            meta_fields.update(_initted = 'bool')

        # create the datasets for meta data tracks

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
            dtype = dict(int = np.int32, float = np.float32, bool = np.bool_, uint8 = np.uint8)[dtype_str]

            if isinstance(shape, int):
                shape = (shape,)

            return dtype, shape, default_value

        for field_name, field_info in meta_fields.items():
            dtype, shape, default_value = parse_field_info(field_info)

            dset_name = f'meta_{field_name}'

            if dset_name not in self.file:
                chunks = (1, *shape) if exists(h5py_compression) else None

                dset = self.file.create_dataset(
                    dset_name,
                    shape = (max_episodes, *shape),
                    dtype = dtype,
                    chunks = chunks,
                    compression = h5py_compression,
                    compression_opts = h5py_compression_opts
                )
            else:
                dset = self.file[dset_name]

            self.meta_data[field_name] = H5DatasetProxy(dset)
            self.meta_shapes[field_name] = shape
            self.meta_dtypes[field_name] = dtype
            self.meta_defaults[field_name] = default_value

        self.internal_meta_fieldnames = {'episode_lens', '_initted'}

        # create the datasets for individual data tracks

        self.shapes = dict()
        self.dtypes = dict()
        self.data = dict()
        self.defaults = dict()
        self.fieldnames = set(fields.keys())

        if not self.fieldnames.isdisjoint(self.meta_fieldnames):
            raise ValueError(f'fields and meta_fields must be disjoint - shared {self.fieldnames & self.meta_fieldnames}')

        for field_name, field_info in fields.items():
            dtype, shape, default_value = parse_field_info(field_info)

            dset_name = f'data_{field_name}'

            if dset_name not in self.file:
                chunks = (1, 1, *shape) if exists(h5py_compression) else None

                dset = self.file.create_dataset(
                    dset_name,
                    shape = (max_episodes, max_timesteps, *shape),
                    dtype = dtype,
                    chunks = chunks,
                    compression = h5py_compression,
                    compression_opts = h5py_compression_opts
                )
            else:
                dset = self.file[dset_name]

            self.data[field_name] = H5DatasetProxy(dset)
            self.shapes[field_name] = shape
            self.dtypes[field_name] = dtype
            self.defaults[field_name] = default_value

        self.memory_namedtuple = namedtuple('Memory', list(fields.keys()))

        self.store_step = 0
        self.should_flush = flush_every_store_step > 0
        self.flush_every_store_step = flush_every_store_step

    def __del__(self):
        with suppress(Exception):
            if self.read_only:
                return self.file.close()

            self.flush()
            self.file.close()

    @property
    def num_episodes(self):
        return self.file.attrs['num_episodes']

    @num_episodes.setter
    def num_episodes(self, value):
        self.file.attrs['num_episodes'] = value

    @property
    def episode_index(self):
        return self.file.attrs['episode_index']

    @episode_index.setter
    def episode_index(self, value):
        self.file.attrs['episode_index'] = value

    @property
    def timestep_index(self):
        return self.file.attrs['timestep_index']

    @timestep_index.setter
    def timestep_index(self, value):
        self.file.attrs['timestep_index'] = value

    @property
    def is_new_episode(self):
        return self.timestep_index == 0

    def __len__(self):
        return (self.episode_lens[:] > 0).sum().item()

    @property
    def episode_lens(self):
        return self.meta_data['episode_lens']

    @property
    def _initted(self):
        return self.meta_data['_initted']

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

        next_indices = np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes
        self._initted[next_indices] = False

    @can_write
    def _lazy_init_episodes(self, indices: ndarray):
        is_initted = self._initted[indices]

        if np.all(is_initted):
            return

        uninit_indices = np.unique(indices[~is_initted])

        # fill meta fields with defaults

        for name, dset in self.meta_data.items():
            if name in self.internal_meta_fieldnames:
                continue
            shape = (len(uninit_indices), *self.meta_shapes[name])
            dtype = self.meta_dtypes[name]
            dset[uninit_indices] = np.full(shape, default(self.meta_defaults[name], 0), dtype=dtype)

        # fill data fields with defaults

        for name, dset in self.data.items():
            shape = (len(uninit_indices), self.max_timesteps, *self.shapes[name])
            dtype = self.dtypes[name]
            dset[uninit_indices] = np.full(shape, default(self.defaults[name], 0), dtype=dtype)

        self._initted[uninit_indices] = True

    @can_write
    def _store_batch(self, data: dict[str, Tensor | ndarray | list | tuple], is_meta = False):
        if len(data) == 0:
            raise ValueError(f'No data provided to {"store_meta_batch" if is_meta else "store_batch"}')

        fieldnames = self.meta_fieldnames if is_meta else self.fieldnames

        if not set(data.keys()).issubset(fieldnames):
            raise ValueError(f'invalid {"meta " if is_meta else ""}field names {set(data.keys()) - fieldnames} - must be a subset of {fieldnames}')

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

        if not is_meta and self.timestep_index >= self.max_timesteps:
            raise ValueError(f'You exceeded the `max_timesteps` ({self.max_timesteps}) set on the replay buffer. Please increase it on init.')

        if not self.circular:
            remaining = self.max_episodes - self.num_episodes

            if remaining <= 0:
                raise ValueError("Buffer full")

            if remaining < batch_size:
                data = {k: v[:remaining] for k, v in data.items()}
                batch_size = remaining

        indices = np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes

        if self.is_new_episode:
            self._lazy_init_episodes(indices)

        for name, values in data.items():
            if is_meta:
                self.store_batch_meta_datapoint(indices, name, values)
            else:
                self.store_batch_datapoint(indices, self.timestep_index, name, values)

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

        self.file.flush()

    @can_write
    @contextmanager
    def one_episode(self, **meta_data):
        if not self.circular and self.num_episodes >= self.max_episodes:
            raise ValueError("Buffer full")

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
            raise ValueError("Buffer full")

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
    def store_datapoint(self, episode_index, timestep_index, name, datapoint):
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
    def store_meta_datapoint(self, episode_index, name, datapoint):
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
    def store_batch_datapoint(self, episode_indices, timestep_index, name, datapoints):
        if is_tensor(datapoints):
            datapoints = datapoints.detach().cpu().numpy()

        if name not in self.fieldnames:
            raise ValueError(f'invalid field name {name} - must be one of {self.fieldnames}')

        self.data[name][episode_indices, timestep_index] = datapoints

    @can_write
    def store_batch_meta_datapoint(self, episode_indices, name, datapoints):
        if is_tensor(datapoints):
            datapoints = datapoints.detach().cpu().numpy()

        if name not in self.meta_fieldnames:
            raise ValueError(f'invalid field name {name} - must be one of {self.meta_fieldnames}')

        self.meta_data[name][episode_indices] = datapoints
    @can_write
    def store(self, **data):
        if self.timestep_index >= self.max_timesteps:
            raise ValueError("Max timesteps exceeded")

        if self.is_new_episode:
            self._lazy_init_episodes(np.array([self.episode_index]))

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
    def update(
        self,
        indices = None,
        **data
    ):
        if len(data) == 0:
            raise ValueError('No data provided to `update`')

        # normalize indices

        if not exists(indices):
            indices = np.where(self.episode_lens[:] > 0)[0]
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

                if value.shape[0] == 1 and len(indices) > 1:
                    value = np.repeat(value, len(indices), axis = 0)

                for i, index in enumerate(indices):
                    self.data[name][index, :time_dim] = value[i]

            elif is_meta:
                target_shape = self.meta_shapes[name]
                value = cast_to_target_shape(value, target_shape, is_time_varying = False)
                self.meta_data[name][indices] = value

        if self.should_flush:
            self.flush()

    def get_all_data(
        self,
        fields = None,
        meta_fields = None,
        device: torch.device | str | None = None
    ):
        self.flush()

        n = self.num_episodes
        if n == 0:
            return dict()

        max_len = self.episode_lens[:n].max()

        all_data = dict()

        data_fields = default(fields, self.fieldnames)

        if not exists(meta_fields):
            meta_data_fields = tuple(f for f in self.meta_fieldnames if f != '_initted')
        else:
            meta_data_fields = meta_fields

        for name in data_fields:
            all_data[name] = from_numpy(self.data[name][:n, :max_len])

        for name in meta_data_fields:
            all_data[name] = from_numpy(self.meta_data[name][:n])

        return tree_map_to_device(all_data, device)

    @beartype
    def dataset(
        self,
        fields = None,
        timestep_level = False,
        filter_meta = None,
        filter_fields = None,
        fieldname_map = None,
        n_steps = None,
        **kwargs
    ) -> Dataset:
        self.flush()

        if len(self) == 0:
            raise ValueError('replay buffer is empty')

        if exists(n_steps):
            raise ValueError('n_steps is not supported by ReplayBufferH5PY - use the memmap ReplayBuffer')

        dataset_klass = ReplayDatasetTimestep if timestep_level else ReplayDatasetTrajectory

        return dataset_klass(
            self,
            fields = fields,
            filter_meta = filter_meta,
            filter_fields = filter_fields,
            fieldname_map = fieldname_map,
            **kwargs
        )

    def get_buffered_storer(self, flush_freq: int):
        storage = defaultdict(list)

        def buffered_storer(force_flush = False, **data):
            if self.read_only:
                raise ValueError('cannot write to read-only buffer')

            for key, value in data.items():
                if key not in self.fieldnames and key not in self.meta_fieldnames:
                    raise ValueError(f"Field {key} not found in buffer fields")

                storage[key].append(value)

            if not (storage and (force_flush or len(next(iter(storage.values()))) >= flush_freq)):
                return

            # validation check for all storage lists having same length
            batch_size = len(next(iter(storage.values())))

            for k, v in storage.items():
                if len(v) != batch_size:
                    raise ValueError(f"Field {k} has different number of episodes in buffer ({len(v)}, expected {batch_size})")

            batch_data = {k: np.stack(v) for k, v in storage.items()}

            self._store_episodes_batch(batch_data)

            storage.clear()
            self.flush()

        return buffered_storer

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

        if return_mask and timestep_level:
            raise ValueError('return_mask is only supported for trajectory-level data')

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

    @can_write
    def _store_episodes_batch(self, data: dict[str, np.ndarray]):
        batch_size = next(iter(data.values())).shape[0]

        if not self.circular and self.num_episodes + batch_size > self.max_episodes:
            raise ValueError(f'The replay buffer is full ({self.max_episodes} episodes) and is not set to be circular. Please set `circular = True` or clear the buffer.')
        indices = np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes

        self._lazy_init_episodes(indices)

        for name, values in data.items():
            if name in self.fieldnames:
                self.data[name][indices] = values
            elif name in self.meta_fieldnames:
                self.meta_data[name][indices] = values

        self.episode_index = (self.episode_index + batch_size) % self.max_episodes
        self.num_episodes += batch_size

        if self.circular:
            self.num_episodes = min(self.num_episodes, self.max_episodes)

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
