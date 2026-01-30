from __future__ import annotations
import h5py
import pickle
import warnings
import numpy as np
from pathlib import Path
from collections import namedtuple, defaultdict
from contextlib import contextmanager

from functools import partial
import einx
import beartype
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Literal

import torch
from torch import tensor, from_numpy as torch_from_numpy, stack, is_tensor, Tensor, arange
from torch.utils.data import Dataset, DataLoader, default_collate

from memmap_replay_buffer.replay_buffer import (
    exists,
    default,
    first,
    cast_to_target_shape,
    xnor,
    is_empty,
    divisible_by,
    from_numpy,
    can_write,
    ReplayDatasetTrajectory,
    ReplayDatasetTimestep,
    FieldInfo,
    PrimitiveType,
    collate_var_time,
    tree_map_to_device
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
        meta_fields: dict[str, FieldInfo] = dict(),
        circular = False,
        overwrite = True,
        read_only = False,
        flush_every_store_step: int = 1,
        h5py_compression: str | None = None,
        h5py_compression_opts: int | Any | None = None
    ):
        self.read_only = read_only

        # folder for data

        if not isinstance(folder, Path):
            folder = Path(folder)

        folder.mkdir(exist_ok = True, parents = True)

        self.folder = folder
        assert folder.is_dir()

        self.h5_path = folder / 'data.h5'
        self.config_path = folder / 'metadata.pkl'

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

        mode = 'w' if overwrite or not self.h5_path.exists() else ('r' if read_only else 'r+')
        self.file = h5py.File(str(self.h5_path), mode)

        if overwrite:
            for key in self.file.keys():
                del self.file[key]

            for key in self.file.attrs.keys():
                del self.file.attrs[key]

        # state management

        if overwrite:
            self.file.attrs['num_episodes'] = 0
            self.file.attrs['episode_index'] = 0
            self.file.attrs['timestep_index'] = 0
        elif 'num_episodes' not in self.file.attrs:
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

        # create the datasets for meta data tracks

        self.meta_shapes = dict()
        self.meta_dtypes = dict()
        self.meta_data = dict()
        self.meta_defaults = dict()
        self.meta_defaults = dict()
        self.meta_fieldnames = set(meta_fields.keys())

        def parse_field_info(field_info):
            if isinstance(field_info, str):
                field_info = (field_info, (), None)

            elif isinstance(field_info, tuple) and len(field_info) == 2:
                field_info = (*field_info, None)

            dtype_str, shape, default_value = field_info
            dtype = dict(int = np.int32, float = np.float32, bool = np.bool_)[dtype_str]

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

                if exists(default_value):
                    dset[:] = default_value
                else:
                    dset[:] = 0
            else:
                dset = self.file[dset_name]

            self.meta_data[field_name] = H5DatasetProxy(dset)
            self.meta_shapes[field_name] = shape
            self.meta_dtypes[field_name] = dtype
            self.meta_defaults[field_name] = default_value

        # create the datasets for individual data tracks

        self.shapes = dict()
        self.dtypes = dict()
        self.data = dict()
        self.defaults = dict()
        self.fieldnames = set(fields.keys())

        assert self.fieldnames.isdisjoint(self.meta_fieldnames), f'fields and meta_fields must be disjoint - shared {self.fieldnames & self.meta_fieldnames}'

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

                if exists(default_value):
                    dset[:] = default_value
                else:
                    dset[:] = 0
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

    def __len__(self):
        return (self.episode_lens[:] > 0).sum().item()

    @property
    def episode_lens(self):
        return self.meta_data['episode_lens']

    @can_write
    def clear(self):
        for name, dset in self.data.items():
            default_value = self.defaults[name]
            dset[:] = default_value if exists(default_value) else 0

        for name, dset in self.meta_data.items():
            default_value = self.meta_defaults[name]
            dset[:] = default_value if exists(default_value) else 0

        self.reset_()
        self.flush()

    @can_write
    def reset_(self):
        self.episode_lens[:] = 0
        self.num_episodes = 0
        self.episode_index = 0
        self.timestep_index = 0

    @can_write
    def advance_episode(self, batch_size = 1):
        if self.timestep_index == 0 and batch_size == 1:
            return

        assert self.circular or self.num_episodes + batch_size <= self.max_episodes

        indices = np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes

        self.episode_lens[indices] = self.timestep_index

        self.episode_index = (self.episode_index + batch_size) % self.max_episodes
        self.timestep_index = 0
        self.num_episodes += batch_size

        if self.circular:
            self.num_episodes = min(self.num_episodes, self.max_episodes)

    @can_write
    def _store_batch(self, data: dict[str, Tensor | ndarray | list | tuple], is_meta = False):
        assert len(data) > 0

        fieldnames = self.meta_fieldnames if is_meta else self.fieldnames
        assert set(data.keys()).issubset(fieldnames)

        batch_size = None

        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                value = tensor(value)
                data[key] = value

            curr_batch_size = value.shape[0]

            if not exists(batch_size):
                batch_size = curr_batch_size

            assert batch_size == curr_batch_size

        if not self.circular:
            remaining = self.max_episodes - self.num_episodes

            if remaining <= 0:
                raise ValueError("Buffer full")

            if remaining < batch_size:
                data = {k: v[:remaining] for k, v in data.items()}
                batch_size = remaining

        indices = np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes

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

    @can_write
    def flush(self):
        if self.timestep_index > 0:
            self.episode_lens[self.episode_index] = self.timestep_index

        self.file.flush()

    @can_write
    @contextmanager
    def one_episode(self, **meta_data):
        if not self.circular and self.num_episodes >= self.max_episodes:
            raise ValueError("Buffer full")

        for name, value in meta_data.items():
            self.store_meta_datapoint(self.episode_index, name, value)

        final_meta_data_store = dict()

        yield final_meta_data_store

        for name, value in final_meta_data_store.items():
            self.store_meta_datapoint(self.episode_index, name, value)

        self.flush()
        self.advance_episode()

    @can_write
    @contextmanager
    def batched_episode(self, batch_size, **meta_batch):
        if not self.circular and self.num_episodes + batch_size > self.max_episodes:
            raise ValueError("Buffer full")

        if len(meta_batch) > 0:
            self.store_meta_batch(**meta_batch)

        yield

        self.flush()
        self.advance_episode(batch_size = batch_size)

    @can_write
    def store_datapoint(self, episode_index, timestep_index, name, datapoint):
        if is_tensor(datapoint):
            datapoint = datapoint.detach().cpu().numpy()

        if is_bearable(datapoint, PrimitiveType):
            datapoint = np.array(datapoint)

        self.data[name][episode_index, timestep_index] = datapoint

    @can_write
    def store_meta_datapoint(self, episode_index, name, datapoint):
        if is_tensor(datapoint):
            datapoint = datapoint.detach().cpu().numpy()

        if is_bearable(datapoint, PrimitiveType):
            datapoint = np.array(datapoint)

        self.meta_data[name][episode_index] = datapoint

    @can_write
    def store_batch_datapoint(self, episode_indices, timestep_index, name, datapoints):
        if is_tensor(datapoints):
            datapoints = datapoints.detach().cpu().numpy()

        self.data[name][episode_indices, timestep_index] = datapoints

    @can_write
    def store_batch_meta_datapoint(self, episode_indices, name, datapoints):
        if is_tensor(datapoints):
            datapoints = datapoints.detach().cpu().numpy()

        self.meta_data[name][episode_indices] = datapoints
    @can_write
    def store(self, **data):
        if self.timestep_index >= self.max_timesteps:
            raise ValueError("Max timesteps exceeded")

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
        if self.timestep_index != 0:
            warnings.warn(f'timestep index is not 0 ({self.timestep_index}) when calling `store_episode`. This will overwrite the current episode from the beginning.')

        assert len(data) > 0, 'No data provided to `store_episode`'

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

            assert is_time_varying or is_meta, f'invalid field name {name} - must be one of {self.fieldnames} or {self.meta_fieldnames}'

            if is_time_varying:
                curr_time_dim = value.shape[0]

                if not exists(time_dim):
                    time_dim = curr_time_dim

                assert time_dim == curr_time_dim, f'all fields must have the same time dimension. field {name} has {curr_time_dim} while previous fields had {time_dim}'
                
                # auto-squeeze/unsqueeze logic for shapes () and (1,)
                value = cast_to_target_shape(value, self.shapes[name], is_time_varying = True)

                assert value.shape[1:] == self.shapes[name], f'field {name} - invalid shape {value.shape[1:]} - shape must be {self.shapes[name]}'

                if time_dim > self.max_timesteps:
                    raise ValueError(f'You exceeded the `max_timesteps` ({self.max_timesteps}) set on the replay buffer. Please increase it on init.')

                self.data[name][self.episode_index, :time_dim] = value

            elif is_meta:
                # auto-squeeze/unsqueeze logic for shapes () and (1,)
                target_shape = self.shapes[name] if name in self.shapes else self.meta_shapes[name]
                value = cast_to_target_shape(value, target_shape, is_time_varying = False)

                assert value.shape == self.meta_shapes[name], f'meta field {name} - invalid shape {value.shape} - shape must be {self.meta_shapes[name]}'
                self.meta_data[name][self.episode_index] = value

        assert exists(time_dim), 'At least one time-varying field must be provided to store_episode'

        self.timestep_index = time_dim
        self.advance_episode()

    def get_all_data(self, fields = None, meta_fields = None):
        self.flush()

        n = self.num_episodes
        if n == 0:
            return dict()

        max_len = self.episode_lens[:n].max()

        all_data = dict()

        data_fields = default(fields, self.fieldnames)
        meta_data_fields = default(meta_fields, self.meta_fieldnames)

        for name in data_fields:
            all_data[name] = from_numpy(self.data[name][:n, :max_len])

        for name in meta_data_fields:
            all_data[name] = from_numpy(self.meta_data[name][:n])

        return all_data

    @beartype
    def dataset(
        self,
        fields = None,
        timestep_level = False,
        filter_meta = None,
        filter_fields = None,
        fieldname_map = None,
        **kwargs
    ) -> Dataset:
        self.flush()

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
            assert not self.read_only, 'cannot write to read-only buffer'

            for key, value in data.items():
                assert key in self.fieldnames or key in self.meta_fieldnames, f"Field {key} not found in buffer fields"
                storage[key].append(value)

            if not (storage and (force_flush or len(next(iter(storage.values()))) >= flush_freq)):
                return

            # validation check for all storage lists having same length
            batch_size = len(next(iter(storage.values())))

            for k, v in storage.items():
                assert len(v) == batch_size, f"Field {k} has different number of episodes in buffer ({len(v)}, expected {batch_size})"

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

        assert self.circular or self.num_episodes + batch_size <= self.max_episodes, "Buffer full"
        indices = np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes

        for name, values in data.items():
            if name in self.fieldnames:
                self.data[name][indices] = values
            elif name in self.meta_fieldnames:
                self.meta_data[name][indices] = values

        self.episode_index = (self.episode_index + batch_size) % self.max_episodes
        self.num_episodes += batch_size

        if self.circular:
            self.num_episodes = min(self.num_episodes, self.max_episodes)

    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()

    @classmethod
    def from_folder(cls, folder: str | Path, read_only: bool = False):
        if isinstance(folder, str):
            folder = Path(folder)

        config_path = folder / 'metadata.pkl'

        with open(str(config_path), 'rb') as f:
            config = pickle.load(f)

        return cls(folder = folder, overwrite = False, read_only = read_only, **config)
