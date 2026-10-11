from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from beartype import beartype
from beartype.typing import Any
from numpy import ndarray
from torch.utils.data import Dataset

from memmap_replay_buffer.replay_buffer import (
    FieldInfo,
    ReplayBufferBase,
    can_write,
    exists,
)

# h5py dataset proxy

class H5DatasetProxy:
    """ numpy-like view over an h5py dataset """

    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, key):
        return self.dset[key]

    def __setitem__(self, key, value):
        self.dset[key] = value

    def __array__(self, dtype = None):
        arr = self.dset[:]
        return arr if not exists(dtype) else arr.astype(dtype)

    @property
    def shape(self):
        return self.dset.shape

    @property
    def dtype(self):
        return self.dset.dtype

    @property
    def ndim(self):
        return len(self.dset.shape)

# hdf5 backed replay buffer

class ReplayBufferH5PY(ReplayBufferBase):

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
        super().__init__(
            folder,
            max_episodes,
            max_timesteps,
            fields,
            meta_fields = meta_fields,
            circular = circular,
            overwrite = overwrite,
            read_only = read_only,
            flush_every_store_step = flush_every_store_step,
            h5py_compression = h5py_compression,
            h5py_compression_opts = h5py_compression_opts
        )

    def _init_state(self, overwrite, extra_config):
        self.h5py_compression = extra_config.get('h5py_compression')
        self.h5py_compression_opts = extra_config.get('h5py_compression_opts')

        self.h5_path = self.folder / 'data.h5'

        if self.read_only:
            assert self.h5_path.exists(), f'data.h5 not found in `{self.folder}` - cannot open in read-only mode'
            mode = 'r'
        else:
            mode = 'w' if overwrite or not self.h5_path.exists() else 'r+'

        self.file = h5py.File(str(self.h5_path), mode)

        if overwrite:
            for key in list(self.file):
                del self.file[key]

            for key in list(self.file.attrs):
                del self.file.attrs[key]

        if overwrite or 'num_episodes' not in self.file.attrs:
            self.file.attrs['num_episodes'] = 0
            self.file.attrs['episode_index'] = 0
            self.file.attrs['timestep_index'] = 0

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

    def _open_array(self, name, shape, dtype, is_meta, overwrite):
        dset_name = f'{"meta" if is_meta else "data"}_{name}'

        if dset_name not in self.file:
            if exists(self.h5py_compression):
                chunks = (1, *shape[1:]) if is_meta else (1, 1, *shape[2:])
            else:
                chunks = None

            dset = self.file.create_dataset(
                dset_name,
                shape = shape,
                dtype = dtype,
                chunks = chunks,
                compression = self.h5py_compression,
                compression_opts = self.h5py_compression_opts
            )
        else:
            dset = self.file[dset_name]

        return H5DatasetProxy(dset)

    def _fill_default(self, array, indices, default_value):
        # h5py does not broadcast for fancy writes, materialize the selection

        array[indices] = np.full((len(indices), *array.shape[1:]), default_value, dtype = array.dtype)

    def _flush_storage(self):
        self.file.flush()

    def _close_storage(self):
        self.file.close()

    def dataset(self, n_steps: int | None = None, **kwargs) -> Dataset:
        if len(self) == 0:
            raise ValueError('replay buffer is empty')

        if exists(n_steps):
            raise ValueError('n_steps is not supported by ReplayBufferH5PY - use the memmap ReplayBuffer')

        return super().dataset(**kwargs)

    def get_buffered_storer(self, flush_freq: int):
        storage = defaultdict(list)

        def buffered_storer(force_flush = False, **data):
            if self.read_only:
                raise ValueError('cannot write to read-only buffer')

            for key, value in data.items():
                if key not in self.fieldnames and key not in self.meta_fieldnames:
                    raise ValueError(f'field {key} not found in buffer fields')

                storage[key].append(value)

            if not (storage and (force_flush or len(next(iter(storage.values()))) >= flush_freq)):
                return

            # all storage lists must hold the same number of episodes

            batch_size = len(next(iter(storage.values())))

            for key, values in storage.items():
                if len(values) != batch_size:
                    raise ValueError(f'field {key} has a different number of episodes ({len(values)}, expected {batch_size})')

            batch_data = {key: np.stack(values) for key, values in storage.items()}

            self._store_episodes_batch(batch_data)

            storage.clear()
            self.flush()

        return buffered_storer

    @can_write
    def _store_episodes_batch(self, data: dict[str, ndarray]):
        batch_size = next(iter(data.values())).shape[0]

        self._check_room(batch_size)

        indices = np.arange(self.episode_index, self.episode_index + batch_size) % self.max_episodes

        # ascending indices, as h5py requires ordered fancy writes

        order = np.argsort(indices)
        indices = indices[order]

        self._lazy_init_episodes(indices)

        for name, values in data.items():
            if name in self.fieldnames:
                self.data[name][indices] = values[order]
            elif name in self.meta_fieldnames:
                self.meta_data[name][indices] = values[order]

        self.episode_index = (self.episode_index + batch_size) % self.max_episodes
        self.num_episodes += batch_size

        if self.circular:
            self.num_episodes = min(self.num_episodes, self.max_episodes)
