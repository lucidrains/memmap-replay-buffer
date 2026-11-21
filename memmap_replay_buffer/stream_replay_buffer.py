from __future__ import annotations

import pickle
from collections import namedtuple
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import einx
import numpy as np
import torch
from beartype import beartype
from numpy import ndarray
from numpy.lib.format import open_memmap
from torch import Tensor, arange, is_tensor, stack, tensor
from torch.utils.data import DataLoader, Dataset, default_collate

from memmap_replay_buffer.replay_buffer import (
    AgeMode,
    FieldInfo,
    FieldStats,
    WindowFilterMode,
    _LazyFieldView,
    _StatsAccumulator,
    age_filter_mask,
    can_write,
    cast_to_target_shape,
    default,
    divisible_by,
    eval_field_predicate,
    exists,
    format_batch,
    from_numpy,
    load_or_write_config,
    parse_field_info,
    resolve_mixture_counts,
    resolve_sample_fields,
    rolling_window_mask,
    select_indices_from_mask,
    to_numpy,
    validate_sample_args,
)

# dataset

class StreamReplayDataset(Dataset):
    """ timestep-level or contiguous window-level dataset view over a StreamReplayBuffer """

    def __init__(
        self,
        replay_buffer: StreamReplayBuffer,
        horizon: int | None = None,
        strict_horizon: bool = True,
        fields: tuple[str, ...] | None = None,
        last_fields: tuple[str, ...] | None = None,
        fieldname_map: dict[str, str] | None = None,
        return_indices: bool = False,
        filter_fields: dict | None = None,
        where: Callable | Tensor | None = None,
        window_filter_mode: WindowFilterMode = 'start',
        min_age: int | None = None,
        max_age: int | None = None,
        age_mode: AgeMode | None = None,
    ):
        if exists(horizon) and horizon < 1:
            raise ValueError(f'horizon must be >= 1, got {horizon}')

        if exists(last_fields) and not exists(horizon):
            raise ValueError('last_fields is only supported when horizon is specified')

        self.replay_buffer = replay_buffer
        self.horizon = horizon
        self.strict_horizon = strict_horizon
        self.fields = default(fields, tuple(replay_buffer.fieldnames))
        self.last_fields = default(last_fields, ())
        self.fieldname_map = default(fieldname_map, dict())
        self.return_indices = return_indices

        for name in self.fields:
            if name not in replay_buffer.fieldnames:
                raise ValueError(f'invalid field `{name}` - must be one of {replay_buffer.fieldnames}')

        for name in self.last_fields:
            if name not in replay_buffer.fieldnames:
                raise ValueError(f'invalid last_field `{name}` - must be one of {replay_buffer.fieldnames}')

        valid_mask = replay_buffer._build_valid_mask(
            horizon = horizon,
            strict_horizon = strict_horizon,
            filter_fields = filter_fields,
            where = where,
            window_filter_mode = window_filter_mode,
            min_age = min_age,
            max_age = max_age,
            age_mode = age_mode,
            fallback_on_empty = False,
        )

        streams, times = valid_mask.nonzero(as_tuple = True)
        self.timepoints = stack((streams, times), dim = -1)

    def out_name(self, name: str) -> str:
        return self.fieldname_map.get(name, name)

    def __len__(self) -> int:
        return len(self.timepoints)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        s, t = self.timepoints[idx].tolist()
        rb = self.replay_buffer
        data = dict()

        if exists(self.horizon):
            h = self.horizon
            cap = rb.capacity
            offsets = np.arange(h, dtype = np.int64)
            time_idx = (t + offsets) % cap

            if self.strict_horizon:
                actual_len = h
                valid_pos = None
            else:
                start_ep = rb._episode_ids[s, t]
                grid_eps = rb._episode_ids[s, time_idx]
                start_stream_age = int((rb._write_counts[s] - 1 - t) % cap)
                step_valid = (grid_eps == start_ep) & (offsets <= start_stream_age) & (offsets < cap)
                valid_pos = np.cumprod(step_valid).astype(bool)
                actual_len = int(valid_pos.sum())

            for name in self.fields:
                arr = np.asarray(rb.data[name][s, time_idx]).copy()
                if exists(valid_pos):
                    arr[~valid_pos] = 0
                data[self.out_name(name)] = from_numpy(arr)

            last_t = (t + actual_len - 1) % cap
            for name in self.last_fields:
                data[self.out_name(f'last_{name}')] = from_numpy(np.asarray(rb.data[name][s, last_t]).copy())

            if not self.strict_horizon:
                data['_lens'] = tensor(actual_len, dtype = torch.long)
        else:
            for name in self.fields:
                data[self.out_name(name)] = from_numpy(np.asarray(rb.data[name][s, t]).copy())

        if self.return_indices:
            data['_indices'] = self.timepoints[idx]

        return data

# buffer

class StreamReplayBuffer:
    """ asynchronous multi-stream circular transition-ring replay buffer

    stores fields in `(num_streams, capacity, *shape)` memmaps (or in-memory when `folder = None`),
    with monotonic `int64` episode IDs and transition ages so parallel environments can step and reset
    independently while keeping in-flight episodes immediately samplable

        `store_step` / `store_steps` - append one transition to one stream or a batch of streams
        `end_episode` / `end_episodes` - advance episode IDs on finished streams
        `store_episode` - write a full trajectory into a stream and end its episode
        `sample` - fast vectorized sampling (timesteps, strict/padded windows, predicates, age, mixtures)
        `field_stats` - chunked valid-masked field statistics (count, mean, std, rms, min, max)
        `dataset` / `dataloader` - PyTorch Dataset / DataLoader views over timesteps or windows
    """

    @beartype
    def __init__(
        self,
        folder: str | Path | None = None,
        num_streams: int = 1,
        capacity: int | None = None,
        fields: dict[str, FieldInfo] | None = None,
        circular: bool = True,
        overwrite: bool = True,
        read_only: bool = False,
        flush_every_store_step: int = 0,
        age_mode: AgeMode = 'stream',
    ):
        if num_streams <= 0:
            raise ValueError(f'num_streams must be positive, got {num_streams}')

        if not exists(capacity) or capacity <= 0:
            raise ValueError(f'capacity must be a positive int, got {capacity}')

        if not exists(fields) or len(fields) == 0:
            raise ValueError('fields must be a non-empty dict')

        self.in_memory = not exists(folder)

        if self.in_memory and read_only:
            raise ValueError('cannot open an in-memory buffer in read-only mode')

        assert not (read_only and overwrite), 'cannot overwrite a buffer in read-only mode'

        if flush_every_store_step < 0:
            raise ValueError(f'flush_every_store_step must be >= 0, got {flush_every_store_step}')

        self.folder = Path(folder) if exists(folder) else None
        self.num_streams = num_streams
        self.capacity = capacity
        self.circular = circular
        self.read_only = read_only
        self.age_mode = age_mode

        self._store_count = 0
        self.should_flush = (not self.in_memory) and flush_every_store_step > 0
        self.flush_every_store_step = flush_every_store_step

        if not self.in_memory:
            if read_only:
                assert self.folder.is_dir(), f'cannot open folder `{self.folder}` in read-only mode - folder does not exist'
            else:
                self.folder.mkdir(exist_ok = True, parents = True)

            assert self.folder.is_dir()

            config = dict(
                buffer_type = 'stream',
                num_streams = num_streams,
                capacity = capacity,
                fields = fields,
                circular = circular,
                age_mode = age_mode,
            )

            load_or_write_config(self.folder, config, overwrite)

        self._init_state(overwrite)
        self._init_fields(fields, overwrite)

    @classmethod
    def from_folder(cls, folder: str | Path, read_only: bool = False) -> StreamReplayBuffer:
        folder = Path(folder)
        config_path = folder / 'metadata.pkl'

        if not config_path.exists():
            raise ValueError(f'metadata.pkl not found in {folder}')

        with open(str(config_path), 'rb') as f:
            config = dict(pickle.load(f))

        if config.pop('buffer_type', None) != 'stream':
            raise ValueError(f'buffer at `{folder}` is not a StreamReplayBuffer - open it with `ReplayBuffer.from_folder`')

        return cls(folder, overwrite = False, read_only = read_only, **config)

    def __del__(self):
        with suppress(Exception):
            self.flush()

    # storage setup

    def _open_array(self, filename: str, shape: tuple[int, ...], dtype, overwrite: bool, fill_value = None):
        if self.in_memory:
            if exists(fill_value):
                return np.full(shape, fill_value, dtype = dtype)
            return np.zeros(shape, dtype = dtype)

        filepath = self.folder / filename
        is_new = overwrite or not filepath.exists()

        if self.read_only:
            assert filepath.exists(), f'`{filepath}` not found - cannot open in read-only mode'
            mode = 'r'
        else:
            mode = 'w+' if is_new else 'r+'

        arr = open_memmap(str(filepath), mode = mode, dtype = dtype, shape = shape)

        if is_new and exists(fill_value) and not self.read_only:
            arr[...] = fill_value

        return arr

    def _init_state(self, overwrite: bool):
        self._write_counts = self._open_array('write_counts.state.npy', (self.num_streams,), np.int64, overwrite, fill_value = 0)

        is_new_ep_ids = self.in_memory or overwrite or not (self.folder / 'current_episode_ids.state.npy').exists()
        self._current_episode_ids = self._open_array('current_episode_ids.state.npy', (self.num_streams,), np.int64, overwrite)
        if is_new_ep_ids and not self.read_only:
            self._current_episode_ids[:] = np.arange(self.num_streams, dtype = np.int64)

        self._next_episode_id = self._open_array('next_episode_id.state.npy', (), np.int64, overwrite, fill_value = self.num_streams)
        self._global_step = self._open_array('global_step.state.npy', (), np.int64, overwrite, fill_value = 0)

        self._episode_ids = self._open_array('episode_ids.internal.npy', (self.num_streams, self.capacity), np.int64, overwrite, fill_value = -1)
        self._write_steps = self._open_array('write_steps.internal.npy', (self.num_streams, self.capacity), np.int64, overwrite, fill_value = -1)

    def _init_fields(self, fields: dict[str, FieldInfo], overwrite: bool):
        self.shapes: dict[str, tuple[int, ...]] = dict()
        self.dtypes: dict[str, Any] = dict()
        self.data: dict[str, ndarray] = dict()
        self.defaults: dict[str, Any] = dict()
        self.fieldnames = set(fields.keys())

        for field_name, field_info in fields.items():
            dtype, shape, default_value = parse_field_info(field_info)

            self.data[field_name] = self._open_array(
                f'{field_name}.data.npy',
                (self.num_streams, self.capacity, *shape),
                dtype,
                overwrite = overwrite,
                fill_value = default(default_value, 0),
            )
            self.shapes[field_name] = shape
            self.dtypes[field_name] = dtype
            self.defaults[field_name] = default_value

        self.memory_namedtuple = namedtuple('Memory', list(fields.keys()))

    # state

    @property
    def num_transitions(self) -> int:
        return int(np.minimum(self._write_counts, self.capacity).sum())

    @property
    def total_writes(self) -> int:
        return int(self._write_counts.sum())

    @property
    def global_step(self) -> int:
        return int(self._global_step.item())

    @property
    def write_counts(self) -> ndarray:
        return self._write_counts

    @property
    def write_ptrs(self) -> ndarray:
        return self._write_counts % self.capacity

    @property
    def episode_ids(self) -> ndarray:
        return self._episode_ids

    @property
    def is_full(self) -> bool:
        return bool((self._write_counts >= self.capacity).all())

    @property
    def active_length(self) -> int:
        return int(min(int(self._write_counts.max()), self.capacity))

    def __len__(self) -> int:
        return self.num_transitions

    def flush(self):
        if self.read_only or self.in_memory:
            return

        for arr in self.data.values():
            arr.flush()

        self._episode_ids.flush()
        self._write_steps.flush()
        self._write_counts.flush()
        self._current_episode_ids.flush()
        self._next_episode_id.flush()
        self._global_step.flush()

    @can_write
    def clear(self):
        self.reset_()
        self.flush()

    @can_write
    def reset_(self):
        self._episode_ids[:] = -1
        self._write_steps[:] = -1
        self._write_counts[:] = 0
        self._current_episode_ids[:] = np.arange(self.num_streams, dtype = np.int64)
        self._next_episode_id[()] = self.num_streams
        self._global_step[()] = 0
        self._store_count = 0

    # writing

    def _validate_stream_index(self, stream_index: int):
        if not (0 <= stream_index < self.num_streams):
            raise ValueError(f'stream_index {stream_index} out of range - must be in [0, {self.num_streams})')

    def _resolve_stream_indices(self, stream_indices) -> ndarray:
        if not exists(stream_indices):
            return np.arange(self.num_streams, dtype = np.int64)

        arr = to_numpy(stream_indices)

        if arr.dtype == np.bool_:
            if arr.ndim != 1 or arr.shape[0] != self.num_streams:
                raise ValueError(f'boolean stream mask must have shape ({self.num_streams},), got {arr.shape}')
            return np.flatnonzero(arr).astype(np.int64)

        streams = np.atleast_1d(arr).astype(np.int64)

        if streams.ndim != 1:
            raise ValueError(f'stream_indices must be 1D, got shape {streams.shape}')

        if streams.size > 0:
            if (streams < 0).any() or (streams >= self.num_streams).any():
                raise ValueError(f'stream_indices out of range [0, {self.num_streams}): {streams}')
            if np.unique(streams).size != streams.size:
                raise ValueError(f'stream_indices must be unique, got duplicates in {streams}')

        return streams

    @can_write
    def end_episode(self, stream_index: int = 0):
        self._validate_stream_index(stream_index)
        next_id = int(self._next_episode_id.item())
        self._current_episode_ids[stream_index] = next_id
        self._next_episode_id[()] = next_id + 1

    @can_write
    def end_episodes(self, stream_indices):
        streams = self._resolve_stream_indices(stream_indices)
        n = streams.size

        if n == 0:
            return

        next_id = int(self._next_episode_id.item())
        self._current_episode_ids[streams] = np.arange(next_id, next_id + n, dtype = np.int64)
        self._next_episode_id[()] = next_id + n

    @can_write
    def store_step(self, stream_index: int = 0, *, end_episode: bool = False, **data):
        self._validate_stream_index(stream_index)

        unknown = set(data.keys()) - self.fieldnames
        if unknown:
            raise ValueError(f'invalid field names {unknown} - must be a subset of {self.fieldnames}')

        if not self.circular and self._write_counts[stream_index] >= self.capacity:
            raise ValueError(f'stream {stream_index} is full (capacity {self.capacity}) and `circular = False`')

        ptr = int(self._write_counts[stream_index] % self.capacity)

        for name in self.memory_namedtuple._fields:
            target_shape = self.shapes[name]
            dtype = self.dtypes[name]
            value = data.get(name)

            if not exists(value):
                self.data[name][stream_index, ptr] = default(self.defaults[name], 0)
                continue

            if np.isscalar(value):
                if target_shape == ():
                    self.data[name][stream_index, ptr] = value
                    continue
                arr = np.full(target_shape, value, dtype = dtype)
            else:
                arr = cast_to_target_shape(to_numpy(value), target_shape, num_leading_dims = 0)
                if arr.shape != target_shape:
                    raise ValueError(f'field `{name}` - invalid shape {arr.shape} - shape must be {target_shape}')

            self.data[name][stream_index, ptr] = arr

        step = int(self._global_step.item())
        self._episode_ids[stream_index, ptr] = self._current_episode_ids[stream_index]
        self._write_steps[stream_index, ptr] = step
        self._global_step[()] = step + 1
        self._write_counts[stream_index] += 1

        if end_episode:
            self.end_episode(stream_index)

        self._store_count += 1
        if self.should_flush and divisible_by(self._store_count, self.flush_every_store_step):
            self.flush()

    @can_write
    def store_steps(self, stream_indices = None, *, end_episodes = None, **data):
        streams = self._resolve_stream_indices(stream_indices)
        num = streams.size

        if num == 0:
            return

        unknown = set(data.keys()) - self.fieldnames
        if unknown:
            raise ValueError(f'invalid field names {unknown} - must be a subset of {self.fieldnames}')

        if not self.circular:
            full_mask = self._write_counts[streams] >= self.capacity
            if full_mask.any():
                raise ValueError(f'streams {streams[full_mask].tolist()} are full (capacity {self.capacity}) and `circular = False`')

        ptrs = self._write_counts[streams] % self.capacity

        for name in self.memory_namedtuple._fields:
            target_shape = self.shapes[name]
            value = data.get(name)

            if not exists(value):
                self.data[name][streams, ptrs] = default(self.defaults[name], 0)
                continue

            arr = to_numpy(value)
            if arr.ndim == 0 or arr.shape[0] != num:
                raise ValueError(f'field `{name}` - expected leading batch size {num}, got shape {arr.shape}')

            arr = cast_to_target_shape(arr, target_shape, num_leading_dims = 1)
            if arr.shape[1:] != target_shape:
                raise ValueError(f'field `{name}` - invalid feature shape {arr.shape[1:]} - expected {target_shape}')

            self.data[name][streams, ptrs] = arr

        start_step = int(self._global_step.item())
        self._episode_ids[streams, ptrs] = self._current_episode_ids[streams]
        self._write_steps[streams, ptrs] = np.arange(start_step, start_step + num, dtype = np.int64)
        self._global_step[()] = start_step + num
        self._write_counts[streams] += 1

        if exists(end_episodes):
            if isinstance(end_episodes, (bool, np.bool_)):
                if end_episodes:
                    self.end_episodes(streams)
            else:
                done_arr = to_numpy(end_episodes).astype(bool)
                if done_arr.shape != (num,):
                    raise ValueError(f'end_episodes must have shape ({num},), got {done_arr.shape}')
                done_streams = streams[done_arr]
                if done_streams.size > 0:
                    self.end_episodes(done_streams)

        self._store_count += 1
        if self.should_flush and divisible_by(self._store_count, self.flush_every_store_step):
            self.flush()

    @can_write
    def store_episode(self, stream_index: int = 0, **data):
        self._validate_stream_index(stream_index)

        if len(data) == 0:
            raise ValueError('No data provided to `store_episode`')

        unknown = set(data.keys()) - self.fieldnames
        if unknown:
            raise ValueError(f'invalid field names {unknown} - must be a subset of {self.fieldnames}')

        parsed: dict[str, ndarray] = dict()
        time_dim = None

        for name, value in data.items():
            arr = to_numpy(value)
            if arr.ndim == 0:
                raise ValueError(f'field `{name}` must have a time dimension')

            if not exists(time_dim):
                time_dim = arr.shape[0]
            elif arr.shape[0] != time_dim:
                raise ValueError(f'all fields must share the same time dimension: `{name}` has {arr.shape[0]} vs {time_dim}')

            arr = cast_to_target_shape(arr, self.shapes[name], num_leading_dims = 1)
            if arr.shape[1:] != self.shapes[name]:
                raise ValueError(f'field `{name}` - invalid feature shape {arr.shape[1:]} - expected {self.shapes[name]}')

            parsed[name] = arr

        if time_dim == 0:
            return

        if not self.circular and int(self._write_counts[stream_index]) + time_dim > self.capacity:
            raise ValueError(f'episode of length {time_dim} exceeds remaining capacity on stream {stream_index} (capacity {self.capacity})')

        start_count = int(self._write_counts[stream_index])
        start_step = int(self._global_step.item())
        ep_id = int(self._current_episode_ids[stream_index])

        # if episode exceeds capacity in circular mode, only the final `capacity` steps survive

        keep = min(time_dim, self.capacity)
        src_start = time_dim - keep
        dst_start = (start_count + src_start) % self.capacity
        first_chunk = min(keep, self.capacity - dst_start)
        second_chunk = keep - first_chunk

        for name in self.memory_namedtuple._fields:
            if name in parsed:
                vals = parsed[name][src_start:]
                self.data[name][stream_index, dst_start:dst_start + first_chunk] = vals[:first_chunk]
                if second_chunk > 0:
                    self.data[name][stream_index, :second_chunk] = vals[first_chunk:]
            else:
                default_val = default(self.defaults[name], 0)
                self.data[name][stream_index, dst_start:dst_start + first_chunk] = default_val
                if second_chunk > 0:
                    self.data[name][stream_index, :second_chunk] = default_val

        step_vals = np.arange(start_step + src_start, start_step + time_dim, dtype = np.int64)
        self._episode_ids[stream_index, dst_start:dst_start + first_chunk] = ep_id
        self._write_steps[stream_index, dst_start:dst_start + first_chunk] = step_vals[:first_chunk]

        if second_chunk > 0:
            self._episode_ids[stream_index, :second_chunk] = ep_id
            self._write_steps[stream_index, :second_chunk] = step_vals[first_chunk:]

        self._write_counts[stream_index] = start_count + time_dim
        self._global_step[()] = start_step + time_dim
        self.end_episode(stream_index)

        self._store_count += 1
        if self.should_flush and divisible_by(self._store_count, self.flush_every_store_step):
            self.flush()

    # masking & filtering

    def _stream_ages_slice(self, start: int, end: int) -> Tensor:
        dtype = torch.int32 if self.capacity < (1 << 30) else torch.int64
        times = arange(start, end, dtype = dtype)
        ptrs = torch.as_tensor(self._write_counts % self.capacity, dtype = dtype)
        return torch.remainder(ptrs[:, None] - 1 - times[None, :], self.capacity)

    def _query_ages_slice(self, start: int, end: int, age_mode: AgeMode | None = None) -> Tensor:
        mode = default(age_mode, self.age_mode)
        if mode == 'stream':
            return self._stream_ages_slice(start, end)
        if mode == 'global':
            return self.global_step - 1 - from_numpy(self._write_steps[:, start:end])
        raise ValueError(f'invalid age_mode `{mode}` - must be "stream" or "global"')

    def _eval_step_filters_slice(
        self,
        start: int,
        end: int,
        filter_fields: dict | None = None,
        where: Callable | Tensor | None = None,
        age_mode: AgeMode | None = None
    ) -> Tensor | None:
        mask = None
        expected_shape = (self.num_streams, end - start)

        if exists(filter_fields):
            for field_name, filter_value in filter_fields.items():
                if field_name not in self.data:
                    raise ValueError(f'invalid field name `{field_name}` - must be one of {self.fieldnames}')

                field_data = from_numpy(self.data[field_name][:, start:end])

                if not callable(filter_value) and field_data.ndim != 2:
                    raise ValueError(f'filter_fields only supports scalar fields, got shape {tuple(field_data.shape[2:])} for `{field_name}`')

                field_mask = eval_field_predicate(field_data, filter_value)

                if field_mask.shape != expected_shape:
                    raise ValueError(f'filter_fields predicate for `{field_name}` must return shape {expected_shape}, got {tuple(field_mask.shape)}')

                mask = field_mask if not exists(mask) else (mask & field_mask)

        if exists(where):
            if is_tensor(where):
                w_mask = where[:, start:end].bool() if where.shape[1] >= end else where.bool()
                if tuple(w_mask.shape) != expected_shape:
                    raise ValueError(f'where mask must have shape {expected_shape}, got {tuple(w_mask.shape)}')
            else:
                getters: dict[str, Callable[[], Tensor]] = {
                    name: (lambda n = name: from_numpy(self.data[n][:, start:end]))
                    for name in self.fieldnames
                }
                getters['_episode_id'] = lambda: from_numpy(self._episode_ids[:, start:end])
                getters['_write_step'] = lambda: from_numpy(self._write_steps[:, start:end])
                getters['_stream_age'] = lambda: self._stream_ages_slice(start, end)
                getters['_global_age'] = lambda: self._query_ages_slice(start, end, age_mode = 'global')
                getters['_age'] = lambda: self._query_ages_slice(start, end, age_mode = age_mode)

                w_mask = where(_LazyFieldView(getters))
                if not is_tensor(w_mask):
                    w_mask = from_numpy(np.asarray(w_mask))
                w_mask = w_mask.bool()

                if tuple(w_mask.shape) != expected_shape:
                    raise ValueError(f'where predicate must return shape {expected_shape}, got {tuple(w_mask.shape)}')

            mask = w_mask if not exists(mask) else (mask & w_mask)

        return mask

    def _apply_window_step_filter(
        self,
        step_mask: Tensor,
        valid_slots: Tensor,
        ep_ids: Tensor,
        stream_age: Tensor,
        active_len: int,
        horizon: int,
        strict_horizon: bool,
        window_filter_mode: WindowFilterMode
    ) -> Tensor:
        if window_filter_mode == 'start' or horizon <= 1:
            return step_mask

        wrapped = active_len == self.capacity

        if strict_horizon:
            return rolling_window_mask(step_mask & valid_slots, horizon = horizon, mode = window_filter_mode, circular = wrapped)

        # exact episode-bounded reduction when strict_horizon = False

        times = arange(active_len, dtype = torch.int64)
        acc = torch.ones_like(valid_slots) if window_filter_mode == 'all' else torch.zeros_like(valid_slots)

        for offset in range(min(horizon, self.capacity)):
            if wrapped:
                t_off = (times + offset) % self.capacity
                in_bounds = torch.ones_like(valid_slots)
            else:
                t_off = (times + offset).clamp(max = active_len - 1)
                in_bounds = (times + offset) < active_len

            in_ep = valid_slots & in_bounds & (ep_ids[:, t_off] == ep_ids) & (stream_age >= offset)
            m_off = step_mask[:, t_off]

            if window_filter_mode == 'all':
                acc &= (~in_ep) | m_off
            else:
                acc |= in_ep & m_off

        return acc

    def _valid_mask_with_context(
        self,
        horizon: int | None = None,
        strict_horizon: bool = True,
        filter_fields: dict | None = None,
        where: Callable | Tensor | None = None,
        window_filter_mode: WindowFilterMode = 'start',
        min_age: int | None = None,
        max_age: int | None = None,
        age_mode: AgeMode | None = None,
        fallback_on_empty: bool = False,
        need_context: bool = False,
    ):
        if window_filter_mode not in ('start', 'all', 'any'):
            raise ValueError(f'window_filter_mode must be one of ("start", "all", "any"), got {window_filter_mode!r}')

        active_len = self.active_length
        if active_len == 0:
            return torch.zeros((self.num_streams, 0), dtype = torch.bool), None, None, None

        ep_ids = from_numpy(self._episode_ids[:, :active_len])
        valid_slots = ep_ids >= 0
        need_stream_age = (
            exists(horizon)
            or ((exists(min_age) or exists(max_age)) and default(age_mode, self.age_mode) == 'stream')
            or need_context
        )
        stream_age = self._stream_ages_slice(0, active_len) if need_stream_age else None

        if exists(horizon):
            if horizon < 1:
                raise ValueError(f'horizon must be >= 1, got {horizon}')

            if strict_horizon:
                if horizon > self.capacity or (active_len < self.capacity and horizon > active_len):
                    return torch.zeros_like(valid_slots), ep_ids, valid_slots, stream_age

                if horizon == 1:
                    valid_mask = valid_slots.clone()
                elif active_len == self.capacity:
                    ep_end = torch.roll(ep_ids, shifts = -(horizon - 1), dims = 1)
                    valid_mask = valid_slots & (ep_ids == ep_end) & (stream_age >= horizon - 1)
                else:
                    n_starts = active_len - horizon + 1
                    valid_mask = torch.zeros_like(valid_slots)
                    valid_mask[:, :n_starts] = (
                        valid_slots[:, :n_starts]
                        & (ep_ids[:, :n_starts] == ep_ids[:, horizon - 1:active_len])
                        & (stream_age[:, :n_starts] >= horizon - 1)
                    )
            else:
                valid_mask = valid_slots.clone()
        else:
            valid_mask = valid_slots.clone()

        step_mask = self._eval_step_filters_slice(
            0,
            active_len,
            filter_fields = filter_fields,
            where = where,
            age_mode = age_mode,
        )

        if exists(step_mask):
            if exists(horizon):
                valid_mask &= self._apply_window_step_filter(
                    step_mask,
                    valid_slots,
                    ep_ids,
                    stream_age,
                    active_len,
                    horizon,
                    strict_horizon,
                    window_filter_mode,
                )
            else:
                valid_mask &= step_mask

        if exists(min_age) or exists(max_age):
            mode = default(age_mode, self.age_mode)
            ages = stream_age if mode == 'stream' else self._query_ages_slice(0, active_len, age_mode = mode)
            aged_mask = valid_mask & age_filter_mask(ages, min_age = min_age, max_age = max_age)

            if not (fallback_on_empty and not aged_mask.any()):
                valid_mask = aged_mask

        return valid_mask, ep_ids, valid_slots, stream_age

    def _build_valid_mask(self, **kwargs) -> Tensor:
        valid_mask, *_ = self._valid_mask_with_context(**kwargs)
        return valid_mask

    # sampling

    def _sample_mixture(
        self,
        mixture: Sequence[tuple[float, dict[str, Any]]],
        batch_size: int,
        base_mask: Tensor,
        ep_ids: Tensor,
        valid_slots: Tensor,
        stream_age: Tensor,
        horizon: int | None,
        strict_horizon: bool,
        window_filter_mode: WindowFilterMode,
        age_mode: AgeMode | None,
        replacement: bool
    ) -> tuple[Tensor, Tensor]:
        allowed_keys = {'min_age', 'max_age', 'age_mode', 'filter_fields', 'where', 'window_filter_mode'}
        counts = resolve_mixture_counts(batch_size, [w for w, _ in mixture])
        active_len = base_mask.shape[1]

        cached_ages: dict[str, Tensor] = dict(stream = stream_age)
        all_streams, all_times = [], []

        for (_, stratum), count in zip(mixture, counts, strict = True):
            if count == 0:
                continue

            unknown = set(stratum.keys()) - allowed_keys
            if unknown:
                raise ValueError(f'unknown mixture stratum keys {unknown} - allowed: {allowed_keys}')

            stratum_mask = base_mask

            s_filter_fields = stratum.get('filter_fields')
            s_where = stratum.get('where')
            s_min_age = stratum.get('min_age')
            s_max_age = stratum.get('max_age')
            s_age_mode = stratum.get('age_mode', age_mode)
            s_win_mode = stratum.get('window_filter_mode', window_filter_mode)

            if exists(s_filter_fields) or exists(s_where):
                step_mask = self._eval_step_filters_slice(
                    0,
                    active_len,
                    filter_fields = s_filter_fields,
                    where = s_where,
                    age_mode = s_age_mode,
                )
                if exists(horizon):
                    step_mask = self._apply_window_step_filter(
                        step_mask,
                        valid_slots,
                        ep_ids,
                        stream_age,
                        active_len,
                        horizon,
                        strict_horizon,
                        s_win_mode,
                    )
                stratum_mask = stratum_mask & step_mask

            if exists(s_min_age) or exists(s_max_age):
                mode_key = default(s_age_mode, self.age_mode)
                if mode_key not in cached_ages:
                    cached_ages[mode_key] = self._query_ages_slice(0, active_len, age_mode = mode_key)
                stratum_mask = stratum_mask & age_filter_mask(cached_ages[mode_key], min_age = s_min_age, max_age = s_max_age)

            if not stratum_mask.any():
                stratum_mask = base_mask

            s_idx, t_idx = select_indices_from_mask(stratum_mask, count, replacement = replacement)
            all_streams.append(s_idx)
            all_times.append(t_idx)

        return torch.cat(all_streams, dim = 0), torch.cat(all_times, dim = 0)

    @beartype
    def sample(
        self,
        batch_size: int,
        horizon: int | None = None,
        *,
        timestep_level: bool | None = None,
        strict_horizon: bool = True,
        fields: tuple[str, ...] | None = None,
        last_fields: tuple[str, ...] | None = None,
        filter_fields: dict | None = None,
        where: Callable | Tensor | None = None,
        window_filter_mode: WindowFilterMode = 'start',
        min_age: int | None = None,
        max_age: int | None = None,
        age_mode: AgeMode | None = None,
        mixture: Sequence[tuple[float, dict[str, Any]]] | None = None,
        replacement: bool = True,
        allow_empty: bool = False,
        fallback_on_empty: bool = False,
        return_indices: bool = False,
        return_mask: bool = False,
        fieldname_map: dict[str, str] | None = None,
        to_named_tuple: bool | tuple[str, ...] | None = None,
        device: torch.device | str | None = None,
    ):
        validate_sample_args(batch_size, horizon, timestep_level, last_fields, return_mask)

        if self.active_length == 0:
            if allow_empty:
                return None
            raise ValueError('replay buffer is empty')

        fieldname_map = default(fieldname_map, dict())

        sample_fields, sample_last_fields, _, return_mask, return_indices = resolve_sample_fields(
            self.fieldnames,
            horizon = horizon,
            fields = fields,
            last_fields = last_fields,
            fieldname_map = fieldname_map,
            to_named_tuple = to_named_tuple,
            return_mask = return_mask,
            return_indices = return_indices,
        )

        base_mask, ep_ids, valid_slots, stream_age = self._valid_mask_with_context(
            horizon = horizon,
            strict_horizon = strict_horizon,
            filter_fields = filter_fields,
            where = where,
            window_filter_mode = window_filter_mode,
            min_age = min_age,
            max_age = max_age,
            age_mode = age_mode,
            fallback_on_empty = fallback_on_empty,
            need_context = exists(mixture),
        )

        if not base_mask.any():
            if allow_empty:
                return None
            raise ValueError('no valid samples matched the query')

        if exists(mixture):
            stream_indices_t, time_indices_t = self._sample_mixture(
                mixture,
                batch_size,
                base_mask,
                ep_ids,
                valid_slots,
                stream_age,
                horizon,
                strict_horizon,
                window_filter_mode,
                age_mode,
                replacement,
            )
        else:
            stream_indices_t, time_indices_t = select_indices_from_mask(base_mask, batch_size, replacement = replacement)

        stream_np = stream_indices_t.numpy()
        time_np = time_indices_t.numpy()

        def out_name(name):
            return fieldname_map.get(name, name)

        batch = dict()

        if exists(horizon):
            offsets = np.arange(horizon, dtype = np.int64)[None, :]
            time_grid = (time_np[:, None] + offsets) % self.capacity
            stream_grid = stream_np[:, None]

            if strict_horizon:
                actual_lens_np = np.full(batch_size, horizon, dtype = np.int64)
                valid_pos = None
            else:
                start_eps = self._episode_ids[stream_np, time_np][:, None]
                grid_eps = self._episode_ids[stream_grid, time_grid]
                start_stream_age = ((self._write_counts[stream_np] - 1 - time_np) % self.capacity)[:, None]
                step_valid = (grid_eps == start_eps) & (offsets <= start_stream_age) & (offsets < self.capacity)
                valid_pos = np.cumprod(step_valid, axis = 1).astype(bool)
                actual_lens_np = valid_pos.sum(axis = 1).astype(np.int64)

            for name in sample_fields:
                gathered = np.asarray(self.data[name][stream_grid, time_grid])
                if exists(valid_pos):
                    gathered = gathered.copy()
                    gathered[~valid_pos] = 0
                batch[out_name(name)] = from_numpy(gathered)

            last_times_np = (time_np + actual_lens_np - 1) % self.capacity

            for name in sample_last_fields:
                gathered = np.asarray(self.data[name][stream_np, last_times_np])
                batch[out_name(f'last_{name}')] = from_numpy(gathered)

            if not strict_horizon:
                batch['_lens'] = from_numpy(actual_lens_np)

            if return_mask:
                batch['_mask'] = torch.ones((batch_size, horizon), dtype = torch.bool) if strict_horizon else from_numpy(valid_pos)
        else:
            for name in sample_fields:
                batch[out_name(name)] = from_numpy(np.asarray(self.data[name][stream_np, time_np]))

        if return_indices:
            batch['_indices'] = stack((stream_indices_t, time_indices_t), dim = -1)

        return format_batch(batch, to_named_tuple = to_named_tuple, device = device)

    # stats

    @beartype
    def field_stats(
        self,
        field: str,
        *,
        filter_fields: dict | None = None,
        where: Callable | Tensor | None = None,
        min_age: int | None = None,
        max_age: int | None = None,
        age_mode: AgeMode | None = None,
        chunk_size: int = 8192,
    ) -> FieldStats:
        if field not in self.fieldnames:
            raise ValueError(f'invalid field `{field}` - must be one of {self.fieldnames}')

        if chunk_size <= 0:
            raise ValueError(f'chunk_size must be positive, got {chunk_size}')

        acc = _StatsAccumulator(self.shapes[field])
        active_len = self.active_length

        if active_len == 0:
            return acc.finalize()

        for start in range(0, active_len, chunk_size):
            end = min(start + chunk_size, active_len)
            valid = from_numpy(self._episode_ids[:, start:end]) >= 0

            step_mask = self._eval_step_filters_slice(
                start,
                end,
                filter_fields = filter_fields,
                where = where,
                age_mode = age_mode,
            )

            if exists(step_mask):
                valid &= step_mask

            if exists(min_age) or exists(max_age):
                ages = self._query_ages_slice(start, end, age_mode = age_mode)
                valid &= age_filter_mask(ages, min_age = min_age, max_age = max_age)

            if not valid.any():
                continue

            chunk_vals = np.asarray(self.data[field][:, start:end])
            acc.update(chunk_vals[valid.numpy()])

        return acc.finalize()

    # dataset / dataloader

    @beartype
    def dataset(
        self,
        horizon: int | None = None,
        strict_horizon: bool = True,
        fields: tuple[str, ...] | None = None,
        last_fields: tuple[str, ...] | None = None,
        filter_fields: dict | None = None,
        where: Callable | Tensor | None = None,
        window_filter_mode: WindowFilterMode = 'start',
        min_age: int | None = None,
        max_age: int | None = None,
        age_mode: AgeMode | None = None,
        fieldname_map: dict[str, str] | None = None,
        return_indices: bool = False,
    ) -> StreamReplayDataset:
        if self.active_length == 0:
            raise ValueError('replay buffer is empty')

        return StreamReplayDataset(
            self,
            horizon = horizon,
            strict_horizon = strict_horizon,
            fields = fields,
            last_fields = last_fields,
            fieldname_map = fieldname_map,
            return_indices = return_indices,
            filter_fields = filter_fields,
            where = where,
            window_filter_mode = window_filter_mode,
            min_age = min_age,
            max_age = max_age,
            age_mode = age_mode,
        )

    @beartype
    def dataloader(
        self,
        batch_size: int,
        horizon: int | None = None,
        strict_horizon: bool = True,
        dataset: Dataset | None = None,
        fields: tuple[str, ...] | None = None,
        last_fields: tuple[str, ...] | None = None,
        filter_fields: dict | None = None,
        where: Callable | Tensor | None = None,
        window_filter_mode: WindowFilterMode = 'start',
        min_age: int | None = None,
        max_age: int | None = None,
        age_mode: AgeMode | None = None,
        fieldname_map: dict[str, str] | None = None,
        return_indices: bool = False,
        return_mask: bool = False,
        to_named_tuple: bool | tuple[str, ...] | None = None,
        shuffle: bool = False,
        device: torch.device | str | None = None,
        **kwargs,
    ) -> DataLoader:
        if self.active_length == 0:
            raise ValueError('replay buffer is empty')

        if isinstance(to_named_tuple, tuple) and exists(fields):
            raise ValueError('cannot specify both fields and to_named_tuple')

        if return_mask and not exists(horizon):
            raise ValueError('return_mask is only supported when horizon is specified')

        if not exists(dataset):
            dataset = self.dataset(
                horizon = horizon,
                strict_horizon = strict_horizon,
                fields = fields,
                last_fields = last_fields,
                filter_fields = filter_fields,
                where = where,
                window_filter_mode = window_filter_mode,
                min_age = min_age,
                max_age = max_age,
                age_mode = age_mode,
                fieldname_map = fieldname_map,
                return_indices = return_indices,
            )

        def collate_fn(data):
            batch = default_collate(data)

            if return_mask:
                if strict_horizon:
                    b = next(iter(batch.values())).shape[0]
                    batch['_mask'] = torch.ones((b, horizon), dtype = torch.bool)
                else:
                    lens = batch['_lens']
                    batch['_mask'] = einx.less('j, i -> i j', arange(horizon, device = lens.device), lens)

            return format_batch(batch, to_named_tuple = to_named_tuple, device = device)

        return DataLoader(dataset, batch_size = batch_size, collate_fn = collate_fn, shuffle = shuffle, **kwargs)
