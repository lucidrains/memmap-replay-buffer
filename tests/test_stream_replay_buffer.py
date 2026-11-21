""" self-explanatory tests for StreamReplayBuffer and backported online sampling features

covers:
    - async multi-stream writes, in-flight episode visibility, and wrap-around defaults
    - disk persistence, read-only guards, cross-class guards, and in-memory mode
    - strict & padded contiguous windows across wrap-around + `last_fields`
    - transition age tracking (`stream` / `global`), `min_age` / `max_age`, empty & fallback behavior
    - callable predicates (`filter_fields`, `filter_meta`, `where`) and `window_filter_mode`
    - stratified `mixture` sampling and empty-stratum fallback
    - chunked masked `field_stats` (scalar & vector)
    - `dataset` / `dataloader` views on `StreamReplayBuffer`
    - parity on episode-major `ReplayBuffer` & `ReplayBufferH5PY` + legacy `metadata.pkl` compatibility
    - end-to-end SuperMimic / LunarLander online training workflow
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from memmap_replay_buffer import (
    FieldStats,
    ReplayBuffer,
    ReplayBufferH5PY,
    ReplayDatasetWindow,
    StreamReplayBuffer,
)

# multi-stream ingestion & in-flight episodes

def test_stream_async_envs_and_inflight_visibility():
    """ independent streams reset asynchronously and unfinished episodes are immediately samplable """

    buf = StreamReplayBuffer(
        folder = None,
        num_streams = 3,
        capacity = 32,
        fields = dict(
            state = ('float', 2),
            step_num = 'int',
        ),
    )

    # stream 0: one finished 4-step episode + one in-flight 3-step episode

    for t in range(4):
        buf.store_step(0, state = [0., float(t)], step_num = t, end_episode = (t == 3))

    for t in range(3):
        buf.store_step(0, state = [10., float(t)], step_num = 10 + t)

    # stream 1: single long in-flight episode (never ended); stream 2 stays empty

    for t in range(6):
        buf.store_step(1, state = [20., float(t)], step_num = 20 + t)

    assert len(buf) == 13
    assert buf.num_transitions == 13
    assert buf.write_counts.tolist() == [7, 6, 0]

    # strict 3-step windows see finished + in-flight episodes and never cross episode boundaries

    batch = buf.sample(200, horizon = 3, strict_horizon = True)
    step_seqs = batch['step_num']

    for seq in step_seqs.tolist():
        assert seq[1] == seq[0] + 1 and seq[2] == seq[0] + 2

    starts = set(step_seqs[:, 0].tolist())
    assert 10 in starts
    assert any(s >= 20 for s in starts)


def test_stream_batched_store_steps_and_end_episodes():
    """ vectorized `store_steps` writes across all or selected streams even after write pointers diverge """

    buf = StreamReplayBuffer(
        num_streams = 4,
        capacity = 20,
        fields = dict(val = 'float', flag = ('bool', (), False)),
    )

    buf.store_steps(
        val = torch.tensor([1., 2., 3., 4.]),
        flag = torch.tensor([True, False, True, False]),
        end_episodes = torch.tensor([False, True, False, True]),
    )

    # extra step on stream 1 causes pointers to diverge across streams

    buf.store_step(1, val = 99., flag = False)
    assert buf.write_ptrs.tolist() == [1, 2, 1, 1]

    # write a subset of streams using a boolean mask

    mask = np.array([True, False, True, False])
    buf.store_steps(stream_indices = mask, val = np.array([10., 30.]), flag = np.array([True, True]))
    assert buf.write_ptrs.tolist() == [2, 2, 2, 1]

    assert buf.episode_ids[0, 0] == buf.episode_ids[0, 1]
    assert buf.episode_ids[1, 0] != buf.episode_ids[1, 1]

    with pytest.raises(ValueError, match = 'unique'):
        buf.store_steps(stream_indices = [0, 0], val = [1., 2.])


def test_stream_omitted_fields_reset_to_defaults_on_circular_wrap():
    """ circularly overwriting a slot resets omitted fields to their declared defaults """

    buf = StreamReplayBuffer(
        num_streams = 1,
        capacity = 4,
        fields = dict(
            x = 'float',
            is_real = ('bool', (), False),
            tag = ('int', (), -7),
        ),
    )

    for i in range(4):
        buf.store_step(0, x = float(i), is_real = True, tag = 42)

    buf.store_step(0, x = 100.)

    assert buf.data['x'][0, 0] == 100.
    assert bool(buf.data['is_real'][0, 0]) is False
    assert int(buf.data['tag'][0, 0]) == -7


def test_stream_non_circular_capacity_guard_and_store_episode_overflow():
    """ `circular = False` raises when full; `store_episode` wraps cleanly even if episode > capacity """

    fixed = StreamReplayBuffer(
        num_streams = 2,
        capacity = 3,
        fields = dict(v = 'int'),
        circular = False,
    )

    for i in range(3):
        fixed.store_step(0, v = i)

    with pytest.raises(ValueError, match = 'full'):
        fixed.store_step(0, v = 3)

    ring = StreamReplayBuffer(
        num_streams = 1,
        capacity = 5,
        fields = dict(v = 'int'),
        circular = True,
    )
    ring.store_episode(0, v = np.arange(12, dtype = np.int32))

    assert len(ring) == 5
    assert int(ring.write_counts[0]) == 12

    batch = ring.sample(1, horizon = 5, strict_horizon = True)
    assert batch['v'][0].tolist() == [7, 8, 9, 10, 11]


# persistence, read-only, and in-memory mode

def test_stream_persistence_read_only_and_cross_class_guards(tmp_path: Path):
    """ disk-backed `StreamReplayBuffer` rehydrates cleanly, enforces read-only, and rejects wrong loaders """

    folder = tmp_path / 'stream_buf'
    fields = dict(state = ('float', 3), reward = 'float', step_id = 'int64')

    buf = StreamReplayBuffer(folder, num_streams = 2, capacity = 10, fields = fields, age_mode = 'stream')

    for t in range(6):
        buf.store_step(0, state = np.full(3, float(t)), reward = float(t), step_id = t)

    buf.end_episode(0)
    buf.store_step(1, state = np.ones(3), reward = 5., step_id = 100)
    buf.flush()

    rehydrated = StreamReplayBuffer.from_folder(folder, read_only = True)

    assert len(rehydrated) == 7
    assert rehydrated.global_step == 7
    assert rehydrated.write_counts.tolist() == [6, 1]
    assert rehydrated.data['step_id'].dtype == np.int64

    with pytest.raises(ValueError, match = 'read-only'):
        rehydrated.store_step(0, state = np.zeros(3), reward = 0., step_id = 0)

    with pytest.raises(ValueError, match = 'different config'):
        StreamReplayBuffer(folder, num_streams = 4, capacity = 10, fields = fields, overwrite = False)

    with pytest.raises(ValueError, match = 'StreamReplayBuffer'):
        ReplayBuffer.from_folder(folder)

    ep_folder = tmp_path / 'ep_buf'
    ReplayBuffer(ep_folder, max_episodes = 4, max_timesteps = 5, fields = dict(x = 'float'))

    with pytest.raises(ValueError, match = 'not a StreamReplayBuffer'):
        StreamReplayBuffer.from_folder(ep_folder)

    writable = StreamReplayBuffer.from_folder(folder, read_only = False)
    writable.clear()

    assert len(writable) == 0
    assert writable.global_step == 0


def test_in_memory_mode_for_both_buffers():
    """ `folder = None` runs purely in RAM for both `StreamReplayBuffer` and `ReplayBuffer` """

    stream_buf = StreamReplayBuffer(None, num_streams = 2, capacity = 16, fields = dict(x = 'float'))
    stream_buf.store_step(0, x = 3.5)

    assert stream_buf.in_memory and stream_buf.folder is None
    assert stream_buf.sample(4)['x'].tolist() == [3.5] * 4

    ep_buf = ReplayBuffer(None, max_episodes = 4, max_timesteps = 8, fields = dict(x = 'float'), track_write_step = True)
    ep_buf.store_episode(x = torch.tensor([1., 2., 3.]))

    assert ep_buf.in_memory and ep_buf.folder is None
    assert ep_buf.sample(1, horizon = 3)['x'][0].tolist() == [1., 2., 3.]

    with pytest.raises(ValueError, match = 'read-only'):
        StreamReplayBuffer(None, num_streams = 1, capacity = 4, fields = dict(x = 'float'), read_only = True)

    with pytest.raises(ValueError, match = 'read-only'):
        ReplayBuffer(None, max_episodes = 2, max_timesteps = 4, fields = dict(x = 'float'), read_only = True)


# contiguous windows across wrap-around, padding, and `last_fields`

def test_stream_strict_horizon_wraps_around_ring_safely():
    """ windows spanning `capacity - 1 -> 0` are sampled intact without crossing episodes or `write_ptr` """

    buf = StreamReplayBuffer(num_streams = 1, capacity = 10, fields = dict(t = 'int'))

    # ep A -> slots 0..6; ep B -> wraps across 7, 8, 9, 0, 1, 2 (slots 3..6 remain from ep A)

    buf.store_episode(0, t = np.arange(0, 7, dtype = np.int32))
    buf.store_episode(0, t = np.arange(100, 106, dtype = np.int32))

    expected_windows = {
        (3, 4, 5, 6),
        (100, 101, 102, 103),
        (101, 102, 103, 104),
        (102, 103, 104, 105),
    }

    batch = buf.sample(300, horizon = 4, strict_horizon = True)
    assert {tuple(row) for row in batch['t'].tolist()} == expected_windows

    assert buf.sample(4, horizon = 11, strict_horizon = True, allow_empty = True) is None


def test_stream_padded_windows_last_fields_and_namedtuple():
    """ `strict_horizon = False` pads short windows, sets `_lens`/`_mask`, and aligns `last_fields` """

    buf = StreamReplayBuffer(
        num_streams = 1,
        capacity = 16,
        fields = dict(
            state = ('float', 2),
            next_state = ('float', 2),
            unused_heavy = ('float', (4, 4)),
        ),
    )

    for t in range(3):
        buf.store_step(0, state = [float(t), 0.], next_state = [float(t + 1), 0.])
    buf.end_episode(0)

    batch = buf.sample(
        batch_size = 50,
        horizon = 5,
        strict_horizon = False,
        fields = ('state',),
        last_fields = ('next_state',),
        return_mask = True,
        return_indices = True,
        fieldname_map = dict(last_next_state = 'final_next_state'),
        to_named_tuple = True,
    )

    assert batch.state.shape == (50, 5, 2)
    assert batch.final_next_state.shape == (50, 2)
    assert batch.lens.shape == (50,)
    assert batch.mask.shape == (50, 5)

    for i in range(50):
        start_t = int(batch.indices[i, 1])
        expected_len = 3 - start_t
        assert int(batch.lens[i]) == expected_len
        assert torch.allclose(batch.state[i, :expected_len, 0], torch.arange(start_t, 3, dtype = torch.float32))
        assert torch.all(batch.state[i, expected_len:] == 0.)
        assert batch.mask[i].tolist() == [True] * expected_len + [False] * (5 - expected_len)
        assert batch.final_next_state[i, 0].item() == 3.0

    # `to_named_tuple` tuple infers `fields` and `last_fields` automatically

    strict_batch = buf.sample(
        batch_size = 8,
        horizon = 3,
        strict_horizon = True,
        to_named_tuple = ('state', 'last_next_state'),
    )

    assert strict_batch.state.shape == (8, 3, 2)
    assert strict_batch.last_next_state.shape == (8, 2)
    assert torch.all(strict_batch.last_next_state[:, 0] == 3.0)


# age tracking, empty pools, and fallback semantics

def test_stream_age_filtering_stream_and_global_modes():
    """ half-open `[min_age, max_age)` partitions transitions cleanly in both `stream` and `global` modes """

    buf = StreamReplayBuffer(num_streams = 2, capacity = 32, fields = dict(step = 'int'), age_mode = 'stream')

    for i in range(10):
        buf.store_step(0, step = i)

    for i in range(4):
        buf.store_step(1, step = 100 + i)

    # stream age < 3 vs >= 3

    recent_stream = buf.sample(300, max_age = 3, age_mode = 'stream')
    assert set(recent_stream['step'].tolist()) == {7, 8, 9, 101, 102, 103}

    old_stream = buf.sample(300, min_age = 3, age_mode = 'stream')
    assert set(old_stream['step'].tolist()) == {0, 1, 2, 3, 4, 5, 6, 100}

    # global age < 4 vs [4, 6)

    recent_global = buf.sample(200, max_age = 4, age_mode = 'global')
    assert set(recent_global['step'].tolist()) == {100, 101, 102, 103}

    mid_global = buf.sample(200, min_age = 4, max_age = 6, age_mode = 'global')
    assert set(mid_global['step'].tolist()) == {8, 9}


def test_stream_empty_fallback_and_replacement_controls():
    """ checks `allow_empty`, `fallback_on_empty`, and `replacement = True / False` """

    buf = StreamReplayBuffer(num_streams = 1, capacity = 16, fields = dict(x = 'int', is_real = 'bool'))

    assert buf.sample(4, allow_empty = True) is None

    with pytest.raises(ValueError, match = 'empty'):
        buf.sample(4)

    buf.store_step(0, x = 1, is_real = True)
    buf.store_step(0, x = 2, is_real = True)
    buf.store_step(0, x = 3, is_real = False)

    assert buf.sample(4, min_age = 100, allow_empty = True) is None

    with pytest.raises(ValueError, match = 'no valid samples'):
        buf.sample(4, min_age = 100)

    # `fallback_on_empty` relaxes age bounds while still respecting `filter_fields`

    fb = buf.sample(32, filter_fields = dict(is_real = True), min_age = 100, fallback_on_empty = True)
    assert set(fb['x'].tolist()).issubset({1, 2})

    # with vs without replacement

    assert buf.sample(16, filter_fields = dict(is_real = True), replacement = True)['x'].shape == (16,)

    b_no_rep = buf.sample(2, filter_fields = dict(is_real = True), replacement = False)
    assert set(b_no_rep['x'].tolist()) == {1, 2}

    with pytest.raises(ValueError, match = 'without replacement'):
        buf.sample(3, filter_fields = dict(is_real = True), replacement = False)


# predicate filtering & `window_filter_mode`

def test_stream_callable_filter_fields_and_where():
    """ supports exact values, callable scalar/vector predicates, and lazy compound `where` """

    buf = StreamReplayBuffer(
        num_streams = 1,
        capacity = 20,
        fields = dict(
            state = ('float', 2),
            terminal = 'float',
            is_real = 'bool',
        ),
    )

    for i in range(10):
        buf.store_step(
            0,
            state = [float(i), 0.],
            terminal = 1.0 if i in (2, 7) else 0.0,
            is_real = (i != 7),
        )

    b1 = buf.sample(16, filter_fields = dict(is_real = True, terminal = lambda x: x > 0.5))
    assert torch.all(b1['state'][:, 0] == 2.0)

    b2 = buf.sample(64, filter_fields = dict(state = lambda s: s.norm(dim = -1) >= 8.0))
    assert set(b2['state'][:, 0].tolist()) == {8.0, 9.0}

    with pytest.raises(ValueError, match = 'scalar'):
        buf.sample(4, filter_fields = dict(state = 0.))

    b3 = buf.sample(16, where = lambda d: d['is_real'] & (d['terminal'] > 0.5) & (d['_age'] >= 5))
    assert torch.all(b3['state'][:, 0] == 2.0)


def test_stream_window_filter_modes():
    """ verifies `window_filter_mode` in ('start', 'all', 'any') on wrapped strict and padded windows """

    buf = StreamReplayBuffer(num_streams = 1, capacity = 8, fields = dict(t = 'int', good = 'bool'))

    # advance 4 steps so the next 8-step episode wraps around index 7 -> 0

    buf.store_episode(0, t = np.arange(4, dtype = np.int32), good = np.zeros(4, dtype = bool))

    good_flags = [True, True, False, True, True, True, False, True]
    buf.store_episode(0, t = np.arange(8, dtype = np.int32), good = np.array(good_flags))

    b_start = buf.sample(200, horizon = 3, filter_fields = dict(good = True), window_filter_mode = 'start')
    assert set(b_start['t'][:, 0].tolist()) == {0, 1, 3, 4, 5}

    b_all = buf.sample(100, horizon = 3, filter_fields = dict(good = True), window_filter_mode = 'all')
    assert set(b_all['t'][:, 0].tolist()) == {3}

    b_any = buf.sample(200, horizon = 3, filter_fields = dict(t = lambda x: x == 6), window_filter_mode = 'any')
    assert set(b_any['t'][:, 0].tolist()) == {4, 5}

    # padded `all` also includes t = 7 (1-step tail window [True])

    b_pad_all = buf.sample(
        200,
        horizon = 3,
        strict_horizon = False,
        filter_fields = dict(good = True),
        window_filter_mode = 'all',
    )
    assert set(b_pad_all['t'][:, 0].tolist()) == {3, 7}


# stratified mixture sampling

def test_stream_stratified_mixture_and_empty_stratum_fallback():
    """ stratified mixtures draw exact stratum counts and fall back to the base pool when a stratum is empty """

    buf = StreamReplayBuffer(num_streams = 1, capacity = 200, fields = dict(step = 'int', is_real = 'bool'))

    for i in range(100):
        buf.store_step(0, step = i, is_real = True)

    batch = buf.sample(
        batch_size = 256,
        filter_fields = dict(is_real = True),
        mixture = [
            (0.35, dict(max_age = 10)),
            (0.65, dict(min_age = 10)),
        ],
    )

    steps = batch['step']
    assert int((steps >= 90).sum()) == int(256 * 0.35)
    assert int((steps < 90).sum()) == 256 - int(256 * 0.35)

    fb_batch = buf.sample(
        batch_size = 64,
        mixture = [
            (0.5, dict(min_age = 999)),
            (0.5, dict(max_age = 10)),
        ],
    )
    assert fb_batch['step'].shape == (64,)


# chunked masked field statistics

def test_stream_field_stats_scalar_and_vector():
    """ `field_stats` computes exact chunked count/mean/std/rms/min/max over valid filtered slots """

    buf = StreamReplayBuffer(
        num_streams = 2,
        capacity = 16,
        fields = dict(
            reward = 'float',
            obs = ('float', 2),
            is_real = 'bool',
        ),
    )

    empty_scalar = buf.field_stats('reward')
    assert isinstance(empty_scalar, FieldStats)
    assert empty_scalar.count == 0 and np.isnan(empty_scalar.rms)

    empty_vec = buf.field_stats('obs')
    assert empty_vec.count == 0 and torch.isnan(empty_vec.mean).all() and empty_vec.mean.shape == (2,)

    real_rewards = [2.0, -4.0, 6.0, 8.0, -2.0]
    real_obs = [[1., 10.], [2., 20.], [3., 30.], [4., 40.], [5., 50.]]

    for i, (r, o) in enumerate(zip(real_rewards, real_obs, strict = True)):
        buf.store_step(i % 2, reward = r, obs = o, is_real = True)

    for _ in range(10):
        buf.store_step(0, reward = 0.0, obs = [999., 999.], is_real = False)

    r_stats = buf.field_stats('reward', filter_fields = dict(is_real = True), chunk_size = 2)
    arr = np.array(real_rewards, dtype = np.float64)

    assert r_stats.count == 5
    assert r_stats.mean == pytest.approx(arr.mean(), rel = 1e-6)
    assert r_stats.std == pytest.approx(arr.std(), rel = 1e-6)
    assert r_stats.rms == pytest.approx(np.sqrt(np.mean(arr ** 2)), rel = 1e-6)
    assert r_stats.min == pytest.approx(arr.min(), rel = 1e-6)
    assert r_stats.max == pytest.approx(arr.max(), rel = 1e-6)

    o_stats = buf.field_stats('obs', filter_fields = dict(is_real = True), chunk_size = 2)
    o_arr = np.array(real_obs, dtype = np.float64)

    assert o_stats.count == 5
    assert torch.allclose(o_stats.mean, torch.tensor(o_arr.mean(axis = 0), dtype = torch.float32))
    assert torch.allclose(o_stats.std, torch.tensor(o_arr.std(axis = 0), dtype = torch.float32))
    assert torch.allclose(o_stats.rms, torch.tensor(np.sqrt(np.mean(o_arr ** 2, axis = 0)), dtype = torch.float32))

    # wrap stream 0 past capacity=16 -> overwrites slot 0 (real reward 2.0), leaving 4 real entries

    for _ in range(4):
        buf.store_step(0, reward = 0.0, obs = [999., 999.], is_real = False)

    wrapped_stats = buf.field_stats('reward', filter_fields = dict(is_real = True), chunk_size = 3)
    surviving = np.array(real_rewards[1:], dtype = np.float64)

    assert wrapped_stats.count == 4
    assert wrapped_stats.rms == pytest.approx(np.sqrt(np.mean(surviving ** 2)), rel = 1e-6)


# dataset / dataloader on StreamReplayBuffer

def test_stream_dataset_and_dataloader():
    """ `StreamReplayBuffer` supports timestep-level, strict window, and padded window dataloaders """

    buf = StreamReplayBuffer(num_streams = 2, capacity = 16, fields = dict(state = ('float', 2), next_state = ('float', 2)))
    buf.store_episode(0, state = np.zeros((5, 2)), next_state = np.ones((5, 2)))
    buf.store_episode(1, state = np.full((3, 2), 2.), next_state = np.full((3, 2), 3.))

    ts_batches = list(buf.dataloader(batch_size = 4, shuffle = False))
    assert sum(b['state'].shape[0] for b in ts_batches) == 8

    win_batches = list(buf.dataloader(
        batch_size = 2,
        horizon = 3,
        strict_horizon = True,
        last_fields = ('next_state',),
        return_mask = True,
        to_named_tuple = ('state', 'last_next_state', '_mask'),
    ))
    assert len(win_batches) == 2
    assert win_batches[0].state.shape == (2, 3, 2)
    assert win_batches[0].last_next_state.shape == (2, 2)
    assert win_batches[0].mask.all()

    pad_batch = next(iter(buf.dataloader(batch_size = 8, horizon = 4, strict_horizon = False, return_mask = True)))
    assert pad_batch['state'].shape == (8, 4, 2)
    assert pad_batch['_lens'].tolist() == [4, 4, 3, 2, 1, 3, 2, 1]
    assert pad_batch['_mask'].sum(dim = -1).tolist() == [4, 4, 3, 2, 1, 3, 2, 1]


# episode-major ReplayBuffer & ReplayBufferH5PY parity + legacy compatibility

@pytest.mark.parametrize('buffer_class', [ReplayBuffer, ReplayBufferH5PY])
def test_episode_buffer_sample_window_uniformity_predicates_and_stats(buffer_class, tmp_path: Path):
    """ backported `sample`, strict `horizon` windows, callable filters, and `field_stats` on both backends """

    buf = buffer_class(
        tmp_path / 'ep_features',
        max_episodes = 6,
        max_timesteps = 12,
        fields = dict(
            state = ('float', 2),
            next_state = ('float', 2),
            reward = 'float',
            step_id = 'long',
        ),
        meta_fields = dict(
            score = 'float',
            is_expert = 'bool',
        ),
    )

    buf.store_episode(
        state = np.stack([np.arange(3), np.zeros(3)], axis = -1).astype(np.float32),
        next_state = np.stack([np.arange(1, 4), np.zeros(3)], axis = -1).astype(np.float32),
        reward = np.array([1., 2., 3.], dtype = np.float32),
        step_id = np.arange(3, dtype = np.int64),
        score = 6.0,
        is_expert = False,
    )
    buf.store_episode(
        state = np.stack([np.arange(10, 20), np.ones(10)], axis = -1).astype(np.float32),
        next_state = np.stack([np.arange(11, 21), np.ones(10)], axis = -1).astype(np.float32),
        reward = np.arange(10, dtype = np.float32),
        step_id = np.arange(10, 20, dtype = np.int64),
        score = 45.0,
        is_expert = True,
    )

    # window-uniform dataset: ep0 (len 3) -> 1 window, ep1 (len 10) -> 8 windows => 9 total

    ds = buf.dataset(horizon = 3, strict_horizon = True, last_fields = ('next_state',))
    assert isinstance(ds, ReplayDatasetWindow)
    assert len(ds) == 9
    assert ds[0]['state'].shape == (3, 2)
    assert ds[0]['last_next_state'][0].item() == 3.0

    dl_batch = next(iter(buf.dataloader(
        batch_size = 9,
        horizon = 3,
        strict_horizon = True,
        last_fields = ('next_state',),
        return_mask = True,
        to_named_tuple = ('state', 'last_next_state', '_mask'),
    )))
    assert dl_batch.state.shape == (9, 3, 2)
    assert dl_batch.mask.all()

    sampled = buf.sample(
        batch_size = 32,
        horizon = 4,
        strict_horizon = True,
        filter_meta = dict(score = lambda s: s > 20.0),
        filter_fields = dict(reward = lambda r: r >= 2.0),
        window_filter_mode = 'all',
        last_fields = ('next_state',),
        to_named_tuple = True,
    )
    assert sampled.state.shape == (32, 4, 2)
    assert (sampled.reward >= 2.0).all()
    assert torch.allclose(sampled.last_next_state[:, 0], sampled.state[:, -1, 0] + 1.0)

    padded = buf.sample(batch_size = 16, horizon = 6, strict_horizon = False, return_mask = True)
    assert padded['state'].shape == (16, 6, 2)
    assert padded['_mask'].shape == (16, 6)

    stats = buf.field_stats('reward', filter_meta = dict(is_expert = True), chunk_episodes = 1)
    expected = np.arange(10, dtype = np.float64)
    assert stats.count == 10
    assert stats.mean == pytest.approx(expected.mean(), rel = 1e-6)
    assert stats.rms == pytest.approx(np.sqrt(np.mean(expected ** 2)), rel = 1e-6)


@pytest.mark.parametrize('buffer_class', [ReplayBuffer, ReplayBufferH5PY])
def test_episode_buffer_track_write_step_age_and_mixture(buffer_class, tmp_path: Path):
    """ `track_write_step = True` enables exact age filtering & meta/recency mixtures on episode buffers """

    folder = tmp_path / 'tracked'
    buf = buffer_class(
        folder,
        max_episodes = 6,
        max_timesteps = 8,
        fields = dict(step = 'int'),
        meta_fields = dict(is_expert = 'bool'),
        track_write_step = True,
        circular = True,
    )

    with buf.one_episode(is_expert = True):
        for i in range(4):
            buf.store(step = i)

    buf.store_episode(step = np.array([10, 11, 12, 13], dtype = np.int32), is_expert = False)

    with buf.batched_episode(batch_size = 2, is_expert = [False, False]):
        buf.store_batch(step = np.array([20, 30], dtype = np.int32))
        buf.store_batch(step = np.array([21, 31], dtype = np.int32))

    assert buf.global_step == 12
    assert set(buf.sample(64, max_age = 4)['step'].tolist()) == {20, 21, 30, 31}
    assert set(buf.sample(64, min_age = 8)['step'].tolist()) == {0, 1, 2, 3}

    # 50% expert + 50% recent online mixture (RLPD-style)

    mix = buf.sample(
        batch_size = 100,
        mixture = [
            (0.5, dict(filter_meta = dict(is_expert = True))),
            (0.5, dict(max_age = 4)),
        ],
    )
    assert int(mix['is_expert'].sum()) == 50
    assert int((~mix['is_expert']).sum()) == 50

    buf.flush()
    if hasattr(buf, 'file'):
        buf.file.close()

    reopened = buffer_class.from_folder(folder, read_only = True)
    assert reopened.track_write_step is True
    assert reopened.global_step == 12
    assert set(reopened.sample(64, max_age = 4)['step'].tolist()) == {20, 21, 30, 31}


def test_legacy_metadata_backwards_compatibility(tmp_path: Path):
    """ default `ReplayBuffer` keeps legacy `metadata.pkl` untouched and reopens read-only folders cleanly """

    folder = tmp_path / 'legacy'
    buf = ReplayBuffer(folder, max_episodes = 4, max_timesteps = 5, fields = dict(x = 'float'))

    with open(folder / 'metadata.pkl', 'rb') as f:
        meta = pickle.load(f)

    assert 'track_write_step' not in meta
    assert 'buffer_type' not in meta
    assert not (folder / 'write_steps.internal.npy').exists()

    buf.store_episode(x = np.array([1., 2., 3.], dtype = np.float32))
    buf.flush()

    ro = ReplayBuffer.from_folder(folder, read_only = True)
    assert ro.sample(4)['x'].shape == (4,)

    with pytest.raises(ValueError, match = 'track_write_step = True'):
        ro.sample(4, max_age = 10)


# end-to-end SuperMimic / LunarLander online training workflow

def test_supermimic_lunar_workflow_end_to_end():
    """ exercises every buffer operation in `supermimic-pytorch/train_lunar.py` via `StreamReplayBuffer` """

    rng = np.random.default_rng(0)
    num_envs = 16
    capacity = 1_000
    dim_state = 8
    dim_action = 2
    wm_horizon = 8
    critic_horizon = 16
    absorbing_steps = wm_horizon - 1
    critic_max_age = 60
    recent_window = 200
    batch_size = 64
    terminal_batch_size = 32

    buffer = StreamReplayBuffer(
        folder = None,
        num_streams = num_envs,
        capacity = capacity,
        fields = dict(
            state = ('float', dim_state),
            action = ('float', dim_action),
            next_state = ('float', dim_state),
            reward = 'float',
            terminal = 'float',
            is_real = ('bool', (), False),
        ),
    )

    def collect_steps(num_steps: int):
        for _ in range(num_steps):
            for env_idx in range(num_envs):
                state = rng.standard_normal(dim_state).astype(np.float32)
                action = rng.uniform(-1., 1., size = dim_action).astype(np.float32)
                next_state = state + 0.1 * rng.standard_normal(dim_state).astype(np.float32)
                terminated = bool(rng.random() < 0.04)
                truncated = (not terminated) and bool(rng.random() < 0.02)
                reward = -100.0 if terminated else float(rng.standard_normal())

                buffer.store_step(
                    env_idx,
                    state = state,
                    action = action,
                    next_state = next_state,
                    reward = reward,
                    terminal = float(terminated),
                    is_real = True,
                )

                if terminated:
                    for _ in range(absorbing_steps):
                        buffer.store_step(
                            env_idx,
                            state = next_state,
                            action = rng.uniform(-1., 1., size = dim_action).astype(np.float32),
                            next_state = next_state,
                            reward = 0.0,
                            terminal = 1.0,
                            is_real = False,
                        )

                if terminated or truncated:
                    buffer.end_episode(env_idx)

    # early warmup -> no aged terminals yet

    collect_steps(30)

    early_terminals = buffer.sample(
        terminal_batch_size,
        min_age = critic_max_age,
        filter_fields = dict(is_real = True, terminal = lambda x: x > 0.5),
        fields = ('state', 'reward'),
        allow_empty = True,
        to_named_tuple = True,
    )
    assert early_terminals is None

    # advance past `critic_max_age` so recent and aged pools are both populated

    collect_steps(80)

    # 1. reward rms strictly over real transitions

    r_stats = buffer.field_stats('reward', filter_fields = dict(is_real = True))
    reward_scale = max(r_stats.rms, 1e-2) if r_stats.count >= 64 else 1.0
    assert r_stats.count > 0 and reward_scale > 1.0

    # 2. world model sequences: H=8, 35% recent + 65% uniform mixture

    wm_batch = buffer.sample(
        batch_size,
        horizon = wm_horizon,
        strict_horizon = True,
        mixture = [(0.35, dict(max_age = recent_window)), (0.65, dict())],
        fields = ('state', 'action', 'next_state', 'terminal', 'reward'),
        to_named_tuple = True,
    )
    assert wm_batch.state.shape == (batch_size, wm_horizon, dim_state)
    assert wm_batch.action.shape == (batch_size, wm_horizon, dim_action)
    assert wm_batch.next_state.shape == (batch_size, wm_horizon, dim_state)
    assert wm_batch.terminal.shape == (batch_size, wm_horizon)
    assert wm_batch.reward.shape == (batch_size, wm_horizon)

    # 3. critic sequences: H=16, max_age with fallback + (B, H + 1, D) states via `last_fields`

    critic_batch = buffer.sample(
        batch_size,
        horizon = critic_horizon,
        strict_horizon = True,
        max_age = critic_max_age,
        fallback_on_empty = True,
        fields = ('state', 'terminal', 'reward'),
        last_fields = ('next_state',),
        to_named_tuple = True,
    )
    critic_states = torch.cat((critic_batch.state, critic_batch.last_next_state[:, None]), dim = 1)
    assert critic_states.shape == (batch_size, critic_horizon + 1, dim_state)
    assert critic_batch.reward.shape == (batch_size, critic_horizon)
    assert critic_batch.terminal.shape == (batch_size, critic_horizon)

    # 4. older real terminal anchors (never absorbing 0.0 rewards)

    term_batch = buffer.sample(
        terminal_batch_size,
        min_age = critic_max_age,
        filter_fields = dict(is_real = True, terminal = lambda x: x > 0.5),
        fields = ('state', 'reward'),
        allow_empty = True,
        to_named_tuple = True,
    )
    assert term_batch is not None
    assert term_batch.state.shape == (terminal_batch_size, dim_state)
    assert term_batch.reward.shape == (terminal_batch_size,)
    assert torch.all(term_batch.reward == -100.0)

    # 5. actor initial states: real states with 35/65 recency mixture

    init_batch = buffer.sample(
        batch_size,
        filter_fields = dict(is_real = True),
        mixture = [(0.35, dict(max_age = recent_window)), (0.65, dict())],
        fields = ('state', 'is_real'),
        to_named_tuple = True,
    )
    assert init_batch.state.shape == (batch_size, dim_state)
    assert init_batch.is_real.all()
