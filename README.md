## memmap-replay-buffer

An easy-to-use numpy memmap replay buffer for RL and other sequence-based learning tasks.

## Install

```bash
$ pip install memmap-replay-buffer
```

## Usage

Supports trajectory-level, timestep-level, and n-step transition dataloading from a single stored buffer.

```python
import torch
from memmap_replay_buffer import ReplayBuffer

# initialize buffer

buffer = ReplayBuffer(
    './replay_data',
    max_episodes = 1000,
    max_timesteps = 500,
    fields = dict(
        state = ('float', (3, 16, 16), 0.),    # type, shape, and optional default value
        action = ('int', 2),
        reward = 'float'                       # default shape is ()
    ),
    meta_fields = dict(
        task_id = 'int'
    ),
    circular = True,
    overwrite = True
)

# store 4 episodes

for _ in range(4):
    with buffer.one_episode(task_id = 1):
        for _ in range(100):
            buffer.store(
                state = torch.randn(3, 16, 16),
                action = torch.randint(0, 4, (2,)).numpy(),
                reward = 1.0
            )

# rehydrate from disk

buffer_rehydrated = ReplayBuffer.from_folder('./replay_data')
assert buffer_rehydrated.num_episodes == 4
```

### Trajectory-level

Variable-length trajectories, automatically padded with mask and lengths.

```python
dataloader = buffer.dataloader(
    batch_size = 2,
    return_mask = True,
    to_named_tuple = ('state', 'action', 'reward', 'task_id', '_mask', '_lens')
)

for state, action, reward, task_id, mask, lens in dataloader:
    assert state.shape   == (2, 100, 3, 16, 16)
    assert action.shape  == (2, 100, 2)
    assert reward.shape  == (2, 100)
    assert task_id.shape == (2,)

    assert lens.shape    == (2,)
    assert mask.shape    == (2, 100)
```

### Timestep-level

Individual timesteps across episodes, with optional `filter_meta` for conditioning.

```python
dataloader = buffer.dataloader(
    batch_size = 8,
    filter_meta = dict(
        task_id = 1
    ),
    to_named_tuple = ('state', 'action', 'task_id'),
    timestep_level = True,
    drop_last = True
)

for state, action, task_id in dataloader:
    assert state.shape   == (8, 3, 16, 16)
    assert action.shape  == (8, 2)
    assert task_id.shape == (8,)
```

### N-step transitions

Fetches `current_fields` at $t$, `next_fields` at $t + n$ (prefixed `next_`), and `sequence_fields` from $t$ to $t + n$ (prefixed `seq_`, zero-padded at episode boundaries). Use `fieldname_map` to remap to your model's kwargs.

```python
dataloader = buffer.dataloader(
    batch_size = 4,
    n_steps = 5,
    current_fields = ('state',),
    next_fields = ('state',),
    sequence_fields = ('action', 'reward'),
    to_named_tuple = ('state', 'next_state', 'action_chunk', 'rewards', 'n_step_lens'),
    fieldname_map = {
        'seq_action': 'action_chunk',
        'seq_reward': 'rewards'
    }
)

for state, next_state, action_chunk, rewards, n_step_lens in dataloader:
    assert state.shape == (4, 3, 16, 16)
    assert next_state.shape == (4, 3, 16, 16)
    assert action_chunk.shape == (4, 5, 2)
    assert rewards.shape == (4, 5)
    assert n_step_lens.shape == (4,)
```

### Storing whole episodes

`store_episode` takes one tensor per field, all sharing the same time dimension (meta fields are scalars or per-episode shapes).

```python
buffer.store_episode(
    state = torch.randn(100, 3, 16, 16),
    action = torch.randint(0, 4, (100, 2)),
    reward = torch.randn(100),
    task_id = 1
)
```

### Batched parallel collection

When collecting from multiple parallel environments, `batched_episode` + `store_batch` keeps every environment at the same timestep. `create_collector` additionally accumulates group-batched data and stores finished episodes for you.

```python
with buffer.batched_episode(batch_size = 4, task_id = [0, 1, 2, 3]):
    for t in range(100):
        buffer.store_batch(
            state = states,     # (4, 3, 16, 16)
            action = actions    # (4, 2)
        )
```

### Updating data in place

`update` overwrites already-stored episodes, e.g. for bootstrapped returns or value targets.

```python
buffer.update(returns = torch.randn(3, 100))            # all populated episodes
buffer.update(episode_ids, returns = returns_for_ids)   # specific episodes
buffer.update(0, returns = returns_for_one)             # scalar index
```

### Computed fields and selective timestep training

Derived fields must be declared in `fields` when the buffer is created, then written after collection. Use `timestep_start` to write contiguous time chunks without loading a whole episode into memory. A common self-play recipe is to compute per-timestep advantages, threshold them into a boolean field, and then let the timestep-level dataset fetch only the selected timesteps.

```python
buffer = ReplayBuffer(
    './replay_data',
    max_episodes = 1000,
    max_timesteps = 500,
    fields = dict(
        state = ('float', (8,)),
        reward = 'float',
        advantage = 'float',
        high_advantage = ('bool', (), False),   # declares the derived mask
    )
)

# compute advantages in episode batches
buffer.update(episode_ids, advantage = advantages)

# or write contiguous time chunks for all episodes
for start in range(0, max_len, 128):
    buffer.update(
        timestep_start = start,
        advantage = advantage_chunk
    )

# set the mask from a comparison of the computed field
buffer.update(high_advantage = advantage > threshold)

# only selected timesteps are returned
dataloader = buffer.dataloader(
    batch_size = 64,
    timestep_level = True,
    filter_fields = dict(high_advantage = True)
)
```

`demo_advantage_filtering.py` contains a complete self-play loop that computes advantages in batches, marks extreme positive advantages, and trains only on those timesteps.

### Asynchronous multi-stream ring buffer & online sampling

For vectorized off-policy or model-based RL where environments reset asynchronously and in-flight episodes must be sampled immediately, `StreamReplayBuffer` stores transitions in a `(num_streams, capacity)` circular ring (either disk-backed memmaps or in-memory with `folder = None`) with automatic monotonic `episode_id` and transition-age tracking.

Both `StreamReplayBuffer` and `ReplayBuffer` support direct vectorized `sample(...)` (without `DataLoader` rebuild overhead), strict window-uniform contiguous sequences (`horizon`, `last_fields`), callable predicates in `filter_fields` / `filter_meta` / `where`, age bounds (`min_age` / `max_age`), stratified `mixture` sampling, and chunked masked `field_stats(...)`.

```python
from memmap_replay_buffer import StreamReplayBuffer

buffer = StreamReplayBuffer(
    folder = './stream_data',   # or None for pure in-memory
    num_streams = 16,
    capacity = 48_000,
    fields = dict(
        state = ('float', 8),
        action = ('float', 2),
        next_state = ('float', 8),
        reward = 'float',
        terminal = 'float',
        is_real = ('bool', (), False),
    )
)

# asynchronous per-env (or batched `store_steps`) writes with live in-flight episodes
buffer.store_step(env_index, state = s, action = a, next_state = s_next, reward = r, terminal = float(terminated), is_real = True)
if terminated or truncated:
    buffer.end_episode(env_index)

# strict unpadded horizon=8 windows, 35% recent + 65% uniform mixture
wm_batch = buffer.sample(
    batch_size = 256,
    horizon = 8,
    mixture = [(0.35, dict(max_age = 4_000)), (0.65, dict())],
    fields = ('state', 'action', 'next_state', 'terminal', 'reward'),
    to_named_tuple = True,
    device = 'cuda'
)

# near-on-policy critic sequences with fallback + final next_state for (B, H + 1, D)
critic_batch = buffer.sample(
    batch_size = 256,
    horizon = 16,
    max_age = 600,
    fallback_on_empty = True,
    fields = ('state', 'terminal', 'reward'),
    last_fields = ('next_state',),
    to_named_tuple = True,
    device = 'cuda'
)

# older real terminal anchors (returns None gracefully if none exist yet)
term_batch = buffer.sample(
    batch_size = 64,
    min_age = 600,
    filter_fields = dict(is_real = True, terminal = lambda x: x > 0.5),
    fields = ('state', 'reward'),
    allow_empty = True,
    to_named_tuple = True,
    device = 'cuda'
)

# chunked masked statistics (count, mean, std, rms, min, max)
stats = buffer.field_stats('reward', filter_fields = dict(is_real = True))
```

On episode-major `ReplayBuffer` / `ReplayBufferH5PY`, pass `track_write_step = True` to enable `min_age` / `max_age` queries (off by default to keep existing `metadata.pkl` folders 100% compatible).

### Pulling everything at once

```python
all_data = buffer.get_all_data()  # dict of tensors, time-padded to the longest episode
```

### Rendering episodes to video

When a field holds one frame per timestep, `save_video` renders an episode to `.mp4` or `.gif` (needs `pip install memmap-replay-buffer[video]`). The image field is auto-detected, and frames may be `uint8` or float in `[0., 1.]`. Pass a list of episodes to concatenate them in time.

```python
buffer.save_video('./episode-0.mp4', episode_index = 0, fps = 30)
buffer.save_video('./episodes-0-3-5.mp4', episode_index = [0, 3, 5], fps = 30)
```

`trajectory_to_video(frames, path, fps)` is exported for plain `(time, height, width, channels)` arrays.

The same is available from the command line, with `--sheet` for a contact sheet and `--serve` for a minimal local frontend. `--episode` and `--field` accept comma-delimited lists - with multiple cameras (image-like fields), one video is written per field (`<name>.<field>.<ext>`) and the frontend plays them side by side, synchronized, with a live frame counter to the side. The frontend has frame-level controls: a timeline scrubber, single-frame stepping (buttons, left/right arrows, shift for 10, or mouse wheel over the videos), play/pause, and slowdown playback (down to 0.1x).

```bash
$ inspect-replay-buffer ./image_data --episode 0,3,5 --field front,wrist --output replay.mp4 --serve 8000
```

### Reopening, read-only, and clearing

`overwrite = False` (or `from_folder`) rehydrates an existing buffer from disk; the stored config is validated against the requested one, so mismatched `fields`/`max_episodes` raise a clear error. `read_only = True` guarantees no files are created or written. `clear()` wipes all episodes.

```python
buffer = ReplayBuffer.from_folder('./replay_data', read_only = True)
buffer = ReplayBuffer('./replay_data', max_episodes = 1000, max_timesteps = 500, fields = ..., overwrite = False)
buffer.clear()
```

### Concatenating buffers

`ConcatReplayBuffer` combines several (read-only) buffers of identical fields into one dataset.

```python
from memmap_replay_buffer import ConcatReplayBuffer

concat = ConcatReplayBuffer(['./replay_data_a', './replay_data_b'])
dataloader = concat.dataloader(batch_size = 2, return_mask = True)
```

### Notes

- `circular = True` overwrites the oldest episodes once the buffer is full; `circular = False` raises when full.
- Fields missing from `store` are filled with their declared default value (or zeros).
- `flush_every_store_step` controls how often the memmaps are flushed to disk while storing (default 1; raise it for faster collection at the cost of durability).
- `ReplayBufferH5PY` (`pip install h5py`) is an HDF5-backed variant with the same interface, optionally gzip-compressed.
