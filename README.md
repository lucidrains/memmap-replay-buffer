## memmap-replay-buffer

An easy-to-use numpy memmap replay buffer for RL and other sequence-based learning tasks.

## Install

```bash
$ pip install memmap-replay-buffer
```

## Usage

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

# train 2 episodes at a time

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

# for loading per timestep

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
