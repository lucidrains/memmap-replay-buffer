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
        state = ('float', (8,), 0.5), # type, shape, and optional default value
        action = 'int',               # default shape is ()
        reward = 'float'
    ),
    meta_fields = dict(
        task_id = 'int'
    ),
    circular = True,
    overwrite = True
)

# store an episode

for _ in range(4):
    with buffer.one_episode(task_id = 1):
        for _ in range(100):
            buffer.store(
                state = torch.randn(8).numpy(),
                action = torch.randint(0, 4, ()),
                reward = 1.0
            )

# rehydrate from disk

buffer_rehydrated = ReplayBuffer.from_config('./replay_data')
assert buffer_rehydrated.num_episodes == 4

# setup dataloader

dataloader = buffer.dataloader(batch_size = 2)

for batch in dataloader:
    state = batch['state']    # (2, 100, 8)
    action = batch['action']  # (2, 100)
    reward = batch['reward']  # (2, 100)
    lens = batch['_lens']     # (2,)

    assert state.shape  == (2, 100, 8)
    assert action.shape == (2, 100)
    assert reward.shape == (2, 100)
    assert lens.shape   == (2,)
```
