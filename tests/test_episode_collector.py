import torch
import pytest
from pathlib import Path
from shutil import rmtree
from memmap_replay_buffer import ReplayBuffer
from memmap_replay_buffer.replay_buffer_h5py import ReplayBufferH5PY

@pytest.mark.parametrize("BufferClass", [ReplayBuffer, ReplayBufferH5PY])
def test_episode_collector(BufferClass):
    path = Path('./test_collector_unit')
    rmtree(path, ignore_errors = True)

    fields = dict(
        obs = ('float', (512,)),
        actions = 'int'
    )
    
    meta_fields = dict(
        rewards = 'float'
    )

    rb = BufferClass(
        path,
        max_episodes = 10,
        max_timesteps = 100,
        fields = fields,
        meta_fields = meta_fields
    )

    num_groups = 4
    num_steps = 3

    collector = rb.create_collector(num_groups = num_groups)

    obs = torch.randn(num_groups, num_steps, 512)
    actions = torch.randint(0, 10, (num_groups, num_steps))
    rewards = torch.randn(num_groups)

    # Append timesteps
    for t in range(num_steps):
        collector.append(
            obs = obs[:, t:t+1],
            actions = actions[:, t:t+1]
        )

    # Store with meta data
    collector.store(rewards = rewards)

    # Verify
    assert rb.num_episodes == num_groups
    data = rb.get_all_data()

    assert torch.allclose(data['obs'][:num_groups], obs.reshape(num_groups, num_steps, 512))
    assert torch.equal(data['actions'][:num_groups], actions.long())
    assert torch.allclose(data['rewards'][:num_groups], rewards)

    if hasattr(rb, 'file'): rb.file.close()
    rmtree(path)
