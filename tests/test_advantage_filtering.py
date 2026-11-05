import numpy as np
import pytest
import torch

from memmap_replay_buffer import ReplayBuffer
from memmap_replay_buffer.replay_buffer_h5py import ReplayBufferH5PY


@pytest.mark.parametrize('buffer_class', [ReplayBuffer, ReplayBufferH5PY])
def test_update_timestep_start(buffer_class, tmp_path):
    buffer = buffer_class(
        tmp_path,
        max_episodes = 2,
        max_timesteps = 6,
        fields = dict(advantage = 'float'),
    )

    buffer.store_episode(advantage = np.zeros(6, dtype = np.float32))
    buffer.store_episode(advantage = np.ones(6, dtype = np.float32))

    # unsorted indices -> values must follow the passed order
    buffer.update(
        indices = [1, 0],
        timestep_start = 2,
        advantage = torch.tensor([[20., 21.], [10., 11.]])
    )

    data = buffer.get_all_data(fields = ('advantage',))['advantage']

    assert torch.equal(data[0], torch.tensor([0., 0., 10., 11., 0., 0.]))
    assert torch.equal(data[1], torch.tensor([1., 1., 20., 21., 1., 1.]))

    # scalar episode index writes (time, *shape) at the offset
    buffer.update(0, timestep_start = 4, advantage = torch.tensor([40., 41.]))
    data = buffer.get_all_data(fields = ('advantage',))['advantage']
    assert torch.equal(data[0], torch.tensor([0., 0., 10., 11., 40., 41.]))

    with pytest.raises(ValueError, match = 'non-negative'):
        buffer.update(timestep_start = -1, advantage = torch.zeros(2, 1))

    with pytest.raises(ValueError, match = 'max_timesteps'):
        buffer.update(timestep_start = 5, advantage = torch.zeros(2, 2))


@pytest.mark.parametrize('buffer_class', [ReplayBuffer, ReplayBufferH5PY])
def test_advantage_threshold_filtering_end_to_end(buffer_class, tmp_path):
    buffer = buffer_class(
        tmp_path / 'advantage',
        max_episodes = 6,
        max_timesteps = 8,
        fields = dict(
            reward = 'float',
            advantage = 'float',
            high_advantage = ('bool', (), False),
        ),
        circular = False,
        overwrite = True,
    )

    rewards = [
        [1., 1., 1., -1., 1., 1.],
        [-1., -1., -1., -1.],
        [1., -1., 1., -1., 1., -1., 1.],
        [-1., 1., -1.],
        [1., 1., 1., 1., 1.],
        [-1., -1.],
    ]

    for episode_rewards in rewards:
        buffer.store_episode(reward = np.asarray(episode_rewards, dtype = np.float32))

    num_episodes = buffer.num_episodes
    episode_lens = torch.from_numpy(buffer.episode_lens[:num_episodes].copy())
    max_len = int(episode_lens.max())

    # 1. compute advantage in episode batches, writing each batch back

    gamma = 0.9

    for start in range(0, num_episodes, 2):
        episode_indices = np.arange(start, min(start + 2, num_episodes))
        batch_len = int(episode_lens[episode_indices].max())

        rewards_batch = torch.from_numpy(buffer.data['reward'][episode_indices, :batch_len].copy())
        valid = torch.arange(batch_len)[None] < episode_lens[episode_indices][:, None]

        returns = torch.zeros_like(rewards_batch)
        running = torch.zeros(len(episode_indices))

        for t in reversed(range(batch_len)):
            running = rewards_batch[:, t] + gamma * running
            returns[:, t] = running

        buffer.update(episode_indices, advantage = returns * valid)

    # 2. set a mask from a threshold comparison, in time chunks

    threshold = 1.

    for start in range(0, max_len, 4):
        end = min(start + 4, max_len)

        advantages = torch.from_numpy(buffer.data['advantage'][:num_episodes, start:end].copy())
        valid = torch.arange(start, end)[None] < episode_lens[:, None]
        high_advantage = (advantages > threshold) & valid

        buffer.update(timestep_start = start, high_advantage = high_advantage)

    # 3. verify the mask matches the comparison exactly

    advantages = torch.from_numpy(buffer.data['advantage'][:num_episodes].copy())
    high_advantage = torch.from_numpy(buffer.data['high_advantage'][:num_episodes].copy()).bool()

    expected = {
        (episode, timestep)
        for episode in range(num_episodes)
        for timestep in range(int(episode_lens[episode]))
        if advantages[episode, timestep].item() > threshold
    }

    assert len(expected) > 0
    assert len(expected) < int(episode_lens.sum())

    for episode in range(num_episodes):
        for timestep in range(int(episode_lens[episode])):
            assert high_advantage[episode, timestep].item() == ((episode, timestep) in expected)

    # 4. timestep-level learning only receives the selected timesteps

    dataset = buffer.dataset(
        timestep_level = True,
        fields = ('advantage',),
        filter_fields = dict(high_advantage = True),
        return_indices = True,
    )

    fetched = {
        tuple(dataset[index]['_indices'].tolist())
        for index in range(len(dataset))
    }

    assert fetched == expected
    assert all(dataset[index]['advantage'].item() > threshold for index in range(len(dataset)))
