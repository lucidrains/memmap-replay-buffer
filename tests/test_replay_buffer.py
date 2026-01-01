import pytest
import torch

def test_replay():
    from memmap_replay_buffer import ReplayBuffer

    replay_buffer = ReplayBuffer(
        './replay_data',
        max_episodes = 10_000,
        max_timesteps = 501,
        fields = dict(
            state = ('float', (8,)),
            action = 'int',
            action_log_prob = 'float',
            reward = 'float',
            value = 'float',
            done = 'bool'
        )
    )

    lens = [3, 5, 4]

    for episode_len in lens:
        with replay_buffer.one_episode():
            for _ in range(episode_len):
                state = torch.randn((8,))
                action = torch.randint(0, 4, ())
                log_prob = torch.randn(())
                reward = torch.randn(())
                value = torch.randn(())
                done = torch.randint(0, 2, ()).bool()

                replay_buffer.store(
                    state = state,
                    action = action,
                    action_log_prob = log_prob,
                    reward = reward,
                    value = value,
                    done = done
                )

    dataset = replay_buffer.dataset()

    assert len(dataset) == 3

    assert torch.is_tensor(dataset[0]['state'])

    dataloader = replay_buffer.dataloader(batch_size = 3)

    assert next(iter(dataloader))['state'].shape[0] == 3

    # we will now consider consecutive pairs of episodes as 2 trials to be used for in-context adaptation
    # but realistically there will be a function that converts a given ReplayBuffer -> Int[batch, episode_indices]

    from torch import stack, arange

    episode_indices = arange(len(replay_buffer))
    remapped_episodes = stack((episode_indices[:-1], episode_indices[1:]))

    dataloader = replay_buffer.dataloader(
        batch_size = 1,
        episode_mapping = remapped_episodes
    )

    assert next(iter(dataloader))['_lens'][0] == (3 + 5) # first and second episodes are concatted together timewise
