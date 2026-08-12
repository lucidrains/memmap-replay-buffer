import torch

from memmap_replay_buffer import ReplayBuffer
from memmap_replay_buffer.replay_buffer import ReplayDatasetTimeWindow


def test_replay_dataset_time_window(tmp_path):
    replay = ReplayBuffer(
        folder = tmp_path,
        max_episodes = 10,
        max_timesteps = 20,
        fields = dict(obs = ('float', (5,))),
        meta_fields = dict(done = ('bool', ()))
    )

    replay.store_episode(obs = torch.ones((15, 5)), done = True)
    replay.store_episode(obs = torch.ones((5, 5)) * 2., done = True)

    dataset = ReplayDatasetTimeWindow(
        replay,
        window_length = 10,
        return_indices = True
    )

    assert len(dataset) == 2

    # verify long episode samples correctly within bounds

    for _ in range(10):
        d = dataset[0]

        assert d['obs'].shape == (10, 5)
        assert torch.all(d['obs'] == 1.)
        assert d['_lens'] == 10

        start = d['_start'].item()
        assert 0 <= start <= 5
        assert d['_reaches_episode_end'] == (start == 5)

    # verify short episode pads correctly and starts at 0

    d = dataset[1]

    assert d['obs'].shape == (10, 5)

    assert torch.all(d['obs'][:5] == 2.)
    assert torch.all(d['obs'][5:] == 0.)

    assert d['_lens'] == 5
    assert d['_start'] == 0
    assert d['_reaches_episode_end']
