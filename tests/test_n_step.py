import pytest
import torch
import numpy as np
from pathlib import Path
import shutil

from memmap_replay_buffer import ReplayBuffer

@pytest.fixture
def buffer(tmp_path):
    folder = tmp_path / 'n_step_data'

    rb = ReplayBuffer(
        folder = folder,
        max_episodes = 4,
        max_timesteps = 10,
        fields = {
            'obs':    ('float', (2,)),
            'action': ('float', (3,)),
            'reward': 'float',
            'done':   'bool',
        },
        meta_fields = {
            'is_expert': 'bool',
        }
    )

    # ep 0 - 5 timesteps, expert
    with rb.one_episode(is_expert = True):
        for i in range(5):
            rb.store(obs = np.array([1., i]), action = np.ones(3) * i, reward = float(i), done = i == 4)

    # ep 1 - 3 timesteps, not expert
    with rb.one_episode(is_expert = False):
        for i in range(3):
            rb.store(obs = np.array([2., i]), action = np.ones(3) * (10 + i), reward = float(10 + i), done = i == 2)

    # ep 2 - 1 timestep (should be excluded, < 2)
    with rb.one_episode(is_expert = True):
        rb.store(obs = np.array([3., 0.]), action = np.zeros(3), reward = 0., done = True)

    return rb


def test_basic_shapes_and_values(buffer):
    """Current, next, and sequence fields with correct padding."""

    dl = buffer.dataloader(
        batch_size = 6,
        n_steps = 3,
        current_fields = ('obs',),
        next_fields = ('obs',),
        sequence_fields = ('reward',),
        shuffle = False,
    )

    batch = next(iter(dl))

    # ep0 has 4 transitions (t=0..3), ep1 has 2 (t=0,1), ep2 excluded => 6 total
    assert batch['obs'].shape == (6, 2)
    assert batch['next_obs'].shape == (6, 2)
    assert batch['seq_reward'].shape == (6, 3)
    assert batch['n_step_lens'].shape == (6,)

    # t=0 in ep0: actual_n = min(3, 4) = 3, next at t=3
    assert torch.allclose(batch['obs'][0], torch.tensor([1., 0.]))
    assert torch.allclose(batch['next_obs'][0], torch.tensor([1., 3.]))
    assert torch.allclose(batch['seq_reward'][0], torch.tensor([0., 1., 2.]))
    assert batch['n_step_lens'][0] == 3

    # t=3 in ep0: actual_n = min(3, 1) = 1, next at t=4, padded
    assert torch.allclose(batch['obs'][3], torch.tensor([1., 3.]))
    assert torch.allclose(batch['next_obs'][3], torch.tensor([1., 4.]))
    assert torch.allclose(batch['seq_reward'][3], torch.tensor([3., 0., 0.]))
    assert batch['n_step_lens'][3] == 1


def test_sequence_field_same_as_current(buffer):
    """SAC use-case: action in both current_fields and sequence_fields."""

    dl = buffer.dataloader(
        batch_size = 6,
        n_steps = 3,
        current_fields = ('obs', 'action'),
        next_fields = ('obs',),
        sequence_fields = ('action', 'reward'),
        shuffle = False,
    )

    batch = next(iter(dl))

    # current action is scalar at t, seq_action is the chunk
    assert batch['action'].shape == (6, 3)        # (batch, action_dim)
    assert batch['seq_action'].shape == (6, 3, 3)  # (batch, n_steps, action_dim)

    # t=0 ep0: action = [0,0,0], seq_action = [[0,0,0],[1,1,1],[2,2,2]]
    assert torch.allclose(batch['action'][0], torch.zeros(3))
    assert torch.allclose(batch['seq_action'][0], torch.tensor([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]]))

    # t=3 ep0: action = [3,3,3], seq_action = [[3,3,3],[0,0,0],[0,0,0]] (padded)
    assert torch.allclose(batch['action'][3], torch.ones(3) * 3)
    assert torch.allclose(batch['seq_action'][3], torch.tensor([[3., 3., 3.], [0., 0., 0.], [0., 0., 0.]]))


def test_fieldname_map(buffer):
    """Custom fieldname_map overrides default prefixes."""

    dl = buffer.dataloader(
        batch_size = 6,
        n_steps = 3,
        current_fields = ('obs',),
        next_fields = ('obs',),
        sequence_fields = ('reward',),
        fieldname_map = {
            'obs': 'state',
            'next_obs': 'next_state',
            'seq_reward': 'rewards',
        },
        shuffle = False,
    )

    batch = next(iter(dl))
    assert 'state' in batch
    assert 'next_state' in batch
    assert 'rewards' in batch
    assert batch['state'].shape == (6, 2)


def test_filter_meta(buffer):
    """filter_meta selects only expert episodes."""

    dl = buffer.dataloader(
        batch_size = 10,
        n_steps = 2,
        current_fields = ('obs',),
        next_fields = ('obs',),
        filter_meta = {'is_expert': True},
        shuffle = False,
    )

    batch = next(iter(dl))

    # only ep0 (expert, 4 transitions), ep2 excluded (1 timestep)
    assert batch['obs'].shape[0] == 4
    assert torch.all(batch['obs'][:, 0] == 1.)  # all from ep0


def test_to_named_tuple(buffer):
    """to_named_tuple creates proper namedtuple batches."""

    dl = buffer.dataloader(
        batch_size = 6,
        n_steps = 2,
        current_fields = ('obs',),
        next_fields = ('obs',),
        sequence_fields = ('reward',),
        to_named_tuple = ('obs', 'next_obs', 'seq_reward', 'n_step_lens'),
        shuffle = False,
    )

    batch = next(iter(dl))
    assert hasattr(batch, 'obs')
    assert hasattr(batch, 'next_obs')
    assert hasattr(batch, 'seq_reward')
    assert hasattr(batch, 'n_step_lens')


def test_metadata_included(buffer):
    """Meta fields are included by default."""

    dl = buffer.dataloader(
        batch_size = 6,
        n_steps = 2,
        current_fields = ('obs',),
        next_fields = ('obs',),
        shuffle = False,
    )

    batch = next(iter(dl))
    assert 'is_expert' in batch


def test_filter_fields_scalar(buffer):
    """filter_fields works for scalar fields like done."""

    ds = buffer.dataset(
        n_steps = 2,
        current_fields = ('obs',),
        next_fields = ('obs',),
        filter_fields = {'done': False},
    )

    assert len(ds) > 0


def test_filter_fields_rejects_multidim(buffer):
    """filter_fields must reject non-scalar fields."""

    with pytest.raises(AssertionError, match = 'scalar'):
        buffer.dataset(
            n_steps = 2,
            current_fields = ('obs',),
            next_fields = ('obs',),
            filter_fields = {'obs': 0.},
        )
