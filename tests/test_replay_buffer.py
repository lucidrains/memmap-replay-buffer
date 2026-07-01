import pytest
import torch
from pathlib import Path

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

def test_read_only():
    from memmap_replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(
        './test_read_only_data',
        max_episodes = 10,
        max_timesteps = 10,
        fields = dict(state = 'float'),
        read_only = True
    )

    with pytest.raises(AssertionError):
        buffer.store(state = 1.0)

    with pytest.raises(AssertionError):
        buffer.clear()

def test_store_batch():
    from memmap_replay_buffer import ReplayBuffer
    import shutil

    folder = './test_batch_data'
    if shutil.os.path.exists(folder):
        shutil.rmtree(folder)

    buffer = ReplayBuffer(
        folder,
        max_episodes = 5,
        max_timesteps = 1,
        fields = dict(state = 'float'),
        circular = False
    )

    # 1. Store batch with enough space
    buffer.store_batch(state = torch.randn(3))
    buffer.advance_episode(3)
    assert buffer.num_episodes == 3

    # 2. Store batch with leftover space (should slice to 2)
    buffer.store_batch(state = torch.randn(4))
    buffer.advance_episode(2)
    assert buffer.num_episodes == 5

    # 3. Store batch when full (should raise error in advance_episode or store_batch)
    with pytest.raises(ValueError):
        buffer.store_batch(state = torch.randn(1))
        buffer.advance_episode(1)

    # 4. Circular buffer wrap-around
    shutil.rmtree(folder)
    buffer = ReplayBuffer(
        folder,
        max_episodes = 5,
        max_timesteps = 5,
        fields = dict(state = 'float'),
        meta_fields = dict(label = 'int'),
        circular = True
    )

    buffer.store_meta_batch(label = torch.tensor([1, 2, 3]))
    buffer.store_batch(state = torch.ones(2))
    buffer.advance_episode(batch_size = 3) # advanced 3 episodes

    # 5. Verify storage at non-zero timestep index
    buffer.timestep_index = 2
    buffer.store_batch(state = torch.zeros(2)) # Should store at index [3, 4] at timestep 2
    buffer.advance_episode(batch_size = 2)

    assert buffer.num_episodes == 5

    data = buffer.get_all_data(fields = ('state',), meta_fields = ('label',))

    # Check meta batch (indices 0, 1, 2)
    assert torch.all(data['label'][:3] == torch.tensor([1, 2, 3]))

    # Check first data batch (indices 0, 1) - wait, indices were 0, 1, 2 for label, then 2 for state advanced 3.
    # Actually 0, 1, 2 were label stored at ep 0, 1, 2.
    # Then store_batch state ones(2) at ep 0, 1.
    # Then advance_episode(3) -> episode_index is 3.
    # Then store_batch zeros(2) at ep 3, 4.

    # Check data storage
    assert torch.all(data['state'][:2, 0] == 1)
    assert torch.all(data['state'][3:5, 2] == 0)

    # 6. Verify robust batch computation
    shutil.rmtree(folder)
    buffer = ReplayBuffer(
        folder,
        max_episodes = 5,
        max_timesteps = 5,
        fields = dict(state = 'float'),
        meta_fields = dict(label = 'int'),
        circular = True
    )

    # Test list input
    buffer.store_meta_batch(label = [1, 2, 3])
    buffer.advance_episode(3)
    assert torch.all(buffer.get_all_data(meta_fields = ('label',))['label'][:3] == torch.tensor([1, 2, 3]))

    # Test empty data assertion
    with pytest.raises(AssertionError):
        buffer.store_batch()

    # Test mismatched batch size assertion
    buffer = ReplayBuffer(
        folder,
        max_episodes = 5,
        max_timesteps = 5,
        fields = dict(state = 'float', action = 'int'),
        circular = True
    )
    with pytest.raises(AssertionError):
        buffer.store_batch(state = torch.ones(3), action = torch.zeros(2))

    # Test invalid field name assertion
    with pytest.raises(AssertionError):
        buffer.store_batch(invalid_field = torch.ones(3))

    # Test invalid meta field name assertion
    with pytest.raises(AssertionError):
        buffer.store_meta_batch(invalid_meta = torch.tensor([1, 2, 3]))

    # 7. Test batched_episode context manager
    shutil.rmtree(folder)
    buffer = ReplayBuffer(
        folder,
        max_episodes = 10,
        max_timesteps = 5,
        fields = dict(state = 'float'),
        meta_fields = dict(label = 'int'),
        circular = True
    )

    with buffer.batched_episode(batch_size = 3, label = [10, 20, 30]):
        buffer.store_batch(state = torch.ones(3))
        buffer.store_batch(state = torch.zeros(3))

    assert buffer.num_episodes == 3
    assert buffer.episode_index == 3

    data = buffer.get_all_data()
    assert torch.all(data['label'][:3] == torch.tensor([10, 20, 30]))
    assert torch.all(data['state'][:3, 0] == 1)
    assert torch.all(data['state'][:3, 1] == 0)
    assert (torch.from_numpy(buffer.episode_lens[:3].copy()) == 2).all()

def test_consistency():
    from memmap_replay_buffer import ReplayBuffer
    import shutil
    import torch
    import numpy as np

    folder_seq = './test_seq'
    folder_batch = './test_batch'

    for f in (folder_seq, folder_batch):
        if shutil.os.path.exists(f):
            shutil.rmtree(f)

    max_episodes = 5
    max_timesteps = 5
    batch_size = 3
    total_episodes = 8 # 8 > 5, so will wrap around

    fields = dict(state = 'float', action = 'int')

    # 1. Sequential Buffer
    buffer_seq = ReplayBuffer(folder_seq, max_episodes, max_timesteps, fields, circular = True)

    for i in range(total_episodes):
        with buffer_seq.one_episode():
            for t in range(max_timesteps):
                buffer_seq.store(state = float(i), action = i)

    # 2. Batched Buffer
    buffer_batch = ReplayBuffer(folder_batch, max_episodes, max_timesteps, fields, circular = True)

    # store 2 batches of 3, then one batch of 2

    # Batch 1 (eps 0, 1, 2)
    with buffer_batch.batched_episode(batch_size = 3):
        for t in range(max_timesteps):
            buffer_batch.store_batch(
                state = torch.tensor([float(0), float(1), float(2)]),
                action = torch.tensor([0, 1, 2])
            )

    # Batch 2 (eps 3, 4, 0)
    with buffer_batch.batched_episode(batch_size = 3):
        for t in range(max_timesteps):
            buffer_batch.store_batch(
                state = torch.tensor([float(3), float(4), float(5)]),
                action = torch.tensor([3, 4, 5])
            )

    # Batch 3 (eps 1, 2)
    with buffer_batch.batched_episode(batch_size = 2):
        for t in range(max_timesteps):
            buffer_batch.store_batch(
                state = torch.tensor([float(6), float(7)]),
                action = torch.tensor([6, 7])
            )

    # 3. Assert Parity
    data_seq = buffer_seq.get_all_data()
    data_batch = buffer_batch.get_all_data()

    for key in data_seq:
        assert torch.all(data_seq[key] == data_batch[key]), f'Mismatched data for {key}'

    assert np.all(buffer_seq.episode_lens == buffer_batch.episode_lens)
    assert buffer_seq.episode_index == buffer_batch.episode_index
    assert buffer_seq.num_episodes == buffer_batch.num_episodes

    # cleanup
    shutil.rmtree(folder_seq)
    shutil.rmtree(folder_batch)

def test_update():
    from memmap_replay_buffer import ReplayBuffer
    import shutil
    import torch
    import numpy as np

    folder = './test_update'
    if shutil.os.path.exists(folder):
        shutil.rmtree(folder)

    buf = ReplayBuffer(
        folder,
        max_episodes = 5,
        max_timesteps = 20,
        fields = dict(
            returns = 'float',
            value = ('float', 10),
        ),
        circular = True,
        overwrite = True
    )

    for ep in range(3):
        with buf.one_episode():
            for t in range(8):
                buf.store(returns=0., value=torch.zeros(10))

    # Test 1: batch update with np.array indices
    indices = np.array([0, 1, 2])
    returns = torch.randn(3, 8)
    values = torch.randn(3, 8, 10)
    buf.update(indices, returns=returns, value=values)
    assert np.allclose(buf.data['returns'][0, 0], returns[0, 0].item(), atol=1e-6)

    # Test 2: scalar index
    returns_s = torch.randn(8)
    buf.update(1, returns=returns_s)
    assert np.allclose(buf.data['returns'][1, 0], returns_s[0].item(), atol=1e-6)

    # Test 3: slice index
    returns_sl = torch.randn(2, 8)
    buf.update(slice(0, 2), returns=returns_sl)
    assert np.allclose(buf.data['returns'][0, 0], returns_sl[0, 0].item(), atol=1e-6)

    # Test 4: indices=None (all populated episodes)
    returns_all = torch.randn(3, 8)
    buf.update(returns=returns_all)
    for i in range(3):
        assert np.allclose(buf.data['returns'][i, 0], returns_all[i, 0].item(), atol=1e-6)

    # Test 5: partial time dimension
    partial = torch.randn(1, 5)
    buf.update(np.array([0]), returns=partial)
    assert np.allclose(buf.data['returns'][0, 4], partial[0, 4].item(), atol=1e-6)

    shutil.rmtree(folder)

def test_slice_by_episode_len():
    from memmap_replay_buffer import ReplayBuffer
    import shutil
    import torch

    folder = './test_slice'
    if shutil.os.path.exists(folder):
        shutil.rmtree(folder)

    buffer = ReplayBuffer(
        folder,
        max_episodes = 2,
        max_timesteps = 10,
        fields = dict(state = 'float'),
    )

    with buffer.one_episode():
        for _ in range(3):
            buffer.store(state=0.0)

    # with default slice_by_episode_len=True, state should have shape (3,)
    dataset_sliced = buffer.dataset(slice_by_episode_len=True)
    assert dataset_sliced[0]['state'].shape[0] == 3

    # with slice_by_episode_len=False, state should have shape (10,)
    dataset_unsliced = buffer.dataset(slice_by_episode_len=False)
    assert dataset_unsliced[0]['state'].shape[0] == 10

    shutil.rmtree(folder)

def test_return_episode_lens():
    from memmap_replay_buffer import ReplayBuffer

    replay_buffer = ReplayBuffer(
        './replay_data_lens',
        max_episodes = 10,
        max_timesteps = 10,
        fields = dict(
            state = 'float',
        )
    )

    with replay_buffer.one_episode():
        for _ in range(5):
            replay_buffer.store(state = torch.randn(()))

    dataset_with_lens = replay_buffer.dataset(return_episode_lens=True)
    assert '_lens' in dataset_with_lens[0]

    dataset_without_lens = replay_buffer.dataset(return_episode_lens=False)
    assert '_lens' not in dataset_without_lens[0]

    with pytest.raises(AssertionError):
        replay_buffer.dataset(slice_by_episode_len=False, return_episode_lens=False)

def test_slice_by_episode_len_multiple_fields():
    from memmap_replay_buffer import ReplayBuffer
    import torch

    replay_buffer = ReplayBuffer(
        './replay_data_slice_multiple',
        max_episodes = 10,
        max_timesteps = 10,
        fields = dict(
            state = ('float', (8,)),
            action = 'int',
            actions = 'int',
            reward = 'float'
        )
    )

    with replay_buffer.one_episode():
        for _ in range(3):
            replay_buffer.store(
                state = torch.randn((8,)),
                action = torch.randint(0, 4, ()),
                actions = torch.randint(0, 4, ()),
                reward = torch.randn(())
            )

    with replay_buffer.one_episode():
        for _ in range(7):
            replay_buffer.store(
                state = torch.randn((8,)),
                action = torch.randint(0, 4, ()),
                actions = torch.randint(0, 4, ()),
                reward = torch.randn(())
            )

    dataset = replay_buffer.dataset(slice_by_episode_len=True)

    assert len(dataset) == 2

    ep1 = dataset[0]
    assert ep1['state'].shape[0] == 3
    assert ep1['action'].shape[0] == 3
    assert ep1['actions'].shape[0] == 3
    assert ep1['reward'].shape[0] == 3

    ep2 = dataset[1]
    assert ep2['state'].shape[0] == 7
    assert ep2['action'].shape[0] == 7
    assert ep2['actions'].shape[0] == 7
    assert ep2['reward'].shape[0] == 7

def test_concat_replay_buffer(tmp_path: Path):
    import numpy as np
    from memmap_replay_buffer import ReplayBuffer, ConcatReplayBuffer

    folder1 = tmp_path / "buf1"
    folder2 = tmp_path / "buf2"

    # Create first buffer
    buf1 = ReplayBuffer(
        folder=folder1,
        max_episodes=2,
        max_timesteps=10,
        fields=dict(
            state=("float", (4,)),
            action="int",
            reward="float"
        ),
        meta_fields=dict(
            reward_sum="float"
        )
    )

    # Store 1 episode in buf1
    with buf1.one_episode(reward_sum=1.0) as meta:
        for t in range(5):
            buf1.store(
                state=np.array([1, 1, 1, 1]) * t,
                action=0,
                reward=1.0
            )

    # Create second buffer
    buf2 = ReplayBuffer(
        folder=folder2,
        max_episodes=3,
        max_timesteps=12,  # can be different
        fields=dict(
            state=("float", (4,)),
            action="int",
            reward="float"
        ),
        meta_fields=dict(
            reward_sum="float"
        )
    )

    # Store 2 episodes in buf2
    with buf2.one_episode(reward_sum=2.0) as meta:
        for t in range(8):
            buf2.store(
                state=np.array([2, 2, 2, 2]) * t,
                action=1,
                reward=2.0
            )

    with buf2.one_episode(reward_sum=3.0) as meta:
        for t in range(3):
            buf2.store(
                state=np.array([3, 3, 3, 3]) * t,
                action=2,
                reward=3.0
            )

    # Create a completely empty buffer
    folder4 = tmp_path / "buf4"
    buf4 = ReplayBuffer(
        folder=folder4,
        max_episodes=2,
        max_timesteps=10,
        fields=dict(state=("float", (4,)), action="int", reward="float"),
        meta_fields=dict(reward_sum="float")
    )

    # Now create ConcatReplayBuffer with an empty buffer included
    concat_buf = ConcatReplayBuffer([folder1, folder4, folder2])

    assert concat_buf.num_episodes == 3
    assert concat_buf.max_episodes == 7
    assert concat_buf.max_timesteps == 12
    assert len(concat_buf) == 3

    # Test dataset
    ds = concat_buf.dataset()
    assert len(ds) == 3

    # First item
    item0 = ds[0]
    assert item0["state"].shape == (5, 4)
    assert item0["reward_sum"].item() == 1.0
    assert torch.all(item0["state"][0] == 0)
    assert torch.all(item0["state"][-1] == 4)
    assert item0["_lens"].item() == 5

    # Second item
    item1 = ds[1]
    assert item1["state"].shape == (8, 4)
    assert item1["reward_sum"].item() == 2.0
    assert item1["_lens"].item() == 8

    # Third item
    item2 = ds[2]
    assert item2["state"].shape == (3, 4)
    assert item2["reward_sum"].item() == 3.0
    assert item2["_lens"].item() == 3

    # Test get_all_data
    all_data = concat_buf.get_all_data()
    # Padded state should be (3, 8, 4) since max length across these episodes is 8
    assert all_data["state"].shape == (3, 8, 4)
    assert all_data["reward_sum"].shape == (3,)
    assert all_data["action"].shape == (3, 8)

    # Ensure padding is zeros
    # item 0 has length 5, so remaining 3 steps should be padded with 0
    assert torch.all(all_data["state"][0, 5:] == 0)
    assert torch.all(all_data["action"][0, 5:] == 0)

    # Test dataloader
    dl = concat_buf.dataloader(batch_size=2)
    batches = list(dl)
    assert len(batches) == 2

    b0 = batches[0]
    assert b0["state"].shape == (2, 8, 4) # max length in this batch is 8 (from item1)

    b1 = batches[1]
    assert b1["state"].shape == (1, 3, 4) # length in this batch is 3 (from item2)

    # Create third buffer to match the concatenated buffers
    folder3 = tmp_path / "buf3"
    buf3 = ReplayBuffer(
        folder=folder3,
        max_episodes=5,
        max_timesteps=12,
        fields=dict(
            state=("float", (4,)),
            action="int",
            reward="float"
        ),
        meta_fields=dict(
            reward_sum="float"
        )
    )

    with buf3.one_episode(reward_sum=1.0) as meta:
        for t in range(5):
            buf3.store(state=np.array([1, 1, 1, 1]) * t, action=0, reward=1.0)

    with buf3.one_episode(reward_sum=2.0) as meta:
        for t in range(8):
            buf3.store(state=np.array([2, 2, 2, 2]) * t, action=1, reward=2.0)

    with buf3.one_episode(reward_sum=3.0) as meta:
        for t in range(3):
            buf3.store(state=np.array([3, 3, 3, 3]) * t, action=2, reward=3.0)

    all_data_buf3 = buf3.get_all_data()

    for k in all_data.keys():
        assert torch.allclose(all_data[k], all_data_buf3[k])

    # test write guard
    with pytest.raises(NotImplementedError):
        concat_buf.clear()

def test_store_meta_after_episode():
    from memmap_replay_buffer import ReplayBuffer
    import shutil

    folder = './test_meta_after_episode'
    if shutil.os.path.exists(folder):
        shutil.rmtree(folder)

    buffer = ReplayBuffer(
        folder,
        max_episodes=1,
        max_timesteps=10,
        fields=dict(reward='float'),
        meta_fields=dict(cum_reward='float'),
        circular=True
    )

    with buffer.one_episode():
        for _ in range(5):
            buffer.store(reward=1.0)

    buffer.store_meta_datapoint(0, 'cum_reward', 5.0)

    dataset = buffer.dataset()
    assert len(dataset) == 1

    data = dataset[0]
    assert torch.all(data['reward'] == 1.0), "Data was improperly zeroed out!"
    assert data['cum_reward'].item() == 5.0
