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
