import pytest
import torch
import numpy as np
from pathlib import Path
from memmap_replay_buffer import ReplayBuffer

def test_persistence(tmp_path):
    folder = tmp_path / "replay_data"
    
    fields = dict(
        state = ('float', (8,), 0.5), # default value 0.5
        action = 'int'
    )
    
    meta_fields = dict(
        invalidated = 'bool'
    )

    # 1. Create and store some data
    rb = ReplayBuffer(
        folder,
        max_episodes = 10,
        max_timesteps = 5,
        fields = fields,
        meta_fields = meta_fields,
        circular = True
    )
    
    with rb.one_episode(episode_lens = 3) as episodes_meta: # testing one_episode with meta_data
        episodes_meta['invalidated'] = False # testing yielding dictionary
        for _ in range(3):
            rb.store(action = 1)
            
    # state should be defaulted to 0.5
    assert np.all(rb.memmaps['state'][0, :3] == 0.5)
    assert rb.num_episodes == 1
    assert rb.episode_index == 1
    
    rb.flush()
    
    # 2. Rehydrate from config
    rb2 = ReplayBuffer.from_config(folder)
    assert rb2.max_episodes == 10
    assert rb2.max_timesteps == 5
    assert rb2.num_episodes == 1
    assert rb2.episode_index == 1
    assert rb2.circular == True
    
    # 3. Continue storing
    with rb2.one_episode():
        for _ in range(2):
            rb2.store(state = torch.ones(8), action = 2)
            
    assert rb2.num_episodes == 2
    assert rb2.episode_index == 2
    
    # 4. Test Circularity
    rb3 = ReplayBuffer(
        tmp_path / "circular_data",
        max_episodes = 2,
        max_timesteps = 5,
        fields = fields,
        circular = True
    )
    
    for i in range(3):
        with rb3.one_episode():
            rb3.store(action = i)
            
    assert rb3.num_episodes == 2
    assert rb3.episode_index == 1 # 0, 1 -> wrap to 0 -> 1

def test_overwrite(tmp_path):
    folder = tmp_path / "overwrite_data"
    
    fields = dict(action = 'int')
    
    rb = ReplayBuffer(folder, 10, 5, fields)
    with rb.one_episode():
        rb.store(action = 1)
        
    assert rb.num_episodes == 1
    
    # Overwrite
    rb2 = ReplayBuffer(folder, 10, 5, fields, overwrite = True)
    assert rb2.num_episodes == 0
    assert rb2.episode_index == 0

def test_clear(tmp_path):
    folder = tmp_path / "clear_data"
    fields = dict(state = ('float', (8,), 0.5))
    
    rb = ReplayBuffer(folder, 10, 5, fields)
    with rb.one_episode():
        rb.store(state = torch.zeros(8))
        
    assert rb.num_episodes == 1
    assert np.all(rb.memmaps['state'][0, 0] == 0)
    
    rb.clear()
    
    assert rb.num_episodes == 0
    assert rb.episode_index == 0
    assert np.all(rb.memmaps['state'][0, 0] == 0.5) # restored default

def test_full_buffer_error(tmp_path):
    folder = tmp_path / "full_data"
    fields = dict(action = 'int')
    
    rb = ReplayBuffer(folder, 2, 5, fields, circular = False)
    
    with rb.one_episode():
        rb.store(action = 1)
        
    with rb.one_episode():
        rb.store(action = 2)
        
    assert rb.num_episodes == 2
    
    with pytest.raises(ValueError, match = "buffer is full"):
        with rb.one_episode():
            rb.store(action = 3)

def test_flush_current_episode(tmp_path):
    folder = tmp_path / "flush_data"
    fields = dict(a = 'int')
    
    rb = ReplayBuffer(folder, 10, 5, fields)
    with rb.one_episode():
        rb.store(a = 1)
        dataset = rb.dataset()
        assert len(dataset) == 1
        assert dataset[0]['a'][0] == 1

def test_full_timestep_error(tmp_path):
    folder = tmp_path / "full_timestep_data"
    fields = dict(action = 'int')
    
    rb = ReplayBuffer(folder, 5, 2, fields)
    
    with rb.one_episode():
        rb.store(action = 1)
        rb.store(action = 2)
        
        with pytest.raises(ValueError, match = "exceeded the `max_timesteps`"):
            rb.store(action = 3)
