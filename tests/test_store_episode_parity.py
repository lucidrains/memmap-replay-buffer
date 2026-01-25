import shutil
from pathlib import Path
import numpy as np
import torch
from memmap_replay_buffer import ReplayBuffer
from memmap_replay_buffer.replay_buffer_h5py import ReplayBufferH5PY

def test_memmap_parity():
    print("Testing ReplayBuffer (memmap) parity...")
    folder1 = Path("test_buffer_store")
    folder2 = Path("test_buffer_store_episode")
    
    if folder1.exists(): shutil.rmtree(folder1)
    if folder2.exists(): shutil.rmtree(folder2)
    
    max_episodes = 5
    max_timesteps = 10
    fields = {'state': ('float', (4,)), 'action': ('int', ())}
    
    buffer_store = ReplayBuffer(folder1, max_episodes, max_timesteps, fields)
    buffer_episode = ReplayBuffer(folder2, max_episodes, max_timesteps, fields)
    
    # Generate dummy data
    states = np.random.randn(8, 4).astype(np.float32)
    actions = np.random.randint(0, 5, size=(8,)).astype(np.int32)
    
    # Store using store() in a loop
    with buffer_store.one_episode():
        for t in range(8):
            buffer_store.store(state=states[t], action=actions[t])
            
    # Store using store_episode()
    buffer_episode.store_episode(state=states, action=actions)
    
    # Compare
    data_store = buffer_store.get_all_data()
    data_episode = buffer_episode.get_all_data()
    
    for key in fields:
        assert np.allclose(data_store[key], data_episode[key]), f"Mismatch in {key}"
        
    print("ReplayBuffer parity test passed!")
    
    shutil.rmtree(folder1)
    shutil.rmtree(folder2)

def test_h5py_parity():
    print("\nTesting ReplayBufferH5PY parity...")
    folder1 = Path("test_h5py_store")
    folder2 = Path("test_h5py_store_episode")
    
    if folder1.exists(): shutil.rmtree(folder1)
    if folder2.exists(): shutil.rmtree(folder2)
    
    max_episodes = 5
    max_timesteps = 10
    fields = {'state': ('float', (4,)), 'action': ('int', ())}
    
    buffer_store = ReplayBufferH5PY(folder1, max_episodes, max_timesteps, fields)
    buffer_episode = ReplayBufferH5PY(folder2, max_episodes, max_timesteps, fields)
    
    # Generate dummy data
    states = np.random.randn(8, 4).astype(np.float32)
    actions = np.random.randint(0, 5, size=(8,)).astype(np.int32)
    
    # Store using store() in a loop
    with buffer_store.one_episode():
        for t in range(8):
            buffer_store.store(state=states[t], action=actions[t])
            
    # Store using store_episode()
    buffer_episode.store_episode(state=states, action=actions)
    
    # Compare
    data_store = buffer_store.get_all_data()
    data_episode = buffer_episode.get_all_data()
    
    for key in fields:
        assert np.allclose(data_store[key], data_episode[key]), f"Mismatch in {key}"
        
    print("ReplayBufferH5PY parity test passed!")
    
    # Explicitly close H5 files before deleting folders
    del buffer_store
    del buffer_episode
    
    shutil.rmtree(folder1)
    shutil.rmtree(folder2)

if __name__ == "__main__":
    test_memmap_parity()
    test_h5py_parity()
