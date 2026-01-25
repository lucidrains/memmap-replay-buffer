import pytest
import torch
import numpy as np

h5py = pytest.importorskip("h5py")
import shutil
from pathlib import Path
from memmap_replay_buffer import ReplayBuffer
from memmap_replay_buffer.replay_buffer_h5py import ReplayBufferH5PY

@pytest.fixture
def temp_folders():
    folder_memmap = Path('./test_pytest_replay_memmap')
    folder_h5py = Path('./test_pytest_replay_h5py')
    
    if folder_memmap.exists(): shutil.rmtree(folder_memmap)
    if folder_h5py.exists(): shutil.rmtree(folder_h5py)
    
    yield folder_memmap, folder_h5py
    
    if folder_memmap.exists(): shutil.rmtree(folder_memmap)
    if folder_h5py.exists(): shutil.rmtree(folder_h5py)

def test_parity_and_images(temp_folders):
    folder_mem, folder_h5 = temp_folders
    
    max_episodes = 5
    max_timesteps = 10
    
    fields = {
        'obs': ('float', (4,)),
        'image': ('float', (32, 32, 3)), # testing images
        'action': 'int'
    }
    
    # Initialize both buffers
    rb_mem = ReplayBuffer(folder_mem, max_episodes, max_timesteps, fields)
    
    # Test H5 with compression
    rb_h5 = ReplayBufferH5PY(
        folder_h5, 
        max_episodes, 
        max_timesteps, 
        fields,
        h5py_compression='gzip',
        h5py_compression_opts=4
    )
    
    # Store some data
    for _ in range(3):
        obs = torch.randn(4)
        image = torch.randn(32, 32, 3)
        action = torch.randint(0, 5, ()).item()
        
        rb_mem.store(obs=obs, image=image, action=action)
        rb_h5.store(obs=obs, image=image, action=action)
        
    rb_mem.advance_episode()
    rb_h5.advance_episode()
    
    # Verify properties
    assert rb_mem.num_episodes == rb_h5.num_episodes
    assert rb_mem.episode_index == rb_h5.episode_index
    assert rb_mem.timestep_index == rb_h5.timestep_index
    
    # Verify retrieval
    data_mem = rb_mem.get_all_data()
    data_h5 = rb_h5.get_all_data()
    
    assert data_mem.keys() == data_h5.keys()
    for k in data_mem:
        torch.testing.assert_close(data_mem[k], data_h5[k])
        
    # Verify dataset retrieval
    ds_mem = rb_mem.dataset()
    ds_h5 = rb_h5.dataset()
    
    assert len(ds_mem) == len(ds_h5)
    item_mem = ds_mem[0]
    item_h5 = ds_h5[0]
    
    for k in item_mem:
        if k != '_lens':
            torch.testing.assert_close(item_mem[k], item_h5[k])

    # Check compression was actually set on H5
    import h5py
    with h5py.File(folder_h5 / 'data.h5', 'r') as f:
        assert f['data_image'].compression == 'gzip'
        assert f['data_image'].chunks is not None
        print(f"\nImage chunk shape: {f['data_image'].chunks}")

def test_store_batch_parity(temp_folders):
    folder_mem, folder_h5 = temp_folders
    
    fields = {'vec': ('float', (8,))}
    rb_mem = ReplayBuffer(folder_mem, 10, 10, fields)
    rb_h5 = ReplayBufferH5PY(folder_h5, 10, 10, fields)
    
    batch_size = 4
    data = torch.randn(batch_size, 8)
    
    rb_mem.store_batch(vec=data)
    rb_h5.store_batch(vec=data)
    
    rb_mem.advance_episode(batch_size=batch_size)
    rb_h5.advance_episode(batch_size=batch_size)
    
    all_mem = rb_mem.get_all_data()
    all_h5 = rb_h5.get_all_data()
    
    torch.testing.assert_close(all_mem['vec'], all_h5['vec'])

def test_buffered_storer_parity(temp_folders):
    _, folder_h5 = temp_folders
    
    max_episodes = 10
    max_timesteps = 5
    fields = {
        'obs': ('float', (4,))
    }
    
    # We use two subfolders for reference and buffered
    ref_folder = folder_h5 / 'ref'
    buf_folder = folder_h5 / 'buf'
    
    rb_ref = ReplayBufferH5PY(ref_folder, max_episodes, max_timesteps, fields, overwrite=True)
    rb_buf = ReplayBufferH5PY(buf_folder, max_episodes, max_timesteps, fields, overwrite=True)
    
    buffered_storer = rb_buf.get_buffered_storer(flush_freq = 4)
    
    np.random.seed(42)
    
    for i in range(8):
        # generate 1 episode data
        length = np.random.randint(1, max_timesteps + 1)
        obs = np.random.randn(max_timesteps, 4).astype(np.float32)
        # ZERO OUT unwritten part to match what store() would leave (zeros)
        obs[length:] = 0.
        
        mask = np.zeros(max_timesteps, dtype=np.float32)
        mask[:length] = 1.
        
        # Store in reference buffer using standard store (immediate)
        for t in range(length):
            rb_ref.store(obs=obs[t])
        rb_ref.advance_episode()
        
        # Store in buffered storer
        buffered_storer(obs=obs, episode_lens=length)
        
    # Final flush for buffered storer
    buffered_storer(force_flush = True)
    
    # Compare
    data_ref = rb_ref.get_all_data()
    data_buf = rb_buf.get_all_data()
    
    assert data_ref.keys() == data_buf.keys()
    
    for k in data_ref:
        np.testing.assert_allclose(data_ref[k], data_buf[k], atol=1e-6)
    
    assert rb_ref.num_episodes == rb_buf.num_episodes
    assert rb_ref.episode_index == rb_buf.episode_index
    assert np.all(rb_ref.episode_lens[:] == rb_buf.episode_lens[:])
